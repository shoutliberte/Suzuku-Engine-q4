#!/usr/bin/env python3
"""Engine-neutral throughput bench: identical chat requests against a q4 or Strata server.

Client-side timing over SSE, the same for both engines:
  prefill t/s = prompt_tokens / TTFT          (TTFT = request sent -> first token chunk)
  decode  t/s = (completion_tokens - 1) / (last token chunk - first token chunk)
Every user message starts with a per-request nonce so no prefix cache can reuse
a previous prompt.  Does not start/stop servers: bench/run-arm.sh does that.

  bench.py run --url http://127.0.0.1:8090 --label q4-abc123 --out bench/results/x.json
  bench.py compare a.json b.json
"""
import argparse, hashlib, json, os, re, statistics, sys, time, urllib.request, uuid
from pathlib import Path

# the prompt texts (ja / code / long8k ...; not in the repository): --eval-dir or Q4_BENCH_EVAL
EVAL = Path(os.environ.get('Q4_BENCH_EVAL') or Path(__file__).resolve().parents[1] / 'models' / 'eval')
# q4 prints these to stderr (the server log) per request: the prefill rate, the decode rate,
# and with a head the MTP acceptance line; any fallback to plain decode is logged by name.
Q4_PREFILL = re.compile(r'prefill (\d+)\+(\d+)/(\d+)\s+([0-9.]+) t/s')
Q4_DECODE = re.compile(r'decode (\d+) tok\s+([0-9.]+) t/s')
Q4_MTP = re.compile(r'mtp draft acceptance = [0-9.]+ \((\d+) accepted / (\d+) drafted, (\d+) gated\), '
                    r'mean len = [0-9.]+ \((\d+) cycles\)')
Q4_MTP_OFF = re.compile(r'MTP cycle failed|MTP cycle recovered|MTP partial repair|MTP head prefill failed|'
                        r'MTP seed failed')
STRATA_LINE = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), '
                         r'(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?')

AGENT_SYS = """You are a coding agent working inside a local git repository. You act only through tools.

Available tools (call exactly one per turn, written as a fenced ```json block {"tool": ..., "args": {...}};
do not use XML tags or any other tool-call markup):
- read_file {"path": string, "offset": int?, "limit": int?}: read a file with line numbers.
- grep {"pattern": string, "path": string?}: regex search over the repository.
- edit {"path": string, "old_string": string, "new_string": string}: exact string replacement.
- exec {"command": string, "timeout_ms": int?}: run a shell command, returns stdout/stderr.

Rules: think briefly, state what you will check and why, then emit one tool call.
Never guess file contents; read before editing. Answer the user in Japanese."""


def sources(eval_dir):
    sums = dict(reversed(l.split()) for l in (eval_dir / 'SHA256SUMS').read_text().splitlines() if l.strip())
    out = {}
    for name in ('ja-notes.txt', 'en-strata-paper.txt', 'code-q4_mtp.c.txt'):
        b = (eval_dir / name).read_bytes()
        if hashlib.sha256(b).hexdigest() != sums[name]:
            sys.exit(f'{name}: sha256 mismatch against {eval_dir}/SHA256SUMS')
        out[name.split('-')[0]] = b.decode()
    return out


def cases(src):
    ja, en, code = src['ja'], src['en'], src['code']
    return [
        dict(name='ja', kind='decode', max_tokens=256, messages=[{'role': 'user', 'content':
             '次の文章を読んで、内容を日本語で詳しく要約してください。重要な論点ごとに見出しをつけてください。\n\n' + ja[:4000]}]),
        dict(name='agent', kind='decode', max_tokens=256, messages=[
             {'role': 'system', 'content': AGENT_SYS},
             {'role': 'user', 'content': 'MTPのverifyサイクルが遅い原因を調べて、修正方針を立ててください。'
              '直前に read_file で src/q4_mtp.c を読んだ結果を下に貼ります。次に確認すべき箇所を決めて、ツールを呼んでください。\n\n'
              + code[:6000]}]),
        dict(name='code', kind='decode', max_tokens=256, messages=[{'role': 'user', 'content':
             'PythonでTTL付きのLRUキャッシュを実装してください。型ヒントとdocstringを付け、pytestのテストも書いてください。'}]),
        dict(name='pf8k', kind='prefill', max_tokens=16, messages=[{'role': 'user', 'content':
             '以下の資料を読み、最後に一文で要約してください。\n\n' + ja + '\n\n' + en[:14000]}]),
        dict(name='pf32k', kind='prefill', max_tokens=16, messages=[{'role': 'user', 'content':
             '以下の資料を読み、最後に一文で要約してください。\n\n' + ja + '\n\n' + en + '\n\n' + code + '\n\n' + en}]),
        # Production-like long requests: a long prompt and a long answer
        # (select with --only long8k,long32k; not in the default set).
        dict(name='long8k', kind='decode', max_tokens=1024, optional=True, messages=[{'role': 'user', 'content':
             '以下の資料を読み、内容を日本語で詳しく解説してください。重要な論点ごとに見出しをつけ、具体的な数値も引用してください。\n\n'
             + ja + '\n\n' + en[:14000]}]),
        # long8k with the system turn Strata's effort_end rendering keeps
        # (the template's xhigh reasoning instruction) — a language A/B.
        dict(name='long8ks', kind='decode', max_tokens=1024, optional=True, messages=[
             {'role': 'system', 'content': 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.'},
             {'role': 'user', 'content':
             '以下の資料を読み、内容を日本語で詳しく解説してください。重要な論点ごとに見出しをつけ、具体的な数値も引用してください。\n\n'
             + ja + '\n\n' + en[:14000]}]),
        dict(name='long32k', kind='decode', max_tokens=1024, optional=True, messages=[{'role': 'user', 'content':
             '以下の資料を読み、内容を日本語で詳しく解説してください。重要な論点ごとに見出しをつけ、具体的な数値も引用してください。\n\n'
             + ja + '\n\n' + en + '\n\n' + code + '\n\n' + en}]),
    ]


# Qwen3.8 Flash Next is specified for temperature 1.0 and must not be run greedy; temp=0 only
# reproduces the pre-2026-10-07 numbers (BUDGET.md).
# presence_penalty 0 explicitly: q4 defaults to 1.5 for thinking-off requests (the
# model card's instruct set), and the bench runs thinking off.
SAMPLE = dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0, presence_penalty=0)
GREEDY = dict(temperature=0, top_k=1, top_p=1, min_p=0)


def request(url, messages, max_tokens, timeout, sample=SAMPLE):
    body = dict(model='bench', messages=messages, max_tokens=max_tokens, stream=True,
                stream_options={'include_usage': True}, **sample, seed=42,
                reasoning_effort='none', chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.monotonic()
    t_first = t_last = None
    text, usage, finish, chunks = [], None, None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            ev = json.loads(data)
            usage = ev.get('usage') or usage
            for ch in ev.get('choices') or []:
                finish = ch.get('finish_reason') or finish
                d = ch.get('delta') or {}
                piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                if d.get('tool_calls'):
                    piece += json.dumps(d['tool_calls'], ensure_ascii=False)
                if piece:
                    now = time.monotonic()
                    t_first = t_first or now
                    t_last = now
                    chunks += 1
                    text.append(piece)
    t_end = time.monotonic()
    if not usage or t_first is None:
        raise RuntimeError(f'no usage or no tokens in stream (usage={usage})')
    pt, ct = usage['prompt_tokens'], usage['completion_tokens']
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    ttft, dec = t_first - t0, t_last - t_first
    return dict(prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached, chunks=chunks, finish=finish,
                ttft_s=ttft, decode_s=dec, wall_s=t_end - t0,
                prefill_tps=pt / ttft, decode_tps=(ct - 1) / dec if ct > 1 and dec > 0 else None,
                text=''.join(text))


def q4_lines(text):
    """The q4 server's per-request lines, or None before its decode line is out."""
    d = Q4_DECODE.findall(text)
    if not d:
        return None
    r = dict(generated=float(d[-1][0]), decode_tps=float(d[-1][1]))
    pf = Q4_PREFILL.findall(text)
    if pf:
        r.update(reused=float(pf[-1][0]), fresh=float(pf[-1][1]), prompt_tokens=float(pf[-1][2]),
                 prefill_tps=float(pf[-1][3]))
    m = Q4_MTP.findall(text)
    if m:
        acc, prop, gated, cyc = (float(x) for x in m[-1])
        r.update(drafts_accepted=acc, drafts=prop, gated=gated, cycles=cyc)
    r['mtp_fallbacks'] = len(Q4_MTP_OFF.findall(text))
    return r


def engine_line(log, offset):
    """The engine's own timing for the request whose output starts at `offset` of its log:
    Strata's one-line summary, or q4's prefill/decode/MTP lines."""
    if not log:
        return None
    deadline = time.monotonic() + 3
    q4 = None
    while True:
        with open(log, 'rb') as f:
            f.seek(offset)
            text = f.read().decode(errors='replace')
        m = STRATA_LINE.findall(text)
        if m:
            break
        got = q4_lines(text)
        if got and (got.get('cycles') or q4 == got):   # MTP line in, or two reads agree (plain decode)
            return got
        q4 = got
        if time.monotonic() > deadline:
            return q4 or dict(error='no timing line')
        time.sleep(0.2)
    if len(m) != 1:
        return dict(error=f'{len(m)} timing lines')
    v = list(m[0])
    keys = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps', 'generated', 'decode_ms', 'decode_tps',
            'drafts_accepted', 'drafts')
    return {k: float(x) for k, x in zip(keys, v) if x != ''}


def spec_note(e):
    """'acc=a/d tok/cyc=x ' from an engine line: the evidence that speculation stayed on."""
    if not e or not e.get('drafts'):
        return ''
    gen, acc = e.get('generated') or 0, e.get('drafts_accepted') or 0
    cyc = e.get('cycles') or (gen - acc)
    note = f"acc={acc:.0f}/{e['drafts']:.0f} tok/cyc={gen / cyc:.2f} " if cyc else ''
    return note + ('MTP-FALLBACK ' if e.get('mtp_fallbacks') else '')


def run(a):
    src = sources(Path(a.eval_dir))
    sample = GREEDY if a.greedy else SAMPLE
    results = dict(label=a.label, url=a.url, started=time.strftime('%F %T'), rounds=a.rounds, sample=sample, runs=[])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'[{a.label}] warmup', flush=True)
    request(a.url, [{'role': 'user', 'content': f'[{uuid.uuid4().hex[:8]}] Reply READY.'}], 8, a.timeout)
    # Strata ran ~10x slow for its first minutes after a fresh start (2026-10-04);
    # warm until two consecutive decode rates agree within 10%.
    prev = None
    for i in range(12):
        w = request(a.url, [{'role': 'user', 'content': f'[{uuid.uuid4().hex[:8]}]\n' + src['en'][:6000] +
                             '\n\nSummarize the text above in Japanese.'}], 96, a.timeout)
        print(f"[{a.label}] warmup {i}: pf={w['prefill_tps']:.1f} dec={w['decode_tps'] or 0:.1f} t/s", flush=True)
        if prev and w['decode_tps'] and abs(w['decode_tps'] - prev) <= 0.1 * prev:
            break
        prev = w['decode_tps']
    else:
        sys.exit(f'[{a.label}] decode rate never settled during warmup')
    sel = set(a.only.split(',')) if a.only else None
    for rnd in range(1, a.rounds + 1):
        for c in cases(src):
            if c.get('optional') and not (sel and c['name'] in sel):
                continue
            if sel and c['name'] not in sel:
                continue
            msgs = json.loads(json.dumps(c['messages']))
            msgs[-1]['content'] = f'[bench {uuid.uuid4().hex[:12]}]\n' + msgs[-1]['content']
            off = Path(a.engine_log).stat().st_size if a.engine_log else 0
            r = request(a.url, msgs, c['max_tokens'], a.timeout, sample)
            r.update(case=c['name'], kind=c['kind'], round=rnd, engine=engine_line(a.engine_log, off))
            if r['cached_tokens']:
                raise RuntimeError(f"{c['name']}: {r['cached_tokens']} cached prompt tokens on a fresh request")
            results['runs'].append(r)
            out.write_text(json.dumps(results, ensure_ascii=False, indent=1) + '\n')
            dt = f"{r['decode_tps']:.1f}" if r['decode_tps'] else '-'
            print(f"[{a.label}] r{rnd} {c['name']:6s} pt={r['prompt_tokens']:6d} ttft={r['ttft_s']:7.2f}s "
                  f"pf={r['prefill_tps']:7.1f} t/s  ct={r['completion_tokens']:4d} dec={dt} t/s  "
                  f"{spec_note(r['engine'])}{r['text'][:60]!r}", flush=True)


def summarize(path):
    d = json.loads(Path(path).read_text())
    by = {}
    for r in d['runs']:
        by.setdefault(r['case'], []).append(r)
    rows = {}
    for name, rs in by.items():
        key = 'prefill_tps' if rs[0]['kind'] == 'prefill' else 'decode_tps'
        vals = [r[key] for r in rs if r[key]]
        rows[name] = dict(kind=rs[0]['kind'], metric=key, median=statistics.median(vals), vals=vals,
                          prompt_tokens=rs[0]['prompt_tokens'], text=rs[0]['text'])
    return d['label'], rows


def compare(a):
    sums = [summarize(p) for p in a.files]
    labels = [s[0] for s in sums]
    print('| case | metric | pt | ' + ' | '.join(labels) + (' | ratio |' if len(sums) == 2 else ' |'))
    print('|---|---|---|' + '---|' * len(sums) + ('---|' if len(sums) == 2 else ''))
    for name in sums[0][1]:
        cells = [s[1].get(name) for s in sums]
        row = f"| {name} | {cells[0]['metric'].split('_')[0]} | {cells[0]['prompt_tokens']} | "
        row += ' | '.join(f"{c['median']:.1f} ({', '.join(f'{v:.1f}' for v in c['vals'])})" if c else '-'
                          for c in cells)
        if len(sums) == 2 and all(cells):
            row += f" | {cells[1]['median'] / cells[0]['median']:.2f}x"
        print(row + ' |')
    if len(sums) == 2:
        for name in sums[0][1]:
            x, y = sums[0][1][name]['text'], sums[1][1].get(name, {}).get('text', '')
            n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            print(f'{name}: outputs agree for the first {n} chars (of {len(x)}/{len(y)})')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--url', required=True)
    r.add_argument('--label', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--rounds', type=int, default=3)
    r.add_argument('--greedy', action='store_true', help='temp 0 / top_k 1 (old BUDGET.md numbers only)')
    r.add_argument('--only', help='comma-separated case names')
    r.add_argument('--engine-log', help="engine log (Strata's own log, or q4's server log): per-request timing and MTP acceptance")
    r.add_argument('--eval-dir', default=str(EVAL))
    r.add_argument('--timeout', type=float, default=1800)
    c = sp.add_parser('compare')
    c.add_argument('files', nargs='+')
    a = p.parse_args()
    run(a) if a.cmd == 'run' else compare(a)


if __name__ == '__main__':
    main()
