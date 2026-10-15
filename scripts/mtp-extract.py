#!/usr/bin/env python3
"""Extract an embedded MTP (nextn) draft layer from a qwen4exp GGUF into a
standalone head GGUF that `q4 --mtp` can load.

Models like Qwen3.8-Flash-Next ship the draft block inside the main file as
`blk.<N>.*` tensors (one extra layer past the regular stack). This copies the
GGUF metadata verbatim and repacks only the draft-layer tensors, producing a
~3 GiB head file instead of pointing the loader at the whole 80+ GiB model.

Usage:
  scripts/mtp-extract.py <model.gguf> -o head.gguf [--layer N]
  scripts/mtp-extract.py --hf [<repo>/<file.gguf>] -o head.gguf

Multi-shard inputs are supported: pass shard 00001 and siblings are found.
--hf reads the GGUF from Hugging Face with HTTP range requests: the header
and the draft layer's tensors only (~3 GiB), not the 80+ GiB model.  The
default is the IQ3E-Q8D-MTP file at a pinned revision; its draft head also
serves the GSQ-RCO quants, which ship without one.  HF_ENDPOINT selects a
mirror, HF_TOKEN is sent when set.
"""
import argparse
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

ALIGN = 32
HF_DEFAULT = ('pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF/'
              'Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf')
HF_REVISION = '75e6411f0157bb4b6d492a9868bc93c57d2b061c'
# scalar value-type sizes; 8=STRING (u64 len), 9=ARRAY
VT_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _u32(buf, p):
    return struct.unpack_from('<I', buf, p)[0], p + 4


def _u64(buf, p):
    return struct.unpack_from('<Q', buf, p)[0], p + 8


def _str(buf, p):
    n, p = _u64(buf, p)
    return buf[p:p + n].decode('utf-8', 'replace'), p + n


def skip_value(buf, p, vt):
    """Return position just past a kv value of type vt."""
    if vt == 8:
        n, p = _u64(buf, p)
        return p + n
    if vt == 9:
        et, p = _u32(buf, p)
        n, p = _u64(buf, p)
        if et == 8:
            for _ in range(n):
                ln, p = _u64(buf, p)
                p += ln
            return p
        return p + n * VT_SIZE[et]
    return p + VT_SIZE[vt]


def read_value(buf, p, vt):
    """Parse a kv value; returns (value, new_pos). Only scalar int/str used."""
    if vt == 8:
        return _str(buf, p)
    if vt in VT_SIZE:
        fmt = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f',
               7: '<?', 10: '<Q', 11: '<q', 12: '<d'}[vt]
        v = struct.unpack_from(fmt, buf, p)[0]
        return v, p + VT_SIZE[vt]
    # ARRAY: skip, keep raw
    p2 = skip_value(buf, p, vt)
    return None, p2


class LocalSrc:
    def __init__(self, path):
        self.name = path
        self.size = os.path.getsize(path)
        self.f = open(path, 'rb')

    def read(self, off, n):
        self.f.seek(off)
        return self.f.read(n)


class HttpSrc:
    """A remote file read by HTTP range requests (each answer must be a
    206 for exactly the range asked: a server that ignores Range sends
    the file's start, which would silently corrupt the head)."""

    def __init__(self, url):
        self.name = url
        self.url = url
        self.hdr = {'User-Agent': 'q4-mtp-extract'}
        tok = os.environ.get('HF_TOKEN')
        if tok:
            self.hdr['Authorization'] = 'Bearer ' + tok
        _, total = self._get(0, 16, want_total=True)
        self.size = total

    def _get(self, off, n, want_total=False):
        rng = f'bytes={off}-{off + n - 1}'
        for attempt in range(6):
            try:
                req = urllib.request.Request(url=self.url, headers=dict(self.hdr, Range=rng))
                with urllib.request.urlopen(req, timeout=60) as r:
                    cr = r.headers.get('Content-Range', '')
                    if r.status != 206 or not cr.startswith(rng.replace('=', ' ') + '/'):
                        raise SystemExit(f'{self.url}: no range support '
                                         f'(status {r.status}, Content-Range {cr!r})')
                    data = r.read()
                    if len(data) != n:
                        raise OSError(f'short read {len(data)} of {n}')
                    return data, int(cr.rsplit('/', 1)[1])
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                    raise SystemExit(f'{self.url}: HTTP {e.code}')
                if attempt == 5:
                    raise SystemExit(f'{self.url}: {e}')
                time.sleep(2 * (attempt + 1))

    def read(self, off, n):
        n = max(0, min(n, self.size - off))
        out = bytearray()
        while len(out) < n:
            k = min(n - len(out), 64 << 20)
            out += self._get(off + len(out), k)[0]
        return bytes(out)


def open_src(path):
    if path.startswith(('http://', 'https://')):
        return HttpSrc(path)
    return LocalSrc(path)


def parse_gguf(path):
    """Return dict with header bytes, kv region, tensor infos, data start."""
    src = open_src(path)
    size = src.size
    win = 1 << 24
    while True:
        head = src.read(0, min(size, win))
        try:
            return _parse_head(src, path, head, size)
        except struct.error:
            if len(head) >= size:
                raise SystemExit(f'{path}: truncated GGUF header')
            win *= 4


def _parse_head(src, path, head, size):
    if head[:4] != b'GGUF':
        raise SystemExit(f'{path}: not a GGUF file')
    ver = _u32(head, 4)[0]
    if ver < 2 or ver > 3:
        raise SystemExit(f'{path}: GGUF version {ver} unsupported')
    n_tensors, n_kv = struct.unpack_from('<QQ', head, 8)
    p = 24
    kv_start = p
    alignment = ALIGN
    for _ in range(n_kv):
        k, p = _str(head, p)
        vt, p = _u32(head, p)
        if k == 'general.alignment':
            v, p = read_value(head, p, vt)
            if v:
                alignment = v
        else:
            p = skip_value(head, p, vt)
    kv_end = p
    tensors = []
    for _ in range(n_tensors):
        name, p = _str(head, p)
        nd, p = _u32(head, p)
        dims = struct.unpack_from('<' + 'Q' * nd, head, p)
        p += 8 * nd
        ty, p = _u32(head, p)
        off, p = _u64(head, p)
        tensors.append({'name': name, 'dims': dims, 'type': ty, 'off': off})
    if p > len(head):
        raise struct.error('header past the read window')
    data_start = ((p + alignment - 1) // alignment) * alignment
    return {'path': path, 'src': src, 'ver': ver, 'n_kv': n_kv,
            'kv': head[kv_start:kv_end], 'infos_end': p,
            'data_start': data_start, 'align': alignment,
            'tensors': tensors, 'size': size}


def tensor_bytes(parts):
    """Compute each tensor's byte size via next-offset / EOF within its file."""
    sizes = {}
    for part in parts:
        ts = sorted(part['tensors'], key=lambda t: t['off'])
        ds, fs = part['data_start'], part['size']
        for i, t in enumerate(ts):
            end = ts[i + 1]['off'] if i + 1 < len(ts) else fs - ds
            sizes[(part['path'], t['name'])] = end - t['off']
    return sizes


def discover_shards(path):
    m = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', path)
    if not m:
        return [path]
    tot = int(m.group(2))
    base = path[:m.start()]
    out = [f'{base}-{i:05d}-of-{tot:05d}.gguf' for i in range(1, tot + 1)]
    for p in out:
        if not p.startswith(('http://', 'https://')) and not os.path.exists(p):
            raise SystemExit(f'missing shard {p}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', nargs='?',
                    help='local GGUF (shard 00001 for split files), or with '
                         '--hf a <repo>/<file.gguf> on Hugging Face')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--hf', action='store_true',
                    help='read from Hugging Face by range requests '
                         f'(default {HF_DEFAULT} @ {HF_REVISION[:7]})')
    ap.add_argument('--revision', default=None,
                    help='with --hf: the commit or branch (default: pinned '
                         'for the default file, main otherwise)')
    ap.add_argument('--layer', type=int, default=None,
                    help='draft layer index (default: layer owning .nextn.)')
    a = ap.parse_args()

    model = a.model
    if a.hf:
        spec = model or HF_DEFAULT
        org, repo, fname = spec.split('/', 2)
        rev = a.revision or (HF_REVISION if spec == HF_DEFAULT else 'main')
        ep = (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
        model = f'{ep}/{org}/{repo}/resolve/{rev}/{fname}'
        print(f'reading {spec} @ {rev[:12]} by range requests')
    elif not model:
        ap.error('a model path is required (or --hf)')

    parts = [parse_gguf(p) for p in discover_shards(model)]
    sizes = tensor_bytes(parts)

    layer = a.layer
    if layer is None:
        pat = re.compile(r'^blk\.(\d+)\.nextn\.')
        cand = [int(pat.match(t['name']).group(1))
                for part in parts for t in part['tensors']
                if pat.match(t['name'])]
        if not cand:
            raise SystemExit('no .nextn. tensors found; pass --layer')
        layer = max(cand)
    prefix = f'blk.{layer}.'
    sel = []
    for part in parts:
        for t in part['tensors']:
            if t['name'].startswith(prefix):
                sel.append((part, t))
    if not sel:
        raise SystemExit(f'no tensors match {prefix}*')
    sel.sort(key=lambda st: st[1]['off'])

    align = parts[0]['align']
    head0 = parts[0]
    tmp = a.out + '.part'
    out = open(tmp, 'wb')
    # header: magic ver n_tensors n_kv
    out.write(b'GGUF')
    out.write(struct.pack('<IQQ', head0['ver'], len(sel), head0['n_kv']))
    kv_off = out.tell()
    out.write(head0['kv'])
    infos = b''
    off = 0
    new = []
    for part, t in sel:
        n = sizes[(part['path'], t['name'])]
        infos += struct.pack('<Q', len(t['name'])) + t['name'].encode()
        infos += struct.pack('<I', len(t['dims']))
        infos += struct.pack('<' + 'Q' * len(t['dims']), *t['dims'])
        infos += struct.pack('<IQ', t['type'], off)
        new.append((part, t, off, n))
        off += ((n + align - 1) // align) * align
    out.write(infos)
    pad = (-out.tell()) % align
    out.write(b'\0' * pad)
    written = 0
    data0 = out.tell()
    for part, t, o, n in new:
        # tensor offsets are relative to the data section
        want = out.tell() - data0
        if want < o:
            out.write(b'\0' * (o - want))
        src = part['src']
        pos = part['data_start'] + t['off']
        left = n
        t0 = time.time()
        while left:
            chunk = src.read(pos, min(left, 1 << 26))
            if not chunk:
                raise SystemExit(f'short read on {t["name"]}')
            out.write(chunk)
            pos += len(chunk)
            left -= len(chunk)
        written += n
        dt = max(time.time() - t0, 1e-3)
        print(f"  {t['name']:<64} {n/1e6:9.1f} MB  {n/1e6/dt:7.1f} MB/s",
              flush=True)
    out.close()
    os.replace(tmp, a.out)
    print(f'{a.out}: {len(sel)} tensors from layer {layer}, '
          f'{os.path.getsize(a.out)/2**30:.2f} GiB')


if __name__ == '__main__':
    main()
