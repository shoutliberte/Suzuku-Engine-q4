# Suzuku-Engine-q4 agent notes

## Resource safety

- Only one q4 process at a time. The resident expert image is tens of GiB.
- The binaries self-contain: `q4_self_scope()` re-execs under
  `systemd-run --user --scope` with `MemoryMax=$Q4_MEM_MAX` and
  `MemorySwapMax=0` whenever no memory-capped cgroup is already in force.
  Opt-outs: `Q4_NOSCOPE=1`, `Q4_MEM_MAX=0`.  The cap bounds only q4's own
  memory: GTT buffers and VRAM evicted by the driver are charged to the
  kernel, not the cgroup — when VRAM is full the driver can dump q4's
  multi-GiB L1 arena into system RAM and OOM the box.  Watch the
  `vram=`/`q4vram=`/`gtt=` fields the serve/q4safe watchdogs log;
  `Q4_VRAM_RESERVE_GIB` (default 2) is the margin that prevents it.
- `serve.sh` wraps via
  `systemd-run --user --scope -p MemoryMax=$Q4_MEM_MAX -p MemorySwapMax=0`.
  For probes/benches use `scripts/q4safe.sh <logprefix> <q4 args...>`
  (same cap + a watchdog that logs MemAvailable/swap/RSS/VRAM and kills q4
  on floor/swap-growth/stall). Quantized weights don't compress, so letting
  the image swap only thrashes.
- `serve.sh` and `q4safe.sh` source `scripts/q4preflight.sh`: they refuse to
  start with another q4 alive, after an unacknowledged q4 `oom-kill` scope, or
  with MemAvailable under `Q4_NEED_GIB`, and they set `oom_score_adj` so a
  global OOM takes q4 rather than the rest of the system.
- After any OOM kill of q4: stop, read `journalctl -b | grep -i oom`, and
  understand the cause before relaunching — the kernel is still reaping the
  resident image. Acknowledge with `systemctl --user reset-failed` only after
  the cause is understood.
- No q4 runs while a large download or other heavy memory use competes for RAM.

## Build / verify

- Build: `make` (hipcc, ROCm; `--offload-arch` via `ROCM_ARCH`, default gfx1100).
- Prefill bench: `Q4_EXPERT_RESIDENT=1 Q4_L1_SLOTS=8192 Q4_IO_THREADS=8 Q4_DRAM_RESERVE_GIB=4 HIP_VISIBLE_DEVICES=0 ./q4 prefill-probe <model> --tokens N --ctx 8192`
  (profiling: `Q4_PROFILE=1`, `Q4_GEMV_PROF=2`; `Q4_PREF_CHUNK` 512..8192, default 6144 — with the MMQ group equal to the chunk each chunk is one expert GEMM (~120 rows per expert); 8192 chunk + 4096 groups was 1070 / 1077 t/s on pf8k / pf32k, 6144 / 6144 is 1186 / 1129 with a smaller lent scratch (static q 27 -> 28), 8192 / 8192 is as fast but lends 0.5 GiB more (q 23, decode lost); the engine also splits a prompt into equal chunks (`Q4_PREF_BALANCE`, a short tail chunk cannot cover its expert stream) — the engine borrows its ~3.3 GiB scratch from the L1 arena tail (L1 lending, below), so the chunk no longer costs decode slots; prefill 1.6× vs 512;
  (`Q4_PROBE_FILE=<utf8 text>`: the default probe text is one paragraph repeated,
  which flatters expert routing and PLE row reuse — use Q4_PROBE_FILE for real numbers.)
  `Q4_PF=0` disables next-layer expert prefetch; `Q4_LOGIT_DUMP=1` prints top logits).
- Prompt QSA indexer score (`qsa_score_wmma_k`, `Q4_QSA_SCORE_WMMA=0`
  reverts, n_tok >= 16): sum_h relu(q_h . pool_b) as a WMMA tile GEMM (32
  tokens x 64 pooled blocks, fp16 operands, fully masked tiles only write
  -1e30).  `qsa_score_k` walked each token's pooled rows serially in fp32
  (~1.3 TFLOPS, 40 -> 440 ms per 5.5K chunk as the context grew to 34K):
  34K prompt 1048 -> 1154 t/s.  The 34K profile per 5.5K chunk otherwise:
  MMQ ~1.5 s, indexed attention 0.65 -> 1.1 s (grows until the 2048
  selection fills), hipBLASLt dense ~0.6 s, GDN ~0.26 s, fp16/bf16
  conversion + dequant glue ~0.4 s.
- Prompt QSA attention over the union of T consecutive tokens' selections
  (`gqa_union_wmma_k` + `qsa_union_k`, `Q4_QSA_UNION`: 4 default, 2, 0 =
  per-token `gqa_prefill_wmma_k`): neighbours pick overlapping 4-position
  blocks (union / one selection: T=4 ~1.4 at 4K context, ~2.2 at 24K+), so
  one block per (kv head, 4 tokens) loads each K/V row once for 48 query
  rows and masks per (token, position).  What the per-tile time was
  (512-token, 28K-context layer, local `pfab` bench on a dumped selection):
  the softmax's lane shuffles (~2K clocks per row; now 8 lanes per row with
  DPP row_xmask), the P*V B operand read as 16 u16 LDS column loads (V is
  now written transposed), K/V gathers at ~0.95 TB/s.  4.9 -> 2.5 ms;
  engine QSA section 0.90 -> 0.56 s per 5.5K chunk at 34K, 34K prompt
  1610 -> 1800, 10K 1596 -> 1743 t/s.  Register prefetch of the next tile,
  64-position tiles and an LDS-only barrier did not help.  Selections must
  not repeat a position within a token (the topk output does not).  A
  position of a union block that no token of the group selected is not
  loaded (zero K/V): the block of the prompt's last cell runs past the
  written cells, and KV chunks paged in from donated expert slots still
  held expert bytes, whose q8 scales decode to inf / NaN — P = 0 times
  NaN turned the last n % 4 tokens of a prompt into NaN rows (NaN logits
  on ~3/4 of pf32k runs after KV paging).  Do not zero KV chunks on map
  instead: hipMemsetAsync over the VMM chunks made the first request take
  ~200 s.  q4-test poisons every unselected cell with NaN scales.
- MMQ launch grid (`Q4_MMQ_TUNE`, RDNA3 only; 0 stock, 1 max_rows grid,
  2 default + compacted grid): `q4_mmq_mul_id` sized grid y from the
  chunk's total rows (81920 at 8192 tokens -> 640 j-tiles per expert,
  ~13M blocks, nearly all returning at once) instead of the largest
  expert's rows; the compacted grid also drops the empty (expert, j-tile)
  pairs (`mmq_tile_prefix_k` + binary search in `mul_mat_q`).  Outputs
  bit-identical.  Microbench (512 experts, 81920 rows): down projections
  ~3x (12 -> 39 TOPS), gate/up ~1.6x (22 -> 35-42 TOPS).  Engine chat:
  10K prompt 1315 -> 1602, 34K 1167 -> 1613 t/s.  The llama.cpp template
  instances depend only on their .cu, so touch them after editing mmq.cuh.
- Prefill routed experts run on `q4_xg` (src/q4_xg.cu, `Q4_XG=0` back to
  MMQ): fp16 WMMA tiles of 128 routed rows x 128 weight rows, the weights
  read straight from their L1 slots (no id-indexed gather, so the MMQ
  weight buffers leave the lend tail: 1368 -> 856 slots) and decoded into
  LDS once per tile with packed fp16 math (grid/table byte -> 1024 + b by
  v_perm, v_pk_add, sign-mask XOR from a 256-entry LDS table, v_pk_mul;
  ~2 VALU per weight), gate/up fused with swiglu writing h as fp16, down
  reading it — no q8_1 activation quantize, no f32 gate/up buffer.
  Weights decode scaled by 256 (sub-block scales ~1e-5 are fp16
  subnormals); h and x saturate at +-65504.  Rows past a tile's count skip
  per 16-row fragment (`xg_mma<NM>`; per 64 wasted ~27% at ~120 rows per
  expert).  Microbench (6144 tokens x top-10, 512 experts): gate/up 15.3
  ms MMQ (+ quantize, swiglu) -> 6.8 ms, down 6.7 -> 3.4 ms; the raw
  WMMA+LDS skeleton is ~5.3 ms (~80 of the ~93 TFLOPS WMMA reaches under
  load).  34K prompt 1865 -> 2130 t/s; ppl drift = MMQ's run-to-run
  noise; q4-test checks every type against a CPU reference (rel ~4e-4,
  MMQ 3.7e-3).  Dead ends: prefetching the next step's x / raw blocks into
  registers (the loop-carried copy forces an immediate vmcnt wait, and the
  VGPRs cost a block per WGP: 6.8 -> 7.6-8.2 ms).
- The prompt's dense projections run on the same loop (`q4_dg_gemm`,
  called from `q4_hgemm_try`; `Q4_DG=0` back to hipBLASLt): fp16
  (dequantized as before) or bf16 weights [N][K], the fp32 activations
  converted while staged into LDS — the separate to_f16 / to_bf16 passes
  (0.88 s of a 34K prompt: every GEMM re-converted its whole input) are
  gone.  Kernel time for a 34K prompt: hipBLASLt 4.15 s + conversions
  0.88 s -> 4.0 s; wall 2175 -> 2247 t/s (3 alternating runs each); ppl
  drift = run-to-run noise.  hc down (N 320, K 10240) is the weakest
  shape (~57 TFLOPS, 135 blocks).
- The prompt's KV chunks are mapped at the request start
  (`engine_kv_fit` -> `q4_sess_kv_reserve`), with the device idle, not
  per 16K step mid-prompt under running kernels.
- Prefill fallbacks (each `=0` restores the previous kernels bit-identically, for bisecting):
  `Q4_WMMA`, `Q4_XG`, `Q4_DG`, `Q4_MMQ` (MMQ expert GEMMs; group cap
  `Q4_MMQ_GROUP`, default = the prefill chunk), `Q4_QSA_PF`, `Q4_PLE_BATCH`,
  `Q4_GDN_SCAN`, `Q4_PF` (next-layer prefetch), `Q4_PF_ROUTE`
  (host routing), `Q4_PF_ASSIGN` (host assignment over device picks).  `Q4_GEMM_BF16W=1` opts into the
  bf16 WMMA dense GEMM (off: measured slower).  `Q4_PF_ROUTE_CHECK=1`
  diffs device vs host route decisions per layer.
- `Q4_L1_STATIC` (default on; needs `Q4_WARM_FILE` + a resident image): the
  top-`q` warm experts per layer are pinned into fixed GPU L1 slots at
  startup and excluded from the DRAM image — `q = (L1_slots − 2·n_exp − 64)
  / (n_layer − 2)`, the remaining `D = slots − n_layer·q` dynamic slots
  cover two layers of staging.  `=0` restores the old placement.
- L1 lending (`Q4_L1_LEND`, default on, engine only): the prompt's big
  device scratch (prefill row buffers, `d_pws`, the MMQ weight gather,
  the GDN scan's n>=64 buffers; `q4_sess_lend_bytes`, ~3.3 GiB at chunk
  8192) is not hipMalloc'd — it is the arena's last `n_tail` slots.  The
  first `q4_sess_prefill` of a prompt borrows them (`q4_expert_lend_begin`:
  drain, unmap, block); `q4_sess_lend_end` (engine, after the head KV
  pass; decode/verify entry points also call it) refills them from the
  DRAM image as ordinary LRU slots (~100 ms for ~1300 experts) and the
  verify uses a permanent 8-row buffer set (its MoE graphs are captured
  once, at the first return).  Same VRAM, ~1300 more decode slots: MTP
  verify misses per layer 21 -> ~15, cpxwait ~20 -> 11-16 ms/cycle,
  greedy agent 30 -> 43 t/s.  `Q4_L1_LEND_FILL=static` keeps the refill
  fixed (measured worse: 19.5 misses/layer).  Lending is never switched
  off after the arena is sized — that would put a second scratch on top
  (2026-10-12: vfree 427 MiB, watchdog kill); an arena too small for the
  tail is a startup error.  The measured-free clamp's `keep` now includes
  the MTP head estimate (it used to come out of the reserve whenever the
  clamp, not the plan, bound the slot count).
- Decode/ppl are NOT bit-reproducible even with `Q4_CPU_EXPERTS=0`: the
  wmoe_* expert-accumulate kernels add each expert's contribution via
  `atomicAdd(y + row, …)` (src/q4_rocm.cu), whose order is timing-dependent,
  so reruns drift ~0.07 mean |Δnll| with ~4% top-1 flips (measured 2026-10-06,
  also at pre-prefill fb0b4f6 — not a regression). Compare ppl .nll dumps by
  magnitude, not equality: a real bug shows max|Δnll| ≫ 2 or flips ≫ 5%.
  `Q4_IO_THREADS=1` is broken (0 io workers → unstaged misses → nll ~8.7
  garbage); keep ≥2.
- GEMV weight-read counters (`Q4_GEMV_PROF=1`) cover every path since
  2026-10-11: `launch()` (single-tensor `q4_hip_gemv_dd`),
  `q4_hip_gemv_m` (multi-segment `gemv_mw_k`) and the routed-expert
  `moe_rows_*` launches charge per type.  Before that, 2.0 of the
  3.9 GiB/token of weight reads (the gdn projections, the hc_* GLU and
  every routed expert) were invisible, so bandwidth numbers derived from
  the old table spoke of about half the traffic.  The table also needs
  `Q4_PROFILE=1` or `Q4_GRAPH=0`: the decode captures each layer as a graph
  and the counters see the recording, not the replays (`q4_fwd.c` turns the
  graphs off while profiling).  `Q4_SHAPES=1` additionally logs the real
  `gemv_m` segment mixes (`MIX n=... in=... nseg=...: [ty=.. r=..] ...`).
  Measured on the GSQ-RCO IQ3_S build: 3.94 GiB/token over 820 calls,
  ~10.6 ms/token of GEMV kernel time → 372 GB/s at the kernel level and
  ~119 GB/s wall clock (33 ms token, ~48% GPU busy).  The warp-per-row
  kernels are occupancy-bound below ~1024 blocks (2-2.5x headroom); the
  iq4_nl/q4_0 decoders now use one 8-byte load + a `v_perm_b32` LUT instead
  of per-u16 loads (identical values and sum order, iq4_nl 158 → 179 GB/s at
  2560 rows).  To re-measure: `Q4_GEMV_PROF=1 Q4_PROFILE=1 prefill-probe
  --tokens 32 --ctx 8192` (probes/graphs off) plus rocprof
  `--kernel-trace` for per-kernel GPU time.
- CPU expert pool layout: the counters every worker writes (`cpx_job`
  next_a / b_done / hold / next_be / a_done_e / hq_rdy, `cpx_pool` live_a /
  done_a) sit on their own cache lines.  Packed, their false sharing
  depended on where the linker put `g_pool`: an unrelated host-side edit
  in q4_rocm.cu moved it to a 64-byte boundary and MTP verify cpxwait went
  20 -> 27 ms/cycle (bisected 2026-10-12); aligned it is 16-18.  When an
  A/B moves CPU-side time with no CPU-side change, rebuild the previous
  commit as a control and check `nm q4 | grep g_pool` (and other hot
  globals) before trusting it.
- Correctness: compare ppl/dump drift against a baseline binary built from
  the previous commit in a `git worktree`.  `q4 ppl` scores in 8-row
  verify windows; `Q4_PPL_PREFILL=N` scores the first N tokens through
  the prompt path (`q4_sess_prefill` chunks, every row) first — needed for
  anything touching prefill buffers (L1 lending on vs off at 8200+800
  tokens: mean |dnll| 0.088 / 4.2% flips, same as off vs off).  For dequant changes,
  `gemvbench parity` (GPU vs the CPU reference, every type) catches
  nibble-mapping and summation-order mistakes before a model load.
- Chat prompts follow chat_template.jinja (checked against a jinja render,
  think tags included): with thinking on, a leading system turn carries the
  reasoning-effort sentence (xhigh by default, low's own sentence, none for
  medium) followed by the leading system/developer messages, trimmed and
  joined by newlines; with thinking off there is no sentence and the
  assistant opens with an empty think block.  The server maps a request's
  effort like Strata's effort_kwargs (`request_think_kwargs`):
  reasoning_effort / reasoning.effort / chat_template_kwargs; none / off /
  minimal / false -> thinking off, low / medium / high / xhigh -> thinking
  on at that effort (OpenCode's levels); none given -> Q4_THINK and
  Q4_EFFORT.  Strata (effort_end) instead keeps the xhigh sentence for
  every request, so bench.py's reasoning_effort "none" gives the engines
  different prompts (~45 tokens; q4 answered long8k in Chinese 3/12 times
  without the sentence, 0/8 with it).  Tools are the template's own block
  (`q4_qwen_tools_block`: "# Tools ... </IMPORTANT>", each tool as
  transformers' `tojson` writes it, i.e. Python json.dumps with
  ensure_ascii=False, sorted by name) ahead of the system text; see the
  OpenCode notes below for what the old compact preamble lost.

## MTP (speculative decoding)

- Turn it on with `Q4_MTP=<head.gguf>` (extracted by
  `scripts/mtp-extract.py`), `Q4_MTP_N=3` and `Q4_MTP_RT=<runtime dir>`
  (the Q2_0 draft-expert pack `experts.bin`, the dense overlay
  `dense.bin`/`dense.txt`, and optionally `draft_vocab.bin` — the layout
  Strata's MTP runtime tooling writes; without it the head's experts
  stream through a small L1 and drafts are much slower;
  `scripts/setup-mtp.sh` builds both next to the model: the head by HTTP
  range reads of the IQ3E GGUF's draft layer, the runtime with Strata's
  tools at a pinned commit).  serve.sh
  turns it on (N=3, `Q4_L1_SLOTS` 3300) when it finds `mtp-head.gguf`
  next to the model (or `Q4_MTP`) and the runtime (`Q4_MTP_RT`, default
  `mtp-rt/` next to the head); `Q4_MTP=0` forces plain.  Greedy on the GSQ-RCO IQ3_S build (2026-10-12, ctx
  262144, L1 lending + batched devtab + K/V-only repair + small-batch
  QSA): ja 63.4-64.4, agent 50.3-52.9, code 63.6-65.5, pf8k 49, pf32k
  49-56 t/s, vs ~37 plain.
- A cycle is draft (up to `Q4_MTP_N` head steps, cut by the confidence
  gate `Q4_MTP_PMIN`, default 0.5) -> verify (one main pass over the
  nd+1 rows, `q4_sess_prefill_tokens`) -> tail (target distribution per
  row) -> commit (GDN rollback + replay of the accepted prefix) -> head KV
  repair (the accepted drafts' cells, by the prompt's K/V-only pass; the
  full head forward it used to run attended over the whole head KV:
  36 ms/cycle at 34K context, now 0.7).  `Q4_MTP_PROF=1` prints ms/cycle
  per phase plus `cpxwait` after each request.  Verify dominates (~43 ms
  at ctx 32K): per layer the critical path is the pre-route GPU work
  (hc mix, attention/GDN, router; ~13 ms/cycle busy) plus the CPU-miss
  job (~500 us x 48 = ~24 ms), the hit experts running beside the job —
  so speeding up post-route GPU work only turns into wait.  The host
  enqueues a window in ~5 ms (`Q4_VPROF=1`: tick / enqueue / total), so
  it is not launch-bound.  rocprof inflates every dispatch gap by ~3 us
  (seambench: 40 tiny kernels/layer 6 -> 174 us/layer under rocprof):
  read its gaps as relative, not absolute.
- Head prompt pass: per prompt chunk only the head's K/V cells are
  written (eh_proj -> attention-side hc mix -> k/v proj -> k norm/rope ->
  append, 64 rows a batch); the head's attention, MoE and logits of
  prompt rows are never read, since every cycle reseeds from the
  verifier's residual.  MTP prefill is within ~2% of plain.
  `Q4_MTP_KVONLY=0` restores the full per-row head forward;
  `Q4_MTP_KVCHECK=1` runs both on every chunk (and every repair)
  in-process and logs the max |diff| of the cells.  When the head KV (`Q4_MTP_KV`, default 65536
  cells) fills, the newest `Q4_MTP_WINDOW` cells (default 32768, at most
  half the capacity) move to the front — before a prompt chunk or at a
  cycle start only.  A cycle failure drops only that request to plain
  decode (`mtp_fail` is cleared per request).
- Draft step: the head top-k (`q4_hip_head_topk`) is one GEMV into a row
  buffer plus a wide block selection (`tkw_k`, 256 threads x 8 values, k
  passes per 2048-row span, repeated until one block).  The old slab
  selection gave each 32768-row slab 4 blocks of 32 threads and was 3.4 ms
  of a 4.65 ms step; a 3-step draft is now ~3.5 ms (was ~14).
  `Q4_TOPK_WIDE=0` restores the slab kernels.  The greedy gate compares
  the softmax p of the draft with `Q4_MTP_PMIN` (`Q4_MTP_GATE_P=0`: the
  old raw-logit compare, which never gated).  `Q4_MTP_SUBSET=1` drafts
  over draft_vocab.bin's rows only: ~1 ms/cycle less draft but code
  tok/cycle 3.08 -> 2.75, so it is off.
- Verify dense GEMVs (2-8 rows): `wgvt_*<NT>` for Q4_K / Q5_K / Q6_K /
  IQ4_XS decode each lane's quants once and reuse them for every row; the
  older `wgvn_*` kept a runtime-indexed `acc[8]` (scratch) and redid the
  dequant per row, so 4 rows cost 2.5x (Q6_K) to 5.3x (Q4_K) one row.
  Now 4 rows cost 0.9-1.4x one row; each row is bit-identical to the
  single-token GEMV (Q6_K shares `q6u_decode`/`q6u_dot` with `wdot_q6k`;
  q4-test checks it).  Greedy ja/code 41/45 -> 45/51 t/s, verify ~57 ->
  ~51 ms/cycle.  `Q4_WGVT=0` restores `wgvn_*`.
- Q4_K / Q5_K / IQ4_XS dense GEMV (`kq16_k<TY, NT, R>`, `Q4_KQ16=0`
  reverts, `Q4_KQ16_R` pins rows per warp): one template for n=1 and the
  2..8-row verify (each row bit-identical across NT and R; explicit fmaf
  + `fp reassociate(off)` so fast-math cannot split them), 16 quant bytes
  per lane, scales from one 16-byte header load, x reused across R rows.
  nbench n=1 340-420 -> 590-670 GB/s, n=4 250-320 -> 330-385; verify
  pre-route 13.6 -> 13.05 ms/pass.  Beyond n=2 the 16B-per-lane map makes
  each x float4 load touch ~16 cache lines per wave (~6 of 24 us at n=4).
- Q6_K: `q6u_dot` / `wdot_q6k` / `wgvt_q6_k` use explicit fmaf with
  `fp reassociate(off)`, so n=1, n-row and `gemv_mw_k` rows are
  bit-identical (fast-math used to split them by ~1e-6), and `q6t_k<NT, R>`
  (`Q4_Q6T`: 0 off, 1 policy, 2 everywhere; `Q4_Q6T_R`) shares x across R
  rows for nt >= 3 on tall tensors (out_dim >= 65536) or nt >= 6: verify LM
  head at 4 rows 835 -> 730 us.  On 2560..10240-row tensors at n3..n4 it
  did not beat wgvt (x re-reads from L2 dominate; R is capped by VGPRs and
  warp count).  BF16 hc GEMVs: batching weight loads lost (about 4 of
  ~10.5 us per call is fixed launch/drain).
- The verify router (`route_dev_np_k`) runs one warp per token (router
  row in registers, same sum order and extraction as the one-warp
  `route_dev_n_k`, bit-identical — q4-test compares both): 54 -> 16 us
  per layer.  End to end it is within noise: the verify's critical path
  is the CPU-miss wait, not GPU work before it.  `Q4_ROUTE_NPAR=0`
  restores the one-warp kernel.
- What the CPU-miss wait is: ~21 token-expert misses per verify layer
  (~16 unique experts x ~2 MB on GSQ-RCO IQ3_S), 630-660 us per job
  (`Q4_STEP_PROF=1` prints `devroute jobs/misses/us per job`), i.e.
  ~50 GB/s — DRAM-bound: 12, 15 and 16 CPU threads give 656 / 634 /
  710 us.  Fewer misses (more L1, better placement) is the lever, not
  CPU kernels or thread count: L1 lending's ~1300 extra LRU slots took
  it to ~15 misses and 440-540 us per job.  The jobs share DRAM with the
  miss warm-up staging (every missed expert is re-uploaded to L1 at the
  next pass's tick, ~1 GB/cycle of H2D DMA): together ~80 GB/s, the
  measured ceiling (one 9950X3D CCD 61 GB/s, both 72-79).
  `Q4_DISP_STAGE=0` drops the staging: misses 15 -> 24 per job, per-miss
  cost 33 -> 26 us, verify unchanged; capping it to the k most-picked
  experts per job lost (k=4: 44.9, k=8: 48.8 ms vs 43.7).  Threads 10 /
  12 / 14: 558 / 500 / 468 us per job; pinning across CCDs
  (`Q4_CPU_PIN`) did nothing.
- Verify timeline (`Q4_VTL=1`, with `Q4_MTP_PROF=1`): GPU wall-clock
  stamps per layer plus the dispatcher's host stamps, printed as ms/pass
  after each request.  Agent prompt, ctx 32K (2026-10-12): pre-route GPU
  16.7, router 0.6, publish (two D2H copies) 1.4, dispatch 0.2, CPU job
  21.4, wait release 0.6, merge+combine 1.3 — seams are ~3 ms; the job
  and the pre-route work are the critical path.  The pre-route weight
  reads are ~3.3 GB/pass (~4 ms at bandwidth), so it is launch/latency
  bound, not bandwidth bound.
- The CPU-miss job is compute- and DRAM-bound about equally, not DRAM
  alone: `Q4_CPX_JOBBENCH=1` (synthetic 12-expert / 16-pair jobs on the
  real pool and image, run once at verify capture) gives 405 us cold vs
  350 us with the experts already cached; `Q4_CPX_REPEAT=1` (rerun each
  verify job back to back) agrees.  Kernels: f16 scales go through F16C
  (`cpx_h2f`; the out-of-line q4_f16_to_f32 per block made IQ4_NL 7.6 GB/s
  per thread, now 19), and Zen 4/5 take AVX-512 VNNI paths (`c5_*`,
  `Q4_CPX512=0` reverts) that are bit-identical to AVX2 (selftest
  compares): Q2_0 6.6 -> 13.7, IQ4_NL 19 -> 23 GB/s.  For the grid formats
  (IQ3_S, IQ3_XXS, IQ2_S) a vpgatherdd decode was no faster than the
  scalar loads (0.83-1.03x) — gathers are slow on Zen 5.  `c6_*` instead
  packs each grid entry into a 16-bit code (the grids hold 8 / 8 / 3
  distinct bytes), looks 32 indices up with vpermi2w over the code table
  and expands with vpermw + vpmultishiftqb + a pshufb value LUT
  (`Q4_CPX_C6=0` back to the gather): IQ3_S 1.79x, IQ3_XXS 1.53x, IQ2_S
  1.17x AVX2 per thread, still bit-identical.  Job bench: cold 425 -> 393
  us, warm 375 -> 311 — the cold job is now mostly DRAM time.  `Q4_STAGE_LATE=1` (queue the warm-up
  staging at job join instead of dispatch) measured no gain.
- Measured dead ends (2026-10-12, verify CPU-miss side):
  - Thread count / placement with the code-table kernels: job bench
    cold 515 / 384 / 387 / 377 us at 8 / 12 / 14 / 16 threads, CCD pins
    (6+6, 8+8) no better; software prefetch of the next rows (`_mm_prefetch`
    2-8 rows ahead) within noise.  The cold job is capped near 62 GB/s.
  - Verify PCIe split (route the top-R image-resident misses to a GPU
    window exec over the host image): the GPU reads the image at 13-22
    GB/s, ~150 us per expert against ~30 us of marginal CPU job time, and
    it queues behind the hit experts: R=1 / 2 verify 42.5 / 45.2 vs 41.3
    ms/pass.  Not kept.
  - Static L1 share: `q` 30 (the planner's value) beats 15 / 8 on the
    agent prompt (misses/pass 853 vs 921 / 1143).
  - Draft gate: `Q4_MTP_GLOG=<file>` logs each cycle's draft p's and the
    accepted count; replayed against the per-row verify cost (2/3/4 rows =
    30/39/45 ms) the best cumulative-p gate (0.45) beats today's pmin 0.5
    by ~1% — not the agent lever.  `Q4_VTL` also splits passes by cycle
    index: the first 10 cycles of a request miss 1.3-1.5x the later ones
    (post-prompt L1 contents), and the agent prompt stays ~20% above ja /
    code throughout.
- KV paging (`Q4_KV_PAGE`, default on with lending): the QSA KV and the
  indexer key caches (raw fp16 keys, pooled keys) are sized for the full
  ctx (4.3 GiB at 262144) but only reserve VA (`q4_hip_vreg_*`); arena 0
  is VMM-backed in 2 MiB chunks, and `n_pool` slots below the lend tail
  hand their chunks to the KV as the context grows (16384-position steps,
  `q4_expert_kv_donate`, a few ms each) and take them back at the next
  shorter request (`engine_kv_fit` -> `q4_sess_kv_trim` +
  `q4_expert_kv_reclaim`).  Every VA keeps one physical chunk across these
  moves (`Q4_VMM_PIN`, see the NaN entry: uploads into a VA remapped to
  another chunk landed in the old one).  The pool is blocked during static_load (static
  experts have no DRAM copy and must never sit there).  At full context
  the experts get what the fixed buffers left them; below it the rest are
  LRU slots: deployed config dynamic slots 2432 -> 4223, temp-1 decode
  ja 62.7 -> 69.5, code 64.7 -> 74.5, agent 52 -> 54.7 t/s.  VMM chunks
  read at hipMalloc bandwidth (stream ~870, slot gathers ~850 GB/s).
- hgemm scratch is lent too (`q4_hgemm_lend`, carved after the GDN scan
  buffers): the fp16 weight dequant (64 MiB), a whole chunk's activation
  slice (T x hc*n_embd fp16, was a fixed 32 MiB that split the 5.5K-row hc
  GEMMs into four) and the hipBLASLt workspace.  The private 128 MiB is
  freed at the first lend; outside a lend window the >= 64-row library
  path is off (the prompt is its only user).  Deployed: fits 3625 -> 3680
  slots, vfree min 2103 MiB; 34K prompt ~1840 -> ~1880 t/s.
- Decode L1 replacement (2026-10-13).  Three fixes to the verify-time
  staging, each env-revertible: (1) a deferred refill bound at
  `q4_expert_dev_tick` kept its evicted occupant's LRU stamp — the
  minimum that made it the victim — so every fresh expert was the next
  victim again: a few hundred slots churned while the rest of L1 kept the
  prompt's experts (`Q4_TICK_LRU=0` restores it).  (2) GPU-side hits
  never refreshed recency (the host only saw misses); verify publishes
  now carry each token's hit slots (`hs`/`hn` in the mailbox) and the
  dispatcher touches them (`Q4_HIT_TOUCH=0`).  (3) TinyLFU admission: a
  per-(layer, expert) routing frequency decayed per pass
  (`Q4_FREQ_DECAY`, 0.97) and a miss only displaces the LRU victim when it
  was routed more often (`Q4_STAGE_ADMIT=0` stages every miss).  The H2D
  link carries ~400 experts per pass and the staging DMA reads DRAM beside
  the CPU-miss jobs: admission cuts staging ~4x for ~20% more misses than
  (1)+(2) alone and still wins (less copy-stream / DRAM contention).
  Greedy ja (same text until a float-level flip ~100 tokens in): verify
  38.5 (none) -> 38.3 (2 alone: not measurable) -> 33.9 (1+2) -> 31.8
  ms/cycle (all) — the stamp fix and admission are the levers; the hit
  refresh is kept because it is cheap and correct.
  Temp-1 bench: ja 70 -> 73, agent ~55 -> ~70, code ~71 -> ~87, long8k ~69
  -> ~77 t/s.  `Q4_ROUTE_TRACE=<file>` dumps every verify layer's routing
  plus a slot snapshot per request for offline policy replay
  (`bench/l1sim.py <trace>`, local; note it modelled the intended LRU, so the
  real-vs-sim gap is what exposed the stamp bug; Belady
  bound on the agent trace: ~0.12 miss rate vs ~0.28 LRU);
  `Q4_SEED`/`Q4_TEMP` pin the chat CLI's sampler for A/Bs.  Open issue,
  plain decode (MTP off, `Q4_DEVROUTE=0`): the prompt's verify-style path
  starts the dispatcher and devtab, but `decode_hip` only ticks with
  decode-side devroute on, so defer[] fills at 512 and later misses just
  evict their victim.  Ticking there made it slower (40 -> 35 t/s: the
  tick's copy_wait puts each token's ~87 uploads on the critical path), so
  it is left as is.
- Verify hit kernels (2026-10-13; rocprof per verify layer, ja at ctx
  262144).  The IQ4_NL down ran one warp per output row: every row paid a
  32-lane reduction per token and re-read the hidden row (103-108 us for
  ~23 MB); `wmoe_iq4_ya_lpr` stages 64 rows of an expert with 16-byte
  loads and gives each row four lanes that decode once and dot every
  token (34 us).  Gate/up (IQ2_S / IQ3_XXS / IQ3_S, `wmoe_*_gu_wd`):
  each lane decodes its oct once into dot8_sg's sign-split words and dots
  every token, row read in place (no LDS copy), 69 / 77 / 89 -> 46 / 51 /
  53 us; the gate/up are instruction-bound (~1000 VALU per row-token at
  ~500 GB/s), not bandwidth-bound.  All window kernels loop a persistent
  grid over (expert, row group) items (the y = 64 plan-cap grid launched
  ~40 empty blocks per live one).  The plan builder (`win_pack_w`, one
  thread per (token, hit), ballot prefix) replaced a one-thread kernel
  (19 -> 5.8 us), and the shared-expert gate dot + sigmoid + add are one
  kernel (`q4_hip_sgate_axpy`, 12 -> 3.9 us).  `Q4_WIN_TILE` 0 restores
  the old kernels, 1 keeps warp-per-row math on staged tiles, 2 default;
  `Q4_SGATE_FUSE=0` reverts the gate.  q4-test compares the windows with
  the old kernels and the gate with a CPU reference; ppl 8200+800 vs the
  old kernels: window mean |dnll| 0.064, 3% flips.  gpuhits 11.5 -> 6.8
  ms/pass, greedy ja verify 32.5 -> 28.5 ms/cycle.  Temp-1 bench
  (serve.sh defaults, 2 rounds): ja 89-91, agent 68-71, code 93-100,
  long8k 81-85 t/s; vfree min 3217 MiB.
- Which side a verify layer waits for: `Q4_VTL` prints `vtl wait side`
  (layers where the CPU job ended after the GPU hits, and the summed
  excess per pass, and the reverse).  With the kernels above, ja:
  CPU-late in ~33 of 48 layers (~6 ms/pass excess), GPU-late in ~15 (~1
  ms).  Pre-route GPU work (~12 ms/pass, ~42% of the pass) is the largest
  fixed block.
- Verify pre-route kernels (2026-10-13, after the hit kernels).  Dense
  n-row GEMVs on int8 activations (`src/q4_mmvq.cu`, now OFF by default,
  `Q4_MMVQ=1` opts in — see the long8k language entry below for why):
  the float n-row kernels re-read the fp32 activation rows per
  output row, which at 4 rows cost more than the weight stream; the rows
  are now quantized once (int8, float scale per 32, int sum per 16) and
  each lane decodes 16 weights of one scale group once and sudot4s them
  against every row.  Kernel time at 4 rows: Q6_K 10240x2560 34.7 -> 27.9
  us (one row 26.2), 2560x6144 22.9 -> 18.4, IQ4_XS 44 -> 25, Q5_K 24.6 ->
  18-21, IQ4_NL 6.0 -> 3.2, the Q6_K lm head 734 -> 611; Q4_K 6144x2560
  ties, so the policy uses it from 3 rows on tensors under 4096 rows only
  (Q6_K / IQ4_NL from 2, IQ4_XS / Q5_K from 3).  `q4_mmvq_hold/release`
  share one quantization across the GDN input projections, QSA q/k/v and
  the shared expert's gate/up.  ppl window: mean |dnll| 0.059, 1.6% flips
  vs float (float vs float 0.058 / 2.4%).  A first try on llama.cpp's
  vec_dot_*_q8_1 (mul_mat_vec_q shape) was slower: it unpacks the weights
  once per activation row.  The hc up GEMV and mulsig_mean are one kernel
  (`hcup_mix_t`, `Q4_HCUP_FUSE=0`; bit-identical), and the GDN scan writes
  the MTP stash itself (`Q4_STASH_SCAN=0`; `Q4_STASH_CHECK=1` compares it
  with the copy).  Greedy ja pre-route 12.21 -> 11.45 ms/pass.  Temp-1
  bench (2 rounds): ms per cycle (tok/cyc / t/s) ja 32.4-33.3 -> 31-32,
  code 31.3 -> 30.0, long8k / agent unchanged within noise; the t/s rows
  (ja 78-95, agent 67-68, code 97-100, long8k 91-93) move with
  acceptance more than with these kernels; tok/cyc shows no systematic
  loss from the int8 activations.  Left: the
  route kernel is ~8 us of device time (17.6 under load) for 4 x top-10
  of 512, and the publish is two D2H copies (x rows, mailbox) — merging
  them needs the x rows and the mailbox adjacent in host memory.
- long8k answers in Chinese ("提示词..."), 2026-10-13.  Two causes,
  found with `Q4_NEXT_PROBE=1` (engine: logs the distribution after a
  forced first token through decode, prefill, 4- and 2-row verify and
  `Q4_NEXT_PROBE_MTP=K` fresh MTP cycles) and by diffing q4's prompt ids
  against Strata's own tokenizer + template:
  (1) The prompt was corrupted.  The server's JSON parser turned `\f` /
  `\b` into the letters f / b (the PDF-derived bench text has form
  feeds: the model read "fPart", "feverything") and did not combine
  `\uD8xx\uDCxx` pairs; q4 also did not `|trim` message bodies as the
  template does.  Fixed (b53b610): ids identical to Strata's; P(词 | 提示)
  0.15-0.69 -> 0.04-0.10; plain decode 3/40 -> 1/40 Chinese starts.
  (2) The int8 n-row GEMVs made a verify's rows depend on the batch size
  (type-specific row policy, int8 rounding): a 2-row verify gave row 0
  different numbers from layer 0's MoE on, and at this near-tie position
  later routing flipped — P(词) 0.23 (2 rows) vs 0.03 (4 rows; float
  kernels 0.034 for both).  MTP verifies that position with 2 rows when
  the draft gate stops the chain: 11/80 Chinese starts with int8, 4/80
  without (plain 1/40, Strata 0/40).  The float n-row kernels are
  bit-identical across n by design; keep that invariant — do not add a
  verify kernel whose per-row result depends on the row count.  Probing
  lessons: P(词) at that position swings 0.05 <-> 0.3 with tiny numeric
  changes, so 2-sample A/Bs on it are noise; the deterministic oracle was
  "layer-0 MoE output of row 0 bit-identical across batch shapes"
  (diagnostic dumps), and the confirmation was 80-120 samples.
  Temp-1 bench with both fixes (int8 off, 2 rounds): ja 82-84, agent
  58-73, code 85-107, long8k 85-92 t/s (Japanese both rounds), pf8k
  2015-2174, pf32k 2284; vfree min 3216 MiB, no NaN, no MTP fallback —
  within acceptance noise of the int8 run (ja 78-95, long8k 91-93).
- L1 replacement study (2026-10-13; `bench/l1sim.py`, traces with
  `Q4_ROUTE_TRACE`): a request touches ~11-13K (layer, expert) keys
  against 3554 dynamic + 1440 static slots; compulsory misses ~0.06,
  Belady ~0.11, the deployed LRU-victim TinyLFU ~0.25 of token-expert
  references.  LFU victims + token-weighted frequency replay ~10% fewer
  misses, but in the engine the extra uploads read DRAM beside the CPU
  jobs: without an admission margin verify got slower (more uploads), with
  `Q4_ADMIT_MARGIN` 1-2 it was inside noise (`Q4_EVICT_LFU`,
  `Q4_FREQ_TOK`, `Q4_ADMIT_MARGIN`, all off).  The miss side is bound by
  the shared DRAM bandwidth, not by the policy.
- Verify seams and QSA selection (2026-10-13, late).  (1) `qsa_topk_k`
  appended the threshold ties with thread 0 scanning the row from block 0
  — usually a single tie, so it walked nearly the whole row serially: 43
  us per verify at a 2.5K context, 307 us at 34K, ~850 at 95K (12 QSA
  layers: 3.7 ms/pass at 34K, ~10 at 95K).  `qsa_topk_p_k` (256 threads)
  picks each radix digit by a suffix scan, takes the count above the
  threshold from the radix's running count, and compacts above / tie
  blocks in ascending order in one coalesced ballot-prefix pass: 28 us at
  34K, same selection and order (q4-test: both kernels vs a CPU reference
  with ties; `Q4_QSA_TOPK_PAR=0` reverts).  Verify pre-route 34K 14.89 ->
  11.98, 2.5K 11.99 -> 11.40 ms/pass; at 95K verify is now 29.5 ms/cycle,
  prefill 2264 t/s.  (2) hc down + inject and ssm beta + alpha read the
  same rows: `gemv_bf16_bn2_t` runs both bf16 tensors in one grid, each
  row through the same body as `gemv_bf16_bn_t` (bit-identical, q4-test
  n 2..8; `Q4_BF16_PAIR=0`): pre-route 11.62 -> 10.99.  (3) The CPU-miss
  wait was a `hipLaunchHostFunc` node; `cpx_spin_k` (one thread) now
  polls a per-layer done word in hipHostMallocCoherent memory against the
  layer's device seq counter (written by the route kernel, so graph
  replays compare with the current publish), and the CPU rows the merge
  reads moved to coherent memory too — host-registered pages are L2-cached
  on the GPU, so a spinning kernel there never sees the host's write
  (`Q4_CPX_SPIN=0` restores the callback).  VTL release 0.54 -> 0.26
  ms/pass; ppl 8200+800 window vs the callback 0.059-0.064 mean |dnll|,
  2.5-3% flips (callback vs callback 0.056 / 2.75%).  Measured and not
  kept: the route kernel's softmax + top-10 is ~8 us of one wave's serial
  latency (standalone: launch 2.8, softmax +2.3, picks +5.4 us); DPP
  reductions instead of 64-bit shuffles did not move it.  A head KV of
  16384 cells (window 8192) cuts the 95K draft 4.37 -> 3.26 ms/cycle but
  acceptance 0.70 -> 0.61 (greedy): slower per token.
  Temp-1 bench with (1)-(3) (2 rounds): ja 85-88, agent 70-83, code
  94-103, long8k 87-93, pf32k decode 67-69 (was 58-61) t/s; pf8k
  1912-2165, pf32k 2114-2312; vfree min 3216 MiB, no NaN or fallback.
  `Q4_MMVQ=2` (opt-in): the int8 n-row GEMVs inside verify batches only,
  for every size 1..8 with a type / shape policy that ignores n (q4-test:
  each row bit-identical for n 1..8, explicit fmaf, no reassociation) —
  batch-invariant, unlike `Q4_MMVQ=1`.  ja pre-route 10.91 -> 10.69
  ms/pass after (2); not default (too small to re-run the long8k language
  check for).
- Review fixes (2026-10-13, night; scouts compared with Strata, all
  findings checked in the code).  (1) Temp > 0 sampling was biased: the
  draft gate stopped the chain when q(drawn draft) < pmin, i.e. proposed
  from q restricted to q >= pmin while the accept test divides by the
  unrestricted q, and the cut position came from p.  Emitted law at draft
  positions i > 0: sum_{d pass} q(d) [outcome] + P(fail) p; with pmin 0.5
  only the mode passes, so a confident head's mode came out with
  p_m (2 - q_m) (p = q = 0.6 / 0.4 -> 0.84).  The gate now compares q_max
  (a function of the distribution, not of the draw); `Q4_MTP_GATE_DRAWN=1`
  restores the old gate for A/Bs only.  Temp-1 numbers before f2d8b05 are
  inflated by the bias (code acceptance 0.78 -> 0.60-0.68) — do not
  compare across it.  Exact alternatives priced with `Q4_MTP_DTSTAT=1`
  (E[accept] on the same verified rows): draft temperature 0.7-1.15 and
  top_p 1 flat, Strata's argmax draft accepted with p(mode) 0.03-0.04
  lower; the gate threshold replayed over `Q4_MTP_GLOG` logs is flat over
  0.3-0.7.  q4-test checks the speculative step's emitted law.  (2) A
  restored session started the draft head from an empty KV, so after a
  prefix-cache hit (every agent turn but the first) drafts saw only this
  request's tokens.  The head's cells are saved beside each snapshot
  (`<snapshot>.mtp`, cells [0, kv_fill), tied by n_kv + id hash, written
  only when `kv_last` says they are real residuals ending at n_kv - 2 /
  n_kv - 1), loaded on a hit, and `q4_mtp_kv_bridge` writes the cell
  pairing the restored residual with the first new token.  Two-turn
  agent chat, turn 2: acceptance 0.66 / 0.66 / 0.59 vs 0.55 / 0.46
  without (`Q4_MTP_SNAP=0`).  The sidecar adds ~4 KB per cell to each
  save (~140 MB at 34K: +0.06 s on the checkpoint, which is on the TTFT
  path).  (3) Restore: a turn cut by its token limit saves its last id
  unfed (n_kv = n_ids - 1) and was rejected, so the next turn recomputed
  everything; it now resumes at n_kv.  An MTP turn cut mid-cycle commits
  drafts it never emitted (n_kv > n_ids, not restorable): the other file
  (.prompt / main) is tried.  (4) The repeat-loop stop (last 8 tokens ==
  the 8 before) cut a 9-column markdown table's |---| row or four equal
  diff lines as a normal stop; it now needs 256 tokens periodic with a
  period <= 64 and reports length.  (5) BPE was quadratic per
  pretokenized piece (16K CJK chars without punctuation 12.6 s, 64K
  letters 20.6 s); heap + linked list, ids identical on 3010 strings and
  the HF fuzz set.  (6) Server: sampling fields left out of a request kept
  the previous request's values (now the startup defaults; top_k is read);
  a failed turn answered 200 with the previous usage (now 400
  context_length_exceeded / 500, an error chunk when streaming).  (7) The
  KV page pool was sized from bytes + 16 chunks, but the reserve maps
  whole chunk spans per region and layer: any ctx off a large power of
  two failed every prompt (ctx 4096: 72 chunks needed, 50 pooled; also the
  old ppl abort at ctx 8192); `q4_kv_page_chunks` counts the spans, ctx
  262144 unchanged (1752 slots).  Measured, no gain: THP on the DRAM image
  (job bench cold 381/390 vs 382/365 us), a non-atomic phase-B add
  instead of the CAS (388/386 vs 381-390 us).  Deliberate, not a bug: the
  splice keys a conversation by its first user message and keeps the
  saved system prompt (OpenCode rewrites it every turn).  Left as is:
  `stop` strings are not read, per-token strstr for <tool_call> over the
  answer (tens of ms per 8K tokens), no chunked request bodies, a
  single-threaded accept loop without socket timeouts.
  Temp-1 bench (2 rounds, exact sampling + the above): ja 93-99, agent
  66-75, code 91-94, long8k 88-93, pf8k / pf32k decode 79-83 / 66-82,
  prefill 2007-2159 / 1946-2305 t/s; vfree min 3214 MiB, no NaN or
  fallback, long8k Japanese both rounds.
- Spin-wait timer (2026-10-14, cbd36bc).  `cpx_spin_k` timed its 60 s
  give-up with `__builtin_amdgcn_s_sendmsg_rtn(131)`, which is the 32-bit
  `s_sendmsg_rtn_b32`: the 100 MHz realtime counter wraps every ~43 s,
  and a wait that straddled the wrap "timed out" at once (~7e-6 per wait).
  The merge then read unfinished CPU rows and the request aborted with
  "CPU-miss pipeline wedged" although every job had completed
  (stuck-layer -1, floor == seq, no gap in the log).  Benches (a few
  thousand waits) never showed it; a long 4-row ppl (~80K waits) did, 5
  of 6 runs.  Use `__builtin_amdgcn_s_sendmsg_rtnl` (b64) or
  `wall_clock64()` for any elapsed-time test on the GPU.  Repro:
  `Q4_PPL_WIN=4 Q4_PPL_PREFILL=2000` ppl over 9000 tokens.
- `Q4_TAIL_SKIP=r` (c52b954, opt-in): Strata's STRATA_ROUTE_TAIL_SKIP —
  a missed expert that every token of a verify window ranks r-th or lower
  is neither computed on the CPU nor staged (weights not renormalised).
  The batched route kernels publish each miss's rank (mailbox `rk[64]`).
  At r = 7 about half the missed pairs go: CPU job 8.6-12.0 -> 4.4-6.2
  ms/pass, 4-row verify -12..-16% (temp 1).  It changes the output: ppl in
  4-row windows mean |dnll| 0.15-0.18 vs 0.06 run-to-run, top-1 flips
  7-9% vs 3%, mean nll within +-0.002 over 3-5K scored tokens.  The CPU
  path itself is as accurate as the GPU one (4-row ppl: int8 CPU 1.3024 /
  1.3037, float CPU 1.2943, all-GPU 1.3012 — run-to-run spread).
- Strata reference (2026-10-14, Strata 0.1.43.1).  The
  0.1.42 "HIP prefill regression" (pf8k ~900 vs 0.1.39's ~1340) was our
  build: the 0.1.42 build-hip had `STRATA_PREFILL_MMQ=OFF` (the CMake
  default; the 0.1.39 build had ON) and the config's runtime
  `STRATA_PREFILL_MMQ=1` cannot enable a path that was not compiled.
  Build with `-DSTRATA_PREFILL_MMQ=ON` (and `-DSTRATA_MMQ_KQUANTS=ON` to
  try `STRATA_DENSE_MMQ`).  Tuning on top (bench pf8k / pf32k):
  `STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096`
  +3-5%, `--prefill auto:32768` +20-28% (1342-1360 / ~1500 -> ~1730 /
  ~1800); `STRATA_DENSE_MMQ=1` -12%; the ring default = 96 anyway.
  Decode: `--adapt-every` default vs 0, `--pool-workers` 12/15/16,
  `STRATA_SH_STREAM=1`, `STRATA_ADAPT_OVERLAP=1`, `--expert-cache` +400
  all inside noise (ms per cycle); `--pcie-frac` default 0.55 -25%
  (keep 0); `STRATA_ROUTE_TAIL_SKIP=7` (an approximation, not default on
  gfx1100) +12-20%.  Strata leaves ~3.05 GiB VRAM free at this config
  (q4 ~3.2).  bench.py's prompts change per run, so greedy runs are not
  same-text: compare ms per cycle (tok/cyc / t/s), not t/s.
  Temp-1 bench, 2 rounds each (2026-10-14): q4 exact ja 73-89, agent
  69-83, code 97-98, long8k 90-92, pf8k / pf32k prefill 1907-2150 /
  2074-2299; q4 + Q4_TAIL_SKIP=7 ja 102-108, agent 83-84, code 95-114,
  long8k 100-110; Strata tuned (above) ja 57-58, agent 56-71, code 61-65,
  long8k 63-68, pf8k / pf32k 1661-1750 / 1763-1779; Strata tuned +
  STRATA_ROUTE_TAIL_SKIP=7 ja 67-74, agent 64-69, code 78-79, long8k 77,
  1736-1779 / 1758-1795.  vfree min q4 3209-3213, Strata 3035-3041 MiB;
  no wedge or MTP fallback.
- Decode expert prefetch (`Q4_DPF=1`, off; measured dead end
  2026-10-13).  Each verify layer L runs layer L+1's router on its MoE
  input (in the graph, after the hits), `pred_sel_k` keeps the best
  `Q4_DPF_N` (4) experts by best-token probability that L+1's devtab
  lacks, and the dispatcher uploads them once L's CPU job joined
  (`Q4_DPF_AT=0`: on arrival) into quiesced slots or slots of layers the
  GPU already passed (the next tick waits for the copy stream); a
  prefetched binding enters `Q4_DPF_AGE` (1500) clock ticks old.
  Offline (`Q4_ROUTE_TRACE_X=1` adds the router inputs to the route
  trace): L+1's router on L's input recalls ~0.64 of L+1's experts per
  token, half of the misses within the best 20 candidates.  In the engine
  (greedy ja, same text): 85% of the uploads hit, misses/pass at 4 rows
  472 -> 371, CPU job 253 -> 220 us, but the prediction adds ~1.1
  ms/pass to the GPU side and the uploads ~0.35 ms to the pre-route work:
  verify 28.74 -> 28.44 ms/pass, inside run-to-run noise.  The uploads
  read the same DRAM the jobs read (a moved read is still a read) and a
  job keeps a per-job floor, so even a free prediction buys ~1 ms/pass.
  A side stream forked around the MoE graph (event fork/join per layer)
  and the prediction ahead of the hits measured the same or worse.
- Lend refill from the prompt (`Q4_LEND_PROMPT`, default on): the
  prefill adds each layer's per-expert token counts (the route/assign
  `ek` table) and `q4_expert_lend_end` refills the lent tail with the
  prompt's most-picked experts not already in L1, before the warm ranks.
  Temp-1 bench (ja/agent/code x3): misses/pass in cycles 0-9 856 -> 818,
  10-29 673 -> 605, 30+ 604 -> 590.
- devtab writes (residency clears/publishes) are batched: one in-order
  scatter kernel per flush on the copy stream instead of one
  `hipStreamWriteValue32` per entry — those were ~940 copy-queue
  dispatches per verify cycle and stretched the compute queue's gaps.
  Stage batches flush at their end, the tick per layer group, a clear
  before its slot is overwritten.  Verify 45.7 -> 43.7 ms/cycle.  The GDN
  stash (4 copies) is one multi-segment copy kernel, the conv copy-back
  one strided copy, the CPU-miss merge one axpy (-0.65 ms).
- Verify QSA attention (2..8 rows): the decode's split-KV kernel with
  the token row on grid.z (`qsa_small_split`, all rows in one launch,
  S capped so n_kvh x S x rows fits the 4 x 128 partials) and a
  warp-per-pooled-block indexer score (`qsa_score_nw_k`).  They ran the
  prefill kernels — one block per (kv head, row), 8 blocks for 4 rows —
  ~750 + ~100 us per QSA layer at a 2.4K context, i.e. ~10 ms of a
  2.4K-prompt verify that the usual 44-token chat probe never shows
  (profile with the bench's prompts).  ja verify 53.5 -> 44-45 ms.
  `Q4_VER_SPLIT=0` / `Q4_QSA_SCORE_NW=0` restore the old kernels.
- Verify QSA attention runs `gqa_split_wmma_k` (`Q4_QSA_SWMMA=0` reverts to
  the scalar split): the prefill WMMA tile kernel cut into split-KV blocks
  (grid kv head x split x row, `Q4_SWMMA_TPS` 64-position tiles per split,
  default 1) leaving the same (m, l, acc) partials for `gqa_dec_comb_k`.
  The scalar `gqa_dec_split_k` ran a 2K selection at ~20 GB/s of KV: 116
  us for one row, 344 us for four (`bench/qsabench.c`, local, CPU-checked); the
  WMMA split takes 69 us for four (fp16 operands, like the prompt path).
  ja verify pre-route 17.2 -> 14.3 ms/pass, verify 44.6 -> 41.8 ms.  A
  single-tile block still costs ~25 us of serial latency (q load, K/V
  rows, two WMMA phases, softmax), so wider batches are the cheaper
  next step; the n_tok == 1 decode paths still use the scalar kernel.
- Verify layer seams: the MoE leaves its CPU-miss rows to the caller
  (`Q4_MERGE_FUSE=0` restores the in-graph axpy) and every hc combine of
  the batched layer loop also writes the grouped rms the next hc mix
  starts with (`Q4_COMB_FUSE=0`): the attention combine with the FFN-side
  rms, the MoE combine (+ CPU rows) with the next layer's attention-side
  rms unless PLE rewrites the residual first.  The 256-thread
  `hc_comb_rms_k` walked each group in two strided passes (~44 us on a
  4-row verify's 16 blocks), so the n-row launcher uses `hc_comb_rms4_k`,
  one float4 per thread (`Q4_COMB_V4=0` reverts; its rms sum order
  differs from grouped_rms_k at float level).  Pre+merge+combine 16.0 ->
  15.1 ms/pass.  `Q4_VTL` also splits passes by row count: agent prompt,
  2 / 3 / 4 rows = 30 / 39 / 45 ms (~21 ms fixed + ~29 us per
  token-expert miss).  `Q4_MTP_PCHAIN=p` additionally stops the draft
  chain once the product of the drafts' p falls under p (off by default).
- Draft / tail trims (2026-10-12, ja chat): the KV repair's eh_proj ran
  n x hc = 12 rows through the batch GEMV (~0.5 ms; `mmd_n` now chunks
  9..32 rows into the 8-row kernels: repair 0.36 -> 0.15 ms/cycle); the
  verify rows' top-k runs each wide pass for all rows in one launch
  (grid.y = row, `Q4_TKR_BATCH=0` per row: logits 1.14 -> 0.97); the head
  router's `softmax_topk_k` picks with warp 0 instead of a serial thread-0
  scan (bit-identical, ~70 -> few us per step); the head attention
  (`q4_hip_qsa_decode_kv`, f32 head KV, device-counted length) runs the
  WMMA split with in-kernel tiles per split (`Q4_KV_SWMMA=0` reverts).
  Draft 3.68 -> 3.35 ms/cycle.  The commit replays the accepted prefix
  with one `gdn_scan` per layer over the stashed inputs (`Q4_COMMIT_SCAN=0`:
  per-token gdn_apply_d; `Q4_COMMIT_CHECK=1` logs max |dS| between them,
  <= 1e-6 of ~8): commit 0.62-0.85 -> 0.47 ms/cycle, the rest is the
  113 MB snapshot restore.  The snapshot itself is written by each
  layer's verify scan as it loads S / conv (`q4_hip_gdn_scan_snap`,
  `Q4_SNAP_SCAN=0`: the old 113 MB copy ahead of the layers;
  `Q4_SNAP_CHECK=1` compares the two — 0 differing values).  The rest of a draft step is the full-vocab
  Q6_K head GEMV (521 MB, ~600 us, at bandwidth).
- Tree drafting does not pay here (`Q4_MTP_TREESTAT=1` logs it per
  request): q4 verifies by speculative sampling, so a rejected position
  already yields a resampled token and an alternative candidate only gains
  through its own continuation.  Temp-1 bench (ja/agent/code x2): ~0.35-0.5
  rejections per cycle, a second candidate (the draft distribution without
  the first, against the residual) accepted with p ~0.33 (~0.4 at depth 0),
  i.e. +~0.11 tokens/cycle (+4%) for one extra depth-0 row, while a verify
  row costs ~6 ms (2 / 3 / 4 rows = 30 / 35-40 / 39 ms) — the CPU-miss work
  grows with every row.
- Verify GDN scan (`gdn_scan_k<W>`, `Q4_GDN_SCAN_W=0` reverts): S and its
  snapshot move through LDS in 32-row x 128-column chunks with 16-byte
  coalesced accesses (the per-thread quarter-column loads hit 4 rows x 8
  columns per wave instruction), the q/k l2 and output rms sums reduce per
  wave (2 barriers per token instead of ~20), conv taps and the <= 8 token
  inputs are loaded ahead of the token loop.  29 -> 22 us per layer at 4
  rows, verify pre-route 13.46 -> 13.09 ms/pass; commit check max |dS|
  ~1e-6.  Measured dead ends for the rest of the pre-route (13 ms/pass,
  mostly kernel time, not launch gaps): x staged in LDS for the n-row
  GEMVs (bit-identical rows keep the per-lane x traffic; Q6_K 2-3x slower
  at one block per CU), R rows per block for the hc down GEMV (x re-reads
  were not the limit: ~600 GB/s at 320 blocks either way), kq16 R=4 at
  n3-4 (mixed +-5%).
- Verify tail (`q4_sess_verify_topk`): the lm head over all verify rows in
  one batched pass plus a device top-k per row, one sync — no 0.95 MiB D2H
  and host scan per row (~3 -> ~1.7 ms/cycle).  `Q4_MTP_VTAIL=0` restores
  per-row `q4_sess_logits_at`.
- Profile a draft or verify step with `rocprofv3 --kernel-trace` around
  `q4 chat` (`scripts/q4safe.sh` with `Q4_SAFE_PIPE=1` and
  `Q4_BIN=<wrapper that execs rocprofv3 ... -- ./q4 "$@">`; q4safe's
  pipe mode forwards only `Q4_*`/`HIP_*`/`ROCM_*` env, so hardcode the
  output dir in the wrapper).

## Images (vision)

- The encoder (mmproj: 27-layer ViT + qwen3vl_merger, 2560-wide output,
  no deepstack) runs on the CPU in a helper, `q4-vision`
  (tools/vision/q4_vision_helper.c, llama.cpp mtmd at the commit pinned in
  scripts/build-vision.sh): weights and buffers in DRAM, no VRAM.  The
  server (src/q4_vision.c) starts it at the first picture inside q4's own
  scope (~1.1 GiB RSS; refused under `Q4_VISION_NEED_GIB` of MemAvailable,
  default serve.sh's watchdog floor `Q4_SAFE_FLOOR` + 2 GiB), talks Strata's strata-vision protocol (`ENC <img> <out>`
  -> `OK n nx ny ms`, SVE1 file), and keeps encodings in an LRU keyed by
  the image bytes + token limits (`Q4_VISION_CACHE_MIB`); entries of the
  running request are never evicted.  The child closes every inherited fd
  (`closefrom(3)`): it once held a client socket open after the server
  closed it, and the client hung.
- Flash attention: the helper turns it off on AVX-512 CPUs (as Strata
  intends).  Off is FP32 attention and here also the fast path: 300
  tokens 0.84 s, 972 tokens 6.0 s, against 2.2 / 19.9 s with it on.
  Strata's own strata-vision ran with it on on this box (output 9.5% rel
  L2 from FP32).  Both binaries give bit-identical output at equal settings.
- Engine: the template's one `<|image_pad|>` per picture expands to the
  picture's n cells (`images_bind`, q4_engine.c).  The model reads the pad
  id there (the PLE hash takes `ple.image_token_id` = the pad) and
  `q4_sess_set_embd_rows` makes `embed_hip` upload the picture's row
  instead of the token embedding (prompt and decode paths; the MTP head
  embeds the pad token like Strata's draft_kv).  The session files key an
  image cell by `img_key(hash, j)` (>= 0x40000000, never a vocab id), so
  two pictures of one size are different prefixes; a splice whose kept
  cells do not match the request's pictures falls back to a full render.
- M-RoPE: every rope kernel (q/k prep, prep3, indexer pool) reads its
  rotary position from the `g_mrope_tab` device symbol when set: pair p
  takes stream p % 3 of the cell's (t, h, w) (mtmd: t = p, h = p + y,
  w = p + x; text after a picture continues at p + max(nx, ny)); cells past
  the prompt use cell - shift.  A symbol, not an argument, so captured
  graphs see it.  Null for any request without pictures (bit-identical
  text path; q4-test checks identity == none and a table vs a CPU ref).
- Two bugs found on the way (not vision-specific): `find_key` scans past
  the end of the object it is given into its siblings, so a message
  without `tool_calls` / `reasoning_content` took a LATER assistant
  turn's (every agent history with a text-only assistant turn before a
  tool-call turn was rendered with duplicated tool calls), and an image
  part's lookup of "text" read the next part; message and content-part
  lookups now run on a copy bounded to the object.  The tokenizer's
  punctuation rule swallowed "<|" after "." ("answer.<|vision_start|>"
  lost its special token); text now splits at special tokens before
  pretokenization, as HF does with added tokens.
- Checks that discriminate: OCR of llama.cpp's tools/mtmd/test-1.jpeg
  (headline, dateline, column text), a red circle in each corner of
  800x600 (h/w sectors and the position continuation), two pictures in one
  message, five squares counted; a second turn hits the encoding cache and
  the session prefix; the same text with a different same-size picture
  misses.

## Rope layout and sampling defaults (2026-10-14)

- qwen4exp's rope is NEOX-style (llama.cpp IMROPE): rotary pair p is
  elements (p, p + n_rot/2).  Every q4 kernel rotated (2p, 2p+1) until
  4e9616c (attention q/k, fused prep3, indexer q, pooled indexer keys, so
  also the MTP head; the CPU path too).  Because q and k shared it,
  short-range attention mostly worked and the 8K-window ppl never showed
  it.  ppl over 40K tokens of unique text: nll -8.0% / -4.8% (two texts),
  the gap growing with depth.  `Q4_ROPE_NEOX=0` restores the old pairs and
  session-file version 5 (new files are 6: their KV holds the rotation).
  When checking long-context quality, use text without near-duplicates:
  bench/real16k.txt + models/eval/*.txt contain two versions of the Strata
  paper and of q4_mtp.c, and the model's confident copying of the older
  version reads as nll 4-7 at 20-34K (an artifact, not decay); a repeated
  block copies at nll 0.00 at every depth.
- OpenCode (2.0.27) sends neither temperature nor top_p for the q4
  provider: an agent's `temperature` / `top_p` are not sent, and a model
  entry's `"temperature": true` / `"attachment": true` / `"reasoning"` are
  "unsupported legacy setting"s it drops (`--print-logs` shows the
  diagnostics; `modalities` is what enables pictures, `variants` with
  `reasoningEffort` send reasoning_effort).  So the server's defaults ARE
  the agent's sampling.  They follow the request's thinking mode, from the
  Qwen3.8 card: thinking 1.0 / 20 / 0.95 / presence 0 (general.sampling),
  thinking off 0.7 / 20 / 0.8 / presence 1.5; the chat log line prints the
  effective values.  `presence_penalty` (OpenAI semantics: every token the
  turn emitted loses it once) is read from requests; the top-k fetches 44
  extra candidates and keeps top_k after the penalty, and MTP verify rows
  see the drafts ahead of them (`q4_pen_mark/push/rewind`), so speculative
  sampling emits what plain decode would (q4-test).  The penalized top-k is
  itself exact only while at most 44 of its 64 candidates were emitted
  before (long thinking-off answers can pass that).  `Q4_DUMP_OUT=<dir>` writes each
  request's body, messages, prompt ids and raw generated text (think,
  answer, tool-call XML).
- Persistent NaN after long generations (found 2026-10-15): copies into a
  remapped VA.  At a request start `engine_kv_fit` trims the KV
  (`q4_sess_kv_trim`: chunks above the prompt back to the pool) and takes
  expert slots back (`q4_expert_kv_reclaim`: their arena VAs mapped to
  pool chunks).  With a LIFO pool a VA came back on a different physical
  chunk than it had before, and the first uploads into the reclaimed
  slots landed in the chunk the VA had held: slot 4082's bytes in slot
  4081's chunk at the same intra-chunk offset, chunk 5249 / 5250 tails
  swapped, the rest of the target stale.  Read back right after the copy
  (`Q4_RECLAIM_CHECK`'s upload watch) the bytes were already wrong.  Not
  SDMA-specific (`HSA_ENABLE_SDMA=0`: 60 of 630 uploads wrong), not fixed
  by a device sync + 500 ms after the remap (1 of 210), no kernel log, no
  duplicate chunk in the VMM audit, devtab consistent; a standalone
  remap toy (one process, 256 MiB) did not reproduce it.  Stale KV bytes
  as IQ weights decode to inf/NaN, so every later token routed to that
  expert gave NaN until restart (the host believed the slot loaded);
  neighbours got another expert's valid bytes, i.e. silently wrong
  output.  Trigger: any reclaim (a shorter request after one that grew
  the KV - the next OpenCode session after a long one reclaimed 539-747
  slots - or the pool's leftover chunk, 1-2 slots most requests).  Fast
  repro (local `fastrep.py`): a 16.1K-token prompt whose decode crosses
  16384, then a short request - every reclaim went wrong (47 of 630
  uploads, 3 NaN in 6 reclaims).
  Fix (`Q4_VMM_PIN`, default on): a VA only ever maps one physical chunk.
  An arena chunk maps its original handle; a KV chunk maps its home - the
  handle it got at its first map, the free one of the highest arena
  chunk - and the KV maps one 16384-position step at a time, so higher
  steps own lower arena chunks and a trim frees exactly the chunks above
  the frontier.  KV growth asks the donor until every needed home is in
  the pool (`q4_hip_vreg_short`); a reclaim maps only chunks whose own
  handle is free (`q4_hip_arena_remap_need` returns UINT32_MAX
  otherwise).  Pinned, the fast repro gave 0 of 2309 uploads wrong over
  22 reclaims of ~105 slots, the same binary with `Q4_VMM_PIN=0` 47 of
  630.  `Q4_KV_RECLAIM=0` keeps the KV's chunks until restart
  (grow-only, 5c91caa); `Q4_VMM_PIN=0 Q4_KV_RECLAIM=1` is the old path.
  Also: a request that sees NaN logits re-copies every L1 slot that
  differs from its DRAM bytes (`q4_expert_slot_repair`), saves no session
  and deletes the prompt checkpoint it wrote (that checkpoint made the
  next hit fail at gen 0 without a forward pass).  Diagnostics:
  `Q4_RECLAIM_CHECK=<n>|end` (reclaimed slots vs the DRAM image per step /
  at request end, plus a read-back of each slot's first upload),
  `Q4_RESTORE_VERIFY=1` (session restore read back),
  `Q4_RECLAIM_SYNC` / `Q4_RECLAIM_SLEEP_MS` (the wait experiments).
  Quality runs before the fix (pelican, Pagoda low / xhigh, A8/A16) ran
  with the hazard present.
- A8 vs A16 for MTP acceptance (2026-10-14): decode quantizes the MoE
  activations to int8 (GPU windows `Q4_WGQ8`, CPU misses `Q4_CPXQ8`; the
  dense GEMVs are fp32, the prompt path fp16).  Both off (float
  activations, the pre-window kernels): E[accept] (`Q4_MTP_DTSTAT`, temp
  1, 4 x 4000-token pelican generations) 0.778 / 0.805 / 0.834 vs int8
  0.793 / 0.825 / 0.807 - no measurable gain; decode 77-82 vs 89-95 t/s.
- Prompt parity with the reference: a request with `"q4_render_only":
  true` builds its prompt (no session splice), writes
  `Q4_DUMP_OUT/render-<n>.ids` and returns empty; compare against Strata's
  `serve/frontend.py` (`openai_to_messages` + `ChatTemplate` +
  `tools/strata_tokenizer.py`) with the tool objects as sent.  A 38-turn
  OpenCode Pagoda session (tools, reasoning, tool results, a picture) is
  id-identical apart from the picture's expanded cells.  Strata itself
  unwraps `{"type": "function", "function": ...}` before rendering;
  transformers / vLLM do not, and neither does q4.
- The compact tools preamble (`Q4_TOOLS_COMPACT=1`, the default until
  2026-10-14) dropped every key named description / title / pattern /
  format / default / ... at every level - including PROPERTIES of those
  names (glob / grep lost `pattern`, question / subagent `description`,
  webfetch `format`, while `required` still listed them) - and every tool
  and parameter description (OpenCode's edit / read / write rules), and
  put a terse format note after the system text: 7.2K vs 9.7K tokens for
  OpenCode's 12 tools.
- Engine parity at depth (`Q4_DUMP_FIRST_LOGITS=<path>` writes each
  request's last-prompt-position logits; Strata's
  `STRATA_DUMP_FIRST_LOGITS` the same): thinking-off prompts over prefixes
  of src/q4_rocm.cu, 1K-50K tokens, two questions per depth, identical
  prompt ids: top-1 equal at 24/24 positions, KL(Strata || q4) 0.0003-0.09
  (mean ~0.016) with no trend in depth; each engine reproduces its own
  logits exactly (Strata 0.1.43 at the deployed config gave the same
  prompt token counts as q4 for reasoning_effort none).  Positions right after `<think>\n` are useless for this
  (p(top-1) ~ 0.999).
- Session cache slots (2026-10-15, `Q4_SESS_SLOTS`, default 4; 1 = the old
  single file): slot 0 is `Q4_SESS_CACHE`, slot k `<path>.<k>`, each with
  main / .prompt / .mtp files (~120 MB each even for a short prompt: the
  GDN state; plus ~10 KB per KV cell).  A request restores the longest
  exact prefix over every slot's main and .prompt, splices the newest slot
  of its conversation, and saves to that slot - anything else to an empty
  or the least recently used slot (never the restore source).  "Same
  conversation" = same first user message AND a common prefix of at least
  half the render up to that message: keyed by the message alone,
  OpenCode's title request (the session's first user message under its
  own short system prompt) became the snapshot the next agent turn
  spliced onto - system prompt and tools gone (the single-file cache had
  the same hazard whenever the title request saved last).  Before the
  slots any other conversation in between (a subagent, the title, another
  session) overwrote the main conversation's snapshot and its .prompt, so
  the next turn re-read the whole context.  Note: the deployed model
  wrapper (`models/.../serve.sh`) sets `Q4_SESS_CACHE` itself, so test
  servers started through it share the production cache files.
  OpenCode in practice (2026-10-15, Pagoda xhigh, 226 requests): 224
  spliced; misses only where the earlier history changed - a new session,
  an auto-compaction (the summary request itself restored 222K of 223K
  from the slot; q4 wrote an accurate 6.7K-char summary and the agent
  carried on from its "Next Move"), and OpenCode replacing older pictures
  with "[This image was removed ...]" text: the kept image cells no
  longer match the request's pictures (`images_bind`: "image k does not
  match the cached conversation") and the splice used to be dropped (~100K
  tokens re-read).  Now kept cells of a picture the request no longer
  carries stay as they are: the server keeps a registry of every picture
  it bound (first cell's key -> hash, n, nx, ny; also appended to
  `<Q4_SESS_CACHE>.imgs` so it survives restarts), the kept run advances
  the M-RoPE position by its extent, and its cells need no rows because
  they are already in the restored KV.  If no restore reaches past them
  the request falls back to the full render (`img_orphan_end`).  Test (4.7K
  system, pictures A / B / C in later turns, the client replacing A with
  the removal text): turn 3 spliced (2016 of 2238 restored), answers
  right; its first logits vs the same conversation with A still sent:
  same top-1, KL 0.0013 (prior answers differed by one token).
- MTP and the session cache: an accepted im_end / eos draft now ends the
  verify window (`q4_mtp_set_stops`), and the engine saves the stop token
  with the turn's ids.  Before, drafts accepted after the stop were
  committed but never emitted (n_kv > n_ids), the main snapshot could not
  be restored, and every agent turn re-read the previous turn's whole
  output from the .prompt snapshot (after a 23K-token reasoning turn, 33K
  tokens again).  Temp-1 bench after these changes (2 rounds; its thinking-off
  requests got presence 1.5, bench.py now sends 0): ja 83-84, agent 64-70,
  code 97-106 t/s, pf8k / pf32k prefill 1892-2152 / 2121-2301.
