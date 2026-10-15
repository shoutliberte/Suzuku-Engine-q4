# Suzuku-Engine-q4: RDNA3 + DRAM offload for Qwen3.8-Flash-Next

> **Status (2026-10-15):** this document was written for the IQ3E build and
> last revised on 2026-10-12. The default model is now the GSQ-RCO IQ3_S
> quant (the same IQ3_S expert payload), MTP is on by default, and later
> work (L1 lending, KV paging, the NEOX rope fix, session slots, vision) is
> logged in [AGENTS.md](AGENTS.md). The README has the current numbers.

Run **Qwen3.8-Flash-Next IQ3E** (pentacoxian IQ3E-Q8D-MTP, one 80 GiB GGUF)
on a consumer RDNA3 box — the reference machine is an RX 7900 XTX
(24 GiB, gfx1100) with ~60 GiB RAM and a fast NVMe — with **prefill speed
and context** first. A general-purpose DRAM-offload runtime is too
slow for this box, and the box is not the 2×32 GB V100 that quant was
packed for. External work this engine refers to is listed in the README.

The GPU and DRAM are shared with a live desktop session. An OOM or a
VRAM exhaust kills the session, so q4 runs as a single process under a
hard memory cap and always leaves RAM and VRAM headroom.

## Where weights live

| weights | size | place |
|---|---:|---|
| Routed experts (48 × 512, top-10) | 45.29 GiB | one tight DRAM image (`Q4_EXPERT_RESIDENT=1`) |
| Dense + shared experts | 4.98 GiB | VRAM |
| PLE n-gram table | 26.82 GiB IQ4_NL | SSD `pread` (~1.4 KB/token). Not in DRAM or VRAM |
| QSA KV, default q8_0 | 0.80 GiB at ctx 65536 | VRAM. f16/f32 only if `Q4_KV` says so |
| GDN recurrent state | 0.11 GiB | VRAM, independent of ctx |
| MTP block inside the GGUF (blk.48) | ~2.9 GiB | peeled from the main forward. Usable as an opt-in draft head via `scripts/mtp-extract.py` + `Q4_MTP=<head.gguf>` (see "MTP (speculative decoding)" below) |

Per-expert mean is 1.89 MiB (max slot 2.22 MiB). A uniform slot array would
be ~53 GiB and does not fit. The resident image is the sum of each layer's
real expert size. GPU upload reads that image directly. There is **no host
mirror** of the GPU slots (that mirror was 9–13 GiB and, together with the
45 GiB image, OOM-killed the host).

`mlock` of the resident image is skipped. `RLIMIT_MEMLOCK` is 8192 KB, and
pinning 45 GiB would freeze the box if the limit were raised. The image is
swappable. Do not browse heavily or delete large files while q4 runs.

UD-Q4_K_XL (71.7 GiB routed, ~1.40 GiB/token) does not fit in 60 GiB RAM.
It remains on disk as the old SSD-offload path. It is not what `serve.sh`
starts.

## VRAM and DRAM floors

```
VRAM planner, GSQ-RCO IQ3_S at ctx 262144 (2026-10-11 server log)
  dense + shared          4.20 GiB
  KV q8 + GDN             3.30 GiB
  QSA indexer             1.26 GiB
  device scratch          3.62 GiB (prefill chunk 8192, Q4_PREF_CHUNK)
  reserve                 2 GiB (Q4_VRAM_RESERVE_GIB, floor 1 GiB)
  GPU expert slots        3545 × 2.48 MiB = 8.79 GiB
                          (2510 with the MTP head: 790 slots reserved)
  (IQ3E, 2026-09-27, 2048-token chunks: dense 4.98, scratch 0.25,
   5168 × 2.22 MiB slots = 11.21 GiB)

DRAM
  resident experts        45.29 GiB
  hard floor              4 GiB   (Q4_DRAM_RESERVE_GIB, cannot be set lower)
  left on the 2026-09-26 run: 11.0 GiB at alloc, 10.6 GiB minimum while running
```

Since 2dcea2f the device scratch is not a separate allocation (L1
lending). The arena gets `n_tail` extra slots (1346 at chunk 8192, the
size of `q4_sess_lend_bytes`): the prompt's row buffers, `d_pws`, the MMQ
weight gather and the GDN scan buffers are carved out of them while a
prompt runs, and decode uses them as LRU expert slots, refilled from the
DRAM image (~100 ms). The measured-free clamp keeps reserve + the
non-lent scratch + the MTP head estimate + 1 GiB free. At ctx 262144
with the MTP head this gives 3686 slots (base 2340 + tail 1346), and
the minimum vfree in the bench is ~2380 MiB. Before this the base was
2510 with the scratch on top, and vfree sat at ~1500.

The old 6 GiB reserve left most of the card empty. Idle display is 0.9 GiB
and sits outside this 23 GiB budget. The indexer used to be allocated on
top of the plan, so the real slack was larger than the label. The 2 GiB
reserve is now unallocated after the indexer. `Q4_VRAM_RESERVE_GIB` below
1 is raised to 1. Putting that margin into weights is what hung the
display.

q8 KV is 34 bytes per 32 values. Widths: ctx 65536 → 0.80 GiB, 90112 →
1.21 GiB, 131072 → 1.6 GiB, 262144 → 3.2 GiB (planner line above includes
GDN, so 3.30 GiB). Session files record the width in `hdr.pad`
(q8=8, f16=2, f32=4) and refuse a mismatch. The MTP head keeps its own
f32 KV; the main KV can stay q8 (the 2026-10 MTP measurements below ran
with q8).

## Expert warm set

A decode step touches `n_layer * top-k` = 480 distinct experts. Those are
what should stay on the GPU. The old rule pinned `n_slots/5` (about 1000
of 5120) after a Japanese warmup, including one-hit experts, and
`find_slot` only evicts a pin when nothing unpinned is left. That froze
L1 that a long prefill needed.

Current rule, after the same 10 Japanese prompts × 16 greedy tokens:

- Cap is 1.5× one decode step (720), and never more than 1/4 of L1.
- Only experts within 8× of the hottest hit count are pinned (minimum 2
  hits), so a single prefill touch stays evictable.
- `Q4_PIN_CAP` overrides with an exact top-N and skips the ratio.
- The set is saved (`Q4_WARM_FILE`, guard is model bytes + layers +
  experts). A larger old file is trimmed to the cap on load. The UD-Q4
  warm file does not load for IQ3E.

On the measured run the cap was 720 and **213** experts qualified, out of
5120 GPU slots. Pins are 4% of L1. `Q4_PIN_ARENA` does nothing in resident
mode (there is no staging arena) and `serve.sh` leaves it off.

## QSA is sparse

The 12 attention layers are Qwen Sparse Attention, not dense GQA. An indexer
(4 query heads, 1 key head, dim 128, weights BF16) mean-pools every 4 raw
keys, RMSNorms, and applies RoPE at the block's first position. Each query
keeps the best 512 blocks (2048 tokens) plus the unfinished tail, at most
2051 positions, then runs GQA on the original K/V. Below that width the
selection is the whole prefix, so short prompts match the dense path.

Until this was wired, every QSA layer attended the full cache. Chunk time
grew by 0.43 s per extra 512 tokens of KV. A 90K fill would have taken
about two hours, and decode at that depth would have been about 2 tok/s.

## MTP (speculative decoding)

Off by default. The extracted blk.48 head drafts up to `Q4_MTP_N` (≤3)
tokens per cycle, and one batched main pass verifies them, so the output
distribution is the plain decoder's. A cycle:

1. **Draft.** Each step is one replayed graph: eh_proj over (token embedding,
   previous estimate), the head's attention over its own KV, the 512 routed
   draft experts from a resident Q2_0 pack (`Q4_MTP_RT`), the hc head mix,
   and a top-k over the lm head on the device. Step 0 reseeds from the real
   residual of the last verified row. After the first draft, one whose
   probability is under `Q4_MTP_PMIN` (0.5) ends the chain.
2. **Verify.** `q4_sess_prefill_tokens` runs the nd+1 rows through the main
   model in one pass. GDN state is snapshotted first, and per-token GDN
   inputs are stashed. The routed MoE is the device-routed hybrid:
   each expert's weights are read once for every token that picked it
   (the "verify window"), L1 hits run on the GPU, and misses run in the
   CPU expert pool. The GPU waits for the pool in-graph (`cpxwait`).
3. **Tail.** One batched lm-head pass over all rows plus a device top-k
   per row gives the target distributions. Acceptance is exact (greedy
   match, or the standard speculative accept/resample at temperature).
4. **Commit and repair.** On a rejection, GDN state rolls back to the
   snapshot and replays the accepted prefix. The head's KV cells for
   accepted tokens are rewritten from the real residuals.

During prompt prefill the head only appends its K/V cells (no attention,
MoE or logits), so MTP prefill costs ~2% over plain. When the head KV
fills, the newest window of cells is compacted to the front.

Where the time goes (greedy, L1 2510 slots, 2026-10-12): draft ~3.5 ms,
verify ~57-66 ms, tail ~1.7 ms, commit+repair ~1.5 ms per cycle at
2.7-3.3 tokens per cycle. Verify is ~38 ms of GPU work, ~16-18 ms waiting
on CPU-expert misses, and ~11 ms of launch gaps across ~3300 kernels.
AGENTS.md lists the switches for each piece.

## Measured, 2026-10-11/12 (OpenAI server, GSQ-RCO IQ3_S)

Client-side SSE timing, thinking off, ctx 262144, 2 rounds. Prompts:
ja (2.5K tokens), agent (2.4K), code (56), pf8k (10.2K), pf32k (34.3K).

| | ja | agent | code | pf8k | pf32k |
|---|---:|---:|---:|---:|---:|
| prefill t/s, plain | 980 | — | — | 1112 | 1106 |
| prefill t/s, MTP | ~930 | ~900 | — | 1037-1073 | 932-1078 |
| decode t/s, plain, greedy | 37.2 | 34-37 | 37.6 | — | — |
| decode t/s, MTP N=3, greedy | 38-43 | 29-32 | 45-46 | 34.6 (16 tok) | 21.6 (16 tok) |
| decode t/s, MTP N=3, temp 1 / top-k 20 / top-p 0.95 | 38-39 | 31-35 | 43-47 | — | — |

MTP rows use `Q4_L1_SLOTS=3300` (2510 main slots after the head's
reservation). Agent-style prompts accept fewer drafts (2.2-2.6 tokens per
cycle vs ~3 for ja/code), which is why MTP loses there.

## Measured, 2026-09-28 (IQ3E, probe and live serve)

One process, `serve.sh` defaults: resident experts, reserve 4 GiB, KV q8,
MTP off, 12 CPU-expert threads, hybrid routing.

| | |
|---|---:|
| Prefill 8,192 (probe) | **474.7 tok/s** |
| Prefill 21,627 (live serve) | **561.8 tok/s** |
| Decode, ~16–21K ctx | **43.6–45.2 tok/s** |
| GPU busy per decode token | ~25 ms |

Decode path shape (all graphed): per layer, **gA** covers the dense/GDN/QSA
front half; a host section routes experts, posts CPU jobs first, stages GPU
hits into a mapped buffer, and hands misses to an async stager thread
(`q4_expert_stage_async`, joined at the next layer boundary by
`q4_expert_stage_join`); **gB′** then runs the GPU hit-exec GEMVs
(fused gate+up `wmoe_*_gu` + down `wmoe_*_ya`, `which`-indirected so slot
ids are read on-device), an in-graph `cpxwait` on the CPU-expert flag, and
the merge. The CPU expert pool pipelines each expert's A-phase (dot) and
B-phase (CAS accumulate into y) so job tails overlap. Adaptive repinning
(`Q4_REPIN_EVERY`, default 64) promotes high-hit slots to pins during
decode, cutting residual CPU misses from ~3.0 to ~2.3 per layer.

QSA decode attention runs the sparse path end to end: `qsa_q_prep` →
`qsa_score_dk` → `hosttopk`/`qsa_tk_sel` (parallelized, was a serial
~55 µs scan) → `gqa_dec_split_k` (warp-per-position scoring + warp-per-head
online softmax, ~5 barriers/tile instead of ~120; 167 → 82 µs/call) →
combine. Small-row GEMVs use warp-per-row kernels
(`wgv_q8_0/iq4_nl/iq3_s/iq2_s/q6_k/bf16/f32` plus `wgv_q4_0/q5_0/q2_0`).
hc_mix dense GEMVs ride `gemv_mw` at ~780 GB/s.

Prefill path: dense GEMMs run `Q4_DGEMW` WMMA instances — `gemm_q8w_k`,
`gemm_q6w_k`, `gemm_q4kw_k`, `gemm_q5kw_k`, and the 0-series
`gemm_q40w_k/q50w_k/q20w_k` (64×64×64 WMMA tiles, f16 smem, 32-element
batched dequant fills; gated `n_batch>=16 && !which`, else the
ncols≤4096 reuse kernel or serial GEMV tail). Expert prefill groups
stage once per group then run `moew2_*` (fused gate+up+silu WMMA) and
`moew_*` (down WMMA) with the same batched dequant. `gqa_prefill_k`
dequants K/V warp-per-position into an f16 tile (2 blocks/CU).
`gdn_scan_k` computes the q/k L2 norms in one reduction pass.

Supported weight formats end-to-end: F32/F16/BF16, Q5_1, Q8_0, Q4_0,
Q5_0, Q2_0 (ggml type 42), Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ2_S —
each with host dequant, decode GEMV, batched/prefill MoE, WMMA dense,
and CPU-expert dot coverage.

Kernel-level numbers that drove the work are in ROCM-NOTES.md.

## Measured, 2026-09-26 (earlier kernels)

One process. `Q4_EXPERT_RESIDENT=1`, reserve 4 GiB, KV q8, MTP off.
Prompt is the repeated Japanese+code paragraph in `prefill-probe`.

Dense path, ctx 65536, GPU L1 5120, after the Japanese warmup:

| | |
|---|---:|
| Prefill 8192 total | 154 s, **53.2 tok/s** |
| Last 512 of that | **39.8 tok/s**, still falling |
| Decode after it | **9.6 tok/s** |
| Min MemAvailable | 10.6 GiB |

Sparse path, ctx **90112** (KV 1.21 GiB, GPU L1 4982 / 10.81 GiB):

| | |
|---|---:|
| Prefill 512 (chunk 1, still dense) | **75.8 tok/s** |
| Prefill 16384 total | 262 s, **62.5 tok/s** |
| Prefill **90112** total | 1514 s, **59.5 tok/s** (25.2 min) |
| Last 512, at 90112 | 8.76 s, **58.4 tok/s** |
| Decode at 16384 (already sparse) | **~12 tok/s** |
| 16K continuation | still the probe's code snippet (`int add(int a, int b)`) |

Chunk time goes 6.8 s → 8.1 s when the sparse path turns on at 2048, then
8.1 s → 8.8 s from there to 90K. Expert SSD reads stayed 0. The full fill
touched MemAvailable **9.6 GiB**. The desktop stayed up and RAM recovered
afterwards. A 16K fill inside the same 90K allocation stayed at 10.9 GiB.

The 64-token continuation after the probe described the prompt as a
repeated pattern. That is a sanity check that q8 KV still produces text,
not a quality eval.

## What is practical

On IQ3E (2026-09-28) plain decode was ~44 tok/s in steady state at
16–21K ctx with GPU busy ≈ wall time. The deployed GSQ-RCO IQ3_S build is
~37 tok/s plain and ~1100 tok/s prefill (2026-10-11, section above). A full 90,112-token
prefill was 25 minutes on the 2026-09-26 kernels (~60 tok/s) and has not
been re-measured on the current ones (~470–560 tok/s at 8–21K).
`serve.sh` defaults to ctx **262144**, the training length.
A 512-token prefill inside a 262144 allocation (2026-09-27) used 5168
expert slots and peaked at 22.40 GiB of 23.99 GiB VRAM, with
MemAvailable staying at 11.0 GiB. OpenCode compaction is the practical
window inside that allocation. Old `session.cache` files from a shorter
ctx are refused.

A long think is fine inside that window. `Q4_MAX_NEW` stays 32768 so a
client that omits `max_tokens` is clamped by remaining context.

Do not run a second q4. Do not drop the DRAM floor below 4 GiB. Do not
set the VRAM reserve below 1 GiB. Do not mlock the resident image. Do not
turn `Q4_PROFILE` on for a full prefill (it syncs every section).

## Code map

| file | role |
|---|---|
| `src/q4_gguf.c` | GGUF index. `peel_mtp` keeps the main forward at 48 layers |
| `src/q4_place.c` | 23 GiB usable, VRAM reserve, q8 KV size |
| `src/q4_expert.c` | tight resident image, GPU L1, direct H2D, stable pins, async miss stager, adaptive repin |
| `src/q4_cpuexp.c` | CPU expert pool; per-expert A→B pipelined jobs, CAS y-accumulate |
| `src/q4_ple.c` | n-gram hash + IQ4_NL gather from SSD |
| `src/q4_quant.c` | Q8_0, Q4_K, Q5_1, Q5_K, Q6_K, IQ4_NL, IQ2_S, IQ3_S |
| `src/q4_moe.c` | top-10, staged hit-exec into gB′ graph, two prefill groups |
| `src/q4_fwd.c` | GDN, QSA (q8/f16/f32 KV), PLE, lm-head, gA/gB′ graph capture |
| `src/q4_rocm.cu` | gfx1100 GEMM/GEMV, warp GEMVs (`wgv_*`, n-row `wgvn_*`), WMMA moew/gemm, QSA kernels, device top-k |
| `src/q4_engine.c` | load, VRAM/slot planning with the MTP reservation, warm file, generate (plain and MTP loops) |
| `src/q4_mtp.c` | MTP draft head: K/V-only prompt pass, draft graph, cycle (draft → verify → tail → commit → repair) |
| `src/q4_mmq.cu`, `src/q4_gemm.cu` | prefill expert MMQ (llama.cpp subset) and the hipBLASLt dense path |
| `src/q4_server.c` | OpenAI `/v1/chat/completions` + browser UI on :8090 |
| `serve.sh` | the safe IQ3E launch (cgroup cap, defaults) |

IQ3E routed mix the kernels have to accept: gate/up mostly IQ2_S, down
mostly IQ4_NL, blk.2 gate/up IQ3_S, a little Q8. Dense and shared are
mostly Q8_0. `output.weight` is Q6_K. Indexer projections can be BF16;
the forward does not read them.
