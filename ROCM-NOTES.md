# ROCm/gfx1100 notes and IQ3E optimization log

> **Status (2026-10-15):** an optimization log up to 2026-10-12 (mostly the
> IQ3E build; some sections are in Japanese). Later kernels and
> measurements are in [AGENTS.md](AGENTS.md).

Companion to DESIGN.md. What an RDNA3 consumer GPU is, what ROCm does to qwen4exp,
and what the optimization work so far measured. Numbers are RX 7900 XTX +
ROCm 7.2.4 (HIP clang 22), `--offload-arch=gfx1100`. References to
MODEL-STRATEGY.md / QUANT-ESTIMATE.md / RDNA3-OPT.md point at unpublished
working notes; their conclusions are summarized inline here.

## Machine facts that shape everything

- RDNA3, **wave32** (`warpSize == 32`). Any reduction or shuffle written
  for wave64 silently corrupts. Always use fixed-width `__shfl_down_sync`
  masks or explicit 64-thread smem trees.
- ~960 GB/s GDDR6 peak. GEMV work is streaming-bound: every dense weight
  byte is re-read each token, so the floor is roughly
  `dense_bytes / BW`. ~4.5 GiB dense+shared ≈ 5 ms/token floor.
- `hipGraph` exists and works, but a measured empty-kernel node costs
  ~2.5 µs — the same as a plain launch. Graphs mainly save *host* time;
  they do not amortize GPU-side per-node cost like CUDA graphs do.
  ~2000 nodes/token ⇒ ~5 ms/token of pure node overhead. Fusing small
  elementwise launches matters as much as kernel speed.
- `rocprofv3 --kernel-trace` sees kernels inside replayed graphs
  (needs `hsa-amd-aqlprofile`). `Grid_Size_X` is **total threads**
  (blocks × blockDim), not blocks.
- Pinned host memory read by a GPU kernel over PCIe works fine (used for
  the CPU-expert merge axpy — no explicit H2D copy needed).
- `hipMalloc` up to ~23 GiB usable of 24 GiB; the desktop session
  holds ~0.9 GiB idle. VRAM reserve floor 1 GiB, default 2.
- Static `__shared__` over ~58 KB compiles and runs here
  (gqa_dec_split_k uses ~58.5 KB) but check before exceeding 64 KB.

## The OOM incident (why every run is capped)

The 45.29 GiB DRAM-resident expert image plus a big session once pushed
q4 to ~22 GiB RSS + ~25 GiB into **zram**. Quantized weights barely
compress, `vm.swappiness` was 150, so the box thrashed for ~an hour,
the kernel OOM-killed q4 — and because q4 shared the desktop's systemd
unit (`OOMPolicy=stop`), the whole graphical session died with it.

Permanent fixes now in the tree:

- `serve.sh` runs q4 under `systemd-run --user --scope
  -p MemoryMax=$Q4_MEM_MAX(52G) -p MemorySwapMax=0`. q4 dies alone, no
  swap, desktop untouched.
- `scripts/q4safe.sh` wraps every experiment: same cap + a watchdog
  logging MemAvailable/swap/RSS/VRAM each second; kills q4 on
  MemAvailable < floor, swap growth > 1 GiB, or log stall > 180 s.
- **Gotcha:** the 45 GiB resident fill itself dips MemAvailable to ~5 GiB
  transiently. The default 6 GiB floor false-triggers during load; use
  `Q4_SAFE_FLOOR=3145728` (3 GiB) for resident-model runs. The 52 G cap
  is the real bound; the floor is only anti-thrash insurance.

## What the model actually executes per decode token

(rocprofv3 @16K ctx, eager probe; ~3500 dispatches/token, ~39 ms GPU
busy before the warp-GEMV work)

| kernel | ms/tok | shape notes |
|---|---:|---|
| `gemv_q8_0_k` | 11.8 | 320×10240 ×102, 10240×320 ×102 (HC down/up), 2560×6144 ×51 (attn out), 2560×640 ×51 (shared-expert down) — 74–580 GB/s, latency-bound |
| `moe_iq2_bk` | 8.9 | routed expert gate/up, grid (640 rows, ~9 experts) |
| `gemv_m_k` | 5.9 | fused GDN qkv+gate+beta+alpha ≈16480 rows/launch, ~690 GB/s |
| `moe_iq4_bk` | 1.75 | expert down-proj |
| `gemv_q6_k_k` | 1.47 | lm head 248320×2560 — already ~840 GB/s |
| `gemv_f32_k` | 1.46 | router 512×2560 ×51 |
| `gdn_tail_k` | 1.36 | delta-net tail ops |
| `copyBuffer` | 1.24 | ~368 small D2D copies |
| `grouped_rms_k` | 1.00 | RMS norms |
| `gqa_dec_split_k`+comb | 0.65 | attention (was 8.0 ms) |
| host `cpuex join` | ~11 | **GPU idles** during CPU expert wait/layer |

## Optimization log (all verified, capped runs)

Prefill (target ≥250 t/s — met; 2048-tok chunks unless noted):

| commit | change | 2048-tok |
|---|---|---|
| — | baseline | ~95 t/s |
| `f092d33` | batch prefill MoE GEMMs per group (found the wave64 reduction bug here) | — |
| `cb4fb73` | next-layer expert prefetch; 2048 chunks | 139 t/s |
| `f36c226` | GQA-shared K/V dequant in QSA, batched PLE, 1-launch GDN scan, tiled f32 GEMM | 207 t/s |
| `6327836` | RDNA3 WMMA f16 for expert + dense q8 GEMMs | **338 t/s** (317 @4K, 303 @8K, ~285 @64K) |

Decode (target ≥50 t/s — landed at ~44; kernels now bandwidth-bound):

| commit | change | 16K ctx |
|---|---|---|
| `74e75eb` | exact parallel radix top-k for the QSA indexer (old atomic/bisect selection was picking ~380/512 wrong blocks, biased early — a silent long-context quality bug; also removed the ctx-length slowdown: 64K went 13→20 t/s) | 20.9 t/s |
| `8ea8d20` | same exact top-k for prefill/eager decode (prefill logit at 8K@16K 15.9→20.3 — model is more confident now) | — |
| `49b1f7d` | split-KV decode attention (64 slices/kv-head, K/V dequant shared across 12 GQA heads, deterministic merge), split-K f32 GEMV (nrows<96), parallel q6 head GEMV | **~28 t/s**, flat to 64K+ |
| `f315d9d` | warp-per-row decode GEMVs (`wgv_*`) for small-to-mid matrices | ~30–32 t/s |
| `99a40b5`, `28c8915`, `dcb4cfd` | host seam off GPU critical path; fused elementwise tails; parallel q8 KV dequant in split attn | ~33–35 t/s |
| `0b684e4`, `deea110` | devroute path (loses to hybrid, kept as `Q4_DEVROUTE=1` for A/B); slot/weight arrays read over PCIe | ~35–37 t/s |
| `434c61b`, `307ed69` | fused hit-exec (gate+up `wmoe_*_gu`, down+accum `wmoe_*_ya`); adaptive re-pin (`Q4_REPIN_EVERY`) | ~37–38 t/s |
| `34cb1c2`, `d2baaba` | `gqa_dec_split` warp-per-pos scores + warp-per-head softmax (120→5 barriers/tile, 167→82 µs); parallel `qsa_tk_sel` | ~42 t/s |
| `412fee8`, `bf5537c`, `3ff4ded`, `32b82b1` | hit-exec+join folded into gB′ graph; cpuexp per-expert A→B pipeline + CAS accum; wmoe 4× unroll + grid.y 64→8; seg8 qkv fusion | **~44.4 t/s** |

Prefill continued (probe, 8192 tok):

| commit | change | 8192-tok |
|---|---|---|
| — | baseline at session start | ~308 t/s |
| `45fb950` | 32-elem batched dequant in WMMA tile fills (moew2/moew/gemm_q8w); warp-per-pos KV dequant in `gqa_prefill_k` | 413.8 → 427.9 t/s |
| `cc625c6` | tiled f32/bf16 GEMM; f16 smem tile in `gqa_prefill_k` (2 blocks/CU) | 472.2 t/s |
| `a0d750a` | `gdn_scan` single-pass q/k L2 norm | **474.7 t/s** |
| `3db911e` | `gemm_q6w_k` WMMA batched GEMM for Q6_K dense (QUANT-ESTIMATE plan-A prerequisite) | — (no in-model path yet) |
| `7d199fc` | Q4_0/Q5_0/Q2_0(type 42) end-to-end (host dequant, wgv/wmoe/moepf/moew, cpuexp, PLE); `gemm_q4kw`/`gemm_q5kw`/`gemm_q40w`/`gemm_q50w`/`gemm_q20w` WMMA dense prefill | 3.78 bpw quant 137 → **580.1 t/s** |

Cross-model check (same box, same probe): an external 3.78 bpw quant
(Q4_0/Q2_0 routed, Q4_K/Q5_K dense, Q4_0 PLE) went 59.7 → 18.2 s for
8192 tok once its dense K-quants stopped falling to the serial GEMV
tail. Decode parity at ~42–43 t/s (bandwidth-bound on resident
experts; dense+shared is only ~3.5 GiB vs IQ3E's ~5 GiB). Lesson for
QUANT-ESTIMATE: simple `d * (q - off)` formats fill WMMA tiles far
cheaper than codebook formats (IQ2_S/IQ3_S grid lookups), and K-quants
now prefill through the same tiled path instead of the scalar GEMV.

Failed experiments, reverted after A/B: 128-token `gemm_q8w` tiles
(−15%, accumulator spills); 32B-aligned vector WMMA operand loads with
transposed Xs (−3%, 160-Byte-stride store bank conflicts). Graph-node
launch overhead measured ~3 µs uniform — no structural holes left;
`Q4_DEVROUTE=1` re-tested twice and still loses (~11%).

Race found and fixed inside `49b1f7d`: the split-attention kernel
re-read reduction scratch across per-head iterations without a barrier —
a faster thread could corrupt another head's softmax state. Symptom was
flaky top-logit collapse (17.4→9.7). Fix: `__syncthreads()` between
heads + include each slice's denominator in the merge.

## Correctness methodology that worked

- `Q4_CPU_EXPERTS=0` for A/B: the hybrid CPU/GPU expert split depends on
  L1 residency and changes FP order by itself.
- Decode serves CUDA-style graphs; the eager path (`decode-probe`) shows
  *larger* FP-reorder drift (~1e-5/call amplified ~100× through GDN
  exp/sigmoid gates → 0.1–0.3 logit drift, near-tie flips). Graphed
  serve-path A/B agrees to ~1e-3. Both deterministic; accept after
  per-call bounded-error checks.
- Baseline comparisons via `git worktree add /tmp/q4base <rev>` builds —
  never stash a dirty tree.
- Diagnostics that lie: reading GPU scratch via hipMemcpy on the NULL
  stream races `g_str` work — `hipDeviceSynchronize()` first.
- Toggles kept for bisect: `Q4_DEC_SPLIT=0`, `Q4_PF=0`, `Q4_AHEAD=1`,
  `Q4_QSA_PF=0`, `Q4_WMMA=0`, `Q4_GNODES=1`, `Q4_SHAPES=1`
  (per-(type,rows,cols,batch) gemv call counts at shutdown).

## Remaining bottlenecks (ranked, 2026-09-28)

Decode GPU busy ≈ 25 ms ≈ wall time; ~1480 nodes/token with ~3 µs
uniform turnaround. Latest trace (decode window, 16–21K ctx):

| kernel | ms/tok | note |
|---|---:|---|
| `gemv_mw` (hc shared expert concat) | 2.8 | ~780 GB/s, bandwidth-bound |
| `wgv_q8_0` (small GEMVs) | 2.8 | 291 calls, 450–600 GB/s |
| `wmoe_iq4_ya` (hit down) | 1.9 | ~300 GB/s, room left |
| `wmoe_iq2_gu` (hit gate+up) | 1.5 | ~300 GB/s |
| `gqa_dec_split` (QSA attn) | 1.0 | 82 µs × 12 layers |
| `gemv_q6_k` (lm head) | 0.9 | warp version now ~840 GB/s |
| `gdn_tail` / `qsa_score_dk` | 1.2 | |
| host CPU-expert wait (`cpxwait`) | ~1.7 | ~35 µs × 48, DRAM-bound |

1. `wmoe_*` hit-exec ~300 GB/s → further unroll/vectorization of the
   dependent 256-block loop is the largest kernel-level slack left.
2. CPU-expert residual wait ~1.7 ms/tok → smaller experts (quant) or
   deeper A/B overlap, not launch-side fixes.
3. Everything else is ~bandwidth-bound; the next real lever is a
   lighter quant — see QUANT-ESTIMATE.md (dense Q8 dominates
   per-token traffic, ~80%; Q6_K dense ≈ +12–14% decode at near-zero
   quality cost). `gemm_q6w_k` landed (`3db911e`); plan A needs only a
   repacked GGUF to A/B.
4. MTP (embedded blk.48 head, extracted via `scripts/mtp-extract.py`):
   two rounds of fixes so far. First round — an 8-row mini buffer set
   for the verify path (was a silent VRAM OOM on cache-hit sessions),
   and `Q4_MTP_SLOTS` read before the L1 reservation estimate. Second
   round — 2–8-row warp GEMV kernels (`wgvn_*`) for the dense verify
   path and a hybrid MoE verify (L1 hits on GPU, misses dispatched to
   the CPU expert pool instead of synchronous PCIe pulls) took a
   4-token verify from ~187 ms to ~65–77 ms. Confidence
   gating (`Q4_MTP_PMIN`, default 0.5) drops low-confidence drafts.
   Measured after the fixes, one run each, ctx 8192:
   | config | R1 (English) | R2 (Japanese) | acceptance | mean len |
   |---|---:|---:|---|---|
   | plain | 51.7 | 50.0 | — | — |
   | MTP N=1 | 39.3 | 30.6 | 82.5 / 63.9% | 1.83 / 0.98 |
   | MTP N=3 | 41.9 | 38.8 | 66.8 / 56.8% | 2.89 / 2.44 |
   | MTP N=3, PMIN=.8 | 44.3 | 31.8 | 75.4 / 64.0% | 2.86 / 1.28 |
   Still slower than plain at the time (2026-09-28). Later work —
   expert reuse across verify tokens, a resident Q2_0 draft-expert pack,
   the K/V-only head prompt pass, the device head top-k and the batched
   verify tail — made it faster than plain on ja/code at greedy and
   temperature 1 (2026-10-12, DESIGN.md "MTP" and the 2026-10 section
   below). Still opt-in (`Q4_MTP`).

   Head memory after the 2026-09-29 OOM: head experts default to page
   cache (`Q4_MTP_RESIDENT=1` restores a resident image, ~2.5 GiB) and
   `Q4_MTP_L2_GIB` defaults to 0 — pinning them on top of the 45 GiB
   main image is what pushed the box into a global OOM the cgroup
   could not see.

Prefill-side status after `7d199fc`: dense K-quants (Q4_K/Q5_K) and
the simple 0-series (Q4_0/Q5_0/Q2_0) all reach WMMA at n_batch>=16;
the `ncols<=4096` reuse block now sits *after* that gate so small
batches and which-gathers keep the cheaper path. MoE expert GEMMs
remain ~60–80 GB/s effective — per-expert call granularity is the
binding constraint, not DRAM bandwidth.

## 2026-09-29: graph capture の ROCm 側挙動（GSQ-RCO 調査で判明）

- 失敗した `hipStreamEndCapture` のあと、g_str が **capture 状態のまま
  取り残される**（`hipStreamQuery` が "operation not permitted when
  stream is capturing"）。以後の全 launch が失敗し、eager フォールバックが
  「hc-attn mix failed」で token 0 死していた。`graph_end` で stuck を
  検出して stream を作り直す + last-error を drain する修正で、
  eager fallback が動くようになった（`f3f2d08`）。
- GSQ-RCO IQ3_S: PPL 実測は IQ3E と同等以上（MODEL-STRATEGY.md）。
  resident は 46.84 GiB + 実行分で RSS ~53.4 GB → memcg 52G では収まらず
  OOM（oom_score_adj=1000 で q4 のみ死亡、デスクトップ無事）。
  非 resident では ~11-14 t/s（SSD 律速の退化モード）。
- 残課題: 非 resident での capture invalidation（全層 retry）と、
  devroute の cpx_wait ハング（IQ3E で実測、q4safe の stall watchdog が
  180 s で kill）。resident 運用では影響なし。

## 2026-09-29 (2): gfx1100 は host-mapped メモリを GPU からポーリングできない

- devroute の `cpx_wait` wedge の根本原因。GPU カーネルが volatile /
  system-scope atomic load で読んでも host の書き込みが見えない
  （隔離テスト: plain 136 iter / atomic 48 iter とも 3 s タイムアウト、
  host 側は更新済み）。L2 に stale が残り、host 書き込みは GPU cache を
  snoop しない。`flat_load glc slc` の inline asm でも不可
  （page fault する。
- 修正: `hipLaunchHostFunc` を compute stream の in-stream wait ノードに
  した（graph capture 内でも replay されることを検証済み）。callback が
  pinned host メモリを読むのでコヒーレント、stream 順序も保たれる。
  `cpx_wait_k` は撤去。45 s 無進行タイムアウト + last-seen 診断は
  host 側に移した（961ffd6）。
- 併せて直した競合: io pool の単一 job 領域を `job_mu` で直列化、
  dispatcher の staging を `stage_batch_try`（飢餓防止）、cpuex の
  `join2` を tag-aware に（done_a overshoot で in-order floor が
  止まる bug）、`q4_moe_dev_stop` + `stage_join` で teardown SEGV 解消。

**訂正（2026-10-11）:** 上の「ポーリングできない」は、テストに使った確保方法での結論だった。
`hipHostMalloc(hipHostMallocMapped | hipHostMallocCoherent)` で確保した flag なら、
1 thread の GPU カーネルが `volatile` load と `__builtin_amdgcn_s_sleep(1)` で
spin して host の書き込みを受け取れる。非公開のマイクロベンチ（`seambench`）で 48 層 × 50 pass を
完走した。1層あたりのコストは以下（作業 kernel 2本込み、host 側の処理 0 µs）:

| 方式 | µs/層 |
|---|---:|
| 継ぎ目なし | 5.8 |
| `hipLaunchHostFunc`（現行の cpx_wait） | 38.9 |
| `hipStreamWaitValue32` | 11.9 |
| GPU spin | 11.8 |

host 側に 200 µs の処理があると、3方式とも同じになる。MTP verify の 48 層では、
差は 1 サイクルあたり ~1.3 ms しかない。cpx_wait の待ち時間の本体は、CPU expert の計算そのものである。

## 部分 resident experts（commit 961ffd6, 6f01105）

- `Q4_EXPERT_RESIDENT=1` が予算内 quota で各層を充填する packed image
  に変更（従来は全量 or 32 GiB L2 フォールバック→swap 崩壊 or 全 SSD）。
- `res_pos[layer*n_exp+eid]` が byte offset の正本（warm set が先に
  枠を取る）、`res_ok` bitmap が充填済みの正本、`res_ptr`/`host_ptr` /
  GPU upload / serial touch は全てこの2つを通る。未充填 expert は
  io pool の bounce fill（一時 staging → L1 slot）または cpuex の
  per-row pread が透過的に処理する。
- `Q4_WARM_FILE`（ja-experts-*.warm）を planner が読んで hot expert を
  resident 化する。IQ3E 48% fill + warm で 22.9 t/s（フル並み）。

## ルーティングプロファイル駆動の resident 選択（bd1ed85, e5d342f）

- `route_dev_k` が mailbox 末尾の `{rseq, routed[16]}` に全 top-k を
  fire-and-forget 公開（wait protocol とは無関係、dispatcher が
  rseq 変化を検出して `route_hits[layer*stride+eid]` を集計）。
  sb/eager/prefill/serial 各パスも同じ表を bump するので全経路で
  ヒストグラムが育つ。
- `Q4_ROUTE_PROF=<path>` で cache_close が非ゼロ件数を warm 形式で
  保存（tmp+rename で atomic）。そのまま `Q4_WARM_FILE` に指せる。
  IQ3E で eval 3 コーパス 9000 tok → 23,095 expert（層あたり
  431-506 hot、90% カバーに 168-354、95% に 220-402 と層差が大きい）。
- planner はプロファイル付き部分 resident 時、各層が同じ hit 率
  カバレッジを持つ最少 expert 数を二分探索する coverage-equalized
  quota（層ごとに可変、早期層は広く後半層は集中という実測に一致）。
- 実測（IQ3E decode-probe、~19-20 GiB budget）: uniform eid 先頭充填
  16.7 t/s vs coverage-equalized 32.3 t/s（SSD miss 2531→1323、
  cpu-miss 1128→538 µs/job）。10.1 GiB・70% カバーの image が
  GPU L1 に収まる構成では 36.4 t/s。つまり**~10-20 GiB の hot-set
  image でフル resident（39.6 t/s）の 8-9 割の速度**が得られる。

## 既知のフレーキー障害: amdhip64 内部の vector assert

- 2026-09-29、モデルA で ppl 3000 tok × 3 ファイル連続評価中に
  `std::vector<unsigned long>::operator[] __n < size()` で SIGABRT。
  q4 側コードに vector は無いので ROCm/libstdc++ 内部。ppl 経路は
  host-callback node を使わないので cpx wait 変更とは無関係と見る。
  同一条件再実行（及び gdb 下 1500 tok × 2）は完走 — 低頻度の競合。
  再発時は core を残して `coredumpctl debug` で bt を取る。

## int8 activation MoE kernels (Q4_WGQ8, gfx11)

- 活性化を 32 要素ごとに int8 量子化 (`xq8_k`, 1 fp32 scale/group)、
  重みを i8x4 に展開して `v_dot4_i32_iu8` (`__builtin_amdgcn_sudot4`)
  で積和。符号付き grid (iq2_s/iq3_s/iq3_xxs) は pos/neg マスクの
  2 ドット分解。SASS で命令生成を確認済み。
- 対象: `wmoe_iq{2,3,3xxs,4xs}_gu_q8` (shared x)、
  `wmoe_iq{2,3,3xxs,4xs}_ya_q8` + `wmoe_q20_ya_q8` (per-expert x slice)。
  バッファは init 時 `moe_stage_bufs` 経由で確保（graph 内 hipMalloc 禁止）。
- 精度: 単独ベンチで GPU 生成 q8 と同一の int8 ref と全形式 0 bad。
  E2E では先頭 ~7 tok 一致後に argmax 分岐 — CPU の Q4_CPXQ8 と同種の
  活性化量子化ノイズ。
- 実モデル計測 (rocprofv3, decode-probe 24 tok):
  iq2_gu 1.58x、iq3_gu 1.22x。iq4_ya は単独 1.7x だが実測 0.91x
  (短い行・L1 重み読み律速で xq8/xd8 追加 load が不利) のため
  **iq4_nl/iq4_xs の ya はデフォルト float、Q4_WGQ8_IQ4YA=1 で opt-in**。
  詳細は RDNA3-OPT.md。

## E2E 比較: 9ffd59c vs 現行 (decode-probe 64tok, RES_BUDGET=20-21GiB,
##   route-prof warm, hybrid CPU+GPU experts, L1 8192, IO 8)

warm mean t/s（同一設定・複数回。軌道は量子化ノイズとキャッシュ履歴で分岐
するため run 間のばらつきが大きい — 特に heretic は GGUF 2 ファイル構成で
初回の page cache 冷えが効く）:

| model | baseline 9ffd59c | 現行 (defaults) | 現行 float のみ |
|---|---|---|---|
| IQ3E (20GiB)      | 30.12          | 23.2*,26.9,24.4,26.6 | 27.8,29.3,27.8 |
| GSQ-RCO IQ3_S     | 19.2,17.2,26.0 | 22.6,21.5,23.9       | 23.2           |
| heretic 3.78bpw   | 18.3,24.3,33.9,27.6 | 33.2,35.7,31.6,31.1 | 34.3,35.1   |

*IQ3E の 23.2 は同一軌道区間の tok1-6 が全ラン 2 倍遅い外れ値（環境要因）。

所見:
- IQ3E は float パスが baseline と logit 完全一致（tok0 16.267）、速度も
  同分布 — コード退行なし。q8 は同一軌道区間の速度も同等で、e2e 差は
  軌道分岐→SSD miss 配置の運。Io-bound regime では GPU 側 ~1ms/tok の
  削減は e2e にほぼ現れない。
- GSQ-RCO は現行の下限が上がる傾向（21.5-23.9 vs 17.2-26.0）。CPXQ8 の
  効果は miss→CPU-expert 経路で発生するが、io 律速で大きくは出ない。
- heretic は experts が Q4_0 系で新 GPU カーネル非対象。測定差は page
  cache/軌道運（float でも 34-35 出る）。
- heretic の warm file を ja-experts（253 hits, 10.2GiB pin）から
  route-prof（22,844 hits, 92.3% coverage, 21GiB pin）に変更すると
  SSD miss が ~3900→~1400-2500/64tok に減る。serve.sh 側で採用済み。

結論: 20-21GiB 帯では全モデル io 律速で、カーネル最適化の e2e 効果は
ノイズ以下〜数%。カーネル側の真の改善（GPU 1.3-1.6x / CPU 1.4-1.6x）
は全常駐・大 budget など演算律速の環境で効く。decode の e2e を上げる
レバーは resident budget と route-prof による hit 率。

## 全常駐計測 (GSQ-RCO IQ3_S, RES_BUDGET=47GiB, image 46.84GiB 全 pin,
##   SSD misses 0 — 演算律速 regime, decode-probe 64tok, CPU threads 12)

| 構成 | warm mean t/s |
|---|---|
| baseline 9ffd59c        | 25.84 / 26.20 |
| 現行 float のみ          | 25.76 |
| 現行 CPXQ8 のみ          | 29.52 |
| 現行 WGQ8 のみ           | 29.06 |
| 現行 defaults           | 27.70 / 30.16 / 29.51 |

→ 演算律速では新カーネル群で **約 +12% e2e**。float は baseline と一致
（回帰なし・数値もクロスバイナリで deterministic）。全常駐時は miss が
CPU expert pool に行くため CPXQ8 (AVX2 int8) の寄与が支配的。
メモリは fill 中 avail ~5GiB まで低下するが watchdog floor (3GiB) 内で
完走。60GiB マシンでは運用ギリギリ — serve 運用は従来どおり budget 運用
が安全。

## 2026-10: MTP サイクルの計測と教訓

- **rocprofv3 で 1 ステップを分解する。** `rocprofv3 --kernel-trace --output-format csv` を
  `q4 chat` に掛ける（AGENTS.md の MTP 節に手順がある）。graph の replay 内の kernel も見える。
  - draft 1 ステップ 4.65 ms のうち 3.45 ms が head の top-k 選択 `tk_sel_k` だった。
    32768 行の slab ごとに 4 block × 32 thread しか使っておらず、ほぼ直列だった。
  - GEMV 1回で行バッファに書き、block 幅の k-pass 選択（`tkw_k`、256 thread × 8 値）を
    1 block になるまで繰り返す方式に替えた。選択は ~0.1 ms、3 ステップの draft は 14 → 3.5 ms。
  - **教訓:** 縮約系の kernel は、grid が CU 数（96）より十分大きいかを最初に見る。
- **全 GPU idle の内訳。** 1 window の kernel を全 stream 合わせて時刻順に並べ、隙間を
  「前の kernel → 次の kernel」で分類する。
  - copy stream の `streamOpsWrite`（residency 表の書き込み）が隙間に散らばって見えた。
    実体は GPU が CPU expert を待っている時間（cpxwait）だった。
  - verify（wall ~74 ms）の内訳は、GPU busy 38 ms、cpxwait ~25 ms、
    20 µs 未満の起動間隔 ~11 ms（1 window 3267 kernel）。
- **cpxwait は DRAM 帯域律速。** 層あたり ~15 個の miss expert × ~1.7-2.1 MB を DRAM から読む
  時間（~0.5 ms/層）と一致する。
  - 1 job（層）あたりの CPU 時間は、12 / 15 / 16 スレッドで 656 / 634 / 710 µs で、ほぼ変わらない。
    ~16 unique expert × ~2 MB ≈ 50 GB/s。CPU pool は pin されておらず、2 CCD にまたがって動く。
  - L1 slot を 2510 → 4897 にすると、miss pair が 21.5 → 14.9 / job、cpxwait が 22 → 12.6 ms/cycle になった
    （2026-10-12、chat、ctx 32768）。「slot を +50% にしても miss は 15% しか減らない」という過去の
    結論（REPORT-mtp-slab）は再現しなかった。
  - 効くのは VRAM を L1 に回すこと。ただし planner の見積もりと実際の確保量がずれている。
    rocprofv3 `--memory-allocation-trace` に出ない ~3.5 GB がある（hipBLASLt、graph など）。
- **リンク配置で性能が変わる（false sharing）。** q4_rocm.cu の host 側だけの変更で、
  verify の cpxwait が 20 → 27 ms に増えた。
  - 原因は、CPU pool の job カウンタ（全 worker が atomic に書く）が、読み取り中心の
    フィールドと同じ cache line に詰まっていたこと。どの組が line を共有するかは
    `g_pool` の 64B 境界からのずれ（リンク位置）で変わる。
  - カウンタ群を `_Alignas(64)` で分離して 16-18 ms になった。
  - **教訓:** CPU 側を変えていないのに CPU 時間が動いたら、前コミットのバイナリを
    worktree で建てて同じセッションで対照を取る。`nm q4 | grep g_pool` で配置も見る。
    作業ツリーを stash してビルドし直すと、オブジェクトの組み合わせが分からなくなる
    （上の「never stash a dirty tree」と同じ）。
