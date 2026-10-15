#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${Q4_MODEL:-}
if [ -z "$MODEL" ]; then
    # download_q4.sh installs into ./models (GSQ-RCO IQ3_S by default,
    # IQ3E with `iq3e`); a sibling models/ next to the repo is also picked up.
    _gsq=GSQ-RCO/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
    _iq3e=IQ3E-Q8D-MTP/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
    for _m in "$ROOT/models/$_gsq" "$ROOT/../models/$_gsq" \
              "$ROOT/models/$_iq3e" "$ROOT/../models/$_iq3e"; do
        if [ -e "$_m" ]; then MODEL=$_m; break; fi
    done
    MODEL=${MODEL:-"$ROOT/models/$_gsq"}
fi
MDIR=$(dirname "$MODEL")
# Train length. QSA attends a fixed 2051 positions, so prefill stays near
# 60 tok/s as the cache grows (measured through 90K). OpenCode compaction
# is the practical window inside this allocation. See DESIGN.md.
CTX=${Q4_CTX:-262144}
HOST=${Q4_HOST:-127.0.0.1}
PORT=${Q4_PORT:-8090}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES

# Defaults tuned on the reference box (RX 7900 XTX 24 GiB / 60 GiB RAM /
# NVMe); every value below is an env override for a different budget.
# Routed experts are one 45.29 GiB DRAM image. PLE stays on SSD.
# Do not run two q4 processes. Keep other heavy memory use closed while it runs.
# 4 GiB stays free for the desktop session. Do not lower it.
# A 10 GiB floor refused residency whenever available RAM was under 55 GiB.
Q4_EXPERT_RESIDENT=${Q4_EXPERT_RESIDENT:-1}
Q4_L2_GIB=${Q4_L2_GIB:-0}              # unused when resident alloc succeeds
Q4_DRAM_RESERVE_GIB=${Q4_DRAM_RESERVE_GIB:-4}
Q4_MLOCK_L2=${Q4_MLOCK_L2:-0}          # resident image is not mlocked (rlimit 8 MB)
Q4_KV=${Q4_KV:-q8}                     # f16 or f32 to widen (MTP runs fine on q8)
Q4_IO_THREADS=${Q4_IO_THREADS:-8}
Q4_PIN_ARENA=${Q4_PIN_ARENA:-0}        # no staging arena in resident mode
# Unallocated VRAM after dense, KV, the QSA indexer and expert L1.
# The desktop on this dGPU uses 1.3-1.7 GiB; first-request init (devroute,
# hipBLASLt tuning, graphs) then spends ~3.3 GiB more — the measured-free
# clamp in q4_engine.c keeps this reserve real.  2 GiB matches the
# planner default in q4_place.c; the watchdog floor is 1024 MiB.
Q4_VRAM_RESERVE_GIB=${Q4_VRAM_RESERVE_GIB:-2}
# The resident image (GSQ-RCO / IQ3E: ~44 GiB after the static L1 set) is
# filled whole: the planner's own budget (MemAvailable minus 4+4+6 GiB of
# reserves) stops at ~40.5 GiB on a 64 GB box, and the rest stages from the
# GGUF on every miss.  The preflight still refuses under 55 GiB
# MemAvailable, and the watchdog below kills q4 before a global OOM.  The
# measured numbers in the README use this.  Q4_RES_BUDGET_GIB= (empty)
# returns to the planner's budget.
Q4_RES_BUDGET_GIB=${Q4_RES_BUDGET_GIB-47}
export Q4_EXPERT_RESIDENT Q4_L2_GIB Q4_DRAM_RESERVE_GIB Q4_MLOCK_L2 Q4_KV
export Q4_RES_BUDGET_GIB
export Q4_IO_THREADS Q4_PIN_ARENA Q4_VRAM_RESERVE_GIB

# Pinned set drives partial-resident fill order. A route profile collected
# with Q4_ROUTE_PROF (saved next to the model) is picked up automatically;
# a stale warm file from another model is rejected by the loader anyway.
# Without one next to the model, the profile shipped in warm/ for that
# model is used (the loader checks it against the model's size and shape).
if [ -z "${Q4_WARM_FILE:-}" ]; then
    for _warm in "$MDIR"/route-prof-*.warm; do
        if [ -f "$_warm" ]; then Q4_WARM_FILE=$_warm; break; fi
    done
fi
if [ -z "${Q4_WARM_FILE:-}" ]; then
    case "$(basename "$MODEL")" in
    *GSQ-RCO-IQ3_S*) _warm=$ROOT/warm/route-prof-gsq-iq3_s.warm ;;
    *IQ3E-Q8D-MTP*)  _warm=$ROOT/warm/route-prof-iq3e.warm ;;
    *)               _warm= ;;
    esac
    if [ -n "$_warm" ] && [ -f "$_warm" ]; then Q4_WARM_FILE=$_warm; fi
fi
Q4_WARM_FILE=${Q4_WARM_FILE:-}
Q4_SESS_CACHE=${Q4_SESS_CACHE:-$ROOT/session.cache}
# CPU expert pool width. 9950X3D has 16 cores; the pool idles between
# tokens (bounded spin -> futex), so 16 does not starve the desktop.
# ~40.5 -> ~42.5 t/s median at 16K ctx.
Q4_CPU_THREADS=${Q4_CPU_THREADS:-12}   # 12 == 16 for decode; keeps CCD1 free
# Device-side routing keeps whole layers in one graph but adds ~60us/layer
# of in-graph nodes (route kernel, mailbox burst, x D2H copy node); the host
# hybrid path overlaps its seam with gA's tail and wins ~11% at 16K ctx.
Q4_DEVROUTE=${Q4_DEVROUTE:-0}
export Q4_CPU_THREADS Q4_DEVROUTE
# int8-activation kernels are on by default and measured faster or neutral
# on this box: Q4_CPXQ8 drives the CPU-expert i8 dot (AVX2 maddubs/madd,
# ~1.4-1.6x on the GSQ formats), Q4_WGQ8 drives the GPU i8-dot MoE kernels
# (v_dot4_i32_iu8, iq2_gu ~1.6x). Both are bit-different from the float path,
# so =0 restores 9ffd59c-identical output for A/B. Q4_WGQ8_IQ4YA=1 opts the
# iq4_nl/iq4_xs down-proj into i8 — measured slower in-model, keep it off.
Q4_THINK=${Q4_THINK:-1}
Q4_SHOW_THINKING=${Q4_SHOW_THINKING:-1}
# Q4_MAX_NEW: the limit when a request gives none (default: the context;
# the model card allows 262144 reasoning tokens).
Q4_MAX_NEW=${Q4_MAX_NEW:-}
# MTP (speculative decoding) is on whenever its head and runtime are found:
# temp-1 bench 2026-10-13 (ctx 262144) ja ~73 / agent ~70 / code ~87 t/s vs
# ~40 plain.  The head GGUF is extracted with scripts/mtp-extract.py; the
# runtime dir (Q2_0 draft-expert pack experts.bin, dense.bin/dense.txt,
# draft_vocab.bin) is what Strata's MTP tooling writes.  Without the
# runtime the head's experts stream through a small L1 and drafts are much
# slower, so MTP stays off then.  Q4_MTP=0 forces plain decode.
#   Q4_MTP=<head.gguf>  default: mtp-head.gguf next to the model
#   Q4_MTP_RT=<dir>     default: mtp-rt/ next to the head
Q4_MTP=${Q4_MTP-}
_mtp_req=$Q4_MTP
if [ -z "$Q4_MTP" ] && [ -e "$MDIR/mtp-head.gguf" ]; then
    Q4_MTP=$MDIR/mtp-head.gguf
fi
if [ "$Q4_MTP" = "0" ]; then
    Q4_MTP=""
elif [ -n "$Q4_MTP" ] && [ ! -e "$Q4_MTP" ]; then
    echo "MTP head not found: $Q4_MTP (Q4_MTP=0 disables)" >&2
    Q4_MTP=""
fi
if [ -n "$Q4_MTP" ]; then
    Q4_MTP_RT=${Q4_MTP_RT:-$(dirname "$Q4_MTP")/mtp-rt}
    if [ ! -e "$Q4_MTP_RT/experts.bin" ]; then
        echo "MTP runtime not found: $Q4_MTP_RT -- plain decode" >&2
        echo "  (scripts/setup-mtp.sh \"$(dirname "$Q4_MTP")\" builds it, or set Q4_MTP_RT)" >&2
        Q4_MTP=""
    fi
elif [ -z "$_mtp_req" ]; then
    echo "no MTP head next to the model: plain decode (scripts/setup-mtp.sh adds it, ~1.8x)" >&2
fi
export Q4_WARM_FILE Q4_SESS_CACHE Q4_THINK Q4_SHOW_THINKING Q4_MAX_NEW Q4_MTP
Q4_MTP_N=${Q4_MTP_N:-3}
export Q4_MTP_N
if [ -n "$Q4_MTP" ]; then export Q4_MTP_RT; fi
# Images: the model's vision encoder (mmproj) runs on the CPU in a helper,
# q4-vision (scripts/build-vision.sh), started at the first picture: ~1.1
# GiB of RAM, no VRAM.  0.9 s at 300 image tokens, 6 s at ~1000 (16 cores).
#   Q4_VISION_MMPROJ=<mmproj.gguf>  default: mmproj*.gguf next to the model
#                                   or one directory up (download_q4.sh);
#                                   0 turns pictures off
#   Q4_VISION_MAX_TOKENS (1024), Q4_VISION_THREADS (one per core)
Q4_VISION_MMPROJ=${Q4_VISION_MMPROJ-}
if [ -z "$Q4_VISION_MMPROJ" ]; then
    for _p in "$MDIR"/mmproj*.gguf "$MDIR"/../mmproj*.gguf; do
        if [ -e "$_p" ]; then Q4_VISION_MMPROJ=$_p; break; fi
    done
fi
if [ "$Q4_VISION_MMPROJ" = "0" ]; then
    Q4_VISION_MMPROJ=""
elif [ -n "$Q4_VISION_MMPROJ" ] && [ -z "${Q4_VISION_BIN:-}" ] &&
     [ ! -x "$ROOT/q4-vision" ]; then
    echo "image encoder not built (scripts/build-vision.sh): pictures off" >&2
    Q4_VISION_MMPROJ=""
fi
export Q4_VISION_MMPROJ
# L1 slots asked of the planner.  With MTP, 3300 is the measured config
# (L1 lending and KV paging then grow the dynamic set to ~4200 below full
# context; VRAM free min ~3.2 GiB beside the desktop).  More takes VRAM the
# reserve does not cover for other GPU users.
if [ -n "$Q4_MTP" ]; then
    Q4_L1_SLOTS=${Q4_L1_SLOTS:-3300}
else
    Q4_L1_SLOTS=${Q4_L1_SLOTS:-8192}   # planner sizes L1; alloc halves on OOM
fi
export Q4_L1_SLOTS

if [ ! -x "$ROOT/q4" ]; then
    echo "q4 is not built. Run make first." >&2
    exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "Model not found: $MODEL" >&2
    echo "Download with: ./download_q4.sh  (or set Q4_MODEL)" >&2
    exit 1
fi

echo "q4 serve  $MODEL"
echo "  ctx=$CTX  http://$HOST:$PORT/  (OpenCode baseURL http://$HOST:$PORT/v1)"
# Own cgroup, hard RAM cap, no swap. Swap is zram and the expert image does
# not compress, so swapping it only thrashes. A global OOM kill inside the
# desktop's unit also stops the session (OOMPolicy=stop). Over the cap only
# q4 dies. Q4_MEM_MAX=0 skips.
# Preflight (scripts/q4preflight.sh): no second q4, no relaunch after an
# unacknowledged OOM kill, enough MemAvailable. q4 is also made the first
# OOM victim so a global OOM does not take the desktop.
#
# 2026-09-29 OOM lesson: the cgroup cap does NOT see amdgpu-pinned/GTT
# pages (~7 GiB unaccounted in the kernel OOM dump), so q4 can push the box
# into a global OOM while under MemoryMax. The kernel oom_reaper could not
# reap the pinned image fast enough and killed the desktop instead.
# Two extra layers now:
#   1. the MTP head keeps its experts in page cache, not resident DRAM
#      (Q4_MTP_RESIDENT=0 default), and
#   2. a 1 s watchdog below kills q4 -9 when MemAvailable < Q4_SAFE_FLOOR
#      (default 3 GiB) or swap grows >1 GiB — BEFORE the kernel OOM. A normal
#      kill lets the driver release pinned pages cleanly; a kernel OOM does
#      not. No stall check here: an idle server legitimately logs nothing.
. "$ROOT/scripts/q4preflight.sh"
q4_preflight || exit 1
q4_oom_first
ulimit -c unlimited 2>/dev/null || true   # keep the amdhip64 abort's core
Q4_MEM_MAX=${Q4_MEM_MAX:-52G}
FLOOR=${Q4_SAFE_FLOOR:-3145728}
MON=${Q4_SERVE_MON:-$ROOT/serve.mon}
gtt0=$(q4_gtt_used)            # eviction baseline: VRAM BOs spill into GTT/RAM
if [ "$Q4_MEM_MAX" != "0" ] && command -v systemd-run >/dev/null 2>&1; then
    (export Q4_CGROUP=1
     exec systemd-run --user --scope --quiet -p MemoryMax="$Q4_MEM_MAX" \
        -p MemorySwapMax=0 "$ROOT/q4" serve "$MODEL" --ctx "$CTX" \
        --host "$HOST" --port "$PORT" "$@") &
    runner=$!
else
    # No cap requested or no systemd-run: the binary still self-wraps in
    # its own scope via q4_self_scope() unless Q4_MEM_MAX=0/Q4_NOSCOPE=1.
    ("$ROOT/q4" serve "$MODEL" --ctx "$CTX" --host "$HOST" --port "$PORT" \
        "$@") &
    runner=$!
fi
: > "$MON"
swap0=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
vfree_lo=0                   # consecutive samples under the VRAM floor
VFLOOR=${Q4_VRAM_FLOOR_MIB:-1024}
jc0=$(date +%s)              # kernel GPU-fault log baseline
jcnext=0
qpid=""
killreason=""
trap 'kill -TERM $qpid $runner 2>/dev/null; exit 130' INT TERM
while kill -0 "$runner" 2>/dev/null; do
    sleep 1
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    sw=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
    if [ -z "$qpid" ]; then qpid=$(pgrep -n -x q4 2>/dev/null || true); fi
    rss=0
    if [ -n "$qpid" ] && [ -r "/proc/$qpid/status" ]; then
        rss=$(awk '/VmRSS/{print $2}' "/proc/$qpid/status" 2>/dev/null || echo 0)
    fi
    gtt=$(q4_gtt_used)
    vram=$(q4_vram_used)
    vfree=$(q4_vram_free_mib)
    desk=$(q4_desk_vram "${qpid:-0}")
    q4vram=0
    if [ -n "$qpid" ]; then q4vram=$(q4_pid_vram "$qpid"); fi
    printf '%s avail=%sG swap=+%sM rss=%sG gtt=%sB vram=%sB q4vram=%sB vfree=%sMiB desk=%s\n' \
        "$(date +%T)" "$((avail/1048576))" "$(( (sw-swap0)/1024 ))" \
        "$((rss/1048576))" "$gtt" "$vram" "$q4vram" "$vfree" "$desk" >> "$MON"
    if [ -n "$qpid" ] && [ -z "$killreason" ]; then
        [ "$avail" -lt "$FLOOR" ] && killreason="MemAvailable<floor"
        [ $((sw - swap0)) -gt 1048576 ] && killreason="swap-grew>1GiB"
        # VRAM exhaustion: the cgroup cap does not see GPU memory — when
        # the card fills, the driver evicts q4's buffers into GTT (kernel
        # memory, not ours) and the desktop dies for RAM.  Kill before the
        # cascade: two sub-floor reads, GTT spill, or a kernel-side
        # pin/submission failure since q4 started.
        if [ "$vfree" -lt "$VFLOOR" ]; then
            vfree_lo=$((vfree_lo + 1))
            [ "$vfree_lo" -ge 2 ] && killreason="vram-free<${VFLOOR}MiB"
        else
            vfree_lo=0
        fi
        [ $((gtt - gtt0)) -gt 2147483648 ] && killreason="gtt-grew>2GiB"
        if [ "$(date +%s)" -ge "$jcnext" ]; then
            jcnext=$(( $(date +%s) + 5 ))
            q4_gpu_klog_bad "$jc0" && killreason="kernel-gpu-err"
        fi
        if [ -n "$killreason" ]; then
            echo "q4 watchdog: $killreason — killing q4 ($qpid)" >&2
            echo "=== WATCHDOG KILL $qpid: $killreason $(date) ===" >> "$MON"
            kill -9 "$qpid" 2>/dev/null
            qpid=""
        fi
    fi
done
wait "$runner"; rc=$?
printf '=== serve done rc=%s reason=%s ===\n' "$rc" "${killreason:-none}" >> "$MON"
exit "$rc"
