#!/bin/sh
# q4safe.sh <logprefix> <q4 args...> — run ./q4 inside a systemd user scope
# (MemoryMax, no swap) with a 1s watchdog logging MemAvailable/swap/q4 RSS/
# top-3 RSS procs/GPU0 VRAM to <logprefix>.mon. Kills q4 -9 when MemAvailable
# < 6 GiB, swap grows >1 GiB over start, or the q4 log stalls for
# Q4_SAFE_STALL s (dumps wchan + gdb backtraces first). Exit = q4 status.
set -u
prefix=$1; shift
MEM=${Q4_SAFE_MEM:-52G}
TO=${Q4_SAFE_TIMEOUT:-900}
STALL=${Q4_SAFE_STALL:-180}
FLOOR=${Q4_SAFE_FLOOR:-3145728}
log="$prefix.log"; mon="$prefix.mon"
: > "$mon"
swap0=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)

. "$(dirname "$0")/q4preflight.sh"
if ! q4_preflight; then
    echo "=== q4safe refused: preflight failed ===" >> "$mon"
    exit 1
fi
gtt0=$(q4_gtt_used)            # eviction baseline: VRAM BOs spill into GTT/RAM
vfree_lo=0                     # consecutive samples under the VRAM floor
VFLOOR=${Q4_VRAM_FLOOR_MIB:-1024}
jc0=$(date +%s)                # kernel GPU-fault log baseline
jcnext=0
# cores on: the amdhip64 vector abort needs coredumpctl to keep it
# (q4 also raises RLIMIT_CORE itself and marks weight arenas DONTDUMP).
ulimit -c unlimited 2>/dev/null || true

# --scope shares stdio but drops stdin pipes; Q4_SAFE_PIPE=1 switches to a
# transient --pipe service so chat-mode probes can feed prompts on stdin.
# A service unit gets a clean environment: forward every Q4_/HIP_/ROCM_
# var (and PATH for timeout/q4 lookup) or probes silently run stock config.
if [ "${Q4_SAFE_PIPE:-0}" = 1 ]; then
    envf="$(cd "$(dirname "$prefix")" && pwd)/$(basename "$prefix").env"
    env | sed -n 's/^\(Q4_[A-Za-z0-9_]*\|HIP_[A-Za-z0-9_]*\|ROCM_[A-Za-z0-9_]*\)=\(.*\)$/\1=\2/p' > "$envf"
    printf 'PATH=%s\nHOME=%s\nQ4_CGROUP=1\n' "$PATH" "$HOME" >> "$envf"
    (q4_oom_first
     exec systemd-run --user --pipe --quiet --same-dir -p "MemoryMax=$MEM" \
        -p MemorySwapMax=0 -p "EnvironmentFile=$envf" \
        timeout "$TO" "${Q4_BIN:-./q4}" "$@") <&0 > "$log" 2>&1 &
else
    (q4_oom_first
     export Q4_CGROUP=1
     exec systemd-run --user --scope --quiet -p "MemoryMax=$MEM" \
        -p MemorySwapMax=0 timeout "$TO" "${Q4_BIN:-./q4}" "$@") <&0 > "$log" 2>&1 &
fi
runner=$!
qpid=""
last_size=0
last_grow=$(date +%s)
killreason=""
while kill -0 "$runner" 2>/dev/null; do
    sleep 1
    now=$(date +%s)
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    sw=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
    if [ -z "$qpid" ]; then qpid=$(pgrep -n -x q4 2>/dev/null || true); fi
    rss=0
    if [ -n "$qpid" ] && [ -r "/proc/$qpid/status" ]; then
        rss=$(awk '/VmRSS/{print $2}' "/proc/$qpid/status" 2>/dev/null || echo 0)
    fi
    top3=$(ps -eo rss,comm --sort=-rss 2>/dev/null | awk 'NR>1 && NR<=4 {printf "%s:%sM ", $2, int($1/1024)}')
    vram=$(timeout 3 rocm-smi --showmeminfo vram 2>/dev/null \
           | awk '/GPU\[0\]/ && /Used/{print $NF; exit}')
    gtt=$(q4_gtt_used)
    vfree=$(q4_vram_free_mib)
    desk=$(q4_desk_vram "${qpid:-0}")
    q4vram=0
    if [ -n "$qpid" ]; then q4vram=$(q4_pid_vram "$qpid"); fi
    size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$size" -gt "$last_size" ]; then last_grow=$now; last_size=$size; fi
    printf '%s avail=%sG swap=+%sM rss=%sG vram=%sB gtt=%sB q4vram=%sB vfree=%sMiB desk=%s logsz=%s top3=%s\n' \
        "$(date +%T)" "$((avail/1048576))" "$(( (sw-swap0)/1024 ))" \
        "$((rss/1048576))" "${vram:--}" "$gtt" "$q4vram" "$vfree" "$desk" \
        "$size" "$top3" >> "$mon"
    if [ -n "$killreason" ] || [ -z "$qpid" ]; then
        :
    else
        [ "$avail" -lt "$FLOOR" ] && killreason="MemAvailable<floor"
        [ $((sw - swap0)) -gt 1048576 ] && killreason="swap-grew>1GiB"
        # VRAM exhaustion: the cgroup cap does not see GPU memory — when
        # the card fills, the driver starts evicting q4's buffers into GTT
        # (charged to the kernel, not us) and the desktop dies for RAM.
        # Kill before that cascade: two sub-floor reads, GTT spill, or a
        # kernel-side pin/submission failure since q4 started.
        if [ "$vfree" -lt "$VFLOOR" ]; then
            vfree_lo=$((vfree_lo + 1))
            [ "$vfree_lo" -ge 2 ] && killreason="vram-free<${VFLOOR}MiB"
        else
            vfree_lo=0
        fi
        [ $((gtt - gtt0)) -gt 2147483648 ] && killreason="gtt-grew>2GiB"
        if [ "$now" -ge "$jcnext" ]; then
            jcnext=$((now + 5))
            q4_gpu_klog_bad "$jc0" && killreason="kernel-gpu-err"
        fi
        if [ $((now - last_grow)) -gt "$STALL" ]; then
            killreason="log-stall>${STALL}s"
            {
                echo "=== STALL DUMP $(date) ==="
                cat "/proc/$qpid/status" 2>/dev/null | head -40
                for t in /proc/"$qpid"/task/*/wchan; do
                    printf '%s %s\n' "$t" "$(cat "$t" 2>/dev/null)"
                done
                if command -v gdb >/dev/null 2>&1; then
                    timeout 30 gdb -p "$qpid" -batch \
                        -ex 'thread apply all bt' 2>&1 | tail -200
                fi
            } >> "$mon" 2>&1
        fi
    fi
    if [ -n "$killreason" ] && [ -n "$qpid" ]; then
        echo "=== WATCHDOG KILL $qpid: $killreason $(date) ===" >> "$mon"
        kill -9 "$qpid" 2>/dev/null
        qpid=""
    fi
done
wait "$runner"; rc=$?
printf '=== q4safe done rc=%s reason=%s ===\n' "$rc" "${killreason:-none}" >> "$mon"
exit "$rc"
