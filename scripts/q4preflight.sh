# q4preflight.sh — sourced by serve.sh and scripts/q4safe.sh before q4 starts.
#
# q4_preflight refuses to start q4 when
#   1. another q4 process is running,
#   2. an earlier q4 scope ended in oom-kill and nobody acknowledged it
#      (relaunching a ~51 GiB resident fill right after an OOM kill, while
#      the kernel is still reaping the last one, is what takes the desktop
#      session down), or
#   3. MemAvailable is below Q4_NEED_GIB (default 55 resident / 16 not).
# Q4_PREFLIGHT=0 skips all checks.
#
# q4_oom_first marks the calling shell (and so the exec'd q4) as the
# preferred OOM victim, so a global OOM takes q4 instead of the desktop.

q4_oom_first() {
    { echo 1000 > /proc/self/oom_score_adj; } 2>/dev/null || true
}

# GTT usage of the discrete GPU. Card numbering is not stable, so resolve by
# PCI id (RX 7900 XTX = 1002:744C) each call; Q4_GTT_PATH overrides. Falls
# back to card1, then prints 0 when nothing matches.
q4_gtt_used() {
    p=${Q4_GTT_PATH:-}
    if [ -z "$p" ]; then
        for d in /sys/class/drm/card*/device; do
            [ -f "$d/uevent" ] || continue
            if grep -q 'PCI_ID=1002:744C' "$d/uevent" 2>/dev/null; then
                p="$d/mem_info_gtt_used"; break
            fi
        done
        [ -z "$p" ] && p=/sys/class/drm/card1/device/mem_info_gtt_used
    fi
    cat "$p" 2>/dev/null || echo 0
}

# dGPU total VRAM usage (bytes): the ~24 GiB card (card0 here is the iGPU,
# its vram_total is 2 GiB). Q4_VRAM_PATH overrides; falls back to card1.
q4_vram_used() {
    p=${Q4_VRAM_PATH:-}
    if [ -z "$p" ]; then
        for d in /sys/class/drm/card*/device; do
            t=$(cat "$d/mem_info_vram_total" 2>/dev/null || true)
            [ -n "$t" ] || continue
            if [ "$t" -gt 15000000000 ] && [ "$t" -lt 30000000000 ]; then
                p="$d/mem_info_vram_used"; break
            fi
        done
        [ -z "$p" ] && p=/sys/class/drm/card1/device/mem_info_vram_used
    fi
    cat "$p" 2>/dev/null || echo 0
}

# One process's VRAM usage (bytes): sum of KFD per-process vram_* counters.
# Entries are u64-cast and a negative pending-eviction count wraps to ~2^64
# (seen on vram_56845), so only plausible bytes (0..32 GiB) are summed.
# Prints 0 when the dir or counters aren't readable.
q4_pid_vram() {
    _d=/sys/class/kfd/kfd/proc/$1
    awk '{v=$1+0; if (v>0 && v<34359738368) s+=v} END{print s+0}'         "$_d"/vram_* 2>/dev/null || echo 0
}

# Free VRAM on the dGPU in MiB: (total - used) of the card whose
# mem_info_vram_total is >16 GiB (card0 here is the 2 GiB iGPU).
q4_vram_free_mib() {
    for d in /sys/class/drm/card*/device; do
        t=$(cat "$d/mem_info_vram_total" 2>/dev/null) || continue
        [ -n "$t" ] || continue
        [ "$t" -gt 17179869184 ] || continue
        u=$(cat "$d/mem_info_vram_used" 2>/dev/null || echo 0)
        echo $(( (t - u) / 1048576 ))
        return
    done
    echo 0
}

# Top 3 non-q4 DRM clients by VRAM, "comm:MiB,comm:MiB,comm:MiB".
# fdinfo reports drm-memory-vram (KiB) per fd; fds duped from one client
# share its drm-client-id, so dedupe on it before summing per pid.  $1 is
# q4's pid (excluded — its own usage is tracked via KFD separately).
q4_desk_vram() {
    _x=${1:-0}
    awk -v excl="$_x" '
        BEGINFILE {
            # /proc entries vanish when a process exits between the glob and
            # the open; ERRNO is set for those — skip them, do not abort.
            if (cid != "" && vram > 0 && pid != excl && !seen[pid, cid]++)
                tot[pid] += vram;
            fn = FILENAME; cid = ""; vram = 0;
            split(FILENAME, a, "/"); pid = a[3];
            if (ERRNO != "") nextfile;
        }
        $1 == "drm-client-id:" { cid = $2 }
        $1 == "drm-memory-vram:" { vram = $2 }
        ENDFILE { if (ERRNO != "") { cid = ""; vram = 0; } }
        END {
            if (cid != "" && vram > 0 && pid != excl && !seen[pid, cid]++)
                tot[pid] += vram;
            n = 0;
            for (p in tot) pp[n++] = p;
            for (i = 0; i < n && i < 3; i++) {
                bi = i;
                for (j = i + 1; j < n; j++)
                    if (tot[pp[j]] > tot[pp[bi]]) bi = j;
                t = pp[i]; pp[i] = pp[bi]; pp[bi] = t;
                comm = pp[i]; cp = "/proc/" pp[i] "/comm";
                if ((getline line < cp) > 0) comm = line;
                close(cp);
                printf "%s%s:%d", (i ? "," : ""), comm, \
                    int(tot[pp[i]] / 1024);
            }
        }' /proc/[0-9]*/fdinfo/* 2>/dev/null
}

# Kernel GPU-fault evidence since epoch $1: amdgpu pin/submission failures
# that precede a driver OOM cascade (seen 2026-10-06: pin failed -> command
# submission OOM -> desktop session died).  Returns 0 when a pattern hits.
q4_gpu_klog_bad() {
    command -v journalctl >/dev/null 2>&1 || return 1
    journalctl -k -q --since "@$1" 2>/dev/null | grep -qE \
        'pin failed|Failed to pin framebuffer|Not enough memory for command submission'
}

q4_preflight() {
    [ "${Q4_PREFLIGHT:-1}" = "0" ] && return 0
    if pgrep -x q4 >/dev/null 2>&1; then
        echo "q4 preflight: another q4 process is running:" >&2
        pgrep -ax q4 >&2
        return 1
    fi
    if command -v systemctl >/dev/null 2>&1; then
        for u in $(systemctl --user list-units --type=scope --state=failed \
                       --plain --no-legend 2>/dev/null | awk '{print $1}'); do
            r=$(systemctl --user show -p Result --value "$u" 2>/dev/null)
            d=$(systemctl --user show -p Description --value "$u" 2>/dev/null)
            case "$r:$d" in
            oom-kill:*q4*)
                echo "q4 preflight: $u was OOM-killed ($d)." >&2
                echo "  Find out why before relaunching (journalctl -b | grep -i oom)," >&2
                echo "  then acknowledge with: systemctl --user reset-failed" >&2
                return 1
                ;;
            esac
        done
    fi
    need=${Q4_NEED_GIB:-}
    if [ -z "$need" ]; then
        # One resident image on this 60.4 GiB box. 55 is MemAvailable
        # *before* the alloc, so 59 cannot occur (MemTotal is 60.4).
        # The figure was ~45 GiB of experts plus ~7 GiB of GTT the cgroup
        # cannot see. Static L1 dropped the main image from 46.84 GiB to
        # 42.81 GiB plain, 43.55 GiB with the MTP head's VRAM reservation
        # (bench/results/q4-mtp3-t1v2.log). The head image is ~2.5 GiB, so
        # main+head is inside that same ~45 GiB and Q4_MTP_RESIDENT does
        # not add a second gate. q4_expert_res_build still refuses a head
        # pin that does not fit after the main image is in.
        if [ "${Q4_EXPERT_RESIDENT:-0}" = "1" ]; then need=55; else need=16; fi
    fi
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -lt "$need" ]; then
        echo "q4 preflight: MemAvailable ${avail} GiB < ${need} GiB needed." >&2
        echo "  Close heavy apps / pause large downloads first (or set Q4_NEED_GIB)." >&2
        return 1
    fi
    gtt=$(q4_gtt_used)
    echo "q4 preflight: ok (MemAvailable ${avail} GiB, gtt $((gtt/1048576)) MiB)" >&2
    return 0
}
