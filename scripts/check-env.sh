#!/bin/sh
# Checks this machine against what q4 needs, and which optional parts are
# set up.  Read-only; prints one line per check and exits 1 if a required
# one fails.
#
#   scripts/check-env.sh
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ROCM=${ROCM_PATH:-/opt/rocm}
fail=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }
warn() { printf '  warn  %s\n' "$*"; }
gib()  { awk -v k="$1" 'BEGIN { printf "%.1f", k / 1048576 }'; }

echo "required"
[ "$(uname -s)" = Linux ] && ok "Linux" || bad "Linux only"
if [ -x "$ROCM/bin/hipcc" ]; then ok "ROCm hipcc ($ROCM)"; else bad "no $ROCM/bin/hipcc (install ROCm, or set ROCM_PATH)"; fi
if ls "$ROCM"/lib/libhipblaslt.so* >/dev/null 2>&1; then ok "hipBLASLt"; else bad "no libhipblaslt in $ROCM/lib"; fi
arch=$("$ROCM/bin/rocminfo" 2>/dev/null | awk '/Name: *gfx/ { print $2; exit }')
case "$arch" in
gfx1100) ok "GPU $arch" ;;
gfx11*)  warn "GPU $arch: built and tuned for gfx1100 (make ROCM_ARCH=$arch; untested)" ;;
"")      bad "no GPU visible to rocminfo (user in the render/video groups?)" ;;
*)       bad "GPU $arch: q4 needs RDNA3 (gfx1100); its kernels use RDNA3 WMMA" ;;
esac
vmax=0
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    v=$(cat "$f"); [ "$v" -gt "$vmax" ] && vmax=$v
done
vg=$(gib $((vmax / 1024)))
if [ "$vmax" -ge 24000000000 ]; then ok "VRAM $vg GiB"; else bad "VRAM $vg GiB: needs a 24 GiB card"; fi
mt=$(awk '/^MemTotal/ { print $2 }' /proc/meminfo)
ma=$(awk '/^MemAvailable/ { print $2 }' /proc/meminfo)
if [ "$mt" -ge 58720256 ]; then ok "RAM $(gib "$mt") GiB"; else bad "RAM $(gib "$mt") GiB: the expert image needs 64 GB (MemAvailable >= 55 GiB at start)"; fi
if [ "$ma" -ge 57671680 ]; then ok "MemAvailable $(gib "$ma") GiB"; else warn "MemAvailable $(gib "$ma") GiB now: serve.sh refuses under 55 GiB (close browsers / other big processes)"; fi
for fl in avx2 fma f16c; do
    grep -qw "$fl" /proc/cpuinfo && ok "CPU $fl" || bad "CPU lacks $fl"
done
grep -qw avx512_vnni /proc/cpuinfo && ok "CPU AVX-512 VNNI (fast CPU expert path)" || warn "no AVX-512 VNNI: the CPU expert kernels fall back to AVX2 (slower)"
command -v gcc >/dev/null 2>&1 && command -v make >/dev/null 2>&1 && ok "gcc, make" || bad "gcc and make are needed"
if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    ok "systemd user session (memory cap scope)"
else
    warn "no systemd user session: q4 runs without its memory cap"
fi

echo "optional"
command -v hf >/dev/null 2>&1 && ok "hf (download_q4.sh)" || warn "no hf: pip install -U huggingface_hub (download_q4.sh)"
python3 -c 'import numpy' 2>/dev/null && ok "python3 + numpy (setup-mtp.sh)" || warn "no python3 numpy (setup-mtp.sh needs it)"
for t in curl unzip cmake; do
    command -v $t >/dev/null 2>&1 && ok "$t" || warn "no $t (setup-mtp.sh / build-vision.sh)"
done

echo "this checkout"
[ -x "$ROOT/q4" ] && ok "q4 built" || warn "q4 not built: make"
model=""
for m in "$ROOT"/models/GSQ-RCO/IQ3_S/*-00001-of-00002.gguf "$ROOT"/../models/GSQ-RCO/IQ3_S/*-00001-of-00002.gguf \
         "$ROOT"/models/IQ3E-Q8D-MTP/*.gguf "$ROOT"/../models/IQ3E-Q8D-MTP/*.gguf; do
    case "$m" in *mtp-head*|*mmproj*) continue ;; esac
    if [ -e "$m" ]; then model=$m; break; fi
done
[ -n "${Q4_MODEL:-}" ] && model=$Q4_MODEL
if [ -n "$model" ] && [ -e "$model" ]; then
    ok "model $model"
    d=$(dirname "$model")
    if [ -e "$d/mtp-head.gguf" ] && [ -e "$d/mtp-rt/experts.bin" ]; then ok "MTP head + runtime"
    else warn "no MTP next to the model: scripts/setup-mtp.sh \"$d\" (~1.8x decode)"; fi
    mp=""
    for f in "$d"/mmproj*.gguf "$d"/../mmproj*.gguf; do [ -e "$f" ] && mp=$f; done
    if [ -n "$mp" ]; then
        [ -x "$ROOT/q4-vision" ] && ok "vision encoder + q4-vision" || warn "mmproj found, q4-vision not built: scripts/build-vision.sh"
    else
        warn "no mmproj: pictures off"
    fi
    free=$(df -Pk "$d" | awk 'NR == 2 { print $4 }')
    [ "$free" -ge 20971520 ] && ok "disk free $(gib "$free") GiB by the model (the session cache in the repo dir can reach ~20 GB at long contexts)" || warn "disk free $(gib "$free") GiB by the model"
else
    warn "no model found: ./download_q4.sh (~84 GB)"
fi
[ "$fail" = 0 ] && echo "all required checks passed" || echo "some required checks failed"
exit "$fail"
