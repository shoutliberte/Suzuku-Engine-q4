#!/bin/sh
# Builds q4-vision (the CPU image encoder, tools/vision) and copies it next
# to the q4 binary, where the server looks for it.
#
#   scripts/build-vision.sh [llama.cpp dir]
#
# Without a directory (or LLAMA_DIR) it downloads llama.cpp at the pinned
# commit below into third_party/llama.cpp-full (not committed): the same
# commit Strata 0.1.43 builds its encoder from, which knows qwen4exp and
# the qwen3vl_merger projector.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
LLAMA_DIR=${1:-${LLAMA_DIR:-$ROOT/third_party/llama.cpp-full}}
if [ ! -e "$LLAMA_DIR/tools/mtmd/mtmd.h" ]; then
    echo "fetching llama.cpp $COMMIT -> $LLAMA_DIR" >&2
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    curl -fL --retry 3 -o "$tmp/l.zip" \
        "https://github.com/ggml-org/llama.cpp/archive/$COMMIT.zip"
    (cd "$tmp" && unzip -q l.zip)
    rm -rf "$LLAMA_DIR"
    mv "$tmp/llama.cpp-$COMMIT" "$LLAMA_DIR"
fi
GEN=""
command -v ninja >/dev/null 2>&1 && GEN="-G Ninja"
cmake -S "$ROOT/tools/vision" -B "$ROOT/build-vision" $GEN \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$LLAMA_DIR"
cmake --build "$ROOT/build-vision" --target q4-vision -j "$(nproc)"
cp "$ROOT/build-vision/bin/q4-vision" "$ROOT/q4-vision"
echo "built $ROOT/q4-vision" >&2
