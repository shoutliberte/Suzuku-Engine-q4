#!/bin/sh
# Prepares MTP (speculative decoding, ~1.8x decode at temp 1) next to the
# model, where serve.sh finds it:
#
#   <model dir>/mtp-head.gguf   the draft layer (q4's head: dense weights)
#   <model dir>/mtp-rt/         the Q2_0 draft-expert pack, dense overlay and
#                               draft vocabulary (Strata's MTP runtime format)
#
#   scripts/setup-mtp.sh [model dir]
#
# Downloads ~3 GiB (the head, by range requests from the IQ3E-Q8D-MTP GGUF;
# the GSQ-RCO GGUFs ship no draft layer) plus ~5 GiB (the BF16 MTP tensors of
# Qwen/Qwen3.8-Flash-Next, also by range requests), and needs ~8 GiB of disk
# in the work dir while packing.  The runtime is built by Strata's own tools
# (MIT, https://github.com/Niko1221/Strata) at a pinned commit, with
# llama.cpp's gguf-py at the commit build-vision.sh uses; both are fetched
# into third_party/ (not committed).  Python 3 with numpy is required.
#
# Env: Q4_MTP_WORK (default <model dir>/.mtp-work, deleted on success unless
# Q4_MTP_KEEP_WORK=1), HF_ENDPOINT (a mirror, both downloads), HF_TOKEN
# (the head download; neither repository is gated).
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STRATA_COMMIT=a0116b2782a5b82622358c04a06496485d12b2d6   # Strata 0.1.43.1
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d    # = build-vision.sh

DIR=${1:-}
if [ -z "$DIR" ]; then
    for _d in "$ROOT/models/GSQ-RCO/IQ3_S" "$ROOT/../models/GSQ-RCO/IQ3_S" \
              "$ROOT/models/IQ3E-Q8D-MTP" "$ROOT/../models/IQ3E-Q8D-MTP"; do
        if [ -d "$_d" ]; then DIR=$_d; break; fi
    done
fi
if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    echo "usage: $0 <model dir>   (no model dir found; run ./download_q4.sh first)" >&2
    exit 1
fi
DIR=$(CDPATH= cd -- "$DIR" && pwd)
PY=${PYTHON:-python3}
"$PY" -c 'import numpy' 2>/dev/null || {
    echo "numpy is required ($PY -m pip install numpy, or your distro's python-numpy)" >&2
    exit 1
}

fetch_zip() {   # <owner/repo> <commit> <dest>
    [ -d "$3" ] && return 0
    echo "fetching $1 ${2%"${2#???????}"} -> $3" >&2
    _t=$(mktemp -d)
    curl -fL --retry 3 -o "$_t/z.zip" "https://github.com/$1/archive/$2.zip"
    (cd "$_t" && unzip -q z.zip)
    mkdir -p "$(dirname "$3")"
    rm -rf "$3"
    mv "$_t"/*-"$2" "$3"
    rm -rf "$_t"
}

# 1. the head: the draft layer of the IQ3E GGUF (a local copy is used when
#    it sits in the model dir, otherwise range requests to Hugging Face)
if [ -e "$DIR/mtp-head.gguf" ]; then
    echo "head: $DIR/mtp-head.gguf exists"
elif [ -e "$DIR/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf" ]; then
    "$PY" "$ROOT/scripts/mtp-extract.py" "$DIR/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf" \
        -o "$DIR/mtp-head.gguf"
else
    "$PY" "$ROOT/scripts/mtp-extract.py" --hf -o "$DIR/mtp-head.gguf.dl"
    # the head extracted from the pinned revision (a local extract of the
    # same file gives the same bytes)
    want=6d815eb854f26c2f3609ab4556c6bc51441a73ca2976d43930da215bc2dec058
    got=$(sha256sum "$DIR/mtp-head.gguf.dl" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "mtp-head.gguf: sha256 $got, expected $want (a mirror or proxy that" >&2
        echo "ignored the range requests?); left as $DIR/mtp-head.gguf.dl" >&2
        exit 1
    fi
    mv "$DIR/mtp-head.gguf.dl" "$DIR/mtp-head.gguf"
fi

# 2. the runtime, with Strata's tools
if [ -e "$DIR/mtp-rt/experts.bin" ] && [ -e "$DIR/mtp-rt/draft_vocab.bin" ]; then
    echo "runtime: $DIR/mtp-rt exists"
    exit 0
fi
STRATA=$ROOT/third_party/strata-$STRATA_COMMIT
LLAMA=$ROOT/third_party/llama.cpp-full
fetch_zip Niko1221/Strata "$STRATA_COMMIT" "$STRATA"
if [ ! -d "$LLAMA/gguf-py/gguf" ]; then
    fetch_zip ggml-org/llama.cpp "$LLAMA_COMMIT" "$LLAMA"
fi
WORK=${Q4_MTP_WORK:-$DIR/.mtp-work}
mkdir -p "$WORK"
export STRATA_GGUF_PY="$LLAMA/gguf-py"
# fetch is resumable and checks every tensor against the pinned SHA256
"$PY" "$STRATA/tools/mtp_fetch.py" fetch --out "$WORK"
"$PY" "$STRATA/tools/mtp_fetch.py" verify --out "$WORK"
"$PY" "$STRATA/tools/mtp_pack.py" --src "$WORK" --experts q2_0 --out "$WORK/mtp-q2_0.gguf"
rm -rf "$DIR/mtp-rt.part"
"$PY" "$STRATA/tools/mtp_rt.py" --gguf "$WORK/mtp-q2_0.gguf" --out "$DIR/mtp-rt.part"
cp "$STRATA/data/draft_vocab.bin" "$DIR/mtp-rt.part/draft_vocab.bin"
rm -rf "$DIR/mtp-rt"
mv "$DIR/mtp-rt.part" "$DIR/mtp-rt"
if [ "${Q4_MTP_KEEP_WORK:-0}" != "1" ]; then rm -rf "$WORK"; fi
echo "MTP ready: $DIR/mtp-head.gguf + $DIR/mtp-rt (serve.sh picks both up)"
