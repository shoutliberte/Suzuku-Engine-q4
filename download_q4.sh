#!/bin/sh
# Downloads a model into ./models (or the directory given), where serve.sh,
# setup-mtp.sh and the Makefile look for it.
#
#   ./download_q4.sh [gsq|iq3e] [dest dir]
#
#   gsq   (default) ISTA-DASLab GSQ-RCO IQ3_S, 2 shards, ~83 GB, plus the
#         vision encoder mmproj (0.9 GB), into models/GSQ-RCO/.  The
#         measured configuration.
#   iq3e  pentacoxian-dev IQ3E-Q8D-MTP, one file, ~86 GB, into
#         models/IQ3E-Q8D-MTP/ (contains its MTP draft layer; setup-mtp.sh
#         then extracts it locally).  No vision encoder of its own.
#
# Needs the Hugging Face CLI (`hf`, from `pip install -U huggingface_hub`).
# Re-running resumes an interrupted download.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
KIND=${1:-gsq}
case "$KIND" in
gsq)
    NAME=GSQ-RCO
    REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
    FILES="IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf mmproj-Qwen3.8-Flash-Next-BF16.gguf"
    SIZE="~84 GB"
    ;;
iq3e)
    NAME=IQ3E-Q8D-MTP
    REPO=pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF
    FILES="Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
    SIZE="~86 GB"
    ;;
*)
    echo "usage: $0 [gsq|iq3e] [dest dir]" >&2
    exit 1
    ;;
esac
DEST=${2:-}
if [ -z "$DEST" ]; then
    # ./models inside the repo; a model already in a sibling ../models is kept
    DEST=$ROOT/models/$NAME
    if [ ! -d "$DEST" ] && [ -d "$ROOT/../models/$NAME" ]; then
        DEST=$ROOT/../models/$NAME
    fi
fi
command -v hf >/dev/null 2>&1 || {
    echo "the Hugging Face CLI is missing: pip install -U huggingface_hub (provides hf)" >&2
    exit 1
}
mkdir -p "$DEST"
echo "Downloading $NAME ($SIZE) into $DEST"
# shellcheck disable=SC2086
hf download "$REPO" $FILES --local-dir "$DEST"
# hf keeps the repo layout (GSQ-RCO/IQ3_S/<shards>, the mmproj one level
# up); serve.sh and setup-mtp.sh know it.  Do not move the files: hf would
# fetch them again on the next run.
MDIR=$DEST
[ "$KIND" = gsq ] && MDIR=$DEST/IQ3_S
echo "done: $MDIR"
echo "next: scripts/setup-mtp.sh \"$MDIR\"   (speculative decoding, ~8 GB more download)"
