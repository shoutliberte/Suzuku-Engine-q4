#!/bin/sh
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec stdbuf -oL "${Q4_BIN:-$ROOT/q4}" "$@"
