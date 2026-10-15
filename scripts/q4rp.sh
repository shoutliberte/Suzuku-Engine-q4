#!/bin/sh
# q4safe Q4_BIN shim: wrap q4 in rocprofv3 kernel stats.  Output dir from
# Q4_RP_DIR (default log/rpres), relative to the repo root (--same-dir).
D="${Q4_RP_DIR:-log/rpres}"
mkdir -p "$D"
exec rocprofv3 --kernel-trace --stats -d "$D" --output-format csv -- ./q4 "$@"
