#!/usr/bin/env python3
"""Evict clean page-cache pages for the given files (posix_fadvise
DONTNEED).  No root needed: it only drops pages this process could
re-fault, which is enough for the q4 PLE/weight shards."""
import os, sys

for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        print(f"dropped {path}")
    finally:
        os.close(fd)
