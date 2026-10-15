# Third-party notices

Suzuku-Engine-q4 is MIT-licensed ([LICENSE](LICENSE)). It contains or
adapts code from the projects below, all under the MIT License. Their
copyright notices and the license text are reproduced here as that license
requires.

Model weights are not part of this repository. The GGUF files, the MTP
tensors and the vision encoder are downloaded from their publishers and are
covered by their own licenses (see each Hugging Face model card).

## DwarfStar (ds4)

<https://github.com/antirez/ds4>

q4 started in September 2026 as a fork of DwarfStar's engine design
(DRAM / SSD expert offload, the GGUF loader, the CPU reference paths) and
has since been rewritten for RDNA3 / ROCm. Parts of the early code (the
GGUF loader, the CPU quantization reference, and the IQ tables in
`src/q4_iq_grids.inc`, which are ggml's) derive from it.

```
Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek
```

## Strata

<https://github.com/Niko1221/Strata>

- `src/q4_hgemm_tuning.hpp`: adapted from `src/prefill/hipblaslt_tuning.hpp`
  (hipBLASLt solution-id table).
- `src/q4_mmq.cu`: the MMQ wrapper and ggml-cuda host shims follow
  Strata's MMQ integration; `third_party/llama.cpp` was copied from the
  llama.cpp tree Strata builds against (Strata 6f32ec0, 2026-10-04).
- `tools/vision/q4_vision_helper.c` speaks Strata's `strata-vision`
  protocol.
- `scripts/setup-mtp.sh` downloads and runs Strata's MTP tools
  (`tools/mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`, `data/draft_vocab.bin`)
  at a pinned commit; they are not redistributed here.

```
Copyright (c) 2026 Niko1221 and the Strata contributors
```

## llama.cpp / ggml

<https://github.com/ggml-org/llama.cpp>

- `third_party/llama.cpp/`: the MMQ kernel subset (headers, `quantize.cu`,
  template instances). `ggml/src/ggml-cuda/mmq.cuh` is modified by q4 (the
  launch grid, `Q4_MMQ_TUNE`).
- `src/q4_iq_grids.inc`: the IQ2 / IQ3 grid and sign tables.
- `scripts/build-vision.sh` downloads llama.cpp at a pinned commit and
  links `q4-vision` against its mtmd library; `scripts/setup-mtp.sh` uses
  its `gguf-py`.

```
Copyright (c) 2023-2026 The ggml authors
```

## The MIT License (text common to all of the above)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
