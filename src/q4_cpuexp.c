#define _GNU_SOURCE
#include "q4.h"

#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* CPU-side routed experts for the single-token decode path.
 *
 * The GPU spends most of a decode step waiting on PCIe H2D copies of expert
 * weights. DRAM bandwidth on the host is ~3x PCIe x4, so running the routed
 * FFNs on CPU cores while the GPU handles shared-expert + attention work is
 * a large net win. Weights are read straight out of the DRAM-resident expert
 * image (or L1/L2 slots; direct GGUF row pread as fallback) -- nothing is
 * dequantized eagerly, GEMVs are fused single-pass dot products.
 */

#define Q4_IQ_DECL static const
#include "q4_iq_grids.inc"

/* f16 -> f32 via F16C: exact for every input (normals, subnormals, inf,
 * nan) like q4_f16_to_f32, but one instruction inline instead of an
 * out-of-line branchy call per block (IQ4_NL pays it every 32 values). */
static inline float cpx_h2f(uint16_t h) { return _cvtsh_ss(h); }

#define CPX_MAX_TH 16
#define CPX_CHUNK_A 64   /* rows per chunk, gate/up (n_ff) */
#define CPX_CHUNK_B 64   /* rows per chunk, down (n_embd) */
#define CPX_MAX_E 64   /* one token is <=16; a verify window unions up to 64 */

/* ---------------- fused row dot kernels (AVX2) ---------------- */

static inline float cpx_hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

/* 8 signed grid bytes -> 8 floats with sign byte applied. */
static inline __m256 cpx_w8(uint64_t gv, uint32_t sg) {
    const __m256i kbit =
        _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    const __m256 ksgn = _mm256_set1_ps(-0.0f);
    __m128i w8 = _mm_cvtsi64_si128((int64_t)gv);
    __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
    __m256i m = _mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_set1_epi32((int)sg), kbit), kbit);
    return _mm256_xor_ps(w, _mm256_and_ps(_mm256_castsi256_ps(m), ksgn));
}

/* Float view of the 1024-entry IQ2_S codebook. A plain 8-float load beats
 * the int8->int32->float convert chain; the table is 32 KiB and stays in L1.
 * Built once on first dispatch. */
static float cpx_fgrid2s[1024][8];
static int cpx_fgrid_ok;
static void cpx_fgrid_init(void) {
    if (cpx_fgrid_ok) return;
    for (int i = 0; i < 1024; i++)
        for (int j = 0; j < 8; j++)
            cpx_fgrid2s[i][j] = (int8_t)((q4_iq2s_grid[i] >> (8 * j)) & 0xff);
    __sync_synchronize();
    cpx_fgrid_ok = 1;
}

static inline __m256 cpx_fw8(int idx, uint32_t sg) {
    const __m256i kbit =
        _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    const __m256 ksgn = _mm256_set1_ps(-0.0f);
    __m256 w = _mm256_loadu_ps(cpx_fgrid2s[idx]);
    __m256i m = _mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_set1_epi32((int)sg), kbit), kbit);
    return _mm256_xor_ps(w, _mm256_and_ps(_mm256_castsi256_ps(m), ksgn));
}

/* IQ2_XS 74B block: u16 qs[32] = 9-bit grid idx + 7-bit sign idx, sc[8]
 * (two 4-bit half-scales, ls = 2*nibble+1), effective d*0.125*ls. */
static float dot_iq2_xs(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 74;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        const uint8_t *sc = blk + 66;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            const float db0 = d * (float)(2 * (sc[ib] & 0xf) + 1);
            const float db1 = d * (float)(2 * (sc[ib] >> 4) + 1);
            uint16_t v[4];
            memcpy(v, blk + 2 + 8 * ib, 8);
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (int l = 0; l < 4; l++) {
                __m256 wv = cpx_w8(q4_iq2xs_grid[v[l] & 511],
                                   q4_ksigns_iq2xs[v[l] >> 9]);
                __m256 xv = _mm256_loadu_ps(xb + ib * 32 + l * 8);
                if (l < 2) v0 = _mm256_fmadd_ps(wv, xv, v0);
                else v1 = _mm256_fmadd_ps(wv, xv, v1);
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db0), v0, va);
            va = _mm256_fmadd_ps(_mm256_set1_ps(db1), v1, va);
        }
    }
    return cpx_hsum(va);
}

/* IQ2_XXS 66B block: four u16 per 32-val group = u32 grid bytes +
 * u32 (4x 7-bit sign idx, top nibble scale ls = 2*nibble+1). */
static float dot_iq2_xxs(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 66;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux32[2];
            memcpy(aux32, blk + 2 + 8 * ib, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const float db = d * (float)(2 * (int)(aux32[1] >> 28) + 1);
            __m256 v = _mm256_setzero_ps();
            for (int l = 0; l < 4; l++) {
                const uint8_t sg =
                    q4_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                __m256 wv = cpx_w8(q4_iq2xxs_grid[aux8[l]], sg);
                __m256 xv = _mm256_loadu_ps(xb + ib * 32 + l * 8);
                v = _mm256_fmadd_ps(wv, xv, v);
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db), v, va);
        }
    }
    return cpx_hsum(va);
}

static float dot_iq2_s(const uint8_t *w, const float *x, uint64_t n) {
    /* One vector accumulator per 256-block: the per-sub-block scales are
     * folded in vectorially instead of paying a serial hsum every 32
     * elements (~4x fewer reductions). */
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *signs = blk + 34;
        const uint8_t *qh = blk + 66;
        const uint8_t *sc = blk + 74;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            const float db0 = d * (0.5f + (float)(sc[ib] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (int l = 0; l < 4; l++) {
                int idx = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                __m256 wv = cpx_fw8(idx, signs[l]);
                __m256 xv = _mm256_loadu_ps(xb + ib * 32 + l * 8);
                if (l < 2) v0 = _mm256_fmadd_ps(wv, xv, v0);
                else v1 = _mm256_fmadd_ps(wv, xv, v1);
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db0), v0, va);
            va = _mm256_fmadd_ps(_mm256_set1_ps(db1), v1, va);
            qs += 4;
            signs += 4;
        }
    }
    return cpx_hsum(va);
}

static float dot_iq3_s(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *signs = blk + 74;
        const uint8_t *sc = blk + 106;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 4; ib++) {
            const float db1 = d * (1.f + 2.f * (float)(sc[ib] & 0xf));
            const float db2 = d * (1.f + 2.f * (float)(sc[ib] >> 4));
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (int half = 0; half < 2; half++) {
                const uint8_t qhb = qh[half];
                for (int l = 0; l < 4; l++) {
                    const int i1 = qs[2 * l] | ((qhb << (8 - 2 * l)) & 256);
                    const int i2 = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256);
                    uint64_t gv = (uint64_t)q4_iq3s_grid[i1] |
                                  ((uint64_t)q4_iq3s_grid[i2] << 32);
                    __m256 wv = cpx_w8(gv, signs[l]);
                    __m256 xv =
                        _mm256_loadu_ps(xb + ib * 64 + half * 32 + l * 8);
                    if (half == 0) v0 = _mm256_fmadd_ps(wv, xv, v0);
                    else v1 = _mm256_fmadd_ps(wv, xv, v1);
                }
                qs += 8;
                signs += 4;
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db1), v0, va);
            va = _mm256_fmadd_ps(_mm256_set1_ps(db2), v1, va);
            qh += 2;
        }
    }
    return cpx_hsum(va);
}

/* IQ3_XXS: 98-byte block = f16 d + 64 grid bytes + 32 sign/scale bytes.
 * Same oct-vector trick as iq3_s: two uint32 grid entries make one w8. */
static float dot_iq3_xxs(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *ss = qs + 64;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux32;
            memcpy(&aux32, ss + 4 * ib, 4);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            __m256 v = _mm256_setzero_ps();
            for (int l = 0; l < 4; l++) {
                const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint64_t gv = (uint64_t)q4_iq3xxs_grid[qs[2 * l]] |
                                    ((uint64_t)q4_iq3xxs_grid[qs[2 * l + 1]]
                                     << 32);
                v = _mm256_fmadd_ps(cpx_w8(gv, sg),
                                    _mm256_loadu_ps(xb + ib * 32 + l * 8), v);
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db), v, va);
            qs += 8;
        }
    }
    return cpx_hsum(va);
}

static const int8_t cpx_iq4nl_kv[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static inline __m256 cpx_iq4nl_vec(const uint8_t *blk, const float *xb,
                                   __m128i kv, __m128i m4) {
    __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(qv, m4));
    __m128i hi =
        _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(qv, 4), m4));
    __m256 v = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_loadu_ps(xb));
    v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
        _mm256_loadu_ps(xb + 8), v);
    v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
        _mm256_loadu_ps(xb + 16), v);
    v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
        _mm256_loadu_ps(xb + 24), v);
    return v;
}

static float dot_iq4_nl(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        va = _mm256_fmadd_ps(_mm256_set1_ps(cpx_h2f(dh)),
                             cpx_iq4nl_vec(blk, x + i * 32, kv, m4), va);
    }
    return cpx_hsum(va);
}

/* IQ4_XS: 136-byte block = f16 d + u16 scales_h + scales_l[4] + qs[128];
 * per-ib (32 vals): same nibble table as IQ4_NL but a 6-bit scale. */
static float dot_iq4_xs(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 136;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        uint16_t sh;
        memcpy(&sh, blk + 2, 2);
        const uint8_t *sl = blk + 4;
        const uint8_t *qs = blk + 8;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                           ((int)((sh >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            __m128i qv = _mm_loadu_si128((const __m128i *)(qs + ib * 16));
            __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(qv, m4));
            __m128i hi = _mm_shuffle_epi8(
                kv, _mm_and_si128(_mm_srli_epi16(qv, 4), m4));
            __m256 v = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
                _mm256_loadu_ps(xb + ib * 32));
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
                _mm256_loadu_ps(xb + ib * 32 + 8), v);
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
                _mm256_loadu_ps(xb + ib * 32 + 16), v);
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
                _mm256_loadu_ps(xb + ib * 32 + 24), v);
            va = _mm256_fmadd_ps(_mm256_set1_ps(dl), v, va);
        }
    }
    return cpx_hsum(va);
}

static float dot_q8_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + i * 32;
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)qs))),
            _mm256_loadu_ps(xb));
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(qs + 8)))),
            _mm256_loadu_ps(xb + 8), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(qs + 16)))),
            _mm256_loadu_ps(xb + 16), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(qs + 24)))),
            _mm256_loadu_ps(xb + 24), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

/* q4_0: same nibble geometry as iq4_nl (lo->0..15, hi->16..31), linear
 * (nibble - 8) map instead of the kvalues table. */
static float dot_q4_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i m8 = _mm_set1_epi8(8);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const float *xb = x + i * 32;
        __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 2));
        __m128i lo = _mm_sub_epi8(_mm_and_si128(qv, m4), m8);
        __m128i hi =
            _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qv, 4), m4), m8);
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
            _mm256_loadu_ps(xb));
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
            _mm256_loadu_ps(xb + 8), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
            _mm256_loadu_ps(xb + 16), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
            _mm256_loadu_ps(xb + 24), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

/* q5_0: q4_0 nibbles (at +6) + 32 high bits (u32 at +2, bit i = elem i),
 * offset -16. */
static float dot_q5_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i m16 = _mm_set1_epi8(16);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 22;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        uint32_t qh;
        memcpy(&qh, blk + 2, 4);
        const float *xb = x + i * 32;
        __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 6));
        /* high bits: elems 0..15 = qh bits 0..15 (lo nibbles),
         * 16..31 = bits 16..31 (hi nibbles) */
        __m128i hlo = _mm_set_epi8(
            (int8_t)((qh >> 15) & 1), (int8_t)((qh >> 14) & 1),
            (int8_t)((qh >> 13) & 1), (int8_t)((qh >> 12) & 1),
            (int8_t)((qh >> 11) & 1), (int8_t)((qh >> 10) & 1),
            (int8_t)((qh >> 9) & 1), (int8_t)((qh >> 8) & 1),
            (int8_t)((qh >> 7) & 1), (int8_t)((qh >> 6) & 1),
            (int8_t)((qh >> 5) & 1), (int8_t)((qh >> 4) & 1),
            (int8_t)((qh >> 3) & 1), (int8_t)((qh >> 2) & 1),
            (int8_t)((qh >> 1) & 1), (int8_t)(qh & 1));
        __m128i hhi = _mm_set_epi8(
            (int8_t)((qh >> 31) & 1), (int8_t)((qh >> 30) & 1),
            (int8_t)((qh >> 29) & 1), (int8_t)((qh >> 28) & 1),
            (int8_t)((qh >> 27) & 1), (int8_t)((qh >> 26) & 1),
            (int8_t)((qh >> 25) & 1), (int8_t)((qh >> 24) & 1),
            (int8_t)((qh >> 23) & 1), (int8_t)((qh >> 22) & 1),
            (int8_t)((qh >> 21) & 1), (int8_t)((qh >> 20) & 1),
            (int8_t)((qh >> 19) & 1), (int8_t)((qh >> 18) & 1),
            (int8_t)((qh >> 17) & 1), (int8_t)((qh >> 16) & 1));
        __m128i lo = _mm_sub_epi8(
            _mm_or_si128(_mm_and_si128(qv, m4), _mm_slli_epi16(hlo, 4)), m16);
        __m128i hi = _mm_sub_epi8(
            _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qv, 4), m4),
                         _mm_slli_epi16(hhi, 4)),
            m16);
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
            _mm256_loadu_ps(xb));
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
            _mm256_loadu_ps(xb + 8), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
            _mm256_loadu_ps(xb + 16), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
            _mm256_loadu_ps(xb + 24), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

/* q2_0: 64-elem block, fp16 d + 16B of 2-bit quads (byte j = elems 4j..4j+3,
 * LSB-first), values (q - 1) * d. */
static float dot_q2_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 64;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const uint8_t *qs = blk + 2;
        const float *xb = x + i * 64;
        int8_t buf[64];
        for (int j = 0; j < 16; j++) {
            const uint32_t qv = qs[j];
            buf[4 * j + 0] = (int8_t)((int)((qv >> 0) & 3) - 1);
            buf[4 * j + 1] = (int8_t)((int)((qv >> 2) & 3) - 1);
            buf[4 * j + 2] = (int8_t)((int)((qv >> 4) & 3) - 1);
            buf[4 * j + 3] = (int8_t)((int)((qv >> 6) & 3) - 1);
        }
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)buf))),
            _mm256_loadu_ps(xb));
        for (int k = 1; k < 8; k++)
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(buf + 8 * k)))),
                _mm256_loadu_ps(xb + 8 * k), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

static float dot_f16(const uint8_t *w, const float *x, uint64_t n) {
    const uint16_t *h = (const uint16_t *)w;
    __m256 v = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i hw = _mm_loadu_si128((const __m128i *)(h + i));
        v = _mm256_fmadd_ps(_mm256_cvtph_ps(hw), _mm256_loadu_ps(x + i), v);
    }
    float acc = cpx_hsum(v);
    for (; i < n; i++) acc += cpx_h2f(h[i]) * x[i];
    return acc;
}

static float dot_bf16(const uint8_t *w, const float *x, uint64_t n) {
    const uint16_t *h = (const uint16_t *)w;
    __m256 v = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i lo = _mm_loadu_si128((const __m128i *)(h + i));
        __m256i w32 = _mm256_cvtepu16_epi32(lo);
        __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(w32, 16));
        v = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), v);
    }
    float acc = cpx_hsum(v);
    for (; i < n; i++) acc += q4_bf16_to_f32(h[i]) * x[i];
    return acc;
}

static float dot_f32(const uint8_t *w, const float *x, uint64_t n) {
    const float *fw = (const float *)w;
    __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(fw + i), _mm256_loadu_ps(x + i),
                             v0);
        v1 = _mm256_fmadd_ps(_mm256_loadu_ps(fw + i + 8),
                             _mm256_loadu_ps(x + i + 8), v1);
    }
    for (; i + 8 <= n; i += 8)
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(fw + i), _mm256_loadu_ps(x + i),
                             v0);
    float acc = cpx_hsum(_mm256_add_ps(v0, v1));
    for (; i < n; i++) acc += fw[i] * x[i];
    return acc;
}

/* Generic fallback: dequant a row then plain dot. */
static float dot_generic(uint32_t type, const uint8_t *w, const float *x,
                         uint64_t n, float *buf) {
    if (n > 8192) return 0.f;
    q4_dequant_row(type, w, n, buf);
    float s = 0.f;
    for (uint64_t i = 0; i < n; i++) s += buf[i] * x[i];
    return s;
}

static float cpx_row_dot(uint32_t type, const uint8_t *w, const float *x,
                         uint64_t ncols, float *dbuf) {
    switch (type) {
    case Q4_T_IQ2_XXS: return dot_iq2_xxs(w, x, ncols);
    case Q4_T_IQ2_XS: return dot_iq2_xs(w, x, ncols);
    case Q4_T_IQ2_S:  return dot_iq2_s(w, x, ncols);
    case Q4_T_IQ3_S:  return dot_iq3_s(w, x, ncols);
    case Q4_T_IQ4_NL: return dot_iq4_nl(w, x, ncols);
    case Q4_T_IQ3_XXS: return dot_iq3_xxs(w, x, ncols);
    case Q4_T_IQ4_XS:  return dot_iq4_xs(w, x, ncols);
    case Q4_T_Q4_0:   return dot_q4_0(w, x, ncols);
    case Q4_T_Q5_0:   return dot_q5_0(w, x, ncols);
    case Q4_T_Q2_0:   return dot_q2_0(w, x, ncols);
    case Q4_T_Q8_0:   return dot_q8_0(w, x, ncols);
    case Q4_T_F16:    return dot_f16(w, x, ncols);
    case Q4_T_BF16:   return dot_bf16(w, x, ncols);
    case Q4_T_F32:    return dot_f32(w, x, ncols);
    default:          return dot_generic(type, w, x, ncols, dbuf);
    }
}

/* ---------------- int8-activation kernels --------------------------------
 * The float path above decodes each weight to fp32 lanes and FMAs against
 * raw x.  The int8 path instead quantizes the activation ONCE per job
 * (x for gate/up, h for down) into per-32-element groups of {i8 codes,
 * fp32 scale, i32 code sum} and computes each 32-value dot in integer:
 * maddubs(u8 magnitudes, i8 activations) -> madd(i16 scales) -> i32, one
 * cvt+fmadd per 32-elem group.  Same format bytes, ~3x fewer ops per
 * weight element; rounding shifts the dot by ~1% (activation quant noise)
 * — gated by Q4_CPXQ8 (default on, =0 restores the float path).
 * Grid magnitudes are the unsigned
 * maddubs operand, sign bits are folded into the activation via sign_epi8,
 * per-16/per-32 weight scales ride in the i16 madd multiplier. */

typedef struct {
    int8_t  q[32];
    float   d;   /* amax/127 */
    int32_t bs;  /* sum(q): offset-correction formats (q2_0/q4_0/...) */
} cpx_q8;        /* 40 B per 32 elements */

static inline int32_t cpx_hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    lo = _mm_add_epi32(lo, _mm256_extracti128_si256(v, 1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4e));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xb1));
    return _mm_cvtsi128_si32(lo);
}

/* x[n] -> o[n/32]; n%32 must be 0 (checked at dispatch).  Rounding is
 * cvtps' nearest-even — matches ggml's 12582912 magic.  amax==0 emits
 * d=0/q=0/bs=0 which contributes nothing to any dot. */
static void cpx_quant8(const float *x, uint64_t n, cpx_q8 *o) {
    const __m256 mabs =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i ord = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t g = 0; g < n / 32; g++) {
        const float *xb = x + 32 * g;
        __m256 x0 = _mm256_loadu_ps(xb), x1 = _mm256_loadu_ps(xb + 8);
        __m256 x2 = _mm256_loadu_ps(xb + 16), x3 = _mm256_loadu_ps(xb + 24);
        __m256 am = _mm256_max_ps(
            _mm256_max_ps(_mm256_and_ps(x0, mabs), _mm256_and_ps(x1, mabs)),
            _mm256_max_ps(_mm256_and_ps(x2, mabs), _mm256_and_ps(x3, mabs)));
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(am),
                              _mm256_extractf128_ps(am, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ps(m, _mm_movehdup_ps(m));
        const float a = _mm_cvtss_f32(m);
        o[g].d = a / 127.f;
        const __m256 sc = _mm256_set1_ps(a > 0.f ? 127.f / a : 0.f);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(x0, sc));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(x1, sc));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(x2, sc));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(x3, sc));
        __m256i p = _mm256_packs_epi16(
            _mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        p = _mm256_permutevar8x32_epi32(p, ord);
        _mm256_storeu_si256((__m256i *)o[g].q, p);
        o[g].bs =
            cpx_hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(one8, p),
                                           one16));
    }
}

/* ksigns_iq2xs as sign bytes: byte k = 0xFF when bit k set, else 0x01 —
 * four scalar u64 loads build a whole 32-value sign vector (the IQ3_XXS
 * aux word holds four 7-bit sign indices).  Lazy init, like cpx_fgrid2s. */
static uint64_t cpx_esigns[128];
static int cpx_esigns_ok;
static void cpx_esigns_init(void) {
    if (cpx_esigns_ok) return;
    for (int i = 0; i < 128; i++) {
        uint64_t r = 0;
        for (int k = 0; k < 8; k++)
            r |= (uint64_t)(((q4_ksigns_iq2xs[i] >> k) & 1) ? 0xFF : 0x01)
                 << (8 * k);
        cpx_esigns[i] = r;
    }
    __sync_synchronize();
    cpx_esigns_ok = 1;
}

/* 32 sign bits -> 32 bytes of -1/+1 (ggml bit_selector pattern). */
static inline __m256i cpx_sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int)m);
    const __m128i lo = _mm_shuffle_epi8(
        bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(
        bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo),
                                                 hi, 1);
    const __m256i sel = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80, 1, 2, 4, 8, 16, 32, 64,
        (char)0x80, 1, 2, 4, 8, 16, 32, 64, (char)0x80, 1, 2, 4, 8, 16,
        32, 64, (char)0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

/* i16 lane scales: lanes 0-7 = a (vals 0-15), 8-15 = b (16-31). */
static inline __m256i cpx_sc16(int a, int b) {
    return _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_set1_epi16((short)a)),
        _mm_set1_epi16((short)b), 1);
}

static inline uint32_t cpx_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static inline uint64_t cpx_u64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* One 32-elem group: g = u8 magnitudes, sgn = i8 ±1 applied to the
 * activation, sc = i16 weight scales -> i32 pair sums. */
static inline __m256i cpx_dot32(__m256i g, __m256i sgn, __m256i sc,
                                const int8_t *yq) {
    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq);
    const __m256i ys = _mm256_sign_epi8(yv, sgn);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc);
}

/* Same, when the decoded values are already signed i8 (IQ4_NL/IQ4_XS
 * kvalues): magnitude via abs, sign applied to the activation.  Requires
 * no zero values (kvalues min |v| = 1) — sign(_,0)=0 would zero a product
 * that should be v*y... v=0 only pairs y with 0 which is correct anyway. */
static inline __m256i cpx_dot32_s(__m256i v, const int8_t *yq) {
    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq);
    return _mm256_maddubs_epi16(_mm256_abs_epi8(v),
                                _mm256_sign_epi8(yv, v));
}

static int cpx_q8ok(uint32_t t) {
    switch (t) {
    case Q4_T_IQ2_XXS:
    case Q4_T_IQ2_XS:
    case Q4_T_IQ2_S:
    case Q4_T_IQ3_S:
    case Q4_T_IQ3_XXS:
    case Q4_T_IQ4_XS:
    case Q4_T_IQ4_NL:
    case Q4_T_Q4_0:
    case Q4_T_Q5_0:
    case Q4_T_Q8_0:
    case Q4_T_Q2_0:
        return 1;
    }
    return 0;
}

static int cpx_q8_env(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_CPXQ8");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* IQ2_XS 74B block: per 32-val group j, four u16 (9-bit grid + 7-bit sign
 * index each) and one scale byte split between the two 16-val halves. */
static float dq8_iq2_xs(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 74;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        for (int j = 0; j < 8; j++) {
            uint16_t v[4];
            memcpy(v, blk + 2 + 8 * j, 8);
            const __m256i g = _mm256_set_epi64x(
                (long long)q4_iq2xs_grid[v[3] & 511],
                (long long)q4_iq2xs_grid[v[2] & 511],
                (long long)q4_iq2xs_grid[v[1] & 511],
                (long long)q4_iq2xs_grid[v[0] & 511]);
            const __m256i sgn = _mm256_set_epi64x(
                (long long)cpx_esigns[v[3] >> 9],
                (long long)cpx_esigns[v[2] >> 9],
                (long long)cpx_esigns[v[1] >> 9],
                (long long)cpx_esigns[v[0] >> 9]);
            const uint8_t sb = blk[66 + j];
            const __m256i sc =
                cpx_sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
            const cpx_q8 *xg = x + 8 * i + j;
            accf = _mm256_fmadd_ps(
                _mm256_set1_ps(d * xg->d),
                _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                accf);
        }
    }
    return cpx_hsum(accf);
}

/* IQ2_XXS 66B block: aux32[0] = four grid bytes, aux32[1] = 4x 7-bit sign
 * index + 4-bit scale. Same dot32 shape as dq8_iq3_xxs but byte-indexed
 * u64 grid. */
static float dq8_iq2_xxs(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 66;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        for (int j = 0; j < 8; j++) {
            uint32_t aux32[2];
            memcpy(aux32, blk + 2 + 8 * j, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const uint32_t aw = aux32[1];
            const __m256i g = _mm256_set_epi64x(
                (long long)q4_iq2xxs_grid[aux8[3]],
                (long long)q4_iq2xxs_grid[aux8[2]],
                (long long)q4_iq2xxs_grid[aux8[1]],
                (long long)q4_iq2xxs_grid[aux8[0]]);
            const __m256i sgn = _mm256_set_epi64x(
                (long long)cpx_esigns[(aw >> 21) & 127],
                (long long)cpx_esigns[(aw >> 14) & 127],
                (long long)cpx_esigns[(aw >> 7) & 127],
                (long long)cpx_esigns[aw & 127]);
            const __m256i sc =
                _mm256_set1_epi16((short)(2 * (int)(aw >> 28) + 1));
            const cpx_q8 *xg = x + 8 * i + j;
            accf = _mm256_fmadd_ps(
                _mm256_set1_ps(d * xg->d),
                _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                accf);
        }
    }
    return cpx_hsum(accf);
}

/* IQ2_S 82B block: qs[32] grid bytes + qs[32] sign bytes + qh[8] + sc[8].
 * Per 32-val group ib (j,half): 4 grid lookups, sign u32, two 16-scales. */
static float dq8_iq2_s(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        for (int j = 0; j < 4; j++) {
            const uint64_t m = cpx_u64(blk + 34 + 8 * j);
            for (int half = 0; half < 2; half++) {
                const int o = 4 * half;
                const uint32_t h = qh[2 * j + half];
                const __m256i g = _mm256_set_epi64x(
                    (long long)q4_iq2s_grid[qs[8 * j + o + 3] |
                                            ((h << 2) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o + 2] |
                                            ((h << 4) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o + 1] |
                                            ((h << 6) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o] |
                                            ((h << 8) & 0x300)]);
                const __m256i sgn =
                    cpx_sgn_vec(half ? (uint32_t)(m >> 32) : (uint32_t)m);
                const uint8_t sb = blk[74 + 2 * j + half];
                const __m256i sc =
                    cpx_sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
                const cpx_q8 *xg = x + 8 * i + 2 * j + half;
                accf = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * xg->d),
                    _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                    accf);
            }
        }
    }
    return cpx_hsum(accf);
}

/* IQ3_S 110B: qs[64] (u32 grid per byte), qh[8], signs[32], scales[4]. */
static float dq8_iq3_s(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        for (int j = 0; j < 4; j++) {
            const uint8_t *q = blk + 2 + 16 * j;
            const uint64_t m = cpx_u64(blk + 74 + 8 * j);
            const uint8_t s = blk[106 + j];
            for (int half = 0; half < 2; half++) {
                const uint8_t *qb = q + 8 * half;
                const uint32_t h = blk[66 + 2 * j + half];
                /* 4 u64 lanes = 8 u32 grid entries: 4 merged loads beat
                 * set_epi32's per-lane insert chain. */
#define G3(k) (uint64_t)q4_iq3s_grid[qb[k] | (((h >> k) & 1u) << 8)]
                const __m256i g = _mm256_set_epi64x(
                    (long long)(G3(6) | (G3(7) << 32)),
                    (long long)(G3(4) | (G3(5) << 32)),
                    (long long)(G3(2) | (G3(3) << 32)),
                    (long long)(G3(0) | (G3(1) << 32)));
#undef G3
                const __m256i sgn =
                    cpx_sgn_vec(half ? (uint32_t)(m >> 32) : (uint32_t)m);
                const __m256i sc = _mm256_set1_epi16(
                    (short)(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1));
                const cpx_q8 *xg = x + 8 * i + 2 * j + half;
                accf = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * xg->d),
                    _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                    accf);
            }
        }
    }
    return cpx_hsum(accf);
}

/* IQ3_XXS 98B: 64 grid bytes + 8 aux u32 (4x 7-bit sign idx + 4-bit sc). */
static float dq8_iq3_xxs(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.25f;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                const uint8_t *q = blk + 2 + 16 * j + 8 * half;
                const uint32_t aw =
                    cpx_u32(blk + 2 + 64 + 8 * j + 4 * half);
#define G3X(k) (uint64_t)q4_iq3xxs_grid[q[k]]
                const __m256i g = _mm256_set_epi64x(
                    (long long)(G3X(6) | (G3X(7) << 32)),
                    (long long)(G3X(4) | (G3X(5) << 32)),
                    (long long)(G3X(2) | (G3X(3) << 32)),
                    (long long)(G3X(0) | (G3X(1) << 32)));
#undef G3X
                const __m256i sgn = _mm256_set_epi64x(
                    (long long)cpx_esigns[(aw >> 21) & 127],
                    (long long)cpx_esigns[(aw >> 14) & 127],
                    (long long)cpx_esigns[(aw >> 7) & 127],
                    (long long)cpx_esigns[aw & 127]);
                const __m256i sc =
                    _mm256_set1_epi16((short)(2 * (int)(aw >> 28) + 1));
                const cpx_q8 *xg = x + 8 * i + 2 * j + half;
                accf = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * xg->d),
                    _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                    accf);
            }
        }
    }
    return cpx_hsum(accf);
}

/* IQ4_NL nibble LUT -> 32 signed values in one 256 reg (lo 16 vals in the
 * low lane, hi in the high). */
static inline __m256i cpx_iq4nl_i8(const uint8_t *q16, __m128i kv,
                                   __m128i m4) {
    const __m128i qv = _mm_loadu_si128((const __m128i *)q16);
    const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(qv, m4));
    const __m128i hi =
        _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(qv, 4), m4));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

/* IQ4_XS 136B: u16 scales_h + scales_l[4] + qs[128], per-32 6-bit scale. */
static float dq8_iq4_xs(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 136;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        uint16_t sh;
        memcpy(&sh, blk + 2, 2);
        const uint8_t *sl = blk + 4;
        const uint8_t *qs = blk + 8;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                           ((int)((sh >> (2 * ib)) & 3) << 4);
            const __m256i v = cpx_iq4nl_i8(qs + 16 * ib, kv, m4);
            const cpx_q8 *xg = x + 8 * i + ib;
            const __m256i p = cpx_dot32_s(v, xg->q);
            accf = _mm256_fmadd_ps(
                _mm256_set1_ps(d * xg->d),
                _mm256_cvtepi32_ps(
                    _mm256_madd_epi16(p, _mm256_set1_epi16((short)(ls - 32)))),
                accf);
        }
    }
    (void)one16;
    return cpx_hsum(accf);
}

/* IQ4_NL 18B block: f16 d + 16 nibble bytes, no sub-scale. */
static float dq8_iq4_nl(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const __m256i v = cpx_iq4nl_i8(blk + 2, kv, m4);
        const cpx_q8 *xg = x + i;
        accf = _mm256_fmadd_ps(
            _mm256_set1_ps(cpx_h2f(dh) * xg->d),
            _mm256_cvtepi32_ps(_mm256_madd_epi16(cpx_dot32_s(v, xg->q),
                                                 one16)),
            accf);
    }
    return cpx_hsum(accf);
}

/* Offset formats: unsigned codes c in u8, value = (c - off) * d, so
 * dot = d*(dx*sum(c*q8) - off*dx*bs).  v' u8 codes x i8 activations. */
static inline __m256i cpx_dot32_u(__m256i v, const int8_t *yq) {
    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(v, yv),
                             _mm256_set1_epi16(1));
}

static float dq8_q4_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    float cs = 0.f;
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 2));
        const __m128i lo = _mm_and_si128(qv, m4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qv, 4), m4);
        const __m256i v =
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        const cpx_q8 *xg = x + i;
        const float dd = cpx_h2f(dh) * xg->d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(dd),
                               _mm256_cvtepi32_ps(cpx_dot32_u(v, xg->q)),
                               accf);
        cs += dd * (float)xg->bs;
    }
    return cpx_hsum(accf) - 8.f * cs;
}

static float dq8_q5_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    float cs = 0.f;
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i m1 = _mm_set1_epi8(0x10);
    const __m128i sel16 = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64,
                                      (char)0x80, 1, 2, 4, 8, 16, 32,
                                      64, (char)0x80);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 22;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        uint32_t qh;
        memcpy(&qh, blk + 2, 4);
        const __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 6));
        /* high bits -> byte lanes: elems 0-15 use qh bits 0-15, 16-31 use
         * bits 16-31.  Broadcast the u16s then test each bit position. */
        const __m128i hv = _mm_set1_epi32((int)qh);
        const __m128i hb_lo = _mm_and_si128(
            _mm_shuffle_epi8(hv, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                               1, 1, 1, 1, 1, 1, 1)),
            sel16);
        const __m128i hb_hi = _mm_and_si128(
            _mm_shuffle_epi8(hv, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                               3, 3, 3, 3, 3, 3, 3)),
            sel16);
        const __m128i lo = _mm_or_si128(
            _mm_and_si128(qv, m4),
            _mm_and_si128(_mm_cmpeq_epi8(hb_lo, sel16), m1));
        const __m128i hi = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(qv, 4), m4),
            _mm_and_si128(_mm_cmpeq_epi8(hb_hi, sel16), m1));
        const __m256i v =
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        const cpx_q8 *xg = x + i;
        const float dd = cpx_h2f(dh) * xg->d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(dd),
                               _mm256_cvtepi32_ps(cpx_dot32_u(v, xg->q)),
                               accf);
        cs += dd * (float)xg->bs;
    }
    return cpx_hsum(accf) - 16.f * cs;
}

static float dq8_q8_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        /* w is already signed i8: abs/sign into the activation.  w = -128
         * wraps abs to 0x80 = u8 128 — still the correct magnitude. */
        const __m256i v =
            _mm256_loadu_si256((const __m256i *)(blk + 2));
        const cpx_q8 *xg = x + i;
        accf = _mm256_fmadd_ps(
            _mm256_set1_ps(cpx_h2f(dh) * xg->d),
            _mm256_cvtepi32_ps(
                _mm256_madd_epi16(cpx_dot32_s(v, xg->q), one16)),
            accf);
    }
    return cpx_hsum(accf);
}

/* Q2_0 18B block: f16 d + 16 bytes of 2-bit quads = 64 vals, two groups.
 * Byte b -> u32 [c0,c1,c2,c3] via a 256-entry unpack table (1 KiB, L1). */
static uint32_t cpx_q20tab[256];
static int cpx_q20tab_ok;
static void cpx_q20tab_init(void) {
    if (cpx_q20tab_ok) return;
    for (int b = 0; b < 256; b++)
        cpx_q20tab[b] = (uint32_t)((b >> 0) & 3) |
                        ((uint32_t)((b >> 2) & 3) << 8) |
                        ((uint32_t)((b >> 4) & 3) << 16) |
                        ((uint32_t)((b >> 6) & 3) << 24);
    __sync_synchronize();
    cpx_q20tab_ok = 1;
}

static float dq8_q2_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    float cs = 0.f;
    const uint64_t nb = n / 64;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const uint8_t *qs = blk + 2;
        const __m256i v0 = _mm256_setr_epi32(
            (int)cpx_q20tab[qs[0]], (int)cpx_q20tab[qs[1]],
            (int)cpx_q20tab[qs[2]], (int)cpx_q20tab[qs[3]],
            (int)cpx_q20tab[qs[4]], (int)cpx_q20tab[qs[5]],
            (int)cpx_q20tab[qs[6]], (int)cpx_q20tab[qs[7]]);
        const __m256i v1 = _mm256_setr_epi32(
            (int)cpx_q20tab[qs[8]], (int)cpx_q20tab[qs[9]],
            (int)cpx_q20tab[qs[10]], (int)cpx_q20tab[qs[11]],
            (int)cpx_q20tab[qs[12]], (int)cpx_q20tab[qs[13]],
            (int)cpx_q20tab[qs[14]], (int)cpx_q20tab[qs[15]]);
        for (int h = 0; h < 2; h++) {
            const cpx_q8 *xg = x + 2 * i + h;
            const float dd = d * xg->d;
            accf = _mm256_fmadd_ps(
                _mm256_set1_ps(dd),
                _mm256_cvtepi32_ps(
                    cpx_dot32_u(h ? v1 : v0, xg->q)),
                accf);
            cs += dd * (float)xg->bs;
        }
    }
    return cpx_hsum(accf) - cs;
}

/* ---- AVX-512 int8 kernels (Zen 4/5: VNNI, VBMI, BITALG) ----------------
 * One routine per format covers 1..CPX_DQ8N_MAX tokens.  Each zmm holds
 * two 32-value groups; dpbusd's i32 lanes are the same 4-byte sums as
 * the AVX2 maddubs+madd lanes and the sub-scale multiply is exact, and
 * each pair is folded into the 8-lane f32 accumulator low group first,
 * then high — the AVX2 group order — so every result is bit-identical
 * to the AVX2 kernels (q4_cpx_selftest compares).  The win is in the
 * weight decode: a gather instead of 8 scalar grid loads + inserts per
 * group, sign bits straight into a k-mask, and 64 values per op. */
#define CPX_DQ8N_MAX 8u   /* tokens per window row (dq8n_* and c5_*) */
#define CPX512                                                              \
    __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni," \
                          "avx512vbmi,avx512bitalg")))

static int cpx_no512;   /* selftest/bench: force the AVX2 kernels */
static int cpx_c6 = 1;  /* code-table grid decode (Q4_CPX_C6=0: gather) */
static void c6_init(void);
static int cpx_has512(void) {
    static int v = -1;
    if (v < 0) {
        const char *c6e = getenv("Q4_CPX_C6");
        if (c6e && c6e[0] == '0') cpx_c6 = 0;
        const char *e = getenv("Q4_CPX512");
        v = !(e && e[0] == '0') && __builtin_cpu_supports("avx512vnni") &&
            __builtin_cpu_supports("avx512vbmi") &&
            __builtin_cpu_supports("avx512bitalg") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512dq") &&
            __builtin_cpu_supports("avx512vl");
        if (v) c6_init();
    }
    return v;
}

/* Two adjacent groups' int8 activations (cpx_q8 is 40 B, not packed). */
CPX512 static inline __m512i c5_y2(const cpx_q8 *a) {
    return _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a[0].q)),
        _mm256_loadu_si256((const __m256i *)a[1].q), 1);
}

/* acc += da * f[0..7], then += db * f[8..15]. */
CPX512 static inline __m256 c5_fold(__m256 acc, __m512 f, float da,
                                    float db) {
    acc = _mm256_fmadd_ps(_mm256_set1_ps(da), _mm512_castps512_ps256(f),
                          acc);
    return _mm256_fmadd_ps(_mm256_set1_ps(db), _mm512_extractf32x8_ps(f, 1),
                           acc);
}

/* sc * sum(g * (y negated where km)) per 4-byte lane, as f32. */
CPX512 static inline __m512 c5_dot(__m512i g, __mmask64 km, __m512i sc,
                                   __m512i y) {
    const __m512i ys = _mm512_mask_sub_epi8(y, km, _mm512_setzero_si512(), y);
    return _mm512_cvtepi32_ps(_mm512_mullo_epi32(
        _mm512_dpbusd_epi32(_mm512_setzero_si512(), g, ys), sc));
}

CPX512 static void c5_iq3_s(const uint8_t *w, const cpx_q8 *const *xs,
                            uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                           10, 11, 12, 13, 14, 15);
    const __m512i one = _mm512_set1_epi32(1);
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        for (int j = 0; j < 4; j++) {
            /* 16 grid indices: qs byte | bit k of the pair's qh u16 << 8 */
            const __m512i qz = _mm512_cvtepu8_epi32(
                _mm_loadu_si128((const __m128i *)(blk + 2 + 16 * j)));
            const uint32_t h16 =
                (uint32_t)blk[66 + 2 * j] | ((uint32_t)blk[67 + 2 * j] << 8);
            const __m512i hb = _mm512_and_si512(
                _mm512_srlv_epi32(_mm512_set1_epi32((int)h16), iota), one);
            const __m512i g = _mm512_i32gather_epi32(
                _mm512_or_si512(qz, _mm512_slli_epi32(hb, 8)),
                (const void *)q4_iq3s_grid, 4);
            const __mmask64 km = _cvtu64_mask64(cpx_u64(blk + 74 + 8 * j));
            const uint8_t s = blk[106 + j];
            const __m512i sc = _mm512_mask_set1_epi32(
                _mm512_set1_epi32(2 * (s & 15) + 1), 0xFF00,
                2 * (s >> 4) + 1);
            const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j;
            for (uint32_t t = 0; t < nt; t++) {
                const cpx_q8 *xg = xs[t] + goff;
                acc[t] = c5_fold(acc[t], c5_dot(g, km, sc, c5_y2(xg)),
                                 d * xg[0].d, d * xg[1].d);
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

CPX512 static void c5_iq2_s(const uint8_t *w, const cpx_q8 *const *xs,
                            uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const __m256i iota2 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i scsh = _mm512_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8,
                                           8, 12, 12, 12, 12);
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        for (int j = 0; j < 4; j++) {
            /* 8 u64 grid entries: qs byte | bits 2k..2k+1 of qh u16 << 8 */
            const __m256i qz = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)(qs + 8 * j)));
            const uint32_t h16 =
                (uint32_t)qh[2 * j] | ((uint32_t)qh[2 * j + 1] << 8);
            const __m256i hb = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((int)h16), iota2),
                _mm256_set1_epi32(3));
            const __m512i g = _mm512_i32gather_epi64(
                _mm256_or_si256(qz, _mm256_slli_epi32(hb, 8)),
                (const void *)q4_iq2s_grid, 8);
            const __mmask64 km = _cvtu64_mask64(cpx_u64(blk + 34 + 8 * j));
            const uint32_t sb16 = (uint32_t)blk[74 + 2 * j] |
                                  ((uint32_t)blk[75 + 2 * j] << 8);
            const __m512i sc = _mm512_or_si512(
                _mm512_slli_epi32(
                    _mm512_and_si512(
                        _mm512_srlv_epi32(_mm512_set1_epi32((int)sb16),
                                          scsh),
                        _mm512_set1_epi32(15)),
                    1),
                _mm512_set1_epi32(1));
            const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j;
            for (uint32_t t = 0; t < nt; t++) {
                const cpx_q8 *xg = xs[t] + goff;
                acc[t] = c5_fold(acc[t], c5_dot(g, km, sc, c5_y2(xg)),
                                 d * xg[0].d, d * xg[1].d);
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

CPX512 static void c5_iq3_xxs(const uint8_t *w, const cpx_q8 *const *xs,
                              uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    /* the eight 7-bit sign indices of an aux pair sit at these bit
     * offsets of (aw0 | aw1 << 32); ksigns[i] = i | parity(i) << 7 */
    const __m128i msh = _mm_setr_epi8(0, 7, 14, 21, 32, 39, 46, 53, 0, 0, 0,
                                      0, 0, 0, 0, 0);
    const __m128i m7 = _mm_set1_epi8(0x7f);
    const __m128i b1 = _mm_set1_epi8(1);
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.25f;
        for (int j = 0; j < 4; j++) {
            const __m512i g = _mm512_i32gather_epi32(
                _mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i *)(blk + 2 + 16 * j))),
                (const void *)q4_iq3xxs_grid, 4);
            const uint64_t aw = cpx_u64(blk + 66 + 8 * j);
            const __m128i s7 = _mm_and_si128(
                _mm_multishift_epi64_epi8(msh,
                                          _mm_cvtsi64_si128((long long)aw)),
                m7);
            const __m128i sb = _mm_or_si128(
                s7, _mm_slli_epi16(
                        _mm_and_si128(_mm_popcnt_epi8(s7), b1), 7));
            const __mmask64 km =
                _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(sb));
            const __m512i sc = _mm512_mask_set1_epi32(
                _mm512_set1_epi32(2 * (int)((uint32_t)aw >> 28) + 1), 0xFF00,
                2 * (int)(aw >> 60) + 1);
            const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j;
            for (uint32_t t = 0; t < nt; t++) {
                const cpx_q8 *xg = xs[t] + goff;
                acc[t] = c5_fold(acc[t], c5_dot(g, km, sc, c5_y2(xg)),
                                 d * xg[0].d, d * xg[1].d);
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

/* IQ4_NL: two 18 B blocks per zmm, 128-bit lanes = lo(A) hi(A) lo(B)
 * hi(B) nibbles through the kvalues pshufb; |v| x (y negated where v<0)
 * is cpx_dot32_s.  An odd tail block takes the AVX2 step. */
CPX512 static void c5_iq4_nl(const uint8_t *w, const cpx_q8 *const *xs,
                             uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const __m512i kv = _mm512_broadcast_i32x4(
        _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv));
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    const uint64_t nb = n / 32;
    uint64_t i = 0;
    for (; i + 2 <= nb; i += 2) {
        const uint8_t *ba = w + i * 18;
        const uint8_t *bb = ba + 18;
        uint16_t dha, dhb;
        memcpy(&dha, ba, 2);
        memcpy(&dhb, bb, 2);
        const float da = cpx_h2f(dha), db = cpx_h2f(dhb);
        const __m128i qa = _mm_loadu_si128((const __m128i *)(ba + 2));
        const __m128i qb = _mm_loadu_si128((const __m128i *)(bb + 2));
        __m512i q = _mm512_inserti32x4(
            _mm512_inserti32x4(
                _mm512_inserti32x4(_mm512_castsi128_si512(qa), qa, 1), qb,
                2),
            qb, 3);
        q = _mm512_and_si512(_mm512_mask_srli_epi16(q, 0xFF00FF00u, q, 4),
                             m4);
        const __m512i v = _mm512_shuffle_epi8(kv, q);
        const __m512i av = _mm512_abs_epi8(v);
        const __mmask64 km = _mm512_movepi8_mask(v);
        for (uint32_t t = 0; t < nt; t++) {
            const cpx_q8 *xg = xs[t] + i;
            const __m512i y = c5_y2(xg);
            const __m512i ys =
                _mm512_mask_sub_epi8(y, km, _mm512_setzero_si512(), y);
            acc[t] = c5_fold(acc[t],
                             _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(
                                 _mm512_setzero_si512(), av, ys)),
                             da * xg[0].d, db * xg[1].d);
        }
    }
    if (i < nb) {
        const __m128i kv1 = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
        const __m128i m41 = _mm_set1_epi8(0x0f);
        const __m256i one16 = _mm256_set1_epi16(1);
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const __m256i v = cpx_iq4nl_i8(blk + 2, kv1, m41);
        for (uint32_t t = 0; t < nt; t++) {
            const cpx_q8 *xg = xs[t] + i;
            acc[t] = _mm256_fmadd_ps(
                _mm256_set1_ps(d * xg->d),
                _mm256_cvtepi32_ps(
                    _mm256_madd_epi16(cpx_dot32_s(v, xg->q), one16)),
                acc[t]);
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

/* Q2_0: one 64-value block per zmm; byte k = bits 2k..2k+1 of qs[16]
 * (vpermb puts qs[2L..] in qword L, multishift picks the 2-bit fields). */
CPX512 static void c5_q2_0(const uint8_t *w, const cpx_q8 *const *xs,
                           uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    float cs[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) {
        acc[t] = _mm256_setzero_ps();
        cs[t] = 0.f;
    }
    static const uint8_t ix[64] = {
        0, 1, 2, 3, 4, 5, 6, 7,   2, 3, 4, 5, 6, 7, 8, 9,
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13,
        8, 9, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 0, 1,
        12, 13, 14, 15, 0, 1, 2, 3, 14, 15, 0, 1, 2, 3, 4, 5};
    const __m512i pidx = _mm512_loadu_si512(ix);
    const __m512i shv = _mm512_set1_epi64(0x0e0c0a0806040200ll);
    const __m512i m3 = _mm512_set1_epi8(3);
    const uint64_t nb = n / 64;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const __m512i src = _mm512_zextsi128_si512(
            _mm_loadu_si128((const __m128i *)(blk + 2)));
        const __m512i v = _mm512_and_si512(
            _mm512_multishift_epi64_epi8(shv,
                                         _mm512_permutexvar_epi8(pidx, src)),
            m3);
        for (uint32_t t = 0; t < nt; t++) {
            const cpx_q8 *xg = xs[t] + 2 * i;
            const float dd0 = d * xg[0].d, dd1 = d * xg[1].d;
            acc[t] = c5_fold(acc[t],
                             _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(
                                 _mm512_setzero_si512(), v, c5_y2(xg))),
                             dd0, dd1);
            cs[t] += dd0 * (float)xg[0].bs;
            cs[t] += dd1 * (float)xg[1].bs;
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]) - cs[t];
}

/* ---- AVX-512 grid decode by code tables (c6_*) ------------------------
 * The IQ grids hold few distinct byte values (IQ3_S 8, IQ3_XXS 8, IQ2_S 3),
 * so each entry packs into a 16-bit code (3 or 2 bits per value).  32
 * indices are looked up at once with vpermi2w over the code table held in
 * zmm registers (64 entries per op, blended by the index's high bits), and
 * the codes expand back to bytes with vpermw + vpmultishiftqb + a pshufb
 * value LUT.  That replaces 8 scalar loads + inserts (AVX2) or a slow
 * vpgatherdd (c5_*) per 32 values.  Dot and fold are c5's, so results stay
 * bit-identical to AVX2. */
static uint16_t c6_t3s[512] __attribute__((aligned(64)));
static uint16_t c6_t3x[256] __attribute__((aligned(64)));
static uint16_t c6_t2s[1024] __attribute__((aligned(64)));
static uint8_t c6_l3s[16] __attribute__((aligned(16)));
static uint8_t c6_l3x[16] __attribute__((aligned(16)));
static uint8_t c6_l2s[16] __attribute__((aligned(16)));
static int c6_ok = -1;

/* Codes for one grid: nb bytes per entry, bits per value; 0 on failure
 * (too many distinct values). */
static int c6_build(const uint8_t *grid, uint32_t ne, uint32_t nb,
                    uint32_t bits, uint16_t *tab, uint8_t *lut) {
    uint8_t uv[16];
    uint32_t nu = 0;
    for (uint32_t i = 0; i < ne * nb; i++) {
        uint32_t u = 0;
        for (; u < nu; u++) if (uv[u] == grid[i]) break;
        if (u == nu) {
            if (nu == (1u << bits)) return 0;
            uv[nu++] = grid[i];
        }
    }
    /* sorted rank so the LUT is monotone (not required, just tidy) */
    for (uint32_t a = 0; a < nu; a++)
        for (uint32_t b = a + 1; b < nu; b++)
            if (uv[b] < uv[a]) { uint8_t t = uv[a]; uv[a] = uv[b]; uv[b] = t; }
    memset(lut, 0, 16);
    for (uint32_t u = 0; u < nu; u++) lut[u] = uv[u];
    for (uint32_t e = 0; e < ne; e++) {
        uint32_t c = 0;
        for (uint32_t k = 0; k < nb; k++) {
            uint32_t u = 0;
            while (uv[u] != grid[e * nb + k]) u++;
            c |= u << (bits * k);
        }
        tab[e] = (uint16_t)c;
    }
    return 1;
}

static void c6_init(void) {
    if (c6_ok >= 0) return;
    int ok = c6_build((const uint8_t *)q4_iq3s_grid, 512, 4, 3, c6_t3s, c6_l3s) &&
             c6_build((const uint8_t *)q4_iq3xxs_grid, 256, 4, 3, c6_t3x, c6_l3x) &&
             c6_build((const uint8_t *)q4_iq2s_grid, 1024, 8, 2, c6_t2s, c6_l2s);
    __sync_synchronize();
    c6_ok = ok;
}

/* 32 word indices -> 32 codes; tab has (1 << lg) entries, lg in 8..10. */
CPX512 static inline __m512i c6_lookup(__m512i idx, const uint16_t *tab,
                                       int lg) {
    __m512i r[16];
    const int np = 1 << (lg - 6);
    for (int p = 0; p < np; p++)
        r[p] = _mm512_permutex2var_epi16(
            _mm512_load_si512(tab + 64 * p), idx,
            _mm512_load_si512(tab + 64 * p + 32));
    int m = np;
    for (int bit = 6; m > 1; bit++) {
        const __mmask32 k =
            _mm512_test_epi16_mask(idx, _mm512_set1_epi16((short)(1 << bit)));
        for (int p = 0; p < m / 2; p++)
            r[p] = _mm512_mask_blend_epi16(k, r[2 * p], r[2 * p + 1]);
        m /= 2;
    }
    return r[0];
}

/* Codes of 16 4-byte entries (words base..base+15 of c) -> 64 bytes. */
CPX512 static inline __m512i c6_x4(__m512i c, int base, __m512i lut) {
    const __m512i perm = _mm512_set_epi16(
        0, 0, base + 15, base + 14, 0, 0, base + 13, base + 12,
        0, 0, base + 11, base + 10, 0, 0, base + 9, base + 8,
        0, 0, base + 7, base + 6, 0, 0, base + 5, base + 4,
        0, 0, base + 3, base + 2, 0, 0, base + 1, base + 0);
    const __m512i ctl = _mm512_set1_epi64(0x1916131009060300ll);
    const __m512i e = _mm512_permutexvar_epi16(perm, c);
    return _mm512_shuffle_epi8(
        lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, e),
                              _mm512_set1_epi8(7)));
}

/* Codes of 8 8-byte entries (words base..base+7) -> 64 bytes (2 bits/value). */
CPX512 static inline __m512i c6_x8(__m512i c, int base, __m512i lut) {
    const __m512i perm = _mm512_set_epi16(
        0, 0, 0, base + 7, 0, 0, 0, base + 6, 0, 0, 0, base + 5,
        0, 0, 0, base + 4, 0, 0, 0, base + 3, 0, 0, 0, base + 2,
        0, 0, 0, base + 1, 0, 0, 0, base + 0);
    const __m512i ctl = _mm512_set1_epi64(0x0e0c0a0806040200ll);
    const __m512i e = _mm512_permutexvar_epi16(perm, c);
    return _mm512_shuffle_epi8(
        lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, e),
                              _mm512_set1_epi8(3)));
}

/* Shared tail: one pair of groups (goff, goff+1) for every token. */
#define C6_DOTS(g, km, sc, d)                                                      for (uint32_t t = 0; t < nt; t++) {                                                const cpx_q8 *xg = xs[t] + goff;                                               acc[t] = c5_fold(acc[t], c5_dot((g), (km), (sc), c5_y2(xg)),                                   (d) * xg[0].d, (d) * xg[1].d);                            }

CPX512 static void c6_iq3_s(const uint8_t *w, const cpx_q8 *const *xs,
                            uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)c6_l3s));
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        for (int h = 0; h < 2; h++) {
            /* indices 32h..32h+31: qs byte | qh bit << 8 */
            __m512i idx = _mm512_cvtepu8_epi16(
                _mm256_loadu_si256((const __m256i *)(blk + 2 + 32 * h)));
            idx = _mm512_mask_add_epi16(idx, _cvtu32_mask32(cpx_u32(blk + 66 + 4 * h)),
                                       idx, _mm512_set1_epi16(256));
            const __m512i c = c6_lookup(idx, c6_t3s, 9);
            for (int q = 0; q < 2; q++) {
                const int j = 2 * h + q;
                const __m512i g = c6_x4(c, 16 * q, lut);
                const __mmask64 km = _cvtu64_mask64(cpx_u64(blk + 74 + 8 * j));
                const uint8_t s = blk[106 + j];
                const __m512i sc = _mm512_mask_set1_epi32(
                    _mm512_set1_epi32(2 * (s & 15) + 1), 0xFF00,
                    2 * (s >> 4) + 1);
                const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j;
                C6_DOTS(g, km, sc, d);
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

CPX512 static void c6_iq3_xxs(const uint8_t *w, const cpx_q8 *const *xs,
                              uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)c6_l3x));
    const __m128i msh = _mm_setr_epi8(0, 7, 14, 21, 32, 39, 46, 53, 0, 0, 0,
                                      0, 0, 0, 0, 0);
    const __m128i m7 = _mm_set1_epi8(0x7f);
    const __m128i b1 = _mm_set1_epi8(1);
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.25f;
        for (int h = 0; h < 2; h++) {
            const __m512i idx = _mm512_cvtepu8_epi16(
                _mm256_loadu_si256((const __m256i *)(blk + 2 + 32 * h)));
            const __m512i c = c6_lookup(idx, c6_t3x, 8);
            for (int q = 0; q < 2; q++) {
                const int j = 2 * h + q;
                const __m512i g = c6_x4(c, 16 * q, lut);
                const uint64_t aw = cpx_u64(blk + 66 + 8 * j);
                const __m128i s7 = _mm_and_si128(
                    _mm_multishift_epi64_epi8(msh, _mm_cvtsi64_si128((long long)aw)),
                    m7);
                const __m128i sb = _mm_or_si128(
                    s7, _mm_slli_epi16(_mm_and_si128(_mm_popcnt_epi8(s7), b1), 7));
                const __mmask64 km = _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(sb));
                const __m512i sc = _mm512_mask_set1_epi32(
                    _mm512_set1_epi32(2 * (int)((uint32_t)aw >> 28) + 1), 0xFF00,
                    2 * (int)(aw >> 60) + 1);
                const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j;
                C6_DOTS(g, km, sc, d);
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

CPX512 static void c6_iq2_s(const uint8_t *w, const cpx_q8 *const *xs,
                            uint32_t nt, uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)c6_l2s));
    /* index k's high bits: qh[k/4] >> 2(k%4) */
    const __m512i qsel = _mm512_set_epi16(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5,
                                          4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2,
                                          1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i qshr = _mm512_set_epi16(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0,
                                          6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0,
                                          6, 4, 2, 0, 6, 4, 2, 0);
    const __m512i scsh = _mm512_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8,
                                           8, 12, 12, 12, 12);
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        const __m512i qhw = _mm512_permutexvar_epi16(
            qsel, _mm512_cvtepu8_epi16(_mm256_castsi128_si256(
                      _mm_loadl_epi64((const __m128i *)(blk + 66)))));
        const __m512i hi = _mm512_slli_epi16(
            _mm512_and_si512(_mm512_srlv_epi16(qhw, qshr),
                             _mm512_set1_epi16(3)),
            8);
        const __m512i idx = _mm512_or_si512(
            _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(blk + 2))),
            hi);
        const __m512i c = c6_lookup(idx, c6_t2s, 10);
        for (int j = 0; j < 4; j++) {
            const __m512i g = c6_x8(c, 8 * j, lut);
            const __mmask64 km = _cvtu64_mask64(cpx_u64(blk + 34 + 8 * j));
            const uint32_t sb16 = (uint32_t)blk[74 + 2 * j] |
                                  ((uint32_t)blk[75 + 2 * j] << 8);
            const __m512i sc = _mm512_or_si512(
                _mm512_slli_epi32(
                    _mm512_and_si512(
                        _mm512_srlv_epi32(_mm512_set1_epi32((int)sb16), scsh),
                        _mm512_set1_epi32(15)),
                    1),
                _mm512_set1_epi32(1));
            const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j;
            C6_DOTS(g, km, sc, d);
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

/* 1 when the AVX-512 path took the row. */
static int c5_dot8n(uint32_t type, const uint8_t *w, const cpx_q8 *const *xs,
                    uint32_t nt, uint64_t n, float *out) {
    if (cpx_no512 || !cpx_has512()) return 0;
    if (cpx_c6 && c6_ok > 0) {
        switch (type) {
        case Q4_T_IQ3_S:   c6_iq3_s(w, xs, nt, n, out); return 1;
        case Q4_T_IQ2_S:   c6_iq2_s(w, xs, nt, n, out); return 1;
        case Q4_T_IQ3_XXS: c6_iq3_xxs(w, xs, nt, n, out); return 1;
        }
    }
    switch (type) {
    case Q4_T_IQ3_S:   c5_iq3_s(w, xs, nt, n, out); return 1;
    case Q4_T_IQ2_S:   c5_iq2_s(w, xs, nt, n, out); return 1;
    case Q4_T_IQ3_XXS: c5_iq3_xxs(w, xs, nt, n, out); return 1;
    case Q4_T_IQ4_NL:  c5_iq4_nl(w, xs, nt, n, out); return 1;
    case Q4_T_Q2_0:    c5_q2_0(w, xs, nt, n, out); return 1;
    }
    return 0;
}

static int cpx_dot8(uint32_t type, const uint8_t *w, const cpx_q8 *xq,
                    uint64_t n, float *out) {
    {
        const cpx_q8 *xs1[1] = {xq};
        if (c5_dot8n(type, w, xs1, 1, n, out)) return 1;
    }
    switch (type) {
    case Q4_T_IQ2_XXS: *out = dq8_iq2_xxs(w, xq, n); return 1;
    case Q4_T_IQ2_XS:  *out = dq8_iq2_xs(w, xq, n);  return 1;
    case Q4_T_IQ2_S:   *out = dq8_iq2_s(w, xq, n);   return 1;
    case Q4_T_IQ3_S:   *out = dq8_iq3_s(w, xq, n);   return 1;
    case Q4_T_IQ3_XXS: *out = dq8_iq3_xxs(w, xq, n); return 1;
    case Q4_T_IQ4_XS:  *out = dq8_iq4_xs(w, xq, n);  return 1;
    case Q4_T_IQ4_NL:  *out = dq8_iq4_nl(w, xq, n);  return 1;
    case Q4_T_Q4_0:    *out = dq8_q4_0(w, xq, n);    return 1;
    case Q4_T_Q5_0:    *out = dq8_q5_0(w, xq, n);    return 1;
    case Q4_T_Q8_0:    *out = dq8_q8_0(w, xq, n);    return 1;
    case Q4_T_Q2_0:    *out = dq8_q2_0(w, xq, n);    return 1;
    }
    return 0;
}

/* Weight-side vectors are built once; each token only pays the int8 dot.
 * out[t] matches dq8_*(w, xs[t], n) bit for bit — the per-token FMA order
 * is the single-activation order. nt == 1 is not used here (callers take
 * cpx_dot8). */

static void dq8n_iq2_s(const uint8_t *w, const cpx_q8 *const *xs, uint32_t nt,
                       uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.125f;
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        for (int j = 0; j < 4; j++) {
            const uint64_t m = cpx_u64(blk + 34 + 8 * j);
            for (int half = 0; half < 2; half++) {
                const int o = 4 * half;
                const uint32_t h = qh[2 * j + half];
                const __m256i g = _mm256_set_epi64x(
                    (long long)q4_iq2s_grid[qs[8 * j + o + 3] |
                                            ((h << 2) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o + 2] |
                                            ((h << 4) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o + 1] |
                                            ((h << 6) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o] |
                                            ((h << 8) & 0x300)]);
                const __m256i sgn =
                    cpx_sgn_vec(half ? (uint32_t)(m >> 32) : (uint32_t)m);
                const uint8_t sb = blk[74 + 2 * j + half];
                const __m256i sc =
                    cpx_sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
                const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j +
                                      (uint32_t)half;
                for (uint32_t t = 0; t < nt; t++) {
                    const cpx_q8 *xg = xs[t] + goff;
                    acc[t] = _mm256_fmadd_ps(
                        _mm256_set1_ps(d * xg->d),
                        _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                        acc[t]);
                }
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

static void dq8n_iq3_s(const uint8_t *w, const cpx_q8 *const *xs, uint32_t nt,
                       uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        for (int j = 0; j < 4; j++) {
            const uint8_t *q = blk + 2 + 16 * j;
            const uint64_t m = cpx_u64(blk + 74 + 8 * j);
            for (int half = 0; half < 2; half++) {
                const uint8_t *qb = q + 8 * half;
                const uint32_t h = blk[66 + 2 * j + half];
#define G3N(k) (uint64_t)q4_iq3s_grid[qb[k] | (((h >> k) & 1u) << 8)]
                const __m256i g = _mm256_set_epi64x(
                    (long long)(G3N(6) | (G3N(7) << 32)),
                    (long long)(G3N(4) | (G3N(5) << 32)),
                    (long long)(G3N(2) | (G3N(3) << 32)),
                    (long long)(G3N(0) | (G3N(1) << 32)));
#undef G3N
                const __m256i sgn =
                    cpx_sgn_vec(half ? (uint32_t)(m >> 32) : (uint32_t)m);
                const uint8_t s = blk[106 + j];
                const __m256i sc = _mm256_set1_epi16(
                    (short)(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1));
                const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j +
                                      (uint32_t)half;
                for (uint32_t t = 0; t < nt; t++) {
                    const cpx_q8 *xg = xs[t] + goff;
                    acc[t] = _mm256_fmadd_ps(
                        _mm256_set1_ps(d * xg->d),
                        _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                        acc[t]);
                }
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

static void dq8n_iq3_xxs(const uint8_t *w, const cpx_q8 *const *xs, uint32_t nt,
                         uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh) * 0.25f;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                const uint8_t *q = blk + 2 + 16 * j + 8 * half;
                const uint32_t aw =
                    cpx_u32(blk + 2 + 64 + 8 * j + 4 * half);
#define G3XN(k) (uint64_t)q4_iq3xxs_grid[q[k]]
                const __m256i g = _mm256_set_epi64x(
                    (long long)(G3XN(6) | (G3XN(7) << 32)),
                    (long long)(G3XN(4) | (G3XN(5) << 32)),
                    (long long)(G3XN(2) | (G3XN(3) << 32)),
                    (long long)(G3XN(0) | (G3XN(1) << 32)));
#undef G3XN
                const __m256i sgn = _mm256_set_epi64x(
                    (long long)cpx_esigns[(aw >> 21) & 127],
                    (long long)cpx_esigns[(aw >> 14) & 127],
                    (long long)cpx_esigns[(aw >> 7) & 127],
                    (long long)cpx_esigns[aw & 127]);
                const __m256i sc =
                    _mm256_set1_epi16((short)(2 * (int)(aw >> 28) + 1));
                const uint32_t goff = 8u * (uint32_t)i + 2u * (uint32_t)j +
                                      (uint32_t)half;
                for (uint32_t t = 0; t < nt; t++) {
                    const cpx_q8 *xg = xs[t] + goff;
                    acc[t] = _mm256_fmadd_ps(
                        _mm256_set1_ps(d * xg->d),
                        _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                        acc[t]);
                }
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

static void dq8n_iq4_xs(const uint8_t *w, const cpx_q8 *const *xs, uint32_t nt,
                        uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 136;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        uint16_t sh;
        memcpy(&sh, blk + 2, 2);
        const uint8_t *sl = blk + 4;
        const uint8_t *qs = blk + 8;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                           ((int)((sh >> (2 * ib)) & 3) << 4);
            const __m256i v = cpx_iq4nl_i8(qs + 16 * ib, kv, m4);
            const __m256i sc = _mm256_set1_epi16((short)(ls - 32));
            const uint32_t goff = 8u * (uint32_t)i + (uint32_t)ib;
            for (uint32_t t = 0; t < nt; t++) {
                const cpx_q8 *xg = xs[t] + goff;
                const __m256i p = cpx_dot32_s(v, xg->q);
                acc[t] = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * xg->d),
                    _mm256_cvtepi32_ps(_mm256_madd_epi16(p, sc)), acc[t]);
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

static void dq8n_iq4_nl(const uint8_t *w, const cpx_q8 *const *xs, uint32_t nt,
                        uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const __m256i v = cpx_iq4nl_i8(blk + 2, kv, m4);
        for (uint32_t t = 0; t < nt; t++) {
            const cpx_q8 *xg = xs[t] + i;
            acc[t] = _mm256_fmadd_ps(
                _mm256_set1_ps(d * xg->d),
                _mm256_cvtepi32_ps(
                    _mm256_madd_epi16(cpx_dot32_s(v, xg->q), one16)),
                acc[t]);
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]);
}

static void dq8n_q2_0(const uint8_t *w, const cpx_q8 *const *xs, uint32_t nt,
                      uint64_t n, float *out) {
    __m256 acc[CPX_DQ8N_MAX];
    float cs[CPX_DQ8N_MAX];
    for (uint32_t t = 0; t < nt; t++) {
        acc[t] = _mm256_setzero_ps();
        cs[t] = 0.f;
    }
    const uint64_t nb = n / 64;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = cpx_h2f(dh);
        const uint8_t *qs = blk + 2;
        const __m256i v0 = _mm256_setr_epi32(
            (int)cpx_q20tab[qs[0]], (int)cpx_q20tab[qs[1]],
            (int)cpx_q20tab[qs[2]], (int)cpx_q20tab[qs[3]],
            (int)cpx_q20tab[qs[4]], (int)cpx_q20tab[qs[5]],
            (int)cpx_q20tab[qs[6]], (int)cpx_q20tab[qs[7]]);
        const __m256i v1 = _mm256_setr_epi32(
            (int)cpx_q20tab[qs[8]], (int)cpx_q20tab[qs[9]],
            (int)cpx_q20tab[qs[10]], (int)cpx_q20tab[qs[11]],
            (int)cpx_q20tab[qs[12]], (int)cpx_q20tab[qs[13]],
            (int)cpx_q20tab[qs[14]], (int)cpx_q20tab[qs[15]]);
        for (int h = 0; h < 2; h++) {
            const __m256i vv = h ? v1 : v0;
            for (uint32_t t = 0; t < nt; t++) {
                const cpx_q8 *xg = xs[t] + 2 * i + h;
                const float dd = d * xg->d;
                acc[t] = _mm256_fmadd_ps(
                    _mm256_set1_ps(dd),
                    _mm256_cvtepi32_ps(cpx_dot32_u(vv, xg->q)), acc[t]);
                cs[t] += dd * (float)xg->bs;
            }
        }
    }
    for (uint32_t t = 0; t < nt; t++) out[t] = cpx_hsum(acc[t]) - cs[t];
}

/* Dot one weight row against nt quantized activations. Specialized types
 * build the weight vectors once; anything else repeats cpx_dot8 (correct,
 * no reuse). */
static int cpx_dot8n(uint32_t type, const uint8_t *w, const cpx_q8 *const *xs,
                     uint32_t nt, uint64_t n, float *out) {
    if (!w || !xs || !out || !nt || nt > CPX_DQ8N_MAX) return 0;
    if (nt == 1) return cpx_dot8(type, w, xs[0], n, out);
    if (c5_dot8n(type, w, xs, nt, n, out)) return 1;
    switch (type) {
    case Q4_T_IQ2_S:   dq8n_iq2_s(w, xs, nt, n, out); return 1;
    case Q4_T_IQ3_S:   dq8n_iq3_s(w, xs, nt, n, out); return 1;
    case Q4_T_IQ3_XXS: dq8n_iq3_xxs(w, xs, nt, n, out); return 1;
    case Q4_T_IQ4_XS:  dq8n_iq4_xs(w, xs, nt, n, out); return 1;
    case Q4_T_IQ4_NL:  dq8n_iq4_nl(w, xs, nt, n, out); return 1;
    case Q4_T_Q2_0:    dq8n_q2_0(w, xs, nt, n, out); return 1;
    default:
        for (uint32_t t = 0; t < nt; t++)
            if (!cpx_dot8(type, w, xs[t], n, out + t)) return 0;
        return 1;
    }
}

/* ---------------- job + worker pool ---------------- */

typedef struct {
    const q4_gguf *g;
    q4_expert_cache *c;
    int32_t layer;
    const float *x;
    int32_t ids[CPX_MAX_E];
    float wts[CPX_MAX_E];
    uint32_t k;
    /* nt == 1 is the single-token job (dispatch2).  A verify window sets
     * nt to 2..8: x/y are token-major, sel/wtn say which tokens picked
     * each unique expert, and h is indexed (e * nt + t). */
    uint32_t nt;
    uint32_t xq_stride;           /* cpx_q8 groups between token x slices */
    uint8_t sel[CPX_MAX_E];
    float wtn[CPX_MAX_E][8];
    float *y;                     /* [n_embd] routed result, or [nt][n_embd] */
    float *h;                     /* [k][n_ff] shared intermediates */
    const q4_tensor *tg, *tu, *td;
    uint64_t og, ou, od;
    uint32_t n_embd, n_ff;
    uint32_t chunks_a;            /* k * ceil(n_ff/CHUNK_A) */
    uint32_t chunks_b;            /* k * ceil(n_embd/CHUNK_B) */
    _Atomic uint64_t gen;         /* == tag of the live generation */
    int ok;
    /* int8-activation path: x/h quantized to cpx_q8 groups at dispatch /
     * at the last phase-A chunk.  q8a: gate/up use xq8.  q8b: down uses
     * hq8[e] once hq_rdy[e] is published by the worker that completed
     * that expert's phase A. */
    int q8a, q8b;
    const cpx_q8 *xq8;            /* [n_embd/32] per-slot buffer */
    cpx_q8 *hq8;                  /* [k][n_ff/32] per-slot buffer */
    uint32_t hgn;                 /* n_ff/32 */
    /* Counters every worker writes, each group on its own cache lines and
     * off the read-mostly fields above. Packed, their sharing depended on
     * where the linker put g_pool: a 64-byte-aligned g_pool cost the
     * verify ~7 ms/cycle of cpxwait (2026-10-12, 20 -> 27 ms). */
    _Alignas(64) _Atomic uint32_t next_a;
    _Alignas(64) _Atomic uint32_t b_done;
    _Alignas(64) _Atomic int      hold;  /* workers pinned on this job object */
    _Alignas(64) _Atomic uint32_t next_be[CPX_MAX_E]; /* phase-B cursors */
    _Alignas(64) _Atomic uint32_t a_done_e[CPX_MAX_E];/* phase-A progress */
    _Alignas(64) _Atomic uint32_t hq_rdy[CPX_MAX_E];
    _Alignas(64) char end_[1];
} cpx_job;

/* Multi-slot job ring: the device-route dispatcher publishes one job per
 * MoE layer per token; with a single job slot each layer's ~200µs compute
 * would serialize (~5 ms/token).  Slots let several layers' jobs overlap —
 * workers sweep all slots claiming chunks, so a job queued behind an active
 * one still starts as soon as a worker frees up. */
#define CPX_SLOTS 8

typedef struct {
    pthread_t       th[CPX_MAX_TH];
    int             n_th;
    int             quit;
    pthread_mutex_t mu;
    pthread_cond_t  cv_job;
    pthread_cond_t  cv_done;
    uint64_t        seq;          /* global issue counter (host side only) */
    /* Job objects are static (one per slot): workers pin a slot with
     * jobs[].hold++ then verify gen — stale pointers can never dangle
     * because the objects are never freed. */
    cpx_job         jobs[CPX_SLOTS];
    _Alignas(64) _Atomic int live_a[CPX_SLOTS];   /* published (release) */
    _Atomic uint64_t tag_a[CPX_SLOTS];   /* publish stamp (seq) per slot */
    _Alignas(64) _Atomic uint64_t done_a[CPX_SLOTS];  /* completion stamp per slot */
    _Alignas(64) char sep_[1];
    /* in-order completion floor for the legacy global wait word: slots
     * finish out of order, so comp[] tracks which seqs are done and the
     * floor advances only across contiguous completions. */
    uint64_t        comp[64];
    uint64_t        done_floor;
    /* per-slot intermediates (jobs overlap — h cannot be shared) */
    float          *hbuf[CPX_SLOTS];
    size_t          hsz[CPX_SLOTS];
    /* per-slot quantized activations (xq8 = n_embd/32 groups,
     * hq8 = k * n_ff/32 groups) */
    cpx_q8         *xqbuf[CPX_SLOTS];
    size_t          xqsz[CPX_SLOTS];
    cpx_q8         *hqbuf[CPX_SLOTS];
    size_t          hqsz[CPX_SLOTS];
    /* per-worker scratch for the pread fallback path */
    uint8_t        *scratch[CPX_MAX_TH];
    uint32_t        scratch_sz[CPX_MAX_TH];
} cpx_pool;

static cpx_pool g_pool;
static int g_pool_init;
static int g_n_threads = -1;

/* Pinned seq words shared with the GPU wait kernel (set by the MoE layer,
 * which owns the host-registered block).  The GPU spins on done >= expect
 * instead of the host joining, so both must stay monotonic. */
static volatile unsigned long long *g_cpx_expect;
static volatile unsigned long long *g_cpx_done;

void q4_cpuex_syncpin(unsigned long long *expect,
                      unsigned long long *done) {
    g_cpx_expect = expect;
    g_cpx_done = done;
}

static int cpx_threads(void) {
    if (g_n_threads >= 0) return g_n_threads;
    const char *e = getenv("Q4_CPU_THREADS");
    int v = e && e[0] ? atoi(e) : 12;
    if (v < 1) v = 1;
    if (v > CPX_MAX_TH) v = CPX_MAX_TH;
    long np = sysconf(_SC_NPROCESSORS_ONLN);
    if (np > 0 && v > np - 1) v = (int)(np - 1);
    if (v < 1) v = 1;
    g_n_threads = v;
    return v;
}

/* Rows [r0,r1) of `part` for expert eid. pack = packed expert base or NULL
 * (then read the needed rows straight out of the GGUF). */
static const uint8_t *cpx_part_rows(cpx_job *j, int w, const q4_tensor *t,
                                    uint64_t part_off_in_pack,
                                    int32_t eid, uint32_t r0, uint32_t r1) {
    const uint64_t rb = q4_row_bytes(t->ggml_type, t->ne[0]);
    const uint64_t slice = t->nbytes / (uint64_t)t->n_experts;
    const uint8_t *pack = q4_expert_host_ptr(j->c, j->layer, eid);
    if (pack) return pack + part_off_in_pack + (uint64_t)r0 * rb;
    /* Fallback: pread only the rows this chunk needs. */
    uint64_t need = (uint64_t)(r1 - r0) * rb;
    cpx_pool *p = &g_pool;
    if (p->scratch_sz[w] < need) {
        free(p->scratch[w]);
        p->scratch[w] = malloc((size_t)need);
        p->scratch_sz[w] = p->scratch[w] ? (uint32_t)need : 0;
    }
    if (!p->scratch[w]) return NULL;
    const q4_file *f = &j->g->files[t->shard];
    uint64_t off = f->data_off + t->offset + slice * (uint64_t)eid +
                   (uint64_t)r0 * rb;
    uint8_t *dst = p->scratch[w];
    uint64_t got = 0;
    while (got < need) {
        ssize_t r = pread(f->fd, dst + got, (size_t)(need - got), (off_t)(off + got));
        if (r <= 0) return NULL;
        got += (uint64_t)r;
    }
    return dst;
}

static inline float cpx_silu(float v) { return v / (1.f + expf(-v)); }

/* Tokens that selected expert e, in token order.  nt is at most 8. */
static uint32_t cpx_sel_ts(const cpx_job *j, uint32_t e, uint32_t ts[8]) {
    uint32_t ns = 0, nt = j->nt > 8u ? 8u : j->nt;
    for (uint32_t t = 0; t < nt; t++)
        if (j->sel[e] & (uint8_t)(1u << t)) ts[ns++] = t;
    return ns;
}

/* Dot one weight row with each selected token.  q8 dequants the row once
 * (cpx_dot8n); a rejected type, or q8 off, re-reads the row per token. */
static void cpx_dot_sel(uint32_t type, const uint8_t *w, const uint32_t *ts,
                        uint32_t ns, const float *xbase, uint32_t xstride,
                        const cpx_q8 *xq, uint32_t xq_stride, int q8,
                        uint64_t n, float *out, float *dbuf) {
    if (q8 && xq && ns) {
        const cpx_q8 *xs[8];
        for (uint32_t s = 0; s < ns; s++)
            xs[s] = xq + (size_t)ts[s] * xq_stride;
        if (cpx_dot8n(type, w, xs, ns, n, out)) return;
    }
    for (uint32_t s = 0; s < ns; s++)
        out[s] = cpx_row_dot(type, w, xbase + (size_t)ts[s] * xstride, n,
                             dbuf);
}

/* Window phase A: one expert-row chunk, hidden stored per selected token. */
static void cpx_chunk_a_win(cpx_job *j, int w, uint32_t i) {
    const uint32_t cpa = (j->n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A;
    const uint32_t e = i / cpa;
    const uint32_t r0 = (i % cpa) * CPX_CHUNK_A;
    const uint32_t r1 = r0 + CPX_CHUNK_A < j->n_ff ? r0 + CPX_CHUNK_A : j->n_ff;
    uint32_t ts[8];
    const uint32_t ns = cpx_sel_ts(j, e, ts);
    if (!ns) return;
    const int32_t eid = j->ids[e];
    const uint8_t *wg = cpx_part_rows(j, w, j->tg, j->og, eid, r0, r1);
    const uint8_t *wu = cpx_part_rows(j, w, j->tu, j->ou, eid, r0, r1);
    if (!wg || !wu) {
        j->ok = 0;
        return;
    }
    const uint64_t rbg = q4_row_bytes(j->tg->ggml_type, j->tg->ne[0]);
    const uint64_t rbu = q4_row_bytes(j->tu->ggml_type, j->tu->ne[0]);
    float dbuf[8192];
    float gv[8], uv[8];
    for (uint32_t r = r0; r < r1; r++) {
        cpx_dot_sel(j->tg->ggml_type, wg + (size_t)(r - r0) * rbg, ts, ns,
                    j->x, j->n_embd, j->xq8, j->xq_stride, j->q8a, j->n_embd,
                    gv, dbuf);
        cpx_dot_sel(j->tu->ggml_type, wu + (size_t)(r - r0) * rbu, ts, ns,
                    j->x, j->n_embd, j->xq8, j->xq_stride, j->q8a, j->n_embd,
                    uv, dbuf);
        for (uint32_t s = 0; s < ns; s++) {
            float *h = j->h + (size_t)(e * j->nt + ts[s]) * j->n_ff;
            h[r] = cpx_silu(gv[s]) * uv[s];
        }
    }
}

static inline void cpx_fadd(float *p, float v);

/* Window phase B: one down-row chunk, scattered into each token's y. */
static void cpx_chunk_be_win(cpx_job *j, int w, uint32_t e, uint32_t i) {
    const uint32_t r0 = i * CPX_CHUNK_B;
    const uint32_t r1 = r0 + CPX_CHUNK_B < j->n_embd ? r0 + CPX_CHUNK_B
                                                    : j->n_embd;
    uint32_t ts[8];
    const uint32_t ns = cpx_sel_ts(j, e, ts);
    if (!ns) return;
    const uint64_t rb = q4_row_bytes(j->td->ggml_type, j->td->ne[0]);
    const uint64_t ncols = j->td->ne[0];
    const uint8_t *wd = cpx_part_rows(j, w, j->td, j->od, j->ids[e], r0, r1);
    if (!wd) {
        j->ok = 0;
        return;
    }
    float dbuf[8192];
    float dv[8];
    const float *hbase = j->h + (size_t)e * j->nt * j->n_ff;
    const cpx_q8 *hq = j->hq8 ? j->hq8 + (size_t)e * j->nt * j->hgn : NULL;
    for (uint32_t r = r0; r < r1; r++) {
        cpx_dot_sel(j->td->ggml_type, wd + (size_t)(r - r0) * rb, ts, ns,
                    hbase, j->n_ff, hq, j->hgn, j->q8b && hq, ncols, dv,
                    dbuf);
        for (uint32_t s = 0; s < ns; s++)
            cpx_fadd(&j->y[(size_t)ts[s] * j->n_embd + r],
                     j->wtn[e][ts[s]] * dv[s]);
    }
}

/* Atomic float add: per-expert phase-B chunks overlap the same y range
 * across experts, so the accumulation must be race-free. */
static inline void cpx_fadd(float *p, float v) {
    uint32_t old;
    memcpy(&old, p, 4);
    for (;;) {
        float nv;
        uint32_t nw;
        memcpy(&nv, &old, 4);
        nv += v;
        memcpy(&nw, &nv, 4);
        if (__atomic_compare_exchange_n((uint32_t *)p, &old, nw, true,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

static void cpx_chunk_a(cpx_job *j, int w, uint32_t i) {
    if (j->nt > 1) {
        cpx_chunk_a_win(j, w, i);
        return;
    }
    const uint32_t cpa = (j->n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A;
    const uint32_t e = i / cpa;
    const uint32_t r0 = (i % cpa) * CPX_CHUNK_A;
    const uint32_t r1 = r0 + CPX_CHUNK_A < j->n_ff ? r0 + CPX_CHUNK_A : j->n_ff;
    const int32_t eid = j->ids[e];
    const uint8_t *wg = cpx_part_rows(j, w, j->tg, j->og, eid, r0, r1);
    const uint8_t *wu = cpx_part_rows(j, w, j->tu, j->ou, eid, r0, r1);
    if (!wg || !wu) {
        j->ok = 0;
        return;
    }
    const uint64_t rbg = q4_row_bytes(j->tg->ggml_type, j->tg->ne[0]);
    const uint64_t rbu = q4_row_bytes(j->tu->ggml_type, j->tu->ne[0]);
    float *h = j->h + (size_t)e * j->n_ff;
    float dbuf[8192];
    for (uint32_t r = r0; r < r1; r++) {
        float gv, uv;
        if (j->q8a) {
            if (!cpx_dot8(j->tg->ggml_type,
                          wg + (size_t)(r - r0) * rbg, j->xq8, j->n_embd,
                          &gv))
                gv = cpx_row_dot(j->tg->ggml_type,
                                 wg + (size_t)(r - r0) * rbg, j->x,
                                 j->n_embd, dbuf);
            if (!cpx_dot8(j->tu->ggml_type,
                          wu + (size_t)(r - r0) * rbu, j->xq8, j->n_embd,
                          &uv))
                uv = cpx_row_dot(j->tu->ggml_type,
                                 wu + (size_t)(r - r0) * rbu, j->x,
                                 j->n_embd, dbuf);
        } else {
            gv = cpx_row_dot(j->tg->ggml_type,
                             wg + (size_t)(r - r0) * rbg, j->x, j->n_embd,
                             dbuf);
            uv = cpx_row_dot(j->tu->ggml_type,
                             wu + (size_t)(r - r0) * rbu, j->x, j->n_embd,
                             dbuf);
        }
        h[r] = cpx_silu(gv) * uv;
    }
}

/* Phase-B chunk for a single expert: gated on that expert's phase A so
 * the down-projection of an early-finishing expert overlaps the gate/up
 * work of the later ones. */
static void cpx_chunk_be(cpx_job *j, int w, uint32_t e, uint32_t i) {
    if (j->nt > 1) {
        cpx_chunk_be_win(j, w, e, i);
        return;
    }
    const uint32_t r0 = i * CPX_CHUNK_B;
    const uint32_t r1 = r0 + CPX_CHUNK_B < j->n_embd ? r0 + CPX_CHUNK_B
                                                    : j->n_embd;
    const uint64_t rb = q4_row_bytes(j->td->ggml_type, j->td->ne[0]);
    const uint64_t ncols = j->td->ne[0];
    float dbuf[8192];
    const uint8_t *wd =
        cpx_part_rows(j, w, j->td, j->od, j->ids[e], r0, r1);
    if (!wd) {
        j->ok = 0;
        return;
    }
    const float *h = j->h + (size_t)e * j->n_ff;
    const cpx_q8 *hq = j->hq8 + (size_t)e * j->hgn;
    const float wt = j->wts[e];
    for (uint32_t r = r0; r < r1; r++) {
        float dv;
        if (j->q8b &&
            cpx_dot8(j->td->ggml_type, wd + (size_t)(r - r0) * rb, hq,
                     ncols, &dv))
            cpx_fadd(&j->y[r], wt * dv);
        else
            cpx_fadd(&j->y[r],
                     wt * cpx_row_dot(j->td->ggml_type,
                                      wd + (size_t)(r - r0) * rb, h,
                                      ncols, dbuf));
    }
}

#define CPX_SPIN 200000 /* ~1 ms of _mm_pause before falling back to sleep */

/* Advance the in-order completion floor and publish it to the pinned word
 * the GPU wait kernel spins on.  Called (under p->mu) by the worker that
 * finishes a job's last chunk. */
static void cpx_note_complete(cpx_pool *p, uint64_t seq) {
    pthread_mutex_lock(&p->mu);
    p->comp[seq & 63] = seq;
    while (p->comp[(p->done_floor + 1) & 63] == p->done_floor + 1) {
        p->done_floor++;
        p->comp[p->done_floor & 63] = 0;
    }
    if (g_cpx_done)
        __atomic_store_n(g_cpx_done, p->done_floor, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->cv_done);
    pthread_mutex_unlock(&p->mu);
}

static int cpx_any_job(cpx_pool *p) {
    for (int s = 0; s < CPX_SLOTS; s++)
        if (atomic_load_explicit(&p->live_a[s], memory_order_acquire) &&
            atomic_load_explicit(&p->done_a[s], memory_order_acquire) !=
                atomic_load_explicit(&p->tag_a[s], memory_order_relaxed))
            return 1;
    return 0;
}

static void *cpx_worker(void *arg) {
    intptr_t w = (intptr_t)arg;
    cpx_pool *p = &g_pool;
    for (;;) {
        int progressed = 0;
        for (int s = 0; s < CPX_SLOTS; s++) {
            uint64_t tag =
                atomic_load_explicit(&p->tag_a[s], memory_order_acquire);
            if (!atomic_load_explicit(&p->live_a[s],
                                      memory_order_acquire) ||
                atomic_load_explicit(&p->done_a[s],
                                     memory_order_acquire) == tag)
                continue;
            cpx_job *j = &p->jobs[s];
            /* Pin the object, then re-verify the generation: the dispatcher
             * may have retired/republished the slot between our tag read and
             * the pin. */
            atomic_fetch_add_explicit(&j->hold, 1, memory_order_seq_cst);
            if (atomic_load_explicit(&j->gen, memory_order_acquire) != tag) {
                atomic_fetch_sub(&j->hold, 1);
                continue;
            }
            const uint32_t cpa =
                (j->n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A;
            const uint32_t cpb =
                (j->n_embd + CPX_CHUNK_B - 1) / CPX_CHUNK_B;
            for (;;) {
                uint32_t i = atomic_fetch_add_explicit(&j->next_a, 1,
                                                       memory_order_relaxed);
                if (i >= j->chunks_a) break;
                cpx_chunk_a(j, (int)w, i);
                progressed = 1;
                /* Per-expert A progress releases that expert's B chunks.
                 * The worker that posts the LAST A chunk also quantizes
                 * that expert's h for the int8 dot path, then publishes
                 * hq_rdy — B chunks gate on it so they never race ahead
                 * of the quantization. */
                const uint32_t ei = i / cpa;
                if (atomic_fetch_add_explicit(&j->a_done_e[ei], 1,
                                              memory_order_release) +
                        1 ==
                        cpa &&
                    j->q8b) {
                    if (j->nt <= 1)
                        cpx_quant8(j->h + (size_t)ei * j->n_ff, j->n_ff,
                                   j->hq8 + (size_t)ei * j->hgn);
                    else {
                        uint32_t ts[8];
                        uint32_t ns = cpx_sel_ts(j, ei, ts);
                        for (uint32_t si = 0; si < ns; si++) {
                            uint32_t t = ts[si];
                            cpx_quant8(j->h +
                                           (size_t)(ei * j->nt + t) * j->n_ff,
                                       j->n_ff,
                                       j->hq8 + (size_t)(ei * j->nt + t) *
                                                    j->hgn);
                        }
                    }
                    atomic_store_explicit(&j->hq_rdy[ei], 1,
                                          memory_order_release);
                }
            }
            /* Claim one B chunk at a time from the first B-ready expert,
             * then rescan: an expert's down projection can start as soon
             * as its own gate/up chunks finish. */
            for (;;) {
                int claimed = 0;
                for (uint32_t e = 0; e < j->k && !claimed; e++) {
                    if (atomic_load_explicit(&j->a_done_e[e],
                                           memory_order_acquire) != cpa)
                        continue;
                    if (j->q8b &&
                        !atomic_load_explicit(&j->hq_rdy[e],
                                              memory_order_acquire))
                        continue;
                    uint32_t i = atomic_fetch_add_explicit(&j->next_be[e],
                                                           1,
                                                           memory_order_relaxed);
                    if (i >= cpb) continue;
                    cpx_chunk_be(j, (int)w, e, i);
                    progressed = 1;
                    claimed = 1;
                    if (atomic_fetch_add(&j->b_done, 1) + 1 == j->chunks_b) {
                        atomic_store_explicit(&p->done_a[s], tag,
                                              memory_order_release);
                        cpx_note_complete(p, tag);
                    }
                }
                if (!claimed) break;
            }
            atomic_fetch_sub(&j->hold, 1);
        }
        if (progressed) continue;
        if (p->quit) return NULL;
        /* Idle: spin briefly (jobs arrive every ~0.5 ms and a condvar wakeup
         * costs tens of us), then sleep until a dispatch broadcasts. */
        uint64_t spins = 0;
        while (spins < CPX_SPIN && !p->quit && !cpx_any_job(p)) {
            _mm_pause();
            spins++;
        }
        if (p->quit) return NULL;
        if (cpx_any_job(p)) continue;
        pthread_mutex_lock(&p->mu);
        while (!p->quit && !cpx_any_job(p))
            pthread_cond_wait(&p->cv_job, &p->mu);
        pthread_mutex_unlock(&p->mu);
        if (p->quit) return NULL;
    }
}

static bool cpx_pool_start(void) {
    if (g_pool_init) return g_pool.n_th > 0;
    g_pool_init = 1;
    memset(&g_pool, 0, sizeof(g_pool));
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.cv_job, NULL);
    pthread_cond_init(&g_pool.cv_done, NULL);
    int want = cpx_threads();
    for (int i = 0; i < want; i++) {
        if (pthread_create(&g_pool.th[g_pool.n_th], NULL, cpx_worker,
                           (void *)(intptr_t)g_pool.n_th) != 0)
            break;
        g_pool.n_th++;
    }
    /* Q4_CPU_PIN=<cpu,cpu,...>: worker i runs on list[i % n] (diagnostic).
     * One 9950X3D CCD reads DRAM at ~61 GB/s, both at ~72-79, but pinning
     * the pool across CCDs did not move the MTP verify jobs (2026-10-12:
     * spread 536, CCD0 569, unpinned ~500 us/job — noise). */
    const char *pin = getenv("Q4_CPU_PIN");
    if (pin && pin[0]) {
        int cpus[64], nc = 0;
        for (const char *q = pin; *q && nc < 64;) {
            cpus[nc++] = atoi(q);
            while (*q && *q != ',') q++;
            if (*q == ',') q++;
        }
        for (int i = 0; i < g_pool.n_th && nc; i++) {
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(cpus[i % nc], &cs);
            (void)pthread_setaffinity_np(g_pool.th[i], sizeof(cs), &cs);
        }
    }
    fprintf(stderr, "q4: cpuexp pool %d threads%s%s\n", g_pool.n_th,
            pin && pin[0] ? " pinned " : "", pin && pin[0] ? pin : "");
    return g_pool.n_th > 0;
}

/* Publish a job into a free slot.  Returns the slot index, -1 on transient
 * backpressure (all slots busy — callers retry after polling/reaping) or
 * -2 on a hard error.  Intermediates are per-slot so overlapping jobs never
 * share h. */
int q4_cpuex_dispatch2(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                       const float *x, const int32_t *ids, const float *wts,
                       uint32_t k, float *y, uint64_t *tag_out) {
    if (!g || !c || !x || !ids || !wts || !y || !cpx_pool_start())
        return -2;
    cpx_fgrid_init();
    const q4_tensor *tg, *tu, *td;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &tg, &tu, &td, &og, &ou, &od, &tot))
        return -2;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (k == 0 || k > CPX_MAX_E) return -2;
    cpx_pool *p = &g_pool;
    pthread_mutex_lock(&p->mu);
    int s = -1;
    for (int i = 0; i < CPX_SLOTS; i++) {
        /* A slot is reusable once its last generation completed and no
         * worker still holds the static job object. */
        if ((!atomic_load_explicit(&p->live_a[i], memory_order_acquire) ||
             atomic_load_explicit(&p->done_a[i], memory_order_acquire) ==
                 atomic_load_explicit(&p->tag_a[i], memory_order_relaxed)) &&
            atomic_load_explicit(&p->jobs[i].hold, memory_order_acquire) ==
                0) {
            s = i;
            break;
        }
    }
    if (s < 0) {
        pthread_mutex_unlock(&p->mu);
        return -1;
    }
    const size_t hneed = (size_t)k * n_ff * sizeof(float);
    if (p->hsz[s] < hneed) {
        free(p->hbuf[s]);
        p->hbuf[s] = (float *)malloc(hneed);
        p->hsz[s] = p->hbuf[s] ? hneed : 0;
    }
    const int q8a = cpx_q8_env() && (n_embd % 32) == 0 &&
                    cpx_q8ok(tg->ggml_type) && cpx_q8ok(tu->ggml_type);
    const int q8b = q8a && (n_ff % 32) == 0 && cpx_q8ok(td->ggml_type);
    const size_t xqneed = q8a ? (n_embd / 32) * sizeof(cpx_q8) : 0;
    const size_t hqneed = q8b ? (size_t)k * (n_ff / 32) * sizeof(cpx_q8)
                              : 0;
    if (p->xqsz[s] < xqneed) {
        free(p->xqbuf[s]);
        p->xqbuf[s] = (cpx_q8 *)malloc(xqneed);
        p->xqsz[s] = p->xqbuf[s] ? xqneed : 0;
    }
    if (p->hqsz[s] < hqneed) {
        free(p->hqbuf[s]);
        p->hqbuf[s] = (cpx_q8 *)malloc(hqneed);
        p->hqsz[s] = p->hqbuf[s] ? hqneed : 0;
    }
    cpx_job *j = &p->jobs[s];
    j->g = g;
    j->c = c;
    j->layer = layer;
    j->x = x;
    j->k = k;
    j->nt = 1;
    memcpy(j->ids, ids, k * sizeof(int32_t));
    memcpy(j->wts, wts, k * sizeof(float));
    j->y = y;
    j->h = p->hbuf[s];
    if (!j->h) {
        pthread_mutex_unlock(&p->mu);
        return -2;
    }
    j->tg = tg;
    j->tu = tu;
    j->td = td;
    j->og = og;
    j->ou = ou;
    j->od = od;
    j->n_embd = n_embd;
    j->n_ff = n_ff;
    j->chunks_a = k * ((n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A);
    j->chunks_b = k * ((n_embd + CPX_CHUNK_B - 1) / CPX_CHUNK_B);
    atomic_store_explicit(&j->next_a, 0, memory_order_relaxed);
    memset(j->next_be, 0, k * sizeof j->next_be[0]);
    memset(j->a_done_e, 0, k * sizeof j->a_done_e[0]);
    atomic_store_explicit(&j->b_done, 0, memory_order_relaxed);
    j->ok = 1;
    j->q8a = q8a && p->xqbuf[s];
    j->q8b = q8b && p->xqbuf[s] && p->hqbuf[s];
    j->xq8 = j->q8a ? p->xqbuf[s] : NULL;
    j->hq8 = j->q8b ? p->hqbuf[s] : NULL;
    j->hgn = n_ff / 32;
    memset(j->hq_rdy, 0, k * sizeof j->hq_rdy[0]);
    if (j->q8a) {
        cpx_fgrid_init();
        cpx_esigns_init();
        cpx_q20tab_init();
        cpx_quant8(x, n_embd, p->xqbuf[s]);
    }
    memset(y, 0, (size_t)n_embd * sizeof(float));
    const uint64_t tag = ++p->seq;
    atomic_store_explicit(&j->gen, tag, memory_order_relaxed);
    atomic_store_explicit(&p->tag_a[s], tag, memory_order_relaxed);
    /* live last: workers see the fully-initialized job. */
    atomic_store_explicit(&p->live_a[s], 1, memory_order_release);
    if (g_cpx_expect)
        __atomic_store_n(g_cpx_expect, tag, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->cv_job);
    pthread_mutex_unlock(&p->mu);
    if (tag_out) *tag_out = tag;
    return s;
}

/* One job for a 2..8 token window.  Unique miss experts are dequantized
 * once and dotted with every token that selected them.  Returns the slot,
 * -1 when every slot is busy, -2 on a hard error (caller falls back to
 * per-token jobs).  y is token-major and is zeroed here, including tokens
 * that contributed no misses. */
int q4_cpuex_dispatch_win(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const float *x, uint32_t n_tok, const int32_t *ids,
                          const float *wts, const uint32_t *tn, float *y,
                          uint64_t *tag_out) {
    if (!g || !c || !x || !ids || !wts || !tn || !y || n_tok < 2 ||
        n_tok > 8 || !cpx_pool_start())
        return -2;
    cpx_fgrid_init();
    const q4_tensor *tg, *tu, *td;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &tg, &tu, &td, &og, &ou, &od, &tot))
        return -2;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    int32_t uids[CPX_MAX_E];
    uint8_t usel[CPX_MAX_E];
    float uw[CPX_MAX_E][8];
    uint32_t k = 0, off = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        if (off + tn[t] > 64) return -2;
        for (uint32_t i = 0; i < tn[t]; i++) {
            const int32_t id = ids[off + i];
            uint32_t u = 0;
            for (; u < k; u++)
                if (uids[u] == id) break;
            if (u == k) {
                if (k >= CPX_MAX_E) return -2;
                uids[k] = id;
                usel[k] = 0;
                memset(uw[k], 0, sizeof uw[k]);
                k++;
            }
            usel[u] = (uint8_t)(usel[u] | (uint8_t)(1u << t));
            uw[u][t] = wts[off + i];
        }
        off += tn[t];
    }
    if (k == 0) return -2;
    cpx_pool *p = &g_pool;
    pthread_mutex_lock(&p->mu);
    int s = -1;
    for (int i = 0; i < CPX_SLOTS; i++) {
        if ((!atomic_load_explicit(&p->live_a[i], memory_order_acquire) ||
             atomic_load_explicit(&p->done_a[i], memory_order_acquire) ==
                 atomic_load_explicit(&p->tag_a[i], memory_order_relaxed)) &&
            atomic_load_explicit(&p->jobs[i].hold, memory_order_acquire) ==
                0) {
            s = i;
            break;
        }
    }
    if (s < 0) {
        pthread_mutex_unlock(&p->mu);
        return -1;
    }
    const size_t hneed = (size_t)k * n_tok * n_ff * sizeof(float);
    if (p->hsz[s] < hneed) {
        free(p->hbuf[s]);
        p->hbuf[s] = (float *)malloc(hneed);
        p->hsz[s] = p->hbuf[s] ? hneed : 0;
    }
    const int q8a = cpx_q8_env() && (n_embd % 32) == 0 &&
                    cpx_q8ok(tg->ggml_type) && cpx_q8ok(tu->ggml_type);
    const int q8b = q8a && (n_ff % 32) == 0 && cpx_q8ok(td->ggml_type);
    const size_t xqneed =
        q8a ? (size_t)n_tok * (n_embd / 32) * sizeof(cpx_q8) : 0;
    const size_t hqneed =
        q8b ? (size_t)k * n_tok * (n_ff / 32) * sizeof(cpx_q8) : 0;
    if (p->xqsz[s] < xqneed) {
        free(p->xqbuf[s]);
        p->xqbuf[s] = (cpx_q8 *)malloc(xqneed);
        p->xqsz[s] = p->xqbuf[s] ? xqneed : 0;
    }
    if (p->hqsz[s] < hqneed) {
        free(p->hqbuf[s]);
        p->hqbuf[s] = (cpx_q8 *)malloc(hqneed);
        p->hqsz[s] = p->hqbuf[s] ? hqneed : 0;
    }
    cpx_job *j = &p->jobs[s];
    j->g = g;
    j->c = c;
    j->layer = layer;
    j->x = x;
    j->k = k;
    j->nt = n_tok;
    j->xq_stride = n_embd / 32;
    memcpy(j->ids, uids, k * sizeof(int32_t));
    memcpy(j->sel, usel, k);
    memcpy(j->wtn, uw, (size_t)k * 8u * sizeof(float));
    j->y = y;
    j->h = p->hbuf[s];
    if (!j->h) {
        pthread_mutex_unlock(&p->mu);
        return -2;
    }
    j->tg = tg;
    j->tu = tu;
    j->td = td;
    j->og = og;
    j->ou = ou;
    j->od = od;
    j->n_embd = n_embd;
    j->n_ff = n_ff;
    j->chunks_a = k * ((n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A);
    j->chunks_b = k * ((n_embd + CPX_CHUNK_B - 1) / CPX_CHUNK_B);
    atomic_store_explicit(&j->next_a, 0, memory_order_relaxed);
    memset(j->next_be, 0, k * sizeof j->next_be[0]);
    memset(j->a_done_e, 0, k * sizeof j->a_done_e[0]);
    atomic_store_explicit(&j->b_done, 0, memory_order_relaxed);
    j->ok = 1;
    j->q8a = q8a && p->xqbuf[s];
    j->q8b = q8b && p->xqbuf[s] && p->hqbuf[s];
    j->xq8 = j->q8a ? p->xqbuf[s] : NULL;
    j->hq8 = j->q8b ? p->hqbuf[s] : NULL;
    j->hgn = n_ff / 32;
    memset(j->hq_rdy, 0, k * sizeof j->hq_rdy[0]);
    if (j->q8a) {
        cpx_fgrid_init();
        cpx_esigns_init();
        cpx_q20tab_init();
        for (uint32_t t = 0; t < n_tok; t++)
            cpx_quant8(x + (size_t)t * n_embd, n_embd,
                       p->xqbuf[s] + (size_t)t * (n_embd / 32));
    }
    memset(y, 0, (size_t)n_tok * n_embd * sizeof(float));
    const uint64_t tag = ++p->seq;
    atomic_store_explicit(&j->gen, tag, memory_order_relaxed);
    atomic_store_explicit(&p->tag_a[s], tag, memory_order_relaxed);
    atomic_store_explicit(&p->live_a[s], 1, memory_order_release);
    if (g_cpx_expect)
        __atomic_store_n(g_cpx_expect, tag, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->cv_job);
    pthread_mutex_unlock(&p->mu);
    if (tag_out) *tag_out = tag;
    return s;
}

/* Tag-aware completion poll: callers that hold a dispatch tag must NOT
 * poll done_a == tag_a — the slot may have been retired and republished
 * (tag_a advanced) while the polled job itself already completed. done_a
 * is monotone (tags come from one global seq), so >= is the right test. */
bool q4_cpuex_tag_done(int slot, uint64_t tag) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS || !tag) return false;
    return atomic_load_explicit(&p->done_a[slot], memory_order_acquire) >=
           tag;
}

/* Wait on a specific dispatch tag: after slot reuse, tag_a advances past
 * the caller's tag — waiting for done_a == (re-read tag) could spin forever
 * (done_a overshoots old tags). Compare >= the captured tag instead. Only
 * clears live when this tag is still the slot's current job — clearing a
 * NEWER job's live flag would leak a running job's slot for re-dispatch. */
bool q4_cpuex_join_tag(int slot, uint64_t tag) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS || !tag) return false;
    uint64_t spins = 0;
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) <
               tag &&
           spins < CPX_SPIN) {
        _mm_pause();
        spins++;
    }
    pthread_mutex_lock(&p->mu);
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) <
           tag)
        pthread_cond_wait(&p->cv_done, &p->mu);
    if (atomic_load_explicit(&p->tag_a[slot], memory_order_relaxed) == tag)
        atomic_store_explicit(&p->live_a[slot], 0, memory_order_release);
    int ok = p->jobs[slot].ok;
    pthread_mutex_unlock(&p->mu);
    return ok != 0;
}

/* Non-blocking completion poll for a dispatched slot. */
bool q4_cpuex_slot_done(int slot) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS) return false;
    return atomic_load_explicit(&p->done_a[slot], memory_order_acquire) ==
               atomic_load_explicit(&p->tag_a[slot], memory_order_relaxed) &&
           atomic_load_explicit(&p->tag_a[slot], memory_order_relaxed) != 0;
}

/* Wait for a dispatched slot, retire it, return its ok flag. */
bool q4_cpuex_join2(int slot) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS) return false;
    uint64_t tag =
        atomic_load_explicit(&p->tag_a[slot], memory_order_acquire);
    uint64_t spins = 0;
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) !=
               tag &&
           spins < CPX_SPIN) {
        _mm_pause();
        spins++;
    }
    pthread_mutex_lock(&p->mu);
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) !=
           tag)
        pthread_cond_wait(&p->cv_done, &p->mu);
    atomic_store_explicit(&p->live_a[slot], 0, memory_order_release);
    int ok = p->jobs[slot].ok;
    pthread_mutex_unlock(&p->mu);
    return ok != 0;
}

/* Wedge diagnostic: dump each slot's publish/complete stamps and the job's
 * internal progress so a stuck job shows WHERE it stopped (A vs B phase,
 * which expert's gate/up never finished). */
void q4_cpuex_diag(void) {
    cpx_pool *p = &g_pool;
    fprintf(stderr, "q4: cpuex floor=%llu seq=%llu slots:",
            (unsigned long long)p->done_floor,
            (unsigned long long)p->seq);
    for (int s = 0; s < CPX_SLOTS; s++) {
        int lv = atomic_load_explicit(&p->live_a[s], memory_order_acquire);
        uint64_t tg = atomic_load_explicit(&p->tag_a[s],
                                           memory_order_relaxed);
        uint64_t dn = atomic_load_explicit(&p->done_a[s],
                                           memory_order_acquire);
        if (!lv && !tg) continue;
        cpx_job *j = &p->jobs[s];
        fprintf(stderr, " [s%d live=%d tag=%llu done=%llu A %u/%u B %u/%u]",
                s, lv, (unsigned long long)tg, (unsigned long long)dn,
                (unsigned)atomic_load(&j->next_a), (unsigned)j->chunks_a,
                (unsigned)atomic_load(&j->b_done), (unsigned)j->chunks_b);
    }
    fprintf(stderr, "\n");
}

/* Retire finished slots so their next dispatch is immediate; returns how
 * many slots are still running. */
int q4_cpuex_retire(void) {
    cpx_pool *p = &g_pool;
    int live = 0;
    for (int s = 0; s < CPX_SLOTS; s++) {
        if (!atomic_load_explicit(&p->live_a[s], memory_order_acquire))
            continue;
        if (atomic_load_explicit(&p->done_a[s], memory_order_acquire) ==
            atomic_load_explicit(&p->tag_a[s], memory_order_relaxed))
            atomic_store_explicit(&p->live_a[s], 0, memory_order_release);
        else
            live++;
    }
    return live;
}

bool q4_cpuex_reap(void) {
    /* Slots self-retire; kept for source compatibility. */
    return true;
}

/* ---------------- selftest ----------------
 * Exercises every int8-activation dot against the float path on random
 * valid encodings (random payload bytes + a proper fp16 scale).  The int8
 * path quantizes x, so it can never be bit-equal — the bound checked is
 * activation-quantization noise: |dq8 - f32| <= tol * (1 + |f32|). */
int q4_cpx_selftest(void) {
    static const struct {
        uint32_t type;
        uint64_t bsz;      /* bytes per super-block */
        uint64_t nblocks;  /* blocks at n=512 */
    } cases[] = {
        {Q4_T_IQ2_XXS, 66,  2},
        {Q4_T_IQ2_XS,  74,  2},
        {Q4_T_IQ2_S,   82,  2},
        {Q4_T_IQ3_S,   110, 2},
        {Q4_T_IQ3_XXS, 98,  2},
        {Q4_T_IQ4_XS,  136, 2},
        {Q4_T_IQ4_NL,  18,  16},
        {Q4_T_Q4_0,    18,  16},
        {Q4_T_Q5_0,    22,  16},
        {Q4_T_Q8_0,    34,  16},
        {Q4_T_Q2_0,    18,  8},
    };
    const uint64_t n = 512;
    cpx_fgrid_init();
    cpx_esigns_init();
    cpx_q20tab_init();
    uint64_t st = 0x9e3779b97f4a7c15ull;
    float x[512], x2[512];
    cpx_q8 xq[512 / 32], xq2[512 / 32];
    uint8_t w[34 * 16];   /* largest case: Q8_0, 544 B for n=512 */
    float wf[512];
    float dbuf[8192];
    int bad = 0;
    for (int trial = 0; trial < 4; trial++) {
        for (uint64_t i = 0; i < n; i++) {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            x[i] = ((float)(int)(st >> 40) / (float)(1 << 22)) * 0.5f;
        }
        cpx_quant8(x, n, xq);
        for (uint64_t i = 0; i < n; i++) {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            x2[i] = ((float)(int)(st >> 40) / (float)(1 << 22)) * 0.5f;
        }
        cpx_quant8(x2, n, xq2);
        const cpx_q8 *xs2[2] = { xq, xq2 };
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
            const uint64_t nbytes = cases[c].bsz * cases[c].nblocks;
            for (uint64_t b = 0; b < nbytes; b++) {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                w[b] = (uint8_t)(st >> 56);
            }
            /* fp16 block scales: d = 0.25 + blk/8 keeps magnitudes sane. */
            for (uint64_t blk = 0; blk < cases[c].nblocks; blk++) {
                uint16_t h16 = _cvtss_sh(0.25f + 0.125f * (float)blk, 0);
                memcpy(w + blk * cases[c].bsz, &h16, 2);
            }
            float ref = cpx_row_dot(cases[c].type, w, x, n, dbuf);
            float got = 0.f;
            if (!cpx_dot8(cases[c].type, w, xq, n, &got)) {
                fprintf(stderr, "cpx q8: type %u not wired\n",
                        cases[c].type);
                bad = 1;
                continue;
            }
            /* Error bound: |x - q*d| <= d/2 per element, so the dot error
             * is bounded by sum |w_i| * dx/2 — absolute, not relative. */
            if (!q4_dequant_row(cases[c].type, w, n, wf)) {
                bad = 1;
                continue;
            }
            float lim = 1e-3f;
            for (uint64_t i = 0; i < n; i++)
                lim += fabsf(wf[i]) * (xq[i / 32].d * 0.5f);
            float err = fabsf(got - ref);
            if (!(err <= lim)) {
                fprintf(stderr,
                        "cpx q8: type %u trial %d got %g ref %g"
                        " (err %g > %g)\n",
                        cases[c].type, trial, got, ref, err, lim);
                bad = 1;
            }
            float b2 = 0.f, o2[2] = { 0.f, 0.f };
            if (!cpx_dot8(cases[c].type, w, xq2, n, &b2) ||
                !cpx_dot8n(cases[c].type, w, xs2, 2, n, o2) ||
                memcmp(&o2[0], &got, sizeof(float)) ||
                memcmp(&o2[1], &b2, sizeof(float))) {
                fprintf(stderr,
                        "cpx q8n: type %u trial %d single %g/%g window %g/%g\n",
                        cases[c].type, trial, got, b2, o2[0], o2[1]);
                bad = 1;
            }
            /* AVX-512 kernels: bit-identical to AVX2 for 1..4 tokens. */
            if (cpx_has512()) {
                const cpx_q8 *xs4[4] = { xq, xq2, xq2, xq };
                for (uint32_t ntt = 1; ntt <= 4; ntt++) {
                    float a5[4] = {0}, a2[4] = {0};
                    cpx_no512 = 0;
                    int ok5 = cpx_dot8n(cases[c].type, w, xs4, ntt, n, a5);
                    cpx_no512 = 1;
                    int ok2 = cpx_dot8n(cases[c].type, w, xs4, ntt, n, a2);
                    cpx_no512 = 0;
                    if (ok5 != ok2 ||
                        (ok5 && memcmp(a5, a2, ntt * sizeof(float)))) {
                        fprintf(stderr,
                                "cpx 512: type %u trial %d nt %u %g/%g\n",
                                cases[c].type, trial, ntt, a5[0], a2[0]);
                        bad = 1;
                    }
                }
            }
        }
    }
    /* quantization round-trip sanity: dequantized codes track x. */
    cpx_quant8(x, n, xq);
    for (uint64_t g = 0; g < n / 32; g++)
        for (int i = 0; i < 32; i++) {
            float xr = (float)xq[g].q[i] * xq[g].d;
            if (fabsf(xr - x[g * 32 + i]) > xq[g].d * 0.51f + 1e-6f) {
                fprintf(stderr, "cpx q8: quant g%llu i%d\n",
                        (unsigned long long)g, i);
                bad = 1;
            }
        }
    if (!bad) printf("ok cpx q8 kernels\n");
    return bad;
}

/* Microbenchmark: row-dot throughput of the float path vs the int8
 * path per format, at n_embd-like row length.  Usage: q4-test --cpxbench */
int q4_cpx_bench(void) {
    static const struct {
        uint32_t type;
        uint64_t bsz, nvals;
        const char *name;
    } cases[] = {
        {Q4_T_IQ2_XXS, 66,  256, "iq2_xxs"},
        {Q4_T_IQ2_XS,  74,  256, "iq2_xs"},
        {Q4_T_IQ2_S,   82,  256, "iq2_s"},
        {Q4_T_IQ3_S,   110, 256, "iq3_s"},
        {Q4_T_IQ3_XXS, 98,  256, "iq3_xxs"},
        {Q4_T_IQ4_XS,  136, 256, "iq4_xs"},
        {Q4_T_IQ4_NL,  18,  32,  "iq4_nl"},
        {Q4_T_Q4_0,    18,  32,  "q4_0"},
        {Q4_T_Q5_0,    22,  32,  "q5_0"},
        {Q4_T_Q8_0,    34,  32,  "q8_0"},
        {Q4_T_Q2_0,    18,  64,  "q2_0"},
        {Q4_T_F16,     2,   1,   "f16"},
    };
    const uint64_t n = 2560;   /* n_embd for the current models */
    const int iters = 40000;
    cpx_fgrid_init();
    cpx_esigns_init();
    cpx_q20tab_init();
    static uint8_t w[8192];
    static float x[2560];
    static cpx_q8 xq[2560 / 32];
    static float dbuf[8192];
    uint64_t st = 42;
    for (uint64_t i = 0; i < n; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        x[i] = ((float)(int)(st >> 40) / (float)(1 << 22)) * 0.5f;
    }
    cpx_quant8(x, n, xq);
    printf("%-8s %10s %10s %10s %8s\n", "type", "f32 GB/s", "q8 GB/s",
           "q8-512", "512/q8");
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const uint64_t nb = n / cases[c].nvals;
        const uint64_t rb = nb * cases[c].bsz;
        if (rb > sizeof(w)) continue;
        for (uint64_t b = 0; b < rb; b++) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            w[b] = (uint8_t)(st >> 56);
        }
        for (uint64_t blk = 0; blk < nb; blk++) {
            uint16_t h16 = _cvtss_sh(0.25f, 0);
            memcpy(w + blk * cases[c].bsz, &h16, 2);
        }
        if (cases[c].type == Q4_T_F16) {
            for (uint64_t i = 0; i < n; i++) {
                uint16_t h16 = _cvtss_sh(x[i] * 0.01f, 0);
                memcpy(w + 2 * i, &h16, 2);
            }
        }
        volatile float sink = 0;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; it++)
            sink += cpx_row_dot(cases[c].type, w, x, n, dbuf);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double t_f = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        float probe;
        if (!cpx_dot8(cases[c].type, w, xq, n, &probe)) {
            /* no int8 path for this format — don't report a fake speedup */
            double gf = rb * (double)iters / t_f / 1e9;
            printf("%-8s %10.1f %10s %8s\n", cases[c].name, gf, "n/a", "n/a");
            continue;
        }
        double tq[2];
        for (int v5 = 0; v5 < 2; v5++) {
            cpx_no512 = !v5;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int it = 0; it < iters; it++) {
                float o;
                if (!cpx_dot8(cases[c].type, w, xq, n, &o)) o = 0.f;
                sink += o;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            tq[v5] = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        }
        cpx_no512 = 0;
        double gf = rb * (double)iters / t_f / 1e9;
        double gq = rb * (double)iters / tq[0] / 1e9;
        double g5 = rb * (double)iters / tq[1] / 1e9;
        printf("%-8s %10.1f %10.1f %10.1f %7.2fx\n", cases[c].name, gf, gq,
               g5, g5 / gq);
        (void)sink;
    }
    return 0;
}
