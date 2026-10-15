/* src/q4_mmvq.cu - 2..8-row dense GEMV on int8 activations (verify).
 * Off by default (Q4_MMVQ=1 turns it on).  The float n-row kernels keep
 * every row bit-identical to the one-row GEMV whatever the batch size;
 * this path does not (its row policy differs by type and n, and int8
 * activations round differently), so a 2-row and a 4-row verify gave row 0
 * different numbers — enough to flip long8k's first answer tokens toward
 * Chinese (MTP decode 4/80 -> 11/80 Chinese starts).  The float kernels
 * re-read the fp32 activation rows per output row; here the rows are
 * quantized once per call to int8 with a float scale per 32 values (plus
 * the int sum per 16, for the Q6_K offset and the K-quant mins), and each
 * lane decodes 16 weights of one scale group once and dots them against
 * every row with sudot4. */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {
#include "q4.h"
}

namespace {

constexpr int MAXK = 16384;
int8_t *g_xq;      /* [8][MAXK] */
float *g_xd;       /* [8][MAXK / 32] */
int32_t *g_xs;     /* [8][MAXK / 16] */
int g_on = -1;      /* 0 off, 1 legacy row policy, 2 verify-invariant */
int g_vscope;       /* inside a verify batch (q4_mmvq_verify) */
const float *g_hold;   /* q4_mmvq_hold: x whose quantization may be reused */
const float *g_cur_x;  /* what g_xq holds now */
int g_cur_k;
uint32_t g_cur_n;

/* one warp per 32-value group: int8 = rint(x / d), d = absmax / 127 */
__global__ void xq8_k(const float *__restrict__ x, int8_t *__restrict__ xq,
                      float *__restrict__ xd, int32_t *__restrict__ xs,
                      int K, int ngrp) {
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (g >= ngrp) return;
    const int lane = threadIdx.x & 31;
    const float v = x[(size_t)g * 32 + lane];
    float a = fabsf(v);
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
    const float d = a / 127.f;
    const int q = d > 0.f ? (int)rintf(v / d) : 0;
    xq[(size_t)g * 32 + lane] = (int8_t)q;
    int s = q;
    for (int o = 8; o; o >>= 1) s += __shfl_xor(s, o, 32);   /* per 16 */
    if (lane == 0) {
        xd[g] = d;
        xs[g * 2] = s;
    }
    if (lane == 16) xs[g * 2 + 1] = s;
    (void)K;
}

__device__ __forceinline__ float h2f(const uint8_t *p) {
    __half h;
    __builtin_memcpy(&h, p, 2);
    return __half2float(h);
}

/* One lane task: 16 consecutive weights of one scale group, decoded into
 * four int8x4 words.  A row's contribution per activation row is
 *   dx(32-group) * (a * dot4(w, xq) + c * sum16(xq)).
 * sgn: the words are signed (sdot4) instead of unsigned 0..63 (sudot4). */
struct mtask {
    uint32_t w[4];
    float a, c;
};

__device__ __forceinline__ uint32_t ld32(const uint8_t *p) {
    return *(const uint32_t *)p;   /* gfx11 global loads take 2-byte alignment */
}

__device__ __forceinline__ uint32_t iq4nl_lut4(uint32_t idx4) {
    const uint32_t sel = idx4 & 0x07070707u;
    const uint32_t mhi = ((idx4 >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t lo = (uint32_t)__builtin_amdgcn_perm(0xF6EADDCFu,
                                                       0xBFAD9881u, sel);
    const uint32_t hi = (uint32_t)__builtin_amdgcn_perm(0x71594535u,
                                                       0x26190D01u, sel);
    return (lo & ~mhi) | (hi & mhi);
}

/* K-quant 6-bit scale / min of sub-block j (ggml get_scale_min_k4) */
__device__ __forceinline__ void k4_scale_min(int j, const uint8_t *q,
                                             int *sc, int *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <int TY> struct mq;
/* Q6_K, 210 B / 256: ql[128] qh[64] scales[16] d */
template <> struct mq<14> {
    static constexpr int bs = 210, qk = 256;
    static constexpr bool sgn = false, has_c = true;
    __device__ static void task(const uint8_t *blk, int g, mtask *t) {
        const int h = g >> 3, q = (g >> 1) & 3, l0 = (g & 1) * 16;
        const uint8_t *qlp = blk + h * 64 + (q & 1) * 32 + l0;
        const uint8_t *qhp = blk + 128 + h * 32 + l0;
        const int sl = 4 * (q >> 1), sh = 2 * q;
#pragma unroll
        for (int i = 0; i < 4; i++)
            t->w[i] = ((ld32(qlp + 4 * i) >> sl) & 0x0F0F0F0Fu) |
                      (((ld32(qhp + 4 * i) >> sh) & 0x03030303u) << 4);
        t->a = h2f(blk + 208) *
               (float)(int8_t)blk[192 + h * 8 + 2 * q + (l0 >> 4)];
        t->c = -32.f * t->a;
    }
    /* first value of task g inside the block */
    __device__ static int e0(int g) {
        return (g >> 3) * 128 + ((g >> 1) & 3) * 32 + (g & 1) * 16;
    }
};
/* Q4_K, 144 B / 256: d dmin scales[12] qs[128] */
template <> struct mq<12> {
    static constexpr int bs = 144, qk = 256;
    static constexpr bool sgn = false, has_c = true;
    __device__ static void task(const uint8_t *blk, int g, mtask *t) {
        const int j = g >> 1, l0 = (g & 1) * 16;
        const uint8_t *qp = blk + 16 + (j >> 1) * 32 + l0;
        const int sh = 4 * (j & 1);
#pragma unroll
        for (int i = 0; i < 4; i++)
            t->w[i] = (ld32(qp + 4 * i) >> sh) & 0x0F0F0F0Fu;
        int sc, m;
        k4_scale_min(j, blk + 4, &sc, &m);
        t->a = h2f(blk) * (float)sc;
        t->c = -h2f(blk + 2) * (float)m;
    }
    __device__ static int e0(int g) { return (g >> 1) * 32 + (g & 1) * 16; }
};
/* Q5_K, 176 B / 256: d dmin scales[12] qh[32] qs[128] */
template <> struct mq<13> {
    static constexpr int bs = 176, qk = 256;
    static constexpr bool sgn = false, has_c = true;
    __device__ static void task(const uint8_t *blk, int g, mtask *t) {
        const int j = g >> 1, l0 = (g & 1) * 16;
        const uint8_t *qp = blk + 48 + (j >> 1) * 32 + l0;
        const uint8_t *hp = blk + 16 + l0;
        const int sh = 4 * (j & 1);
#pragma unroll
        for (int i = 0; i < 4; i++)
            t->w[i] = ((ld32(qp + 4 * i) >> sh) & 0x0F0F0F0Fu) |
                      (((ld32(hp + 4 * i) >> j) & 0x01010101u) << 4);
        int sc, m;
        k4_scale_min(j, blk + 4, &sc, &m);
        t->a = h2f(blk) * (float)sc;
        t->c = -h2f(blk + 2) * (float)m;
    }
    __device__ static int e0(int g) { return (g >> 1) * 32 + (g & 1) * 16; }
};
/* IQ4_XS, 136 B / 256: d scales_h scales_l[4] qs[128] */
template <> struct mq<23> {
    static constexpr int bs = 136, qk = 256;
    static constexpr bool sgn = true, has_c = false;
    __device__ static void task(const uint8_t *blk, int g, mtask *t) {
        const int ib = g >> 1, hi = g & 1;
        const uint8_t *qp = blk + 8 + ib * 16;
#pragma unroll
        for (int i = 0; i < 4; i++)
            t->w[i] = iq4nl_lut4((ld32(qp + 4 * i) >> (4 * hi)) & 0x0F0F0F0Fu);
        const uint32_t shv = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8);
        const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf) |
                       (int)(((shv >> (2 * ib)) & 3u) << 4);
        t->a = h2f(blk) * (float)(ls - 32);
        t->c = 0.f;
    }
    __device__ static int e0(int g) { return (g >> 1) * 32 + (g & 1) * 16; }
};
/* IQ4_NL, 18 B / 32: d qs[16] */
template <> struct mq<20> {
    static constexpr int bs = 18, qk = 32;
    static constexpr bool sgn = true, has_c = false;
    __device__ static void task(const uint8_t *blk, int g, mtask *t) {
        const uint8_t *qp = blk + 2;
#pragma unroll
        for (int i = 0; i < 4; i++)
            t->w[i] = iq4nl_lut4((ld32(qp + 4 * i) >> (4 * g)) & 0x0F0F0F0Fu);
        t->a = h2f(blk);
        t->c = 0.f;
    }
    __device__ static int e0(int g) { return g * 16; }
};

/* warp per output row; lanes stride the row's 16-value tasks */
/* Explicit fmaf + no reassociation: a row's result must not depend on
 * NT (Q4_MMVQ=2 runs every verify batch size, 1..8, through here). */
template <int TY, int NT>
__global__ void __launch_bounds__(256)
mq8_k(const uint8_t *__restrict__ w, const int8_t *__restrict__ xq,
      const float *__restrict__ xd, const int32_t *__restrict__ xs,
      float *__restrict__ y, int K, int rows, float scale) {
#pragma clang fp reassociate(off) contract(off)
    using M = mq<TY>;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    constexpr int tpb = M::qk / 16;
    const int nb = K / M::qk;
    const uint8_t *wr = w + (size_t)row * nb * M::bs;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    for (int task = lane; task < nb * tpb; task += 32) {
        const int b = task / tpb, g = task % tpb;
        mtask m;
        M::task(wr + (size_t)b * M::bs, g, &m);
        const int e0 = b * M::qk + M::e0(g);
        /* every row's activation loads first, then the dots */
        int4 xv[NT];
        float xdv[NT];
        int xsv[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) {
            xv[t] = *(const int4 *)(xq + (size_t)t * K + e0);
            xdv[t] = xd[(size_t)t * (K >> 5) + (e0 >> 5)];
            if (M::has_c) xsv[t] = xs[(size_t)t * (K >> 4) + (e0 >> 4)];
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            int s = 0;
            s = __builtin_amdgcn_sudot4(M::sgn, (int)m.w[0], true, xv[t].x, s, false);
            s = __builtin_amdgcn_sudot4(M::sgn, (int)m.w[1], true, xv[t].y, s, false);
            s = __builtin_amdgcn_sudot4(M::sgn, (int)m.w[2], true, xv[t].z, s, false);
            s = __builtin_amdgcn_sudot4(M::sgn, (int)m.w[3], true, xv[t].w, s, false);
            float v = m.a * (float)s;
            if (M::has_c) v = fmaf(m.c, (float)xsv[t], v);
            acc[t] = fmaf(xdv[t], v, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        float a = acc[t];
        for (int o = 16; o; o >>= 1) a += __shfl_xor(a, o, 32);
        if (lane == 0) y[(size_t)t * rows + row] = scale * a;
    }
}

template <int TY>
bool mq_launch(uint32_t n, const uint8_t *w, float *y, int K, int rows,
               float scale, hipStream_t st) {
    if (K % mq<TY>::qk) return false;
    const dim3 g((rows + 7) / 8), b(256);
    switch (n) {
#define MQ_CASE(N) \
    case N: mq8_k<TY, N><<<g, b, 0, st>>>(w, g_xq, g_xd, g_xs, y, K, rows, scale); break;
    MQ_CASE(1) MQ_CASE(2) MQ_CASE(3) MQ_CASE(4) MQ_CASE(5) MQ_CASE(6)
    MQ_CASE(7) MQ_CASE(8)
#undef MQ_CASE
    default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

} // namespace

/* Allocate outside any stream capture (the verify graphs capture GEMVs). */
extern "C" void q4_mmvq_warm(void) {
    if (g_xq) return;
    if (hipMalloc((void **)&g_xq, (size_t)8 * MAXK) != hipSuccess ||
        hipMalloc((void **)&g_xd, (size_t)8 * (MAXK / 32) * 4) != hipSuccess ||
        hipMalloc((void **)&g_xs, (size_t)8 * (MAXK / 16) * 4) != hipSuccess)
        g_xq = nullptr;
}

/* Q4_MMVQ: 0 / unset off; 1 the type- and n-dependent row policy below
 * (2..8 rows anywhere — it made a verify row depend on the batch size);
 * 2 verify batches only, 1..8 rows, a per-type/shape policy that ignores
 * n, so a row comes out the same whatever the batch. */
extern "C" int q4_mmvq_on(void) {
    if (g_on < 0) {
        const char *e = getenv("Q4_MMVQ");
        g_on = e ? atoi(e) : 0;
        if (g_on < 0 || g_on > 2) g_on = 0;
    }
    return g_on;
}
extern "C" void q4_mmvq_verify(int on) { g_vscope = on; }
extern "C" void q4_mmvq_set(int v) { g_on = v; }

/* false: type / shape not covered (caller keeps its float path). */
extern "C" bool q4_mmvq_n(uint32_t ty, const uint8_t *d_w, uint64_t nrows,
                          uint64_t ncols, const float *d_x, float *d_y,
                          float scale, uint32_t n) {
    const int mode = q4_mmvq_on();
    if (!g_xq || !mode || n < 1 || n > 8 || !d_w || !d_x || !d_y ||
        ncols > (uint64_t)MAXK || (ncols % 32) || nrows > 0x7fffffffu)
        return false;
    if (mode == 2) {
        /* verify batches of any size; the type / shape choice only */
        if (!g_vscope) return false;
        switch (ty) {
        case 14u: case 20u: case 23u: case 13u: break;
        case 12u: if (nrows >= 4096) return false; break;
        default: return false;
        }
    } else {
        /* where the int8 kernels beat the float n-row ones (kernel time,
         * rocprof, 2026-10-13): Q6_K / IQ4_NL from 2 rows, IQ4_XS / Q5_K
         * from 3, Q4_K from 3 on short tensors only (6144 x 2560 ties) */
        if (n < 2) return false;
        switch (ty) {
        case 14u: case 20u: break;
        case 23u: case 13u: if (n < 3) return false; break;
        case 12u: if (n < 3 || nrows >= 4096) return false; break;
        default: return false;
        }
    }
    hipStream_t st = (hipStream_t)q4_hip_stream();
    const int K = (int)ncols, ngrp = (int)(n * (ncols / 32));
    const bool reuse = g_hold && g_hold == d_x && g_cur_x == d_x &&
                       g_cur_k == K && g_cur_n >= n;
    if (!reuse) {
        xq8_k<<<(ngrp + 7) / 8, 256, 0, st>>>(d_x, g_xq, g_xd, g_xs, K, ngrp);
        g_cur_x = d_x;
        g_cur_k = K;
        g_cur_n = n;
    }
    const int R = (int)nrows;
    switch (ty) {
    case 14u: return mq_launch<14>(n, d_w, d_y, K, R, scale, st);
    case 12u: return mq_launch<12>(n, d_w, d_y, K, R, scale, st);
    case 13u: return mq_launch<13>(n, d_w, d_y, K, R, scale, st);
    case 23u: return mq_launch<23>(n, d_w, d_y, K, R, scale, st);
    case 20u: return mq_launch<20>(n, d_w, d_y, K, R, scale, st);
    default: return false;
    }
}

/* Activation sharing: between hold(x) and release(), GEMVs on the same x
 * (same K, no more rows) reuse its quantization — the caller guarantees x
 * is not rewritten inside the scope. */
extern "C" void q4_mmvq_hold(const float *d_x) {
    g_hold = d_x;
    g_cur_x = nullptr;   /* x may have changed since the last call */
}
extern "C" void q4_mmvq_release(void) {
    g_hold = nullptr;
    g_cur_x = nullptr;
}
