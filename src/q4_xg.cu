/* src/q4_xg.cu - prefill routed-expert GEMMs on fp16 WMMA.
 *
 * One launch per projection over a chunk's routed rows (sorted by expert).
 * A block takes one tile of up to XG_NT routed rows of one expert and
 * XG_NR weight rows; the weights are read straight out of the expert's L1
 * slot (no id-indexed gather), dequantized into LDS as fp16 once per tile,
 * and the activations are fp16 rows gathered through atok.
 *   gate/up: weight rows f0..f0+63 of gate and the same of up; the epilogue
 *            writes h = silu(g) * u as fp16 (the down GEMM's input).
 *   down:    128 weight rows, fp32 output.
 * Accumulation is fp32 (v_wmma_f32_16x16x16_f16). */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "q4.h"
}

#define Q4_IQ_DECL static __constant__
#include "q4_iq_grids.inc"

typedef _Float16 xg_h16 __attribute__((ext_vector_type(16)));
typedef float xg_f8 __attribute__((ext_vector_type(8)));

static hipStream_t xg_str(void) { return (hipStream_t)q4_hip_stream(); }

#define XG_NT 128  /* routed rows (tokens) per tile */
#define XG_NR 128  /* weight rows per tile */
#define XG_KC 64   /* k per LDS step */
#define XG_LD (XG_KC + 8)
/* Weights are decoded scaled by XG_WS (the epilogue divides it out):
 * small IQ sub-block scales (d * 0.125 ~ 1e-5) would otherwise land in
 * fp16's subnormal range. */
#define XG_WS 256.f

__device__ static inline uint32_t xg_u16(const uint8_t *p) {
    return *(const uint16_t *)p;
}
/* 2-byte aligned dword (IQ blocks are 82 / 98 / 110 / 18 bytes). */
__device__ static inline uint32_t xg_u32(const uint8_t *p) {
    return xg_u16(p) | xg_u16(p + 2) << 16;
}
__device__ static inline float xg_h2f(uint32_t b) {
    return __half2float(__ushort_as_half((unsigned short)b));
}
/* Packed fp16 decode.  A grid / table byte b becomes the fp16 1024 + b by
 * a v_perm with 0x64 high bytes (exact for b < 1024), one v_pk_add takes
 * the bias off, a sign mask (bit 15 / 31) flips it, one v_pk_mul scales:
 * ~2 VALU per weight instead of ~5 through fp32. */
__device__ static inline uint32_t xg_h2u(__half2 h) {
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}
__device__ static inline __half2 xg_u2h(uint32_t u) {
    __half2 h;
    memcpy(&h, &u, 4);
    return h;
}
/* fp16 pairs (1024 + byte0, 1024 + byte1) / (.. byte2, .. byte3) of w */
__device__ static inline __half2 xg_pb01(uint32_t w) {
    return xg_u2h(__builtin_amdgcn_perm(0x64646464u, w, 0x04010400u));
}
__device__ static inline __half2 xg_pb23(uint32_t w) {
    return xg_u2h(__builtin_amdgcn_perm(0x64646464u, w, 0x04030402u));
}
/* 8 unsigned grid bytes (g0: elements 0..3, g1: 4..7), sign masks m
 * (xg_sgn), scale dl2 -> 4 packed dwords. */
__device__ static inline void xg_g8(uint32_t g0, uint32_t g1, uint4 m,
                                    __half2 dl2, uint32_t *o) {
    const __half2 off = __float2half2_rn(-1024.f);
    o[0] = xg_h2u(__hmul2(xg_u2h(xg_h2u(__hadd2(xg_pb01(g0), off)) ^ m.x), dl2));
    o[1] = xg_h2u(__hmul2(xg_u2h(xg_h2u(__hadd2(xg_pb23(g0), off)) ^ m.y), dl2));
    o[2] = xg_h2u(__hmul2(xg_u2h(xg_h2u(__hadd2(xg_pb01(g1), off)) ^ m.z), dl2));
    o[3] = xg_h2u(__hmul2(xg_u2h(xg_h2u(__hadd2(xg_pb23(g1), off)) ^ m.w), dl2));
}

/* iq4_nl's table + 128 as a register LUT (two v_perm_b32 byte selects
 * blended on nibble bit 3): byte j = kvalues[idx4 byte j & 15] + 128. */
__device__ static inline uint32_t xg_kv4(uint32_t idx4) {
    const uint32_t sel = idx4 & 0x07070707u;
    const uint32_t mhi = ((idx4 >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t lo =
        (uint32_t)__builtin_amdgcn_perm(0x766A5D4Fu, 0x3F2D1801u, sel);
    const uint32_t hi =
        (uint32_t)__builtin_amdgcn_perm(0xF1D9C5B5u, 0xA6998D81u, sel);
    return (lo & ~mhi) | (hi & mhi);
}
/* 4 bytes of 16 packed nibbles at w (lo nibbles: elements 4q.., hi:
 * 16 + 4q..) -> o[2q], o[2q+1], o[8+2q], o[9+2q]. */
__device__ static inline void xg_nl8(uint32_t w, __half2 d2, uint32_t *o,
                                     int q) {
    const __half2 off = __float2half2_rn(-1152.f);
    const uint32_t lo = xg_kv4(w), hi = xg_kv4(w >> 4);
    o[2 * q] = xg_h2u(__hmul2(__hadd2(xg_pb01(lo), off), d2));
    o[2 * q + 1] = xg_h2u(__hmul2(__hadd2(xg_pb23(lo), off), d2));
    o[8 + 2 * q] = xg_h2u(__hmul2(__hadd2(xg_pb01(hi), off), d2));
    o[9 + 2 * q] = xg_h2u(__hmul2(__hadd2(xg_pb23(hi), off), d2));
}

/* 32 contiguous weights of one row starting at k (a multiple of 32) as 16
 * packed fp16 dwords, in two halves so the raw loads can run a step
 * ahead: ld() fetches the block bytes into r[], dec() decodes them (sgn =
 * the block's 256-entry sign-mask table). */
template <int TY> struct XgT;

template <> struct XgT<Q4_T_IQ2_S> {
    __device__ static void ld(const uint8_t *row, uint32_t k, uint32_t *r) {
        const uint8_t *b = row + (k >> 8) * 82u;
        const uint32_t ib = (k >> 5) & 7u;
        r[0] = xg_u16(b) | (uint32_t)b[74 + ib] << 16 | (uint32_t)b[66 + ib] << 24;
        r[1] = xg_u32(b + 2 + ib * 4);
        r[2] = xg_u32(b + 34 + ib * 4);
    }
    __device__ static void dec(const uint32_t *r, uint32_t *o,
                               const uint4 *sgn) {
        const float d = xg_h2f(r[0] & 0xffffu);
        const uint32_t sc = (r[0] >> 16) & 0xffu, qh = r[0] >> 24;
        const uint32_t qs = r[1], sg = r[2];
        const __half2 dl0 = __float2half2_rn(d * (0.5f + (float)(sc & 0xf)) * (0.25f * XG_WS));
        const __half2 dl1 = __float2half2_rn(d * (0.5f + (float)(sc >> 4)) * (0.25f * XG_WS));
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t idx =
                ((qs >> (8 * l)) & 0xffu) | ((qh << (8 - 2 * l)) & 0x300u);
            const uint64_t gv = q4_iq2s_grid[idx];
            xg_g8((uint32_t)gv, (uint32_t)(gv >> 32), sgn[(sg >> (8 * l)) & 0xffu],
                  l < 2 ? dl0 : dl1, o + 4 * l);
        }
    }
};

template <> struct XgT<Q4_T_IQ3_XXS> {
    __device__ static void ld(const uint8_t *row, uint32_t k, uint32_t *r) {
        const uint8_t *b = row + (k >> 8) * 98u;
        const uint32_t ib = (k >> 5) & 7u;
        r[0] = xg_u16(b);
        r[1] = xg_u32(b + 66 + 4 * ib);
        r[2] = xg_u32(b + 2 + ib * 8);
        r[3] = xg_u32(b + 6 + ib * 8);
    }
    __device__ static void dec(const uint32_t *r, uint32_t *o,
                               const uint4 *sgn) {
        const float d = xg_h2f(r[0]);
        const uint32_t aux = r[1];
        const __half2 db = __float2half2_rn(d * (0.5f + (float)(aux >> 28)) * (0.5f * XG_WS));
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t si = (aux >> (7 * l)) & 127u;
            const uint32_t s = si | (__popc(si) & 1u) << 7;
            const uint32_t qq = l < 2 ? r[2] : r[3];
            const uint32_t g1 = q4_iq3xxs_grid[(qq >> (16 * (l & 1))) & 0xffu];
            const uint32_t g2 =
                q4_iq3xxs_grid[(qq >> (16 * (l & 1) + 8)) & 0xffu];
            xg_g8(g1, g2, sgn[s], db, o + 4 * l);
        }
    }
};

template <> struct XgT<Q4_T_IQ3_S> {
    __device__ static void ld(const uint8_t *row, uint32_t k, uint32_t *r) {
        const uint8_t *b = row + (k >> 8) * 110u;
        const uint32_t e0 = k & 255u, p = e0 >> 6, hf = (e0 >> 5) & 1u;
        const uint32_t sc = b[106 + p];
        r[0] = xg_u16(b) | (hf ? (sc >> 4) : (sc & 0xf)) << 16 |
               (uint32_t)b[66 + p * 2 + hf] << 24;
        const uint8_t *qs = b + 2 + p * 16 + hf * 8;
        r[1] = xg_u32(qs);
        r[2] = xg_u32(qs + 4);
        r[3] = xg_u32(b + 74 + p * 8 + hf * 4);
    }
    __device__ static void dec(const uint32_t *r, uint32_t *o,
                               const uint4 *sgn) {
        const float d = xg_h2f(r[0] & 0xffffu);
        const __half2 db =
            __float2half2_rn(d * XG_WS * (1.f + 2.f * (float)((r[0] >> 16) & 0xfu)));
        const uint32_t qhb = r[0] >> 24, sg = r[3];
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t qq = l < 2 ? r[1] : r[2];
            const uint32_t i1 = ((qq >> (16 * (l & 1))) & 0xffu) |
                                ((qhb << (8 - 2 * l)) & 256u);
            const uint32_t i2 = ((qq >> (16 * (l & 1) + 8)) & 0xffu) |
                                ((qhb << (7 - 2 * l)) & 256u);
            xg_g8(q4_iq3s_grid[i1], q4_iq3s_grid[i2],
                  sgn[(sg >> (8 * l)) & 0xffu], db, o + 4 * l);
        }
    }
};

template <> struct XgT<Q4_T_IQ4_XS> {
    __device__ static void ld(const uint8_t *row, uint32_t k, uint32_t *r) {
        const uint8_t *b = row + (k >> 8) * 136u;
        const uint32_t ib = (k >> 5) & 7u;
        const uint32_t sh = xg_u16(b + 2);
        const uint32_t ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                            ((sh >> (2 * ib)) & 3u) << 4;
        r[0] = xg_u16(b) | ls << 16;
        const uint8_t *qs = b + 8 + ib * 16;
#pragma unroll
        for (int q = 0; q < 4; q++) r[1 + q] = xg_u32(qs + 4 * q);
    }
    __device__ static void dec(const uint32_t *r, uint32_t *o,
                               const uint4 *) {
        const __half2 dl = __float2half2_rn(xg_h2f(r[0] & 0xffffu) *
                                            (float)((int)(r[0] >> 16) - 32) * XG_WS);
#pragma unroll
        for (int q = 0; q < 4; q++) xg_nl8(r[1 + q], dl, o, q);
    }
};

template <> struct XgT<Q4_T_IQ4_NL> {
    __device__ static void ld(const uint8_t *row, uint32_t k, uint32_t *r) {
        const uint8_t *b = row + (k >> 5) * 18u;
        r[0] = xg_u16(b);
#pragma unroll
        for (int q = 0; q < 4; q++) r[1 + q] = xg_u32(b + 2 + 4 * q);
    }
    __device__ static void dec(const uint32_t *r, uint32_t *o,
                               const uint4 *) {
        const __half2 d2 =
            __hmul2(xg_u2h(r[0] * 0x10001u), __float2half2_rn(XG_WS));
#pragma unroll
        for (int q = 0; q < 4; q++) xg_nl8(r[1 + q], d2, o, q);
    }
};

template <> struct XgT<Q4_T_Q2_0> {
    __device__ static void ld(const uint8_t *row, uint32_t k, uint32_t *r) {
        const uint8_t *b = row + (k >> 6) * 18u;
        const uint8_t *qs = b + 2 + ((k & 63u) >> 2);
        r[0] = xg_u16(b);
        r[1] = xg_u32(qs);
        r[2] = xg_u32(qs + 4);
    }
    __device__ static void dec(const uint32_t *r, uint32_t *o,
                               const uint4 *) {
        const float d = xg_h2f(r[0]) * XG_WS;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t w = r[1 + h];
#pragma unroll
            for (int j = 0; j < 16; j += 2)
                o[h * 8 + j / 2] = xg_h2u(__floats2half2_rn(
                    d * (float)((int)((w >> (2 * j)) & 3u) - 1),
                    d * (float)((int)((w >> (2 * j + 2)) & 3u) - 1)));
        }
    }
};

/* One LDS step of a wave's MMA: NM 16-row fragments (of 4) x 2.  BF: the
 * LDS tiles hold bf16 (v_wmma_f32_16x16x16_bf16), else fp16. */
typedef short xg_s16 __attribute__((ext_vector_type(16)));
template <int NM, int BF = 0>
__device__ static inline void xg_mma(const __half (*As)[XG_LD],
                                     const __half (*Bs)[XG_LD],
                                     xg_f8 (&acc)[4][2], uint32_t th,
                                     uint32_t fg, uint32_t l) {
#pragma unroll
    for (uint32_t ks = 0; ks < XG_KC; ks += 16) {
        if (BF) {
            const xg_s16 b0 = *(const xg_s16 *)&Bs[fg * 16 + (l & 15u)][ks];
            const xg_s16 b1 =
                *(const xg_s16 *)&Bs[XG_NR / 2 + fg * 16 + (l & 15u)][ks];
#pragma unroll
            for (int m = 0; m < NM; m++) {
                const xg_s16 a =
                    *(const xg_s16 *)&As[th * 64 + m * 16 + (l & 15u)][ks];
                acc[m][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    a, b0, acc[m][0]);
                acc[m][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    a, b1, acc[m][1]);
            }
        } else {
            const xg_h16 b0 = *(const xg_h16 *)&Bs[fg * 16 + (l & 15u)][ks];
            const xg_h16 b1 =
                *(const xg_h16 *)&Bs[XG_NR / 2 + fg * 16 + (l & 15u)][ks];
#pragma unroll
            for (int m = 0; m < NM; m++) {
                const xg_h16 a =
                    *(const xg_h16 *)&As[th * 64 + m * 16 + (l & 15u)][ks];
                acc[m][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a, b0, acc[m][0]);
                acc[m][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a, b1, acc[m][1]);
            }
        }
    }
}

/* tiles[i] = {first routed row, rows (<= XG_NT), expert's L1 slot, 0}.
 * GU: out = h (fp16, [row][ldo]); else out = fp32 [row][ldo]. */
template <int TY, int GU>
__global__ __launch_bounds__(256) void xg_k(
    const uint8_t *__restrict__ abase, uint64_t astride, uint64_t woff0,
    uint64_t woff1, uint32_t rowb, uint32_t K, const int4 *__restrict__ tiles,
    const __half *__restrict__ x, const int32_t *__restrict__ atok,
    uint32_t ldx, void *__restrict__ out, uint32_t ldo) {
    __shared__ __align__(16) __half As[XG_NT][XG_LD];
    __shared__ __align__(16) __half Bs[XG_NR][XG_LD];
    /* sign masks: entry s = 8 sign bits -> 4 dwords, bit 2q -> 15, bit
     * 2q + 1 -> 31 of dword q */
    __shared__ uint4 sgn[256];
    {
        const uint32_t s = threadIdx.x;
        uint32_t m[4];
        for (int q = 0; q < 4; q++)
            m[q] = ((s >> (2 * q)) & 1u) << 15 | ((s >> (2 * q + 1)) & 1u) << 31;
        sgn[s] = make_uint4(m[0], m[1], m[2], m[3]);
    }
    __syncthreads();
    const int4 tl = tiles[blockIdx.y];
    const uint32_t tid = threadIdx.x, w = tid >> 5, l = tid & 31u;
    const uint32_t f0 = blockIdx.x * (GU ? XG_NR / 2 : XG_NR);
    const uint8_t *wb = abase + (uint64_t)tl.z * astride;
    /* dequant unit: weight row br, 32-run `run` of each k step */
    const uint32_t br = tid >> 1, run = tid & 1u;
    const uint8_t *wrow =
        GU ? (br < XG_NR / 2 ? wb + woff0 + (uint64_t)(f0 + br) * rowb
                             : wb + woff1 + (uint64_t)(f0 + br - XG_NR / 2) * rowb)
           : wb + woff0 + (uint64_t)(f0 + br) * rowb;
    /* activation unit: tile row xt, 32 halves at xc */
    const uint32_t xt = tid >> 1, xc = (tid & 1u) * 32u;
    const int32_t rr = tl.x + (int32_t)(xt < (uint32_t)tl.y ? xt : tl.y - 1);
    const __half *xr =
        x + (uint64_t)(atok ? atok[rr] : rr) * ldx + xc;
    /* wave tile: 64 rows (th) x 16 + 16 weight rows (fg, fg + 64); frags
     * past the tile's rows are skipped (wave-uniform). */
    const uint32_t th = w & 1u, fg = w >> 1;
    const uint32_t nm = __builtin_amdgcn_readfirstlane(
        (uint32_t)tl.y > th * 64
            ? min(4u, ((uint32_t)tl.y - th * 64 + 15) / 16)
            : 0u);
    xg_f8 acc[4][2] = {};
    for (uint32_t k0 = 0; k0 < K; k0 += XG_KC) {
        {
            uint4 xv[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
                xv[q] = ((const uint4 *)(xr + k0))[q];
            uint32_t rw[5], o[16];
            XgT<TY>::ld(wrow, k0 + run * 32u, rw);
            XgT<TY>::dec(rw, o, sgn);
#pragma unroll
            for (int q = 0; q < 4; q++)
                *(uint4 *)&Bs[br][run * 32u + 8 * q] =
                    make_uint4(o[4 * q], o[4 * q + 1], o[4 * q + 2], o[4 * q + 3]);
#pragma unroll
            for (int q = 0; q < 4; q++) *(uint4 *)&As[xt][xc + 8 * q] = xv[q];
        }
        __syncthreads();
        switch (nm) {
        case 4: xg_mma<4>(As, Bs, acc, th, fg, l); break;
        case 3: xg_mma<3>(As, Bs, acc, th, fg, l); break;
        case 2: xg_mma<2>(As, Bs, acc, th, fg, l); break;
        case 1: xg_mma<1>(As, Bs, acc, th, fg, l); break;
        default: break;
        }
        __syncthreads();
    }
    const uint32_t fc = fg * 16 + (l & 15u);
#pragma unroll
    for (int m = 0; m < 4; m++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t t = th * 64 + m * 16 + 2 * i + (l >> 4);
            if (t >= (uint32_t)tl.y) continue;
            const uint64_t r = (uint64_t)(tl.x + t) * ldo;
            if (GU) {
                const float g = acc[m][0][i] * (1.f / XG_WS),
                            u = acc[m][1][i] * (1.f / XG_WS);
                ((__half *)out)[r + f0 + fc] = __float2half(
                    fminf(fmaxf(g / (1.f + __expf(-g)) * u, -65504.f), 65504.f));
            } else {
                ((float *)out)[r + f0 + fc] = acc[m][0][i] * (1.f / XG_WS);
                ((float *)out)[r + f0 + XG_NR / 2 + fc] =
                    acc[m][1][i] * (1.f / XG_WS);
            }
        }
}

/* Dense prompt GEMM (the projections hipBLASLt ran): y[t][f] = scale *
 * w[f] . x[t] over 128-token x 128-row tiles with the xg loop; w is fp16
 * (dequantized) or bf16 [N][K], x fp32 [T][K] converted while it is
 * staged into LDS (no separate f32 -> 16-bit pass).  K % 64 == 0. */
__device__ static inline uint32_t dg_pk(float a, float b, int bf) {
    if (bf) {
        uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
        ua += 0x7fffu + ((ua >> 16) & 1u);
        ub += 0x7fffu + ((ub >> 16) & 1u);
        return (ua >> 16) | (ub & 0xffff0000u);
    }
    a = fminf(fmaxf(a, -65504.f), 65504.f);
    b = fminf(fmaxf(b, -65504.f), 65504.f);
    return xg_h2u(__floats2half2_rn(a, b));
}

template <int BF>
__global__ __launch_bounds__(256) void dg_k(
    const uint16_t *__restrict__ w, uint32_t N, uint32_t K,
    const float *__restrict__ x, uint32_t T, float *__restrict__ y,
    uint32_t ldy, float scale) {
    __shared__ __align__(16) __half As[XG_NT][XG_LD];
    __shared__ __align__(16) __half Bs[XG_NR][XG_LD];
    const uint32_t tid = threadIdx.x, wv = tid >> 5, l = tid & 31u;
    const uint32_t f0 = blockIdx.x * XG_NR, t0 = blockIdx.y * XG_NT;
    const uint32_t nt = T - t0 < XG_NT ? T - t0 : XG_NT;
    const uint32_t br = tid >> 1, hf = (tid & 1u) * 32u;
    const uint32_t fr = f0 + br < N ? f0 + br : N - 1;
    const uint16_t *wr = w + (uint64_t)fr * K + hf;
    const float *xr = x + (uint64_t)(t0 + (br < nt ? br : nt - 1)) * K + hf;
    const uint32_t th = wv & 1u, fg = wv >> 1;
    const uint32_t nm = __builtin_amdgcn_readfirstlane(
        nt > th * 64 ? min(4u, (nt - th * 64 + 15) / 16) : 0u);
    xg_f8 acc[4][2] = {};
    for (uint32_t k0 = 0; k0 < K; k0 += XG_KC) {
        {
            uint4 wq[4];
#pragma unroll
            for (int q = 0; q < 4; q++) wq[q] = ((const uint4 *)(wr + k0))[q];
            uint32_t xo[16];
#pragma unroll
            for (int q = 0; q < 8; q++) {
                const float4 v = ((const float4 *)(xr + k0))[q];
                xo[2 * q] = dg_pk(v.x, v.y, BF);
                xo[2 * q + 1] = dg_pk(v.z, v.w, BF);
            }
#pragma unroll
            for (int q = 0; q < 4; q++) {
                *(uint4 *)&Bs[br][hf + 8 * q] = wq[q];
                *(uint4 *)&As[br][hf + 8 * q] =
                    make_uint4(xo[4 * q], xo[4 * q + 1], xo[4 * q + 2], xo[4 * q + 3]);
            }
        }
        __syncthreads();
        switch (nm) {
        case 4: xg_mma<4, BF>(As, Bs, acc, th, fg, l); break;
        case 3: xg_mma<3, BF>(As, Bs, acc, th, fg, l); break;
        case 2: xg_mma<2, BF>(As, Bs, acc, th, fg, l); break;
        case 1: xg_mma<1, BF>(As, Bs, acc, th, fg, l); break;
        default: break;
        }
        __syncthreads();
    }
    const uint32_t c0 = f0 + fg * 16 + (l & 15u), c1 = c0 + XG_NR / 2;
#pragma unroll
    for (int m = 0; m < 4; m++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t t = th * 64 + m * 16 + 2 * i + (l >> 4);
            if (t >= nt) continue;
            float *yr = y + (uint64_t)(t0 + t) * ldy;
            if (c0 < N) yr[c0] = acc[m][0][i] * scale;
            if (c1 < N) yr[c1] = acc[m][1][i] * scale;
        }
}

static int g_dg = -1;
extern "C" void q4_dg_set(int on) { g_dg = on; }
extern "C" int q4_dg_on(void) {
    if (g_dg < 0) {
        const char *e = getenv("Q4_DG");
        g_dg = !(e && e[0] == '0');
    }
    return g_dg;
}

/* y (ldy) = scale * W x for T rows; W fp16 or bf16 [N][K]. */
extern "C" bool q4_dg_gemm(const void *w, int bf16, uint32_t N, uint32_t K,
                           const float *x, uint32_t T, float *y, uint32_t ldy,
                           float scale) {
    if (!N || !T) return true;
    if (K % XG_KC || !w || !x || !y) return false;
    dim3 grid((N + XG_NR - 1) / XG_NR, (T + XG_NT - 1) / XG_NT);
    if (bf16)
        dg_k<1><<<grid, 256, 0, xg_str()>>>((const uint16_t *)w, N, K, x, T,
                                             y, ldy, scale);
    else
        dg_k<0><<<grid, 256, 0, xg_str()>>>((const uint16_t *)w, N, K, x, T,
                                             y, ldy, scale);
    return hipGetLastError() == hipSuccess;
}

__global__ static void xg_cvt_k(const float *__restrict__ x,
                                __half *__restrict__ y, uint64_t n4) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = ((const float4 *)x)[i];
    /* saturate: an out-of-range row must not turn into inf / NaN */
    v.x = fminf(fmaxf(v.x, -65504.f), 65504.f);
    v.y = fminf(fmaxf(v.y, -65504.f), 65504.f);
    v.z = fminf(fmaxf(v.z, -65504.f), 65504.f);
    v.w = fminf(fmaxf(v.w, -65504.f), 65504.f);
    __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
    uint2 o;
    memcpy(&o.x, &a, 4);
    memcpy(&o.y, &b, 4);
    ((uint2 *)y)[i] = o;
}


extern "C" int q4_xg_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_XG");
        v = !(e && e[0] == '0');
    }
    return v;
}

extern "C" bool q4_xg_gu_type(uint32_t t) {
    return t == Q4_T_IQ2_S || t == Q4_T_IQ3_XXS || t == Q4_T_IQ3_S ||
           t == Q4_T_IQ4_XS;
}
extern "C" bool q4_xg_dn_type(uint32_t t) {
    return t == Q4_T_IQ4_NL || t == Q4_T_Q2_0;
}
extern "C" bool q4_xg_shape_ok(uint32_t n_embd, uint32_t n_ff) {
    return n_embd % XG_NR == 0 && n_ff % (XG_NR / 2) == 0 &&
           n_embd % 256 == 0 && n_ff % XG_KC == 0;
}

/* Host tile table: tiles of <= XG_NT routed rows per expert.  ek/ebase:
 * per-expert row counts and first rows; slot: expert -> L1 slot. */
extern "C" uint32_t q4_xg_tiles(const uint32_t *ek, const uint32_t *ebase,
                                const int32_t *slot, uint32_t n_exp,
                                int32_t *out, uint32_t cap) {
    uint32_t n = 0;
    for (uint32_t e = 0; e < n_exp; e++)
        for (uint32_t t = 0; t < ek[e]; t += XG_NT) {
            if (n >= cap) return 0;
            out[4 * n + 0] = (int32_t)(ebase[e] + t);
            out[4 * n + 1] = (int32_t)(ek[e] - t < XG_NT ? ek[e] - t : XG_NT);
            out[4 * n + 2] = slot[e];
            out[4 * n + 3] = (int32_t)e;
            n++;
        }
    return n;
}
extern "C" uint32_t q4_xg_tile_cap(uint32_t rows, uint32_t n_exp) {
    return rows / XG_NT + n_exp;
}

extern "C" bool q4_xg_cvt(const float *x, void *x16, uint64_t n) {
    if (!n) return true;
    if (n % 4) return false;
    const uint64_t n4 = n / 4;
    xg_cvt_k<<<(unsigned)((n4 + 255) / 256), 256, 0, xg_str()>>>(
        x, (__half *)x16, n4);
    return hipGetLastError() == hipSuccess;
}

template <int TY, int GU>
static void xg_launch(dim3 grid, const uint8_t *abase, uint64_t astride,
                      uint64_t w0, uint64_t w1, uint32_t rowb, uint32_t K,
                      const int32_t *tiles, const void *x,
                      const int32_t *atok, uint32_t ldx, void *out,
                      uint32_t ldo) {
    xg_k<TY, GU><<<grid, 256, 0, xg_str()>>>(
        abase, astride, w0, w1, rowb, K, (const int4 *)tiles,
        (const __half *)x, atok, ldx, out, ldo);
}

/* h[row][n_ff] (fp16) = silu(gate . x) * (up . x), x16 rows via atok. */
extern "C" bool q4_xg_gu(uint32_t type, const uint8_t *abase,
                         uint64_t astride, uint64_t og, uint64_t ou,
                         uint32_t n_embd, uint32_t n_ff,
                         const int32_t *d_tiles, uint32_t n_tiles,
                         const void *x16, const int32_t *d_atok, void *h16) {
    if (!n_tiles) return true;
    if (!q4_xg_shape_ok(n_embd, n_ff)) return false;
    const uint32_t rowb = (uint32_t)q4_row_bytes(type, n_embd);
    dim3 grid(n_ff / (XG_NR / 2), n_tiles);
#define XG_GU(T)                                                              \
    case T:                                                                   \
        xg_launch<T, 1>(grid, abase, astride, og, ou, rowb, n_embd, d_tiles,  \
                        x16, d_atok, n_embd, h16, n_ff);                      \
        break;
    switch (type) {
        XG_GU(Q4_T_IQ2_S)
        XG_GU(Q4_T_IQ3_XXS)
        XG_GU(Q4_T_IQ3_S)
        XG_GU(Q4_T_IQ4_XS)
    default:
        return false;
    }
#undef XG_GU
    return hipGetLastError() == hipSuccess;
}

/* out[row][n_embd] (fp32) = down . h[row]. */
extern "C" bool q4_xg_dn(uint32_t type, const uint8_t *abase,
                         uint64_t astride, uint64_t od, uint32_t n_ff,
                         uint32_t n_embd, const int32_t *d_tiles,
                         uint32_t n_tiles, const void *h16, float *out) {
    if (!n_tiles) return true;
    if (!q4_xg_shape_ok(n_embd, n_ff)) return false;
    const uint32_t rowb = (uint32_t)q4_row_bytes(type, n_ff);
    dim3 grid(n_embd / XG_NR, n_tiles);
#define XG_DN(T)                                                              \
    case T:                                                                   \
        xg_launch<T, 0>(grid, abase, astride, od, 0, rowb, n_ff, d_tiles,     \
                        h16, NULL, n_ff, out, n_embd);                        \
        break;
    switch (type) {
        XG_DN(Q4_T_IQ4_NL)
        XG_DN(Q4_T_Q2_0)
    default:
        return false;
    }
#undef XG_DN
    return hipGetLastError() == hipSuccess;
}
