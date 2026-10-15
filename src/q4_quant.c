#include "q4.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define QK 32
#define QK_K 256

#define Q4_IQ_DECL static const
#include "q4_iq_grids.inc"

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static uint8_t iq_byte(uint64_t v, int j) {
    return (uint8_t)((v >> (8 * j)) & 0xffu);
}

float q4_f16_to_f32(uint16_t h) {
    uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            e = 127 - 14;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3ffu;
            u = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

float q4_bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

const char *q4_type_name(uint32_t ggml_type) {
    switch (ggml_type) {
    case Q4_T_F32: return "F32";
    case Q4_T_F16: return "F16";
    case Q4_T_Q5_1: return "Q5_1";
    case Q4_T_Q8_0: return "Q8_0";
    case Q4_T_Q4_K: return "Q4_K";
    case Q4_T_Q5_K: return "Q5_K";
    case Q4_T_Q6_K: return "Q6_K";
    case Q4_T_IQ4_NL: return "IQ4_NL";
    case Q4_T_IQ3_S: return "IQ3_S";
    case Q4_T_IQ2_XXS: return "IQ2_XXS";
    case Q4_T_IQ2_XS: return "IQ2_XS";
    case Q4_T_IQ2_S: return "IQ2_S";
    case Q4_T_IQ3_XXS: return "IQ3_XXS";
    case Q4_T_IQ4_XS: return "IQ4_XS";
    case Q4_T_BF16: return "BF16";
    case Q4_T_Q4_0: return "Q4_0";
    case Q4_T_Q5_0: return "Q5_0";
    case Q4_T_Q2_0: return "Q2_0";
    default: return "?";
    }
}

uint64_t q4_row_bytes(uint32_t ggml_type, uint64_t ncols) {
    switch (ggml_type) {
    case Q4_T_F32: return ncols * 4ull;
    case Q4_T_F16:
    case Q4_T_BF16: return ncols * 2ull;
    case Q4_T_Q8_0: return (ncols / QK) * 34ull;
    case Q4_T_Q4_0: return (ncols / QK) * 18ull;
    case Q4_T_Q5_0: return (ncols / QK) * 22ull;
    case Q4_T_Q5_1: return (ncols / QK) * 24ull;
    case Q4_T_Q2_0:
        if (ncols % 64) return 0;
        return (ncols / 64) * 18ull;
    case Q4_T_Q4_K: return (ncols / QK_K) * 144ull;
    case Q4_T_Q5_K: return (ncols / QK_K) * 176ull;
    case Q4_T_Q6_K:
        if (ncols % QK_K) return 0;
        return (ncols / QK_K) * 210ull;
    case Q4_T_IQ2_XXS:
        if (ncols % QK_K) return 0;
        return (ncols / QK_K) * 66ull;
    case Q4_T_IQ2_XS:
        if (ncols % QK_K) return 0;
        return (ncols / QK_K) * 74ull;
    case Q4_T_IQ2_S:
        if (ncols % QK_K) return 0;
        return (ncols / QK_K) * 82ull;
    case Q4_T_IQ3_S:
        if (ncols % QK_K) return 0;
        return (ncols / QK_K) * 110ull;
    case Q4_T_IQ4_NL:
        if (ncols % QK) return 0;
        return (ncols / QK) * 18ull;
    case Q4_T_IQ3_XXS:
        if (ncols % QK_K) return 0;
        return (ncols / QK_K) * 98ull;
    case Q4_T_IQ4_XS:
        if (ncols % QK_K) return 0;
        return (ncols / QK_K) * 136ull;
    default: return 0;
    }
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dequant_q8_0(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK;
    for (uint64_t i = 0; i < nb; i++) {
        uint16_t dh;
        memcpy(&dh, src + i * 34, 2);
        const float d = q4_f16_to_f32(dh);
        const int8_t *qs = (const int8_t *)(src + i * 34 + 2);
        for (int j = 0; j < QK; j++) dst[i * QK + j] = (float)qs[j] * d;
    }
}

static void dequant_q4_0(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK;
    for (uint64_t i = 0; i < nb; i++) {
        uint16_t dh;
        memcpy(&dh, src + i * 18, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = src + i * 18 + 2;
        for (int j = 0; j < QK / 2; j++) {
            dst[i * QK + j] = d * (float)((int)(qs[j] & 0xf) - 8);
            dst[i * QK + j + QK / 2] = d * (float)((int)(qs[j] >> 4) - 8);
        }
    }
}

static void dequant_q5_0(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK;
    for (uint64_t i = 0; i < nb; i++) {
        uint16_t dh;
        memcpy(&dh, src + i * 22, 2);
        const float d = q4_f16_to_f32(dh);
        uint32_t qh;
        memcpy(&qh, src + i * 22 + 2, 4);
        const uint8_t *qs = src + i * 22 + 6;
        for (int j = 0; j < QK / 2; j++) {
            const uint8_t xh0 = ((qh >> j) << 4) & 0x10;
            const uint8_t xh1 = (qh >> (j + 12)) & 0x10;
            dst[i * QK + j] = d * (float)((int)((qs[j] & 0xf) | xh0) - 16);
            dst[i * QK + j + QK / 2] =
                d * (float)((int)((qs[j] >> 4) | xh1) - 16);
        }
    }
}

/* Q2_0 (ggml type 42): 64-value blocks, fp16 scale + 16 bytes of 2-bit
 * quants, values (q - 1) * d. */
static void dequant_q2_0(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / 64;
    for (uint64_t i = 0; i < nb; i++) {
        uint16_t dh;
        memcpy(&dh, src + i * 18, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = src + i * 18 + 2;
        for (int j = 0; j < 64; j++)
            dst[i * 64 + j] =
                d * (float)((int)((qs[j / 4] >> ((j % 4) * 2)) & 3) - 1);
    }
}

static void dequant_q5_1(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 24;
        uint16_t d16, m16;
        memcpy(&d16, blk, 2);
        memcpy(&m16, blk + 2, 2);
        const float d = q4_f16_to_f32(d16);
        const float m = q4_f16_to_f32(m16);
        uint32_t qh;
        memcpy(&qh, blk + 4, 4);
        const uint8_t *qs = blk + 8;
        for (int j = 0; j < QK / 2; j++) {
            const uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            const uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            const int x0 = (qs[j] & 0x0F) | xh0;
            const int x1 = (qs[j] >> 4) | xh1;
            dst[i * QK + j] = (float)x0 * d + m;
            dst[i * QK + j + QK / 2] = (float)x1 * d + m;
        }
    }
}

static void dequant_q4_k(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    float *y = dst;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 144;
        uint16_t d16, m16;
        memcpy(&d16, blk, 2);
        memcpy(&m16, blk + 2, 2);
        const float d = q4_f16_to_f32(d16);
        const float minv = q4_f16_to_f32(m16);
        const uint8_t *scales = blk + 4;
        const uint8_t *q = blk + 16;
        int is = 0;
        uint8_t sc, mv;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &mv);
            const float d1 = d * sc, m1 = minv * mv;
            get_scale_min_k4(is + 1, scales, &sc, &mv);
            const float d2 = d * sc, m2 = minv * mv;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void dequant_q5_k(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    float *y = dst;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 176;
        uint16_t d16, m16;
        memcpy(&d16, blk, 2);
        memcpy(&m16, blk + 2, 2);
        const float d = q4_f16_to_f32(d16);
        const float minv = q4_f16_to_f32(m16);
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *ql = blk + 16 + QK_K / 8;
        int is = 0;
        uint8_t sc, mv;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &mv);
            const float d1 = d * sc, m1 = minv * mv;
            get_scale_min_k4(is + 1, scales, &sc, &mv);
            const float d2 = d * sc, m2 = minv * mv;
            for (int l = 0; l < 32; l++)
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void dequant_iq4_nl(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        float *y = dst + i * QK;
        for (int j = 0; j < QK / 2; j++) {
            const uint8_t q = qs[j];
            y[j] = d * (float)kvalues_iq4nl[q & 0xf];
            y[j + QK / 2] = d * (float)kvalues_iq4nl[q >> 4];
        }
    }
}

static void dequant_iq2_s(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *scales = blk + 74;
        const uint8_t *signs = qs + 32;
        float *y = dst + i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ib++) {
            const float db0 = d * (0.5f + (float)(scales[ib] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const float dl = l < 2 ? db0 : db1;
                const int idx = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                const uint64_t gv = q4_iq2s_grid[idx];
                const uint8_t sg = signs[l];
                for (int j = 0; j < 8; j++) {
                    const float s = (sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f;
                    *y++ = dl * (float)iq_byte(gv, j) * s;
                }
            }
            qs += 4;
            signs += 4;
        }
    }
}

/* IQ2_XS 74B block: f16 d, qs[32] u16 (9-bit grid index + 7-bit sign-table
 * index), scales[8] (two 4-bit half-scales per 32-group, ls = 2*nibble+1,
 * effective d * 0.125 * ls). */
static void dequant_iq2_xs(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 74;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint16_t *qs = (const uint16_t *)(blk + 2);
        const uint8_t *scales = blk + 66;
        float *y = dst + i * QK_K;
        for (int ib = 0; ib < 8; ib++) {
            for (int l = 0; l < 4; l++) {
                const uint16_t v = qs[l];
                const uint64_t gv = q4_iq2xs_grid[v & 511];
                const uint8_t sg = q4_ksigns_iq2xs[v >> 9];
                const float dl = d * 0.125f *
                                 (float)(2 * ((scales[ib] >> (4 * (l >> 1))) & 0xf) + 1);
                for (int j = 0; j < 8; j++) {
                    y[j] = dl * (float)(uint8_t)iq_byte(gv, j) *
                           ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

/* IQ2_XXS 66B block: f16 d + qs[32] u16. Per 32-val group the four u16 are
 * two u32 words: aux8[0..3] = iq2xxs grid bytes, aux32[1] = four 7-bit sign
 * indexes + a 4-bit scale in the top nibble (effective d * 0.125 *
 * (2*sc+1)). */
static void dequant_iq2_xxs(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 66;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        float *y = dst + i * QK_K;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux32[2];
            memcpy(aux32, blk + 2 + 8 * ib, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const float dl =
                d * 0.125f * (float)(2 * (int)(aux32[1] >> 28) + 1);
            for (int l = 0; l < 4; l++) {
                const uint64_t gv = q4_iq2xxs_grid[aux8[l]];
                const uint8_t sg =
                    q4_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++) {
                    y[j] = dl * (float)(uint8_t)iq_byte(gv, j) *
                           ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

static void dequant_iq3_s(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *signs = blk + 74;
        const uint8_t *scales = blk + 106;
        float *y = dst + i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const float db1 = d * (1.f + 2.f * (float)(scales[ib / 2] & 0xf));
            const float db2 = d * (1.f + 2.f * (float)(scales[ib / 2] >> 4));
            for (int half = 0; half < 2; half++) {
                const float db = half == 0 ? db1 : db2;
                const uint8_t qhb = qh[half];
                for (int l = 0; l < 4; l++) {
                    const int i1 = qs[2 * l] | ((qhb << (8 - 2 * l)) & 256);
                    const int i2 = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256);
                    const uint32_t g1 = q4_iq3s_grid[i1];
                    const uint32_t g2 = q4_iq3s_grid[i2];
                    const uint8_t sg = signs[l];
                    for (int j = 0; j < 4; j++) {
                        y[j] = db * (float)iq_byte(g1, j) *
                               ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f);
                        y[j + 4] = db * (float)iq_byte(g2, j) *
                                   ((sg & q4_kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                    }
                    y += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    }
}

static void dequant_iq3_xxs(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *ss = qs + 64;
        float *y = dst + i * QK_K;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux32;
            memcpy(&aux32, ss + 4 * ib, 4);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint32_t g1 = q4_iq3xxs_grid[qs[2 * l]];
                const uint32_t g2 = q4_iq3xxs_grid[qs[2 * l + 1]];
                for (int j = 0; j < 4; j++) {
                    y[j] = db * (float)iq_byte(g1, j) *
                           ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f);
                    y[j + 4] = db * (float)iq_byte(g2, j) *
                               ((sg & q4_kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

static void dequant_iq4_xs(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 136;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        uint16_t sh;
        memcpy(&sh, blk + 2, 2);
        const uint8_t *sl = blk + 4;
        const uint8_t *qs = blk + 8;
        float *y = dst + i * QK_K;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                           ((int)((sh >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; j++) {
                y[j] = dl * (float)kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

static void dequant_q6_k(const uint8_t *src, uint64_t n, float *dst) {
    const uint64_t nb = n / QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 210;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        uint16_t dh;
        memcpy(&dh, blk + 208, 2);
        const float d = q4_f16_to_f32(dh);
        float *y = dst + i * QK_K;
        for (int n128 = 0; n128 < QK_K; n128 += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)(ql[l] & 0xF) | ((int)(qh[l] & 3) << 4);
                const int q2 = (int)(ql[l + 32] & 0xF) | (((int)(qh[l] >> 2) & 3) << 4);
                const int q3 = (int)(ql[l] >> 4) | (((int)(qh[l] >> 4) & 3) << 4);
                const int q4 = (int)(ql[l + 32] >> 4) | (((int)(qh[l] >> 6) & 3) << 4);
                y[l] = d * (float)sc[is] * (float)(q1 - 32);
                y[l + 32] = d * (float)sc[is + 2] * (float)(q2 - 32);
                y[l + 64] = d * (float)sc[is + 4] * (float)(q3 - 32);
                y[l + 96] = d * (float)sc[is + 6] * (float)(q4 - 32);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void dequant_row_impl(uint32_t type, const uint8_t *src, uint64_t ncols,
                             float *dst) {
    switch (type) {
    case Q4_T_F32:
        memcpy(dst, src, ncols * 4);
        break;
    case Q4_T_F16:
        for (uint64_t i = 0; i < ncols; i++) {
            uint16_t h;
            memcpy(&h, src + i * 2, 2);
            dst[i] = q4_f16_to_f32(h);
        }
        break;
    case Q4_T_BF16:
        for (uint64_t i = 0; i < ncols; i++) {
            uint16_t h;
            memcpy(&h, src + i * 2, 2);
            dst[i] = q4_bf16_to_f32(h);
        }
        break;
    case Q4_T_Q4_0:
        dequant_q4_0(src, ncols, dst);
        break;
    case Q4_T_Q5_0:
        dequant_q5_0(src, ncols, dst);
        break;
    case Q4_T_Q2_0:
        dequant_q2_0(src, ncols, dst);
        break;
    case Q4_T_Q8_0:
        dequant_q8_0(src, ncols, dst);
        break;
    case Q4_T_Q5_1:
        dequant_q5_1(src, ncols, dst);
        break;
    case Q4_T_Q4_K:
        dequant_q4_k(src, ncols, dst);
        break;
    case Q4_T_Q5_K:
        dequant_q5_k(src, ncols, dst);
        break;
    case Q4_T_Q6_K:
        dequant_q6_k(src, ncols, dst);
        break;
    case Q4_T_IQ4_NL:
        dequant_iq4_nl(src, ncols, dst);
        break;
    case Q4_T_IQ2_XXS:
        dequant_iq2_xxs(src, ncols, dst);
        break;
    case Q4_T_IQ2_XS:
        dequant_iq2_xs(src, ncols, dst);
        break;
    case Q4_T_IQ2_S:
        dequant_iq2_s(src, ncols, dst);
        break;
    case Q4_T_IQ3_S:
        dequant_iq3_s(src, ncols, dst);
        break;
    case Q4_T_IQ3_XXS:
        dequant_iq3_xxs(src, ncols, dst);
        break;
    case Q4_T_IQ4_XS:
        dequant_iq4_xs(src, ncols, dst);
        break;
    default:
        memset(dst, 0, ncols * sizeof(float));
        break;
    }
}

static float dot_f32(const float *a, const float *b, uint64_t n) {
    float s = 0;
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        s += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] +
             a[i + 3] * b[i + 3] + a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5] +
             a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
    }
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

bool q4_gemv(uint32_t ggml_type, const uint8_t *w, uint64_t nrows, uint64_t ncols,
             const float *x, float *y, float scale) {
    const uint64_t rb = q4_row_bytes(ggml_type, ncols);
    if (!w || !x || !y || nrows == 0 || ncols == 0 || rb == 0) return false;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        float tmp[2560];
        float *row = ncols <= 2560 ? tmp : NULL;
        float *heap = NULL;
        if (!row) {
            heap = malloc(ncols * sizeof(float));
            row = heap;
        }
        if (row) {
#ifdef _OPENMP
#pragma omp for
#endif
            for (uint64_t r = 0; r < nrows; r++) {
                dequant_row_impl(ggml_type, w + r * rb, ncols, row);
                y[r] += scale * dot_f32(row, x, ncols);
            }
        }
        free(heap);
    }
    return true;
}

bool q4_dequant_row(uint32_t ggml_type, const uint8_t *src, uint64_t ncols,
                    float *dst) {
    if (!src || !dst || ncols == 0) return false;
    if (ggml_type != Q4_T_F32 && q4_row_bytes(ggml_type, ncols) == 0)
        return false;
    dequant_row_impl(ggml_type, src, ncols, dst);
    return true;
}

void q4_silu(float *x, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

void q4_sigmoid(float *x, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

float q4_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

/* Planar Q2_0 expert → ggml Q2_0 rows.
 * Planes, one expert, H=2560 FF=640, block = fp16 scale + 16 code bytes:
 *   gate/up codes  at 0,        1280 slots × 40 × 16
 *   down codes     at 819200,   2560 rows  × 10 × 16
 *   gate/up scales at 1228800,  1280 slots × 40 × 2
 *   down scales    at 1331200,  2560 rows  × 10 × 2
 * Gate/up slot 2r is gate row r and slot 2r+1 is up row r (same for scales).
 * ggml pack: gate rows at 0 (640 × 40 × 18), up at 460800, down at 921600. */
static void q2_pack_row(uint8_t *dst, const uint8_t *codes, const uint8_t *scales,
                        uint32_t nb) {
    for (uint32_t blk = 0; blk < nb; blk++) {
        memcpy(dst + (size_t)blk * 18, scales + (size_t)blk * 2, 2);
        memcpy(dst + (size_t)blk * 18 + 2, codes + (size_t)blk * 16, 16);
    }
}

bool q4_q2_planar_to_ggml(const uint8_t *planar, uint8_t *ggml) {
    if (!planar || !ggml) return false;
    const uint32_t gu_nb = Q4_Q2_PACK_H / 64u;          /* 40 */
    const uint32_t dn_nb = Q4_Q2_PACK_FF / 64u;         /* 10 */
    const uint32_t gu_crow = gu_nb * 16u;               /* 640 */
    const uint32_t gu_srow = gu_nb * 2u;                /* 80 */
    const uint32_t dn_crow = dn_nb * 16u;               /* 160 */
    const uint32_t dn_srow = dn_nb * 2u;                /* 20 */
    const uint32_t gu_grow = gu_nb * 18u;               /* 720 */
    const uint32_t dn_grow = dn_nb * 18u;               /* 180 */
    const uint8_t *gu_c = planar;
    const uint8_t *dn_c = planar + 819200u;
    const uint8_t *gu_s = planar + 1228800u;
    const uint8_t *dn_s = planar + 1331200u;
    for (uint32_t r = 0; r < Q4_Q2_PACK_FF; r++) {
        uint32_t sg = r * 2u;
        uint32_t su = sg + 1u;
        q2_pack_row(ggml + (size_t)r * gu_grow,
                    gu_c + (size_t)sg * gu_crow, gu_s + (size_t)sg * gu_srow,
                    gu_nb);
        q2_pack_row(ggml + 460800u + (size_t)r * gu_grow,
                    gu_c + (size_t)su * gu_crow, gu_s + (size_t)su * gu_srow,
                    gu_nb);
    }
    for (uint32_t r = 0; r < Q4_Q2_PACK_H; r++)
        q2_pack_row(ggml + 921600u + (size_t)r * dn_grow,
                    dn_c + (size_t)r * dn_crow, dn_s + (size_t)r * dn_srow,
                    dn_nb);
    return true;
}

void q4_rms_norm(const float *x, const float *w, float *y, uint64_t n, float eps) {
    double ss = 0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    if (w) {
        for (uint64_t i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
    } else {
        for (uint64_t i = 0; i < n; i++) y[i] = x[i] * scale;
    }
}

void q4_l2_norm(float *x, uint64_t n, float eps) {
    double ss = 0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)ss + eps);
    for (uint64_t i = 0; i < n; i++) x[i] *= scale;
}

void q4_softmax(float *x, uint64_t n) {
    float m = x[0];
    for (uint64_t i = 1; i < n; i++)
        if (x[i] > m) m = x[i];
    float s = 0;
    for (uint64_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    const float inv = s > 0 ? 1.0f / s : 0;
    for (uint64_t i = 0; i < n; i++) x[i] *= inv;
}

void q4_topk(const float *logits, uint32_t n, uint32_t k, int32_t *ids, float *wts,
             bool renormalize) {
    if (k > n) k = n;
    for (uint32_t i = 0; i < k; i++) {
        ids[i] = -1;
        wts[i] = -INFINITY;
    }
    for (uint32_t i = 0; i < n; i++) {
        const float v = logits[i];
        uint32_t p = k;
        while (p > 0 && (ids[p - 1] < 0 || v > wts[p - 1])) p--;
        if (p == k) continue;
        for (uint32_t j = k - 1; j > p; j--) {
            ids[j] = ids[j - 1];
            wts[j] = wts[j - 1];
        }
        ids[p] = (int32_t)i;
        wts[p] = v;
    }
    if (renormalize) {
        float s = 0;
        for (uint32_t i = 0; i < k; i++) s += wts[i];
        if (s > 0) {
            const float inv = 1.0f / s;
            for (uint32_t i = 0; i < k; i++) wts[i] *= inv;
        }
    }
}
