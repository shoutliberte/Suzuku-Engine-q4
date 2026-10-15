#include "q4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *m) {
    fprintf(stderr, "FAIL %s\n", m);
    return 1;
}

int main(int argc, char **argv) {
    q4_self_scope(argc, argv);
    q4_install_crashdump();
    if (argc > 1 && strcmp(argv[1], "--cpxbench") == 0)
        return q4_cpx_bench();
    if (argc > 1 && strcmp(argv[1], "--crash") == 0)
        abort();   /* crash-handler smoke test */
    const char *path = argc > 1 ? argv[1] : NULL;
    if (!path) {
        uint64_t b = 0;
        if (!q4_parse_gib("9GB", &b) || b != 9ull * Q4_GIB) return fail("parse 9GB");
        if (q4_parse_gib("0GB", &b)) return fail("parse 0");
        {
            float W[6] = {1, 2, 3, 4, 5, 6};
            float x[3] = {1, 1, 1};
            float y[2] = {0, 0};
            if (!q4_gemv(Q4_T_F32, (const uint8_t *)W, 2, 3, x, y, 1.0f))
                return fail("gemv f32");
            if (fabsf(y[0] - 6.f) > 1e-4f || fabsf(y[1] - 15.f) > 1e-4f)
                return fail("gemv values");
        }
        {
            /* One super-block, x = 1. IQ2_S/IQ3_S grids at index 0 are
             * constant bytes, so a zero codeword has a closed-form dot. */
            uint8_t iq2[82 * 2];
            memset(iq2, 0, sizeof(iq2));
            uint16_t one = 0x3c00;
            memcpy(iq2, &one, 2);
            memcpy(iq2 + 82, &one, 2);
            memset(iq2 + 82 + 74, 0x11, 8); /* both scale nibbles = 1 */
            float x[256], y[2];
            for (int i = 0; i < 256; i++) x[i] = 1.f;
            y[0] = y[1] = 0.f;
            if (!q4_gemv(Q4_T_IQ2_S, iq2, 2, 256, x, y, 1.f))
                return fail("iq2 gemv");
            /* d=1, grid byte 8, scale 0 → 0.125*8=1; scale nibble 1 → 0.375*8=3 */
            if (fabsf(y[0] - 256.f) > 1e-2f || fabsf(y[1] - 768.f) > 1e-2f)
                return fail("iq2 values");
            uint8_t iq3[110];
            memset(iq3, 0, sizeof(iq3));
            memcpy(iq3, &one, 2);
            y[0] = 0.f;
            if (!q4_gemv(Q4_T_IQ3_S, iq3, 1, 256, x, y, 1.f))
                return fail("iq3 gemv");
            if (fabsf(y[0] - 256.f) > 1e-2f) return fail("iq3 values");
            uint8_t iq4[18];
            memset(iq4, 0, sizeof(iq4));
            memcpy(iq4, &one, 2);
            float x32[32], y4 = 0.f;
            for (int i = 0; i < 32; i++) x32[i] = 1.f;
            if (!q4_gemv(Q4_T_IQ4_NL, iq4, 1, 32, x32, &y4, 1.f))
                return fail("iq4 gemv");
            if (fabsf(y4 - (-127.f * 32.f)) > 1e-2f) return fail("iq4 values");
            uint8_t q6[210];
            memset(q6, 0, sizeof(q6));
            memset(q6 + 192, 1, 16);
            memcpy(q6 + 208, &one, 2);
            y[0] = 0.f;
            if (!q4_gemv(Q4_T_Q6_K, q6, 1, 256, x, y, 1.f))
                return fail("q6 gemv");
            if (fabsf(y[0] - (-32.f * 256.f)) > 1e-2f) return fail("q6 values");
            if (q4_hip_init()) {
                /* Batched IQ GEMM must match the CPU dot, including a which[] gather. */
                {
                    const int cases[4][3] = {{Q4_T_IQ2_XS, 40, 512},
                                             {Q4_T_IQ2_S, 40, 512},
                                             {Q4_T_IQ3_S, 40, 512},
                                             {Q4_T_IQ4_NL, 40, 64}};
                    for (int ci = 0; ci < 4; ci++) {
                        uint32_t ty = (uint32_t)cases[ci][0];
                        int nrows = cases[ci][1], ncols = cases[ci][2], nbat = 40;
                        uint64_t rb = q4_row_bytes(ty, (uint64_t)ncols);
                        uint8_t *w = calloc((size_t)nrows * rb, 1);
                        float *xb = calloc((size_t)nbat * ncols, 4);
                        float *yref = calloc((size_t)nbat * nrows, 4);
                        float *ygot = calloc((size_t)nbat * nrows, 4);
                        int32_t which[40];
                        if (!w || !xb || !yref || !ygot || rb == 0) return fail("iq batch alloc");
                        for (int r = 0; r < nrows; r++) {
                            uint16_t d = (uint16_t)(0x3c00 + (r & 3));
                            memcpy(w + (size_t)r * rb, &d, 2);
                            w[(size_t)r * rb + 4] = (uint8_t)(r * 3);
                        }
                        for (int i = 0; i < nbat * ncols; i++)
                            xb[i] = ((i * 17) % 11) * 0.05f - 0.2f;
                        for (int t = 0; t < nbat; t++) which[t] = nbat - 1 - t;
                        for (int t = 0; t < nbat; t++)
                            if (!q4_gemv(ty, w, (uint64_t)nrows, (uint64_t)ncols,
                                         xb + (size_t)which[t] * ncols,
                                         yref + (size_t)t * nrows, 1.f))
                                return fail("iq batch cpu");
                        uint8_t *dw = q4_hip_malloc((size_t)nrows * rb);
                        float *dx = q4_hip_malloc((size_t)nbat * ncols * 4);
                        float *dy = q4_hip_malloc((size_t)nbat * nrows * 4);
                        int32_t *dwhich = q4_hip_malloc(sizeof(which));
                        if (!dw || !dx || !dy || !dwhich) return fail("iq batch gpu alloc");
                        if (!q4_hip_h2d(dw, w, (size_t)nrows * rb) ||
                            !q4_hip_h2d(dx, xb, (size_t)nbat * ncols * 4) ||
                            !q4_hip_h2d(dwhich, which, sizeof(which)) ||
                            !q4_hip_gemv_dd_n_which(ty, dw, (uint64_t)nrows,
                                                    (uint64_t)ncols, dx, dy, 1.f,
                                                    (uint32_t)nbat, dwhich) ||
                            !q4_hip_d2h(ygot, dy, (size_t)nbat * nrows * 4))
                            return fail("iq batch gpu");
                        for (int i = 0; i < nbat * nrows; i++) {
                            float e = fabsf(ygot[i] - yref[i]);
                            float lim = 2e-3f * (1.f + fabsf(yref[i]));
                            if (e > lim) {
                                fprintf(stderr, "iq batch ty %u i %d got %g ref %g\n",
                                        ty, i, ygot[i], yref[i]);
                                return fail("iq batch values");
                            }
                        }
                        q4_hip_free(dw);
                        q4_hip_free(dx);
                        q4_hip_free(dy);
                        q4_hip_free(dwhich);
                        free(w);
                        free(xb);
                        free(yref);
                        free(ygot);
                    }
                }
                /* Q8_0 which-gather: n_batch>=16 must not take the contiguous
                 * gemm (it ignores which). The reuse kernel dequants a row
                 * once and is what a resident head expert runs. */
                {
                    const uint32_t ty = Q4_T_Q8_0;
                    const int nrows = 32, ncols = 64, nbat = 24, nsrc = 40;
                    uint64_t rb = q4_row_bytes(ty, (uint64_t)ncols);
                    uint8_t *w = calloc((size_t)nrows * rb, 1);
                    float *xb = calloc((size_t)nsrc * ncols, 4);
                    float *yref = calloc((size_t)nbat * nrows, 4);
                    float *ygot = calloc((size_t)nbat * nrows, 4);
                    float *yre2 = calloc((size_t)nbat * nrows, 4);
                    int32_t which[24];
                    if (!w || !xb || !yref || !ygot || !yre2 || rb == 0)
                        return fail("q8 reuse alloc");
                    for (int r = 0; r < nrows; r++) {
                        uint32_t nb = (uint32_t)ncols / 32u;
                        for (uint32_t bi = 0; bi < nb; bi++) {
                            uint8_t *blk = w + (size_t)r * rb + (size_t)bi * 34u;
                            uint16_t d = 0x3c00; /* scale 1 */
                            memcpy(blk, &d, 2);
                            for (int j = 0; j < 32; j++)
                                blk[2 + j] = (uint8_t)((int8_t)((r + (int)bi + j) % 7 - 3));
                        }
                    }
                    for (int i = 0; i < nsrc * ncols; i++)
                        xb[i] = ((i * 17) % 11) * 0.05f - 0.2f;
                    for (int t = 0; t < nbat; t++) which[t] = nsrc - 1 - t;
                    for (int t = 0; t < nbat; t++)
                        if (!q4_gemv(ty, w, (uint64_t)nrows, (uint64_t)ncols,
                                     xb + (size_t)which[t] * ncols,
                                     yref + (size_t)t * nrows, 1.f))
                            return fail("q8 reuse cpu");
                    uint8_t *dw = q4_hip_malloc((size_t)nrows * rb);
                    float *dx = q4_hip_malloc((size_t)nsrc * ncols * 4);
                    float *dy = q4_hip_malloc((size_t)nbat * nrows * 4);
                    int32_t *dwhich = q4_hip_malloc(sizeof(which));
                    if (!dw || !dx || !dy || !dwhich) return fail("q8 reuse gpu alloc");
                    if (!q4_hip_h2d(dw, w, (size_t)nrows * rb) ||
                        !q4_hip_h2d(dx, xb, (size_t)nsrc * ncols * 4) ||
                        !q4_hip_h2d(dwhich, which, sizeof(which)) ||
                        !q4_hip_gemv_dd_n_which(ty, dw, (uint64_t)nrows,
                                                (uint64_t)ncols, dx, dy, 1.f,
                                                (uint32_t)nbat, dwhich) ||
                        !q4_hip_d2h(ygot, dy, (size_t)nbat * nrows * 4))
                        return fail("q8 which gemv");
                    if (!q4_hip_q8_reuse(dw, (uint64_t)nrows, (uint64_t)ncols,
                                         dx, dwhich, dy, 1.f, (uint32_t)nbat) ||
                        !q4_hip_d2h(yre2, dy, (size_t)nbat * nrows * 4))
                        return fail("q8 reuse gpu");
                    for (int i = 0; i < nbat * nrows; i++) {
                        float lim = 2e-3f * (1.f + fabsf(yref[i]));
                        if (fabsf(ygot[i] - yref[i]) > lim ||
                            fabsf(yre2[i] - yref[i]) > lim) {
                            fprintf(stderr,
                                    "q8 reuse i %d gemv %g reuse %g ref %g\n",
                                    i, ygot[i], yre2[i], yref[i]);
                            return fail("q8 reuse values");
                        }
                    }
                    q4_hip_free(dw);
                    q4_hip_free(dx);
                    q4_hip_free(dy);
                    q4_hip_free(dwhich);
                    free(w);
                    free(xb);
                    free(yref);
                    free(ygot);
                    free(yre2);
                }
                /* MMQ (vendored kernels): expert-id-indexed
                 * weights + ragged bounds + gathered q8_1 activations vs a
                 * dequantized CPU reference.  Shapes are the GSQ-RCO
                 * layer's: gate+up [1280,2560], down [2560,640]. */
                {
                    static const struct { uint32_t ty; uint64_t wr, wc; } mmc[] = {
                        {Q4_T_IQ3_XXS, 1280, 2560}, {Q4_T_IQ3_S, 1280, 2560},
                        {Q4_T_IQ2_S, 1280, 2560},   {Q4_T_IQ4_XS, 1280, 2560},
                        {Q4_T_IQ4_NL, 2560, 640},   {Q4_T_Q2_0, 2560, 640},
                    };
                    const uint32_t n_exp = 8, rows = 300, nsrc = 320;
                    /* ragged: one empty expert, one with a single row. */
                    const uint32_t per[8] = {100, 0, 1, 40, 60, 30, 50, 19};
                    int32_t bounds[9] = {0};
                    int32_t ids[300];
                    uint32_t maxr = 0;
                    for (uint32_t e = 0; e < n_exp; e++) {
                        bounds[e + 1] = bounds[e] + (int32_t)per[e];
                        if (per[e] > maxr) maxr = per[e];
                    }
                    for (uint32_t r = 0; r < rows; r++) {
                        uint32_t e = 0;
                        while (e + 1 < n_exp && r >= (uint32_t)bounds[e + 1]) e++;
                        ids[r] = (int32_t)((r * 37 + e * 11) % nsrc);
                    }
                    int32_t *d_ids = q4_hip_malloc(sizeof(ids));
                    int32_t *d_bnd = q4_hip_malloc(sizeof(bounds));
                    if (!d_ids || !d_bnd ||
                        !q4_hip_h2d(d_ids, ids, sizeof(ids)) ||
                        !q4_hip_h2d(d_bnd, bounds, sizeof(bounds)))
                        return fail("mmq meta");
                    for (uint32_t ci = 0; ci < 6; ci++) {
                        uint32_t ty = mmc[ci].ty;
                        /* blocks/row: the i-quants here are QK_K=256,
                         * IQ4_NL 32, Q2_0 64; every block leads fp16 d. */
                        uint64_t qk = ty == Q4_T_IQ4_NL ? 32 :
                                      ty == Q4_T_Q2_0 ? 64 : 256;
                        uint64_t wr = mmc[ci].wr, wc = mmc[ci].wc;
                        uint64_t rb = q4_row_bytes(ty, wc);
                        uint64_t eb = wr * rb;
                        uint8_t *w = malloc((size_t)(n_exp * eb));
                        float *x = malloc((size_t)nsrc * wc * 4);
                        float *got = malloc((size_t)rows * wr * 4);
                        if (!w || !x || !got || !rb) return fail("mmq alloc");
                        uint64_t rng = 88172645463325252ull + ci;
                        for (uint64_t i = 0; i < n_exp * eb; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            w[i] = (uint8_t)rng;
                        }
                        /* every block leads with fp16 d: keep it sane */
                        uint64_t nb = wc / qk, bs = rb / nb;
                        const uint16_t ds[3] = {0x2a66, 0x2e66, 0x30a3};
                        for (uint64_t e = 0; e < n_exp; e++)
                            for (uint64_t r = 0; r < wr; r++)
                                for (uint64_t b = 0; b < nb; b++)
                                    memcpy(w + e * eb + r * rb + b * bs,
                                           &ds[(e + r + b) % 3], 2);
                        for (uint64_t i = 0; i < (uint64_t)nsrc * wc; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            x[i] = ((int32_t)(rng % 41) - 20) * 0.025f;
                        }
                        uint8_t *dw = q4_hip_malloc((size_t)(n_exp * eb));
                        float *dx = q4_hip_malloc((size_t)nsrc * wc * 4);
                        void *dxq = q4_hip_malloc(q4_mmq_q8_bytes(rows, wc));
                        float *dst = q4_hip_malloc((size_t)rows * wr * 4);
                        if (!dw || !dx || !dxq || !dst ||
                            !q4_hip_h2d(dw, w, (size_t)(n_exp * eb)) ||
                            !q4_hip_h2d(dx, x, (size_t)nsrc * wc * 4) ||
                            !q4_mmq_quantize(dx, d_ids, dxq, ty, wc, rows) ||
                            !q4_mmq_mul_id(dw, ty, wr, wc, eb, n_exp, dxq,
                                           d_bnd, maxr, rows, dst, wr) ||
                            !q4_hip_d2h(got, dst, (size_t)rows * wr * 4))
                            return fail("mmq gpu run");
                        double num = 0.0, den = 0.0;
                        float *row = malloc((size_t)wc * 4);
                        if (!row) return fail("mmq ref alloc");
                        for (uint32_t r = 0; r < rows; r++) {
                            uint32_t e = 0;
                            while (e + 1 < n_exp &&
                                   r >= (uint32_t)bounds[e + 1]) e++;
                            const uint8_t *we = w + e * eb;
                            const float *xr = x + (size_t)ids[r] * wc;
                            for (uint64_t j = 0; j < wr; j++) {
                                if (!q4_dequant_row(ty, we + j * rb, wc, row))
                                    return fail("mmq ref dequant");
                                double acc = 0.0;
                                for (uint64_t k = 0; k < wc; k++)
                                    acc += (double)row[k] * xr[k];
                                double d = acc - got[(size_t)r * wr + j];
                                num += d * d;
                                den += acc * acc;
                            }
                        }
                        double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
                        printf("mmq type %u (%s): rel L2 %.5f\n", ty,
                               q4_type_name(ty), rel);
                        free(row);
                        q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dxq);
                        q4_hip_free(dst);
                        free(w); free(x); free(got);
                        if (rel >= 1e-2) return fail("mmq accuracy");
                    }
                    /* q4_xg (fp16 WMMA expert GEMMs) on the same ragged
                     * routing: weights addressed by slot (expert e at
                     * slot n_exp-1-e), gate/up fused with swiglu into
                     * fp16 h, down from fp16 h. */
                    for (uint32_t ci = 0; ci < 6; ci++) {
                        uint32_t ty = mmc[ci].ty;
                        const int gu = ci < 4;
                        uint64_t qk = ty == Q4_T_IQ4_NL ? 32 :
                                      ty == Q4_T_Q2_0 ? 64 : 256;
                        uint64_t wr = mmc[ci].wr, wc = mmc[ci].wc;
                        uint64_t rb = q4_row_bytes(ty, wc), eb = wr * rb;
                        uint64_t nb = wc / qk, bs = rb / nb;
                        uint64_t no = gu ? wr / 2 : wr;
                        uint8_t *w = malloc((size_t)(n_exp * eb));
                        float *x = malloc((size_t)nsrc * wc * 4);
                        uint16_t *x16 = malloc((size_t)nsrc * wc * 2);
                        float *got = malloc((size_t)rows * no * 4);
                        uint16_t *got16 = malloc((size_t)rows * no * 2);
                        float *row = malloc((size_t)wc * 4);
                        float *row2 = malloc((size_t)wc * 4);
                        if (!w || !x || !x16 || !got || !got16 || !row || !row2)
                            return fail("xg alloc");
                        uint64_t rng = 0x9E3779B97F4A7C15ull + ci;
                        for (uint64_t i = 0; i < n_exp * eb; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            w[i] = (uint8_t)rng;
                        }
                        /* model-like scales (weights ~1e-2; h stays
                         * well inside fp16), incl. sub-block scales in
                         * fp16's subnormal range */
                        const uint16_t ds[3] = {0x0466, 0x1266, 0x18a3};
                        for (uint64_t e = 0; e < n_exp; e++)
                            for (uint64_t r = 0; r < wr; r++)
                                for (uint64_t b = 0; b < nb; b++)
                                    memcpy(w + e * eb + r * rb + b * bs,
                                           &ds[(e + r + b) % 3], 2);
                        for (uint64_t i = 0; i < (uint64_t)nsrc * wc; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            x[i] = ((int32_t)(rng % 41) - 20) * 0.025f;
                        }
                        int32_t slot[8], tl[4 * 16];
                        uint32_t ek8[8], eb8[8];
                        for (uint32_t e = 0; e < n_exp; e++) {
                            slot[e] = (int32_t)(n_exp - 1 - e);
                            ek8[e] = per[e];
                            eb8[e] = (uint32_t)bounds[e];
                        }
                        uint32_t nt = q4_xg_tiles(ek8, eb8, slot, n_exp, tl, 16);
                        uint8_t *dw = q4_hip_malloc((size_t)(n_exp * eb));
                        float *dx = q4_hip_malloc((size_t)nsrc * wc * 4);
                        void *dx16 = q4_hip_malloc((size_t)nsrc * wc * 2);
                        void *dout = q4_hip_malloc((size_t)rows * no * 4);
                        int32_t *dtl = q4_hip_malloc(sizeof(tl));
                        int ok = nt && dw && dx && dx16 && dout && dtl;
                        for (uint32_t e = 0; ok && e < n_exp; e++)
                            ok = q4_hip_h2d(dw + (size_t)slot[e] * eb,
                                            w + e * eb, (size_t)eb);
                        ok = ok && q4_hip_h2d(dx, x, (size_t)nsrc * wc * 4) &&
                             q4_hip_h2d(dtl, tl, (size_t)nt * 16) &&
                             q4_xg_cvt(dx, dx16, (uint64_t)nsrc * wc) &&
                             q4_hip_d2h(x16, dx16, (size_t)nsrc * wc * 2);
                        /* down reads routed rows directly: row r = x16[r] */
                        ok = ok &&
                             (gu ? q4_xg_gu(ty, dw, eb, 0, rb * (wr / 2),
                                            (uint32_t)wc, (uint32_t)(wr / 2),
                                            dtl, nt, dx16, d_ids, dout) &&
                                       q4_hip_d2h(got16, dout, (size_t)rows * no * 2)
                                 : q4_xg_dn(ty, dw, eb, 0, (uint32_t)wc,
                                            (uint32_t)wr, dtl, nt, dx16,
                                            dout) &&
                                       q4_hip_d2h(got, dout, (size_t)rows * no * 4));
                        if (!ok) return fail("xg gpu run");
                        double num = 0.0, den = 0.0;
                        for (uint32_t r = 0; r < rows; r++) {
                            uint32_t e = 0;
                            while (e + 1 < n_exp &&
                                   r >= (uint32_t)bounds[e + 1]) e++;
                            const uint8_t *we = w + e * eb;
                            const uint16_t *xr =
                                x16 + (size_t)(gu ? ids[r] : (int32_t)r) * wc;
                            for (uint64_t j = 0; j < no; j++) {
                                double a = 0.0, b2 = 0.0, ref, gv;
                                if (!q4_dequant_row(ty, we + j * rb, wc, row))
                                    return fail("xg ref dequant");
                                for (uint64_t k = 0; k < wc; k++)
                                    a += (double)row[k] * q4_f16_to_f32(xr[k]);
                                if (gu) {
                                    if (!q4_dequant_row(ty, we + (j + no) * rb,
                                                        wc, row2))
                                        return fail("xg ref dequant");
                                    for (uint64_t k = 0; k < wc; k++)
                                        b2 += (double)row2[k] *
                                              q4_f16_to_f32(xr[k]);
                                    ref = a / (1.0 + exp(-a)) * b2;
                                    gv = q4_f16_to_f32(got16[(size_t)r * no + j]);
                                } else {
                                    ref = a;
                                    gv = got[(size_t)r * no + j];
                                }
                                num += (ref - gv) * (ref - gv);
                                den += ref * ref;
                            }
                        }
                        double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
                        printf("xg %s type %u (%s): rel L2 %.5f\n",
                               gu ? "gate/up" : "down", ty, q4_type_name(ty),
                               rel);
                        q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dx16);
                        q4_hip_free(dout); q4_hip_free(dtl);
                        free(w); free(x); free(x16); free(got); free(got16);
                        free(row); free(row2);
                        if (!(rel < 5e-3)) return fail("xg accuracy");
                    }
                    q4_hip_free(d_ids);
                    q4_hip_free(d_bnd);
                }
                /* q4_dg (dense prompt GEMM, fp16 / bf16 weights, fp32 x
                 * staged to 16-bit in the kernel) vs a CPU reference on
                 * ragged N and T. */
                for (int bf = 0; bf < 2; bf++) {
                    const uint32_t N = 200, K = 320, T = 150, ldy = 210;
                    uint16_t *w = malloc((size_t)N * K * 2);
                    float *x = malloc((size_t)T * K * 4), *got = malloc((size_t)T * ldy * 4);
                    if (!w || !x || !got) return fail("dg alloc");
                    uint64_t rng = 0x5DEECE66Dull + (uint64_t)bf;
                    for (size_t i = 0; i < (size_t)N * K; i++) {
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        const float v = ((int32_t)(rng % 201) - 100) * 0.0005f;
                        uint32_t u;
                        memcpy(&u, &v, 4);
                        w[i] = bf ? (uint16_t)(u >> 16) : 0;
                    }
                    if (!bf) {
                        float *wf = malloc((size_t)N * K * 4);
                        void *dwf = q4_hip_malloc((size_t)N * K * 4);
                        void *dwh = q4_hip_malloc((size_t)N * K * 2);
                        rng = 0x5DEECE66Dull;
                        for (size_t i = 0; i < (size_t)N * K; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            wf[i] = ((int32_t)(rng % 201) - 100) * 0.0005f;
                        }
                        if (!wf || !dwf || !dwh ||
                            !q4_hip_h2d(dwf, wf, (size_t)N * K * 4) ||
                            !q4_hip_f32_to_f16(dwh, dwf, (uint64_t)N * K) ||
                            !q4_hip_d2h(w, dwh, (size_t)N * K * 2))
                            return fail("dg f16 weights");
                        free(wf); q4_hip_free(dwf); q4_hip_free(dwh);
                    }
                    for (size_t i = 0; i < (size_t)T * K; i++) {
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        x[i] = ((int32_t)(rng % 401) - 200) * 0.01f;
                    }
                    void *dw = q4_hip_malloc((size_t)N * K * 2);
                    float *dx = q4_hip_malloc((size_t)T * K * 4);
                    float *dy = q4_hip_malloc((size_t)T * ldy * 4);
                    if (!dw || !dx || !dy || !q4_hip_h2d(dw, w, (size_t)N * K * 2) ||
                        !q4_hip_h2d(dx, x, (size_t)T * K * 4) ||
                        !q4_dg_gemm(dw, bf, N, K, dx, T, dy, ldy, 0.5f) ||
                        !q4_hip_d2h(got, dy, (size_t)T * ldy * 4))
                        return fail("dg gpu run");
                    double num = 0.0, den = 0.0;
                    for (uint32_t t = 0; t < T; t++)
                        for (uint32_t n = 0; n < N; n++) {
                            double acc = 0.0;
                            for (uint32_t k = 0; k < K; k++) {
                                const uint16_t h = w[(size_t)n * K + k];
                                const double wv = bf ? (double)q4_bf16_to_f32(h)
                                                     : (double)q4_f16_to_f32(h);
                                acc += wv * x[(size_t)t * K + k];
                            }
                            acc *= 0.5;
                            const double d = acc - got[(size_t)t * ldy + n];
                            num += d * d;
                            den += acc * acc;
                        }
                    const double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
                    printf("dg %s: rel L2 %.5f\n", bf ? "bf16" : "fp16", rel);
                    q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                    free(w); free(x); free(got);
                    if (!(rel < (bf ? 8e-3 : 2e-3))) return fail("dg accuracy");
                }
                /* hgemm: library dense path vs CPU fp64 reference per type,
                 * plus old-vs-new launch() agreement at n_batch 64. */
                {
                    static const struct { uint32_t ty; float tol; } hc[] = {
                        {Q4_T_Q6_K, 5e-3f},   {Q4_T_Q5_K, 5e-3f},
                        {Q4_T_Q4_K, 5e-3f},   {Q4_T_IQ4_XS, 5e-3f},
                        {Q4_T_IQ4_NL, 5e-3f}, {Q4_T_Q8_0, 5e-3f},
                        {Q4_T_BF16, 1e-2f},
                    };
                    const uint32_t wr = 256, wc = 2560, nbat = 200;
                    float *x = malloc((size_t)nbat * wc * 4);
                    float *got = malloc((size_t)nbat * wr * 4);
                    float *row = malloc((size_t)wc * 4);
                    if (!x || !got || !row) return fail("hgemm alloc");
                    uint64_t rng = 0x9e3779b97f4a7c15ull;
                    for (uint64_t i = 0; i < (uint64_t)nbat * wc; i++) {
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        x[i] = ((int32_t)(rng % 61) - 30) * 0.04f;
                    }
                    for (uint32_t ci = 0; ci < 7; ci++) {
                        uint32_t ty = hc[ci].ty;
                        uint64_t rb = q4_row_bytes(ty, wc);
                        uint8_t *w = malloc((size_t)wr * rb);
                        if (!w || !rb) return fail("hgemm walloc");
                        for (uint64_t i = 0; i < (uint64_t)wr * rb; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            w[i] = (uint8_t)rng;
                        }
                        /* keep every fp16 d/dmin field small; for BF16 the
                         * whole tensor is fp16-shaped, so write real ones. */
                        const uint16_t ds[3] = {0x2a66, 0x2e66, 0x30a3};
                        if (ty == Q4_T_BF16) {
                            static const uint16_t bv[4] = {0x3d8f, 0xbe0a,
                                                           0x3f0a, 0x3c23};
                            uint16_t *w16 = (uint16_t *)w;
                            for (uint64_t i = 0; i < (uint64_t)wr * wc; i++)
                                w16[i] = bv[i & 3];
                        } else {
                            uint64_t qk = ty == Q4_T_IQ4_NL ? 32 :
                                          ty == Q4_T_Q8_0 ? 32 : 256;
                            uint64_t nb = wc / qk, bs = rb / nb;
                            for (uint64_t r = 0; r < wr; r++)
                                for (uint64_t b = 0; b < nb; b++) {
                                    uint8_t *bp = w + r * rb + b * bs;
                                    if (ty == Q4_T_Q6_K)
                                        memcpy(bp + bs - 2,
                                               &ds[(r + b) % 3], 2);
                                    else
                                        memcpy(bp, &ds[(r + b) % 3], 2);
                                    if (ty == Q4_T_Q4_K || ty == Q4_T_Q5_K)
                                        memcpy(bp + 2, &ds[(r + b + 1) % 3],
                                               2);
                                }
                        }
                        uint8_t *dw = q4_hip_malloc((size_t)wr * rb);
                        float *dx = q4_hip_malloc((size_t)nbat * wc * 4);
                        float *dy = q4_hip_malloc((size_t)nbat * wr * 4);
                        if (!dw || !dx || !dy ||
                            !q4_hip_h2d(dw, w, (size_t)wr * rb) ||
                            !q4_hip_h2d(dx, x, (size_t)nbat * wc * 4) ||
                            !q4_hip_gemv_dd_n(ty, dw, wr, wc, dx, dy, 1.0f,
                                              nbat) ||
                            !q4_hip_d2h(got, dy, (size_t)nbat * wr * 4))
                            return fail("hgemm gpu run");
                        double num = 0.0, den = 0.0;
                        for (uint64_t t = 0; t < nbat; t++) {
                            const float *xr = x + t * wc;
                            for (uint64_t r = 0; r < wr; r++) {
                                if (!q4_dequant_row(ty, w + r * rb, wc, row))
                                    return fail("hgemm ref dequant");
                                double acc = 0.0;
                                for (uint64_t k = 0; k < wc; k++)
                                    acc += (double)row[k] * xr[k];
                                double d = acc - got[t * wr + r];
                                num += d * d;
                                den += acc * acc;
                            }
                        }
                        double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
                        printf("hgemm type %u (%s): rel L2 %.5f\n", ty,
                               q4_type_name(ty), rel);
                        free(w);
                        q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                        if (rel >= hc[ci].tol) return fail("hgemm accuracy");
                    }
                    /* old kernel vs library path at n_batch 64 (Q6_K). */
                    {
                        const uint32_t ty = Q4_T_Q6_K;
                        uint64_t rb = q4_row_bytes(ty, wc);
                        uint8_t *w = malloc((size_t)wr * rb);
                        float *a = malloc((size_t)64 * wr * 4);
                        float *b = malloc((size_t)64 * wr * 4);
                        if (!w || !a || !b) return fail("hgemm ab alloc");
                        for (uint64_t i = 0; i < (uint64_t)wr * rb; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            w[i] = (uint8_t)rng;
                        }
                        const uint16_t ds[3] = {0x2a66, 0x2e66, 0x30a3};
                        for (uint64_t r = 0; r < wr; r++)
                            for (uint64_t j = 0; j < wc / 256; j++)
                                memcpy(w + r * rb + j * 210 + 208,
                                       &ds[(r + j) % 3], 2);
                        uint8_t *dw = q4_hip_malloc((size_t)wr * rb);
                        float *dx = q4_hip_malloc((size_t)64 * wc * 4);
                        float *dy = q4_hip_malloc((size_t)64 * wr * 4);
                        if (!dw || !dx || !dy ||
                            !q4_hip_h2d(dw, w, (size_t)wr * rb) ||
                            !q4_hip_h2d(dx, x, (size_t)64 * wc * 4))
                            return fail("hgemm ab up");
                        setenv("Q4_HGEMM", "0", 1);
                        if (!q4_hip_gemv_dd_n(ty, dw, wr, wc, dx, dy, 1.0f,
                                              64) ||
                            !q4_hip_d2h(a, dy, (size_t)64 * wr * 4))
                            return fail("hgemm old path");
                        setenv("Q4_HGEMM", "1", 1);
                        if (!q4_hip_gemv_dd_n(ty, dw, wr, wc, dx, dy, 1.0f,
                                              64) ||
                            !q4_hip_d2h(b, dy, (size_t)64 * wr * 4))
                            return fail("hgemm new path");
                        double num = 0.0, den = 0.0;
                        for (uint64_t i = 0; i < (uint64_t)64 * wr; i++) {
                            double d = (double)a[i] - b[i];
                            num += d * d;
                            den += (double)a[i] * a[i];
                        }
                        double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
                        printf("hgemm old-vs-new (Q6_K): rel L2 %.5f\n", rel);
                        free(w); free(a); free(b);
                        q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                        if (rel >= 2e-2) return fail("hgemm ab accuracy");
                    }
                    free(x); free(got); free(row);
                }
                /* gdnrec: pipelined GDN scan vs gdn_scan_k — core, final S
                 * and final hist must match to fp-reorder error. */
                {
                    const uint32_t n_kh = 16, n_vh = 48, d = 128, ksz = 4;
                    const uint32_t ch = 2 * d * n_kh + d * n_vh;
                    const float eps = 1e-5f, qscale = 1.f / sqrtf(128.f);
                    uint64_t rng = 0x243f6a8885a308d3ull;
                    float *cw = malloc((size_t)ch * ksz * 4);
                    float *Av = malloc((size_t)n_vh * 4);
                    float *dt = malloc((size_t)n_vh * 4);
                    float *nw = malloc((size_t)d * 4);
                    if (!cw || !Av || !dt || !nw) return fail("gdn alloc");
                    for (uint64_t i = 0; i < (uint64_t)ch * ksz; i++) {
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        cw[i] = ((int32_t)(rng % 41) - 20) * 0.01f;
                    }
                    for (uint32_t i = 0; i < n_vh; i++) {
                        Av[i] = -0.02f - (i % 5) * 0.01f;
                        dt[i] = 0.05f * (int32_t)(i % 7) - 0.15f;
                    }
                    for (uint32_t i = 0; i < d; i++)
                        nw[i] = 0.5f + 0.1f * (i % 9);
                    static const uint32_t Ts[2] = {70, 300};
                    for (uint32_t ti = 0; ti < 2; ti++) {
                        const uint32_t T = Ts[ti];
                        float *hist0 = malloc(3 * (size_t)ch * 4);
                        float *S0 = malloc((size_t)n_vh * d * d * 4);
                        float *qkv = malloc((size_t)T * ch * 4);
                        float *beta = malloc((size_t)T * n_vh * 4);
                        float *alpha = malloc((size_t)T * n_vh * 4);
                        float *z = malloc((size_t)T * n_vh * d * 4);
                        if (!hist0 || !S0 || !qkv || !beta || !alpha || !z)
                            return fail("gdn talloc");
                        for (uint64_t i = 0; i < (size_t)T * ch; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            qkv[i] = ((int32_t)(rng % 81) - 40) * 0.05f;
                        }
                        for (uint64_t i = 0; i < 3 * (size_t)ch; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            hist0[i] = ((int32_t)(rng % 61) - 30) * 0.03f;
                        }
                        for (uint64_t i = 0; i < (size_t)n_vh * d * d; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            S0[i] = ((int32_t)(rng % 51) - 25) * 0.004f;
                        }
                        for (uint64_t i = 0; i < (size_t)T * n_vh; i++) {
                            beta[i] = ((int32_t)(i % 31) - 15) * 0.2f;
                            alpha[i] = -0.5f - 0.03f * (i % 11);
                        }
                        for (uint64_t i = 0; i < (size_t)T * n_vh * d; i++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            z[i] = ((int32_t)(rng % 61) - 30) * 0.1f;
                        }
                        /* two runs over identical private copies. */
                        float *hist1 = malloc(3 * (size_t)ch * 4);
                        float *hist2 = malloc(3 * (size_t)ch * 4);
                        float *S1 = malloc((size_t)n_vh * d * d * 4);
                        float *S2 = malloc((size_t)n_vh * d * d * 4);
                        float *c1 = malloc((size_t)T * n_vh * d * 4);
                        float *c2 = malloc((size_t)T * n_vh * d * 4);
                        if (!hist1 || !hist2 || !S1 || !S2 || !c1 || !c2)
                            return fail("gdn oalloc");
                        memcpy(hist1, hist0, 3 * (size_t)ch * 4);
                        memcpy(hist2, hist0, 3 * (size_t)ch * 4);
                        memcpy(S1, S0, (size_t)n_vh * d * d * 4);
                        memcpy(S2, S0, (size_t)n_vh * d * d * 4);
                        float *dh = q4_hip_malloc(3 * (size_t)ch * 4);
                        float *dS = q4_hip_malloc((size_t)n_vh * d * d * 4);
                        float *dq = q4_hip_malloc((size_t)T * ch * 4);
                        float *dw = q4_hip_malloc((size_t)ch * ksz * 4);
                        float *db = q4_hip_malloc((size_t)T * n_vh * 4);
                        float *da = q4_hip_malloc((size_t)T * n_vh * 4);
                        float *dA = q4_hip_malloc((size_t)n_vh * 4);
                        float *dd = q4_hip_malloc((size_t)n_vh * 4);
                        float *dn = q4_hip_malloc((size_t)d * 4);
                        float *dz = q4_hip_malloc((size_t)T * n_vh * d * 4);
                        float *dc = q4_hip_malloc((size_t)T * n_vh * d * 4);
                        if (!dh || !dS || !dq || !dw || !db || !da || !dA ||
                            !dd || !dn || !dz || !dc ||
                            !q4_hip_h2d(dq, qkv, (size_t)T * ch * 4) ||
                            !q4_hip_h2d(dw, cw, (size_t)ch * ksz * 4) ||
                            !q4_hip_h2d(db, beta, (size_t)T * n_vh * 4) ||
                            !q4_hip_h2d(da, alpha, (size_t)T * n_vh * 4) ||
                            !q4_hip_h2d(dA, Av, (size_t)n_vh * 4) ||
                            !q4_hip_h2d(dd, dt, (size_t)n_vh * 4) ||
                            !q4_hip_h2d(dn, nw, (size_t)d * 4) ||
                            !q4_hip_h2d(dz, z, (size_t)T * n_vh * d * 4))
                            return fail("gdn up");
                        setenv("Q4_GDN_REC", "0", 1);
                        if (!q4_hip_h2d(dh, hist1, 3 * (size_t)ch * 4) ||
                            !q4_hip_h2d(dS, S1, (size_t)n_vh * d * d * 4) ||
                            !q4_hip_gdn_scan(dh, dq, dw, ch, ksz, dS, db, da,
                                             dA, dd, dn, dz, dc, T, n_vh,
                                             n_kh, d, eps, qscale) ||
                            !q4_hip_d2h(c1, dc, (size_t)T * n_vh * d * 4) ||
                            !q4_hip_d2h(S1, dS, (size_t)n_vh * d * d * 4) ||
                            !q4_hip_d2h(hist1, dh, 3 * (size_t)ch * 4))
                            return fail("gdn old");
                        setenv("Q4_GDN_REC", "1", 1);
                        if (!q4_hip_h2d(dh, hist2, 3 * (size_t)ch * 4) ||
                            !q4_hip_h2d(dS, S2, (size_t)n_vh * d * d * 4) ||
                            !q4_hip_gdn_scan(dh, dq, dw, ch, ksz, dS, db, da,
                                             dA, dd, dn, dz, dc, T, n_vh,
                                             n_kh, d, eps, qscale) ||
                            !q4_hip_d2h(c2, dc, (size_t)T * n_vh * d * 4) ||
                            !q4_hip_d2h(S2, dS, (size_t)n_vh * d * d * 4) ||
                            !q4_hip_d2h(hist2, dh, 3 * (size_t)ch * 4))
                            return fail("gdn new");
                        double cn = 0, cd = 0, sn = 0, sd = 0, hn = 0, hd2 = 0;
                        for (uint64_t i = 0; i < (size_t)T * n_vh * d; i++) {
                            double df = (double)c1[i] - c2[i];
                            cn += df * df; cd += (double)c1[i] * c1[i];
                        }
                        for (uint64_t i = 0; i < (size_t)n_vh * d * d; i++) {
                            double df = (double)S1[i] - S2[i];
                            sn += df * df; sd += (double)S1[i] * S1[i];
                        }
                        for (uint64_t i = 0; i < 3 * (size_t)ch; i++) {
                            double df = (double)hist1[i] - hist2[i];
                            hn += df * df; hd2 += (double)hist1[i] * hist1[i];
                        }
                        double cr = cd ? sqrt(cn / cd) : 0,
                               sr = sd ? sqrt(sn / sd) : 0,
                               hr = hd2 ? sqrt(hn / hd2) : 0;
                        printf("gdnrec T=%u: core %.2e  S %.2e  hist %.2e\n",
                               T, cr, sr, hr);
                        if (cr >= 1e-4 || sr >= 1e-4 || hr >= 1e-4)
                            return fail("gdnrec accuracy");
                        free(hist0); free(S0); free(qkv); free(beta);
                        free(alpha); free(z); free(hist1); free(hist2);
                        free(S1); free(S2); free(c1); free(c2);
                        q4_hip_free(dh); q4_hip_free(dS); q4_hip_free(dq);
                        q4_hip_free(dw); q4_hip_free(db); q4_hip_free(da);
                        q4_hip_free(dA); q4_hip_free(dd); q4_hip_free(dn);
                        q4_hip_free(dz); q4_hip_free(dc);
                    }
                    free(cw); free(Av); free(dt); free(nw);
                }
                /* qsa wmma prefill: flash kernel vs gqa_prefill_k, dense and
                 * indexed, q8_0 KV (kv_f16=2). */
                {
                    const uint32_t n_head = 24, n_kvh = 2, hd = 256;
                    const uint32_t gqa = n_head / n_kvh;
                    const float scale = 1.f / sqrtf((float)hd);
                    const uint64_t q8b = hd / 32 * 34;
                    uint64_t rng = 0x9e3779b97f4a7c15ull;
                    /* (a) dense: kv_base 100, n_tok 300. */
                    {
                        const uint32_t kv_base = 100, n_tok = 300,
                                       npos = kv_base + n_tok;
                        float *q = malloc((size_t)n_tok * n_head * hd * 4);
                        float *kf = malloc((size_t)npos * n_kvh * hd * 4);
                        float *vf = malloc((size_t)npos * n_kvh * hd * 4);
                        float *gt = malloc((size_t)n_tok * n_head * hd * 4);
                        float *a1 = malloc((size_t)n_tok * n_head * hd * 4);
                        float *a2 = malloc((size_t)n_tok * n_head * hd * 4);
                        if (!q || !kf || !vf || !gt || !a1 || !a2)
                            return fail("qsaw alloc");
                        for (uint64_t e = 0; e < (uint64_t)n_tok * n_head * hd;
                             e++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            q[e] = ((int32_t)(rng % 61) - 30) * 0.04f;
                            gt[e] = ((int32_t)(rng % 41) - 20) * 0.1f;
                        }
                        for (uint64_t e = 0; e < (uint64_t)npos * n_kvh * hd;
                             e++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            kf[e] = ((int32_t)(rng % 51) - 25) * 0.04f;
                            vf[e] = ((int32_t)(rng % 51) - 25) * 0.04f;
                        }
                        float *dq = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                        float *dg = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                        float *do_ = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                        void *dk8 = q4_hip_malloc(npos * n_kvh * q8b);
                        void *dv8 = q4_hip_malloc(npos * n_kvh * q8b);
                        float *dk = q4_hip_malloc((size_t)npos * n_kvh * hd * 4);
                        float *dv = q4_hip_malloc((size_t)npos * n_kvh * hd * 4);
                        if (!dq || !dg || !do_ || !dk8 || !dv8 || !dk || !dv ||
                            !q4_hip_h2d(dq, q, (size_t)n_tok * n_head * hd * 4) ||
                            !q4_hip_h2d(dg, gt, (size_t)n_tok * n_head * hd * 4) ||
                            !q4_hip_h2d(dk, kf, (size_t)npos * n_kvh * hd * 4) ||
                            !q4_hip_h2d(dv, vf, (size_t)npos * n_kvh * hd * 4) ||
                            !q4_hip_f32_to_q8(dk8, dk, npos * n_kvh * hd) ||
                            !q4_hip_f32_to_q8(dv8, dv, npos * n_kvh * hd))
                            return fail("qsaw up");
                        setenv("Q4_QSA_WMMA", "0", 1);
                        if (!q4_hip_qsa_decode_n(dq, dk8, dv8, dg, do_, n_head,
                                                 n_kvh, hd, n_tok, kv_base,
                                                 npos, scale, 2) ||
                            !q4_hip_d2h(a1, do_, (size_t)n_tok * n_head * hd * 4))
                            return fail("qsaw dense old");
                        setenv("Q4_QSA_WMMA", "1", 1);
                        if (!q4_hip_qsa_decode_n(dq, dk8, dv8, dg, do_, n_head,
                                                 n_kvh, hd, n_tok, kv_base,
                                                 npos, scale, 2) ||
                            !q4_hip_d2h(a2, do_, (size_t)n_tok * n_head * hd * 4))
                            return fail("qsaw dense new");
                        double num = 0, den = 0;
                        for (uint64_t e = 0; e < (uint64_t)n_tok * n_head * hd;
                             e++) {
                            double df = (double)a1[e] - a2[e];
                            num += df * df;
                            den += (double)a1[e] * a1[e];
                        }
                        double rel = den ? sqrt(num / den) : 0;
                        printf("qsa wmma dense: rel L2 %.5f\n", rel);
                        if (rel >= 3e-3) return fail("qsaw dense");
                        /* (b) indexed: width 2051, n_tok 200, some -1/dups. */
                        const uint32_t width = 2051, n_tok2 = 200,
                                       max_kv = 4096;
                        float *q2 = realloc(q, (size_t)n_tok2 * n_head * hd * 4);
                        float *gt2 = realloc(gt, (size_t)n_tok2 * n_head * hd * 4);
                        float *b1 = realloc(a1, (size_t)n_tok2 * n_head * hd * 4);
                        float *b2 = realloc(a2, (size_t)n_tok2 * n_head * hd * 4);
                        q = q2; gt = gt2; a1 = b1; a2 = b2;
                        int32_t *idx = malloc((size_t)n_tok2 * width * 4);
                        float *kf2 = realloc(kf, (size_t)max_kv * n_kvh * hd * 4);
                        float *vf2 = realloc(vf, (size_t)max_kv * n_kvh * hd * 4);
                        kf = kf2; vf = vf2;
                        if (!q || !gt || !a1 || !a2 || !idx || !kf || !vf)
                            return fail("qsaw realloc");
                        for (uint64_t e = 0; e < (uint64_t)n_tok2 * n_head * hd;
                             e++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            q[e] = ((int32_t)(rng % 61) - 30) * 0.04f;
                            gt[e] = ((int32_t)(rng % 41) - 20) * 0.1f;
                        }
                        for (uint64_t e = 0; e < (uint64_t)max_kv * n_kvh * hd;
                             e++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            kf[e] = ((int32_t)(rng % 51) - 25) * 0.04f;
                            vf[e] = ((int32_t)(rng % 51) - 25) * 0.04f;
                        }
                        for (uint64_t e = 0; e < (uint64_t)n_tok2 * width; e++) {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                            idx[e] = (rng % 11) == 0 ? -1
                                                   : (int32_t)(rng % max_kv);
                        }
                        int32_t *didx = q4_hip_malloc((size_t)n_tok2 * width * 4);
                        float *dk2 = q4_hip_malloc((size_t)max_kv * n_kvh * hd * 4);
                        float *dv2 = q4_hip_malloc((size_t)max_kv * n_kvh * hd * 4);
                        void *dk82 = q4_hip_malloc((size_t)max_kv * n_kvh * q8b);
                        void *dv82 = q4_hip_malloc((size_t)max_kv * n_kvh * q8b);
                        if (!didx || !dk2 || !dv2 || !dk82 || !dv82 ||
                            !q4_hip_h2d(dq, q, (size_t)n_tok2 * n_head * hd * 4) ||
                            !q4_hip_h2d(dg, gt, (size_t)n_tok2 * n_head * hd * 4) ||
                            !q4_hip_h2d(didx, idx, (size_t)n_tok2 * width * 4) ||
                            !q4_hip_h2d(dk2, kf, (size_t)max_kv * n_kvh * hd * 4) ||
                            !q4_hip_h2d(dv2, vf, (size_t)max_kv * n_kvh * hd * 4) ||
                            !q4_hip_f32_to_q8(dk82, dk2, (size_t)max_kv * n_kvh * hd) ||
                            !q4_hip_f32_to_q8(dv82, dv2, (size_t)max_kv * n_kvh * hd))
                            return fail("qsaw idx up");
                        setenv("Q4_QSA_WMMA", "0", 1);
                        if (!q4_hip_qsa_decode_idx(dq, dk82, dv82, dg, do_,
                                                   didx, width, n_head, n_kvh,
                                                   hd, n_tok2, max_kv, scale,
                                                   2) ||
                            !q4_hip_d2h(a1, do_,
                                        (size_t)n_tok2 * n_head * hd * 4))
                            return fail("qsaw idx old");
                        setenv("Q4_QSA_WMMA", "1", 1);
                        setenv("Q4_QSA_UNION", "0", 1);
                        if (!q4_hip_qsa_decode_idx(dq, dk82, dv82, dg, do_,
                                                   didx, width, n_head, n_kvh,
                                                   hd, n_tok2, max_kv, scale,
                                                   2) ||
                            !q4_hip_d2h(a2, do_,
                                        (size_t)n_tok2 * n_head * hd * 4))
                            return fail("qsaw idx new");
                        unsetenv("Q4_QSA_UNION");
                        num = 0; den = 0;
                        for (uint64_t e = 0; e < (uint64_t)n_tok2 * n_head * hd;
                             e++) {
                            double df = (double)a1[e] - a2[e];
                            num += df * df;
                            den += (double)a1[e] * a1[e];
                        }
                        rel = den ? sqrt(num / den) : 0;
                        printf("qsa wmma indexed: rel L2 %.5f\n", rel);
                        if (!(rel < 3e-3)) return fail("qsaw indexed");
                        /* (c) union of T consecutive tokens' selections
                         * (gqa_union_wmma_k): duplicate-free per token like
                         * the topk output; neighbours share most positions,
                         * an odd n_tok leaves a short last group. */
                        for (uint32_t t = 0; t < n_tok2; t++) {
                            const uint32_t off = (t / 3u) * 97u + (t % 3u) * 5u;
                            for (uint32_t w = 0; w < width; w++) {
                                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                                const int32_t pp =
                                    (int32_t)((w * 3u + off) % max_kv);
                                /* p % 4 == 3 is never selected, so every
                                 * union block has a cell nobody picked */
                                idx[(size_t)t * width + w] =
                                    (rng % 13) == 0 || (pp & 3) == 3 ? -1 : pp;
                            }
                        }
                        if (!q4_hip_h2d(didx, idx, (size_t)n_tok2 * width * 4))
                            return fail("qsaw union up");
                        const uint32_t n_tok3 = n_tok2 - 1;
                        /* positions nobody selects get NaN q8 scales: the
                         * union loads whole 4-position blocks (the block of
                         * the prompt's last cell runs past what was
                         * written) and must not touch them */
                        {
                            const size_t qb = (size_t)max_kv * n_kvh * q8b;
                            uint8_t *hk8 = malloc(qb), *hv8 = malloc(qb);
                            uint8_t *used = calloc(max_kv, 1);
                            if (!hk8 || !hv8 || !used ||
                                !q4_hip_d2h(hk8, dk82, qb) ||
                                !q4_hip_d2h(hv8, dv82, qb))
                                return fail("qsaw union poison");
                            for (uint64_t e = 0; e < (uint64_t)n_tok3 * width; e++)
                                if (idx[e] >= 0) used[idx[e]] = 1;
                            const uint16_t nanh = 0x7e00;
                            for (uint32_t p = 0; p < max_kv; p++) {
                                if (used[p]) continue;
                                for (uint32_t h = 0; h < n_kvh; h++)
                                    for (uint32_t b = 0; b < hd / 32; b++) {
                                        size_t o = ((size_t)p * n_kvh + h) * q8b + b * 34;
                                        memcpy(hk8 + o, &nanh, 2);
                                        memcpy(hv8 + o, &nanh, 2);
                                    }
                            }
                            if (!q4_hip_h2d(dk82, hk8, qb) || !q4_hip_h2d(dv82, hv8, qb))
                                return fail("qsaw union poison up");
                            free(hk8); free(hv8); free(used);
                        }
                        setenv("Q4_QSA_WMMA", "0", 1);
                        if (!q4_hip_qsa_decode_idx(dq, dk82, dv82, dg, do_,
                                                   didx, width, n_head, n_kvh,
                                                   hd, n_tok3, max_kv, scale,
                                                   2) ||
                            !q4_hip_d2h(a1, do_,
                                        (size_t)n_tok3 * n_head * hd * 4))
                            return fail("qsaw union ref");
                        unsetenv("Q4_QSA_WMMA");
                        for (int ut = 2; ut <= 4; ut += 2) {
                            setenv("Q4_QSA_UNION", ut == 2 ? "2" : "4", 1);
                            if (!q4_hip_qsa_decode_idx(dq, dk82, dv82, dg, do_,
                                                       didx, width, n_head,
                                                       n_kvh, hd, n_tok3,
                                                       max_kv, scale, 2) ||
                                !q4_hip_d2h(a2, do_,
                                            (size_t)n_tok3 * n_head * hd * 4))
                                return fail("qsaw union");
                            num = 0; den = 0;
                            for (uint64_t e = 0;
                                 e < (uint64_t)n_tok3 * n_head * hd; e++) {
                                double df = (double)a1[e] - a2[e];
                                num += df * df;
                                den += (double)a1[e] * a1[e];
                            }
                            rel = den ? sqrt(num / den) : 0;
                            printf("qsa union T=%d: rel L2 %.5f\n", ut, rel);
                            if (!(rel < 3e-3)) return fail("qsaw union");
                        }
                        unsetenv("Q4_QSA_UNION");
                        free(q); free(kf); free(vf); free(gt); free(a1);
                        free(a2); free(idx);
                        q4_hip_free(dq); q4_hip_free(dg); q4_hip_free(do_);
                        q4_hip_free(dk8); q4_hip_free(dv8); q4_hip_free(dk);
                        q4_hip_free(dv); q4_hip_free(didx); q4_hip_free(dk2);
                        q4_hip_free(dv2); q4_hip_free(dk82); q4_hip_free(dv82);
                    }
                }
                /* QSA: online softmax must match a dense reference for f32 and f16 KV. */
                {
                    const int n_head = 2, n_kvh = 1, hd = 32, n_tok = 2, npos = 4;
                    const int kv_base = 0;
                    float scale = 1.f / sqrtf((float)hd);
                    float *q = calloc((size_t)n_tok * n_head * hd, 4);
                    float *k = calloc((size_t)npos * n_kvh * hd, 4);
                    float *v = calloc((size_t)npos * n_kvh * hd, 4);
                    float *gate = calloc((size_t)n_tok * n_head * hd, 4);
                    float *out = calloc((size_t)n_tok * n_head * hd, 4);
                    float *ref = calloc((size_t)n_tok * n_head * hd, 4);
                    if (!q || !k || !v || !gate || !out || !ref) return fail("qsa alloc");
                    for (int i = 0; i < npos * hd; i++) {
                        k[i] = ((i * 17) % 11) * 0.05f - 0.2f;
                        v[i] = ((i * 13) % 9) * 0.07f;
                    }
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        q[i] = ((i * 3) % 7) * 0.1f;
                        gate[i] = ((i % 5) - 2) * 0.3f;
                    }
                    int gqa = n_head / n_kvh;
                    for (int t = 0; t < n_tok; t++) {
                        int n_kv = kv_base + t + 1;
                        for (int h = 0; h < n_head; h++) {
                            int kh = h / gqa;
                            float sc[8];
                            float m = -1e30f;
                            for (int p = 0; p < n_kv; p++) {
                                float dot = 0.f;
                                for (int d = 0; d < hd; d++)
                                    dot += q[(t * n_head + h) * hd + d] *
                                           k[(p * n_kvh + kh) * hd + d];
                                sc[p] = dot * scale;
                                if (sc[p] > m) m = sc[p];
                            }
                            float sum = 0.f;
                            for (int p = 0; p < n_kv; p++) {
                                sc[p] = expf(sc[p] - m);
                                sum += sc[p];
                            }
                            for (int d = 0; d < hd; d++) {
                                float a = 0.f;
                                for (int p = 0; p < n_kv; p++)
                                    a += sc[p] * v[(p * n_kvh + kh) * hd + d];
                                float gv = 1.f / (1.f + expf(-gate[(t * n_head + h) * hd + d]));
                                ref[(t * n_head + h) * hd + d] = a / sum * gv;
                            }
                        }
                    }
                    float *dq = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                    float *dk = q4_hip_malloc((size_t)npos * hd * 4);
                    float *dv = q4_hip_malloc((size_t)npos * hd * 4);
                    float *dg = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                    float *dout = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                    void *dk16 = q4_hip_malloc((size_t)npos * hd * 2);
                    void *dv16 = q4_hip_malloc((size_t)npos * hd * 2);
                    if (!dq || !dk || !dv || !dg || !dout || !dk16 || !dv16)
                        return fail("qsa hip alloc");
                    q4_hip_h2d(dq, q, (size_t)n_tok * n_head * hd * 4);
                    q4_hip_h2d(dk, k, (size_t)npos * hd * 4);
                    q4_hip_h2d(dv, v, (size_t)npos * hd * 4);
                    q4_hip_h2d(dg, gate, (size_t)n_tok * n_head * hd * 4);
                    if (!q4_hip_qsa_decode_n(dq, dk, dv, dg, dout, n_head, n_kvh, hd,
                                             n_tok, kv_base, npos, scale, 0))
                        return fail("qsa f32 launch");
                    q4_hip_d2h(out, dout, (size_t)n_tok * n_head * hd * 4);
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        if (fabsf(out[i] - ref[i]) > 1e-3f) {
                            fprintf(stderr, "qsa f32 [%d] got %g ref %g\n", i, out[i], ref[i]);
                            return fail("qsa f32");
                        }
                    }
                    if (!q4_hip_f32_to_f16(dk16, dk, (uint64_t)npos * hd) ||
                        !q4_hip_f32_to_f16(dv16, dv, (uint64_t)npos * hd) ||
                        !q4_hip_qsa_decode_n(dq, dk16, dv16, dg, dout, n_head, n_kvh, hd,
                                             n_tok, kv_base, npos, scale, 1))
                        return fail("qsa f16 launch");
                    q4_hip_d2h(out, dout, (size_t)n_tok * n_head * hd * 4);
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        if (fabsf(out[i] - ref[i]) > 2e-2f) {
                            fprintf(stderr, "qsa f16 [%d] got %g ref %g\n", i, out[i], ref[i]);
                            return fail("qsa f16");
                        }
                    }
                    void *dk8 = q4_hip_malloc((size_t)npos * hd / 32 * 34);
                    void *dv8 = q4_hip_malloc((size_t)npos * hd / 32 * 34);
                    if (!dk8 || !dv8) return fail("qsa q8 alloc");
                    if (!q4_hip_f32_to_q8(dk8, dk, (uint64_t)npos * hd) ||
                        !q4_hip_f32_to_q8(dv8, dv, (uint64_t)npos * hd) ||
                        !q4_hip_qsa_decode_n(dq, dk8, dv8, dg, dout, n_head, n_kvh, hd,
                                             n_tok, kv_base, npos, scale, 2))
                        return fail("qsa q8 launch");
                    q4_hip_d2h(out, dout, (size_t)n_tok * n_head * hd * 4);
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        if (fabsf(out[i] - ref[i]) > 8e-2f) {
                            fprintf(stderr, "qsa q8 [%d] got %g ref %g\n", i, out[i], ref[i]);
                            return fail("qsa q8");
                        }
                    }
                    q4_hip_free(dq); q4_hip_free(dk); q4_hip_free(dv);
                    q4_hip_free(dg); q4_hip_free(dout);
                    q4_hip_free(dk16); q4_hip_free(dv16);
                    q4_hip_free(dk8); q4_hip_free(dv8);
                    free(q); free(k); free(v); free(gate); free(out); free(ref);
                }
                float g0 = 0.f, g1 = 0.f, g3 = 0.f, g4 = 0.f, g6 = 0.f;
                float yg[2];
                if (!q4_hip_gemv(Q4_T_IQ2_S, iq2, 2, 256, x, yg, 1.f))
                    return fail("iq2 hip");
                g0 = yg[0];
                g1 = yg[1];
                if (fabsf(g0 - 256.f) > 0.5f || fabsf(g1 - 768.f) > 0.5f)
                    return fail("iq2 hip values");
                if (!q4_hip_gemv(Q4_T_IQ3_S, iq3, 1, 256, x, &g3, 1.f) ||
                    fabsf(g3 - 256.f) > 0.5f)
                    return fail("iq3 hip");
                if (!q4_hip_gemv(Q4_T_IQ4_NL, iq4, 1, 32, x32, &g4, 1.f) ||
                    fabsf(g4 - (-127.f * 32.f)) > 0.5f)
                    return fail("iq4 hip");
                if (!q4_hip_gemv(Q4_T_Q6_K, q6, 1, 256, x, &g6, 1.f) ||
                    fabsf(g6 - (-32.f * 256.f)) > 0.5f)
                    return fail("q6 hip");
                /* gemv_m mixed-type: one fused launch over the GDN seg set
                 * (Q6_K + Q4_K + BF16 + IQ3_S) must reproduce per-segment
                 * gemv_dd outputs. */
                {
                    const uint32_t in = 256;
                    const uint32_t ty[4] = { Q4_T_Q6_K, Q4_T_Q4_K,
                                             Q4_T_BF16, Q4_T_IQ3_S };
                    const uint32_t rows[4] = { 3, 2, 4, 2 };
                    uint8_t *hw[4] = {0}; float *hrefs[4] = {0};
                    uint8_t *dw[4] = {0}; float *do_[4] = {0};
                    float xin[256];
                    for (int i = 0; i < 256; i++)
                        xin[i] = ((i * 13) % 17) * 0.03125f - 0.25f;
                    float *dx = q4_hip_malloc(in * 4);
                    bool ok = dx && q4_hip_h2d(dx, xin, in * 4);
                    for (int s = 0; s < 4 && ok; s++) {
                        uint64_t rb = q4_row_bytes(ty[s], in);
                        uint32_t n = rows[s];
                        hw[s] = malloc((size_t)n * rb);
                        hrefs[s] = malloc((size_t)n * 4);
                        dw[s] = q4_hip_malloc((size_t)n * rb);
                        do_[s] = q4_hip_malloc((size_t)n * 4);
                        ok = hw[s] && hrefs[s] && dw[s] && do_[s] && rb;
                        if (!ok) break;
                        for (uint32_t r = 0; r < n; r++) {
                            uint8_t *w = hw[s] + (size_t)r * rb;
                            for (uint64_t b = 0; b < rb; b++)
                                w[b] = (uint8_t)((b * 31u + r * 17u + 7u) & 0xffu);
                            /* sane scales: f16 d fields -> ~0.5+r*0.25 */
                            uint16_t d16 = (uint16_t)(0x3800 + r * 0x0400);
                            memcpy(w, &d16, 2);
                            if (ty[s] == Q4_T_Q6_K)
                                memcpy(w + 208, &d16, 2);
                            else if (ty[s] == Q4_T_Q4_K) {
                                uint16_t dm = 0x3400; /* dmin 0.25 */
                                memcpy(w + 2, &dm, 2);
                            } else if (ty[s] == Q4_T_BF16) {
                                for (uint32_t e = 0; e < in; e++)
                                    ((uint16_t *)w)[e] =
                                        (uint16_t)(0x3c00 + ((e + r) & 0x3f));
                            }
                        }
                        ok = q4_hip_h2d(dw[s], hw[s], (size_t)n * rb) &&
                             q4_hip_gemv_dd(ty[s], dw[s], n, in, dx,
                                            do_[s], 1.f) &&
                             q4_hip_d2h(hrefs[s], do_[s], (size_t)n * 4);
                    }
                    q4_seg_t segs[4];
                    float *dgot = q4_hip_malloc(11 * 4);
                    float hgot[11];
                    uint32_t off = 0;
                    for (int s = 0; s < 4; s++) {
                        segs[s] = (q4_seg_t){ dw[s], dgot + off,
                                              rows[s], ty[s], 0.f };
                        off += rows[s];
                    }
                    ok = ok && dgot && q4_hip_gemv_m(segs, 4, dx, in) &&
                         q4_hip_d2h(hgot, dgot, sizeof(hgot));
                    if (!ok) { fprintf(stderr, "gemv_m launch/copy failed\n");
                               return fail("gemv_m run"); }
                    off = 0;
                    for (int s = 0; s < 4; s++) {
                        for (uint32_t r = 0; r < rows[s]; r++) {
                            float a = hgot[off + r], b = hrefs[s][r];
                            float e = fabsf(a - b);
                            if (!(e <= 2e-3f * (1.f + fabsf(b)))) {
                                fprintf(stderr,
                                        "gemv_m seg %d row %u: got %g ref %g\n",
                                        s, r, a, b);
                                return fail("gemv_m values");
                            }
                        }
                        off += rows[s];
                    }
                    for (int s = 0; s < 4; s++) {
                        free(hw[s]); free(hrefs[s]);
                        q4_hip_free(dw[s]); q4_hip_free(do_[s]);
                    }
                    q4_hip_free(dx); q4_hip_free(dgot);
                }
                /* bf16 batched GEMM (gemm_wmma_bf16_k) vs fp32 host ref:
                 * bf16 rounding allows ~1e-2 relative on these magnitudes; a
                 * fragment-layout bug lands far outside that. */
                {
                    const uint32_t in = 256, nr = 5, nb = 17;
                    uint16_t *hw = malloc((size_t)nr * in * 2);
                    float *hx = malloc((size_t)nb * in * 4);
                    float *hy = malloc((size_t)nb * nr * 4);
                    uint16_t *dw = q4_hip_malloc((size_t)nr * in * 2);
                    float *dx = q4_hip_malloc((size_t)nb * in * 4);
                    float *dy = q4_hip_malloc((size_t)nb * nr * 4);
                    bool ok = hw && hx && hy && dw && dx && dy;
                    for (uint32_t i = 0; ok && i < nr * in; i++)
                        hw[i] = (uint16_t)(0x3b80u +
                                ((uint16_t)((i * 37u) % 127u)));
                    for (uint32_t i = 0; ok && i < nb * in; i++)
                        hx[i] = ((int)((i * 29u) % 89u) - 44) * 0.0078125f;
                    ok = ok && q4_hip_h2d(dw, hw, (size_t)nr * in * 2) &&
                         q4_hip_h2d(dx, hx, (size_t)nb * in * 4) &&
                         q4_hip_gemv_dd_n(Q4_T_BF16, (const uint8_t *)dw, nr,
                                          in, dx, dy, 1.f, nb) &&
                         q4_hip_d2h(hy, dy, (size_t)nb * nr * 4);
                    if (!ok) return fail("bf16 gemm run");
                    for (uint32_t t = 0; t < nb; t++)
                        for (uint32_t r = 0; r < nr; r++) {
                            double acc = 0;
                            for (uint32_t i = 0; i < in; i++) {
                                uint32_t u =
                                    (uint32_t)hw[(size_t)r * in + i] << 16;
                                float wf;
                                memcpy(&wf, &u, 4);
                                acc += (double)wf * hx[(size_t)t * in + i];
                            }
                            float got = hy[(size_t)t * nr + r];
                            float e = fabsf(got - (float)acc);
                            if (!(e <= 1e-2f * (1.f + fabsf((float)acc)))) {
                                fprintf(stderr,
                                        "bf16 gemm t%u r%u: got %g ref %g\n",
                                        t, r, got, (float)acc);
                                return fail("bf16 gemm values");
                            }
                        }
                    free(hw); free(hx); free(hy);
                    q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                }
                /* 2..8 token bf16: narrow rows use one block per row, short
                 * rows use the sub-warp kernel.  Each token matches the
                 * single-token launch, and the silu epilogue matches
                 * scale_silu(dot). */
                {
                    struct { uint32_t nr, nc; } shapes[] = {
                        { 32, 2048 }, { 64, 320 }, { 1040, 2048 },
                    };
                    const uint32_t nb = 4;
                    for (int si = 0; si < 3; si++) {
                        uint32_t nr = shapes[si].nr, nc = shapes[si].nc;
                        uint16_t *hw = calloc((size_t)nr * nc, 2);
                        float *hx = malloc((size_t)nb * nc * 4);
                        float *hy = malloc((size_t)nb * nr * 4);
                        float *h1 = malloc((size_t)nr * 4);
                        uint16_t *dw = q4_hip_malloc((size_t)nr * nc * 2);
                        float *dx = q4_hip_malloc((size_t)nb * nc * 4);
                        float *dy = q4_hip_malloc((size_t)nb * nr * 4);
                        float *dy1 = q4_hip_malloc((size_t)nr * 4);
                        if (!hw || !hx || !hy || !h1 || !dw || !dx || !dy ||
                            !dy1)
                            return fail("bf16n alloc");
                        for (uint32_t i = 0; i < nr * nc; i++)
                            hw[i] = (uint16_t)(0x3c00u + (i * 17u) % 200u);
                        for (uint32_t i = 0; i < nb * nc; i++)
                            hx[i] = ((int)(i * 13u) % 41 - 20) * 0.015625f;
                        if (!q4_hip_h2d(dw, hw, (size_t)nr * nc * 2) ||
                            !q4_hip_h2d(dx, hx, (size_t)nb * nc * 4) ||
                            !q4_hip_gemv_dd_n(Q4_T_BF16, (const uint8_t *)dw,
                                              nr, nc, dx, dy, 1.f, nb) ||
                            !q4_hip_d2h(hy, dy, (size_t)nb * nr * 4))
                            return fail("bf16n run");
                        for (uint32_t t = 0; t < nb; t++) {
                            if (!q4_hip_gemv_dd(Q4_T_BF16, (const uint8_t *)dw,
                                                nr, nc, dx + (size_t)t * nc,
                                                dy1, 1.f) ||
                                !q4_hip_d2h(h1, dy1, (size_t)nr * 4))
                                return fail("bf16n n1");
                            for (uint32_t r = 0; r < nr; r++) {
                                float a = hy[(size_t)t * nr + r], s1 = h1[r];
                                if (a != s1) {
                                    fprintf(stderr,
                                            "bf16n shape %u t%u r%u: batch %g "
                                            "single %g\n",
                                            nr, t, r, a, s1);
                                    return fail("bf16n match");
                                }
                            }
                        }
                        double num = 0, den = 0;
                        for (uint32_t t = 0; t < nb; t++)
                            for (uint32_t r = 0; r < nr; r += nr > 64 ? 16 : 1) {
                                double acc = 0;
                                for (uint32_t k = 0; k < nc; k++) {
                                    uint32_t u = (uint32_t)hw[(size_t)r * nc + k]
                                                 << 16;
                                    float wf;
                                    memcpy(&wf, &u, 4);
                                    acc += (double)wf * hx[(size_t)t * nc + k];
                                }
                                double d = acc - hy[(size_t)t * nr + r];
                                num += d * d;
                                den += acc * acc;
                            }
                        double rel = den > 0 ? sqrt(num / den) : sqrt(num);
                        if (rel >= 1e-3) {
                            fprintf(stderr, "bf16n nr %u rel %.5f\n", nr, rel);
                            return fail("bf16n ref");
                        }
                        free(hw); free(hx); free(hy); free(h1);
                        q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                        q4_hip_free(dy1);
                    }
                    {
                        const uint32_t nr = 8, nc = 2048, nt = 4;
                        const float act = 0.25f;
                        uint16_t *hw = calloc((size_t)nr * nc, 2);
                        float *hx = malloc((size_t)nt * nc * 4);
                        float *hy = malloc((size_t)nt * nr * 4);
                        float *h1 = malloc((size_t)nr * 4);
                        uint16_t *dw = q4_hip_malloc((size_t)nr * nc * 2);
                        float *dx = q4_hip_malloc((size_t)nt * nc * 4);
                        float *dy = q4_hip_malloc((size_t)nt * nr * 4);
                        float *dy1 = q4_hip_malloc((size_t)nr * 4);
                        if (!hw || !hx || !hy || !h1 || !dw || !dx || !dy ||
                            !dy1)
                            return fail("bf16n act alloc");
                        for (uint32_t i = 0; i < nr * nc; i++)
                            hw[i] = (uint16_t)(0x3c00u + (i * 19u) % 180u);
                        for (uint32_t i = 0; i < nt * nc; i++)
                            hx[i] = ((int)(i * 11u) % 31 - 15) * 0.03125f;
                        if (!q4_hip_h2d(dw, hw, (size_t)nr * nc * 2) ||
                            !q4_hip_h2d(dx, hx, (size_t)nt * nc * 4) ||
                            !q4_hip_gemv_dd_n_act(Q4_T_BF16,
                                                  (const uint8_t *)dw, nr, nc,
                                                  dx, dy, 1.f, act, nt) ||
                            !q4_hip_d2h(hy, dy, (size_t)nt * nr * 4))
                            return fail("bf16n act run");
                        for (uint32_t t = 0; t < nt; t++) {
                            if (!q4_hip_gemv_dd_act(Q4_T_BF16,
                                                    (const uint8_t *)dw, nr, nc,
                                                    dx + (size_t)t * nc, dy1,
                                                    1.f, act) ||
                                !q4_hip_d2h(h1, dy1, (size_t)nr * 4))
                                return fail("bf16n act n1");
                            for (uint32_t r = 0; r < nr; r++)
                                if (hy[(size_t)t * nr + r] != h1[r])
                                    return fail("bf16n act match");
                        }
                        free(hw); free(hx); free(hy); free(h1);
                        q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                        q4_hip_free(dy1);
                    }
                    {
                        const uint32_t ty = Q4_T_Q6_K, nr = 16, nc = 512, nt = 4;
                        uint64_t rb = q4_row_bytes(ty, nc);
                        uint8_t *w = malloc((size_t)nr * rb);
                        float *hx = malloc((size_t)nt * nc * 4);
                        float *hy = malloc((size_t)nt * nr * 4);
                        float *h1 = malloc((size_t)nr * 4);
                        if (!w || !rb || !hx || !hy || !h1)
                            return fail("q6n alloc");
                        for (uint64_t i = 0; i < (uint64_t)nr * rb; i++)
                            w[i] = (uint8_t)(i * 17u);
                        const uint16_t ds = 0x2e66;
                        for (uint32_t r = 0; r < nr; r++)
                            for (uint32_t blk = 0; blk < nc / 256; blk++)
                                memcpy(w + (size_t)r * rb + blk * 210 + 208,
                                       &ds, 2);
                        for (uint32_t i = 0; i < nt * nc; i++)
                            hx[i] = ((int)(i * 9u) % 23 - 11) * 0.0625f;
                        uint8_t *dw = q4_hip_malloc((size_t)nr * rb);
                        float *dx = q4_hip_malloc((size_t)nt * nc * 4);
                        float *dy = q4_hip_malloc((size_t)nt * nr * 4);
                        float *dy1 = q4_hip_malloc((size_t)nr * 4);
                        if (!dw || !dx || !dy || !dy1 ||
                            !q4_hip_h2d(dw, w, (size_t)nr * rb) ||
                            !q4_hip_h2d(dx, hx, (size_t)nt * nc * 4) ||
                            !q4_hip_gemv_dd_n(ty, dw, nr, nc, dx, dy, 1.f, nt) ||
                            !q4_hip_d2h(hy, dy, (size_t)nt * nr * 4))
                            return fail("q6n run");
                        for (uint32_t t = 0; t < nt; t++) {
                            if (!q4_hip_gemv_dd(ty, dw, nr, nc,
                                                dx + (size_t)t * nc, dy1, 1.f) ||
                                !q4_hip_d2h(h1, dy1, (size_t)nr * 4))
                                return fail("q6n n1");
                            for (uint32_t r = 0; r < nr; r++)
                                if (hy[(size_t)t * nr + r] != h1[r]) {
                                    fprintf(stderr,
                                            "q6n t%u r%u: batch %g single %g\n",
                                            t, r, hy[(size_t)t * nr + r], h1[r]);
                                    return fail("q6n match");
                                }
                        }
                        free(w); free(hx); free(hy); free(h1);
                        q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                        q4_hip_free(dy1);
                    }
                }
                /* hc_comb_rms: fused combine+norm must match the two-kernel
                 * sequence bit-for-bit (same per-element math). */
                {
                    const uint32_t hc = 4, ne = 512;
                    const uint32_t tot = hc * ne;
                    float hres[2048], hblk[512], hinj[4], hwn[2048];
                    for (uint32_t i = 0; i < tot; i++)
                        hres[i] = ((int)(i * 37 % 101) - 50) * 0.01171875f;
                    for (uint32_t i = 0; i < ne; i++)
                        hblk[i] = ((int)(i * 53 % 89) - 44) * 0.0078125f;
                    for (uint32_t i = 0; i < hc; i++)
                        hinj[i] = ((int)(i * 29 % 41) - 20) * 0.0625f;
                    for (uint32_t i = 0; i < tot; i++)
                        hwn[i] = 0.75f + ((i * 7) % 19) * 0.015625f;
                    float *dra = q4_hip_malloc(tot * 4);
                    float *drb = q4_hip_malloc(tot * 4);
                    float *db = q4_hip_malloc(ne * 4);
                    float *di = q4_hip_malloc(hc * 4);
                    float *dw = q4_hip_malloc(tot * 4);
                    float *dya = q4_hip_malloc(tot * 4);
                    float *dyb = q4_hip_malloc(tot * 4);
                    bool ok = dra && drb && db && di && dw && dya && dyb &&
                              q4_hip_h2d(dra, hres, tot * 4) &&
                              q4_hip_h2d(drb, hres, tot * 4) &&
                              q4_hip_h2d(db, hblk, ne * 4) &&
                              q4_hip_h2d(di, hinj, hc * 4) &&
                              q4_hip_h2d(dw, hwn, tot * 4);
                    ok = ok && q4_hip_hc_combine(dra, db, di, ne, hc) &&
                         q4_hip_grouped_rms(dra, dw, dya, ne, hc, 1e-6f) &&
                         q4_hip_hc_comb_rms(drb, db, di, dw, dyb, ne, hc,
                                            1e-6f);
                    float yra[2048], yrb[2048], rra[2048], rrb[2048];
                    ok = ok && q4_hip_d2h(yra, dya, tot * 4) &&
                         q4_hip_d2h(yrb, dyb, tot * 4) &&
                         q4_hip_d2h(rra, dra, tot * 4) &&
                         q4_hip_d2h(rrb, drb, tot * 4);
                    if (!ok) return fail("hc_comb_rms run");
                    for (uint32_t i = 0; i < tot; i++) {
                        if (yra[i] != yrb[i] || rra[i] != rrb[i]) {
                            fprintf(stderr, "hc_comb_rms i=%u y %g vs %g\n",
                                    i, yra[i], yrb[i]);
                            return fail("hc_comb_rms values");
                        }
                    }
                    q4_hip_free(dra); q4_hip_free(drb); q4_hip_free(db);
                    q4_hip_free(di); q4_hip_free(dw); q4_hip_free(dya);
                    q4_hip_free(dyb);
                }
                /* qsa_prep3: fused q/k/idx prep must match the three
                 * standalone prep launches. */
                {
                    const uint32_t nh = 2, nk = 1, ni = 1;
                    const uint32_t hd = 64, ihd = 32, nr = 16;
                    const float base = 1e4f, eps = 1e-6f;
                    float hqg[2 * 128], hk[64], hik[32];
                    for (int i = 0; i < 256; i++)
                        hqg[i] = ((int)(i * 23 % 67) - 33) * 0.015625f;
                    for (int i = 0; i < 64; i++)
                        hk[i] = ((int)(i * 41 % 53) - 26) * 0.015625f;
                    for (int i = 0; i < 32; i++)
                        hik[i] = ((int)(i * 19 % 47) - 23) * 0.015625f;
                    float *dqg = q4_hip_malloc(2 * 128 * 4);
                    float *dk = q4_hip_malloc(64 * 4);
                    float *dk2 = q4_hip_malloc(64 * 4);
                    float *dik = q4_hip_malloc(32 * 4);
                    float *dik2 = q4_hip_malloc(32 * 4);
                    float *dq = q4_hip_malloc(2 * 64 * 4);
                    float *dg = q4_hip_malloc(2 * 64 * 4);
                    float *dq2 = q4_hip_malloc(2 * 64 * 4);
                    float *dg2 = q4_hip_malloc(2 * 64 * 4);
                    bool ok = dqg && dk && dk2 && dik && dik2 && dq && dg &&
                              dq2 && dg2 &&
                              q4_hip_h2d(dqg, hqg, sizeof(hqg)) &&
                              q4_hip_h2d(dk, hk, sizeof(hk)) &&
                              q4_hip_h2d(dk2, hk, sizeof(hk)) &&
                              q4_hip_h2d(dik, hik, sizeof(hik)) &&
                              q4_hip_h2d(dik2, hik, sizeof(hik));
                    /* reference: three standalone launches at pos 37 */
                    ok = ok &&
                         q4_hip_qsa_q_prep(dqg, NULL, dq, dg, nh, hd, nr,
                                           37, base, eps) &&
                         q4_hip_qsa_k_prep(dk, NULL, nk, hd, nr, 37, base,
                                           eps) &&
                         q4_hip_qsa_k_prep(dik, NULL, ni, ihd, nr, 37, base,
                                           eps);
                    q4_hip_pos_push(37);
                    ok = ok &&
                         q4_hip_qsa_prep3_d(dqg, NULL, dq2, dg2, nh, dk2,
                                            NULL, nk, dik2, NULL, ni, hd,
                                            ihd, nr, base, eps);
                    float hq[128], hq2[128], hgo[128], hgo2[128];
                    float hko[64], hko2[64], hio[32], hio2[32];
                    ok = ok && q4_hip_d2h(hq, dq, sizeof(hq)) &&
                         q4_hip_d2h(hq2, dq2, sizeof(hq2)) &&
                         q4_hip_d2h(hgo, dg, sizeof(hgo)) &&
                         q4_hip_d2h(hgo2, dg2, sizeof(hgo2)) &&
                         q4_hip_d2h(hko, dk, sizeof(hko)) &&
                         q4_hip_d2h(hko2, dk2, sizeof(hko2)) &&
                         q4_hip_d2h(hio, dik, sizeof(hio)) &&
                         q4_hip_d2h(hio2, dik2, sizeof(hio2));
                    if (!ok) return fail("qsa_prep3 run");
                    for (int i = 0; i < 128; i++)
                        if (hq[i] != hq2[i] || hgo[i] != hgo2[i])
                            return fail("qsa_prep3 q values");
                    for (int i = 0; i < 64; i++)
                        if (hko[i] != hko2[i]) return fail("qsa_prep3 k");
                    for (int i = 0; i < 32; i++)
                        if (hio[i] != hio2[i]) return fail("qsa_prep3 idx");
                    q4_hip_free(dqg); q4_hip_free(dk); q4_hip_free(dk2);
                    q4_hip_free(dik); q4_hip_free(dik2); q4_hip_free(dq);
                    q4_hip_free(dg); q4_hip_free(dq2); q4_hip_free(dg2);
                }
                /* M-RoPE table (images): pair p rotates by stream p % 3 of
                 * its cell's (t, h, w); cells past the table at cell -
                 * shift; an identity table is bit-identical to none. */
                {
                    const uint32_t hd = 64, nr = 64, nt = 3;
                    const float base = 1e4f, eps = 1e-6f;
                    float hk[3 * 64];
                    for (int i = 0; i < 3 * 64; i++)
                        hk[i] = ((int)(i * 29 % 61) - 30) * 0.03125f;
                    const int32_t tab[5 * 3] = {0, 0, 0,  1, 1, 1,  2, 2, 2,
                                                2, 3, 2,  2, 3, 3};
                    float *dk = q4_hip_malloc(sizeof(hk));
                    float out[3][3 * 64];
                    bool ok = dk != NULL;
                    /* run 0: no table; 1: identity; 2: the image table
                     * from cell 2 (cells 2..4), 3: cells 4..6 (two past) */
                    const int32_t idt[7 * 3] = {0, 0, 0, 1, 1, 1, 2, 2, 2,
                                                3, 3, 3, 4, 4, 4, 5, 5, 5,
                                                6, 6, 6};
                    for (int r = 0; r < 3 && ok; r++) {
                        ok = q4_hip_mrope_set(r == 0 ? NULL : r == 1 ? idt : tab,
                                              r == 1 ? 7 : 5, r == 2 ? 1 : 0) &&
                             q4_hip_h2d(dk, hk, sizeof(hk)) &&
                             q4_hip_qsa_k_prep_n(dk, NULL, 1, hd, nr, 2, base,
                                                 eps, nt) &&
                             q4_hip_d2h(out[r], dk, sizeof(hk));
                    }
                    float past[3 * 64];
                    ok = ok && q4_hip_h2d(dk, hk, sizeof(hk)) &&
                         q4_hip_qsa_k_prep_n(dk, NULL, 1, hd, nr, 4, base, eps,
                                             nt) &&
                         q4_hip_d2h(past, dk, sizeof(hk)) &&
                         q4_hip_mrope_set(NULL, 0, 0);
                    if (!ok) return fail("mrope run");
                    if (memcmp(out[0], out[1], sizeof(hk)))
                        return fail("mrope identity table not bit-identical");
                    double md = 0;
                    for (int run = 0; run < 2; run++)
                        for (uint32_t t = 0; t < nt; t++) {
                            const float *src = hk + t * hd;
                            const float *got = run ? past + t * hd : out[2] + t * hd;
                            const int cell = (run ? 4 : 2) + (int)t;
                            float ss = 0, v[64];
                            for (uint32_t i = 0; i < hd; i++) ss += src[i] * src[i];
                            const float sc = 1.f / sqrtf(ss / hd + eps);
                            for (uint32_t i = 0; i < hd; i++) v[i] = src[i] * sc;
                            for (uint32_t p = 0; p < nr / 2; p++) {
                                const int pos = cell < 5 ? tab[cell * 3 + p % 3]
                                                         : cell - 1;
                                const double f = pow(base, -(double)p / (nr / 2));
                                const double a = pos * f;
                                /* NEOX pairs (p, p + n_rot/2) */
                                const double x0 = v[p], x1 = v[p + nr / 2];
                                const double r0 = x0 * cos(a) - x1 * sin(a);
                                const double r1 = x0 * sin(a) + x1 * cos(a);
                                md = fmax(md, fabs(r0 - got[p]));
                                md = fmax(md, fabs(r1 - got[p + nr / 2]));
                            }
                        }
                    if (md > 2e-4) {
                        fprintf(stderr, "mrope max diff %g\n", md);
                        return fail("mrope table rotation");
                    }
                    q4_hip_free(dk);
                    printf("mrope table ok (max diff %.2g)\n", md);
                }
                /* Batched path dequants each row once. Four identical tokens. */
                {
                    float xb[4 * 256];
                    for (int t = 0; t < 4; t++)
                        memcpy(xb + t * 256, x, 256 * sizeof(float));
                    uint8_t *dw = q4_hip_malloc(82ull * 2);
                    float *dx = q4_hip_malloc(4ull * 256 * 4);
                    float *dy = q4_hip_malloc(4ull * 2 * 4);
                    float yh[8];
                    if (!dw || !dx || !dy) return fail("iq2 batch alloc");
                    q4_hip_h2d(dw, iq2, 82ull * 2);
                    q4_hip_h2d(dx, xb, 4ull * 256 * 4);
                    if (!q4_hip_gemv_dd_n(Q4_T_IQ2_S, dw, 2, 256, dx, dy, 1.f, 4))
                        return fail("iq2 batch launch");
                    q4_hip_d2h(yh, dy, sizeof(yh));
                    for (int t = 0; t < 4; t++) {
                        if (fabsf(yh[t * 2] - 256.f) > 0.5f ||
                            fabsf(yh[t * 2 + 1] - 768.f) > 0.5f) {
                            fprintf(stderr, "iq2 batch t%d %g %g\n", t, yh[t * 2],
                                    yh[t * 2 + 1]);
                            return fail("iq2 batch");
                        }
                    }
                    q4_hip_free(dw);
                    q4_hip_free(dx);
                    q4_hip_free(dy);
                }
                /* route_dev_k: the warp top-k extraction must reproduce the
                 * serial softmax -> topk -> renormalize -> devtab split,
                 * including index-ascending order on exact prob ties. */
                {
                    const uint32_t ne = 512, tk = 10;
                    float lg[512], pr[512];
                    int32_t tab[512];
                    for (uint32_t i = 0; i < ne; i++) {
                        lg[i] = (float)(((i * 2654435761u) >> 8) & 0xffff) *
                                    6e-5f - 2.f;
                        tab[i] = (i % 3 == 0) ? (int32_t)(i / 3) : -1;
                    }
                    lg[7] = lg[71] = lg[500] = 3.f;   /* exact tie at max */
                    lg[100] = lg[200] = lg[300] = 2.f; /* ties mid-pack */
                    memcpy(pr, lg, sizeof(pr));
                    q4_softmax(pr, ne);
                    int32_t rids[16];
                    float rwt[16];
                    memcpy(pr, lg, sizeof(pr));
                    q4_softmax(pr, ne);
                    q4_topk(pr, ne, tk, rids, rwt, true);
                    uint32_t xnh = 0, xnm = 0, nres = 0;
                    int32_t xes[64], xmids[64];
                    float xew[64], xmwt[64];
                    for (uint32_t k = 0; k < tk; k++) {
                        int32_t slot = tab[rids[k]];
                        if (slot >= 0) { xes[xnh] = slot; xew[xnh] = rwt[k]; xnh++; }
                        else { xmids[xnm] = rids[k]; xmwt[xnm] = rwt[k]; xnm++; }
                    }
                    float *d_rlog = (float *)q4_hip_malloc(ne * 4);
                    int32_t *d_tab = (int32_t *)q4_hip_malloc(ne * 4);
                    int32_t *d_es = (int32_t *)q4_hip_malloc(64 * 4);
                    float *d_ew = (float *)q4_hip_malloc(64 * 4);
                    int32_t *d_nh = (int32_t *)q4_hip_malloc(4);
                    uint8_t *d_mail = (uint8_t *)q4_hip_malloc(1280);
                    unsigned long long *d_seq =
                        (unsigned long long *)q4_hip_malloc(16);
                    unsigned long long z[2] = {0, 0};
                    q4_hip_h2d(d_rlog, lg, ne * 4);
                    q4_hip_h2d(d_tab, tab, ne * 4);
                    q4_hip_h2d(d_seq, z, 16);
                    if (!d_rlog || !d_tab || !d_es || !d_ew || !d_nh ||
                        !d_mail || !d_seq)
                        return fail("route alloc");
                    int32_t *d_res_s = (int32_t *)q4_hip_malloc(64 * 4);
                    float *d_res_w = (float *)q4_hip_malloc(64 * 4);
                    int32_t *d_res_n = (int32_t *)q4_hip_malloc(4);
                    if (!d_res_s || !d_res_w || !d_res_n)
                        return fail("route res alloc");
                    if (!q4_hip_route_dev(d_rlog, d_tab, d_es, d_ew, d_nh,
                                          NULL, d_res_s, d_res_w, d_res_n,
                                          64, d_mail, d_seq, NULL, NULL,
                                          ne, tk))
                        return fail("route launch");
                    /* Contract: topk 0 or >64 must be rejected — the
                     * kernel's s_pay staging is 64-deep and >64 would be a
                     * shared-memory OOB write. */
                    if (q4_hip_route_dev(d_rlog, d_tab, d_es, d_ew, d_nh,
                                         NULL, d_res_s, d_res_w, d_res_n,
                                         64, d_mail, d_seq, NULL, NULL,
                                         ne, 0) ||
                        q4_hip_route_dev(d_rlog, d_tab, d_es, d_ew, d_nh,
                                         NULL, d_res_s, d_res_w, d_res_n,
                                         64, d_mail, d_seq, NULL, NULL,
                                         ne, 65))
                        return fail("route topk bounds");
                    /* restab == NULL ⇒ every L1 miss lands in the mailbox. */
                    int32_t hres_n = -1;
                    q4_hip_d2h(&hres_n, d_res_n, 4);
                    if (hres_n != 0) return fail("route res_n null");
                    int32_t hes[64];
                    float hew[64];
                    int32_t hnh = -1;
                    uint8_t hmail[1280];
                    q4_hip_d2h(&hnh, d_nh, 4);
                    q4_hip_d2h(hes, d_es, 64 * 4);
                    q4_hip_d2h(hew, d_ew, 64 * 4);
                    q4_hip_d2h(hmail, d_mail, 1280);
                    if ((uint32_t)hnh != xnh) return fail("route nhit");
                    for (uint32_t i = 0; i < xnh; i++) {
                        if (hes[i] != xes[i]) return fail("route hit id");
                        if (fabsf(hew[i] - xew[i]) > 1e-6f)
                            return fail("route hit wt");
                    }
                    /* staging layout: ids[64]@0 wts[64]@256 n@512 ntok@516
                     * tn@520 hs@552 hn@1064 rk@1096 routed@1160 rseq@1224
                     * seq@1232 */
                    uint32_t m_n, m_ntok;
                    unsigned long long m_seq;
                    memcpy(&m_n, hmail + 512, 4);
                    memcpy(&m_ntok, hmail + 516, 4);
                    memcpy(&m_seq, hmail + 1232, 8);
                    if (m_n != xnm || m_ntok != 0 || m_seq != 1)
                        return fail("route mail hdr");
                    for (uint32_t i = 0; i < xnm; i++) {
                        int32_t id;
                        float w;
                        memcpy(&id, hmail + i * 4, 4);
                        memcpy(&w, hmail + 256 + i * 4, 4);
                        if (id != xmids[i]) return fail("route miss id");
                        if (fabsf(w - xmwt[i]) > 1e-6f)
                            return fail("route miss wt");
                    }
                    /* Resident-table split: mark two of the miss experts as
                     * resident slots; they must move to the res work list in
                     * top-k order and out of the mailbox payload. */
                    {
                        int32_t rtab[512];
                        memset(rtab, 0, sizeof(rtab));
                        nres = xnm > 2 ? 2 : xnm;
                        for (uint32_t i = 0; i < nres; i++)
                            rtab[xmids[i]] = 40 + (int32_t)i;
                        int32_t *d_rtab =
                            (int32_t *)q4_hip_malloc(ne * 4);
                        if (!d_rtab) return fail("route rtab alloc");
                        q4_hip_h2d(d_rtab, rtab, ne * 4);
                        if (!q4_hip_route_dev(d_rlog, d_tab, d_es, d_ew,
                                              d_nh, d_rtab, d_res_s, d_res_w,
                                              d_res_n, 64, d_mail, d_seq,
                                              NULL, NULL, ne, tk))
                            return fail("route res launch");
                        int32_t hrn = -1, hrs[64];
                        float hrw[64];
                        q4_hip_d2h(&hrn, d_res_n, 4);
                        q4_hip_d2h(hrs, d_res_s, 64 * 4);
                        q4_hip_d2h(hrw, d_res_w, 64 * 4);
                        q4_hip_d2h(hmail, d_mail, 1280);
                        if ((uint32_t)hrn != nres)
                            return fail("route res count");
                        for (uint32_t i = 0; i < nres; i++) {
                            if (hrs[i] != rtab[xmids[i]] - 1)
                                return fail("route res slot");
                            if (fabsf(hrw[i] - xmwt[i]) > 1e-6f)
                                return fail("route res wt");
                        }
                        memcpy(&m_n, hmail + 512, 4);
                        if (m_n != xnm - nres)
                            return fail("route res mail n");
                        for (uint32_t i = 0; i < xnm - nres; i++) {
                            int32_t id;
                            memcpy(&id, hmail + i * 4, 4);
                            if (id != xmids[nres + i])
                                return fail("route res miss id");
                        }
                        /* res_cap=1: only the first resident miss takes the
                         * GPU path; the second falls back to the mailbox. */
                        if (nres == 2 &&
                            !q4_hip_route_dev(d_rlog, d_tab, d_es, d_ew,
                                              d_nh, d_rtab, d_res_s, d_res_w,
                                              d_res_n, 1, d_mail, d_seq,
                                              NULL, NULL, ne, tk))
                            return fail("route cap launch");
                        if (nres == 2) {
                            q4_hip_d2h(&hrn, d_res_n, 4);
                            q4_hip_d2h(hrs, d_res_s, 4);
                            q4_hip_d2h(hmail, d_mail, 1280);
                            memcpy(&m_n, hmail + 512, 4);
                            if (hrn != 1 || hrs[0] != rtab[xmids[0]] - 1 ||
                                m_n != xnm - 1)
                                return fail("route cap split");
                            memcpy(&m_n, hmail + 0, 4);
                            if (m_n != xmids[1])
                                return fail("route cap miss order");
                        }
                        q4_hip_free(d_rtab);
                    }
                    /* Batched variant: 4 tokens x topk 10, same refs. */
                    {
                        const uint32_t nt = 4;
                        float lg4[4 * 512], pr4[512];
                        for (uint32_t t = 0; t < nt; t++)
                            for (uint32_t i = 0; i < ne; i++) {
                                uint32_t h = (i + t * 977) * 2654435761u;
                                lg4[t * ne + i] =
                                    (float)((h >> 8) & 0xffff) * 6e-5f - 2.f;
                            }
                        lg4[1 * ne + 3] = lg4[1 * ne + 9] = 4.f;
                        lg4[3 * ne + 2] = lg4[3 * ne + 2 + 64] = 3.5f;
                        float *d_rl4 = (float *)q4_hip_malloc(nt * ne * 4);
                        int32_t *d_es4 = (int32_t *)q4_hip_malloc(8 * 64 * 4);
                        float *d_ew4 = (float *)q4_hip_malloc(8 * 64 * 4);
                        int32_t *d_nh4 = (int32_t *)q4_hip_malloc(8 * 4);
                        q4_hip_h2d(d_rl4, lg4, nt * ne * 4);
                        unsigned long long *d_rseq =
                            (unsigned long long *)q4_hip_malloc(8);
                        q4_hip_h2d(d_rseq, z, 8);
                        if (getenv("Q4_TEST_ROUTE_BENCH")) {
                            /* own mailbox + seq: the checks below are exact */
                            void *d_bm = q4_hip_malloc(1280);
                            unsigned long long *d_bs =
                                (unsigned long long *)q4_hip_malloc(8);
                            q4_hip_h2d(d_bs, z, 8);
                            /* back-to-back launches: per-call device time
                             * (+ launch floor, shown by a 1-element fill) */
                            for (int pass = 0; pass < 2; pass++) {
                                q4_hip_stream_sync();
                                struct timespec ta, tb;
                                clock_gettime(CLOCK_MONOTONIC, &ta);
                                for (int it = 0; it < 2000; it++) {
                                    if (pass)
                                        (void)q4_hip_fill(d_ew4, 0.f, 1);
                                    else
                                        (void)q4_hip_route_dev_n(
                                            d_rl4, d_tab, d_es4, d_ew4, d_nh4,
                                            d_bm, d_bs, NULL, NULL, ne, tk,
                                            nt);
                                }
                                q4_hip_stream_sync();
                                clock_gettime(CLOCK_MONOTONIC, &tb);
                                printf("route bench %s: %.2f us/call\n",
                                       pass ? "fill(1)" : "route_dev_n",
                                       ((tb.tv_sec - ta.tv_sec) * 1e6 +
                                        (tb.tv_nsec - ta.tv_nsec) * 1e-3) /
                                           2000);
                            }
                            q4_hip_free(d_bm);
                            q4_hip_free((void *)d_bs);
                        }
                        if (!q4_hip_route_dev_n(d_rl4, d_tab, d_es4, d_ew4,
                                                d_nh4, d_mail, d_seq, d_rseq,
                                                NULL, ne, tk, nt))
                            return fail("route n launch");
                        uint8_t hm4[1280];
                        int32_t hnh4[8], hes4[8 * 64];
                        float hew4[8 * 64];
                        q4_hip_d2h(hm4, d_mail, 1280);
                        q4_hip_d2h(hnh4, d_nh4, 8 * 4);
                        q4_hip_d2h(hes4, d_es4, 8 * 64 * 4);
                        q4_hip_d2h(hew4, d_ew4, 8 * 64 * 4);
                        uint32_t mn, mnt, mtn[8];
                        unsigned long long mseq;
                        memcpy(&mn, hm4 + 512, 4);
                        memcpy(&mnt, hm4 + 516, 4);
                        memcpy(mtn, hm4 + 520, 32);
                        memcpy(&mseq, hm4 + 1232, 8);
                        /* seq: launch1 ->1; res launch +1 iff misses remain
                         * (xnm>nres); cap launch +1 iff it ran (nres==2,
                         * which implies xnm-1>0 misses left); batch +1. */
                        {
                            uint32_t want = 2 + (xnm > nres) +
                                            (nres == 2 ? 1u : 0u);
                            if (mnt != nt || mseq != want)
                                return fail("route n hdr");
                        }
                        uint32_t mo = 0;
                        for (uint32_t t = 0; t < nt; t++) {
                            memcpy(pr4, lg4 + t * ne, sizeof(pr4));
                            q4_softmax(pr4, ne);
                            q4_topk(pr4, ne, tk, rids, rwt, true);
                            uint32_t nh = 0;
                            for (uint32_t k = 0; k < tk; k++) {
                                int32_t slot = tab[rids[k]];
                                if (slot >= 0) {
                                    if ((uint32_t)hes4[t * 64 + nh] != (uint32_t)slot ||
                                        fabsf(hew4[t * 64 + nh] - rwt[k]) > 1e-6f)
                                        return fail("route n hit");
                                    nh++;
                                } else {
                                    int32_t id;
                                    float w;
                                    memcpy(&id, hm4 + mo * 4, 4);
                                    memcpy(&w, hm4 + 256 + mo * 4, 4);
                                    if (id != rids[k] ||
                                        fabsf(w - rwt[k]) > 1e-6f)
                                        return fail("route n miss");
                                    /* the miss's top-k rank (Q4_TAIL_SKIP) */
                                    if (hm4[1096 + mo] != k)
                                        return fail("route n miss rank");
                                    mo++;
                                }
                            }
                            if ((uint32_t)hnh4[t] != nh || mtn[t] != tk - nh)
                                return fail("route n count");
                            /* the publish's hit slots (host LRU refresh) */
                            uint32_t mhn;
                            memcpy(&mhn, hm4 + 1064 + t * 4, 4);
                            if (mhn != nh) return fail("route n hn");
                            for (uint32_t i = 0; i < nh; i++) {
                                int32_t hsv;
                                memcpy(&hsv, hm4 + 552 + (t * 16 + i) * 4, 4);
                                if (hsv != hes4[t * 64 + i])
                                    return fail("route n hs");
                            }
                        }
                        if (mn != mo) return fail("route n total");
                        /* The one-warp kernel and the warp-per-token one
                         * (the default above) must agree bit for bit. */
                        {
                            unsigned long long *d_pf =
                                (unsigned long long *)q4_hip_malloc(64);
                            q4_hip_fill((float *)d_pf, 0.f, 16);
                            q4_hip_route_npar_set(0);
                            uint8_t hm5[1280];
                            int32_t hnh5[8], hes5[8 * 64];
                            float hew5[8 * 64];
                            if (!q4_hip_route_dev_n(d_rl4, d_tab, d_es4, d_ew4,
                                                    d_nh4, d_mail, d_seq, d_rseq,
                                                    d_pf, ne, tk, nt))
                                return fail("route n1 launch");
                            q4_hip_d2h(hm5, d_mail, 1280);
                            q4_hip_d2h(hnh5, d_nh4, 8 * 4);
                            q4_hip_d2h(hes5, d_es4, 8 * 64 * 4);
                            q4_hip_d2h(hew5, d_ew4, 8 * 64 * 4);
                            for (uint32_t t = 0; t < nt; t++) {
                                if (hnh5[t] != hnh4[t])
                                    return fail("route n par nh");
                                for (int32_t i = 0; i < hnh4[t]; i++)
                                    if (hes5[t * 64 + i] != hes4[t * 64 + i] ||
                                        memcmp(&hew5[t * 64 + i],
                                               &hew4[t * 64 + i], 4))
                                        return fail("route n par hit");
                            }
                            /* ids/wts of the misses, n, ntok, tn */
                            if (memcmp(hm5, hm4, mo * 4) ||
                                memcmp(hm5 + 256, hm4 + 256, mo * 4) ||
                                memcmp(hm5 + 512, hm4 + 512, 40))
                                return fail("route n par mail");
                            q4_hip_route_npar_set(1);
                            q4_hip_free(d_pf);
                        }
                        q4_hip_free(d_rl4); q4_hip_free(d_es4);
                        q4_hip_free(d_ew4); q4_hip_free(d_nh4);
                    }
                    q4_hip_free(d_rlog); q4_hip_free(d_tab);
                    q4_hip_free(d_es); q4_hip_free(d_ew); q4_hip_free(d_nh);
                    q4_hip_free(d_res_s); q4_hip_free(d_res_w);
                    q4_hip_free(d_res_n);
                    q4_hip_free(d_mail); q4_hip_free(d_seq);
                }
                /* route_pf_k + assign_pf_k vs the host block they replace:
                 * ids must match q4_softmax+q4_topk(...,true) exactly, wts
                 * within expf-ulp tolerance, and the (a_tok, rsrc, rw)
                 * assignment rows must be a hole-free per-expert
                 * partitioning that maps back to the same (token, weight).
                 * n_tok=17 with gsz=8 covers the partial tail group and
                 * n_tok%4 != 0 (route_pf's 4-token blocks). */
                {
                    const uint32_t n_tok = 17, n_exp = 512, topk = 10,
                                   gsz = 8;
                    const uint32_t n_g = (n_tok + gsz - 1) / gsz;
                    const size_t nn = (size_t)n_tok * topk;
                    float *hlog = malloc((size_t)n_tok * n_exp * 4);
                    float *lref = malloc((size_t)n_tok * n_exp * 4);
                    int32_t *h_ids = malloc(nn * 4), *h_ids2 = malloc(nn * 4);
                    float *h_wts = malloc(nn * 4), *h_wts2 = malloc(nn * 4);
                    uint32_t *h_ek = malloc((size_t)n_g * 512 * 4);
                    uint32_t *h_eb = malloc((size_t)n_g * 512 * 4);
                    int32_t *h_at = malloc((size_t)n_g * gsz * topk * 4);
                    int32_t *h_rs = malloc(nn * 4);
                    float *h_rw = malloc(nn * 4);
                    int32_t *h_stg = malloc((size_t)n_g * 512 * 4);
                    uint32_t *h_stn = malloc(n_g * 4);
                    bool ok = hlog && lref && h_ids && h_ids2 && h_wts &&
                              h_wts2 && h_ek && h_eb && h_at && h_rs &&
                              h_rw && h_stg && h_stn;
                    for (uint32_t i = 0; ok && i < n_tok * n_exp; i++)
                        hlog[i] = ((int)((i * 2654435761u) >> 16) - 32768) *
                                  0.00048828125f;
                    /* host reference */
                    if (ok) {
                        memcpy(lref, hlog, (size_t)n_tok * n_exp * 4);
                        for (uint32_t t = 0; t < n_tok; t++) {
                            q4_softmax(lref + t * n_exp, n_exp);
                            q4_topk(lref + t * n_exp, n_exp, topk,
                                    h_ids + t * topk, h_wts + t * topk,
                                    true);
                        }
                        memset(h_ek, 0, (size_t)n_g * 512 * 4);
                        for (uint32_t t = 0; t < n_tok; t++)
                            for (uint32_t k = 0; k < topk; k++) {
                                int32_t e = h_ids[t * topk + k];
                                if (e >= 0) h_ek[(t / gsz) * 512 + e]++;
                            }
                        memset(h_stn, 0, n_g * 4);
                        for (uint32_t gno = 0; gno < n_g; gno++) {
                            uint32_t b = 0;
                            for (uint32_t e = 0; e < n_exp; e++) {
                                h_eb[gno * 512 + e] = b;
                                b += h_ek[gno * 512 + e];
                                if (h_ek[gno * 512 + e])
                                    h_stg[gno * 512 + h_stn[gno]++] =
                                        (int32_t)e;
                            }
                        }
                    }
                    /* device path */
                    float *d_rlog = q4_hip_malloc((size_t)n_tok * n_exp * 4);
                    int32_t *d_ids = q4_hip_malloc(nn * 4);
                    float *d_wts = q4_hip_malloc(nn * 4);
                    int32_t *d_atok =
                        q4_hip_malloc((size_t)n_g * gsz * topk * 4);
                    int32_t *d_rsrc = q4_hip_malloc(nn * 4);
                    float *d_rw = q4_hip_malloc(nn * 4);
                    /* ek|stg|stgn share one contiguous block — the layout
                     * q4_moe_layer_dev_n's single D2H relies on. */
                    uint32_t *d_blk =
                        q4_hip_malloc((size_t)n_g * 1025 * 4);
                    uint32_t *d_ek = d_blk;
                    int32_t *d_stg = (int32_t *)(d_blk + (size_t)n_g * 512);
                    uint32_t *d_stn = d_blk + 2 * (size_t)n_g * 512;
                    uint32_t *u32r = malloc((size_t)n_g * 1025 * 4);
                    ok = ok && d_rlog && d_ids && d_wts && d_atok &&
                         d_rsrc && d_rw && d_blk && u32r;
                    ok = ok &&
                         q4_hip_h2d(d_rlog, hlog,
                                    (size_t)n_tok * n_exp * 4) &&
                         q4_hip_route_pf(d_rlog, d_ids, d_wts, n_tok,
                                         n_exp, topk) &&
                         q4_hip_assign_pf(d_ids, d_wts, d_atok, d_rsrc,
                                          d_rw, d_stg, d_stn, d_ek, gsz,
                                          topk, n_exp, n_tok) &&
                         q4_hip_d2h(h_ids2, d_ids, nn * 4) &&
                         q4_hip_d2h(h_wts2, d_wts, nn * 4) &&
                         q4_hip_d2h(h_rs, d_rsrc, nn * 4) &&
                         q4_hip_d2h(h_rw, d_rw, nn * 4) &&
                         q4_hip_d2h(h_at, d_atok,
                                    (size_t)n_g * gsz * topk * 4) &&
                         q4_hip_d2h(u32r, d_ek, (size_t)n_g * 1025 * 4);
                    if (!ok) return fail("pf route run");
                    /* ids exact, wts within expf-ulp distance */
                    for (uint32_t i = 0; i < nn; i++) {
                        if (h_ids[i] != h_ids2[i])
                            return fail("pf ids");
                        float d = fabsf(h_wts[i] - h_wts2[i]);
                        if (!(d <= 1e-6f * (1.f + fabsf(h_wts[i]))))
                            return fail("pf wts");
                    }
                    /* ek + stg + stgn match (u32r: ek | stg | stgn) */
                    for (uint32_t i = 0; i < n_g * 512; i++)
                        if (u32r[i] != h_ek[i]) return fail("pf ek");
                    for (uint32_t gno = 0; gno < n_g; gno++) {
                        if (u32r[2 * n_g * 512 + gno] != h_stn[gno])
                            return fail("pf stgn");
                        for (uint32_t i = 0; i < h_stn[gno]; i++)
                            if ((int32_t)u32r[n_g * 512 + gno * 512 + i] !=
                                h_stg[gno * 512 + i])
                                return fail("pf stg");
                    }
                    /* assignment rows: every pick's rsrc lands in its
                     * expert's range, a_tok maps back to the token, rw is
                     * the pick weight; each row in the group's used range
                     * is claimed exactly once. */
                    {
                        uint8_t *seen =
                            calloc((size_t)n_g * gsz * topk, 1);
                        if (!seen) return fail("pf assign alloc");
                        for (uint32_t t = 0; t < n_tok; t++) {
                            const uint32_t gno = t / gsz;
                            const uint32_t seg = gno * gsz * topk;
                            for (uint32_t k = 0; k < topk; k++) {
                                int32_t e = h_ids[t * topk + k];
                                int32_t p = h_rs[t * topk + k];
                                if (e < 0) {
                                    if (p != -1 || h_rw[t * topk + k])
                                        return fail("pf rsrc pad");
                                    continue;
                                }
                                uint32_t base = h_eb[gno * 512 + e],
                                         cnt = h_ek[gno * 512 + e];
                                if (p < (int32_t)base ||
                                    (uint32_t)p >= base + cnt)
                                    return fail("pf rsrc range");
                                if (h_at[seg + p] != (int32_t)t)
                                    return fail("pf atok backmap");
                                /* rw is a verbatim copy of the kernel's
                                 * d_wts — compare to h_wts2, not the
                                 * host's (1ulp-different) h_wts. */
                                if (h_rw[t * topk + k] !=
                                    h_wts2[t * topk + k])
                                    return fail("pf rw");
                                if (seen[seg + p])
                                    return fail("pf rsrc dup");
                                seen[seg + p] = 1;
                            }
                        }
                        /* hole-free: each group's used rows == sum(ek) */
                        for (uint32_t gno = 0; gno < n_g; gno++) {
                            uint32_t tot = 0;
                            for (uint32_t e = 0; e < n_exp; e++)
                                tot += h_ek[gno * 512 + e];
                            for (uint32_t p = 0; p < tot; p++)
                                if (!seen[gno * gsz * topk + p])
                                    return fail("pf rsrc hole");
                        }
                        free(seen);
                    }
                    free(hlog); free(lref); free(h_ids); free(h_ids2);
                    free(h_wts); free(h_wts2); free(h_ek); free(h_eb);
                    free(h_at); free(h_rs); free(h_rw); free(h_stg);
                    free(h_stn); free(u32r);
                    q4_hip_free(d_rlog); q4_hip_free(d_ids);
                    q4_hip_free(d_wts); q4_hip_free(d_atok);
                    q4_hip_free(d_rsrc); q4_hip_free(d_rw);
                    q4_hip_free(d_blk); /* ek|stg|stgn aliases */
                }
                /* route_pf writeback contract: topk up to 64 must land —
                 * the shared staging is 64-deep but a warp has 32 lanes,
                 * so picks 32..63 only reach d_ids/d_wts via the strided
                 * writeback loop.  topk=64 forces the second stride for
                 * every lane; ids must match q4_topk exactly. */
                {
                    const uint32_t n_tok = 5, n_exp = 512, topk = 64;
                    const size_t nn = (size_t)n_tok * topk;
                    float hlog[5 * 512];
                    int32_t h_ids[320], h_ids2[320];
                    float h_wts[320], h_wts2[320], lref[512];
                    for (uint32_t i = 0; i < n_tok * n_exp; i++)
                        hlog[i] = ((int)((i * 2654435761u) >> 16) - 32768) *
                                  0.00048828125f;
                    for (uint32_t t = 0; t < n_tok; t++) {
                        memcpy(lref, hlog + t * n_exp, sizeof(lref));
                        q4_softmax(lref, n_exp);
                        q4_topk(lref, n_exp, topk, h_ids + t * topk,
                                h_wts + t * topk, true);
                    }
                    float *d_rlog = q4_hip_malloc(n_tok * n_exp * 4);
                    int32_t *d_ids = q4_hip_malloc(nn * 4);
                    float *d_wts = q4_hip_malloc(nn * 4);
                    if (!d_rlog || !d_ids || !d_wts)
                        return fail("pf64 alloc");
                    /* Poison the device buffers: picks that stay
                     * unwritten must fail loudly, not match by luck. */
                    int32_t poison[320];
                    memset(poison, 0xA5, sizeof(poison));
                    bool ok = q4_hip_h2d(d_rlog, hlog, n_tok * n_exp * 4) &&
                              q4_hip_h2d(d_ids, poison, nn * 4) &&
                              q4_hip_h2d(d_wts, poison, nn * 4);
                    ok = ok && q4_hip_route_pf(d_rlog, d_ids, d_wts, n_tok,
                                               n_exp, topk) &&
                         q4_hip_d2h(h_ids2, d_ids, nn * 4) &&
                         q4_hip_d2h(h_wts2, d_wts, nn * 4);
                    if (!ok) return fail("pf64 run");
                    for (uint32_t i = 0; i < nn; i++) {
                        if (h_ids[i] != h_ids2[i])
                            return fail("pf64 ids");
                        float d = fabsf(h_wts[i] - h_wts2[i]);
                        if (!(d <= 1e-6f * (1.f + fabsf(h_wts[i]))))
                            return fail("pf64 wts");
                    }
                    /* Contract: topk 0 or >64 rejected by the wrapper. */
                    if (q4_hip_route_pf(d_rlog, d_ids, d_wts, n_tok,
                                        n_exp, 0) ||
                        q4_hip_route_pf(d_rlog, d_ids, d_wts, n_tok,
                                        n_exp, 65))
                        return fail("pf64 topk bounds");
                    q4_hip_free(d_rlog); q4_hip_free(d_ids);
                    q4_hip_free(d_wts);
                }
                /* cpx_wait inside a captured graph bakes its ctx pointer
                 * into the host node — it must not come from the recycled
                 * eager ring.  Capture one wait, wrap the ring with >1024
                 * eager waits, then replay with the ring's last pair
                 * unsatisfiable: an aliased ctx would spin on the wrong
                 * done/expect until the 45 s timeout trips. */
                {
                    volatile unsigned long long d1 = 1, e1 = 1;
                    volatile unsigned long long d2 = 1, e2 = 1;
                    q4_hip_sync();
                    if (!q4_hip_graph_begin()) return fail("cpx cap begin");
                    if (!q4_hip_cpx_wait((void *)&d1, (void *)&e1, NULL))
                        return fail("cpx cap wait");
                    void *exec = q4_hip_graph_end();
                    if (!exec) return fail("cpx cap end");
                    for (int i = 0; i < 1200; i++)
                        if (!q4_hip_cpx_wait((void *)&d2, (void *)&e2, NULL))
                            return fail("cpx eager wait");
                    q4_hip_sync();
                    /* Ring's hot slots now hold (&d2,&e2): poison them so
                     * an aliased replay waits forever (45 s timeout). */
                    d2 = 0; e2 = ~0ull;
                    if (!q4_hip_graph_launch(exec))
                        return fail("cpx graph launch");
                    q4_hip_sync();
                    if (q4_hip_cpx_wait_timeouts())
                        return fail("cpx graph ctx aliased");
                    q4_hip_graph_free(exec);
                }
                /* Device top-k matches q4_topk, including ties (strict >,
                 * the earlier index stays). A replayed graph must read the
                 * logit row at launch time, not the values from capture. */
                {
                    float small[5] = {1.f, 5.f, 5.f, 3.f, 5.f};
                    int32_t ids[3], got[3];
                    float w[3], gw[3];
                    q4_topk(small, 5, 3, ids, w, false);
                    float *d = q4_hip_malloc(5 * sizeof(float));
                    if (!d || !q4_hip_h2d(d, small, 5 * sizeof(float)) ||
                        !q4_hip_topk(d, 5, 3, got, gw))
                        return fail("topk small run");
                    for (int ti = 0; ti < 3; ti++)
                        if (ids[ti] != got[ti] || gw[ti] != w[ti])
                            return fail("topk small");
                    const uint32_t ntk = 4096, ktk = 20;
                    float *row = malloc((size_t)ntk * sizeof(float));
                    int32_t *hid = malloc((size_t)ktk * sizeof(int32_t));
                    int32_t *gid = malloc((size_t)ktk * sizeof(int32_t));
                    float *hw = malloc((size_t)ktk * sizeof(float));
                    float *gw2 = malloc((size_t)ktk * sizeof(float));
                    float *dl = q4_hip_malloc((size_t)ntk * sizeof(float));
                    if (!row || !hid || !gid || !hw || !gw2 || !dl)
                        return fail("topk large alloc");
                    for (uint32_t ti = 0; ti < ntk; ti++)
                        row[ti] = (float)((ti * 17u) % 50u);
                    row[10] = 1000.f;
                    row[4000] = 1000.f;
                    q4_topk(row, ntk, ktk, hid, hw, false);
                    if (!q4_hip_h2d(dl, row, (size_t)ntk * sizeof(float)) ||
                        !q4_hip_topk(dl, ntk, ktk, gid, gw2))
                        return fail("topk large run");
                    for (uint32_t ti = 0; ti < ktk; ti++)
                        if (hid[ti] != gid[ti] || hw[ti] != gw2[ti])
                            return fail("topk large");
                    /* Head-sized rows (full and draft_vocab subset), coarse
                     * values so ties cross block boundaries. */
                    {
                        const uint32_t vn[2] = {248320u, 106299u};
                        const uint32_t vk[3] = {1u, 20u, 128u};
                        float *vr = malloc((size_t)vn[0] * sizeof(float));
                        float *dv = q4_hip_malloc((size_t)vn[0] * sizeof(float));
                        int32_t vh[128], vg[128];
                        float vw[128], vgw[128];
                        if (!vr || !dv) return fail("topk vocab alloc");
                        for (int ni = 0; ni < 2; ni++) {
                            unsigned st = 12345u + (unsigned)ni;
                            for (uint32_t ti = 0; ti < vn[ni]; ti++) {
                                st = st * 1103515245u + 12345u;
                                vr[ti] = (float)((st >> 16) % 2000u) * 0.25f;
                            }
                            if (!q4_hip_h2d(dv, vr, (size_t)vn[ni] * sizeof(float)))
                                return fail("topk vocab h2d");
                            for (int ki = 0; ki < 3; ki++) {
                                q4_topk(vr, vn[ni], vk[ki], vh, vw, false);
                                if (!q4_hip_topk(dv, vn[ni], vk[ki], vg, vgw))
                                    return fail("topk vocab run");
                                for (uint32_t ti = 0; ti < vk[ki]; ti++)
                                    if (vh[ti] != vg[ti] || vw[ti] != vgw[ti]) {
                                        fprintf(stderr,
                                                "topk vocab n=%u k=%u i=%u host %d/%g dev %d/%g\n",
                                                vn[ni], vk[ki], ti, vh[ti], vw[ti],
                                                vg[ti], vgw[ti]);
                                        return fail("topk vocab");
                                    }
                            }
                        }
                        q4_hip_free(dv);
                        free(vr);
                    }
                    /* Row gather (the draft_vocab subset head). */
                    {
                        uint32_t gs[5 * 525], gd[3 * 525];
                        int32_t gi[3] = {4, 0, 2};
                        for (uint32_t ti = 0; ti < 5u * 525u; ti++) gs[ti] = ti * 7u + 1u;
                        uint8_t *dgs = q4_hip_malloc(sizeof gs);
                        uint8_t *dgd = q4_hip_malloc(sizeof gd);
                        int32_t *dgi = q4_hip_malloc(sizeof gi);
                        if (!dgs || !dgd || !dgi || !q4_hip_h2d(dgs, gs, sizeof gs) ||
                            !q4_hip_h2d(dgi, gi, sizeof gi) ||
                            !q4_hip_gather_rows(dgd, dgs, 525u * 4u, dgi, 3) ||
                            !q4_hip_stream_sync() || !q4_hip_d2h(gd, dgd, sizeof gd))
                            return fail("gather rows run");
                        for (uint32_t r = 0; r < 3; r++)
                            for (uint32_t ti = 0; ti < 525u; ti++)
                                if (gd[r * 525u + ti] != gs[(uint32_t)gi[r] * 525u + ti])
                                    return fail("gather rows");
                        q4_hip_free(dgs);
                        q4_hip_free(dgd);
                        q4_hip_free(dgi);
                    }
                    float seq[5] = {9.f, 0.f, 0.f, 8.f, 1.f};
                    q4_topk(seq, 5, 3, ids, w, false);
                    /* Capture while d still holds the previous row, then
                     * overwrite it. A baked copy would return the old ids. */
                    if (!q4_hip_graph_begin()) return fail("topk graph begin");
                    if (!q4_hip_topk_launch(d, 5, 3))
                        return fail("topk graph record");
                    void *ex = q4_hip_graph_end();
                    if (!ex) return fail("topk graph end");
                    if (!q4_hip_h2d(d, seq, 5 * sizeof(float)) ||
                        !q4_hip_graph_launch(ex) || !q4_hip_stream_sync())
                        return fail("topk graph replay");
                    q4_hip_topk_read(got, gw, 3);
                    for (int ti = 0; ti < 3; ti++)
                        if (ids[ti] != got[ti] || gw[ti] != w[ti])
                            return fail("topk graph values");
                    q4_hip_graph_free(ex);
                    float src[32], dst[32];
                    for (int ti = 0; ti < 32; ti++) src[ti] = (float)(ti + 1);
                    float *d0 = q4_hip_malloc(32 * sizeof(float));
                    float *d1 = q4_hip_malloc(32 * sizeof(float));
                    if (!d0 || !d1 ||
                        !q4_hip_h2d(d0, src, 32 * sizeof(float)) ||
                        !q4_hip_fill(d1, -1.f, 32) ||
                        !q4_hip_copy_async(d1, d0, 32) ||
                        !q4_hip_stream_sync() ||
                        !q4_hip_d2h(dst, d1, 32 * sizeof(float)))
                        return fail("copy_async run");
                    for (int ti = 0; ti < 32; ti++)
                        if (dst[ti] != src[ti]) return fail("copy_async");
                    q4_hip_free(d0);
                    q4_hip_free(d1);
                    q4_hip_free(dl);
                    q4_hip_free(d);
                    free(row);
                    free(hid);
                    free(gid);
                    free(hw);
                    free(gw2);
                }
            }
        }
        {
            float logits[5] = {0.1f, 0.9f, 0.2f, 0.8f, 0.3f};
            int32_t ids[2];
            float w[2];
            q4_topk(logits, 5, 2, ids, w, false);
            if (ids[0] != 1 || ids[1] != 3) return fail("topk ids");
        }
        {
            float lg[8] = {0.1f, 3.f, 1.2f, 3.f, 0.4f, 2.2f, 0.f, 1.5f};
            int32_t a[8], bids[8], sel[8];
            float pa[8], pb[8], sw[8];
            int na = q4_dist_build(lg, 8, 0.8f, 4, 0.9f, a, pa);
            q4_topk(lg, 8, 4, sel, sw, false);
            int ndist = q4_dist_from_topk(sel, sw, 4, 0.8f, 0.9f, bids, pb);
            if (na < 1 || na != ndist) return fail("dist from topk n");
            for (int ti = 0; ti < na; ti++) {
                if (a[ti] != bids[ti]) return fail("dist from topk id");
                if (fabsf(pa[ti] - pb[ti]) > 1e-6f)
                    return fail("dist from topk p");
            }
            int nc = q4_dist_build(lg, 8, 1.f, 4, 0.2f, a, pa);
            if (nc != 1 || a[0] != 1) return fail("dist topp cut");
            if (q4_dist_from_topk(NULL, sw, 4, 1.f, 1.f, bids, pb) != 0)
                return fail("dist from topk null");
        }
        {
            /* Speculative step law: a draft drawn from q, accepted by
             * q4_spec_accept or replaced by its residual sample, must
             * emit the target p — including draft-only and target-only
             * ids (the lists are truncated independently). */
            const int32_t tid[4] = {7, 3, 9, 1};
            const float tp[4] = {0.45f, 0.30f, 0.15f, 0.10f};
            const int32_t did[3] = {3, 7, 5};
            const float dp[3] = {0.55f, 0.35f, 0.10f};
            const int nt = 400000;
            int cnt[10] = {0};
            unsigned rng = 12345u;
            for (int ti = 0; ti < nt; ti++) {
                int32_t d = q4_dist_sample(did, dp, 3, &rng), rs = -1;
                int32_t x = q4_spec_accept(tid, tp, 4, did, dp, 3, d, &rng,
                                           &rs)
                                ? d
                                : rs;
                if (x < 0 || x > 9) return fail("spec law id");
                cnt[x]++;
            }
            for (int a = 0; a < 10; a++) {
                const float want = q4_dist_prob(tid, tp, 4, a);
                if (fabsf((float)cnt[a] / nt - want) > 0.005f)
                    return fail("spec law p");
            }
        }
        {
            /* Presence penalty: the top-k over k + extra candidates, kept
             * to k after the penalty, equals the distribution over the
             * penalized full row; mark / rewind restore the history. */
            enum { NV = 300 };
            float lg[NV], pl[NV];
            for (int i = 0; i < NV; i++)
                lg[i] = sinf((float)i * 1.7f) * 3.f + (i % 7 == 0 ? 2.f : 0.f);
            q4_pen_begin(1.5f, NV);
            const int32_t hist[6] = {0, 7, 14, 21, 7, 28};
            for (int i = 0; i < 6; i++) q4_pen_push(hist[i]);
            const int mk = q4_pen_mark();
            q4_pen_push(35);
            memcpy(pl, lg, sizeof(lg));
            const int32_t pen[6] = {0, 7, 14, 21, 28, 35};
            for (int i = 0; i < 6; i++) pl[pen[i]] -= 1.5f;
            int32_t want[Q4_DIST_MAX], got[Q4_DIST_MAX], sel[Q4_DIST_MAX];
            float pw[Q4_DIST_MAX], pg[Q4_DIST_MAX], sv[Q4_DIST_MAX];
            q4_pen_begin(0.f, NV); /* reference: no penalty, penalized row */
            const int nw = q4_dist_build(pl, NV, 0.9f, 20, 0.95f, want, pw);
            q4_pen_begin(1.5f, NV);
            for (int i = 0; i < 6; i++) q4_pen_push(hist[i]);
            q4_pen_push(35);
            const int ng = q4_dist_build(lg, NV, 0.9f, 20, 0.95f, got, pg);
            const int kc = 20 + q4_pen_extra();
            q4_topk(lg, NV, (uint32_t)kc, sel, sv, false);
            int32_t g2[Q4_DIST_MAX];
            float p2[Q4_DIST_MAX];
            const int n2 = q4_dist_from_topk_k(sel, sv, kc, 20, 0.9f, 0.95f,
                                               g2, p2);
            if (nw < 1 || ng != nw || n2 != nw) return fail("penalty dist n");
            for (int i = 0; i < nw; i++)
                if (got[i] != want[i] || g2[i] != want[i] ||
                    fabsf(pg[i] - pw[i]) > 1e-6f || fabsf(p2[i] - pw[i]) > 1e-6f)
                    return fail("penalty dist");
            q4_pen_rewind(mk); /* 35 is no longer penalized, 7 still is */
            pl[35] += 1.5f;
            q4_pen_begin(0.f, NV);
            const int nr = q4_dist_build(pl, NV, 0.9f, 20, 0.95f, want, pw);
            q4_pen_begin(1.5f, NV);
            for (int i = 0; i < 6; i++) q4_pen_push(hist[i]);
            q4_pen_push(35);
            q4_pen_rewind(q4_pen_mark() - 1);
            const int nr2 = q4_dist_build(lg, NV, 0.9f, 20, 0.95f, got, pg);
            if (nr != nr2) return fail("penalty rewind n");
            for (int i = 0; i < nr; i++)
                if (got[i] != want[i] || fabsf(pg[i] - pw[i]) > 1e-6f)
                    return fail("penalty rewind");
            q4_pen_begin(0.f, NV);
        }
        {
            const char *xml =
                "ok then\n"
                "<tool_call>\n"
                "<function=bash>\n"
                "<parameter=command>\n"
                "ls -la /tmp\n"
                "</parameter>\n"
                "<parameter=timeout>\n"
                "30\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n";
            char *prefix = NULL, **names = NULL, **args = NULL;
            int n = 0;
            if (q4_qwen_parse_tool_calls(xml, &prefix, &names, &args, &n) != 1)
                return fail("parse xml n");
            if (!prefix || strcmp(prefix, "ok then") != 0) return fail("parse xml prefix");
            if (!names || strcmp(names[0], "bash") != 0) return fail("parse xml name");
            if (!args || !strstr(args[0], "\"command\":\"ls -la /tmp\""))
                return fail("parse xml command");
            if (!strstr(args[0], "\"timeout\":30")) return fail("parse xml timeout");
            q4_qwen_free_tool_calls(names, args, n);
            free(prefix);
            const char *hermes =
                "<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"a.c\"}}\n"
                "</tool_call>";
            prefix = NULL; names = NULL; args = NULL; n = 0;
            if (q4_qwen_parse_tool_calls(hermes, &prefix, &names, &args, &n) != 1)
                return fail("parse hermes n");
            if (strcmp(names[0], "read") != 0) return fail("parse hermes name");
            if (!strstr(args[0], "\"path\":\"a.c\"")) return fail("parse hermes args");
            q4_qwen_free_tool_calls(names, args, n);
            free(prefix);
            char *pre = q4_qwen_tools_preamble(
                "[{\"type\":\"function\",\"function\":{\"name\":\"bash\"}}]", 1);
            if (!pre || !strstr(pre, "<tools>") || !strstr(pre, "\"name\":\"bash\"") ||
                !strstr(pre, "<tool_call>"))
                return fail("tools preamble");
            free(pre);
            {
                char fat[4096];
                memset(fat, 'x', 3000);
                fat[3000] = 0;
                char req[4500];
                snprintf(req, sizeof(req),
                         "[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
                         "\"description\":\"%s\",\"parameters\":{\"type\":\"object\","
                         "\"properties\":{\"command\":{\"type\":\"string\","
                         "\"description\":\"run a shell command\"}},"
                         "\"required\":[\"command\"]}}}]",
                         fat);
                char *c = q4_qwen_compact_tools_json(req);
                if (!c || strstr(c, "description") || strstr(c, "xxx") ||
                    !strstr(c, "\"name\":\"bash\"") || !strstr(c, "\"command\""))
                    return fail("compact tools json");
                if (strlen(c) > 400) return fail("compact tools still fat");
                char *p2 = q4_qwen_tools_preamble(req, 1);
                if (!p2 || strstr(p2, fat) || strlen(p2) > 800)
                    return fail("preamble still fat");
                free(c);
                free(p2);
                char *ord = q4_qwen_compact_tools_json(
                    "[{\"function\":{\"name\":\"write\"}},{\"function\":{\"name\":\"bash\"}}]");
                if (!ord) return fail("compact sort");
                const char *wb = strstr(ord, "\"name\":\"write\"");
                const char *bb = strstr(ord, "\"name\":\"bash\"");
                if (!wb || !bb || bb > wb) return fail("compact sort order");
                free(ord);
            }
        }
        {
            const char *w = "こんにちは！";
            if (q4_utf8_safe_end(w, 3) != 3) return fail("utf8 end of こ");
            if (q4_utf8_safe_end(w, 4) != 3) return fail("utf8 cut inside ん");
            if (q4_utf8_safe_end(w, 6) != 6) return fail("utf8 end of こん");
            if (q4_utf8_safe_end(w, 15) != 15) return fail("utf8 whole greeting");
            if (q4_utf8_safe_end("abc", 2) != 2) return fail("utf8 ascii");
        }
        {
            /* Planar gate/up is parity order: up row 0 is slot 1, not the
             * row after every gate row. A slot-order pack dequants up as 0. */
            uint8_t *src = calloc(1, Q4_Q2_PACK_BLOB);
            uint8_t *dst = calloc(1, Q4_Q2_PACK_BLOB);
            if (!src || !dst) return fail("q2 planar alloc");
            if (q4_q2_planar_to_ggml(NULL, dst)) return fail("q2 planar null");
            uint16_t one = 0x3C00;
            memcpy(src + 1228800, &one, 2);
            memset(src, 0x1B, 16);
            memset(src + 640, 0xE4, 16);
            memcpy(src + 1228800 + 80, &one, 2);
            memcpy(src + 1331200 + 7 * 20 + 4, &one, 2);
            memset(src + 819200 + 7 * 160 + 32, 0x1B, 16);
            if (!q4_q2_planar_to_ggml(src, dst)) return fail("q2 planar convert");
            float row[Q4_Q2_PACK_H];
            if (!q4_dequant_row(Q4_T_Q2_0, dst, Q4_Q2_PACK_H, row))
                return fail("q2 dequant gate");
            if (row[0] != 2.f || row[1] != 1.f || row[2] != 0.f || row[3] != -1.f)
                return fail("q2 gate values");
            if (!q4_dequant_row(Q4_T_Q2_0, dst + 460800, Q4_Q2_PACK_H, row))
                return fail("q2 dequant up");
            /* 0xE4 is LSB-first quants 0,1,2,3 → -1,0,1,2 at scale 1. */
            if (row[0] != -1.f || row[1] != 0.f || row[2] != 1.f || row[3] != 2.f)
                return fail("q2 up interleave");
            if (!q4_dequant_row(Q4_T_Q2_0, dst + 921600 + 7 * 180, Q4_Q2_PACK_FF,
                                row))
                return fail("q2 dequant down");
            if (row[128] != 2.f || row[129] != 1.f || row[130] != 0.f ||
                row[131] != -1.f)
                return fail("q2 down plane");
            if (!q4_dequant_row(Q4_T_Q2_0, dst + 720, Q4_Q2_PACK_H, row))
                return fail("q2 dequant gate1");
            if (row[0] != 0.f || row[63] != 0.f) return fail("q2 gate row1");
            free(src);
            free(dst);
        }
        if (q4_cpx_selftest()) return fail("cpx q8 kernels");
        if (q4_hip_init()) {
            /* Verify window: the tiled down kernels + warp-built plan vs
             * the warp-per-row kernels + serial plan.  Same per-row dot
             * order, so only the atomicAdd order differs. */
            const uint32_t gts[3] = {Q4_T_IQ3_S, Q4_T_IQ2_S, Q4_T_IQ3_XXS};
            const uint32_t dts[3] = {Q4_T_IQ4_NL, Q4_T_Q2_0, Q4_T_IQ4_NL};
            for (int ci = 0; ci < 6; ci++) {
                const uint32_t gt = gts[ci % 3], dt = dts[ci % 3];
                const int tile_v = 1 + ci / 3;
                const uint32_t in_dim = 2560, n_ff = 640, n_embd = 2560;
                const uint32_t ns = 16, nt = 4;
                const uint64_t gb = gt == Q4_T_IQ3_S ? 110
                                    : gt == Q4_T_IQ3_XXS ? 98 : 82;
                const uint64_t grb = (in_dim / 256) * gb;
                const uint64_t drb = dt == Q4_T_Q2_0 ? (n_ff / 64) * 18
                                                     : (n_ff / 32) * 18;
                const uint64_t og = 0, ou = n_ff * grb,
                               od = 2 * n_ff * grb;
                const uint64_t stride = (od + n_embd * drb + 4095) & ~4095ull;
                uint8_t *w = malloc(ns * stride);
                float *x = malloc((size_t)nt * in_dim * 4);
                float *y0 = malloc((size_t)nt * n_embd * 4);
                float *y1 = malloc((size_t)nt * n_embd * 4);
                if (!w || !x || !y0 || !y1) return fail("win alloc");
                uint64_t rng = 0x2545F4914F6CDD1Dull + (uint64_t)ci;
                for (uint64_t i = 0; i < ns * stride; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    w[i] = (uint8_t)rng;
                }
                const uint16_t ds = 0x1266;
                for (uint32_t e = 0; e < ns; e++) {
                    uint8_t *we = w + (uint64_t)e * stride;
                    for (uint64_t r = 0; r < 2ull * n_ff; r++)
                        for (uint64_t b = 0; b < in_dim / 256; b++)
                            memcpy(we + r * grb + b * gb, &ds, 2);
                    const uint64_t dbs = 18;
                    for (uint64_t r = 0; r < n_embd; r++)
                        for (uint64_t b = 0; b < drb / dbs; b++)
                            memcpy(we + od + r * drb + b * dbs, &ds, 2);
                }
                for (uint32_t i = 0; i < nt * in_dim; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    x[i] = ((int32_t)(rng % 41) - 20) * 0.05f;
                }
                int32_t es[4 * 64], nh[4];
                float ew[4 * 64];
                for (uint32_t t = 0; t < nt; t++) {
                    nh[t] = 3 + (int32_t)t * 2;   /* 3, 5, 7, 9 hits */
                    for (int32_t i = 0; i < nh[t]; i++) {
                        es[t * 64 + i] = (int32_t)((t * 3 + (uint32_t)i * 5) % ns);
                        ew[t * 64 + i] = 0.1f + 0.05f * (float)i;
                    }
                }
                uint8_t *dw = q4_hip_malloc(ns * stride);
                float *dx = q4_hip_malloc((size_t)nt * in_dim * 4);
                float *dh = q4_hip_malloc((size_t)64 * n_ff * 4);
                float *dy = q4_hip_malloc((size_t)nt * n_embd * 4);
                int32_t *des = q4_hip_malloc(sizeof(es));
                float *dew = q4_hip_malloc(sizeof(ew));
                int32_t *dnh = q4_hip_malloc(sizeof(nh));
                int ok = dw && dx && dh && dy && des && dew && dnh &&
                         q4_hip_h2d(dw, w, ns * stride) &&
                         q4_hip_h2d(dx, x, (size_t)nt * in_dim * 4) &&
                         q4_hip_h2d(des, es, sizeof(es)) &&
                         q4_hip_h2d(dew, ew, sizeof(ew)) &&
                         q4_hip_h2d(dnh, nh, sizeof(nh)) &&
                         q4_hip_moe_win_ok(gt, gt, dt, in_dim, n_ff, nt);
                for (int tv = 0; ok && tv < 2; tv++) {
                    q4_hip_win_tile_set(tv ? tile_v : 0);
                    ok = q4_hip_fill(dy, 0.f, (uint64_t)nt * n_embd) &&
                         q4_hip_moe_exec_win(dw, stride, des, dew, dnh, nt,
                                             gt, og, gt, ou, dt, od, dx, dh,
                                             dy, n_ff, in_dim, n_embd) &&
                         q4_hip_d2h(tv ? y1 : y0, dy,
                                    (size_t)nt * n_embd * 4);
                }
                q4_hip_win_tile_set(-1);
                if (!ok) return fail("win gpu run");
                double num = 0.0, den = 0.0;
                for (uint32_t i = 0; i < nt * n_embd; i++) {
                    num += ((double)y1[i] - y0[i]) * ((double)y1[i] - y0[i]);
                    den += (double)y0[i] * y0[i];
                }
                double rel = den > 0.0 ? sqrt(num / den) : 1.0;
                printf("win tile %d %s/%s: rel L2 %.2e\n", tile_v,
                       q4_type_name(gt), q4_type_name(dt), rel);
                q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dh);
                q4_hip_free(dy); q4_hip_free(des); q4_hip_free(dew);
                q4_hip_free(dnh);
                free(w); free(x); free(y0); free(y1);
                if (!(den > 0.0) || !(rel < 1e-5)) return fail("win tile");
            }
        }
        if (q4_hip_init()) {
            /* fused shared-expert gate: y += sigmoid(w . x) * sh per row */
            enum { N = 2560, T = 4 };
            static float w[N], x[T * N], sh[T * N], y[T * N], got[T * N];
            uint64_t rng = 0x9E3779B97F4A7C15ull;
            for (int i = 0; i < N; i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                w[i] = ((int)(rng % 201) - 100) * 1e-3f;
            }
            for (int i = 0; i < T * N; i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                x[i] = ((int)(rng % 201) - 100) * 1e-2f;
                sh[i] = ((int)((rng >> 20) % 201) - 100) * 1e-2f;
                y[i] = ((int)((rng >> 40) % 201) - 100) * 1e-2f;
            }
            float *dw = q4_hip_malloc(sizeof(w)), *dx = q4_hip_malloc(sizeof(x));
            float *dsh = q4_hip_malloc(sizeof(sh)), *dy = q4_hip_malloc(sizeof(y));
            if (!dw || !dx || !dsh || !dy || !q4_hip_h2d(dw, w, sizeof(w)) ||
                !q4_hip_h2d(dx, x, sizeof(x)) || !q4_hip_h2d(dsh, sh, sizeof(sh)) ||
                !q4_hip_h2d(dy, y, sizeof(y)) ||
                !q4_hip_sgate_axpy(dy, dsh, dw, dx, N, T) ||
                !q4_hip_d2h(got, dy, sizeof(got)))
                return fail("sgate gpu run");
            double md = 0.0;
            for (int t = 0; t < T; t++) {
                double d = 0.0;
                for (int i = 0; i < N; i++) d += (double)w[i] * x[t * N + i];
                const double sg = 1.0 / (1.0 + exp(-d));
                for (int i = 0; i < N; i++) {
                    const double ref = y[t * N + i] + sg * sh[t * N + i];
                    const double e = fabs(ref - got[t * N + i]);
                    if (e > md) md = e;
                }
            }
            printf("sgate fuse: max abs err %.2e\n", md);
            q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dsh); q4_hip_free(dy);
            if (!(md < 1e-5)) return fail("sgate fuse");
        }
        if (q4_hip_init()) {
            /* int8-activation n-row GEMV vs a CPU dequant + double dot */
            q4_mmvq_warm();
            q4_mmvq_set(1);
            struct { uint32_t ty, bs, qk; uint32_t dof[2]; } T[5] = {
                {Q4_T_Q6_K, 210, 256, {208, 208}}, {Q4_T_Q4_K, 144, 256, {0, 2}},
                {Q4_T_Q5_K, 176, 256, {0, 2}}, {Q4_T_IQ4_XS, 136, 256, {0, 0}},
                {Q4_T_IQ4_NL, 18, 32, {0, 0}}};
            const uint32_t K = 2560, R = 256;
            for (int ci = 0; ci < 5 * 7; ci++) {
                const int ti = ci % 5;
                const uint32_t NT = 2 + (uint32_t)(ci / 5);
                const uint64_t rb = (uint64_t)K / T[ti].qk * T[ti].bs;
                uint8_t *w = malloc(rb * R);
                float *x = malloc((size_t)NT * K * 4), *y = malloc((size_t)NT * R * 4);
                float *row = malloc((size_t)K * 4);
                uint64_t rng = 0x51ED2701u + (uint64_t)ti;
                for (uint64_t i = 0; i < rb * R; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    w[i] = (uint8_t)rng;
                }
                const uint16_t hd[2] = {0x1c00, 0x1800};   /* 2^-8, 2^-9 */
                for (uint64_t b = 0; b < rb * R / T[ti].bs; b++)
                    for (int k = 0; k < 2; k++)
                        memcpy(w + b * T[ti].bs + T[ti].dof[k], &hd[k], 2);
                for (uint32_t i = 0; i < NT * K; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    x[i] = ((int)(rng % 2001) - 1000) * 1e-3f;
                }
                uint8_t *dw = q4_hip_malloc(rb * R);
                float *dx = q4_hip_malloc((size_t)NT * K * 4);
                float *dy = q4_hip_malloc((size_t)NT * R * 4);
                if (!dw || !dx || !dy || !q4_hip_h2d(dw, w, rb * R) ||
                    !q4_hip_h2d(dx, x, (size_t)NT * K * 4))
                    return fail("mmvq gpu run");
                if (!q4_mmvq_n(T[ti].ty, dw, R, K, dx, dy, 1.f, NT)) {
                    /* outside the type's row policy: float kernels */
                    q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                    free(w); free(x); free(y); free(row);
                    continue;
                }
                if (!q4_hip_d2h(y, dy, (size_t)NT * R * 4))
                    return fail("mmvq gpu run");
                double num = 0.0, den = 0.0;
                for (uint32_t r = 0; r < R; r++) {
                    if (!q4_dequant_row(T[ti].ty, w + r * rb, K, row))
                        return fail("mmvq ref dequant");
                    for (uint32_t t = 0; t < NT; t++) {
                        double a = 0.0;
                        for (uint32_t k = 0; k < K; k++)
                            a += (double)row[k] * x[t * K + k];
                        const double e = a - y[t * R + r];
                        num += e * e;
                        den += a * a;
                    }
                }
                const double rel = den > 0.0 ? sqrt(num / den) : 1.0;
                if (!(rel < 2e-2) || NT == 2 || NT == 4)
                    printf("mmvq %s n%u: rel L2 %.2e\n", q4_type_name(T[ti].ty),
                           NT, rel);
                q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                free(w); free(x); free(y); free(row);
                if (!(rel < 2e-2)) return fail("mmvq accuracy");
            }
            q4_mmvq_set(-1);
        }
        if (q4_hip_init()) {
            /* Q4_MMVQ=2 (verify-invariant): each row identical for NT 1..8 */
            q4_mmvq_warm();
            q4_mmvq_set(2);
            q4_mmvq_verify(1);
            struct { uint32_t ty, bs, qk; uint32_t dof[2]; uint32_t R; } T[5] = {
                {Q4_T_Q6_K, 210, 256, {208, 208}, 512}, {Q4_T_Q4_K, 144, 256, {0, 2}, 256},
                {Q4_T_Q5_K, 176, 256, {0, 2}, 512}, {Q4_T_IQ4_XS, 136, 256, {0, 0}, 512},
                {Q4_T_IQ4_NL, 18, 32, {0, 0}, 512}};
            const uint32_t K = 2560;
            for (int ti = 0; ti < 5; ti++) {
                const uint32_t R = T[ti].R;
                const uint64_t rb = (uint64_t)K / T[ti].qk * T[ti].bs;
                uint8_t *w = malloc(rb * R);
                float *x = malloc((size_t)8 * K * 4);
                float *y8 = malloc((size_t)8 * R * 4), *yn = malloc((size_t)8 * R * 4);
                uint64_t rng = 0x7A11u + (uint64_t)ti;
                for (uint64_t i = 0; i < rb * R; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    w[i] = (uint8_t)rng;
                }
                const uint16_t hd[2] = {0x1c00, 0x1800};
                for (uint64_t b = 0; b < rb * R / T[ti].bs; b++)
                    for (int k = 0; k < 2; k++)
                        memcpy(w + b * T[ti].bs + T[ti].dof[k], &hd[k], 2);
                for (uint32_t i = 0; i < 8 * K; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    x[i] = ((int)(rng % 2001) - 1000) * 1e-3f;
                }
                uint8_t *dw = q4_hip_malloc(rb * R);
                float *dx = q4_hip_malloc((size_t)8 * K * 4);
                float *dy = q4_hip_malloc((size_t)8 * R * 4);
                if (!dw || !dx || !dy || !q4_hip_h2d(dw, w, rb * R) ||
                    !q4_hip_h2d(dx, x, (size_t)8 * K * 4) ||
                    !q4_mmvq_n(T[ti].ty, dw, R, K, dx, dy, 1.f, 8) ||
                    !q4_hip_d2h(y8, dy, (size_t)8 * R * 4))
                    return fail("mmvq inv run");
                for (uint32_t NT = 1; NT < 8; NT++) {
                    if (!q4_mmvq_n(T[ti].ty, dw, R, K, dx, dy, 1.f, NT) ||
                        !q4_hip_d2h(yn, dy, (size_t)NT * R * 4))
                        return fail("mmvq inv run n");
                    int nd = 0;
                    for (size_t i = 0; i < (size_t)NT * R; i++) nd += memcmp(&yn[i], &y8[i], 4) != 0;
                    if (nd) {
                        printf("mmvq inv %s n%u: %d of %u values differ from n8\n",
                               q4_type_name(T[ti].ty), NT, nd, NT * R);
                        return fail("mmvq batch invariance");
                    }
                }
                q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy);
                free(w); free(x); free(y8); free(yn);
            }
            q4_mmvq_verify(0);
            q4_mmvq_set(-1);
            printf("mmvq verify-invariant: rows identical for n 1..8 (5 types)\n");
        }
        if (q4_hip_init()) {
            /* fused hc up + mulsig_mean vs the two-kernel path: identical */
            enum { KI = 320, NE = 2560, HC = 4, NT = 4 };
            uint16_t *w = malloc((size_t)HC * NE * KI * 2);
            float *x = malloc((size_t)NT * KI * 4), *a = malloc((size_t)NT * HC * NE * 4);
            float *y0 = malloc((size_t)NT * NE * 4), *y1 = malloc((size_t)NT * NE * 4);
            uint64_t rng = 0xC0FFEEull;
            for (size_t i = 0; i < (size_t)HC * NE * KI; i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                const float f = ((int)(rng % 2001) - 1000) * 1e-4f;
                uint32_t u;
                memcpy(&u, &f, 4);
                w[i] = (uint16_t)(u >> 16);
            }
            for (int i = 0; i < NT * KI; i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                x[i] = ((int)(rng % 2001) - 1000) * 1e-3f;
            }
            for (int i = 0; i < NT * HC * NE; i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                a[i] = ((int)(rng % 2001) - 1000) * 1e-3f;
            }
            uint8_t *dw = q4_hip_malloc((size_t)HC * NE * KI * 2);
            float *dx = q4_hip_malloc((size_t)NT * KI * 4);
            float *da = q4_hip_malloc((size_t)NT * HC * NE * 4);
            float *dg = q4_hip_malloc((size_t)NT * HC * NE * 4);
            float *dy = q4_hip_malloc((size_t)NT * NE * 4);
            if (!dw || !dx || !da || !dg || !dy ||
                !q4_hip_h2d(dw, w, (size_t)HC * NE * KI * 2) ||
                !q4_hip_h2d(dx, x, (size_t)NT * KI * 4) ||
                !q4_hip_h2d(da, a, (size_t)NT * HC * NE * 4) ||
                !q4_hip_gemv_dd_n(Q4_T_BF16, dw, HC * NE, KI, dx, dg, 1.f, NT) ||
                !q4_hip_mulsig_mean_n(da, dg, dy, NE, HC, NT) ||
                !q4_hip_d2h(y0, dy, (size_t)NT * NE * 4) ||
                !q4_hip_hcup_mix_n(Q4_T_BF16, dw, dx, da, dy, KI, NE, HC, NT) ||
                !q4_hip_d2h(y1, dy, (size_t)NT * NE * 4))
                return fail("hcup gpu run");
            int nd = 0;
            for (int i = 0; i < NT * NE; i++) nd += memcmp(&y0[i], &y1[i], 4) != 0;
            printf("hcup fuse: %d of %d values differ\n", nd, NT * NE);
            q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(da); q4_hip_free(dg);
            q4_hip_free(dy);
            free(w); free(x); free(a); free(y0); free(y1);
            if (nd) return fail("hcup fuse");
        }
        if (q4_hip_init()) {
            /* bf16 pair launch vs the two single n-row calls: identical */
            const struct { uint32_t K, r0, r1; float a0; } P[] = {
                { 10240, 320, 24, 0.25f }, { 2560, 48, 48, 0.f } };
            for (int pi = 0; pi < 2; pi++) {
                const uint32_t K = P[pi].K, r0 = P[pi].r0, r1 = P[pi].r1;
                const size_t nw = (size_t)(r0 + r1) * K;
                uint16_t *w = malloc(nw * 2);
                float *x = malloc((size_t)8 * K * 4);
                float *y0 = malloc((size_t)8 * (r0 + r1) * 4), *y1 = malloc((size_t)8 * (r0 + r1) * 4);
                uint64_t rng = 0xBEEFull + pi;
                for (size_t i = 0; i < nw; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    const float f = ((int)(rng % 2001) - 1000) * 1e-4f;
                    uint32_t u;
                    memcpy(&u, &f, 4);
                    w[i] = (uint16_t)(u >> 16);
                }
                for (size_t i = 0; i < (size_t)8 * K; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    x[i] = ((int)(rng % 2001) - 1000) * 1e-3f;
                }
                uint8_t *dw = q4_hip_malloc(nw * 2);
                float *dx = q4_hip_malloc((size_t)8 * K * 4);
                float *dy0 = q4_hip_malloc((size_t)8 * r0 * 4), *dy1 = q4_hip_malloc((size_t)8 * r1 * 4);
                if (!dw || !dx || !dy0 || !dy1 || !q4_hip_h2d(dw, w, nw * 2) ||
                    !q4_hip_h2d(dx, x, (size_t)8 * K * 4))
                    return fail("bf16 pair alloc");
                const uint8_t *dw1 = dw + (size_t)r0 * K * 2;
                for (uint32_t NT = 2; NT <= 8; NT++) {
                    const bool a_ok = P[pi].a0 != 0.f
                        ? q4_hip_gemv_dd_n_act(Q4_T_BF16, dw, r0, K, dx, dy0, 1.f, P[pi].a0, NT)
                        : q4_hip_gemv_dd_n(Q4_T_BF16, dw, r0, K, dx, dy0, 1.f, NT);
                    if (!a_ok || !q4_hip_gemv_dd_n(Q4_T_BF16, dw1, r1, K, dx, dy1, 1.f, NT) ||
                        !q4_hip_d2h(y0, dy0, (size_t)NT * r0 * 4) ||
                        !q4_hip_d2h(y0 + (size_t)NT * r0, dy1, (size_t)NT * r1 * 4))
                        return fail("bf16 pair single");
                    q4_hip_bf16_pair_set(1);
                    if (!q4_hip_gemv_bf16_n2(dw, dy0, r0, P[pi].a0, dw1, dy1, r1, 0.f, dx, K, NT) ||
                        !q4_hip_d2h(y1, dy0, (size_t)NT * r0 * 4) ||
                        !q4_hip_d2h(y1 + (size_t)NT * r0, dy1, (size_t)NT * r1 * 4))
                        return fail("bf16 pair launch");
                    q4_hip_bf16_pair_set(-1);
                    int nd = 0;
                    for (size_t i = 0; i < (size_t)NT * (r0 + r1); i++) nd += memcmp(&y0[i], &y1[i], 4) != 0;
                    if (nd) {
                        printf("bf16 pair K%u n%u: %d values differ\n", K, NT, nd);
                        return fail("bf16 pair");
                    }
                }
                q4_hip_free(dw); q4_hip_free(dx); q4_hip_free(dy0); q4_hip_free(dy1);
                free(w); free(x); free(y0); free(y1);
            }
            printf("bf16 pair: identical to the single calls (n 2..8)\n");
        }
        if (q4_hip_init()) {
            /* QSA top-k (both kernels) vs a CPU reference: blocks above the
             * k-th key ascending, then threshold ties ascending, then the
             * tail positions; scores quantized coarsely so ties are common */
            enum { NT = 4, STRIDE = 12000, KB = 512, RATIO = 4, WIDTH = 2048 + 8 };
            const uint32_t pos0s[] = { 1500, 2700, 9000, 34000, 47000 };
            float *sc = malloc((size_t)NT * STRIDE * 4);
            int32_t *got = malloc((size_t)NT * WIDTH * 4), *ref = malloc((size_t)NT * WIDTH * 4);
            uint32_t *key = malloc((size_t)STRIDE * 4), *srt = malloc((size_t)STRIDE * 4);
            float *dsc = q4_hip_malloc((size_t)NT * STRIDE * 4);
            int32_t *dsel = q4_hip_malloc((size_t)NT * WIDTH * 4);
            if (!sc || !got || !ref || !key || !srt || !dsc || !dsel)
                return fail("qsa topk alloc");
            uint64_t rng = 0x5EEDull;
            for (int q = 0; q < 2; q++)
            for (size_t pi = 0; pi < sizeof pos0s / sizeof pos0s[0]; pi++) {
                const uint32_t pos0 = pos0s[pi] < (STRIDE - 8) * RATIO ? pos0s[pi] : (STRIDE - 8) * RATIO;
                for (size_t i = 0; i < (size_t)NT * STRIDE; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    const int v = (int)(rng % (q ? 64 : 200001)) - (q ? 20 : 100000);
                    sc[i] = v * (q ? 0.25f : 1e-4f);
                }
                for (uint32_t t = 0; t < NT; t++) {
                    const uint32_t pos = pos0 + t, n_vis = (pos + 1) / RATIO;
                    const uint32_t need = n_vis < KB ? n_vis : KB;
                    int32_t *r = ref + (size_t)t * WIDTH;
                    uint32_t w = 0;
                    if (need == n_vis) {
                        for (; w < WIDTH; w++) r[w] = w <= pos ? (int32_t)w : -1;
                        continue;
                    }
                    for (uint32_t b = 0; b < n_vis; b++) {
                        uint32_t u;
                        memcpy(&u, &sc[(size_t)t * STRIDE + b], 4);
                        key[b] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                        srt[b] = key[b];
                    }
                    /* the need-th largest key (partial selection sort is too slow) */
                    uint32_t lo = 0, hi = 0xffffffffu;
                    while (lo < hi) {   /* largest thr with count(key >= thr) >= need */
                        const uint32_t mid = (uint32_t)(lo + ((uint64_t)hi - lo + 1) / 2);
                        uint32_t c = 0;
                        for (uint32_t b = 0; b < n_vis; b++) c += srt[b] >= mid;
                        if (c >= need) lo = mid; else hi = mid - 1;
                    }
                    const uint32_t thr = lo;
                    uint32_t nch = 0;
                    for (uint32_t b = 0; b < n_vis; b++)
                        if (key[b] > thr) for (uint32_t j = 0; j < RATIO; j++, nch++) r[w++] = (int32_t)(b * RATIO + j);
                    for (uint32_t b = 0; b < n_vis && nch < need * RATIO; b++)
                        if (key[b] == thr) for (uint32_t j = 0; j < RATIO; j++, nch++) r[w++] = (int32_t)(b * RATIO + j);
                    uint32_t tail = n_vis * RATIO;
                    for (; w < WIDTH; w++, tail++) r[w] = tail <= pos ? (int32_t)tail : -1;
                }
                for (int par = 0; par < 2; par++) {
                    q4_qsa_topk_par_set(par);
                    if (!q4_hip_h2d(dsc, sc, (size_t)NT * STRIDE * 4) ||
                        !q4_hip_qsa_topk(dsc, dsel, NT, 0, STRIDE, KB, RATIO, pos0, WIDTH) ||
                        !q4_hip_d2h(got, dsel, (size_t)NT * WIDTH * 4))
                        return fail("qsa topk run");
                    int nd = 0;
                    for (int i = 0; i < NT * WIDTH; i++) nd += got[i] != ref[i];
                    if (nd) {
                        printf("qsa topk %s pos %u ties %d: %d of %d differ\n",
                               par ? "par" : "old", pos0, q, nd, NT * WIDTH);
                        return fail("qsa topk");
                    }
                }
            }
            q4_qsa_topk_par_set(-1);
            printf("qsa topk: old and parallel kernels match the CPU reference\n");
            q4_hip_free(dsc); q4_hip_free(dsel);
            free(sc); free(got); free(ref); free(key); free(srt);
        }
        printf("ok unit (no model)\n");
        return 0;
    }
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return fail("open");
    if (g.n_tensors == 0) return fail("no tensors");
    if (g.bytes_total == 0) return fail("zero size");
    if (g.n_files == 0) return fail("no files");
    if (g.arch && strcmp(g.arch, "qwen4exp") == 0) {
        if (g.n_expert == 0) return fail("qwen4exp has no experts");
        if (g.bytes_routed == 0) return fail("no routed bytes");
        if (g.per_expert_bytes == 0) return fail("per-expert");
    }
    q4_machine m = q4_machine_this_pc(8192);
    q4_place p = q4_place_plan(&g, &m);
    {
        uint64_t sum = g.bytes_dense + g.bytes_shared + g.bytes_routed + g.bytes_ple;
        if (sum != g.bytes_total) {
            fprintf(stderr, "kind sum %.3f vs total %.3f GiB\n",
                    (double)sum / Q4_GIB, (double)g.bytes_total / Q4_GIB);
            q4_gguf_close(&g);
            return fail("kind partition");
        }
    }
    if (g.arch && strcmp(g.arch, "qwen4exp") == 0) {
        if (g.ple_n_heads == 0 || g.bytes_ple == 0) return fail("PLE meta");
        if (g.hc_mult != 4) return fail("hc_mult");
        if (g.n_vocab == 0) return fail("n_vocab");
        if (g.ssm_d_inner == 0 || g.indexer_top_k == 0) return fail("ssm/qsa kv");
        q4_ple *ple = q4_ple_open(&g);
        if (!ple) return fail("ple open");
        int32_t toks[4] = {1, 2, 3, 4};
        int32_t rows[4 * 16];
        if (!q4_ple_hash(ple, toks, 4, NULL, rows)) {
            q4_ple_close(ple);
            q4_gguf_close(&g);
            return fail("ple hash");
        }
        float *emb = malloc(16 * 160 * sizeof(float));
        if (!emb || !q4_ple_gather(ple, rows, 16, emb)) {
            free(emb);
            q4_ple_close(ple);
            q4_gguf_close(&g);
            return fail("ple gather");
        }
        free(emb);
        q4_ple_close(ple);
    }
    q4_gguf_print(&g, stdout);
    q4_place_print(&p, &g, stdout);
    if (g.tok_vocab && g.n_tok_vocab) {
        q4_tok *tok = q4_tok_open(&g);
        if (!tok) {
            q4_gguf_close(&g);
            return fail("tok open");
        }
        int32_t ids[64];
        int n = q4_tok_encode(tok, "こんにちは", ids, 64);
        char back[64];
        int dn = q4_tok_decode(tok, ids, n, back, (int)sizeof(back));
        if (n <= 0 || n > 8 || dn <= 0 || strcmp(back, "こんにちは") != 0) {
            fprintf(stderr, "tok ja n=%d dec='%s'\n", n, back);
            q4_tok_close(tok);
            q4_gguf_close(&g);
            return fail("tok encode ja");
        }
        printf("tok encode ja %d tokens  roundtrip ok\n", n);
        /* Qwen2 regex splits each digit; whole-string BPE would not. */
        int nd = q4_tok_encode(tok, "123", ids, 64);
        if (nd != 3) {
            fprintf(stderr, "tok digits n=%d (want 3)\n", nd);
            q4_tok_close(tok);
            q4_gguf_close(&g);
            return fail("tok encode digits");
        }
        int ns = q4_tok_encode(tok, "hello world", ids, 64);
        if (ns < 2) {
            fprintf(stderr, "tok hello world n=%d\n", ns);
            q4_tok_close(tok);
            q4_gguf_close(&g);
            return fail("tok encode words");
        }
        {
            /* the template's |trim (Python str.strip) on user content, and a
             * form feed kept as its byte token (it used to reach the
             * tokenizer as the letter f through the JSON parser) */
            const char *r1[] = {"user"};
            const char *t1[] = {" \n page\fbreak \n\n\xe3\x80\x80"};
            const char *t2[] = {"page\fbreak"};
            int32_t a1[64], a2[64], ff[4];
            const int n1 = q4_tok_apply_messages(tok, r1, t1, 1, a1, 64);
            const int n2 = q4_tok_apply_messages(tok, r1, t2, 1, a2, 64);
            const int nf = q4_tok_encode(tok, "\f", ff, 4);
            char fb[8] = {0};
            const int fl = nf == 1 ? q4_tok_decode(tok, ff, 1, fb, 8) : 0;
            if (n1 <= 0 || n1 != n2 || memcmp(a1, a2, (size_t)n1 * 4) ||
                fl != 1 || fb[0] != '\f') {
                fprintf(stderr, "tok trim n %d/%d, form feed n %d len %d\n",
                        n1, n2, nf, fl);
                q4_tok_close(tok);
                q4_gguf_close(&g);
                return fail("tok template trim / form feed");
            }
            printf("tok template trim + form feed ok\n");
        }
        {
            putenv("Q4_THINK=1");
            const char *roles[] = {"user", "assistant", "user"};
            const char *texts[] = {"hi", "answer only", "next"};
            int32_t hist[256];
            int nh = q4_tok_apply_messages(tok, roles, texts, 3, hist, 256);
            int32_t te = q4_tok_id(tok, "</think>");
            int32_t th = q4_tok_id(tok, "<think>");
            int saw_close = 0, open_unclosed = 0;
            for (int i = 0; i < nh; i++) {
                if (hist[i] == te) saw_close = 1;
                if (hist[i] == th && !saw_close) open_unclosed = 1;
            }
            /* History answer must sit after a closed think; only the final
             * assistant-open may leave <think> unclosed. */
            int last_th = -1, last_te = -1;
            for (int i = 0; i < nh; i++) {
                if (hist[i] == th) last_th = i;
                if (hist[i] == te) last_te = i;
            }
            if (nh <= 0 || last_th < 0 || last_te < 0 || last_te >= last_th) {
                fprintf(stderr, "tok hist think nh=%d last_th=%d last_te=%d\n",
                        nh, last_th, last_te);
                q4_tok_close(tok);
                q4_gguf_close(&g);
                return fail("tok history think close");
            }
            (void)open_unclosed;
            printf("tok history think closed then reopen  %d tokens\n", nh);
            int32_t tc = q4_tok_id(tok, "<tool_call>");
            int32_t tce = q4_tok_id(tok, "</tool_call>");
            int32_t ids2[8];
            int n2 = q4_tok_encode(tok, "<tool_call>", ids2, 8);
            if (tc < 0 || tce < 0 || n2 != 1 || ids2[0] != tc) {
                fprintf(stderr, "tok tool_call id=%d enc_n=%d enc0=%d\n", tc, n2,
                        n2 > 0 ? ids2[0] : -1);
                q4_tok_close(tok);
                q4_gguf_close(&g);
                return fail("tok tool_call special");
            }
        }
        q4_tok_close(tok);
        printf("tok digits 3  hello world %d tokens\n", ns);
    }
    {
        uint64_t kv = q4_qsa_kv_bytes(&g, 65536);
        if (g.n_layer && kv == 0) {
            q4_gguf_close(&g);
            return fail("qsa kv 64k");
        }
        printf("QSA KV 64K %.2f GiB\n", (double)kv / Q4_GIB);
    }
    q4_gguf_close(&g);
    printf("ok model\n");
    return 0;
}
