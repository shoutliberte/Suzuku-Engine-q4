/* q4_gemm.cu — hipBLASLt/hipBLAS dense batched GEMM for prefill.
 *
 * The library path replaces the warp-per-row dense kernels when the batch
 * is large: quantized weights are dequantized to fp16 (block dequant
 * transcribed from the quant dequant used by the GGUF formats, MIT;
 * see README References), activations converted to
 * fp16/bf16 with a ±65504 clamp, and the product runs on the tensor cores
 * (fp32 accumulate):  y[t][r] = scale * sum_k W[r][k]*x[t][k].
 * hipBLASLt algo selection: tuned solution ids from
 * Q4_HIPBLASLT_TUNING (env path, arch/version-guarded) or the library
 * heuristic when unset; hipblasGemmEx is the last resort.
 * hipBLASLt wrapper and the tuning-table loader are adapted (MIT;
 * see README References). */

extern "C" {
#include "q4.h"
}

#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#include <unistd.h>

#include "q4_hgemm_tuning.hpp"

static hipStream_t hgemm_str(void) { return (hipStream_t)q4_hip_stream(); }

/* ---- dequant: native GGUF block rows -> fp16 ---------------------------- */

namespace {

__device__ __forceinline__ float h2f_hg(const uint8_t *p) {
    return __half2float(__ushort_as_half((uint16_t)(p[0] | (p[1] << 8))));
}

__constant__ int8_t kv_iq4nl_hg[16] = {-127, -104, -83, -65, -49, -35,
                                       -22,  -10,  1,   13,  25,  38,
                                       53,   69,   89,  113};

__device__ __forceinline__ void scale_min_k4_hg(int j, const uint8_t *q,
                                                int &d, int &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct H16 {
    uint16_t v;
};
__device__ __forceinline__ void put16(H16 *o, int i, float v) {
    o[i].v = __half_as_ushort(__float2half_rn(v));
}

/* Elements l0..l0+7 (l0 = 0, 8, 16, 24) of one 32-element group of a block
 * row into out[0..7]; same arithmetic per element as the old 32-element
 * loop, so the f16 values are unchanged. */
template <int TYPE>
__device__ __forceinline__ void group8_hg(const uint8_t *row_blocks,
                                          int gi_in_row, int l0, H16 *out) {
    if constexpr (TYPE == 8) {                    // Q8_0: d + 32 int8
        const uint8_t *b = row_blocks + (size_t)gi_in_row * 34;
        const float d = h2f_hg(b);
        for (int j = l0; j < l0 + 8; ++j)
            put16(out, j - l0, (float)(int8_t)b[2 + j] * d);
    } else if constexpr (TYPE == 12) {            // Q4_K: d dmin scales[12] qs[128]
        const uint8_t *b = row_blocks + (size_t)(gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f_hg(b), dmin = h2f_hg(b + 2);
        int sc, m;
        scale_min_k4_hg(gi, b + 4, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const uint8_t *q = b + 16 + 32 * j64;
        for (int l = l0; l < l0 + 8; ++l)
            put16(out, l - l0, d1 * (float)(hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {            // Q5_K: + qh[32]
        const uint8_t *b = row_blocks + (size_t)(gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f_hg(b), dmin = h2f_hg(b + 2);
        int sc, m;
        scale_min_k4_hg(gi, b + 4, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const uint8_t *qh = b + 16;
        const uint8_t *ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t)(1u << (2 * j64 + hi));
        for (int l = l0; l < l0 + 8; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put16(out, l - l0, d1 * (float)(nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {            // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t *b = row_blocks + (size_t)(gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t *ql = b + 64 * n;
        const uint8_t *qh = b + 128 + 32 * n;
        const int8_t *sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * n;
        const float d = h2f_hg(b + 208);
        for (int l = l0; l < l0 + 8; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0)
                q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1)
                q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2)
                q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else
                q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put16(out, l - l0, d * (float)sc[is + 2 * qu] * (float)(q - 32));
        }
    } else if constexpr (TYPE == 20) {            // IQ4_NL: d + 32 nib
        const uint8_t *b = row_blocks + (size_t)gi_in_row * 18;
        const float d = h2f_hg(b);
        for (int j = l0; j < l0 + 8; ++j)
            put16(out, j - l0,
                  d * (float)(j < 16 ? kv_iq4nl_hg[b[2 + j] & 0xf]
                                     : kv_iq4nl_hg[b[2 + j - 16] >> 4]));
    } else if constexpr (TYPE == 23) {            // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t *b = row_blocks + (size_t)(gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f_hg(b);
        const uint16_t scales_h = (uint16_t)(b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const uint8_t *qs = b + 8 + 16 * ib;
        for (int j = l0; j < l0 + 8; ++j)
            put16(out, j - l0,
                  dl * (float)(j < 16 ? kv_iq4nl_hg[qs[j] & 0xf]
                                      : kv_iq4nl_hg[qs[j - 16] >> 4]));
    }
}

/* Thread per 8 outputs, one 16-byte store: consecutive lanes write
 * consecutive 16 bytes (the thread-per-32 version stored 2 bytes at a
 * 64-byte lane stride: Q6_K 2560x10240 took ~0.9 ms). */
template <int TYPE>
__global__ void dequant_f16_k(const uint8_t *__restrict__ blocks,
                              int64_t row_bytes, int64_t row0, int64_t rows,
                              int64_t groups_per_row, H16 *__restrict__ out) {
    const int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t g = e >> 2;
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    const int l0 = (int)(e & 3) * 8;
    union { H16 h[8]; uint4 u; } o;
    group8_hg<TYPE>(blocks + (row0 + r) * row_bytes, (int)gi, l0, o.h);
    *(uint4 *)(out + r * groups_per_row * 32 + gi * 32 + l0) = o.u;
}

bool dequant_geom(int type, int &block_elems, int &block_bytes) {
    switch (type) {
    case Q4_T_Q8_0:   block_elems = 32;  block_bytes = 34;  return true;
    case Q4_T_Q4_K:   block_elems = 256; block_bytes = 144; return true;
    case Q4_T_Q5_K:   block_elems = 256; block_bytes = 176; return true;
    case Q4_T_Q6_K:   block_elems = 256; block_bytes = 210; return true;
    case Q4_T_IQ4_NL: block_elems = 32;  block_bytes = 18;  return true;
    case Q4_T_IQ4_XS: block_elems = 256; block_bytes = 136; return true;
    }
    return false;
}

bool dequant_f16(int type, const void *blocks, int64_t row0, int64_t rows,
                 int64_t cols, uint16_t *out) {
    int be = 0, bb = 0;
    if (!dequant_geom(type, be, bb) || cols % be != 0 || rows <= 0)
        return false;
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32,
                  total = rows * gpr;
    if (total > (int64_t)1 << 40) return false;
    const unsigned grid = (unsigned)((total * 4 + 255) / 256);
    const uint8_t *p = (const uint8_t *)blocks;
    hipStream_t st = hgemm_str();
#define DQ(TY)                                                                  \
    dequant_f16_k<TY><<<grid, 256, 0, st>>>(p, row_bytes, row0, rows, gpr,      \
                                          (H16 *)out);                          \
    break
    switch (type) {
    case Q4_T_Q8_0:   DQ(Q4_T_Q8_0);
    case Q4_T_Q4_K:   DQ(Q4_T_Q4_K);
    case Q4_T_Q5_K:   DQ(Q4_T_Q5_K);
    case Q4_T_Q6_K:   DQ(Q4_T_Q6_K);
    case Q4_T_IQ4_NL: DQ(Q4_T_IQ4_NL);
    case Q4_T_IQ4_XS: DQ(Q4_T_IQ4_XS);
    default: return false;
    }
#undef DQ
    return hipGetLastError() == hipSuccess;
}

/* ---- fp32 -> fp16/bf16 activation conversion ---------------------------- */

__device__ __forceinline__ uint16_t hf_sat(float f) {
    /* NaN stays NaN; finite values clamp at fp16's last representable. */
    return __half_as_ushort(
        __float2half_rn(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)));
}
__device__ __forceinline__ uint16_t bf_rne(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__global__ void to_f16_k(const float *__restrict__ x, uint16_t *__restrict__ y,
                         int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = hf_sat(x[i]);
}
__global__ void to_bf16_k(const float *__restrict__ x, uint16_t *__restrict__ y,
                          int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf_rne(x[i]);
}

bool to_u16(const float *x, uint16_t *y, int64_t n, bool bf16) {
    if (n <= 0) return true;
    const unsigned grid = (unsigned)((n + 255) / 256);
    if (bf16)
        to_bf16_k<<<grid, 256, 0, hgemm_str()>>>(x, y, n);
    else
        to_f16_k<<<grid, 256, 0, hgemm_str()>>>(x, y, n);
    return hipGetLastError() == hipSuccess;
}

/* ---- hipBLASLt / hipblasGemmEx state ------------------------------------ */

constexpr int64_t W_ELEMS = 32ll << 20; /* fp16 elems: weight dequant scratch */
constexpr int64_t X_ELEMS = 16ll << 20; /* fp16 elems: activation slice      */
constexpr size_t WS_BYTES = 32u << 20;

struct LtKey {
    int dtype, t, n, k, ldy;
    uint32_t alpha_bits, beta_bits;
    bool operator<(const LtKey &o) const {
        return std::tie(dtype, n, k, ldy, t, alpha_bits, beta_bits) <
               std::tie(o.dtype, o.n, o.k, o.ldy, o.t, o.alpha_bits,
                        o.beta_bits);
    }
};
struct LtAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};
struct LtDescs {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    ~LtDescs() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }
    bool init(hipDataType type, int t, int n, int k, int ldy) {
        const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        return hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F,
                                         HIP_R_32F) ==
                   HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                               &ta, sizeof(ta)) ==
                   HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB,
                                               &tb, sizeof(tb)) ==
                   HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatrixLayoutCreate(&a, type, k, n, k) ==
                   HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatrixLayoutCreate(&b, type, k, t, k) ==
                   HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) ==
                   HIPBLAS_STATUS_SUCCESS;
    }
};

struct LtState {
    hipblasLtHandle_t handle = nullptr;
    hipblasHandle_t hb = nullptr;
    uint16_t *wq = nullptr, *xq = nullptr;
    void *ws = nullptr;
    int64_t x_elems = X_ELEMS; /* xq capacity (fp16 elems) */
    bool lend = false;         /* buffers come from q4_hgemm_lend only */
    q4hg::TuningTable table;
    bool table_ok = false;
    std::map<LtKey, LtAlgo> cache;
    uint64_t lt = 0, gemmex = 0, tune_hits = 0, tune_miss = 0;
};

LtState &lt(void) {
    static LtState s;
    if (!s.handle) {
        const char *err = "";
        if (hipblasLtCreate(&s.handle) != HIPBLAS_STATUS_SUCCESS)
            std::fprintf(stderr, "q4 hgemm: hipBLASLt create failed\n");
        if (s.handle &&
            hipblasCreate(&s.hb) != HIPBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "q4 hgemm: hipblasCreate failed\n");
            hipblasLtDestroy(s.handle);
            s.handle = nullptr;
        }
        if (s.handle) {
            if (!s.lend &&
                (hipMalloc(&s.wq, (size_t)W_ELEMS * 2) != hipSuccess ||
                 hipMalloc(&s.xq, (size_t)X_ELEMS * 2) != hipSuccess ||
                 hipMalloc(&s.ws, WS_BYTES) != hipSuccess)) {
                std::fprintf(stderr, "q4 hgemm: scratch alloc failed\n");
                if (s.wq) hipFree(s.wq);
                if (s.xq) hipFree(s.xq);
                if (s.ws) hipFree(s.ws);
                s.wq = s.xq = nullptr;
                s.ws = nullptr;
                hipblasLtDestroy(s.handle);
                hipblasDestroy(s.hb);
                s.handle = nullptr;
                return s;
            }
            if (s.ws) (void)hipblasSetWorkspace(s.hb, s.ws, WS_BYTES);
            std::atexit([] {
                if (lt().table_ok)
                    std::fprintf(stderr,
                                 "q4 hgemm: tune hits=%llu misses=%llu "
                                 "lt=%llu gemmex=%llu\n",
                                 (unsigned long long)lt().tune_hits,
                                 (unsigned long long)lt().tune_miss,
                                 (unsigned long long)lt().lt,
                                 (unsigned long long)lt().gemmex);
            });
            const char *path = std::getenv("Q4_HIPBLASLT_TUNING");
            std::string dflt;
            if (!path) {
                /* bundled table next to the binary (data/hipblaslt/), used
                 * only if readable; Q4_HIPBLASLT_TUNING="" forces heuristic */
                char exe[4096];
                const ssize_t en =
                    readlink("/proc/self/exe", exe, sizeof(exe) - 1);
                if (en > 0) {
                    exe[en] = 0;
                    dflt = exe;
                    const auto sl = dflt.find_last_of('/');
                    if (sl != std::string::npos) dflt.resize(sl);
                    dflt += "/data/hipblaslt/gfx1100-hipblaslt-100202.txt";
                    std::ifstream probe(dflt);
                    if (probe.good()) path = dflt.c_str();
                }
            }
            if (path && *path) {
                int version = 0, device = 0;
                hipDeviceProp_t props{};
                std::string arch;
                (void)hipGetDevice(&device);
                if (hipblasLtGetVersion(s.handle, &version) ==
                        HIPBLAS_STATUS_SUCCESS &&
                    hipGetDeviceProperties(&props, device) == hipSuccess) {
                    arch = props.gcnArchName;
                    const auto colon = arch.find(':');
                    if (colon != std::string::npos) arch.resize(colon);
                }
                std::string load_err;
                if (!arch.empty() &&
                    s.table.load(path, arch, version, load_err)) {
                    s.table_ok = true;
                    std::fprintf(stderr,
                                 "q4 hgemm: hipBLASLt tuning (%zu rows, %s, "
                                 "version %d)\n",
                                 s.table.rows().size(), arch.c_str(),
                                 version);
                } else {
                    err = load_err.c_str();
                    std::fprintf(stderr,
                                 "q4 hgemm: tuning rejected (%s); heuristic\n",
                                 err);
                }
            }
        }
    }
    return s;
}

LtAlgo resolve(LtState &s, q4hg::InputType type, int t, int n, int k, int ldy,
               float alpha, float beta) {
    uint32_t ab = 0, bb = 0;
    std::memcpy(&ab, &alpha, 4);
    std::memcpy(&bb, &beta, 4);
    const LtKey key{int(type), t, n, k, ldy, ab, bb};
    const auto cached = s.cache.find(key);
    if (cached != s.cache.end()) return cached->second;

    LtAlgo resolved;
    LtDescs desc;
    const hipDataType input =
        type == q4hg::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input, t, n, k, ldy))
        return s.cache.emplace(key, resolved).first->second;

    bool verbose = std::getenv("Q4_HIPBLASLT_VERBOSE") != nullptr;
    if (s.table_ok) {
        const q4hg::TuningRow *row = s.table.closest(type, n, k, ldy, t);
        if (row) {
            std::vector<int> ids{row->solution_id};
            std::vector<hipblasLtMatmulHeuristicResult_t> cand;
            if (hipblaslt_ext::getAlgosFromIndex(s.handle, ids, cand) ==
                    HIPBLAS_STATUS_SUCCESS &&
                !cand.empty() && cand.front().state == HIPBLAS_STATUS_SUCCESS &&
                hipblaslt_ext::getIndexFromAlgo(cand.front().algo) ==
                    row->solution_id) {
                size_t need = 0;
                if (hipblaslt_ext::matmulIsAlgoSupported(
                        s.handle, desc.op, &alpha, desc.a, desc.b, &beta,
                        desc.c, desc.c, cand.front().algo, need) ==
                        HIPBLAS_STATUS_SUCCESS &&
                    need <= WS_BYTES) {
                    resolved.supported = true;
                    resolved.algo = cand.front().algo;
                    resolved.workspace_bytes = need;
                    s.tune_hits++;
                    return s.cache.emplace(key, resolved).first->second;
                }
            }
            if (verbose)
                std::fprintf(stderr,
                             "q4 hgemm: tuned solution %d unusable T=%d N=%d "
                             "K=%d\n",
                             row->solution_id, t, n, k);
        } else {
            s.tune_miss++;
        }
    }

    /* Heuristic fallback (also the default when no table is loaded). */
    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS) {
        uint64_t max_ws = WS_BYTES;
        (void)hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
            sizeof(max_ws));
        hipblasLtMatmulHeuristicResult_t res[8];
        int nres = 0;
        if (hipblasLtMatmulAlgoGetHeuristic(s.handle, desc.op, desc.a, desc.b,
                                            desc.c, desc.c, pref, 8, res,
                                            &nres) == HIPBLAS_STATUS_SUCCESS) {
            for (int i = 0; i < nres; i++) {
                if (res[i].state != HIPBLAS_STATUS_SUCCESS ||
                    res[i].workspaceSize > WS_BYTES)
                    continue;
                resolved.supported = true;
                resolved.algo = res[i].algo;
                resolved.workspace_bytes = res[i].workspaceSize;
                break;
            }
        }
        hipblasLtMatmulPreferenceDestroy(pref);
    }
    if (verbose && !resolved.supported)
        std::fprintf(stderr,
                     "q4 hgemm: no Lt algo T=%d N=%d K=%d ldy=%d (hipBLAS)\n",
                     t, n, k, ldy);
    return s.cache.emplace(key, resolved).first->second;
}

/* one GEMM: y[t][r] = alpha*sum_k w[r][k]*x[t][k] + beta*y[t][r],
 * x [T][K] fp16/bf16, w [N][K] same type, y [T][ldy] fp32. */
bool gemm_u16(LtState &s, const uint16_t *x, const uint16_t *w, float *y,
              int64_t T, int64_t N, int64_t K, int64_t ldy, float alpha,
              float beta, bool bf16) {
    if (T <= 0 || N <= 0 || K <= 0 || T > INT32_MAX || N > INT32_MAX ||
        K > INT32_MAX || ldy < N)
        return false;
    const q4hg::InputType ty =
        bf16 ? q4hg::InputType::bf16 : q4hg::InputType::f16;
    LtAlgo res =
        resolve(s, ty, (int)T, (int)N, (int)K, (int)ldy, alpha, beta);
    if (res.supported) {
        LtDescs desc;
        if (desc.init(bf16 ? HIP_R_16BF : HIP_R_16F, (int)T, (int)N, (int)K,
                      (int)ldy)) {
            if (hipblasLtMatmul(s.handle, desc.op, &alpha, w, desc.a, x,
                                desc.b, &beta, y, desc.c, y, desc.c,
                                &res.algo, s.ws,
                                WS_BYTES, hgemm_str()) ==
                HIPBLAS_STATUS_SUCCESS) {
                s.lt++;
                return true;
            }
        }
    }
    if (std::getenv("Q4_HIPBLASLT_VERBOSE"))
        std::fprintf(stderr,
                     "q4 hgemm: hipblasGemmEx T=%lld N=%lld K=%lld bf16=%d\n",
                     (long long)T, (long long)N, (long long)K, (int)bf16);
    /* hipblasGemmEx: same column-major shape. */
    const hipblasStatus_t st = hipblasGemmEx(
        s.hb, HIPBLAS_OP_T, HIPBLAS_OP_N, (int)N, (int)T, (int)K, &alpha, w,
        bf16 ? HIP_R_16BF : HIP_R_16F, (int)K, x,
        bf16 ? HIP_R_16BF : HIP_R_16F, (int)K, &beta, y, HIP_R_32F, (int)ldy,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    s.gemmex++;
    return st == HIPBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" bool q4_hgemm_on(void) {
    const char *e = std::getenv("Q4_HGEMM");
    return !e || e[0] != '0';
}

extern "C" bool q4_hgemm_supported(uint32_t type) {
    if (type == Q4_T_BF16) return true;
    int be = 0, bb = 0;
    return dequant_geom((int)type, be, bb) && be > 0;
}

extern "C" uint64_t q4_hgemm_scratch_bytes(void) {
    return q4_hgemm_on() ? (uint64_t)(W_ELEMS + X_ELEMS) * 2 + WS_BYTES : 0;
}

/* L1 lending (engine prompts): the weight-dequant, activation and
 * workspace buffers are borrowed from the arena tail with the rest of the
 * prompt scratch, the activation slice sized for a whole chunk (the fixed
 * 32 MiB one split a 5.5K-token hc GEMM into four).  The first call frees
 * the private buffers for good; outside a lend window (NULLs) the library
 * path is off and q4_hip_gemv_* falls back to its kernels (only prompt
 * batches of >= 64 rows come here).  128 MiB more expert slots in decode. */
extern "C" void q4_hgemm_lend_need(uint32_t T, uint64_t max_k, uint64_t *wq,
                                   uint64_t *xq, uint64_t *ws) {
    const bool on = q4_hgemm_on();
    *wq = on ? (uint64_t)W_ELEMS * 2 : 0;
    *xq = on ? (uint64_t)T * max_k * 2 : 0;
    *ws = on ? (uint64_t)WS_BYTES : 0;
}

extern "C" void q4_hgemm_lend(void *wq, void *xq, uint64_t xq_bytes,
                              void *ws) {
    LtState &s = lt();
    if (!s.lend) {
        if (s.wq) (void)hipFree(s.wq);
        if (s.xq) (void)hipFree(s.xq);
        if (s.ws) (void)hipFree(s.ws);
        s.lend = true;
    }
    s.wq = (uint16_t *)wq;
    s.xq = (uint16_t *)xq;
    s.ws = ws;
    s.x_elems = xq ? (int64_t)(xq_bytes / 2) : 0;
    if (ws && s.hb) (void)hipblasSetWorkspace(s.hb, ws, WS_BYTES);
}

extern "C" bool q4_hgemm_try(uint32_t type, const uint8_t *d_w, uint64_t nrows,
                             uint64_t ncols, const float *d_x, float *d_y,
                             float scale, uint32_t n_batch) {
    if (!q4_hgemm_supported(type) || !nrows || !ncols || n_batch > INT32_MAX)
        return false;
    if ((int64_t)ncols <= 0) return false;
    const bool bf16 = type == Q4_T_BF16;
    if (!bf16) {
        int be = 0, bb = 0;
        if (!dequant_geom((int)type, be, bb) || ncols % (uint64_t)be)
            return false;
    }
    LtState &s = lt();
    if (!s.handle || !s.wq || !s.xq || !s.ws) return false;
    if (std::getenv("Q4_HIPBLASLT_VERBOSE"))
        std::fprintf(stderr, "q4 hgemm: try ty=%u N=%llu K=%llu T=%u\n", type,
                     (unsigned long long)nrows, (unsigned long long)ncols,
                     n_batch);
    const int64_t T = n_batch, N = (int64_t)nrows, K = (int64_t)ncols,
                  ldy = N;
    const int64_t xrows =
        s.x_elems / K > 0 ? (s.x_elems / K < T ? s.x_elems / K : T) : 1;
    if (bf16) {
        if (q4_dg_on() && K % 64 == 0)
            return q4_dg_gemm(d_w, 1, (uint32_t)N, (uint32_t)K, d_x,
                              (uint32_t)T, d_y, (uint32_t)ldy, scale);
        for (int64_t t0 = 0; t0 < T; t0 += xrows) {
            const int64_t n = T - t0 < xrows ? T - t0 : xrows;
            if (!to_u16(d_x + t0 * K, s.xq, n * K, true) ||
                !gemm_u16(s, s.xq, (const uint16_t *)d_w, d_y + t0 * ldy, n,
                          N, K, ldy, scale, 0.0f, true))
                return false;
        }
        return true;
    }
    const int64_t wrows = W_ELEMS / K;
    if (wrows <= 0) return false;
    for (int64_t r0 = 0; r0 < N; r0 += wrows) {
        const int64_t n = N - r0 < wrows ? N - r0 : wrows;
        if (!dequant_f16((int)type, d_w, r0, n, K, s.wq)) return false;
        if (q4_dg_on() && K % 64 == 0) {
            if (!q4_dg_gemm(s.wq, 0, (uint32_t)n, (uint32_t)K, d_x,
                            (uint32_t)T, d_y + r0, (uint32_t)ldy, scale))
                return false;
            continue;
        }
        for (int64_t t0 = 0; t0 < T; t0 += xrows) {
            const int64_t m = T - t0 < xrows ? T - t0 : xrows;
            if (!to_u16(d_x + t0 * K, s.xq, m * K, false) ||
                !gemm_u16(s, s.xq, s.wq, d_y + t0 * ldy + r0, m, n, K, ldy,
                          scale, 0.0f, false))
                return false;
        }
    }
    return true;
}
