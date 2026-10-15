/* src/q4_mmq.cu - prefill expert GEMMs through the vendored MMQ kernels
 * (MIT; third_party/llama.cpp).  The wrapper shape and the host-side
 * shims below are adapted under that license (after Strata's MMQ
 * integration); see THIRD_PARTY_NOTICES.md.
 * Errors return false; only a violated ggml internal assert aborts. */
extern "C" {
#include "q4.h"
}

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

/* ---- ggml-cuda host symbols (shims; see THIRD_PARTY_NOTICES.md) ------ */
[[noreturn]] void ggml_cuda_error(const char *stmt, const char *func,
                                  const char *file, int line, const char *msg) {
    std::fprintf(stderr, "ggml-cuda (q4 mmq): %s: %s\n  in %s at %s:%d\n", msg,
                 stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_set_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }

void ggml_abort(const char *file, int line, const char *fmt, ...) {
    std::fprintf(stderr, "ggml-cuda (q4 mmq) %s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
    std::abort();
}

void ggml_log_internal(enum ggml_log_level level, const char *format, ...) {
    (void) level;
    va_list ap;
    va_start(ap, format);
    std::vfprintf(stderr, format, ap);
    va_end(ap);
}

namespace {

unsigned parse_hex(const std::string &value) {
    if (value.empty()) return 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 16);
    return end == value.c_str() || *end != '\0' ? 0 : (unsigned) parsed;
}

/* GGML's AMD cc encoding: OFFSET_AMD + gfx major * 0x100 + minor/stepping.
 * gfx1100 -> GGML_CUDA_CC_RDNA3 (0x01001100). */
int ggml_cuda_parse_amd_id(const char *device_name) {
    if (device_name == nullptr) return GGML_CUDA_CC_OFFSET_AMD;
    std::string arch(device_name);
    if (arch.compare(0, 3, "gfx") == 0) arch.erase(0, 3);
    const size_t suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);
    const size_t generic = arch.rfind("-generic");
    if (generic != std::string::npos && generic + 8 == arch.size())
        arch.resize(generic);
    unsigned major = 0, minor = 0;
    const size_t separator = arch.find('-');
    if (separator != std::string::npos) {
        major = parse_hex(arch.substr(0, separator));
        minor = parse_hex(arch.substr(separator + 1)) * 0x10;
    } else if (arch.size() >= 3) {
        minor = parse_hex(arch.substr(arch.size() - 2));
        major = parse_hex(arch.substr(0, arch.size() - 2));
    }
    return GGML_CUDA_CC_OFFSET_AMD + (int) (major * 0x100 + minor);
}

/* Buffers are kept and reused: MMQ asks for the same few sizes every launch
 * (its stream-k fixup tiles). */
struct CachingPool : ggml_cuda_pool {
    struct Buf { void *p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void *alloc(size_t size, size_t *actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto &b : bufs)
            if (!b.used && b.size >= size) {
                b.used = true;
                *actual_size = b.size;
                return b.p;
            }
        void *p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, size));
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void *ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto &b : bufs)
            if (b.p == ptr) {
                b.used = false;
                return;
            }
    }
    ~CachingPool() override {
        for (auto &b : bufs) (void) cudaFree(b.p);
    }
};

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

__global__ void iota_kernel(int32_t *dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

/* silu(gate)*up, non-interleaved: gu[r] = [gate n_ff | up n_ff]. */
__global__ void swiglu_kernel(const float *__restrict__ gu,
                              float *__restrict__ h, int64_t rows,
                              int64_t n_ff) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float g = gu[r * 2 * n_ff + k], u = gu[r * 2 * n_ff + n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

/* Expert gather into id-indexed buffers: ordinal i's L1 slot -> eid[i]'s
 * gate+up slice (gu + eid*gu_bytes) and down slice (d + eid*d_bytes). */
__global__ void gather16_k(const uint8_t *__restrict__ abase,
                           uint64_t astride, uint64_t goff, uint64_t doff,
                           uint64_t gu_bytes, uint64_t d_bytes,
                           uint64_t gu_dst, uint64_t d_dst,
                           const int32_t *__restrict__ eid,
                           const int32_t *__restrict__ slot,
                           uint8_t *__restrict__ gu, uint8_t *__restrict__ d) {
    const int i = (int) blockIdx.y;
    const uint8_t *src = abase + (uint64_t) slot[i] * astride;
    uint4 *gd = (uint4 *) (gu + (uint64_t) eid[i] * gu_dst);
    uint4 *dd = (uint4 *) (d + (uint64_t) eid[i] * d_dst);
    const uint64_t na = gu_bytes / 16, nc = d_bytes / 16;
    const uint64_t pa = (gu_dst - gu_bytes) / 16,
                 pc = (d_dst - d_bytes) / 16;
    const uint64_t t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const uint4 z = make_uint4(0, 0, 0, 0);
    /* the last weight row's k-iteration over-reads into the pad; it must
     * always be zeros, so the gather rewrites it every call (another
     * layer's stride may have put real bytes there). */
    if (t < na) gd[t] = ((const uint4 *) (src + goff))[t];
    else if (t < na + nc) dd[t - na] = ((const uint4 *) (src + doff))[t - na];
    else if (t < na + nc + pa) gd[na + (t - na - nc)] = z;
    else if (t < na + nc + pa + pc) dd[nc + (t - na - nc - pa)] = z;
}

__global__ void gather1_k(const uint8_t *__restrict__ abase, uint64_t astride,
                          uint64_t goff, uint64_t doff, uint64_t gu_bytes,
                          uint64_t d_bytes, uint64_t gu_dst, uint64_t d_dst,
                          const int32_t *__restrict__ eid,
                          const int32_t *__restrict__ slot,
                          uint8_t *__restrict__ gu, uint8_t *__restrict__ d) {
    const int i = (int) blockIdx.y;
    const uint8_t *src = abase + (uint64_t) slot[i] * astride;
    uint8_t *gd = gu + (uint64_t) eid[i] * gu_dst;
    uint8_t *dd = d + (uint64_t) eid[i] * d_dst;
    const uint64_t t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t gp = gu_dst - gu_bytes, dp = d_dst - d_bytes;
    if (t < gu_bytes) gd[t] = src[goff + t];
    else if (t < gu_bytes + d_bytes) dd[t - gu_bytes] = src[doff + t - gu_bytes];
    else if (t < gu_bytes + d_bytes + gp) gd[t - d_bytes] = 0;
    else if (t < gu_bytes + d_bytes + gp + dp) dd[t - gu_dst] = 0;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

cudaStream_t g_str_mmq() { return (cudaStream_t) q4_hip_stream(); }

ggml_backend_cuda_context *mmq_ctx() {
    static ggml_backend_cuda_context *ctx = [] {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
        return new ggml_backend_cuda_context(dev);
    }();
    return ctx;
}

int32_t *g_iota = nullptr;
int64_t g_iota_n = 0;

uint8_t *g_gu = nullptr, *g_dn = nullptr;
uint64_t g_gu_b = 0, g_dn_b = 0;
/* Lend mode (q4_mmq_wbuf_lend): g_gu/g_dn point into borrowed L1 slots
 * while a prompt is processed and are NULL otherwise — never hipMalloc'd
 * or freed here. */
int g_wb_lend = 0;

}  // namespace

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

std::unique_ptr<ggml_cuda_pool>
ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

const ggml_cuda_device_info &ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto &d = in.devices[id];
            d.cc = ggml_cuda_parse_amd_id(prop.gcnArchName);
            if ((d.cc & 0xff00) == 0)
                d.cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100 +
                       prop.minor * 0x10;
            d.smpbo = prop.sharedMemPerBlock;
            d.integrated = false;
            d.supports_cooperative_launch = false;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.warp_size = prop.warpSize;
            d.vmm = false;
            d.total_vram = prop.totalGlobalMem;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

/* ---- q4 C ABI --------------------------------------------------------- */

extern "C" int q4_mmq_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MMQ");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* Q4_MMQ_TUNE (RDNA3 only; other GPUs always run the stock launch):
 *   0  stock launch: grid sized by the total row count (tiles beyond an
 *      expert's rows are dead blocks - ~13M of them per 8K-token GEMM)
 *   1  grid sized by the largest expert (max_rows)
 *   2  (default) compacted (expert, j-tile) grid via a tile prefix table
 * q4_mmq_set_tune/q4_mmq_set_opt are for the microbench. */
static int g_tune = -1;
static int g_opt = -1;   /* >0: tile-size target override (bench/tuning) */
extern "C" void q4_mmq_set_tune(int v) { g_tune = v; }
extern "C" void q4_mmq_set_opt(int v) { g_opt = v; }
static int mmq_tune(void) {
    if (g_tune < 0) {
        const char *e = getenv("Q4_MMQ_TUNE");
        g_tune = e && e[0] ? atoi(e) : 2;
    }
    return g_tune;
}
static bool mmq_rdna3(void) {
    const ggml_cuda_device_info &info = ggml_cuda_info();
    const int id = ggml_cuda_get_device();
    return id < info.device_count && GGML_CUDA_CC_IS_RDNA3(info.devices[id].cc);
}

extern "C" uint32_t q4_mmq_group(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MMQ_GROUP");
        v = e && e[0] ? atoi(e) : (int) q4_pref_chunk();
        if (v < 512) v = 512;
        if (v > 8192) v = 8192;
    }
    return (uint32_t) v;
}

static int mmq_type_ok(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q2_0:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_Q8_0:
        return 1;
    default:
        return 0;
    }
}

extern "C" bool q4_mmq_supported(uint32_t type, uint64_t w_rows) {
    if (!mmq_type_ok((ggml_type) type)) return false;
    /* mul_mat_q_case's choice: "fallback" configs when the rows are not a
     * multiple of 128; then a tile size whose config exists for this card
     * and fits its shared memory. */
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info &info = ggml_cuda_info();
    if (info.device_count <= 0) return false;
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c =
                ggml_cuda_mmq_get_config((ggml_type) type, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT &&
                  mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (type < GGML_TYPE_COUNT && !said[type]) {
                said[type] = true;
                std::fprintf(stderr,
                             "q4 mmq: no tile for type %u (%llu rows) on GPU "
                             "%d (cc %d, %zu shmem): non-MMQ path\n",
                             type, (unsigned long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

extern "C" uint64_t q4_mmq_q8_bytes(uint64_t rows, uint64_t cols) {
    return rows * (uint64_t) pad512((int64_t) cols) /
                   (4 * QK8_1) * sizeof(block_q8_1_mmq) +
           128 * sizeof(block_q8_1_mmq);
}

static int mmq_dbg(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_MMQ_DEBUG") != nullptr;
    return v;
}

static bool mmq_sync(const char *what) {
    if (mmq_dbg()) {
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaGetLastError() != cudaSuccess) {
            std::fprintf(stderr, "q4 mmq: fault after %s\n", what);
            return false;
        }
        return true;
    }
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool q4_mmq_quantize(const float *x, const int32_t *ids, void *xq,
                                uint32_t w_type, uint64_t cols,
                                uint64_t rows) {
    if (!x || !xq || !cols) return false;
    if ((int64_t) rows <= 0) return true;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) w_type, (int64_t) cols,
                           (int64_t) cols, (int64_t) (rows * cols),
                           (int64_t) (rows * cols), pad512((int64_t) cols),
                           (int64_t) rows, 1, 1, g_str_mmq());
    return mmq_sync("quantize");
}

template <ggml_type T>
static bool mmq_case(ggml_backend_cuda_context &ctx, mmq_args &a,
                     uint64_t expert_stride_bytes, cudaStream_t s) {
    a.stride_row_x = a.ncols_x / ggml_cuda_type_traits<T>::qk;
    a.stride_channel_x =
        (int64_t) (expert_stride_bytes / ggml_cuda_type_traits<T>::bs);
    mul_mat_q_case<T>(ctx, a, s);
    return mmq_sync("mul_id");
}

extern "C" bool q4_mmq_mul_id(const void *w, uint32_t type, uint64_t w_rows,
                              uint64_t w_cols, uint64_t expert_stride_bytes,
                              uint32_t n_expert, const void *xq,
                              const int32_t *bounds, uint32_t max_rows,
                              uint64_t total_rows, float *dst,
                              uint64_t ld_dst) {
    if (!w || !xq || !bounds || !dst || !n_expert || !w_cols || !ld_dst)
        return false;
    if (!total_rows || !max_rows) return true;
    /* ids_dst = identity: dst row i is activation row i.  The kernel loads
     * J-row id tiles unconditionally, so it may read up to J-1 (127) ints
     * past the last row — keep that much slack in the buffer. */
    if (g_iota_n < (int64_t) total_rows + 256) {
        int64_t want = (int64_t) total_rows + 256;
        int32_t *nb = nullptr;
        if (cudaMalloc(&nb, want * sizeof(int32_t)) != cudaSuccess)
            return false;
        if (g_iota) (void) cudaFree(g_iota);
        g_iota = nb;
        g_iota_n = want;
        iota_kernel<<<blocks(want), 256, 0, g_str_mmq()>>>(g_iota, want);
    }
    const int tune = mmq_rdna3() ? mmq_tune() : 0;
    mmq_args a = {(const char *) w, (ggml_type) type, (const int *) xq,
                  g_iota, bounds, dst, nullptr,
                  (int64_t) w_cols, (int64_t) w_rows, (int64_t) total_rows, 0,
                  (int64_t) total_rows, (int64_t) ld_dst,
                  (int64_t) n_expert, (int64_t) n_expert, 0, 0, 0,
                  1, 1, 0, 0, 0,
                  tune >= 1 ? (int64_t) max_rows : (int64_t) total_rows,
                  g_opt > 0 ? (int64_t) g_opt : (int64_t) max_rows,
                  tune >= 2, (int64_t) total_rows};
    ggml_backend_cuda_context *ctx = mmq_ctx();
    switch ((ggml_type) type) {
    case GGML_TYPE_Q2_0:
        return mmq_case<GGML_TYPE_Q2_0>(*ctx, a, expert_stride_bytes, g_str_mmq());
    case GGML_TYPE_IQ2_S:
        return mmq_case<GGML_TYPE_IQ2_S>(*ctx, a, expert_stride_bytes, g_str_mmq());
    case GGML_TYPE_IQ3_XXS:
        return mmq_case<GGML_TYPE_IQ3_XXS>(*ctx, a, expert_stride_bytes, g_str_mmq());
    case GGML_TYPE_IQ3_S:
        return mmq_case<GGML_TYPE_IQ3_S>(*ctx, a, expert_stride_bytes, g_str_mmq());
    case GGML_TYPE_IQ4_NL:
        return mmq_case<GGML_TYPE_IQ4_NL>(*ctx, a, expert_stride_bytes, g_str_mmq());
    case GGML_TYPE_IQ4_XS:
        return mmq_case<GGML_TYPE_IQ4_XS>(*ctx, a, expert_stride_bytes, g_str_mmq());
    case GGML_TYPE_Q8_0:
        return mmq_case<GGML_TYPE_Q8_0>(*ctx, a, expert_stride_bytes, g_str_mmq());
    default:
        return false;
    }
}

/* MMQ's last k-iteration reads a full span past the row's blocks when the
 * block count doesn't divide it; on the final weight row that lands just
 * past the expert's region, and whatever bytes sit there feed the dot via
 * the scale field (NaN-scale garbage poisons it even with zeroed xq).  Each
 * gathered expert therefore carries a zero pad >= one span, a multiple of
 * the row bytes so the stride stays divisible by the block size, and a
 * multiple of 16 so the fast gather path stays vectorized. */
extern "C" uint64_t q4_mmq_pad(uint64_t row_bytes) {
    uint64_t p = row_bytes;
    while (p < 1024 || (p & 15)) p += row_bytes;
    return p;
}

/* Max gate+up / down expert-bytes over the layers MMQ can actually run
 * (gate type == up type, both tiled).  Both the planner's scratch
 * accounting and q4_mmq_wbuf's one-time allocation use this, so they
 * cannot disagree. */
extern "C" void q4_mmq_wbuf_size(const q4_gguf *g, uint64_t *gu_max,
                                 uint64_t *d_max) {
    static const q4_gguf *last_g = nullptr;
    static uint64_t gu = 0, d = 0;
    if (g != last_g) {
        gu = d = 0;
        if (g && q4_mmq_on()) {
            uint64_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
            for (int32_t L = 0; L < g->n_layer; L++) {
                const q4_tensor *tg, *tu, *td;
                uint64_t og, ou, od, tot;
                if (!q4_expert_parts(g, L, &tg, &tu, &td, &og, &ou, &od,
                                     &tot))
                    continue;
                /* q4_xg reads these layers' slots directly */
                if (q4_xg_on() && tg->ggml_type == tu->ggml_type &&
                    q4_xg_gu_type(tg->ggml_type) &&
                    q4_xg_dn_type(td->ggml_type) &&
                    q4_xg_shape_ok((uint32_t)g->n_embd, (uint32_t)n_ff))
                    continue;
                if (tg->ggml_type == tu->ggml_type &&
                    q4_mmq_supported(tg->ggml_type, 2ull * n_ff) &&
                    q4_mmq_supported(td->ggml_type, (uint64_t) g->n_embd)) {
                    const uint64_t gp =
                        q4_mmq_pad(q4_row_bytes(tg->ggml_type, g->n_embd));
                    const uint64_t dp =
                        q4_mmq_pad(q4_row_bytes(td->ggml_type, n_ff));
                    if (od - og + gp > gu) gu = od - og + gp;
                    if (tot - od + dp > d) d = tot - od + dp;
                }
            }
        }
        last_g = g;
    }
    if (gu_max) *gu_max = gu;
    if (d_max) *d_max = d;
}

/* Expert-id-indexed GU/D weight buffers, allocated once at the
 * q4_mmq_wbuf_size maxima (the planner counts them in scratch_bytes). */
extern "C" bool q4_mmq_wbuf(uint64_t gu_bytes, uint64_t d_bytes,
                            uint32_t n_exp, uint8_t **gu, uint8_t **d) {
    if (!gu || !d || !n_exp) return false;
    /* MMQ loads a full ITER_K span per iteration; for weight rows whose
     * block count is not a multiple (640 cols = 20 IQ4_NL blocks vs
     * 8-block iterations) the last row's tail read runs past the last
     * expert's matrix — pad the buffer so it stays mapped.  The excess
     * k-terms multiply the zero-padded xq planes, so results stay exact. */
    constexpr uint64_t tail = 65536;
    uint64_t gneed = gu_bytes * n_exp + tail,
             dneed = d_bytes * n_exp + tail;
    if (g_wb_lend) {
        /* Borrowed buffers only: outside a lend window (decode) there are
         * none, and growing here would allocate ~1.3 GiB behind the
         * planner's back. The caller falls back to the non-MMQ kernels. */
        if (!g_gu || !g_dn || gneed > g_gu_b || dneed > g_dn_b) return false;
        *gu = g_gu;
        *d = g_dn;
        return true;
    }
    if (gneed > g_gu_b) {
        uint8_t *nb = nullptr;
        if (cudaMalloc(&nb, gneed) != cudaSuccess) return false;
        /* Zero-fill: the gather writes only experts used by the current
         * layer.  The last weight row's over-read lands in the NEXT
         * expert's region — garbage bytes there can hold NaN fp16 scales,
         * and NaN*0 (padded y blocks) still yields NaN. */
        if (cudaMemsetAsync(nb, 0, gneed, g_str_mmq()) != cudaSuccess)
            return false;
        if (g_gu) (void) cudaFree(g_gu);
        g_gu = nb;
        g_gu_b = gneed;
    }
    if (dneed > g_dn_b) {
        uint8_t *nb = nullptr;
        if (cudaMalloc(&nb, dneed) != cudaSuccess) return false;
        if (cudaMemsetAsync(nb, 0, dneed, g_str_mmq()) != cudaSuccess)
            return false;
        if (g_dn) (void) cudaFree(g_dn);
        g_dn = nb;
        g_dn_b = dneed;
    }
    *gu = g_gu;
    *d = g_dn;
    return true;
}

/* Bytes q4_mmq_wbuf needs for (gu, d) at the q4_mmq_wbuf_size maxima. */
extern "C" void q4_mmq_wbuf_need(const q4_gguf *g, uint64_t *gu_need,
                                 uint64_t *d_need) {
    uint64_t gu = 0, d = 0;
    q4_mmq_wbuf_size(g, &gu, &d);
    const uint32_t n_exp = g && g->n_expert ? g->n_expert : 0;
    *gu_need = gu && n_exp ? gu * n_exp + 65536 : 0;
    *d_need = d && n_exp ? d * n_exp + 65536 : 0;
}

/* Point the MMQ weight buffers at borrowed memory (gu/d non-NULL) or drop
 * them (NULL).  The first call switches q4_mmq_wbuf to lend mode for good:
 * it never allocates again.  Each borrow zero-fills (see q4_mmq_wbuf: the
 * last row's over-read must not meet NaN fp16 scales). */
extern "C" bool q4_mmq_wbuf_lend(void *gu, uint64_t gu_bytes, void *d,
                                 uint64_t d_bytes) {
    if (!g_wb_lend) {
        if (g_gu) (void)cudaFree(g_gu);
        if (g_dn) (void)cudaFree(g_dn);
        g_wb_lend = 1;
    }
    g_gu = (uint8_t *)gu;
    g_dn = (uint8_t *)d;
    g_gu_b = gu ? gu_bytes : 0;
    g_dn_b = d ? d_bytes : 0;
    if (gu && cudaMemsetAsync(gu, 0, gu_bytes, g_str_mmq()) != cudaSuccess)
        return false;
    if (d && cudaMemsetAsync(d, 0, d_bytes, g_str_mmq()) != cudaSuccess)
        return false;
    return true;
}

extern "C" bool q4_mmq_gather(const uint8_t *abase, uint64_t astride,
                              uint64_t goff, uint64_t gu_bytes, uint64_t doff,
                              uint64_t d_bytes, uint64_t gu_dst, uint64_t d_dst,
                              const int32_t *d_eid, const int32_t *d_slot,
                              uint32_t n, uint8_t *d_gu, uint8_t *d_d) {
    if (!abase || !d_eid || !d_slot || !d_gu || !d_d || !n) return !n;
    const uintptr_t al =
        (uintptr_t) abase | astride | goff | gu_bytes | doff | d_bytes |
        gu_dst | d_dst | (uintptr_t) d_gu | (uintptr_t) d_d;
    if (mmq_dbg())
        std::fprintf(stderr, "mmq ptr gu=%p+%llu d=%p+%llu abase=%p\n",
                     (void *) d_gu, (unsigned long long) (gu_dst * n),
                     (void *) d_d, (unsigned long long) (d_dst * n),
                     (void *) abase),
        std::fflush(stderr);
    dim3 grid(blocks((int64_t) (gu_dst + d_dst) / (al % 16 ? 1 : 16)), n);
    if (al % 16 == 0)
        gather16_k<<<grid, 256, 0, g_str_mmq()>>>(
            abase, astride, goff, doff, gu_bytes, d_bytes, gu_dst, d_dst,
            d_eid, d_slot, d_gu, d_d);
    else
        gather1_k<<<grid, 256, 0, g_str_mmq()>>>(
            abase, astride, goff, doff, gu_bytes, d_bytes, gu_dst, d_dst,
            d_eid, d_slot, d_gu, d_d);
    return mmq_sync("gather");
}

extern "C" bool q4_mmq_swiglu(const float *gu, float *h, uint64_t rows,
                              uint64_t n_ff) {
    if (!gu || !h) return false;
    if (!rows || !n_ff) return true;
    swiglu_kernel<<<blocks((int64_t) (rows * n_ff)), 256, 0, g_str_mmq()>>>(
        gu, h, (int64_t) rows, (int64_t) n_ff);
    return mmq_sync("swiglu");
}
