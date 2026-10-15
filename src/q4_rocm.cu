#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

extern "C" {
#include "q4.h"
}

#define Q4_IQ_DECL static __constant__
#include "q4_iq_grids.inc"

static __constant__ int8_t q4_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

#define Q4_HIP_CHECK(x)                                                        \
    do {                                                                       \
        hipError_t _e = (x);                                                   \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "q4 hip: %s (%s:%d)\n", hipGetErrorString(_e),     \
                    __FILE__, __LINE__);                                       \
            return false;                                                      \
        }                                                                      \
    } while (0)

static bool g_ok = false;
static float *g_dx = nullptr;
static float *g_dy = nullptr;
static uint8_t *g_dw = nullptr;
static size_t g_dx_n = 0, g_dy_n = 0, g_dw_n = 0;
/* Out-of-place write-back target for gdn_scan_k's shared q/k conv rows:
 * blocks may finish (and write) before stragglers have read hist. */
static float *g_scanqk = nullptr;
static size_t g_scanqk_n = 0;
/* gdnrec pipeline scratch: conv+silu h [T][ch], pre-norm output oc
 * [T][nvh*d], gates+beta [2][T][nvh].  Grown lazily to the largest T seen
 * (the planner reserves q4_pref_chunk, see scratch_bytes). */
static float *g_scanh = nullptr;
static size_t g_scanh_n = 0;
static float *g_scanoc = nullptr;
static size_t g_scanoc_n = 0;
static float *g_scangb = nullptr;
static size_t g_scangb_n = 0;
/* Lend mode (q4_hip_gdn_scan_lend): the rec-path buffers above point into
 * borrowed L1 slots during a prompt and are NULL otherwise. */
static int g_scan_lend = 0;
static uint32_t *g_tk_hist, *g_tk_st, *g_tk_cnt, *g_tk_off;
static int32_t *g_tk_chosen;
static float *g_attn_part; /* (n_kvh<=4,S,gqa<=12,hd+2) split-dec partials */
static int32_t *g_un_blk;               /* prompt QSA union (qsa_union_k) */
static uint32_t *g_un_mask, *g_un_cnt;
static float *g_sk_part;
static struct { uint32_t t; uint64_t r, c, b, n; } g_shapes[256];
static int g_shape_n;
/* Q4_SHAPES for the multi-segment GEMV: which segment mixes actually run. */
typedef struct {
    uint64_t n;
    uint32_t in, nseg, ty[8], rows[8];
} q4_segmix_t;
static q4_segmix_t g_mixes[64];
static int g_mix_n;
static int shapes = -1;   /* Q4_SHAPES, resolved on first gemv_m call */
/* Expert arenas: 0 = main model, 1 = MTP draft head. Separate because the
 * slot sizes differ and the caches are independent LRU sets. */
/* Arenas 2/3 are the sb verify miss-scratch shadows of 0/1 (see
 * moe_dev_n_sb): unique CPU-miss experts are DMA'd in and computed by the
 * same wmoe rows instead of joining the CPU pool. */
#define Q4_HIP_ARENAS 4
static uint8_t *g_exp[Q4_HIP_ARENAS] = {nullptr, nullptr, nullptr, nullptr};
static uint64_t g_exp_slot[Q4_HIP_ARENAS] = {0, 0};
static uint32_t g_exp_n[Q4_HIP_ARENAS] = {0, 0};

/* ---- KV paging: arena 0 in 2 MiB physical chunks ------------------------
 * With q4_hip_arena_vmm(1) the main expert arena is a reserved VA range
 * backed chunk by chunk (hipMemCreate + hipMemMap) instead of a hipMalloc,
 * so slots the expert cache blocks can hand their physical chunks to the
 * context-sized buffers (QSA KV, indexer keys), which reserve VA only and
 * map chunks as the context grows (q4_hip_vreg_*).  Unmapped chunks sit in
 * a pool between owners.  Same physical VRAM as the fixed layout at full
 * context; at a short context the KV's share stays expert slots.  Reads
 * from VMM chunks measured the same bandwidth as hipMalloc (stream ~870,
 * 2 MiB-slot gathers ~850 GB/s).  Callers drain the device before any
 * unmap. */
#define Q4_VCH (2ull << 20)
static bool g_avmm_want;
static uint8_t *g_ava;              /* arena 0 VA when VMM-backed */
static uint64_t g_ava_len;
static hipMemGenericAllocationHandle_t *g_ach;
static uint8_t *g_ach_map;
static uint32_t g_ach_n;
static hipMemGenericAllocationHandle_t *g_vpool;
static uint32_t g_vpool_n, g_vpool_cap;
struct q4_vreg {
    uint8_t *va;
    uint64_t len;
    uint32_t nch;
    hipMemGenericAllocationHandle_t *h;
    hipMemGenericAllocationHandle_t *home; /* pinned: the chunk's only handle */
    uint8_t *map;
};
#define Q4_VREG_MAX 8
static q4_vreg g_vreg[Q4_VREG_MAX];

static hipMemAllocationProp vmm_prop(void) {
    hipMemAllocationProp prop = {};
    int dev = 0;
    (void)hipGetDevice(&dev);
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    return prop;
}

static bool vmm_map(uint8_t *va, hipMemGenericAllocationHandle_t h) {
    hipMemAllocationProp prop = vmm_prop();
    hipMemAccessDesc ad = {};
    ad.location = prop.location;
    ad.flags = hipMemAccessFlagsProtReadWrite;
    if (hipMemMap(va, Q4_VCH, 0, h, 0) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    if (hipMemSetAccess(va, Q4_VCH, &ad, 1) != hipSuccess) {
        (void)hipMemUnmap(va, Q4_VCH);
        (void)hipGetLastError();
        return false;
    }
    return true;
}

/* Pinned pairing (Q4_VMM_PIN, default on): a VA is only ever mapped to one
 * physical chunk - an arena chunk to the handle it was created with, a KV
 * chunk to the handle it got at its first map (its home).  A copy into a VA
 * that had been remapped to a different chunk landed in the chunk the VA had
 * before (2026-10-15: slot uploads after a KV trim + reclaim, with SDMA and
 * with blit kernels, a sync and 500 ms after the remap included).  With one
 * chunk per VA a stale translation still names the right memory.  A handle
 * moves between exactly two VAs: its arena chunk and the KV chunk that took
 * it as home. */
static int g_vpin = -1;
static hipMemGenericAllocationHandle_t *g_ach_orig;
static std::unordered_set<uint64_t> g_claimed; /* handles that are a KV home */
static std::unordered_map<uint64_t, uint32_t> g_hidx; /* handle -> arena chunk */

static bool vpin(void) {
    if (g_vpin < 0) {
        const char *e = getenv("Q4_VMM_PIN");
        g_vpin = !(e && e[0] == '0');
    }
    return g_vpin != 0;
}

extern "C" int q4_hip_vmm_pinned(void) { return vpin() ? 1 : 0; }

static int pool_find(hipMemGenericAllocationHandle_t h) {
    for (uint32_t i = g_vpool_n; i-- > 0;)
        if (g_vpool[i] == h) return (int)i;
    return -1;
}

/* Remove pool entry i keeping the order (the newest stays on top). */
static void pool_remove(int i) {
    memmove(g_vpool + i, g_vpool + i + 1,
            (size_t)(g_vpool_n - (uint32_t)i - 1) * sizeof(*g_vpool));
    g_vpool_n--;
}

static bool vpool_push(hipMemGenericAllocationHandle_t h) {
    if (g_vpool_n == g_vpool_cap) {
        uint32_t nc = g_vpool_cap ? 2 * g_vpool_cap : 1024;
        void *p = realloc(g_vpool, (size_t)nc * sizeof(*g_vpool));
        if (!p) return false;
        g_vpool = (hipMemGenericAllocationHandle_t *)p;
        g_vpool_cap = nc;
    }
    g_vpool[g_vpool_n++] = h;
    return true;
}

static q4_vreg *vreg_of(const void *va);

/* Unmap a region and release its VA; its chunks go to the pool.  Pinned
 * VMM: the regions are reserved once per session (q4_sess_open); a VA range
 * reserved again later must not inherit the released homes' VAs. */
static bool vreg_free(void *va) {
    q4_vreg *v = vreg_of(va);
    if (!v) return false;
    for (uint32_t c = 0; c < v->nch; c++)
        if (v->map[c]) {
            (void)hipMemUnmap(v->va + (uint64_t)c * Q4_VCH, Q4_VCH);
            (void)vpool_push(v->h[c]);
        }
    for (uint32_t c = 0; c < v->nch; c++)
        if (v->home && v->home[c]) g_claimed.erase((uint64_t)v->home[c]);
    (void)hipMemAddressFree(v->va, v->len);
    free(v->h);
    free(v->home);
    free(v->map);
    *v = q4_vreg{};
    return true;
}

static void vmm_arena_free(void) {
    if (!g_ava) return;
    for (int r = 0; r < Q4_VREG_MAX; r++)
        if (g_vreg[r].va) (void)vreg_free(g_vreg[r].va);
    for (uint32_t c = 0; c < g_ach_n; c++) {
        if (g_ach_map[c]) (void)hipMemUnmap(g_ava + (uint64_t)c * Q4_VCH, Q4_VCH);
        if (g_ach[c]) (void)hipMemRelease(g_ach[c]);
    }
    for (uint32_t i = 0; i < g_vpool_n; i++) (void)hipMemRelease(g_vpool[i]);
    (void)hipMemAddressFree(g_ava, g_ava_len);
    free(g_ach);
    free(g_ach_orig);
    g_ach_orig = nullptr;
    g_claimed.clear();
    g_hidx.clear();
    free(g_ach_map);
    g_ach = nullptr;
    g_ach_map = nullptr;
    g_ach_n = 0;
    g_ava = nullptr;
    g_ava_len = 0;
    g_vpool_n = 0;
}

static bool vmm_arena_alloc(uint64_t n) {
    hipMemAllocationProp prop = vmm_prop();
    const uint32_t nch = (uint32_t)((n + Q4_VCH - 1) / Q4_VCH);
    const uint64_t len = (uint64_t)nch * Q4_VCH;
    void *va = nullptr;
    if (hipMemAddressReserve(&va, len, Q4_VCH, 0, 0) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    g_ava = (uint8_t *)va;
    g_ava_len = len;
    g_ach = (hipMemGenericAllocationHandle_t *)calloc(nch, sizeof(*g_ach));
    g_ach_orig = (hipMemGenericAllocationHandle_t *)calloc(nch, sizeof(*g_ach_orig));
    g_ach_map = (uint8_t *)calloc(nch, 1);
    g_ach_n = nch;
    if (!g_ach || !g_ach_orig || !g_ach_map) {
        vmm_arena_free();
        return false;
    }
    for (uint32_t c = 0; c < nch; c++) {
        if (hipMemCreate(&g_ach[c], Q4_VCH, &prop, 0) != hipSuccess) {
            (void)hipGetLastError();
            g_ach[c] = 0;
            vmm_arena_free();
            return false;
        }
        if (!vmm_map(g_ava + (uint64_t)c * Q4_VCH, g_ach[c])) {
            vmm_arena_free();
            return false;
        }
        g_ach_orig[c] = g_ach[c];
        g_hidx[(uint64_t)g_ach[c]] = c;
        g_ach_map[c] = 1;
    }
    return true;
}

extern "C" void q4_hip_arena_vmm(int on) { g_avmm_want = on != 0; }
extern "C" int q4_hip_arena_is_vmm(void) { return g_ava != nullptr; }
extern "C" uint64_t q4_hip_vchunk(void) { return Q4_VCH; }
extern "C" uint32_t q4_hip_vpool_n(void) { return g_vpool_n; }

/* Make arena 0's donated byte range exactly [lo, hi): chunks wholly inside
 * it are unmapped into the pool, every other chunk is mapped (taking pool
 * chunks back).  False when the pool cannot cover a remap. */
extern "C" bool q4_hip_arena_donate(uint64_t lo, uint64_t hi) {
    if (!g_ava) return false;
    for (uint32_t c = 0; c < g_ach_n; c++) {
        const uint64_t a = (uint64_t)c * Q4_VCH, b = a + Q4_VCH;
        const bool in = a >= lo && b <= hi;
        if (in && g_ach_map[c]) {
            if (hipMemUnmap(g_ava + a, Q4_VCH) != hipSuccess) {
                (void)hipGetLastError();
                return false;
            }
            g_ach_map[c] = 0;
            if (!vpool_push(g_ach[c])) return false;
            g_ach[c] = 0;
        } else if (!in && !g_ach_map[c]) {
            int i = vpin() ? pool_find(g_ach_orig[c]) : (int)g_vpool_n - 1;
            if (i < 0) return false;
            hipMemGenericAllocationHandle_t h = g_vpool[i];
            if (!vmm_map(g_ava + a, h)) return false;
            pool_remove(i);
            g_ach[c] = h;
            g_ach_map[c] = 1;
        }
    }
    return true;
}

/* Q4_VMM_AUDIT: every physical chunk must be in exactly one place - an
 * arena chunk, a region chunk or the free pool.  Logs duplicates (a chunk
 * mapped twice aliases two buffers) and returns their count. */
extern "C" uint32_t q4_hip_vmm_audit(const char *tag) {
    struct ent { uint64_t h; int where; uint32_t idx; };
    uint32_t n = g_vpool_n;
    for (uint32_t c = 0; c < g_ach_n; c++) n += g_ach_map[c];
    for (int r = 0; r < Q4_VREG_MAX; r++)
        if (g_vreg[r].va)
            for (uint32_t c = 0; c < g_vreg[r].nch; c++) n += g_vreg[r].map[c];
    ent *e = (ent *)malloc((size_t)(n ? n : 1) * sizeof(ent));
    if (!e) return 0;
    uint32_t k = 0;
    for (uint32_t i = 0; i < g_vpool_n; i++) e[k++] = {(uint64_t)g_vpool[i], -1, i};
    for (uint32_t c = 0; c < g_ach_n; c++)
        if (g_ach_map[c]) e[k++] = {(uint64_t)g_ach[c], 0, c};
    for (int r = 0; r < Q4_VREG_MAX; r++)
        if (g_vreg[r].va)
            for (uint32_t c = 0; c < g_vreg[r].nch; c++)
                if (g_vreg[r].map[c]) e[k++] = {(uint64_t)g_vreg[r].h[c], 1 + r, c};
    std::sort(e, e + k, [](const ent &x, const ent &y) { return x.h < y.h; });
    uint32_t dup = 0;
    for (uint32_t i = 1; i < k; i++)
        if (e[i].h == e[i - 1].h) {
            if (dup < 8)
                fprintf(stderr, "q4 vmm audit %s: chunk %llx in %d[%u] and %d[%u] "
                        "(-1 pool, 0 arena, r+1 region)\n", tag,
                        (unsigned long long)e[i].h, e[i - 1].where, e[i - 1].idx,
                        e[i].where, e[i].idx);
            dup++;
        }
    fprintf(stderr, "q4 vmm audit %s: %u chunks (pool %u), %u duplicated\n", tag,
            k, g_vpool_n, dup);
    free(e);
    return dup;
}

/* Chunks a q4_hip_arena_donate(lo, hi) would have to map back. */
extern "C" uint32_t q4_hip_arena_remap_need(uint64_t lo, uint64_t hi) {
    uint32_t n = 0;
    for (uint32_t c = 0; c < g_ach_n; c++) {
        const uint64_t a = (uint64_t)c * Q4_VCH, b = a + Q4_VCH;
        const bool need = !g_ach_map[c] && !(a >= lo && b <= hi);
        /* pinned: a chunk whose own handle the KV still holds cannot come back */
        if (need && vpin() && pool_find(g_ach_orig[c]) < 0) return UINT32_MAX;
        n += need;
    }
    return n;
}

/* Context-sized buffer: VA only; chunks are mapped by q4_hip_vreg_map. */
extern "C" void *q4_hip_vreg_alloc(uint64_t bytes) {
    if (!bytes) return nullptr;
    int r = 0;
    while (r < Q4_VREG_MAX && g_vreg[r].va) r++;
    if (r == Q4_VREG_MAX) return nullptr;
    q4_vreg *v = &g_vreg[r];
    const uint32_t nch = (uint32_t)((bytes + Q4_VCH - 1) / Q4_VCH);
    void *va = nullptr;
    if (hipMemAddressReserve(&va, (uint64_t)nch * Q4_VCH, Q4_VCH, 0, 0) !=
        hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    v->h = (hipMemGenericAllocationHandle_t *)calloc(nch, sizeof(*v->h));
    v->home = (hipMemGenericAllocationHandle_t *)calloc(nch, sizeof(*v->home));
    v->map = (uint8_t *)calloc(nch, 1);
    if (!v->h || !v->home || !v->map) {
        free(v->h);
        free(v->home);
        free(v->map);
        (void)hipMemAddressFree(va, (uint64_t)nch * Q4_VCH);
        *v = q4_vreg{};
        return nullptr;
    }
    v->va = (uint8_t *)va;
    v->len = (uint64_t)nch * Q4_VCH;
    v->nch = nch;
    return va;
}

static q4_vreg *vreg_of(const void *va) {
    for (int r = 0; r < Q4_VREG_MAX; r++)
        if (g_vreg[r].va && g_vreg[r].va == va) return &g_vreg[r];
    return nullptr;
}

/* Unmapped chunks overlapping [off, off + bytes). */
extern "C" uint32_t q4_hip_vreg_need(const void *va, uint64_t off,
                                     uint64_t bytes) {
    q4_vreg *v = vreg_of(va);
    if (!v || !bytes) return 0;
    uint64_t c0 = off / Q4_VCH, c1 = (off + bytes + Q4_VCH - 1) / Q4_VCH;
    if (c1 > v->nch) c1 = v->nch;
    uint32_t n = 0;
    for (uint64_t c = c0; c < c1; c++) n += !v->map[c];
    return n;
}

/* Chunks the pool cannot supply for mapping all n ranges right now: with
 * pinning, homes not in the pool plus first maps beyond the unclaimed
 * handles; without, the plain count beyond the pool. */
extern "C" uint32_t q4_hip_vreg_short(void *const *vas, const uint64_t *offs,
                                      const uint64_t *lens, int n) {
    uint32_t need = 0, miss = 0, fresh = 0;
    for (int k = 0; k < n; k++) {
        q4_vreg *v = vreg_of(vas[k]);
        if (!v || !lens[k]) continue;
        uint64_t c0 = offs[k] / Q4_VCH, c1 = (offs[k] + lens[k] + Q4_VCH - 1) / Q4_VCH;
        if (c1 > v->nch) c1 = v->nch;
        for (uint64_t c = c0; c < c1; c++) {
            if (v->map[c]) continue;
            need++;
            if (!vpin()) continue;
            if (v->home[c]) miss += pool_find(v->home[c]) < 0;
            else fresh++;
        }
    }
    if (!vpin()) return need > g_vpool_n ? need - g_vpool_n : 0;
    uint32_t unclaimed = 0;
    for (uint32_t i = 0; i < g_vpool_n; i++)
        unclaimed += !g_claimed.count((uint64_t)g_vpool[i]);
    return miss + (fresh > unclaimed ? fresh - unclaimed : 0);
}

/* Map pool chunks under [off, off + bytes); false when the pool ran dry. */
extern "C" bool q4_hip_vreg_map(void *va, uint64_t off, uint64_t bytes) {
    q4_vreg *v = vreg_of(va);
    if (!v) return false;
    uint64_t c0 = off / Q4_VCH, c1 = (off + bytes + Q4_VCH - 1) / Q4_VCH;
    if (c1 > v->nch) c1 = v->nch;
    for (uint64_t c = c0; c < c1; c++) {
        if (v->map[c]) continue;
        int i = -1;
        if (!vpin()) {
            i = (int)g_vpool_n - 1;
        } else if (v->home[c]) {
            i = pool_find(v->home[c]);
        } else {
            /* first map: the unowned handle of the highest arena chunk.  The
             * KV grows a position step at a time (sess_kv_reserve) and the
             * arena frontier moves down, so later steps own lower chunks:
             * trimming the top steps frees the chunks just above the
             * frontier, which is what a reclaim maps back first. */
            int64_t best = -1;
            for (int k = (int)g_vpool_n - 1; k >= 0; k--) {
                if (g_claimed.count((uint64_t)g_vpool[k])) continue;
                auto it = g_hidx.find((uint64_t)g_vpool[k]);
                const int64_t ci = it == g_hidx.end() ? 0 : (int64_t)it->second;
                if (ci > best) { best = ci; i = k; }
            }
        }
        if (i < 0) return false;
        hipMemGenericAllocationHandle_t h = g_vpool[i];
        if (!vmm_map(v->va + c * Q4_VCH, h)) return false;
        pool_remove(i);
        if (vpin() && !v->home[c]) {
            v->home[c] = h;
            g_claimed.insert((uint64_t)h);
        }
        v->h[c] = h;
        v->map[c] = 1;
    }
    return true;
}

/* Unmap every chunk of the region that overlaps none of the n kept ranges
 * (offs[i], lens[i]); returns the chunks given back to the pool. */
extern "C" uint32_t q4_hip_vreg_trim(void *va, const uint64_t *offs,
                                     const uint64_t *lens, int n) {
    q4_vreg *v = vreg_of(va);
    if (!v) return 0;
    uint32_t put = 0;
    for (uint32_t c = 0; c < v->nch; c++) {
        if (!v->map[c]) continue;
        const uint64_t a = (uint64_t)c * Q4_VCH, b = a + Q4_VCH;
        bool keep = false;
        for (int i = 0; i < n && !keep; i++)
            keep = lens[i] && offs[i] < b && offs[i] + lens[i] > a;
        if (keep) continue;
        if (hipMemUnmap(v->va + a, Q4_VCH) != hipSuccess) {
            (void)hipGetLastError();
            continue;
        }
        v->map[c] = 0;
        if (!vpool_push(v->h[c])) break;
        v->h[c] = 0;
        put++;
    }
    return put;
}

/* Mapped bytes of a region (for logs). */
extern "C" uint64_t q4_hip_vreg_mapped(const void *va) {
    q4_vreg *v = vreg_of(va);
    if (!v) return 0;
    uint64_t n = 0;
    for (uint32_t c = 0; c < v->nch; c++) n += v->map[c];
    return n * Q4_VCH;
}

static hipStream_t g_copy = nullptr;
/* All compute launches go on this stream so stream capture can graph them.
 * hipStreamLegacy capture is unsupported on ROCm, so a real stream it is. */
static hipStream_t g_str = nullptr;
/* 1 while g_str is capturing a graph: blocking/legacy-stream calls are
 * illegal then, so helpers must take their pure-kernel path. */
static int g_capturing = 0;

/* Step-profiler marker events (allocated up front so nothing is created
 * during graph capture). */
#define Q4_MK_SLOTS 16
static hipEvent_t g_mk0[Q4_MK_SLOTS], g_mk1[Q4_MK_SLOTS];
static int g_mk_armed[Q4_MK_SLOTS];

static bool exp_ok(int arena) { return arena >= 0 && arena < Q4_HIP_ARENAS; }

static bool grow(void **p, size_t *cap, size_t need) {
    if (*cap >= need && *p) return true;
    if (*p) (void)hipFree(*p);
    *p = nullptr;
    *cap = 0;
    if (hipMalloc(p, need) != hipSuccess) {
        *p = nullptr;
        (void)hipGetLastError();
        return false;
    }
    *cap = need;
    return true;
}

static bool moe_stage_bufs(void);
static bool xq8_bufs(void);
static uint32_t *g_bar; /* grid-barrier counter/flag for gdn_tail_k */
static int32_t *g_pos_d;   /* decode position read by graph-captured kernels */
__device__ int g_rope_neox = 1; /* rotary pair layout, see rope_pair */
static int32_t *g_kv_d;    /* KV slot, when it is not the rope position */
static uint32_t g_last_graph_nodes;

extern "C" bool q4_hip_init(void) {
    if (g_ok) return true;
    setenv("HIP_VISIBLE_DEVICES", "0", 0);
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) return false;
    if (hipSetDevice(0) != hipSuccess) return false;
    if (hipStreamCreateWithFlags(&g_str, hipStreamNonBlocking) != hipSuccess)
        g_str = nullptr; /* nullptr == legacy default: still correct, uncapturable */
    if (hipStreamCreate(&g_copy) != hipSuccess) g_copy = nullptr;
    g_ok = true;
    /* Pre-create step-profiler marker events here: creating them lazily would
     * hit hipEventCreate inside a graph capture and invalidate it. */
    {
        const char *e = getenv("Q4_STEP_PROF");
        if (e && e[0] == '1')
            for (int i = 0; i < Q4_MK_SLOTS; i++) {
                (void)hipEventCreate(&g_mk0[i]);
                (void)hipEventCreate(&g_mk1[i]);
            }
    }
    (void)moe_stage_bufs(); /* avoid first-time hipMalloc inside a capture */
    if (hipMalloc((void **)&g_bar, 2 * sizeof(uint32_t)) == hipSuccess)
        (void)hipMemset(g_bar, 0, 2 * sizeof(uint32_t));
    (void)hipMalloc((void **)&g_pos_d, sizeof(int32_t));
    (void)hipMalloc((void **)&g_kv_d, sizeof(int32_t));
    {
        const char *rn = getenv("Q4_ROPE_NEOX");
        const int neox = !(rn && rn[0] == '0');
        (void)hipMemcpyToSymbol(HIP_SYMBOL(g_rope_neox), &neox, sizeof(neox));
        if (!neox) fprintf(stderr, "q4: Q4_ROPE_NEOX=0 (old interleaved pairs)\n");
    }
    return true;
}

extern "C" void q4_hip_experts_free(void) {
    for (int a = 0; a < Q4_HIP_ARENAS; a++) {
        if (a == 0 && g_ava) vmm_arena_free();
        else if (g_exp[a]) (void)hipFree(g_exp[a]);
        g_exp[a] = nullptr;
        g_exp_slot[a] = 0;
        g_exp_n[a] = 0;
    }
}

static void cpx_dyn_free_all(void); /* defined beside cpx_wait_ctx */

extern "C" void q4_hip_shutdown(void) {
    q4_hip_experts_free();
    cpx_dyn_free_all();
    if (g_copy) {
        (void)hipStreamDestroy(g_copy);
        g_copy = nullptr;
    }
    if (g_dx) (void)hipFree(g_dx);
    if (g_dy) (void)hipFree(g_dy);
    if (g_dw) (void)hipFree(g_dw);
    if (g_bar) {
        (void)hipFree(g_bar);
        g_bar = nullptr;
    }
    if (g_pos_d) {
        (void)hipFree(g_pos_d);
        g_pos_d = nullptr;
    }
    if (g_kv_d) {
        (void)hipFree(g_kv_d);
        g_kv_d = nullptr;
    }
    if (g_scanqk) (void)hipFree(g_scanqk);
    if (!g_scan_lend) {
        if (g_scanh) (void)hipFree(g_scanh);
        if (g_scanoc) (void)hipFree(g_scanoc);
        if (g_scangb) (void)hipFree(g_scangb);
    }
    if (g_tk_hist) (void)hipFree(g_tk_hist);
    if (g_tk_st) (void)hipFree(g_tk_st);
    if (g_tk_cnt) (void)hipFree(g_tk_cnt);
    if (g_tk_off) (void)hipFree(g_tk_off);
    if (g_tk_chosen) (void)hipFree(g_tk_chosen);
    if (g_attn_part) (void)hipFree(g_attn_part);
    if (g_sk_part) (void)hipFree(g_sk_part);
    if (g_un_blk) (void)hipFree(g_un_blk);
    if (g_un_mask) (void)hipFree(g_un_mask);
    if (g_un_cnt) (void)hipFree(g_un_cnt);
    g_un_blk = nullptr;
    g_un_mask = g_un_cnt = nullptr;
    g_tk_hist = g_tk_st = g_tk_cnt = g_tk_off = nullptr;
    g_tk_chosen = nullptr;
    g_attn_part = nullptr;
    g_sk_part = nullptr;
    if (getenv("Q4_SHAPES"))
        for (int i = 0; i < g_shape_n; i++)
            fprintf(stderr, "SHAPE ty=%u r=%llu c=%llu nb=%llu n=%llu\n",
                    g_shapes[i].t, (unsigned long long)g_shapes[i].r,
                    (unsigned long long)g_shapes[i].c,
                    (unsigned long long)g_shapes[i].b,
                    (unsigned long long)g_shapes[i].n);
    if (getenv("Q4_SHAPES"))
        for (int i = 0; i < g_mix_n; i++) {
            fprintf(stderr, "MIX n=%llu in=%u nseg=%u:",
                    (unsigned long long)g_mixes[i].n, g_mixes[i].in,
                    g_mixes[i].nseg);
            for (uint32_t s = 0; s < g_mixes[i].nseg; s++)
                fprintf(stderr, " [ty=%u r=%u]", g_mixes[i].ty[s],
                        g_mixes[i].rows[s]);
            fprintf(stderr, "\n");
        }
    g_dx = g_dy = nullptr;
    g_dw = nullptr;
    g_scanqk = nullptr;
    g_scanh = g_scanoc = g_scangb = nullptr;
    g_dx_n = g_dy_n = g_dw_n = g_scanqk_n = 0;
    g_scanh_n = g_scanoc_n = g_scangb_n = 0;
    g_ok = false;
}

extern "C" bool q4_hip_experts_alloc_a(int arena, uint32_t n_slots,
                                       uint64_t slot_bytes) {
    if (!g_ok || !exp_ok(arena) || n_slots == 0 || slot_bytes == 0) return false;
    if (arena == 0 && g_ava) vmm_arena_free();
    else if (g_exp[arena]) (void)hipFree(g_exp[arena]);
    g_exp[arena] = nullptr;
    g_exp_n[arena] = 0;
    g_exp_slot[arena] = 0;
    size_t n = (size_t)n_slots * (size_t)slot_bytes;
    if (arena == 0 && g_avmm_want) {
        if (!vmm_arena_alloc(n)) return false;
        g_exp[0] = g_ava;
        g_exp_n[0] = n_slots;
        g_exp_slot[0] = slot_bytes;
        return true;
    }
    if (hipMalloc((void **)&g_exp[arena], n) != hipSuccess) {
        g_exp[arena] = nullptr;
        (void)hipGetLastError();
        return false;
    }
    g_exp_n[arena] = n_slots;
    g_exp_slot[arena] = slot_bytes;
    return true;
}

extern "C" bool q4_hip_experts_alloc(uint32_t n_slots, uint64_t slot_bytes) {
    return q4_hip_experts_alloc_a(0, n_slots, slot_bytes);
}

extern "C" uint32_t q4_hip_experts_n_a(int arena) {
    return exp_ok(arena) && g_exp[arena] ? g_exp_n[arena] : 0;
}


extern "C" uint32_t q4_hip_experts_n(void) {
    return q4_hip_experts_n_a(0);
}

/* Synchronous copy that still orders after g_str work. With a NonBlocking
 * compute stream, a blocking hipMemcpy on the legacy stream does NOT wait
 * for g_str kernels — it must be enqueued on g_str itself. During graph
 * capture it degrades to a memcpy node (host-visible results only after the
 * replay; capture regions must not read dst on the host right away). */
static bool sync_copy(void *dst, const void *src, size_t n,
                      hipMemcpyKind kind) {
    if (!g_str) return hipMemcpy(dst, src, n, kind) == hipSuccess;
    if (g_capturing)
        return hipMemcpyAsync(dst, src, n, kind, g_str) == hipSuccess;
    if (hipMemcpyAsync(dst, src, n, kind, g_str) != hipSuccess) return false;
    return hipStreamSynchronize(g_str) == hipSuccess;
}

extern "C" bool q4_hip_experts_put_a(int arena, uint32_t slot,
                                     const uint8_t *host, uint64_t n) {
    if (!exp_ok(arena) || !g_exp[arena] || slot >= g_exp_n[arena] || !host ||
        n == 0)
        return false;
    if (n > g_exp_slot[arena]) n = g_exp_slot[arena];
    return sync_copy(g_exp[arena] + (size_t)slot * (size_t)g_exp_slot[arena],
                     host, (size_t)n, hipMemcpyHostToDevice);
}

extern "C" bool q4_hip_experts_put(uint32_t slot, const uint8_t *host, uint64_t n) {
    return q4_hip_experts_put_a(0, slot, host, n);
}

/* Q4_RECLAIM_CHECK: the first upload into each slot of the watched range
 * (the last reclaim) is waited for and read back. */
static uint32_t g_watch_lo, g_watch_hi;
static uint8_t *g_watch_seen;
static pthread_mutex_t g_watch_mu = PTHREAD_MUTEX_INITIALIZER;

extern "C" void q4_hip_experts_watch(uint32_t lo, uint32_t hi) {
    pthread_mutex_lock(&g_watch_mu);
    free(g_watch_seen);
    g_watch_seen = hi > lo ? (uint8_t *)calloc(hi - lo, 1) : nullptr;
    g_watch_lo = lo;
    g_watch_hi = g_watch_seen ? hi : lo;
    pthread_mutex_unlock(&g_watch_mu);
}

static void watch_put(int arena, uint32_t slot, const uint8_t *host, uint64_t n,
                      hipStream_t st) {
    pthread_mutex_lock(&g_watch_mu);
    if (arena != 0 || slot < g_watch_lo || slot >= g_watch_hi ||
        g_watch_seen[slot - g_watch_lo]) {
        pthread_mutex_unlock(&g_watch_mu);
        return;
    }
    g_watch_seen[slot - g_watch_lo] = 1;
    uint8_t *d = g_exp[arena] + (size_t)slot * (size_t)g_exp_slot[arena];
    (void)hipStreamSynchronize(st);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (buf && hipMemcpy(buf, d, (size_t)n, hipMemcpyDeviceToHost) == hipSuccess) {
        uint64_t nd = 0, f = n, l = 0;
        for (uint64_t i = 0; i < n; i++)
            if (buf[i] != host[i]) {
                nd++;
                if (f == n) f = i;
                l = i;
            }
        const uint64_t a = (uint64_t)slot * g_exp_slot[arena];
        if (nd) {
            fprintf(stderr, "q4 reclaim watch: slot %u first upload BAD right after "
                    "its copy: %llu of %llu bytes differ at [%llu, %llu], slot at "
                    "chunk %llu+%llu\n", slot, (unsigned long long)nd,
                    (unsigned long long)n, (unsigned long long)f,
                    (unsigned long long)l, (unsigned long long)(a / Q4_VCH),
                    (unsigned long long)(a % Q4_VCH));
            fprintf(stderr, "  got ");
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", buf[f + i < n ? f + i : 0]);
            fprintf(stderr, "  want ");
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", host[f + i < n ? f + i : 0]);
            fputc('\n', stderr);
            /* a second copy of the same bytes, waited for */
            if (hipMemcpyAsync(d, host, (size_t)n, hipMemcpyHostToDevice, st) == hipSuccess &&
                hipStreamSynchronize(st) == hipSuccess &&
                hipMemcpy(buf, d, (size_t)n, hipMemcpyDeviceToHost) == hipSuccess)
                fprintf(stderr, "  second copy: %s\n",
                        memcmp(buf, host, (size_t)n) ? "STILL BAD" : "lands");
        } else {
            fprintf(stderr, "q4 reclaim watch: slot %u first upload ok\n", slot);
        }
    }
    free(buf);
    pthread_mutex_unlock(&g_watch_mu);
}

extern "C" bool q4_hip_experts_put_async_a(int arena, uint32_t slot,
                                           const uint8_t *host, uint64_t n) {
    if (!exp_ok(arena) || !g_exp[arena] || slot >= g_exp_n[arena] || !host ||
        n == 0)
        return false;
    if (n > g_exp_slot[arena]) n = g_exp_slot[arena];
    hipStream_t st = g_copy ? g_copy : 0;
    Q4_HIP_CHECK(hipMemcpyAsync(g_exp[arena] + (size_t)slot * (size_t)g_exp_slot[arena],
                                host, (size_t)n, hipMemcpyHostToDevice, st));
    if (g_watch_hi > g_watch_lo) watch_put(arena, slot, host, n, st);
    return true;
}

extern "C" bool q4_hip_experts_put_async(uint32_t slot, const uint8_t *host,
                                         uint64_t n) {
    return q4_hip_experts_put_async_a(0, slot, host, n);
}

extern "C" void q4_hip_copy_wait(void) {
    if (g_copy) (void)hipStreamSynchronize(g_copy);
    else (void)hipDeviceSynchronize();
}

/* GPU-side ordering for copy-stream uploads: fence records an event after
 * everything queued on g_copy so far; stream_wait makes g_str's next work
 * wait on it without blocking the host.  Used by the sb miss-scratch path
 * so the CPU pool isn't replaced by a per-layer host drain. */
static hipEvent_t g_copy_ev;
extern "C" bool q4_hip_copy_fence(void) {
    if (!g_copy) return true;
    if (g_capturing) return false;   /* eventRecord inside capture is illegal */
    if (!g_copy_ev &&
        hipEventCreateWithFlags(&g_copy_ev, hipEventDisableTiming) !=
            hipSuccess)
        return false;
    return hipEventRecord(g_copy_ev, g_copy) == hipSuccess;
}
extern "C" bool q4_hip_stream_wait_copy(void) {
    if (!g_copy_ev) return true;
    return hipStreamWaitEvent(g_str, g_copy_ev, 0) == hipSuccess;
}

extern "C" const uint8_t *q4_hip_experts_dev_a(int arena, uint32_t slot) {
    if (!exp_ok(arena) || !g_exp[arena] || slot >= g_exp_n[arena]) return nullptr;
    return g_exp[arena] + (size_t)slot * (size_t)g_exp_slot[arena];
}

extern "C" const uint8_t *q4_hip_experts_base_a(int arena) {
    return exp_ok(arena) ? g_exp[arena] : nullptr;
}
extern "C" uint64_t q4_hip_experts_stride_a(int arena) {
    return exp_ok(arena) ? g_exp_slot[arena] : 0;
}

extern "C" const uint8_t *q4_hip_experts_dev(uint32_t slot) {
    return q4_hip_experts_dev_a(0, slot);
}

extern "C" bool q4_hip_ok(void) { return g_ok; }

/* The compute stream all prefill/decode kernels run on (q4_mmq.cu queues
 * its MMQ/gather work on the same stream). nullptr == legacy default. */
extern "C" void *q4_hip_stream(void) { return (void *)g_str; }


/* Free VRAM right now (after the dense upload, before L1 sizing): the
 * planner budgets L1 slots off the real number, not the static 23 GiB
 * guess — the desktop's share is already subtracted by the driver. */
extern "C" uint64_t q4_hip_vram_free(void) {
    if (!g_ok) return 0;
    size_t fr = 0, tot = 0;
    if (hipMemGetInfo(&fr, &tot) != hipSuccess || !fr) return 0;
    return (uint64_t)fr;
}

extern "C" bool q4_hip_host_register(void *p, size_t n) {
    if (!g_ok || !p || !n) return false;
    if (hipHostRegister(p, n, hipHostRegisterMapped) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    return true;
}
/* Device-accessible alias of a host-registered buffer (zero-copy). */
extern "C" void *q4_hip_host_devptr(void *p) {
    if (!g_ok || !p) return nullptr;
    void *d = nullptr;
    if (hipHostGetDevicePointer(&d, p, 0) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    return d;
}
extern "C" void q4_hip_host_unregister(void *p) {
    if (g_ok && p) (void)hipHostUnregister(p);
}

extern "C" void q4_hip_sync(void) {
    if (g_ok) (void)hipDeviceSynchronize();
}

extern "C" void q4_hip_clear(void) {
    (void)hipGetLastError();
}

extern "C" void *q4_hip_malloc(size_t n) {
    if (!g_ok || n == 0) return nullptr;
    void *p = nullptr;
    if (hipMalloc(&p, n) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    return p;
}

extern "C" void q4_hip_free(void *p) {
    if (p && !vreg_free(p)) (void)hipFree(p);
}

extern "C" bool q4_hip_h2d(void *d, const void *h, size_t n) {
    if (!d || !h || n == 0) return false;
    return sync_copy(d, h, n, hipMemcpyHostToDevice);
}

extern "C" bool q4_hip_d2h(void *h, const void *d, size_t n) {
    if (!d || !h || n == 0) return false;
    return sync_copy(h, d, n, hipMemcpyDeviceToHost);
}

extern "C" int q4_hip_clock_khz(void) {
    static int khz;
    if (!khz) {
        int v = 0;
        if (hipDeviceGetAttribute(&v, hipDeviceAttributeClockRate, 0) !=
            hipSuccess)
            return 0;
        khz = v;
    }
    return khz;
}

/* ---- GEMV weight-read counters (profiling only) ----
 * Defined here so the multi-segment and routed-expert paths below can
 * charge their weight bytes before launch(). */
#define QG_PHASES 8
#define QG_PH_NONE 0
#define QG_PH_DRAFT 1
#define QG_PH_VERIFY 2
#define QG_PH_LOGITS 3
#define QG_PH_COMMIT 4
#define QG_PH_REPAIR 5
#define QG_PH_OTHER 6
static const char *const qg_ph_name[QG_PHASES] = {
    "none", "draft", "verify", "logits", "commit", "repair", "other", "?"
};
static uint64_t g_gemv_calls[QG_PHASES][64];
static uint64_t g_gemv_bytes[QG_PHASES][64];
static double g_gemv_ms[QG_PHASES][64];
/* Host-set phase index for the counters above (Q4_GEMV_PHASE). One thread
 * enqueues, so a plain int is enough. */
static int g_gemv_phase;
#define Q4_GEV_MAX 200000
static hipEvent_t g_gev_a[Q4_GEV_MAX], g_gev_b[Q4_GEV_MAX];
static uint8_t g_gev_ty[Q4_GEV_MAX];
static uint32_t g_gev_n;
static int g_gemv_mode = -1;
static int gemv_mode(void) {
    if (g_gemv_mode < 0) {
        const char *e = getenv("Q4_GEMV_PROF");
        g_gemv_mode = e ? atoi(e) : 0;
    }
    return g_gemv_mode;
}
/* Routed-expert GEMV byte accounting.  The wmoe_* kernels go through
 * moe_rows_gu / moe_rows_ya / moe_rows instead of launch(), so without this
 * the expert traffic (the largest single read stream in decode) never shows
 * up in the gemv-prof table.  mult = tensors per expert (2 for the fused
 * gate+up path). */
static uint32_t g_moe_nhits_hint;   /* last host-staged hit count (prof) */
static void moe_count(uint32_t ty, uint32_t out_dim, uint32_t in_dim,
                      uint32_t ne, uint32_t mult) {
    if (gemv_mode() <= 0 || ty >= 64 || !ne) return;
    /* The graphable exec path passes the hit count through device memory
     * (n_d): the host-staged value is what the replay reads, so use it. */
    uint32_t cnt = ne ? ne : (g_moe_nhits_hint ? g_moe_nhits_hint : 1u);
    uint64_t rb = (uint64_t)q4_row_bytes(ty, in_dim);
    g_gemv_calls[g_gemv_phase][ty] += cnt;
    g_gemv_bytes[g_gemv_phase][ty] += rb * out_dim * cnt * mult;
}
/* Phase tag packed with the type: type < 64 so 2 bits are free. */
static void gev_rec(int which, uint32_t type) {
    /* During graph capture eventRecord would be an illegal host call that
     * leaves a sticky error; profiling only covers eager launches then. */
    if (g_gev_n >= Q4_GEV_MAX || g_capturing) return;
    uint32_t i = g_gev_n;
    if (which == 0) {
        if (!g_gev_a[i])
            if (hipEventCreate(&g_gev_a[i]) != hipSuccess) return;
        if (hipEventRecord(g_gev_a[i], g_str) != hipSuccess) return;
        g_gev_ty[i] = (uint8_t)(type | ((uint32_t)g_gemv_phase << 6));
    } else {
        if (!g_gev_b[i])
            if (hipEventCreate(&g_gev_b[i]) != hipSuccess) return;
        if (hipEventRecord(g_gev_b[i], g_str) != hipSuccess) return;
        g_gev_n++;
    }
}
#define GEV_PRE(t) do { if (gemv_mode() >= 2) gev_rec(0, (t)); } while (0)
#define GEV_POST() do { if (gemv_mode() >= 2) gev_rec(1, 0); } while (0)

__global__ static void gemv_q8_0_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t blocks = (in_dim + 31u) / 32u;
    const uint8_t *wr = w + (size_t)row * (size_t)blocks * 34ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
            uint32_t i0 = b * 32u;
            uint32_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
            __half dh;
            memcpy(&dh, wr + b * 34u, 2);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2);
            float dot = 0.f;
            for (uint32_t i = 0; i < n; i++) dot += (float)qs[i] * xt[i0 + i];
            acc += __half2float(dh) * dot;
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__device__ static float bf16_f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

__global__ static void gemv_bf16_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const uint16_t *wr = (const uint16_t *)(w + (size_t)row * in_dim * 2u);
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            acc += bf16_f(wr[i]) * xt[i];
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

/* bf16 helpers: unpack a 4B pair into two f32 without extract/convert. */
__device__ static inline void bf16x2_f(uint32_t pr, float *lo, float *hi) {
    uint32_t ul = pr << 16, uh = pr & 0xffff0000u;
    memcpy(lo, &ul, 4);
    memcpy(hi, &uh, 4);
}

/* Warp-per-row bf16 dot: fastest mode needs in_dim%8==0 and 16B-aligned
 * w/x (uint4 + float4x2); otherwise the 4B-pair mode (in_dim%2 + 4B
 * aligned w); scalar tail if neither holds. */
__device__ static float wdot_bf16(const uint8_t *wr, const float *x,
                                  uint32_t in_dim, uint32_t lane) {
    float acc = 0.f;
    if (!(in_dim & 7u) && ((((uintptr_t)wr | (uintptr_t)x) & 15u) == 0)) {
        const uint4 *w4 = (const uint4 *)wr;
        const float4 *x4 = (const float4 *)x;
        const uint32_t n8 = in_dim >> 3;
        for (uint32_t i = lane; i < n8; i += 32u) {
            const uint4 p = w4[i];
            const float4 xa = x4[2u * i], xb = x4[2u * i + 1u];
            float a, b;
            bf16x2_f(p.x, &a, &b);
            acc = fmaf(a, xa.x, fmaf(b, xa.y, acc));
            bf16x2_f(p.y, &a, &b);
            acc = fmaf(a, xa.z, fmaf(b, xa.w, acc));
            bf16x2_f(p.z, &a, &b);
            acc = fmaf(a, xb.x, fmaf(b, xb.y, acc));
            bf16x2_f(p.w, &a, &b);
            acc = fmaf(a, xb.z, fmaf(b, xb.w, acc));
        }
    } else if (!(in_dim & 1u) && (((uintptr_t)wr & 3u) == 0)) {
        const uint32_t *wr32 = (const uint32_t *)wr;
        const uint32_t n2 = in_dim >> 1;
        for (uint32_t i = lane; i < n2; i += 32u) {
            uint32_t pr = wr32[i];
            float lo, hi;
            uint32_t ul = pr << 16, uh = pr & 0xffff0000u;
            memcpy(&lo, &ul, 4);
            memcpy(&hi, &uh, 4);
            acc = fmaf(lo, x[2u * i], fmaf(hi, x[2u * i + 1u], acc));
        }
    } else {
        const uint16_t *wr16 = (const uint16_t *)wr;
        for (uint32_t i = lane; i < in_dim; i += 32u)
            acc = fmaf(bf16_f(wr16[i]), x[i], acc);
    }
    return acc;
}

/* Warp-per-row bf16 GEMV for the decode path (n_batch==1): each warp owns a
 * row, lanes take 4B bf16 pairs and shuffle-reduce — no shared memory, no
 * block barriers.  The old block-per-row kernel spent most of its time in
 * the __syncthreads reduction with only ~20B/thread of real work. */
__global__ static void wgv_bf16_k(float *out, const uint8_t *w,
                                  const float *x, uint32_t in_dim,
                                  uint32_t out_dim, float scale) {
    const uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (wid >= out_dim) return;
    const uint8_t *wr = w + (size_t)wid * in_dim * 2u;
    float acc = wdot_bf16(wr, x, in_dim, lane);
    for (uint32_t o = 16; o; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffull, acc, o);
    if (lane == 0) out[wid] = acc * scale;
}

/* Vectorized warp/sub-warp bf16 GEMV: G lanes cooperate on one row, so a
 * warp covers 32/G rows.  uint4 weight loads (8 bf16) + float4 x loads cut
 * the instruction count ~4x vs the 4B scalar path and give each lane more
 * bytes in flight — the dominant win for in_dim<=~2k rows that spent most
 * of the kernel in the 5-deep shfl reduce.  Requires in_dim%8==0 and 16B
 * alignment on w and x. */
template <uint32_t G>
__global__ static void wgv_bf16v_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale, float act) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t rpw = 32u / G;
    const uint32_t row = wid * rpw + lane / G;
    const uint32_t gl = lane % G;
    if (row >= out_dim) return;
    const uint4 *w4 = (const uint4 *)(w + (size_t)row * in_dim * 2u);
    const float4 *x4 = (const float4 *)x;
    const uint32_t n8 = in_dim >> 3;
    float acc = 0.f;
    for (uint32_t i = gl; i < n8; i += G) {
        const uint4 p = w4[i];
        const float4 xa = x4[2u * i], xb = x4[2u * i + 1u];
        float a, b;
        bf16x2_f(p.x, &a, &b);
        acc = fmaf(a, xa.x, fmaf(b, xa.y, acc));
        bf16x2_f(p.y, &a, &b);
        acc = fmaf(a, xa.z, fmaf(b, xa.w, acc));
        bf16x2_f(p.z, &a, &b);
        acc = fmaf(a, xb.x, fmaf(b, xb.y, acc));
        bf16x2_f(p.w, &a, &b);
        acc = fmaf(a, xb.z, fmaf(b, xb.w, acc));
    }
    for (uint32_t o = G >> 1; o; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffull, acc, o);
    if (!gl) {
        float v = acc * scale;
        if (act != 0.f) {
            v *= act;
            v = v / (1.f + expf(-v));
        }
        out[row] = v;
    }
}

/* bf16 block-per-row for narrow outputs with big in_dim (hc_*_down, the
 * ssm_alpha/beta projections, indexer_k): one 256-thread block per row,
 * 16B weight loads, warp-reduce then 8-way shared reduce.  Warp-per-row on
 * out_dim=320 only filled 40 blocks — half the GPU idle and serial 20KB
 * rows. */
__global__ static void gemv_bf16_bk(float *out, const uint8_t *w,
                                    const float *x, uint32_t in_dim,
                                    uint32_t out_dim, float scale, float act) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const uint4 *w4 = (const uint4 *)(w + (size_t)row * in_dim * 2u);
    const float4 *x4 = (const float4 *)x;
    const uint32_t n8 = in_dim >> 3;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    float acc = 0.f;
    for (uint32_t i = threadIdx.x; i < n8; i += blockDim.x) {
        const uint4 p = w4[i];
        const float4 xa = x4[2u * i], xb = x4[2u * i + 1u];
        float a, b;
        bf16x2_f(p.x, &a, &b);
        acc = fmaf(a, xa.x, fmaf(b, xa.y, acc));
        bf16x2_f(p.y, &a, &b);
        acc = fmaf(a, xa.z, fmaf(b, xa.w, acc));
        bf16x2_f(p.z, &a, &b);
        acc = fmaf(a, xb.x, fmaf(b, xb.y, acc));
        bf16x2_f(p.w, &a, &b);
        acc = fmaf(a, xb.z, fmaf(b, xb.w, acc));
    }
    for (uint32_t o = 16; o; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffull, acc, o);
    __shared__ float sh[8];
    if (!lane) sh[warp] = acc;
    __syncthreads();
    if (!threadIdx.x) {
        float s = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < 8; i++) s += sh[i];
        float v = s * scale;
        if (act != 0.f) {
            v *= act;
            v = v / (1.f + expf(-v));
        }
        out[row] = v;
    }
}

/* Tiled many-token GEMM (prefill), same shape as gemm_q8_0_k: each block
 * owns a 32x32 (row x token) tile, stages dequantized W and f32 X tiles in
 * shared, one thread per (1 row, 4 toks). W is re-read only T/32 times.
 * WT is float (f32 weights) or uint16_t (bf16). */
__device__ static inline float wt_f(float v) { return v; }
__device__ static inline float wt_f(uint16_t v) { return bf16_f(v); }
template<typename WT>
__global__ static void gemm_tiled_k(float *out, const WT *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim,
                                   float scale, uint32_t n_batch,
                                   const int32_t *which) {
    __shared__ float ws[32][33];
    __shared__ float xs[32][33];
    const uint32_t r0 = blockIdx.x * 32u;
    const uint32_t t0 = blockIdx.y * 32u;
    const uint32_t nb = (in_dim + 31u) / 32u;
    const uint32_t wr = threadIdx.x / 8u;
    const uint32_t kk = (threadIdx.x % 8u) * 4u;
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t tr = warp * 4u + lane / 8u;
    const uint32_t tt = (lane % 8u) * 4u;
    float acc[4] = {};
    for (uint32_t kb = 0; kb < nb; kb++) {
        uint32_t i0 = kb * 32u;
        {
            uint32_t r = r0 + wr;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (r < out_dim) {
                const WT *xr = w + (size_t)r * in_dim + i0 + kk;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    v[j] = i0 + kk + j < in_dim ? wt_f(xr[j]) : 0.f;
            }
            ws[wr][kk + 0] = v[0];
            ws[wr][kk + 1] = v[1];
            ws[wr][kk + 2] = v[2];
            ws[wr][kk + 3] = v[3];
        }
        {
            uint32_t t = t0 + wr;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (t < n_batch) {
                const float *xr =
                    x + (size_t)(which ? which[t] : (int32_t)t) * in_dim + i0 +
                    kk;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    v[j] = i0 + kk + j < in_dim ? xr[j] : 0.f;
            }
            xs[wr][kk + 0] = v[0];
            xs[wr][kk + 1] = v[1];
            xs[wr][kk + 2] = v[2];
            xs[wr][kk + 3] = v[3];
        }
        __syncthreads();
#pragma unroll 8
        for (uint32_t k = 0; k < 32u; k++) {
            float wv = ws[tr][k];
#pragma unroll
            for (int j = 0; j < 4; j++) acc[j] += wv * xs[tt + j][k];
        }
        __syncthreads();
    }
    {
        uint32_t r = r0 + tr;
        if (r < out_dim) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t t = t0 + tt + (uint32_t)j;
                if (t < n_batch) out[(size_t)t * out_dim + r] = acc[j] * scale;
            }
        }
    }
}

__global__ static void gemv_f32_k(float *out, const float *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim, float scale,
                                  uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const float *wr = w + (size_t)row * in_dim;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            acc += wr[i] * xt[i];
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__device__ static inline float f16d(const uint8_t *p) {
    __half h;
    memcpy(&h, p, 2);
    return __half2float(h);
}

__device__ static inline float iq_u8(uint64_t v, int j) {
    return (float)((v >> (8 * j)) & 0xffu);
}

__device__ static float dot_iq2s(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *scales = blk + 74;
    const uint8_t *signs = qs + 32;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        const float db0 = d * (0.5f + (float)(scales[ib] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const float dl = l < 2 ? db0 : db1;
            const int idx = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
            const uint64_t gv = q4_iq2s_grid[idx];
            const uint8_t sg = signs[l];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float s = (sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f;
                acc += dl * iq_u8(gv, j) * s * x[xi++];
            }
        }
        qs += 4;
        signs += 4;
    }
    return acc;
}

/* iq2_xs 74B block: u16 qs[32] (9-bit grid + 7-bit sign-table idx),
 * scales[8] (per-16 nibble, ls = 2*n+1), effective d * 0.125 * ls. */
/* iq2_xxs 66B block: f16 d + qs[32] u16 -> per 32-group aux32[2]:
 * aux8[0..3] = iq2xxs grid bytes, aux32[1] = 4x 7-bit sign idx +
 * 4-bit scale nibble (dl = d * 0.125 * (2*sc+1)). */
__device__ static float dot_iq2xxs(const uint8_t *blk, const float *x) {
    const float d = f16d(blk) * 0.125f;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t aux32[2];
        memcpy(aux32, blk + 2 + ib * 8, 8);
        const uint8_t *aux8 = (const uint8_t *)aux32;
        const float db = d * (float)(2 * (int)(aux32[1] >> 28) + 1);
        for (int l = 0; l < 4; l++) {
            const uint64_t gv = q4_iq2xxs_grid[aux8[l]];
            const uint8_t sg = q4_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float s = (sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f;
                acc += db * iq_u8(gv, j) * s * x[xi++];
            }
        }
    }
    return acc;
}

__device__ static float dot_iq2xs(const uint8_t *blk, const float *x) {
    const float d = f16d(blk) * 0.125f;
    const uint8_t *scales = blk + 66;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        const float db0 = d * (float)(2 * (scales[ib] & 0xf) + 1);
        const float db1 = d * (float)(2 * (scales[ib] >> 4) + 1);
        for (int l = 0; l < 4; l++) {
            const float dl = l < 2 ? db0 : db1;
            uint16_t v;
            memcpy(&v, blk + 2 + ib * 8 + 2 * l, 2);
            const uint64_t gv = q4_iq2xs_grid[v & 511u];
            const uint8_t sg = q4_ksigns_iq2xs[v >> 9];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float s = (sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f;
                acc += dl * iq_u8(gv, j) * s * x[xi++];
            }
        }
    }
    return acc;
}

__device__ static float dot_iq3s(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *signs = blk + 74;
    const uint8_t *scales = blk + 106;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib += 2) {
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
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc += db * iq_u8(g1, j) *
                           ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f) * x[xi + j];
                    acc += db * iq_u8(g2, j) *
                           ((sg & q4_kmask_iq2xs[j + 4]) ? -1.f : 1.f) *
                           x[xi + 4 + j];
                }
                xi += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
    return acc;
}

__device__ static float dot_iq3xxs(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *ss = qs + 64;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t aux32;
        memcpy(&aux32, ss + 4 * ib, 4);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint32_t g1 = q4_iq3xxs_grid[qs[2 * l]];
            const uint32_t g2 = q4_iq3xxs_grid[qs[2 * l + 1]];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                acc += db * iq_u8(g1, j) *
                       ((sg & (1u << j)) ? -1.f : 1.f) * x[xi + j];
                acc += db * iq_u8(g2, j) *
                       ((sg & (1u << (j + 4))) ? -1.f : 1.f) * x[xi + 4 + j];
            }
            xi += 8;
        }
        qs += 8;
    }
    return acc;
}

__device__ static float dot_iq4xs(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const uint8_t *sl = blk + 4;
    const uint8_t *qs = blk + 8;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                       ((int)((sh >> (2 * ib)) & 3u) << 4);
        const float dl = d * (float)(ls - 32);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            acc += dl * (float)q4_kvalues_iq4nl[qs[j] & 0xf] * x[xi + j];
            acc += dl * (float)q4_kvalues_iq4nl[qs[j] >> 4] * x[xi + 16 + j];
        }
        qs += 16;
        xi += 32;
    }
    return acc;
}

__device__ static float dot_iq4nl(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    float acc = 0.f;
    for (int j = 0; j < 16; j++) {
        const uint8_t q = qs[j];
        acc += d * (float)q4_kvalues_iq4nl[q & 0xf] * x[j];
        acc += d * (float)q4_kvalues_iq4nl[q >> 4] * x[j + 16];
    }
    return acc;
}

__device__ static float dot_q40(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    float acc = 0.f;
    for (int j = 0; j < 16; j++) {
        acc += d * (float)((int)(qs[j] & 0xf) - 8) * x[j];
        acc += d * (float)((int)(qs[j] >> 4) - 8) * x[j + 16];
    }
    return acc;
}

__device__ static float dot_q50(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint8_t *qs = blk + 6;
    float acc = 0.f;
    for (int j = 0; j < 16; j++) {
        acc += d * (float)((int)((qs[j] & 0xf) | (((qh >> j) & 1u) << 4)) - 16) *
               x[j];
        acc += d * (float)((int)((qs[j] >> 4) |
                                 (((qh >> (j + 16)) & 1u) << 4)) - 16) *
               x[j + 16];
    }
    return acc;
}

__device__ static float dot_q20(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    float acc = 0.f;
    for (int j = 0; j < 64; j++)
        acc += d * (float)((int)((qs[j >> 2] >> ((j & 3) * 2)) & 3u) - 1) * x[j];
    return acc;
}

__device__ static float dot_q6k(const uint8_t *blk, const float *x) {
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    const float d = f16d(blk + 208);
    float acc = 0.f;
    for (int n128 = 0; n128 < 256; n128 += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = (int)(ql[l] & 0xF) | ((int)(qh[l] & 3) << 4);
            const int q2 = (int)(ql[l + 32] & 0xF) | (((int)(qh[l] >> 2) & 3) << 4);
            const int q3 = (int)(ql[l] >> 4) | (((int)(qh[l] >> 4) & 3) << 4);
            const int q4 = (int)(ql[l + 32] >> 4) | (((int)(qh[l] >> 6) & 3) << 4);
            acc += d * (float)sc[is] * (float)(q1 - 32) * x[n128 + l];
            acc += d * (float)sc[is + 2] * (float)(q2 - 32) * x[n128 + l + 32];
            acc += d * (float)sc[is + 4] * (float)(q3 - 32) * x[n128 + l + 64];
            acc += d * (float)sc[is + 6] * (float)(q4 - 32) * x[n128 + l + 96];
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc;
}


#define Q4_GEMV_BLK(name, dotfn, blk_bytes, blk_vals)                          \
    __global__ static void name(float *out, const uint8_t *w, const float *x, \
                                uint32_t in_dim, uint32_t out_dim, float scale,\
                                uint32_t n_batch, const int32_t *which) {      \
        uint32_t row = blockIdx.x;                                             \
        if (row >= out_dim) return;                                            \
        uint32_t nb = in_dim / (blk_vals);                                     \
        const uint8_t *wr = w + (size_t)row * nb * (blk_bytes);                \
        __shared__ float sh[64];                                               \
        for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {     \
            const float *xt =                                                  \
                x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;      \
            float acc = 0.f;                                                   \
            for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)            \
                acc += dotfn(wr + (size_t)b * (blk_bytes),                     \
                             xt + (size_t)b * (blk_vals));                     \
            sh[threadIdx.x] = acc;                                             \
            __syncthreads();                                                   \
            for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {               \
                if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];   \
                __syncthreads();                                               \
            }                                                                  \
            if (threadIdx.x == 0)                                              \
                out[(size_t)tok * out_dim + row] = sh[0] * scale;              \
            __syncthreads();                                                   \
        }                                                                      \
    }

Q4_GEMV_BLK(gemv_iq2xxs_k, dot_iq2xxs, 66u, 256u)
Q4_GEMV_BLK(gemv_iq2xs_k, dot_iq2xs, 74u, 256u)
Q4_GEMV_BLK(gemv_iq2_s_k, dot_iq2s, 82u, 256u)
Q4_GEMV_BLK(gemv_iq3_s_k, dot_iq3s, 110u, 256u)
Q4_GEMV_BLK(gemv_iq4_nl_k, dot_iq4nl, 18u, 32u)
Q4_GEMV_BLK(gemv_iq3xxs_k, dot_iq3xxs, 98u, 256u)
Q4_GEMV_BLK(gemv_iq4xs_k, dot_iq4xs, 136u, 256u)
Q4_GEMV_BLK(gemv_q4_0_k, dot_q40, 18u, 32u)
Q4_GEMV_BLK(gemv_q5_0_k, dot_q50, 22u, 32u)
Q4_GEMV_BLK(gemv_q2_0_k, dot_q20, 18u, 64u)
/* Head logits gemv: split every 256-value block into 16 units of 4 quads;
 * each thread reads one unit's ql/qh as uint32 vectors. Deterministic: the
 * per-row result is the fixed blockDim reduction of thread partials. */
__global__ static void gemv_q6_k_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim,
                                   float scale, uint32_t n_batch,
                                   const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 210u;
    __shared__ float sh[128];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt =
            x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        uint32_t nu = nb * 16u;
        for (uint32_t u = threadIdx.x; u < nu; u += blockDim.x) {
            const uint8_t *blk = wr + (size_t)(u >> 4) * 210u;
            uint32_t qu = u & 15u;
            uint32_t h2 = qu >> 3, l4 = (qu & 7u) * 4u;
            uint32_t qlv0 = *(const uint32_t *)(blk + h2 * 64u + l4);
            uint32_t qlv1 = *(const uint32_t *)(blk + h2 * 64u + l4 + 32u);
            uint32_t qhv = *(const uint32_t *)(blk + 128u + h2 * 32u + l4);
            const int8_t *sc = (const int8_t *)(blk + 192u) + h2 * 8;
            float d = f16d(blk + 208u);
            uint32_t is = l4 >> 4;
            const float *xt4 = xt + (u >> 4) * 256u + h2 * 128u + l4;
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t qlo = (qlv0 >> (j * 8)) & 0xffu;
                uint32_t qlh = (qlv1 >> (j * 8)) & 0xffu;
                uint32_t qhb = (qhv >> (j * 8)) & 0xffu;
                acc += d * (float)sc[is] *
                       (float)(int((qlo & 15u) | ((qhb & 3u) << 4)) - 32) *
                       xt4[j];
                acc += d * (float)sc[is + 2u] *
                       (float)(int((qlh & 15u) | (((qhb >> 2) & 3u) << 4)) -
                               32) *
                       xt4[j + 32u];
                acc += d * (float)sc[is + 4u] *
                       (float)(int((qlo >> 4) | (((qhb >> 4) & 3u) << 4)) - 32) *
                       xt4[j + 64u];
                acc += d * (float)sc[is + 6u] *
                       (float)(int((qlh >> 4) | (((qhb >> 6) & 3u) << 4)) - 32) *
                       xt4[j + 96u];
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

/* Batched routed-expert rows: grid.x = output rows, grid.y = expert index.
 * Each expert's weight pack lives at base + slots[e] * stride; x is either
 * shared (xs = 0) or per-expert (xs = in_dim). One launch per stage instead
 * of one launch per expert — removes ~3*n small kernels per MoE layer. */
#define Q4_MOE_BLK(name, dotfn, bb, bv)                                        \
    __global__ static void name(float *out, const uint8_t *base,               \
                                uint64_t stride, const int32_t *slots,         \
                                const float *x, uint32_t xs,                   \
                                uint32_t in_dim, uint32_t out_dim) {           \
        uint32_t e = blockIdx.y, row = blockIdx.x;                             \
        uint32_t nb = in_dim / (bv);                                           \
        const uint8_t *wr = base + (size_t)slots[e] * (stride) +               \
                            (size_t)row * nb * (bb);                           \
        const float *xt = x + (size_t)e * xs;                                  \
        __shared__ float sh[64];                                               \
        float acc = 0.f;                                                       \
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)                \
            acc += dotfn(wr + (size_t)b * (bb), xt + (size_t)b * (bv));        \
        sh[threadIdx.x] = acc;                                                 \
        __syncthreads();                                                       \
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {                   \
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];       \
            __syncthreads();                                                   \
        }                                                                      \
        if (threadIdx.x == 0)                                                  \
            out[(size_t)e * out_dim + row] = sh[0];                            \
    }
Q4_MOE_BLK(moe_iq2xxs_bk, dot_iq2xxs, 66u, 256u)
Q4_MOE_BLK(moe_iq2xs_bk, dot_iq2xs, 74u, 256u)
Q4_MOE_BLK(moe_iq2_bk, dot_iq2s, 82u, 256u)
Q4_MOE_BLK(moe_iq3_bk, dot_iq3s, 110u, 256u)
Q4_MOE_BLK(moe_iq4_bk, dot_iq4nl, 18u, 32u)
Q4_MOE_BLK(moe_iq3xxs_bk, dot_iq3xxs, 98u, 256u)
Q4_MOE_BLK(moe_iq4xs_bk, dot_iq4xs, 136u, 256u)
Q4_MOE_BLK(moe_q40_bk, dot_q40, 18u, 32u)
Q4_MOE_BLK(moe_q50_bk, dot_q50, 22u, 32u)
Q4_MOE_BLK(moe_q20_bk, dot_q20, 18u, 64u)

/* ---- warp-per-row decode GEMVs (Q4_WARPGV, default on) -------------------
 * Decode GEMVs are latency-bound: one 64-thread block per row underfills the
 * GPU and reads quant bytes serially.  These kernels give each warp one
 * output row and map the 32 lanes onto 8-element "octs" inside each quant
 * block, so every warp-load is fully coalesced (q8_0: lane -> 16 elems of
 * block l/2; iq2_s/iq3_s: lane -> oct lane of the single block per iter;
 * iq4_nl: lane -> quarter (l&3) of block l/4).  Lane-local fp32 accumulate,
 * then a fixed-order __shfl_down_sync tree — deterministic and graph-safe.
 * The fp sum order differs from the 64-thread kernels; diffs are bounded
 * reorder noise (~1e-5 relative), same as the gemv_f32_sk path. */
static int wgv_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_WARPGV");
        v = !(e && e[0] == '0');
    }
    return v;
}

static int wgvn_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_WGVN");
        v = !(e && e[0] == '0');
    }
    return v;
}

__device__ static inline float wgv_shfl_red(float acc) {
    for (uint32_t o = 16; o; o >>= 1)
        acc += __shfl_down_sync(0xffffffffull, acc, o);
    return acc;
}

/* q8_0: lane covers 16 elems (half block) via 8x u16 + 4x float2; warp covers
 * 16 blocks (512 vals) per iter.  Requires in_dim % 32 == 0. */
/* 8-float group load for the wgd_* oct decoders: a float4 pair when the
 * pointer is 16B-aligned (engine activations always are — callers offset
 * by multiples of 8 elems), scalar otherwise. */
__device__ static inline void xload8(const float *x, float v[8]) {
    if (!((uintptr_t)x & 15u)) {
        const float4 a = *(const float4 *)x, b = *(const float4 *)(x + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = x[j];
    }
}

__device__ static inline void xload4(const float *x, float v[4]) {
    if (!((uintptr_t)x & 15u)) {
        const float4 a = *(const float4 *)x;
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    } else {
#pragma unroll
        for (int j = 0; j < 4; j++) v[j] = x[j];
    }
}

__device__ static float wgd_q8h(const uint8_t *blk, const float *x,
                                uint32_t m) {
    const float d = f16d(blk);
    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        uint16_t q2;
        memcpy(&q2, blk + 2 + m + 2 * j, 2);
        float2 xv = *(const float2 *)(x + m + 2 * j);
        dot += (float)(int8_t)(q2 & 0xff) * xv.x +
               (float)(int8_t)(q2 >> 8) * xv.y;
    }
    return d * dot;
}

/* iq2_s: lane covers oct 8*lane..+7 of a 256-block (ib = lane/4, l = lane&3,
 * j = 0..7).  qs/sign byte index == lane → coalesced 32B warp loads. */
__device__ static float wgd_iq2s(const uint8_t *blk, const float *x,
                                 uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    const float sc = (l < 2) ? (float)(blk[74 + ib] & 0xf)
                             : (float)(blk[74 + ib] >> 4);
    const float dl = d * (0.5f + sc) * 0.25f;
    const int idx = blk[2 + ib * 4 + l] |
                    ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    const uint64_t gv = q4_iq2s_grid[idx];
    const uint8_t sg = blk[2 + 32 + ib * 4 + l];
    float xv[8];
    xload8(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float s = (sg & (1u << j)) ? -1.f : 1.f;
        acc += iq_u8(gv, j) * s * xv[j];
    }
    return dl * acc;
}

/* iq2_xs: lane covers oct 8*lane..+7 of a 256-block (ib = lane/4,
 * l = lane&3).  u16 v = 9-bit grid idx + 7-bit sign idx; scale nibble by
 * l<2. */
/* iq2_xxs: lane covers oct 8*lane..+7 (ib = lane>>2, l = lane&3). */
__device__ static float wgd_iq2xxs(const uint8_t *blk, const float *x,
                                   uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    uint32_t aux32[2];
    memcpy(aux32, blk + 2 + ib * 8, 8);
    const float sc = (float)(aux32[1] >> 28);
    const float dl = d * 0.125f * (2.f * sc + 1.f);
    const uint64_t gv = q4_iq2xxs_grid[(aux32[0] >> (8 * l)) & 0xffu];
    const uint8_t sg = q4_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
    float xv[8];
    xload8(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float s = (sg & (1u << j)) ? -1.f : 1.f;
        acc += iq_u8(gv, j) * s * xv[j];
    }
    return dl * acc;
}

__device__ static float wgd_iq2xs(const uint8_t *blk, const float *x,
                                  uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    const uint8_t sb = blk[66 + ib];
    const float sc = (l < 2) ? (float)(sb & 0xf) : (float)(sb >> 4);
    const float dl = d * 0.125f * (2.f * sc + 1.f);
    uint16_t v;
    memcpy(&v, blk + 2 + ib * 8 + 2 * l, 2);
    const uint64_t gv = q4_iq2xs_grid[v & 511u];
    const uint8_t sg = q4_ksigns_iq2xs[v >> 9];
    float xv[8];
    xload8(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float s = (sg & (1u << j)) ? -1.f : 1.f;
        acc += iq_u8(gv, j) * s * xv[j];
    }
    return dl * acc;
}

/* iq3_s: lane covers oct 8*lane..+7 of a 256-block (p = lane/8,
 * half = (lane>>2)&1, l = lane&3, jj = 0..7). */
__device__ static float wgd_iq3s(const uint8_t *blk, const float *x,
                                 uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t p = lane >> 3, half = (lane >> 2) & 1u, l = lane & 3u;
    const float sc = (half == 0) ? (float)(blk[106 + p] & 0xf)
                                 : (float)(blk[106 + p] >> 4);
    const float db = d * (1.f + 2.f * sc);
    const uint8_t qhb = blk[66 + p * 2 + half];
    const int i1 = blk[2 + p * 16 + half * 8 + 2 * l] |
                   ((qhb << (8 - 2 * l)) & 256);
    const int i2 = blk[2 + p * 16 + half * 8 + 2 * l + 1] |
                   ((qhb << (7 - 2 * l)) & 256);
    const uint32_t g1 = q4_iq3s_grid[i1];
    const uint32_t g2 = q4_iq3s_grid[i2];
    const uint8_t sg = blk[74 + p * 8 + half * 4 + l];
    float xv[8];
    xload8(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int jj = 0; jj < 8; jj++) {
        const uint32_t gi = jj & 3u;
        const uint32_t gv = (jj < 4) ? g1 : g2;
        const float s = (sg & (1u << jj)) ? -1.f : 1.f;
        acc += (float)((gv >> (8 * gi)) & 0xffu) * s * xv[jj];
    }
    return db * acc;
}

/* iq3_xxs: lane covers oct 8*lane..+7 of a 256-block (ib = lane>>2,
 * l = lane&3).  qs[96] = 64 grid bytes + 32 sign/scale bytes (aux32). */
__device__ static float wgd_iq3xxs(const uint8_t *blk, const float *x,
                                   uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
    const uint8_t *q3 = blk + 2 + ib * 8;
    const uint32_t g1 = q4_iq3xxs_grid[q3[2 * l]];
    const uint32_t g2 = q4_iq3xxs_grid[q3[2 * l + 1]];
    float xv[8];
    xload8(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint32_t gi = j & 3u;
        const uint32_t gv = (j < 4) ? g1 : g2;
        const float s = (sg & (1u << j)) ? -1.f : 1.f;
        acc += (float)((gv >> (8 * gi)) & 0xffu) * s * xv[j];
    }
    return db * acc;
}

/* iq4_xs: lane covers oct 8*lane..+7 of a 256-block (ib = lane>>2,
 * q = lane&3).  Block: f16 d, u16 scales_h, scales_l[4], qs[128]. */
__device__ static float wgd_iq4xs(const uint8_t *blk, const float *x,
                                  uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, q = lane & 3u;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)(ls - 32);
    const uint8_t *qb = blk + 8 + ib * 16 + (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    float xv[8];
    xload8(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint8_t nib = hi ? (qb[j] >> 4) : (qb[j] & 0xf);
        acc += (float)q4_kvalues_iq4nl[nib] * xv[j];
    }
    return dl * acc;
}

/* iq4_nl: lane covers quarter (lane&3)*8..+7 of a 32-block via 4x u16 qs
 * loads (warp covers 8 blocks/iter). */
/* iq4_nl 16-entry table as a register LUT: two v_perm_b32 byte selects
 * blended on nibble bit3.  Beats 8 serialized __constant__ byte gathers.
 * (Defined below; the float decoder needs it, so it is forward-declared.) */
__device__ static inline uint32_t kv_iq4nl_lut4(uint32_t idx4);
__device__ static inline uint32_t kv_q40_lut4(uint32_t idx4);

/* iq4_nl: lane covers the 8 values of one quarter-quarter of a 32-elem
 * block.  One 8-byte load fetches all 16 nibbles this lane decodes; the
 * 16-entry table lookup then runs as two v_perm_b32 LUTs (kv_iq4nl_lut4)
 * instead of 8 scalar gathers from __constant__. */
__device__ static float wgd_iq4nl(const uint8_t *blk, const float *x,
                                  uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t q = lane & 3u;
    const uint32_t byte0 = (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    float xv[8];
    xload8(x, xv);
    uint64_t qb8;
    memcpy(&qb8, blk + 2 + byte0, 8);
    uint32_t w0 = (uint32_t)(qb8 & 0xffffffffu), w1 = (uint32_t)(qb8 >> 32);
    if (hi) {
        w0 >>= 4;
        w1 >>= 4;
    }
    const uint32_t wa = kv_iq4nl_lut4(w0 & 0x0f0f0f0fu);
    const uint32_t wb = kv_iq4nl_lut4(w1 & 0x0f0f0f0fu);
    const int8_t val[8] = {
        (int8_t)wa,       (int8_t)(wa >> 8),
        (int8_t)(wa >> 16), (int8_t)(wa >> 24),
        (int8_t)wb,       (int8_t)(wb >> 8),
        (int8_t)(wb >> 16), (int8_t)(wb >> 24)
    };
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++)
        acc += (float)val[2 * j] * xv[2 * j] + (float)val[2 * j + 1] * xv[2 * j + 1];
    return d * acc;
}

/* q4_0: same lane geometry as iq4nl (quarter (lane&3)*8..+7), linear
 * nibble map instead of the table. */
__device__ static float wgd_q40(const uint8_t *blk, const float *x,
                                uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t q = lane & 3u;
    const uint32_t byte0 = (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    float xv[8];
    xload8(x, xv);
    /* one 8-byte load covers the 16 nibbles this lane decodes */
    uint64_t qb8;
    memcpy(&qb8, blk + 2 + byte0, 8);
    uint32_t w0 = (uint32_t)(qb8 & 0xffffffffu), w1 = (uint32_t)(qb8 >> 32);
    if (hi) {
        w0 >>= 4;
        w1 >>= 4;
    }
    const uint32_t wa = kv_q40_lut4(w0 & 0x0f0f0f0fu);
    const uint32_t wb = kv_q40_lut4(w1 & 0x0f0f0f0fu);
    const int8_t val[8] = {
        (int8_t)wa,       (int8_t)(wa >> 8),
        (int8_t)(wa >> 16), (int8_t)(wa >> 24),
        (int8_t)wb,       (int8_t)(wb >> 8),
        (int8_t)(wb >> 16), (int8_t)(wb >> 24)
    };
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++)
        acc += (float)val[2 * j] * xv[2 * j] + (float)val[2 * j + 1] * xv[2 * j + 1];
    return d * acc;
}

/* q5_0: q4_0 layout + 32 high bits at blk+2 (bit i = elem i), offset -16. */
__device__ static float wgd_q50(const uint8_t *blk, const float *x,
                                uint32_t lane) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint32_t q = lane & 3u;
    const uint32_t byte0 = (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    const uint32_t e0 = q * 8u;
    float xv[8];
    xload8(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint16_t qb;
        memcpy(&qb, blk + 6 + byte0 + 2 * j, 2);
        const uint32_t ea = e0 + 2 * j, eb = ea + 1;
        const int v0 = (int)((hi ? ((qb >> 4) & 0xf) : (qb & 0xf)) |
                             (((qh >> ea) & 1u) << 4)) - 16;
        const int v1 = (int)((hi ? (qb >> 12) : ((qb >> 8) & 0xf)) |
                             (((qh >> eb) & 1u) << 4)) - 16;
        acc += (float)v0 * xv[2 * j] + (float)v1 * xv[2 * j + 1];
    }
    return d * acc;
}

/* q2_0: 64-elem block = fp16 d + 16B of 2-bit quads; x is pre-offset to the
 * lane's 4-elem group, byte picks the packed byte.  Values (q - 1) * d. */
__device__ static float wgd_q20(const uint8_t *blk, const float *x,
                                uint32_t byte) {
    const float d = f16d(blk);
    const uint32_t qv = blk[2 + byte];
    float xv[4];
    xload4(x, xv);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++)
        acc += (float)((int)((qv >> (j * 2)) & 3u) - 1) * xv[j];
    return d * acc;
}

/* ---- int8-activation helpers (gfx11 v_dot4_i32_iu8) --------------------
 * The wmoe_* float path spends most of its issue slots on per-element
 * load+cvt+fma after the weight decode.  With activations pre-quantized
 * to per-32 int8 groups (xq8_k below), the same decoders emit packed
 * i8x4 and reduce 4 products per instruction.  Q4_WGQ8=0 falls back. */

/* iq4_nl 16-entry table as a register LUT: two v_perm_b32 byte selects
 * blended on nibble bit3.  Beats 8 serialized __constant__ byte gathers. */
__device__ static inline uint32_t kv_iq4nl_lut4(uint32_t idx4) {
    const uint32_t sel = idx4 & 0x07070707u;
    const uint32_t mhi = ((idx4 >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t lo = (uint32_t)__builtin_amdgcn_perm(0xF6EADDCFu,
                                                       0xBFAD9881u, sel);
    const uint32_t hi = (uint32_t)__builtin_amdgcn_perm(0x71594535u,
                                                       0x26190D01u, sel);
    return (lo & ~mhi) | (hi & mhi);
}

/* q4_0's {-8..7} nibble map in the same register-LUT form: byte j of the
 * result is table[idx & 0x0f]. */
__device__ static inline uint32_t kv_q40_lut4(uint32_t idx4) {
    const uint32_t sel = idx4 & 0x07070707u;
    const uint32_t mhi = ((idx4 >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t lo = (uint32_t)__builtin_amdgcn_perm(0xFFFEFDFCu,
                                                       0xFBF9FAF8u, sel);
    const uint32_t hi = (uint32_t)__builtin_amdgcn_perm(0x07060504u,
                                                       0x03020100u, sel);
    return (lo & ~mhi) | (hi & mhi);
}

/* iq4_nl int8: same lane geometry as wgd_iq4nl — quarter (lane&3)*8 of a
 * 32-element block — but returns an i32 partial dot and the block scale
 * via *dp so the caller multiplies by dw*dx once per 8 elements. */
__device__ static int32_t wgd_iq4nl_i8(const uint8_t *blk, const int8_t *xq,
                                       uint32_t lane, float *dp) {
    *dp = f16d(blk);
    uint64_t qv;
    memcpy(&qv, blk + 2 + (lane & 1u) * 8u, 8);
    if (lane & 2u) qv >>= 4;
    const uint32_t w0 = kv_iq4nl_lut4((uint32_t)qv);
    const uint32_t w1 = kv_iq4nl_lut4((uint32_t)(qv >> 32));
    uint32_t xl, xh;
    memcpy(&xl, xq + (lane & 3u) * 8u, 4);
    memcpy(&xh, xq + (lane & 3u) * 8u + 4u, 4);
    int32_t s = 0;
    s = __builtin_amdgcn_sudot4(true, (int)w0, true, (int)xl, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)w1, true, (int)xh, s, false);
    return s;
}

/* q2_0 int8: lane covers elements byte*4..+3 of a 64-element block.  The
 * 2-bit field map {0,1,2,3} -> {-1,0,1,2} is a 4-entry byte LUT. */
__device__ static int32_t wgd_q20_i8(const uint8_t *blk, uint32_t xu,
                                     uint32_t byte, float *dp) {
    *dp = f16d(blk);
    const uint32_t f = blk[2 + byte];
    const uint32_t sel = (f & 3u) | (((f >> 2) & 3u) << 8) |
                         (((f >> 4) & 3u) << 16) | (((f >> 6) & 3u) << 24);
    const uint32_t w = (uint32_t)__builtin_amdgcn_perm(0, 0x020100FFu, sel);
    int32_t s = 0;
    s = __builtin_amdgcn_sudot4(true, (int)w, true, (int)xu, s, false);
    return s;
}

/* Sign-byte -> per-byte 0xFF masks.  M = bits{7,14,21,28} spreads low 4
 * bits to byte MSBs; >>7 then *0xFF widens them.  No packed-negate exists
 * on gfx11, so signed-magnitude octs split into two masked u8 dots. */
__device__ static inline void sg_masks(uint32_t sg, uint32_t *m0,
                                       uint32_t *m1) {
    *m0 = ((((sg & 0x0Fu) * 0x10204080u) & 0x80808080u) >> 7) * 0xFFu;
    *m1 = (((((sg >> 4) & 0x0Fu) * 0x10204080u) & 0x80808080u) >> 7) * 0xFFu;
}

/* Oct dot for sign-magnitude grids: gv holds 8 unsigned magnitudes, sg
 * marks the negative bytes.  s = dot(w&~m, x) - dot(w&m, x). */
__device__ static inline int32_t dot8_sg(uint64_t gv, uint32_t sg,
                                         const int8_t *xq) {
    uint32_t m0, m1, xl, xh;
    sg_masks(sg, &m0, &m1);
    memcpy(&xl, xq, 4);
    memcpy(&xh, xq + 4, 4);
    const uint32_t w0 = (uint32_t)gv, w1 = (uint32_t)(gv >> 32);
    int32_t p = 0, n = 0;
    p = __builtin_amdgcn_sudot4(false, (int)(w0 & ~m0), true, (int)xl, p,
                                false);
    p = __builtin_amdgcn_sudot4(false, (int)(w1 & ~m1), true, (int)xh, p,
                                false);
    n = __builtin_amdgcn_sudot4(false, (int)(w0 & m0), true, (int)xl, n,
                                false);
    n = __builtin_amdgcn_sudot4(false, (int)(w1 & m1), true, (int)xh, n,
                                false);
    return p - n;
}

/* iq2_s int8: same lane geometry as wgd_iq2s (oct 8*lane..+7). */
__device__ static int32_t wgd_iq2s_i8(const uint8_t *blk, const int8_t *xq,
                                      uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    const float sc = (l < 2) ? (float)(blk[74 + ib] & 0xf)
                             : (float)(blk[74 + ib] >> 4);
    *dp = d * (0.5f + sc) * 0.25f;
    const int idx = blk[2 + ib * 4 + l] |
                    ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    const uint64_t gv = q4_iq2s_grid[idx];
    const uint8_t sg = blk[2 + 32 + ib * 4 + l];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq2_xxs int8: same lane geometry as wgd_iq2xxs. */
__device__ static int32_t wgd_iq2xxs_i8(const uint8_t *blk, const int8_t *xq,
                                        uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    uint32_t aux32[2];
    memcpy(aux32, blk + 2 + ib * 8, 8);
    *dp = d * 0.125f * (2.f * (float)(aux32[1] >> 28) + 1.f);
    const uint64_t gv = q4_iq2xxs_grid[(aux32[0] >> (8 * l)) & 0xffu];
    const uint8_t sg = q4_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq2_xs int8: same lane geometry as wgd_iq2xs. */
__device__ static int32_t wgd_iq2xs_i8(const uint8_t *blk, const int8_t *xq,
                                       uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    const uint8_t sb = blk[66 + ib];
    const float sc = (l < 2) ? (float)(sb & 0xf) : (float)(sb >> 4);
    *dp = d * 0.125f * (2.f * sc + 1.f);
    uint16_t v;
    memcpy(&v, blk + 2 + ib * 8 + 2 * l, 2);
    const uint64_t gv = q4_iq2xs_grid[v & 511u];
    const uint8_t sg = q4_ksigns_iq2xs[v >> 9];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq3_s int8: same lane geometry as wgd_iq3s. */
__device__ static int32_t wgd_iq3s_i8(const uint8_t *blk, const int8_t *xq,
                                      uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t p = lane >> 3, half = (lane >> 2) & 1u, l = lane & 3u;
    const float sc = (half == 0) ? (float)(blk[106 + p] & 0xf)
                                 : (float)(blk[106 + p] >> 4);
    *dp = d * (1.f + 2.f * sc);
    const uint8_t qhb = blk[66 + p * 2 + half];
    const int i1 = blk[2 + p * 16 + half * 8 + 2 * l] |
                   ((qhb << (8 - 2 * l)) & 256);
    const int i2 = blk[2 + p * 16 + half * 8 + 2 * l + 1] |
                   ((qhb << (7 - 2 * l)) & 256);
    uint64_t gv = (uint64_t)q4_iq3s_grid[i2] << 32 | q4_iq3s_grid[i1];
    const uint8_t sg = blk[74 + p * 8 + half * 4 + l];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq3_xxs int8: same lane geometry as wgd_iq3xxs. */
__device__ static int32_t wgd_iq3xxs_i8(const uint8_t *blk, const int8_t *xq,
                                        uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    *dp = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
    const uint8_t *q3 = blk + 2 + ib * 8;
    uint64_t gv = (uint64_t)q4_iq3xxs_grid[q3[2 * l + 1]] << 32 |
                  q4_iq3xxs_grid[q3[2 * l]];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq4_xs int8: same lane geometry as wgd_iq4xs — nibble LUT like iq4_nl,
 * no sign byte. */
__device__ static int32_t wgd_iq4xs_i8(const uint8_t *blk, const int8_t *xq,
                                       uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, q = lane & 3u;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    *dp = d * (float)(ls - 32);
    uint64_t qv;
    memcpy(&qv, blk + 8 + ib * 16 + (q & 1u) * 8u, 8);
    if (q >> 1) qv >>= 4;
    const uint32_t w0 = kv_iq4nl_lut4((uint32_t)qv);
    const uint32_t w1 = kv_iq4nl_lut4((uint32_t)(qv >> 32));
    uint32_t xl, xh;
    memcpy(&xl, xq + lane * 8u, 4);
    memcpy(&xh, xq + lane * 8u + 4u, 4);
    int32_t s = 0;
    s = __builtin_amdgcn_sudot4(true, (int)w0, true, (int)xl, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)w1, true, (int)xh, s, false);
    return s;
}

/* f32: lane covers 8 consecutive floats (2x float4); warp = 256/iter. */
__device__ static float wgd_f32o(const float *wr, const float *x) {
    const float4 a = *(const float4 *)(wr);
    const float4 b = *(const float4 *)(wr + 4);
    const float4 xa = *(const float4 *)(x);
    const float4 xb = *(const float4 *)(x + 4);
    return a.x * xa.x + a.y * xa.y + a.z * xa.z + a.w * xa.w +
           b.x * xb.x + b.y * xb.y + b.z * xb.z + b.w * xb.w;
}

#define Q4_WGV_ROW()                                                           \
    uint32_t row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);        \
    uint32_t lane = threadIdx.x & 31u;                                         \
    if (row >= out_dim) return

/* ---- row split ------------------------------------------------------------
 * The warp-per-row kernels lose 2-2.5x on short grids: 320 blocks over 96
 * CUs is 3.3 per CU, so the last "wave" only fills a third of the machine
 * and the drain dominates (the gemvbench occ sweep: iq4_nl 640 cols goes
 * 138 GB/s at 2560 rows to 468 at 131072 — same per-warp work, only the
 * block count differs).  Splitting one row over nsp warps multiplies the
 * block count by nsp without touching the work, which shrinks the
 * quantisation; index 0 of each group sums the partials in split order so
 * the result stays deterministic.  nsp==1 is the original mapping, ranges
 * and add order, bit for bit.
 *
 * Requires out_dim % (8/nsp) == 0 so the early return stays uniform per
 * warp and __syncthreads is never reached by a partial block.  wgv_nsp()
 * enforces that and only asks for a split when the grid needs it. */
#define Q4_WGV_SP_MAX 8u

#define Q4_WGV_ROW_SP(nsp)                                                     \
    const uint32_t q4_rpb = (blockDim.x >> 5) / (nsp);                         \
    uint32_t row = blockIdx.x * q4_rpb + ((threadIdx.x >> 5) / (nsp));         \
    uint32_t q4_sp = (threadIdx.x >> 5) % (nsp);                               \
    uint32_t lane = threadIdx.x & 31u;                                         \
    if (row >= out_dim) return

/* block-range boundary of split k: 0 and nb are exact, the interior ones
 * round up so every block lands in exactly one split's range. */
__device__ static inline uint32_t q4_sp_bound(uint32_t k, uint32_t nsp,
                                              uint32_t nb, uint32_t step) {
    if (k == 0u) return 0u;
    if (k >= nsp) return nb;
    uint32_t e = (k * nb) / nsp;
    return (e + (step - 1u)) & ~(step - 1u);
}

/* merge nsp already-warp-reduced partials of one row.  Sm must be
 * __shared__ float[8][Q4_WGV_SP_MAX]; the trailing __syncthreads keeps the
 * next expert iteration from overwriting a slot that is still being read. */
#define Q4_WGV_SP_MERGE(sm, v)                                                 \
    if ((nsp) > 1u) {                                                          \
        (sm)[(threadIdx.x >> 5) / (nsp)][(threadIdx.x >> 5) % (nsp)] = (v);    \
        __syncthreads();                                                       \
        if (q4_sp == 0u) {                                                     \
            uint32_t q4_ri = (threadIdx.x >> 5) / (nsp);                       \
            (v) = (sm)[q4_ri][0];                                              \
            for (uint32_t q4_k = 1u; q4_k < (nsp); q4_k++)                     \
                (v) += (sm)[q4_ri][q4_k];                                      \
        }                                                                      \
        __syncthreads();                                                       \
    }
/* same for the gate+up kernels, which carry two accumulators and must not
 * pay two sync pairs. */
#define Q4_WGV_SP_MERGE2(sm1, sm2, a, b)                                       \
    if ((nsp) > 1u) {                                                          \
        uint32_t q4_gi = (threadIdx.x >> 5) / (nsp);                            \
        uint32_t q4_gs = (threadIdx.x >> 5) % (nsp);                            \
        (sm1)[q4_gi][q4_gs] = (a);                                             \
        (sm2)[q4_gi][q4_gs] = (b);                                             \
        __syncthreads();                                                       \
        if (q4_sp == 0u) {                                                     \
            (a) = (sm1)[q4_gi][0];                                             \
            (b) = (sm2)[q4_gi][0];                                             \
            for (uint32_t q4_k = 1u; q4_k < (nsp); q4_k++) {                   \
                (a) += (sm1)[q4_gi][q4_k];                                     \
                (b) += (sm2)[q4_gi][q4_k];                                     \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }

/* ---- per-type warp dots: wr points at one packed row; returns the
 * lane-local partial (caller does wgv_shfl_red).  Each body is the same
 * lane mapping the standalone wgv_* kernel uses — the multi-projection
 * gemv_mw dispatches to these per segment. ---- */

__device__ static float wdot_q8_0(const uint8_t *wr, const float *x,
                                  uint32_t in_dim, uint32_t lane) {
    const uint32_t nb = in_dim >> 5;
    const uint32_t b = lane >> 1, m = (lane & 1u) * 16u;
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 16u) {
        uint32_t bb = b0 + b;
        if (bb < nb)
            acc += wgd_q8h(wr + (size_t)bb * 34ull, x + bb * 32u, m);
    }
    return acc;
}

/* Per-format warp-per-row kernels: one output row per warp. */
__global__ static void wgv_q8_0_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale, float act) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 5;
    const uint8_t *wr = w + (size_t)row * nb * 34ull;
    float acc = wdot_q8_0(wr, x, in_dim, lane);
    acc = wgv_shfl_red(acc);
    if (lane == 0) {
        float v = acc * scale;
        if (act != 0.f) {
            v *= act;
            v = v / (1.f + expf(-v));
        }
        out[row] = v;
    }
}

#define Q4_WDOT_IQ256(name, octfn, bb)                                         \
    __device__ static float wdot_##name(const uint8_t *wr, const float *x,     \
                                        uint32_t in_dim, uint32_t lane) {     \
        uint32_t nb = in_dim >> 8;                                             \
        float acc = 0.f;                                                       \
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {                             \
            acc += octfn(wr + (size_t)b0 * (bb),                               \
                         x + b0 * 256u + lane * 8u, lane);                     \
            if (b0 + 1 < nb)                                                   \
                acc += octfn(wr + (size_t)(b0 + 1) * (bb),                     \
                             x + (b0 + 1) * 256u + lane * 8u, lane);           \
        }                                                                      \
        return acc;                                                            \
    }
#define Q4_WGV_IQ256(name, octfn, bb)                                          \
    Q4_WDOT_IQ256(name, octfn, bb)                                             \
    __global__ static void wgv_##name(float *out, const uint8_t *w,            \
                                const float *x, uint32_t in_dim,               \
                                uint32_t out_dim, float scale) {               \
        Q4_WGV_ROW();                                                          \
        uint32_t nb = in_dim >> 8;                                             \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        float acc = wdot_##name(wr, x, in_dim, lane);                          \
        acc = wgv_shfl_red(acc);                                               \
        if (lane == 0) out[row] = acc * scale;                                 \
    }
Q4_WGV_IQ256(iq2xxs_k, wgd_iq2xxs, 66ull)
Q4_WGV_IQ256(iq2xs_k, wgd_iq2xs, 74ull)
Q4_WGV_IQ256(iq2_s_k, wgd_iq2s, 82ull)
Q4_WGV_IQ256(iq3_s_k, wgd_iq3s, 110ull)
Q4_WGV_IQ256(iq3xxs_k, wgd_iq3xxs, 98ull)
Q4_WGV_IQ256(iq4xs_k, wgd_iq4xs, 136ull)

__device__ static float wdot_iq4nl(const uint8_t *wr, const float *x,
                                   uint32_t in_dim, uint32_t lane) {
    const uint32_t nb = in_dim >> 5;
    const uint32_t b = lane >> 2;
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
        uint32_t bb = b0 + b;
        if (bb < nb)
            acc += wgd_iq4nl(wr + (size_t)bb * 18ull,
                             x + bb * 32u + (lane & 3u) * 8u, lane);
    }
    return acc;
}
__global__ static void wgv_iq4_nl_k(float *out, const uint8_t *w,
                                    const float *x, uint32_t in_dim,
                                    uint32_t out_dim, float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 5;
    const uint8_t *wr = w + (size_t)row * nb * 18ull;
    float acc = wdot_iq4nl(wr, x, in_dim, lane);
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

__device__ static float wdot_f32(const uint8_t *wr_, const float *x,
                                 uint32_t in_dim, uint32_t lane) {
    const float *wr = (const float *)wr_;
    float acc = 0.f;
    uint32_t e0 = lane * 8u;
    for (; e0 + 8u <= in_dim; e0 += 256u)
        acc += wgd_f32o(wr + e0, x + e0);
    for (uint32_t e = (in_dim & ~7u) + lane; e < in_dim; e += 32u)
        acc += wr[e] * x[e];
    return acc;
}
__global__ static void wgv_f32_k(float *out, const float *w, const float *x,
                                 uint32_t in_dim, uint32_t out_dim,
                                 float scale) {
    Q4_WGV_ROW();
    const float *wr = w + (size_t)row * in_dim;
    float acc = wdot_f32((const uint8_t *)wr, x, in_dim, lane);
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

/* Warp-per-row Q6_K decode GEMV (the lm_head — the largest GEMV per token).
 * Same math as gemv_q6_k_k with the 16-elem units strided across lanes and a
 * shuffle reduce; the block version serialised 16 units over 128 threads and
 * paid a full shared-memory reduction per row. */
/* One Q6_K unit (16 values: 4 scale groups x 4) of a warp-per-row dot:
 * decode, then the per-group sums. Shared by wdot_q6k and the n-row
 * wgvt_q6_k so both produce the same bits for a row. */
struct q6u_t {
    float q[4][4];
    float ds[4];
};

__device__ static __forceinline__ void q6u_decode(const uint8_t *blk,
                                                  uint32_t h2, uint32_t l4,
                                                  q6u_t *u) {
    const uint32_t qlv0 = *(const uint32_t *)(blk + h2 * 64u + l4);
    const uint32_t qlv1 = *(const uint32_t *)(blk + h2 * 64u + l4 + 32u);
    const uint32_t qhv = *(const uint32_t *)(blk + 128u + h2 * 32u + l4);
    const int8_t *sc = (const int8_t *)(blk + 192u) + h2 * 8;
    const float d = f16d(blk + 208u);
    const uint32_t is = l4 >> 4;
    u->ds[0] = d * (float)sc[is];
    u->ds[1] = d * (float)sc[is + 2u];
    u->ds[2] = d * (float)sc[is + 4u];
    u->ds[3] = d * (float)sc[is + 6u];
#pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
        const uint32_t qlo = (qlv0 >> (j * 8)) & 0xffu;
        const uint32_t qlh = (qlv1 >> (j * 8)) & 0xffu;
        const uint32_t qhb = (qhv >> (j * 8)) & 0xffu;
        u->q[0][j] = (float)(int((qlo & 15u) | ((qhb & 3u) << 4)) - 32);
        u->q[1][j] = (float)(int((qlh & 15u) | (((qhb >> 2) & 3u) << 4)) - 32);
        u->q[2][j] = (float)(int((qlo >> 4) | (((qhb >> 4) & 3u) << 4)) - 32);
        u->q[3][j] = (float)(int((qlh >> 4) | (((qhb >> 6) & 3u) << 4)) - 32);
    }
}

/* 4 scale groups x float4 x loads: d*sc factors out, x comes in 16B. */
__device__ static __forceinline__ float q6u_dot(const q6u_t *u,
                                                const float *xt4) {
#pragma clang fp reassociate(off)
    const float4 xa = *(const float4 *)xt4;
    const float4 xb = *(const float4 *)(xt4 + 32u);
    const float4 xc = *(const float4 *)(xt4 + 64u);
    const float4 xd = *(const float4 *)(xt4 + 96u);
    const float xv[4] = { xa.x, xa.y, xa.z, xa.w };
    const float yv[4] = { xb.x, xb.y, xb.z, xb.w };
    const float zv[4] = { xc.x, xc.y, xc.z, xc.w };
    const float wv[4] = { xd.x, xd.y, xd.z, xd.w };
    /* explicit fma chains, no reassociation: the same bits as q6t_core's
     * unit sum for any NT / R (and in every Q6_K kernel that shares this). */
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
        s0 = __builtin_fmaf(u->q[0][j], xv[j], s0);
        s1 = __builtin_fmaf(u->q[1][j], yv[j], s1);
        s2 = __builtin_fmaf(u->q[2][j], zv[j], s2);
        s3 = __builtin_fmaf(u->q[3][j], wv[j], s3);
    }
    float p = u->ds[0] * s0;
    p = __builtin_fmaf(u->ds[1], s1, p);
    p = __builtin_fmaf(u->ds[2], s2, p);
    return __builtin_fmaf(u->ds[3], s3, p);
}

__device__ static float wdot_q6k(const uint8_t *wr, const float *x,
                                 uint32_t in_dim, uint32_t lane) {
#pragma clang fp reassociate(off)
    uint32_t nb = in_dim / 256u;
    float acc = 0.f;
    uint32_t nu = nb * 16u;
    for (uint32_t u = lane; u < nu; u += 32u) {
        const uint8_t *blk = wr + (size_t)(u >> 4) * 210u;
        uint32_t qu = u & 15u;
        uint32_t h2 = qu >> 3, l4 = (qu & 7u) * 4u;
        q6u_t un;
        q6u_decode(blk, h2, l4, &un);
        acc += q6u_dot(&un, x + (u >> 4) * 256u + h2 * 128u + l4);
    }
    return acc;
}
__global__ static void wgv_q6_k_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 210u;
    float acc = wdot_q6k(wr, x, in_dim, lane);
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

/* 2..8 tokens.  Each token calls wdot_q6k, so the row matches the
 * single-token warp kernel.  The block-per-token fallback re-reads the
 * row and pays a shared reduction per token. */
__global__ static void wgvn_q6_k(float *out, const uint8_t *w, const float *x,
                                 uint32_t in_dim, uint32_t out_dim,
                                 float scale, uint32_t nt) {
    Q4_WGV_ROW();
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 210u;
    for (uint32_t t = 0; t < nt; t++) {
        float acc = wdot_q6k(wr, x + (size_t)t * in_dim, in_dim, lane);
        acc = wgv_shfl_red(acc);
        if (lane == 0) out[(size_t)t * out_dim + row] = acc * scale;
    }
}

__device__ static void scale_min_k4_dev(int is, const uint8_t *sc,
                                        float *d1, float *m1);

/* q4_k/q5_k warp dots: 8 sub-blocks of 32 elems per 256-el superblock;
 * lane l covers 8 elements of sub-block (l>>2) at position (l&3)*8. */
__device__ static float wdot_q4k(const uint8_t *wr, const float *x,
                                 uint32_t in_dim, uint32_t lane) {
    const uint32_t nb = in_dim / 256u;
    const uint32_t is = lane >> 2;            /* sub-block 0..7 */
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;    /* 8 elems inside sub-block */
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 144u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *qs = blk + 16u + j64 * 32u + pos;
        const float *xb = x + b * 256u + xoff;
        const float4 xa = *(const float4 *)xb;
        const float4 xbv = *(const float4 *)(xb + 4u);
        const float xr[8] = { xa.x,  xa.y,  xa.z,  xa.w,
                              xbv.x, xbv.y, xbv.z, xbv.w };
        float ds = 0.f, xs = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t q = half ? (uint32_t)(qs[j] >> 4)
                                    : (uint32_t)(qs[j] & 0xFu);
            ds += (float)q * xr[j];
            xs += xr[j];
        }
        acc += d1 * ds - m1 * xs;
    }
    return acc;
}
__global__ static void wgv_q4_k_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 144u;
    float acc = wdot_q4k(wr, x, in_dim, lane);
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

__device__ static float wdot_q5k(const uint8_t *wr, const float *x,
                                 uint32_t in_dim, uint32_t lane) {
    const uint32_t nb = in_dim / 256u;
    const uint32_t is = lane >> 2;
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 176u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *ql = blk + 48u + j64 * 32u + pos;
        const uint8_t *qh = blk + 16u + pos;
        const float *xb = x + b * 256u + xoff;
        const float4 xa = *(const float4 *)xb;
        const float4 xbv = *(const float4 *)(xb + 4u);
        const float xr[8] = { xa.x,  xa.y,  xa.z,  xa.w,
                              xbv.x, xbv.y, xbv.z, xbv.w };
        float ds = 0.f, xs = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t q = ((half ? (uint32_t)(ql[j] >> 4)
                                     : (uint32_t)(ql[j] & 0xFu))) |
                               (((uint32_t)(qh[j] >> is) & 1u) << 4);
            ds += (float)q * xr[j];
            xs += xr[j];
        }
        acc += d1 * ds - m1 * xs;
    }
    return acc;
}
__global__ static void wgv_q5_k_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 176u;
    float acc = wdot_q5k(wr, x, in_dim, lane);
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

/* 32-value-block warp dots (q4_0/q5_0): same geometry as wgv_iq4_nl_k —
 * lane covers a quarter of each block, 8 blocks per warp-iter. */
#define Q4_WGV_B32(name, octfn, bb)                                            \
    __device__ static float wdot_##name(const uint8_t *wr, const float *x,     \
                                        uint32_t in_dim, uint32_t lane) {     \
        const uint32_t nb = in_dim >> 5;                                       \
        const uint32_t b = lane >> 2;                                          \
        float acc = 0.f;                                                       \
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {                             \
            uint32_t bb2 = b0 + b;                                             \
            if (bb2 < nb)                                                      \
                acc += octfn(wr + (size_t)bb2 * (bb),                          \
                             x + bb2 * 32u + (lane & 3u) * 8u, lane);          \
        }                                                                      \
        return acc;                                                            \
    }                                                                          \
    __global__ static void wgv_##name(float *out, const uint8_t *w,            \
                                const float *x, uint32_t in_dim,               \
                                uint32_t out_dim, float scale) {               \
        Q4_WGV_ROW();                                                          \
        const uint32_t nb = in_dim >> 5;                                       \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        float acc = wdot_##name(wr, x, in_dim, lane);                          \
        acc = wgv_shfl_red(acc);                                               \
        if (lane == 0) out[row] = acc * scale;                                 \
    }
Q4_WGV_B32(q4_0_k, wgd_q40, 18ull)
Q4_WGV_B32(q5_0_k, wgd_q50, 22ull)

/* q2_0: 64-elem blocks, one byte (4 elems) per lane, 2 blocks/warp-iter. */
__device__ static float wdot_q20(const uint8_t *wr, const float *x,
                                 uint32_t in_dim, uint32_t lane) {
    const uint32_t nb = in_dim >> 6;
    const uint32_t b = lane >> 4;
    const uint32_t byte = lane & 15u;
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
        uint32_t bb = b0 + b;
        if (bb < nb)
            acc += wgd_q20(wr + (size_t)bb * 18ull,
                           x + bb * 64u + byte * 4u, byte);
    }
    return acc;
}
__global__ static void wgv_q2_0_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 6;
    const uint8_t *wr = w + (size_t)row * nb * 18ull;
    float acc = wdot_q20(wr, x, in_dim, lane);
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

/* ---- wgvn: warp-per-row for 2..8-token batches (MTP verify, short
 * prefill).  Each warp owns one output row and carries n_batch
 * accumulators; acc[t] receives adds in exactly the order the n_batch==1
 * wgv kernel applies them, so out[t*out_dim + row] is bit-identical to a
 * single-token call on x_t. ---- */
#define WGVN_MAX 8u
#define Q4_WGVN_EPILOG()                                                       \
    for (uint32_t t = 0; t < nt; t++) {                                        \
        float v = wgv_shfl_red(acc[t]);                                        \
        if (lane == 0) out[(size_t)t * out_dim + row] = v * scale;             \
    }

__global__ static void wgvn_q8_0_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 5;
    const uint8_t *wr = w + (size_t)row * nb * 34ull;
    const uint32_t b = lane >> 1, m = (lane & 1u) * 16u;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 16u) {
        uint32_t bb = b0 + b;
        if (bb < nb) {
            const uint8_t *blk = wr + (size_t)bb * 34ull;
            for (uint32_t t = 0; t < nt; t++)
                acc[t] += wgd_q8h(blk, x + (size_t)t * in_dim + bb * 32u, m);
        }
    }
    Q4_WGVN_EPILOG();
}

#define Q4_WGVN_IQ256(name, octfn, bb)                                         \
    __global__ static void name(float *out, const uint8_t *w, const float *x,  \
                                uint32_t in_dim, uint32_t out_dim,             \
                                float scale, uint32_t nt) {                    \
        Q4_WGV_ROW();                                                          \
        uint32_t nb = in_dim >> 8;                                             \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        float acc[WGVN_MAX];                                                   \
        for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;                        \
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {                             \
            const uint8_t *b0p = wr + (size_t)b0 * (bb);                       \
            const uint8_t *b1p = wr + (size_t)(b0 + 1u) * (bb);                \
            for (uint32_t t = 0; t < nt; t++) {                                \
                const float *xt = x + (size_t)t * in_dim + b0 * 256u +         \
                                    lane * 8u;                                 \
                acc[t] += octfn(b0p, xt, lane);                                \
                if (b0 + 1 < nb)                                               \
                    acc[t] += octfn(b1p, xt + 256u, lane);                     \
            }                                                                  \
        }                                                                      \
        Q4_WGVN_EPILOG();                                                      \
    }
Q4_WGVN_IQ256(wgvn_iq2xxs_k, wgd_iq2xxs, 66ull)
Q4_WGVN_IQ256(wgvn_iq2xs_k, wgd_iq2xs, 74ull)
Q4_WGVN_IQ256(wgvn_iq2_s_k, wgd_iq2s, 82ull)
Q4_WGVN_IQ256(wgvn_iq3_s_k, wgd_iq3s, 110ull)
Q4_WGVN_IQ256(wgvn_iq3xxs_k, wgd_iq3xxs, 98ull)
Q4_WGVN_IQ256(wgvn_iq4xs_k, wgd_iq4xs, 136ull)

/* 32-value-block geometry for iq4_nl / q4_0 / q5_0: lane covers a quarter
 * of each block, 8 blocks per warp-iter — same as wgv_iq4_nl_k/B32. */
#define Q4_WGVN_B32(name, octfn, bb)                                           \
    __global__ static void name(float *out, const uint8_t *w, const float *x,  \
                                uint32_t in_dim, uint32_t out_dim,             \
                                float scale, uint32_t nt) {                    \
        Q4_WGV_ROW();                                                          \
        const uint32_t nb = in_dim >> 5;                                       \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        const uint32_t b = lane >> 2;                                          \
        float acc[WGVN_MAX];                                                   \
        for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;                        \
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {                             \
            uint32_t bb2 = b0 + b;                                             \
            if (bb2 < nb) {                                                    \
                const uint8_t *blk = wr + (size_t)bb2 * (bb);                  \
                for (uint32_t t = 0; t < nt; t++)                              \
                    acc[t] += octfn(blk, x + (size_t)t * in_dim +              \
                                    bb2 * 32u + (lane & 3u) * 8u, lane);       \
            }                                                                  \
        }                                                                      \
        Q4_WGVN_EPILOG();                                                      \
    }
Q4_WGVN_B32(wgvn_iq4_nl_k, wgd_iq4nl, 18ull)
Q4_WGVN_B32(wgvn_q4_0_k, wgd_q40, 18ull)
Q4_WGVN_B32(wgvn_q5_0_k, wgd_q50, 22ull)

__global__ static void wgvn_q2_0_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 6;
    const uint8_t *wr = w + (size_t)row * nb * 18ull;
    const uint32_t b = lane >> 4;
    const uint32_t byte = lane & 15u;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
        uint32_t bb = b0 + b;
        if (bb < nb) {
            const uint8_t *blk = wr + (size_t)bb * 18ull;
            for (uint32_t t = 0; t < nt; t++)
                acc[t] += wgd_q20(blk, x + (size_t)t * in_dim + bb * 64u +
                                  byte * 4u, byte);
        }
    }
    Q4_WGVN_EPILOG();
}

__global__ static void wgvn_f32_k(float *out, const float *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale, uint32_t nt) {
    Q4_WGV_ROW();
    const float *wr = w + (size_t)row * in_dim;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    uint32_t e0 = lane * 8u;
    for (; e0 + 8u <= in_dim; e0 += 256u)
        for (uint32_t t = 0; t < nt; t++)
            acc[t] += wgd_f32o(wr + e0, x + (size_t)t * in_dim + e0);
    for (uint32_t e = (in_dim & ~7u) + lane; e < in_dim; e += 32u)
        for (uint32_t t = 0; t < nt; t++)
            acc[t] += wr[e] * x[(size_t)t * in_dim + e];
    Q4_WGVN_EPILOG();
}

__global__ static void wgvn_bf16_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint16_t *wr = (const uint16_t *)(w + (size_t)row * in_dim * 2u);
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    if (!(in_dim & 1u) && (((uintptr_t)w & 3u) == 0)) {
        const uint32_t *wr32 = (const uint32_t *)wr;
        const uint32_t n2 = in_dim >> 1;
        for (uint32_t i = lane; i < n2; i += 32u) {
            uint32_t pr = wr32[i];
            float lo, hi;
            uint32_t ul = pr << 16, uh = pr & 0xffff0000u;
            memcpy(&lo, &ul, 4);
            memcpy(&hi, &uh, 4);
            for (uint32_t t = 0; t < nt; t++)
                acc[t] = fmaf(lo, x[(size_t)t * in_dim + 2u * i],
                              fmaf(hi, x[(size_t)t * in_dim + 2u * i + 1u],
                                   acc[t]));
        }
    } else {
        for (uint32_t i = lane; i < in_dim; i += 32u)
            for (uint32_t t = 0; t < nt; t++)
                acc[t] = fmaf(bf16_f(wr[i]), x[(size_t)t * in_dim + i],
                              acc[t]);
    }
    for (uint32_t t = 0; t < nt; t++) {
        for (uint32_t o = 16; o; o >>= 1)
            acc[t] += __shfl_xor_sync(0xffffffffull, acc[t], o);
        if (lane == 0) out[(size_t)t * out_dim + row] = acc[t] * scale;
    }
}

/* 2..8 token bf16, one 256-thread block per row.  Same per-token add order
 * as gemv_bf16_bk: the weight uint4 is loaded once and applied to every
 * token.  Warp-per-row on out_dim=320 (hc_*_down, hc_*_inject, ssm_beta /
 * ssm_alpha) only fills ~40 blocks. */
template <uint32_t NT>
__device__ __forceinline__ static void bf16_bn_row(float *out, const uint8_t *w,
                                                   const float *x, uint32_t in_dim,
                                                   uint32_t out_dim, uint32_t row,
                                                   float scale, float act) {
    const uint4 *w4 = (const uint4 *)(w + (size_t)row * in_dim * 2u);
    const uint32_t n8 = in_dim >> 3;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    float acc[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    for (uint32_t i = threadIdx.x; i < n8; i += blockDim.x) {
        const uint4 p = w4[i];
        float a, b;
        bf16x2_f(p.x, &a, &b);
        float a1, b1, a2, b2, a3, b3;
        bf16x2_f(p.y, &a1, &b1);
        bf16x2_f(p.z, &a2, &b2);
        bf16x2_f(p.w, &a3, &b3);
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            const float4 *x4 = (const float4 *)(x + (size_t)t * in_dim);
            const float4 xa = x4[2u * i], xb = x4[2u * i + 1u];
            acc[t] = fmaf(a, xa.x, fmaf(b, xa.y, acc[t]));
            acc[t] = fmaf(a1, xa.z, fmaf(b1, xa.w, acc[t]));
            acc[t] = fmaf(a2, xb.x, fmaf(b2, xb.y, acc[t]));
            acc[t] = fmaf(a3, xb.z, fmaf(b3, xb.w, acc[t]));
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        float a = acc[t];
        for (uint32_t o = 16; o; o >>= 1)
            a += __shfl_xor_sync(0xffffffffull, a, o);
        acc[t] = a;
    }
    __shared__ float sh[NT][8];
    if (!lane) {
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) sh[t][warp] = acc[t];
    }
    __syncthreads();
    if (!threadIdx.x) {
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float s = 0.f;
#pragma unroll
            for (uint32_t i = 0; i < 8; i++) s += sh[t][i];
            float v = s * scale;
            if (act != 0.f) {
                v *= act;
                v = v / (1.f + expf(-v));
            }
            out[(size_t)t * out_dim + row] = v;
        }
    }
}

template <uint32_t NT>
__global__ static void gemv_bf16_bn_t(float *out, const uint8_t *w,
                                      const float *x, uint32_t in_dim,
                                      uint32_t out_dim, float scale,
                                      float act) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    bf16_bn_row<NT>(out, w, x, in_dim, out_dim, row, scale, act);
}

/* Two bf16 tensors over the same activation rows in one launch (hc down +
 * inject, ssm beta + alpha): blocks [0, r0) run tensor 0, the rest tensor
 * 1, each row exactly as gemv_bf16_bn_t computes it. */
template <uint32_t NT>
__global__ static void gemv_bf16_bn2_t(float *o0, const uint8_t *w0,
                                       uint32_t r0, float a0, float *o1,
                                       const uint8_t *w1, uint32_t r1,
                                       float a1, const float *x,
                                       uint32_t in_dim) {
    const uint32_t row = blockIdx.x;
    if (row < r0)
        bf16_bn_row<NT>(o0, w0, x, in_dim, r0, row, 1.f, a0);
    else if (row - r0 < r1)
        bf16_bn_row<NT>(o1, w1, x, in_dim, r1, row - r0, 1.f, a1);
}

/* 2..8 token bf16, sub-warp groups.  G matches wgv_bf16v_k so a short row
 * (hc_*_up, in_dim 320) retires several rows per warp, and each weight
 * uint4 is applied to every token. */
template <uint32_t G, uint32_t NT>
__global__ static void wgvn_bf16v_t(float *out, const uint8_t *w,
                                    const float *x, uint32_t in_dim,
                                    uint32_t out_dim, float scale,
                                    float act) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t rpw = 32u / G;
    const uint32_t row = wid * rpw + lane / G;
    const uint32_t gl = lane % G;
    if (row >= out_dim) return;
    const uint4 *w4 = (const uint4 *)(w + (size_t)row * in_dim * 2u);
    const uint32_t n8 = in_dim >> 3;
    float acc[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    for (uint32_t i = gl; i < n8; i += G) {
        const uint4 p = w4[i];
        float a, b, a1, b1, a2, b2, a3, b3;
        bf16x2_f(p.x, &a, &b);
        bf16x2_f(p.y, &a1, &b1);
        bf16x2_f(p.z, &a2, &b2);
        bf16x2_f(p.w, &a3, &b3);
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            const float4 *x4 = (const float4 *)(x + (size_t)t * in_dim);
            const float4 xa = x4[2u * i], xb = x4[2u * i + 1u];
            acc[t] = fmaf(a, xa.x, fmaf(b, xa.y, acc[t]));
            acc[t] = fmaf(a1, xa.z, fmaf(b1, xa.w, acc[t]));
            acc[t] = fmaf(a2, xb.x, fmaf(b2, xb.y, acc[t]));
            acc[t] = fmaf(a3, xb.z, fmaf(b3, xb.w, acc[t]));
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        float a = acc[t];
        for (uint32_t o = G >> 1; o; o >>= 1)
            a += __shfl_xor_sync(0xffffffffull, a, o);
        if (!gl) {
            float v = a * scale;
            if (act != 0.f) {
                v *= act;
                v = v / (1.f + expf(-v));
            }
            out[(size_t)t * out_dim + row] = v;
        }
    }
}

__global__ static void wgvn_q4_k_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 144u;
    const uint32_t is = lane >> 2;
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 144u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *qs = blk + 16u + j64 * 32u + pos;
        for (uint32_t t = 0; t < nt; t++) {
            const float *xb = x + (size_t)t * in_dim + b * 256u + xoff;
            float ds = 0.f, xs = 0.f;
#pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t q = half ? (uint32_t)(qs[j] >> 4)
                                        : (uint32_t)(qs[j] & 0xFu);
                ds += (float)q * xb[j];
                xs += xb[j];
            }
            acc[t] += d1 * ds - m1 * xs;
        }
    }
    Q4_WGVN_EPILOG();
}

__global__ static void wgvn_q5_k_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 176u;
    const uint32_t is = lane >> 2;
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 176u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *ql = blk + 48u + j64 * 32u + pos;
        const uint8_t *qh = blk + 16u + pos;
        for (uint32_t t = 0; t < nt; t++) {
            const float *xb = x + (size_t)t * in_dim + b * 256u + xoff;
            float ds = 0.f, xs = 0.f;
#pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t q = ((half ? (uint32_t)(ql[j] >> 4)
                                          : (uint32_t)(ql[j] & 0xFu))) |
                                   (((uint32_t)(qh[j] >> is) & 1u) << 4);
                ds += (float)q * xb[j];
                xs += xb[j];
            }
            acc[t] += d1 * ds - m1 * xs;
        }
    }
    Q4_WGVN_EPILOG();
}

/* ---- n-row GEMV templated on the row count (MTP verify, 2..8 rows) ----
 * The wgvn_* kernels above take nt at run time: acc[WGVN_MAX] is indexed
 * by a runtime loop (scratch, not registers) and the K-quant/iq4_xs ones
 * redo the dequant inside the token loop.  nbench (2026-10-12): n=4 cost
 * 2.5x n=1 on Q6_K and 4.5-5.3x on Q4_K/Q5_K — no weight reuse at all.
 * These keep the n=1 kernels' lane geometry and per-token expression
 * order (so each row is bit-identical to the single-token GEMV), but
 * decode each lane's quants once into registers and reuse them for all
 * NT token rows.  Q4_WGVT=0 keeps the wgvn_* kernels. */
static int wgvt_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_WGVT");
        v = !(e && e[0] == '0');
    }
    return v;
}

#define Q4_WGVT_ACC0()                                                         \
    float acc[NT];                                                             \
    _Pragma("unroll") for (int t = 0; t < NT; t++) acc[t] = 0.f

#define Q4_WGVT_EPILOG()                                                       \
    _Pragma("unroll") for (int t = 0; t < NT; t++) {                           \
        const float v = wgv_shfl_red(acc[t]);                                  \
        if (lane == 0) out[(size_t)t * out_dim + row] = v * scale;             \
    }

/* 8 x values of token row t at element offset off, as xr[0..7]. */
#define Q4_WGVT_X8(t, off)                                                     \
    const float *xb = x + (size_t)(t) * in_dim + (off);                        \
    const float4 xa = *(const float4 *)xb;                                     \
    const float4 xc = *(const float4 *)(xb + 4u);                              \
    const float xr[8] = { xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w }

/* K-quant (q4_k/q5_k) tail: wdot_q4k's ds/xs sums over the decoded q[]. */
#define Q4_WGVT_KQ(off)                                                        \
    _Pragma("unroll") for (int t = 0; t < NT; t++) {                           \
        Q4_WGVT_X8(t, off);                                                    \
        float ds = 0.f, xs = 0.f;                                              \
        _Pragma("unroll") for (uint32_t j = 0; j < 8; j++) {                   \
            ds += qf[j] * xr[j];                                               \
            xs += xr[j];                                                       \
        }                                                                      \
        acc[t] += d1 * ds - m1 * xs;                                           \
    }

template <int NT>
__global__ static void wgvt_q4_k(float *out, const uint8_t *w, const float *x,
                                 uint32_t in_dim, uint32_t out_dim,
                                 float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 144u;
    const uint32_t is = lane >> 2, j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    Q4_WGVT_ACC0();
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 144u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint2 qv = *(const uint2 *)(blk + 16u + j64 * 32u + pos);
        float qf[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t by = ((j < 4 ? qv.x : qv.y) >> ((j & 3) * 8)) & 0xffu;
            qf[j] = (float)(half ? (by >> 4) : (by & 0xFu));
        }
        Q4_WGVT_KQ(b * 256u + xoff);
    }
    Q4_WGVT_EPILOG();
}

template <int NT>
__global__ static void wgvt_q5_k(float *out, const uint8_t *w, const float *x,
                                 uint32_t in_dim, uint32_t out_dim,
                                 float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 176u;
    const uint32_t is = lane >> 2, j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    Q4_WGVT_ACC0();
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 176u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint2 lv = *(const uint2 *)(blk + 48u + j64 * 32u + pos);
        const uint2 hv = *(const uint2 *)(blk + 16u + pos);
        float qf[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t sh = (j & 3) * 8;
            const uint32_t lb = ((j < 4 ? lv.x : lv.y) >> sh) & 0xffu;
            const uint32_t hb = ((j < 4 ? hv.x : hv.y) >> sh) & 0xffu;
            qf[j] = (float)((half ? (lb >> 4) : (lb & 0xFu)) |
                            (((hb >> is) & 1u) << 4));
        }
        Q4_WGVT_KQ(b * 256u + xoff);
    }
    Q4_WGVT_EPILOG();
}

template <int NT>
__global__ static void wgvt_iq4_xs(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 8;
    const uint8_t *wr = w + (size_t)row * nb * 136u;
    const uint32_t ib = lane >> 2, q = lane & 3u, hi = q >> 1;
    Q4_WGVT_ACC0();
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 136u;
        const float d = f16d(blk);
        uint32_t shv;
        memcpy(&shv, blk + 2, 2);
        const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                       ((int)((shv >> (2 * ib)) & 3u) << 4);
        const float dl = d * (float)(ls - 32);
        const uint2 qv = *(const uint2 *)(blk + 8 + ib * 16 + (q & 1u) * 8u);
        float qf[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t by = ((j < 4 ? qv.x : qv.y) >> ((j & 3) * 8)) & 0xffu;
            qf[j] = (float)q4_kvalues_iq4nl[hi ? (by >> 4) : (by & 0xFu)];
        }
        /* wgd_iq4xs: dl * sum(kvalue * x) */
#pragma unroll
        for (int t = 0; t < NT; t++) {
            Q4_WGVT_X8(t, b * 256u + lane * 8u);
            float a = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) a += qf[j] * xr[j];
            acc[t] += dl * a;
        }
    }
    Q4_WGVT_EPILOG();
}

/* Q6_K: wdot_q6k's unit geometry (16 units of 16 values per superblock,
 * strided over lanes); 4 scale groups x 4 values per unit. */
template <int NT>
__global__ static void wgvt_q6_k(float *out, const uint8_t *w, const float *x,
                                 uint32_t in_dim, uint32_t out_dim,
                                 float scale) {
#pragma clang fp reassociate(off)
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 210u;
    const uint32_t nu = nb * 16u;
    Q4_WGVT_ACC0();
    for (uint32_t u = lane; u < nu; u += 32u) {
        const uint8_t *blk = wr + (size_t)(u >> 4) * 210u;
        const uint32_t qu = u & 15u;
        const uint32_t h2 = qu >> 3, l4 = (qu & 7u) * 4u;
        q6u_t un;
        q6u_decode(blk, h2, l4, &un);
        const uint32_t xo = (u >> 4) * 256u + h2 * 128u + l4;
#pragma unroll
        for (int t = 0; t < NT; t++)
            acc[t] += q6u_dot(&un, x + (size_t)t * in_dim + xo);
    }
    Q4_WGVT_EPILOG();
}

#define Q4_WGVT_CASE(kern)                                                     \
    switch (nt) {                                                              \
    case 2: kern<2><<<g, 256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale); break; \
    case 3: kern<3><<<g, 256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale); break; \
    case 4: kern<4><<<g, 256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale); break; \
    case 5: kern<5><<<g, 256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale); break; \
    case 6: kern<6><<<g, 256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale); break; \
    case 7: kern<7><<<g, 256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale); break; \
    default: kern<8><<<g, 256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale); break; \
    }

/* ---- Q4_Q6T: Q6_K dense GEMV, R rows per warp, one template for n = 1..8 --
 * wdot_q6k's unit geometry (a lane owns one 16-value unit of a superblock,
 * 16 units per superblock, 32 lanes = two superblocks per iteration) but
 * each lane decodes the units of R rows once and then, per token, loads x
 * once (4 float4) for all R rows: x traffic per weight byte drops R-fold
 * (x is fp32: at n=8 it was 40x the bytes of the 6-bit weights).  The next
 * iteration's packed bytes are loaded before the current one is consumed.
 * Explicit fmaf + reassociate(off): every row rounds the same for any NT
 * and R, so n=1 and the n-row batch are bit-identical per row.   Q4_Q6T=0 restores
 * wdot_q6k / wgv_q6_k_k / wgvt_q6_k; Q4_Q6T_R=1/2/4 pins the rows per warp. */
static int q6t_mode(void) {  /* 0 off, 1 policy (default), 2 always */
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_Q6T");
        v = e ? atoi(e) : 1;
    }
    return v;
}

struct Q6Pk { uint32_t a, b, c, s, d; };

__device__ __forceinline__ Q6Pk q6_load(const uint8_t *blk, uint32_t h2,
                                        uint32_t l4) {
    Q6Pk p;
    p.a = *(const uint32_t *)(blk + h2 * 64u + l4);
    p.b = *(const uint32_t *)(blk + h2 * 64u + l4 + 32u);
    p.c = *(const uint32_t *)(blk + 128u + h2 * 32u + l4);
    const uint8_t *sc = blk + 192u + h2 * 8u + (l4 >> 4);
    p.s = (uint32_t)sc[0] | ((uint32_t)sc[2] << 8) | ((uint32_t)sc[4] << 16) |
          ((uint32_t)sc[6] << 24);
    unsigned short us;
    memcpy(&us, blk + 208u, 2);
    p.d = us;
    return p;
}

template <int NT, int R>
__device__ __forceinline__ void q6t_core(float (&acc)[R][NT],
                                         const uint8_t *const (&wr)[R],
                                         const float *x, uint32_t in_dim,
                                         uint32_t lane) {
#pragma clang fp reassociate(off)
    const uint32_t nu = (in_dim >> 8) * 16u;
    const uint32_t qu = lane & 15u;
    const uint32_t h2 = qu >> 3, l4 = (qu & 7u) * 4u;
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) acc[r][t] = 0.f;
    Q6Pk cur[R];
#pragma unroll
    for (int r = 0; r < R; r++)
        cur[r] = q6_load(wr[r] + (size_t)(lane < nu ? lane >> 4 : 0) * 210u, h2, l4);
    for (uint32_t u = lane; u < nu; u += 32u) {
        const uint32_t un = u + 32u < nu ? u + 32u : u;
        Q6Pk nxt[R];
#pragma unroll
        for (int r = 0; r < R; r++)
            nxt[r] = q6_load(wr[r] + (size_t)(un >> 4) * 210u, h2, l4);
        float q[R][4][4], ds[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
            __half hh;
            const unsigned short us = (unsigned short)cur[r].d;
            memcpy(&hh, &us, 2);
            const float d = __half2float(hh);
#pragma unroll
            for (int g = 0; g < 4; g++)
                ds[r][g] = d * (float)(int)(int8_t)(cur[r].s >> (8 * g));
#pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                const uint32_t qlo = (cur[r].a >> (j * 8)) & 0xffu;
                const uint32_t qlh = (cur[r].b >> (j * 8)) & 0xffu;
                const uint32_t qhb = (cur[r].c >> (j * 8)) & 0xffu;
                q[r][0][j] = (float)(int((qlo & 15u) | ((qhb & 3u) << 4)) - 32);
                q[r][1][j] = (float)(int((qlh & 15u) | (((qhb >> 2) & 3u) << 4)) - 32);
                q[r][2][j] = (float)(int((qlo >> 4) | (((qhb >> 4) & 3u) << 4)) - 32);
                q[r][3][j] = (float)(int((qlh >> 4) | (((qhb >> 6) & 3u) << 4)) - 32);
            }
        }
        const uint32_t xo = (u >> 4) * 256u + h2 * 128u + l4;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float *xt = x + (size_t)t * in_dim + xo;
            const float4 xa = *(const float4 *)xt;
            const float4 xb = *(const float4 *)(xt + 32u);
            const float4 xc = *(const float4 *)(xt + 64u);
            const float4 xd = *(const float4 *)(xt + 96u);
            const float xg[4][4] = {
                { xa.x, xa.y, xa.z, xa.w }, { xb.x, xb.y, xb.z, xb.w },
                { xc.x, xc.y, xc.z, xc.w }, { xd.x, xd.y, xd.z, xd.w } };
#pragma unroll
            for (int r = 0; r < R; r++) {
                float p = 0.f;
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    float s = 0.f;
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        s = __builtin_fmaf(q[r][g][j], xg[g][j], s);
                    p = __builtin_fmaf(ds[r][g], s, p);
                }
                acc[r][t] += p;
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) cur[r] = nxt[r];
    }
}

template <int NT, int R>
__global__ __launch_bounds__(256) static void
q6t_k(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
      uint32_t out_dim, float scale) {
    const uint32_t wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = wid * R;
    if (row0 >= out_dim) return;
    const uint64_t rb = (uint64_t)(in_dim >> 8) * 210ull;
    const uint8_t *wr[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const uint32_t rr = row0 + r < out_dim ? row0 + r : out_dim - 1u;
        wr[r] = w + (size_t)rr * rb;
    }
    float acc[R][NT];
    q6t_core<NT, R>(acc, wr, x, in_dim, lane);
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = wgv_shfl_red(acc[r][t]);
            if (lane == 0 && row0 + r < out_dim)
                out[(size_t)t * out_dim + row0 + r] = v * scale;
        }
}

static bool q6t_ok(const uint8_t *w, const float *x, uint32_t in_dim) {
    return q6t_mode() && !(in_dim & 255u) && !((uintptr_t)x & 15u) &&
           !((uintptr_t)w & 3u);
}

static int q6t_rows_env(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_Q6T_R");
        v = e ? atoi(e) : 0;
    }
    return v;
}

#define Q4_Q6T_RUN(NT_, R_)                                                    \
    q6t_k<NT_, R_><<<dim3((out_dim + 8u * (R_) - 1u) / (8u * (R_))), 256, 0,   \
                     g_str>>>(y, w, x, in_dim, out_dim, scale)
#define Q4_Q6T_NT(R_)                                                          \
    switch (nt) {                                                              \
    case 1: Q4_Q6T_RUN(1, R_); break;                                          \
    case 2: Q4_Q6T_RUN(2, R_); break;                                          \
    case 3: Q4_Q6T_RUN(3, R_); break;                                          \
    case 4: Q4_Q6T_RUN(4, R_); break;                                          \
    case 5: Q4_Q6T_RUN(5, R_); break;                                          \
    case 6: Q4_Q6T_RUN(6, R_); break;                                          \
    case 7: Q4_Q6T_RUN(7, R_); break;                                          \
    default: Q4_Q6T_RUN(8, R_); break;                                         \
    }

/* Which n-row Q6_K kernel: nbench (2026-10-12) says R=2 pays only for the
 * long tensors (R=4 at n=3..4, R=2 beyond) and at n>=6; the short ones lose
 * warps to the row sharing.  All of them give the same bits per row. */
static bool q6t_use(uint32_t nt, uint32_t out_dim) {
    const int m = q6t_mode();
    if (m == 2) return true;
    return m == 1 && ((nt >= 3 && out_dim >= 65536u) || nt >= 6);
}

static bool q6t_launch(float *y, const uint8_t *w, const float *x,
                       uint32_t in_dim, uint32_t out_dim, float scale,
                       uint32_t nt) {
    if (nt < 1 || nt > 8 || !q6t_ok(w, x, in_dim)) return false;
    int R = q6t_rows_env();
    if (R != 1 && R != 2 && R != 4) R = nt <= 2 ? 1 : (nt <= 4 ? 4 : 2);
    switch (R) {
    case 1: Q4_Q6T_NT(1); break;
    case 2: Q4_Q6T_NT(2); break;
    default: Q4_Q6T_NT(4); break;
    }
    return true;
}

/* ---- Q4_KQ16: Q4_K / Q5_K / IQ4_XS dense GEMV, 16-value lanes, R rows per
 * warp ----------------------------------------------------------------
 * The wdot_q4k / wgvt_* geometry (8 values per lane, 4 lanes per 32-value
 * sub-block, one superblock per warp iteration) read x four times per
 * weight byte it consumed for Q4_K (x is fp32: 8x the bytes of the 4-bit
 * weights) and re-decoded scales per 8 values.  Here a lane covers 16
 * bytes of quants (Q4_K/Q5_K: both nibbles = two 16-value sub-blocks;
 * IQ4_XS: one whole 32-value sub-block), 8 lanes per superblock, 4
 * superblocks per warp iteration; the scales are decoded from one 16-byte
 * header load; each x element is loaded once and used for R rows of the
 * warp (R = 1/2/4 at launch), so x traffic per weight byte drops R-fold.
 * ONE template serves n = 1 (NT=1) and the n-row verify (NT=2..8), and the
 * per-row arithmetic does not depend on R or NT, so every row is
 * bit-identical across all of them.  Q4_KQ16=0 restores the old kernels,
 * Q4_KQ16_R=1/2/4 pins the rows per warp (default by nt: 1,1,2,2,4..).
 * nbench 2026-10-12: n=1 350-400 -> 600-700 GB/s; n=4 270-320 -> 350-420.
 * Beyond n=2 the lane geometry is the limit: the 16B-per-lane weight map
 * makes each x float4 load touch 16 cache lines per wave (lanes sit 64-256 B
 * apart), ~6 us of the 24 us n=4 time on an 8.8 MB tensor; a dword-
 * interleaved map would fix that at the price of decoding all 8 scales per
 * lane (not done). */
static int kq16_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_KQ16");
        v = !(e && e[0] == '0');
    }
    return v;
}

enum { KQ_Q4K = 0, KQ_Q5K = 1, KQ_IQ4XS = 2 };

/* d1 = d * scale, m1 = dmin * min of sub-block j, as scale_min_k4_dev.
 * Branch-free: j is a per-lane constant, the 12 scale bytes are h.y/z/w
 * (bytes 0-3, 4-7, 8-11).  j < 4: s = q[j] & 63, m = q[j+4] & 63;
 * else s = (q[j+4] & 15) | (q[j-4] >> 6) << 4, m = (q[j+4] >> 4) |
 * (q[j] >> 6) << 4, where q[j-4], q[j] are in h.y, h.z and q[j+4] in h.w. */
__device__ static inline void kq_sm(const uint4 &h, uint32_t j, float d,
                                    float dmin, float &d1, float &m1) {
    const uint32_t sh = (j & 3u) * 8u;
    const uint32_t ys = h.y >> sh, zs = h.z >> sh, ws = h.w >> sh;
    const uint32_t s_lo = ys & 63u, m_lo = zs & 63u;
    const uint32_t s_hi = (ws & 0xFu) | ((ys & 0xC0u) >> 2);
    const uint32_t m_hi = ((ws >> 4) & 0xFu) | ((zs & 0xC0u) >> 2);
    const bool hi = j >= 4u;
    d1 = d * (float)(hi ? s_hi : s_lo);
    m1 = dmin * (float)(hi ? m_hi : m_lo);
}

__device__ static inline float kq_h2f(uint32_t bits) {
    __half hh;
    const unsigned short us = (unsigned short)bits;
    memcpy(&hh, &us, 2);
    return __half2float(hh);
}

__device__ static inline float kq_ub(uint32_t w, int k) {
    return (float)((w >> (8 * k)) & 0xffu);
}
__device__ static inline float kq_sb(uint32_t w, int k) {
    return (float)(int)(int8_t)(w >> (8 * k));
}

/* 16 consecutive x floats at p (16B aligned). */
__device__ static inline void kq_x16(const float *p, float v[16]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float4 a = *(const float4 *)(p + 4 * i);
        v[4 * i] = a.x; v[4 * i + 1] = a.y; v[4 * i + 2] = a.z;
        v[4 * i + 3] = a.w;
    }
}

struct KqPk { uint4 h, ql, qh; };

/* One lane's packed bytes of superblock b of one row. */
template <int TY>
__device__ __forceinline__ KqPk kq_load(const uint8_t *wr, uint32_t b,
                                        uint32_t u) {
    KqPk p;
    if (TY == KQ_IQ4XS) {
        const uint8_t *blk = wr + (size_t)b * 136u;
        const uint2 hd = *(const uint2 *)blk;
        const uint2 qa = *(const uint2 *)(blk + 8u + u * 16u);
        const uint2 qb = *(const uint2 *)(blk + 16u + u * 16u);
        p.h = make_uint4(hd.x, hd.y, 0u, 0u);
        p.ql = make_uint4(qa.x, qa.y, qb.x, qb.y);
        p.qh = make_uint4(0u, 0u, 0u, 0u);
    } else {
        const uint32_t j64 = u >> 1, p16 = (u & 1u) * 16u;
        const uint8_t *blk = wr + (size_t)b * (TY == KQ_Q4K ? 144u : 176u);
        p.h = *(const uint4 *)blk;
        if (TY == KQ_Q4K) {
            p.ql = *(const uint4 *)(blk + 16u + j64 * 32u + p16);
            p.qh = make_uint4(0u, 0u, 0u, 0u);
        } else {
            p.ql = *(const uint4 *)(blk + 48u + j64 * 32u + p16);
            p.qh = *(const uint4 *)(blk + 16u + p16);
        }
    }
    return p;
}

template <int TY, int NT, int R>
__device__ __forceinline__ void kq16_compute(float (&acc)[R][NT],
                                             const KqPk (&pk)[R],
                                             const float *x, uint32_t in_dim,
                                             uint32_t b, uint32_t u) {
    if (TY == KQ_IQ4XS) {
        const uint32_t xo = u * 32u;
        float dl[R];
        float kv[R][32];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint32_t ls =
                ((pk[r].h.y >> (4u * (u & 1u) + 8u * (u >> 1))) & 0xfu) |
                ((((pk[r].h.x >> 16) >> (2u * u)) & 3u) << 4);
            dl[r] = kq_h2f(pk[r].h.x) * (float)((int)ls - 32);
            const uint32_t qw[4] = { pk[r].ql.x, pk[r].ql.y, pk[r].ql.z,
                                     pk[r].ql.w };
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t lo = kv_iq4nl_lut4(qw[k] & 0x0f0f0f0fu);
                const uint32_t hi = kv_iq4nl_lut4((qw[k] >> 4) & 0x0f0f0f0fu);
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    kv[r][4 * k + m] = kq_sb(lo, m);
                    kv[r][16 + 4 * k + m] = kq_sb(hi, m);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float *xb = x + (size_t)t * in_dim + b * 256u + xo;
            float xv[32];
            kq_x16(xb, xv);
            kq_x16(xb + 16, xv + 16);
#pragma unroll
            for (int r = 0; r < R; r++) {
                float a = 0.f;
#pragma unroll
                for (int i = 0; i < 32; i++)
                    a = __builtin_fmaf(kv[r][i], xv[i], a);
                acc[r][t] = __builtin_fmaf(dl[r], a, acc[r][t]);
            }
        }
    } else {
        const uint32_t j64 = u >> 1, p16 = (u & 1u) * 16u;
        const uint32_t jlo = j64 * 2u, jhi = jlo + 1u;
        const uint32_t xo = j64 * 64u + p16;
        float d1l[R], m1l[R], d1h[R], m1h[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float d = kq_h2f(pk[r].h.x), dm = kq_h2f(pk[r].h.x >> 16);
            kq_sm(pk[r].h, jlo, d, dm, d1l[r], m1l[r]);
            kq_sm(pk[r].h, jhi, d, dm, d1h[r], m1h[r]);
        }
        /* Dequantize once per row (w = d1 * q - m1, the reference's own
         * expression), then every token is a plain FMA chain over the
         * widened weights: no per-token sum of x, no per-token widening. */
        float wl[R][16], wh[R][16];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint32_t qw[4] = { pk[r].ql.x, pk[r].ql.y, pk[r].ql.z,
                                     pk[r].ql.w };
            const uint32_t hw[4] = { pk[r].qh.x, pk[r].qh.y, pk[r].qh.z,
                                     pk[r].qh.w };
#pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t lo = qw[k] & 0x0f0f0f0fu;
                uint32_t hi = (qw[k] >> 4) & 0x0f0f0f0fu;
                if (TY == KQ_Q5K) {
                    lo |= ((hw[k] >> jlo) & 0x01010101u) << 4;
                    hi |= ((hw[k] >> jhi) & 0x01010101u) << 4;
                }
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    wl[r][4 * k + m] = __builtin_fmaf(d1l[r], kq_ub(lo, m), -m1l[r]);
                    wh[r][4 * k + m] = __builtin_fmaf(d1h[r], kq_ub(hi, m), -m1h[r]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float *xb = x + (size_t)t * in_dim + b * 256u + xo;
            float xl[16], xh[16];
            kq_x16(xb, xl);
            kq_x16(xb + 32, xh);
#pragma unroll
            for (int r = 0; r < R; r++) {
                float dsl = 0.f, dsh = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    dsl = __builtin_fmaf(wl[r][i], xl[i], dsl);
                    dsh = __builtin_fmaf(wh[r][i], xh[i], dsh);
                }
                acc[r][t] += dsl + dsh;
            }
        }
    }
}

/* Lane-local partials of R rows x NT tokens.  wr[r] = row r's packed bytes,
 * x = token-0 activations (token t at x + t*in_dim).  The next superblock's
 * weight bytes are loaded before the current one is consumed. */
template <int TY, int NT, int R>
__device__ __forceinline__ void kq16_core(float (&acc)[R][NT],
                                          const uint8_t *const (&wr)[R],
                                          const float *x, uint32_t in_dim,
                                          uint32_t lane) {
    /* no fast-math reassociation/contraction: every row must round the same
     * for any NT / R (explicit fmaf). */
#pragma clang fp reassociate(off)
    const uint32_t nb = in_dim >> 8;
    const uint32_t sbi = lane >> 3, u = lane & 7u;
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) acc[r][t] = 0.f;
    KqPk cur[R];
#pragma unroll
    for (int r = 0; r < R; r++)
        cur[r] = kq_load<TY>(wr[r], sbi < nb ? sbi : nb - 1u, u);
    for (uint32_t b = sbi; b < nb; b += 4u) {
        const uint32_t bn = b + 4u < nb ? b + 4u : b;
        KqPk nxt[R];
#pragma unroll
        for (int r = 0; r < R; r++) nxt[r] = kq_load<TY>(wr[r], bn, u);
        kq16_compute<TY, NT, R>(acc, cur, x, in_dim, b, u);
#pragma unroll
        for (int r = 0; r < R; r++) cur[r] = nxt[r];
    }
}

__host__ __device__ static inline uint64_t kq16_rowbytes(int ty) {
    return ty == KQ_Q4K ? 144ull : (ty == KQ_Q5K ? 176ull : 136ull);
}

/* gemv_mw_k segment dot: the lane-local partial of one row (R = NT = 1). */
template <int TY>
__device__ static float wdot_kq16(const uint8_t *wr, const float *x,
                                  uint32_t in_dim, uint32_t lane) {
    float acc[1][1];
    const uint8_t *const w1[1] = { wr };
    kq16_core<TY, 1, 1>(acc, w1, x, in_dim, lane);
    return acc[0][0];
}

template <int TY, int NT, int R>
__global__ __launch_bounds__(256) static void
kq16_k(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
       uint32_t out_dim, float scale) {
    const uint32_t wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = wid * R;
    if (row0 >= out_dim) return;
    const uint64_t rb = (uint64_t)(in_dim >> 8) * kq16_rowbytes(TY);
    const uint8_t *wr[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const uint32_t rr = row0 + r < out_dim ? row0 + r : out_dim - 1u;
        wr[r] = w + (size_t)rr * rb;
    }
    float acc[R][NT];
    kq16_core<TY, NT, R>(acc, wr, x, in_dim, lane);
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = wgv_shfl_red(acc[r][t]);
            if (lane == 0 && row0 + r < out_dim)
                out[(size_t)t * out_dim + row0 + r] = v * scale;
        }
}

static int kq16_rows_env(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_KQ16_R");
        v = e ? atoi(e) : 0;
    }
    return v;
}

#define Q4_KQ16_RUN(TY_, NT_, R_)                                              \
    kq16_k<TY_, NT_, R_><<<dim3((out_dim + 8u * (R_) - 1u) / (8u * (R_))),     \
                           256, 0, g_str>>>(y, w, x, in_dim, out_dim, scale)
#define Q4_KQ16_NT(TY_, R_)                                                    \
    switch (nt) {                                                              \
    case 1: Q4_KQ16_RUN(TY_, 1, R_); break;                                    \
    case 2: Q4_KQ16_RUN(TY_, 2, R_); break;                                    \
    case 3: Q4_KQ16_RUN(TY_, 3, R_); break;                                    \
    case 4: Q4_KQ16_RUN(TY_, 4, R_); break;                                    \
    case 5: Q4_KQ16_RUN(TY_, 5, R_); break;                                    \
    case 6: Q4_KQ16_RUN(TY_, 6, R_); break;                                    \
    case 7: Q4_KQ16_RUN(TY_, 7, R_); break;                                    \
    default: Q4_KQ16_RUN(TY_, 8, R_); break;                                   \
    }
#define Q4_KQ16_TY(TY_)                                                        \
    switch (R) {                                                               \
    case 1: Q4_KQ16_NT(TY_, 1); break;                                         \
    case 2: Q4_KQ16_NT(TY_, 2); break;                                         \
    default: Q4_KQ16_NT(TY_, 4); break;                                        \
    }

/* Alignment/shape gate shared by every caller, independent of nt, so the
 * n=1 and n-row paths always agree on which kernel family a tensor uses. */
static bool kq16_ok(uint32_t type, const uint8_t *w, const float *x,
                    uint32_t in_dim) {
    if (!kq16_on() || (in_dim & 255u) || ((uintptr_t)x & 15u)) return false;
    switch (type) {
    case Q4_T_Q4_K:
    case Q4_T_Q5_K:   return ((uintptr_t)w & 15u) == 0;
    case Q4_T_IQ4_XS: return ((uintptr_t)w & 7u) == 0;
    default:          return false;
    }
}

static bool kq16_launch(uint32_t type, float *y, const uint8_t *w,
                        const float *x, uint32_t in_dim, uint32_t out_dim,
                        float scale, uint32_t nt) {
    if (nt < 1 || nt > 8 || !kq16_ok(type, w, x, in_dim)) return false;
    /* rows per warp: x is re-read once per warp, so more rows amortize it
     * until the accumulators/widened weights crowd the registers.
     * nbench (RX 7900 XTX): R=1 best at n=1..2, R=2 at 3..4, R=4 beyond. */
    int R = kq16_rows_env();
    if (R != 1 && R != 2 && R != 4) R = nt <= 2 ? 1 : (nt <= 4 ? 2 : 4);
    switch (type) {
    case Q4_T_Q4_K:   Q4_KQ16_TY(KQ_Q4K); break;
    case Q4_T_Q5_K:   Q4_KQ16_TY(KQ_Q5K); break;
    default:          Q4_KQ16_TY(KQ_IQ4XS); break;
    }
    return true;
}

/* 2..8 rows, x 16-byte aligned with in_dim % 256 == 0 (every token row
 * stays aligned), w 8-byte aligned.  False: the caller's wgvn_* path. */
static bool wgvt_launch(uint32_t type, float *y, const uint8_t *w,
                        const float *x, uint32_t in_dim, uint32_t out_dim,
                        float scale, uint32_t nt) {
    if (nt >= 2 && nt <= 8 &&
        kq16_launch(type, y, w, x, in_dim, out_dim, scale, nt))
        return true;
    if (type == Q4_T_Q6_K && nt >= 2 && nt <= 8 && q6t_use(nt, out_dim) &&
        q6t_launch(y, w, x, in_dim, out_dim, scale, nt))
        return true;
    if (!wgvt_on() || nt < 2 || nt > 8 || (in_dim & 255u) ||
        ((uintptr_t)x & 15u) || ((uintptr_t)w & 7u))
        return false;
    if (type == Q4_T_Q6_K && ((uintptr_t)w & 3u)) return false;
    const dim3 g((out_dim + 7u) / 8u);
    switch (type) {
    case Q4_T_Q4_K: Q4_WGVT_CASE(wgvt_q4_k); break;
    case Q4_T_Q5_K: Q4_WGVT_CASE(wgvt_q5_k); break;
    case Q4_T_Q6_K: Q4_WGVT_CASE(wgvt_q6_k); break;
    case Q4_T_IQ4_XS: Q4_WGVT_CASE(wgvt_iq4_xs); break;
    default: return false;
    }
    return true;
}

/* Warp-per-(row, expert) for the batched routed-expert stages.  grid.y is a
 * small fixed stripe count (WMOE_GY): each block loops e += gridDim.y so the
 * launch stays small when the device-side count is only known at exec —
 * 64-way grids spent most of their block-dispatch time on early exits. */
#define WMOE_GY 8u
#define Q4_WMOE_IQ256(name, octfn, bb)                                         \
    __global__ static void name(float *out, const uint8_t *base,               \
                                uint64_t stride, const int32_t *slots,         \
                                const float *x, uint32_t xs,                   \
                                uint32_t in_dim, uint32_t out_dim,             \
                                const int32_t *n_d, uint32_t nsp) {            \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];                            \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 4u);                  \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 4u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = slo; b0 < shi; b0 += 4u) {                       \
                acc += octfn(wr + (size_t)b0 * (bb),                           \
                             xt + b0 * 256u + lane * 8u, lane);                \
                if (b0 + 1 < shi)                                              \
                    acc += octfn(wr + (size_t)(b0 + 1) * (bb),                 \
                                 xt + (b0 + 1) * 256u + lane * 8u, lane);      \
                if (b0 + 2 < shi)                                              \
                    acc += octfn(wr + (size_t)(b0 + 2) * (bb),                 \
                                 xt + (b0 + 2) * 256u + lane * 8u, lane);      \
                if (b0 + 3 < shi)                                              \
                    acc += octfn(wr + (size_t)(b0 + 3) * (bb),                 \
                                 xt + (b0 + 3) * 256u + lane * 8u, lane);      \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            Q4_WGV_SP_MERGE(q4_sm_a, acc)                                       \
            if (lane == 0 && q4_sp == 0u) out[(size_t)e * out_dim + row] = acc;\
        }                                                                      \
    }
Q4_WMOE_IQ256(wmoe_iq2xxs_bk, wgd_iq2xxs, 66ull)
Q4_WMOE_IQ256(wmoe_iq2xs_bk, wgd_iq2xs, 74ull)
Q4_WMOE_IQ256(wmoe_iq2_bk, wgd_iq2s, 82ull)
Q4_WMOE_IQ256(wmoe_iq3_bk, wgd_iq3s, 110ull)
Q4_WMOE_IQ256(wmoe_iq3xxs_bk, wgd_iq3xxs, 98ull)
Q4_WMOE_IQ256(wmoe_iq4xs_bk, wgd_iq4xs, 136ull)

/* gate+up fused: one warp does both rows of the same expert, sharing the
 * slot lookup and x read — halves launches and fixed per-block cost. */
#define Q4_WMOE_GU(name, octfn, bb)                                            \
    __global__ static void name(float *og, float *ou, const uint8_t *bg,       \
                                const uint8_t *bu, uint64_t stride,            \
                                const int32_t *slots, const float *x,          \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d,          \
                                uint32_t act, uint32_t nsp) {                  \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX], q4_sm_b[8][Q4_WGV_SP_MAX];  \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 4u);                  \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 4u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wg = bg + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const uint8_t *wu = bu + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            float ag = 0.f, au = 0.f;                                          \
            for (uint32_t b0 = slo; b0 < shi; b0 += 4u) {                       \
                ag += octfn(wg + (size_t)b0 * (bb),                            \
                            xt + b0 * 256u + lane * 8u, lane);                 \
                au += octfn(wu + (size_t)b0 * (bb),                            \
                            xt + b0 * 256u + lane * 8u, lane);                 \
                if (b0 + 1 < shi) {                                            \
                    ag += octfn(wg + (size_t)(b0 + 1) * (bb),                  \
                                xt + (b0 + 1) * 256u + lane * 8u, lane);       \
                    au += octfn(wu + (size_t)(b0 + 1) * (bb),                  \
                                xt + (b0 + 1) * 256u + lane * 8u, lane);       \
                }                                                              \
                if (b0 + 2 < shi) {                                            \
                    ag += octfn(wg + (size_t)(b0 + 2) * (bb),                  \
                                xt + (b0 + 2) * 256u + lane * 8u, lane);       \
                    au += octfn(wu + (size_t)(b0 + 2) * (bb),                  \
                                xt + (b0 + 2) * 256u + lane * 8u, lane);       \
                }                                                              \
                if (b0 + 3 < shi) {                                            \
                    ag += octfn(wg + (size_t)(b0 + 3) * (bb),                  \
                                xt + (b0 + 3) * 256u + lane * 8u, lane);       \
                    au += octfn(wu + (size_t)(b0 + 3) * (bb),                  \
                                xt + (b0 + 3) * 256u + lane * 8u, lane);       \
                }                                                              \
            }                                                                  \
            ag = wgv_shfl_red(ag);                                             \
            au = wgv_shfl_red(au);                                             \
            Q4_WGV_SP_MERGE2(q4_sm_a, q4_sm_b, ag, au)                          \
            if (lane == 0 && q4_sp == 0u) {                                    \
                og[(size_t)e * out_dim + row] =                                \
                    act ? ag / (1.f + expf(-ag)) * au : ag;                    \
                if (!act) ou[(size_t)e * out_dim + row] = au;                   \
            }                                                                  \
        }                                                                      \
    }
Q4_WMOE_GU(wmoe_iq2xxs_gu_bk, wgd_iq2xxs, 66ull)
Q4_WMOE_GU(wmoe_iq2xs_gu_bk, wgd_iq2xs, 74ull)
Q4_WMOE_GU(wmoe_iq2_gu_bk, wgd_iq2s, 82ull)
Q4_WMOE_GU(wmoe_iq3_gu_bk, wgd_iq3s, 110ull)
Q4_WMOE_GU(wmoe_iq3xxs_gu_bk, wgd_iq3xxs, 98ull)
Q4_WMOE_GU(wmoe_iq4xs_gu_bk, wgd_iq4xs, 136ull)

/* int8-activation gate+up: same geometry, but x comes pre-quantized in
 * per-32 i8 groups (xq8_k).  The oct's activation scale is
 * xd8[b0*8 + lane/4]; the int8 partial is scaled by dl*dx once per oct. */
#define Q4_WMOE_GU8(name, octfn, bb)                                           \
    __global__ static void name(float *og, float *ou, const uint8_t *bg,       \
                                const uint8_t *bu, uint64_t stride,            \
                                const int32_t *slots, const int8_t *xq8,       \
                                const float *xd8, uint32_t in_dim,             \
                                uint32_t out_dim, const int32_t *n_d,          \
                                uint32_t act, uint32_t nsp) {                  \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX],                            \
            q4_sm_b[8][Q4_WGV_SP_MAX];                                         \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 4u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wg = bg + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const uint8_t *wu = bu + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            float ag = 0.f, au = 0.f;                                          \
            for (uint32_t b0 = q4_sp_bound(q4_sp, nsp, nb, 4u); b0 < shi;      \
                 b0 += 4u) {                                                   \
                const uint32_t bb0 = b0, bb1 = b0 + 1, bb2 = b0 + 2,           \
                               bb3 = b0 + 3;                                   \
                if (bb0 < shi) {                                               \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb0 * (bb),                \
                                       xq8 + bb0 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb0 * (bb),                \
                                       xq8 + bb0 * 256u, lane, &du);           \
                    const float dx = xd8[bb0 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
                if (bb1 < shi) {                                               \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb1 * (bb),                \
                                       xq8 + bb1 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb1 * (bb),                \
                                       xq8 + bb1 * 256u, lane, &du);           \
                    const float dx = xd8[bb1 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
                if (bb2 < shi) {                                               \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb2 * (bb),                \
                                       xq8 + bb2 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb2 * (bb),                \
                                       xq8 + bb2 * 256u, lane, &du);           \
                    const float dx = xd8[bb2 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
                if (bb3 < shi) {                                               \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb3 * (bb),                \
                                       xq8 + bb3 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb3 * (bb),                \
                                       xq8 + bb3 * 256u, lane, &du);           \
                    const float dx = xd8[bb3 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
            }                                                                  \
            ag = wgv_shfl_red(ag);                                             \
            au = wgv_shfl_red(au);                                             \
            Q4_WGV_SP_MERGE2(q4_sm_a, q4_sm_b, ag, au)                          \
            if (lane == 0 && q4_sp == 0u) {                                    \
                og[(size_t)e * out_dim + row] =                                \
                    act ? ag / (1.f + expf(-ag)) * au : ag;                    \
                if (!act) ou[(size_t)e * out_dim + row] = au;                  \
            }                                                                  \
        }                                                                      \
    }
Q4_WMOE_GU8(wmoe_iq2xxs_gu_q8, wgd_iq2xxs_i8, 66ull)
Q4_WMOE_GU8(wmoe_iq2xs_gu_q8, wgd_iq2xs_i8, 74ull)
Q4_WMOE_GU8(wmoe_iq2_gu_q8, wgd_iq2s_i8, 82ull)
Q4_WMOE_GU8(wmoe_iq3_gu_q8, wgd_iq3s_i8, 110ull)
Q4_WMOE_GU8(wmoe_iq3xxs_gu_q8, wgd_iq3xxs_i8, 98ull)
Q4_WMOE_GU8(wmoe_iq4xs_gu_q8, wgd_iq4xs_i8, 136ull)

__global__ static void wmoe_iq4_gu_bk(float *og, float *ou, const uint8_t *bg,
                                      const uint8_t *bu, uint64_t stride,
                                      const int32_t *slots, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d,
                                      uint32_t act, uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX], q4_sm_b[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 8u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 8u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wg =
            bg + (size_t)slots[e] * stride + (size_t)row * nb * 18ull;
        const uint8_t *wu =
            bu + (size_t)slots[e] * stride + (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 2;
        float ag = 0.f, au = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < shi) {
                ag += wgd_iq4nl(wg + (size_t)bb * 18ull,
                                xt + bb * 32u + (lane & 3u) * 8u, lane);
                au += wgd_iq4nl(wu + (size_t)bb * 18ull,
                                xt + bb * 32u + (lane & 3u) * 8u, lane);
            }
        }
        ag = wgv_shfl_red(ag);
        au = wgv_shfl_red(au);
        Q4_WGV_SP_MERGE2(q4_sm_a, q4_sm_b, ag, au)
        if (lane == 0 && q4_sp == 0u) {
            og[(size_t)e * out_dim + row] =
                act ? ag / (1.f + expf(-ag)) * au : ag;
            if (!act) ou[(size_t)e * out_dim + row] = au;
        }
    }
}

/* Down-projection with fused weighted accumulate: each row atomically adds
 * w[e]*dot into y, removing the n_expert×n_embd intermediate + wsum pass. */
#define Q4_WMOE_YA(name, octfn, bb)                                            \
    __global__ static void name(float *y, const uint8_t *base,                 \
                                uint64_t stride, const int32_t *slots,         \
                                const float *wts, const float *x,              \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d,          \
                                uint32_t nsp) {                                \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];                            \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 4u);                  \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 4u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = slo; b0 < shi; b0 += 4u) {                       \
                acc += octfn(wr + (size_t)b0 * (bb),                           \
                             xt + b0 * 256u + lane * 8u, lane);                \
                if (b0 + 1 < shi)                                              \
                    acc += octfn(wr + (size_t)(b0 + 1) * (bb),                 \
                                 xt + (b0 + 1) * 256u + lane * 8u, lane);      \
                if (b0 + 2 < shi)                                              \
                    acc += octfn(wr + (size_t)(b0 + 2) * (bb),                 \
                                 xt + (b0 + 2) * 256u + lane * 8u, lane);      \
                if (b0 + 3 < shi)                                              \
                    acc += octfn(wr + (size_t)(b0 + 3) * (bb),                 \
                                 xt + (b0 + 3) * 256u + lane * 8u, lane);      \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            Q4_WGV_SP_MERGE(q4_sm_a, acc)                                       \
            if (lane == 0 && q4_sp == 0u)                                      \
                atomicAdd(y + row, acc * wts[e]);                              \
        }                                                                      \
    }
Q4_WMOE_YA(wmoe_iq2xxs_ya_bk, wgd_iq2xxs, 66ull)
Q4_WMOE_YA(wmoe_iq2xs_ya_bk, wgd_iq2xs, 74ull)
Q4_WMOE_YA(wmoe_iq2_ya_bk, wgd_iq2s, 82ull)
Q4_WMOE_YA(wmoe_iq3_ya_bk, wgd_iq3s, 110ull)
Q4_WMOE_YA(wmoe_iq3xxs_ya_bk, wgd_iq3xxs, 98ull)
Q4_WMOE_YA(wmoe_iq4xs_ya_bk, wgd_iq4xs, 136ull)

/* int8-activation IQ256 down-proj: per-expert quantized x slices; oct's
 * activation scale is xd8[e][b0*8 + lane/4]. */
#define Q4_WMOE_YA8(name, octfn, bb)                                           \
    __global__ static void name(float *y, const uint8_t *base,                 \
                                uint64_t stride, const int32_t *slots,         \
                                const float *wts, const int8_t *xq8,           \
                                const float *xd8, uint32_t in_dim,             \
                                uint32_t out_dim, const int32_t *n_d,          \
                                uint32_t nsp) {                                \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];                            \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 4u);                  \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 4u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const int8_t *xq = xq8 + (size_t)e * in_dim;                       \
            const float *xd = xd8 + (size_t)e * (in_dim >> 5);                 \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = slo; b0 < shi; b0 += 4u) {                       \
                for (uint32_t k = 0; k < 4u; k++) {                            \
                    const uint32_t bi = b0 + k;                                \
                    if (bi < shi) {                                            \
                        float dw;                                              \
                        const int32_t s = octfn(wr + (size_t)bi * (bb),        \
                                                xq + bi * 256u, lane, &dw);    \
                        acc += dw * xd[bi * 8u + (lane >> 2)] * (float)s;      \
                    }                                                          \
                }                                                              \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            Q4_WGV_SP_MERGE(q4_sm_a, acc)                                       \
            if (lane == 0 && q4_sp == 0u)                                      \
                atomicAdd(y + row, acc * wts[e]);                              \
        }                                                                      \
    }
Q4_WMOE_YA8(wmoe_iq2xxs_ya_q8, wgd_iq2xxs_i8, 66ull)
Q4_WMOE_YA8(wmoe_iq2xs_ya_q8, wgd_iq2xs_i8, 74ull)
Q4_WMOE_YA8(wmoe_iq2_ya_q8, wgd_iq2s_i8, 82ull)
Q4_WMOE_YA8(wmoe_iq3_ya_q8, wgd_iq3s_i8, 110ull)
Q4_WMOE_YA8(wmoe_iq3xxs_ya_q8, wgd_iq3xxs_i8, 98ull)
Q4_WMOE_YA8(wmoe_iq4xs_ya_q8, wgd_iq4xs_i8, 136ull)

__global__ static void wmoe_iq4_ya_bk(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d,
                                      uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 8u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 8u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 2;
        float acc = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < shi)
                acc += wgd_iq4nl(wr + (size_t)bb * 18ull,
                                 xt + bb * 32u + (lane & 3u) * 8u, lane);
        }
        acc = wgv_shfl_red(acc);
        Q4_WGV_SP_MERGE(q4_sm_a, acc)
        if (lane == 0 && q4_sp == 0u) atomicAdd(y + row, acc * wts[e]);
    }
}

__global__ static void wmoe_iq4_bk(float *out, const uint8_t *base,
                                   uint64_t stride, const int32_t *slots,
                                   const float *x, uint32_t xs,
                                   uint32_t in_dim, uint32_t out_dim,
                                   const int32_t *n_d, uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 8u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 8u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 2;
        float acc = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < shi)
                acc += wgd_iq4nl(wr + (size_t)bb * 18ull,
                                 xt + bb * 32u + (lane & 3u) * 8u, lane);
        }
        acc = wgv_shfl_red(acc);
        Q4_WGV_SP_MERGE(q4_sm_a, acc)
        if (lane == 0 && q4_sp == 0u) out[(size_t)e * out_dim + row] = acc;
    }
}

/* 32-value-block expert variants (q4_0/q5_0): same shape as wmoe_iq4_*. */
#define Q4_WMOE_B32(name, octfn, bb)                                           \
    __global__ static void name(float *out, const uint8_t *base,               \
                                uint64_t stride, const int32_t *slots,         \
                                const float *x, uint32_t xs,                   \
                                uint32_t in_dim, uint32_t out_dim,             \
                                const int32_t *n_d, uint32_t nsp) {            \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        const uint32_t nb = in_dim >> 5;                                       \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];                            \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 8u);                  \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 8u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            const uint32_t b = lane >> 2;                                      \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = slo; b0 < shi; b0 += 8u) {                       \
                uint32_t bb2 = b0 + b;                                         \
                if (bb2 < shi)                                                 \
                    acc += octfn(wr + (size_t)bb2 * (bb),                      \
                                 xt + bb2 * 32u + (lane & 3u) * 8u, lane);     \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            Q4_WGV_SP_MERGE(q4_sm_a, acc)                                       \
            if (lane == 0 && q4_sp == 0u)                                      \
                out[(size_t)e * out_dim + row] = acc;                          \
        }                                                                      \
    }
Q4_WMOE_B32(wmoe_q40_bk, wgd_q40, 18ull)
Q4_WMOE_B32(wmoe_q50_bk, wgd_q50, 22ull)

#define Q4_WMOE_GU32(name, octfn, bb)                                          \
    __global__ static void name(float *og, float *ou, const uint8_t *bg,       \
                                const uint8_t *bu, uint64_t stride,            \
                                const int32_t *slots, const float *x,          \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d,          \
                                uint32_t act, uint32_t nsp) {                  \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        const uint32_t nb = in_dim >> 5;                                       \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX],                            \
            q4_sm_b[8][Q4_WGV_SP_MAX];                                         \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 8u);                  \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 8u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wg = bg + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const uint8_t *wu = bu + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            const uint32_t b = lane >> 2;                                      \
            float ag = 0.f, au = 0.f;                                          \
            for (uint32_t b0 = slo; b0 < shi; b0 += 8u) {                      \
                uint32_t bb2 = b0 + b;                                         \
                if (bb2 < shi) {                                               \
                    ag += octfn(wg + (size_t)bb2 * (bb),                       \
                                xt + bb2 * 32u + (lane & 3u) * 8u, lane);      \
                    au += octfn(wu + (size_t)bb2 * (bb),                       \
                                xt + bb2 * 32u + (lane & 3u) * 8u, lane);      \
                }                                                              \
            }                                                                  \
            ag = wgv_shfl_red(ag);                                             \
            au = wgv_shfl_red(au);                                             \
            Q4_WGV_SP_MERGE2(q4_sm_a, q4_sm_b, ag, au)                          \
            if (lane == 0 && q4_sp == 0u) {                                    \
                og[(size_t)e * out_dim + row] =                                \
                    act ? ag / (1.f + expf(-ag)) * au : ag;                    \
                if (!act) ou[(size_t)e * out_dim + row] = au;                  \
            }                                                                  \
        }                                                                      \
    }
Q4_WMOE_GU32(wmoe_q40_gu_bk, wgd_q40, 18ull)
Q4_WMOE_GU32(wmoe_q50_gu_bk, wgd_q50, 22ull)

#define Q4_WMOE_YA32(name, octfn, bb)                                          \
    __global__ static void name(float *y, const uint8_t *base,                 \
                                uint64_t stride, const int32_t *slots,         \
                                const float *wts, const float *x,              \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d,          \
                                uint32_t nsp) {                                \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        const uint32_t nb = in_dim >> 5;                                       \
        __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];                            \
        Q4_WGV_ROW_SP(nsp);                                                    \
        const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 8u);                  \
        const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 8u);             \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            const uint32_t b = lane >> 2;                                      \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = slo; b0 < shi; b0 += 8u) {                       \
                uint32_t bb2 = b0 + b;                                         \
                if (bb2 < shi)                                                  \
                    acc += octfn(wr + (size_t)bb2 * (bb),                      \
                                 xt + bb2 * 32u + (lane & 3u) * 8u, lane);     \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            Q4_WGV_SP_MERGE(q4_sm_a, acc)                                       \
            if (lane == 0 && q4_sp == 0u)                                      \
                atomicAdd(y + row, acc * wts[e]);                              \
        }                                                                      \
    }
Q4_WMOE_YA32(wmoe_q40_ya_bk, wgd_q40, 18ull)
Q4_WMOE_YA32(wmoe_q50_ya_bk, wgd_q50, 22ull)

/* q2_0 experts: 64-elem blocks, one byte (4 elems) per lane, 2 blocks per
 * warp-iter (lane>>4 picks the block, lane&15 the byte). */
__global__ static void wmoe_q20_bk(float *out, const uint8_t *base,
                                   uint64_t stride, const int32_t *slots,
                                   const float *x, uint32_t xs,
                                   uint32_t in_dim, uint32_t out_dim,
                                   const int32_t *n_d, uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 2u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 2u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float acc = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < shi)
                acc += wgd_q20(wr + (size_t)bb * 18ull,
                               xt + bb * 64u + byte * 4u, byte);
        }
        acc = wgv_shfl_red(acc);
        Q4_WGV_SP_MERGE(q4_sm_a, acc)
        if (lane == 0 && q4_sp == 0u) out[(size_t)e * out_dim + row] = acc;
    }
}

__global__ static void wmoe_q20_gu_bk(float *og, float *ou, const uint8_t *bg,
                                      const uint8_t *bu, uint64_t stride,
                                      const int32_t *slots, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d,
                                      uint32_t act, uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX],
        q4_sm_b[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 2u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 2u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wg = bg + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const uint8_t *wu = bu + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float ag = 0.f, au = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < shi) {
                ag += wgd_q20(wg + (size_t)bb * 18ull,
                              xt + bb * 64u + byte * 4u, byte);
                au += wgd_q20(wu + (size_t)bb * 18ull,
                              xt + bb * 64u + byte * 4u, byte);
            }
        }
        ag = wgv_shfl_red(ag);
        au = wgv_shfl_red(au);
        Q4_WGV_SP_MERGE2(q4_sm_a, q4_sm_b, ag, au)
        if (lane == 0 && q4_sp == 0u) {
            og[(size_t)e * out_dim + row] =
                act ? ag / (1.f + expf(-ag)) * au : ag;
            if (!act) ou[(size_t)e * out_dim + row] = au;
        }
    }
}

__global__ static void wmoe_q20_ya_bk(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d, uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 2u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 2u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float acc = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < shi)
                acc += wgd_q20(wr + (size_t)bb * 18ull,
                               xt + bb * 64u + byte * 4u, byte);
        }
        acc = wgv_shfl_red(acc);
        Q4_WGV_SP_MERGE(q4_sm_a, acc)
        if (lane == 0 && q4_sp == 0u) atomicAdd(y + row, acc * wts[e]);
    }
}

/* int8-activation ya variants: x slices were quantized into per-32 i8
 * groups by xq8_k (see below).  Same warp-per-row geometry as the float
 * kernels; the per-element fma tail collapses into i32 dots scaled once
 * per weight block. */
__global__ static void wmoe_iq4_ya_q8(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const int8_t *xq8,
                                      const float *xd8, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d,
                                      uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 8u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 8u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const int8_t *xq = xq8 + (size_t)e * in_dim;
        const float *xd = xd8 + (size_t)e * (in_dim >> 5);
        const uint32_t b = lane >> 2;
        float acc = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < shi) {
                float dw;
                int32_t s = wgd_iq4nl_i8(wr + (size_t)bb * 18ull,
                                         xq + bb * 32u, lane, &dw);
                acc += dw * xd[bb] * (float)s;
            }
        }
        acc = wgv_shfl_red(acc);
        Q4_WGV_SP_MERGE(q4_sm_a, acc)
        if (lane == 0 && q4_sp == 0u) atomicAdd(y + row, acc * wts[e]);
    }
}

__global__ static void wmoe_q20_ya_q8(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const int8_t *xq8,
                                      const float *xd8, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d,
                                      uint32_t nsp) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    __shared__ float q4_sm_a[8][Q4_WGV_SP_MAX];
    Q4_WGV_ROW_SP(nsp);
    const uint32_t slo = q4_sp_bound(q4_sp, nsp, nb, 2u);
    const uint32_t shi = q4_sp_bound(q4_sp + 1u, nsp, nb, 2u);
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const int8_t *xq = xq8 + (size_t)e * in_dim;
        const float *xd = xd8 + (size_t)e * (in_dim >> 5);
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float acc = 0.f;
        for (uint32_t b0 = slo; b0 < shi; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < shi) {
                /* block bb = elements bb*64..+63 -> q8 groups 2*bb + hi.
                 * lane's 4 elems sit at group offset (byte&7)*4. */
                const uint32_t grp = bb * 2u + (byte >> 3);
                float dw;
                uint32_t xu;
                memcpy(&xu, xq + grp * 32u + (byte & 7u) * 4u, 4);
                int32_t s = wgd_q20_i8(wr + (size_t)bb * 18ull, xu, byte,
                                       &dw);
                acc += dw * xd[grp] * (float)s;
            }
        }
        acc = wgv_shfl_red(acc);
        Q4_WGV_SP_MERGE(q4_sm_a, acc)
        if (lane == 0 && q4_sp == 0u) atomicAdd(y + row, acc * wts[e]);
    }
}

__global__ static void moe_act_k(float *g, const float *u, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        g[i] = v / (1.f + expf(-v)) * u[i];
    }
}

/* Device-count variant: n = *n_d * n_ff — grid is sized for the worst case. */
__global__ static void moe_act_dk(float *g, const float *u, uint32_t n_ff,
                                  const int32_t *n_d) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)*n_d * n_ff) {
        float v = g[i];
        g[i] = v / (1.f + expf(-v)) * u[i];
    }
}

/* y[i] += sum_e w[e] * dd[e][i] — replaces n separate axpy launches. */
__global__ static void moe_wsum_k(float *y, const float *dd, const float *w,
                                  uint32_t ne, uint32_t dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float s = 0.f;
    for (uint32_t e = 0; e < ne; e++) s += w[e] * dd[(size_t)e * dim + i];
    y[i] += s;
}

__global__ static void moe_wsum_dk(float *y, const float *dd, const float *w,
                                   const int32_t *n_d, uint32_t dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    uint32_t ne = (uint32_t)*n_d;
    float s = 0.f;
    for (uint32_t e = 0; e < ne; e++) s += w[e] * dd[(size_t)e * dim + i];
    y[i] += s;
}

/* ---- multi-segment GEMV: up to 4 weight tensors sharing one x, one
 * launch (gdn qkv+gate+beta+alpha, shared-expert gate+up, ...). ---- */
struct seg4_p {
    const uint8_t *w[8];
    float *o[8];
    uint32_t rows[8];
    uint32_t ty[8];
    uint64_t rb[8];     /* per-segment row stride (q4_row_bytes) */
    uint32_t off[8];    /* cumulative row starts */
    float act[8];       /* per-seg scale+silu epilogue (0 = none) */
    uint32_t kq[8];     /* per-seg: kq16 dot (Q4_K/Q5_K/IQ4_XS) */
    uint32_t nseg;
};

__global__ static void gemv_m_k(seg4_p S, const float *x, uint32_t in_dim) {
    uint32_t g = blockIdx.x, s = 0;
    while (s + 1 < S.nseg && g >= S.off[s + 1]) s++;
    uint32_t row = g - S.off[s];
    const uint8_t *w = S.w[s];
    __shared__ float sh[64];
    float acc = 0.f;
    if (S.ty[s] == Q4_T_Q8_0) {
        uint32_t nb = in_dim / 32u;
        const uint8_t *wr = w + (size_t)row * nb * 34ull;
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
            __half dh; memcpy(&dh, wr + b * 34u, 2);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2);
            float dot = 0.f;
            for (uint32_t i = 0; i < 32; i++)
                dot += (float)qs[i] * x[b * 32u + i];
            acc += __half2float(dh) * dot;
        }
    } else { /* F32 row */
        const float *wr = (const float *)w + (size_t)row * in_dim;
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            acc += wr[i] * x[i];
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t t = blockDim.x >> 1; t > 0; t >>= 1) {
        if (threadIdx.x < t) sh[threadIdx.x] += sh[threadIdx.x + t];
        __syncthreads();
    }
    if (threadIdx.x == 0) S.o[s][row] = sh[0];
}

/* Warp-per-row variant of gemv_m_k: warp handles concatenated row g; the
 * segment search is warp-uniform.  Each segment dispatches to the same
 * wdot_* body its standalone wgv kernel runs, so results are bit-identical
 * to per-tensor gemv calls (same lane mapping, same add order). */
__global__ static void gemv_mw_k(seg4_p S, const float *x, uint32_t in_dim) {
    uint32_t g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t s = 0;
    while (s + 1 < S.nseg && g >= S.off[s + 1]) s++;
    uint32_t row = g - S.off[s];
    if (row >= S.rows[s]) return;
    const uint8_t *wr = S.w[s] + (size_t)row * S.rb[s];
    float acc;
    switch (S.ty[s]) {
    case Q4_T_Q8_0:    acc = wdot_q8_0(wr, x, in_dim, lane);     break;
    case Q4_T_IQ2_XXS: acc = wdot_iq2xxs_k(wr, x, in_dim, lane); break;
    case Q4_T_IQ2_XS:  acc = wdot_iq2xs_k(wr, x, in_dim, lane);  break;
    case Q4_T_IQ2_S:   acc = wdot_iq2_s_k(wr, x, in_dim, lane);  break;
    case Q4_T_IQ3_S:   acc = wdot_iq3_s_k(wr, x, in_dim, lane);  break;
    case Q4_T_IQ3_XXS: acc = wdot_iq3xxs_k(wr, x, in_dim, lane); break;
    case Q4_T_IQ4_XS:  acc = S.kq[s] ? wdot_kq16<KQ_IQ4XS>(wr, x, in_dim, lane)
                                     : wdot_iq4xs_k(wr, x, in_dim, lane);  break;
    case Q4_T_IQ4_NL:  acc = wdot_iq4nl(wr, x, in_dim, lane);    break;
    case Q4_T_Q4_0:    acc = wdot_q4_0_k(wr, x, in_dim, lane);   break;
    case Q4_T_Q5_0:    acc = wdot_q5_0_k(wr, x, in_dim, lane);   break;
    case Q4_T_Q2_0:    acc = wdot_q20(wr, x, in_dim, lane);      break;
    case Q4_T_Q6_K:    acc = wdot_q6k(wr, x, in_dim, lane);      break;
    case Q4_T_Q4_K:    acc = S.kq[s] ? wdot_kq16<KQ_Q4K>(wr, x, in_dim, lane)
                                     : wdot_q4k(wr, x, in_dim, lane);      break;
    case Q4_T_Q5_K:    acc = S.kq[s] ? wdot_kq16<KQ_Q5K>(wr, x, in_dim, lane)
                                     : wdot_q5k(wr, x, in_dim, lane);      break;
    case Q4_T_BF16:    acc = wdot_bf16(wr, x, in_dim, lane);     break;
    default:           acc = wdot_f32(wr, x, in_dim, lane);      break;
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) {
        float v = acc;
        const float a = S.act[s];
        if (a != 0.f) {
            v *= a;
            v = v / (1.f + expf(-v));
        }
        S.o[s][row] = v;
    }
}

/* Per-segment fast-path gate, mirroring the wgv dispatch conditions in
 * q4_hip_gemv_dd so a segment here computes exactly what a standalone
 * wgv_* call would. */
static bool gemv_mw_ty_ok(uint32_t ty, uint32_t in_dim, const float *x) {
    switch (ty) {
    case Q4_T_F32:  return true;   /* scalar tail covers any in_dim */
    case Q4_T_BF16: return true;   /* wdot_bf16 has pair/scalar fallbacks */
    case Q4_T_Q8_0:
    case Q4_T_Q4_0:
    case Q4_T_Q5_0:
    case Q4_T_IQ4_NL: return in_dim % 32u == 0;
    case Q4_T_Q2_0:  return in_dim % 64u == 0;
    case Q4_T_Q6_K:
    case Q4_T_Q4_K:
    case Q4_T_Q5_K:  return in_dim % 256u == 0 && ((uintptr_t)x & 15u) == 0;
    case Q4_T_IQ2_XXS:
    case Q4_T_IQ2_XS:
    case Q4_T_IQ2_S:
    case Q4_T_IQ3_S:
    case Q4_T_IQ3_XXS:
    case Q4_T_IQ4_XS: return in_dim % 256u == 0;
    default: return false;
    }
}

extern "C" bool q4_hip_gemv_m(const q4_seg_t *segs, uint32_t nseg,
                              const float *d_x, uint32_t in_dim) {
    if (!g_ok || !segs || !d_x || !nseg || nseg > 8 || !in_dim) return false;
    if (shapes < 0) shapes = getenv("Q4_SHAPES") != NULL;
    if (shapes) {
        bool dup = false;
        for (int i = 0; i < g_mix_n; i++) {
            if (g_mixes[i].nseg != nseg || g_mixes[i].in != in_dim) continue;
            bool same = true;
            for (uint32_t s = 0; s < nseg; s++)
                if (g_mixes[i].ty[s] != segs[s].ty ||
                    g_mixes[i].rows[s] != segs[s].rows) {
                    same = false;
                    break;
                }
            if (same) {
                g_mixes[i].n++;
                dup = true;
                break;
            }
        }
        if (!dup && g_mix_n < 64) {
            q4_segmix_t *m = &g_mixes[g_mix_n++];
            m->n = 1;
            m->in = in_dim;
            m->nseg = (uint32_t)nseg;
            for (uint32_t s = 0; s < nseg; s++) {
                m->ty[s] = segs[s].ty;
                m->rows[s] = segs[s].rows;
            }
        }
    }
    seg4_p P;
    memset(&P, 0, sizeof P);
    P.nseg = nseg;
    uint32_t tot = 0;
    bool mw = wgv_on(), basic = true;
    for (uint32_t i = 0; i < nseg; i++) {
        if (!segs[i].w || !segs[i].out || !segs[i].rows) return false;
        uint64_t rb = q4_row_bytes(segs[i].ty, in_dim);
        if (!rb) return false;
        if (!gemv_mw_ty_ok(segs[i].ty, in_dim, d_x)) mw = false;
        if (segs[i].ty != Q4_T_Q8_0 && segs[i].ty != Q4_T_F32)
            basic = false;
        if (segs[i].act != 0.f) basic = false;  /* act is warp-path only */
        P.w[i] = segs[i].w;
        P.o[i] = segs[i].out;
        P.rows[i] = segs[i].rows;
        P.ty[i] = segs[i].ty;
        P.rb[i] = rb;
        P.act[i] = segs[i].act;
        P.kq[i] = kq16_ok(segs[i].ty, segs[i].w, d_x, in_dim) ? 1u : 0u;
        P.off[i] = tot;
        tot += segs[i].rows;
    }
    if (mw) {
        if (gemv_mode() > 0)
            for (uint32_t i = 0; i < nseg; i++)
                if (segs[i].ty < 64) {
                    g_gemv_calls[g_gemv_phase][segs[i].ty]++;
                    g_gemv_bytes[g_gemv_phase][segs[i].ty] +=
                        (uint64_t)segs[i].rows * P.rb[i];
                }
        GEV_PRE(segs[0].ty);
        gemv_mw_k<<<dim3((tot + 7u) / 8u), 256, 0, g_str>>>(P, d_x, in_dim);
        GEV_POST();
    } else if (basic) {
        gemv_m_k<<<dim3(tot), 64, 0, g_str>>>(P, d_x, in_dim);
    } else {
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* Small elementwise fusions (each replaces 2 launches). */
__global__ static void mul_sig_k(float *y, const float *a, const float *g,
                                 uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / (1.f + expf(-g[i]));
}
extern "C" bool q4_hip_mul_sig(float *y, const float *a, const float *g,
                               uint64_t n) {
    if (!g_ok || !y || !a || !g || !n) return false;
    mul_sig_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(y, a, g, n);
    return hipGetLastError() == hipSuccess;
}
/* Fused mul_sig + mean_hc for the decode hc_mix tail: one node instead of
 * two, and skips materialising the gated intermediate.  Same expression
 * and same accumulation order as the two kernels it replaces. */
__global__ static void mulsig_mean_k(const float *a, const float *g,
                                     float *mixed, uint32_t n_embd,
                                     uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float s = 0.f;
    for (uint32_t c = 0; c < hc; c++) {
        size_t o = (size_t)c * n_embd + i;
        s += a[o] / (1.f + expf(-g[o]));
    }
    mixed[i] = s / (float)hc;
}
extern "C" bool q4_hip_mulsig_mean(const float *a, const float *g, float *mixed,
                                   uint32_t n_embd, uint32_t hc) {
    if (!g_ok || !a || !g || !mixed || !n_embd || !hc) return false;
    mulsig_mean_k<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(a, g, mixed,
                                                               n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

/* Batched mulsig_mean: token-major [n_tok][hc*n_embd] inputs, mixed is
 * [n_tok][n_embd]. One thread per (token, embd). */
__global__ static void mulsig_mean_n_k(const float *a, const float *g,
                                       float *mixed, uint32_t n_embd,
                                       uint32_t hc, uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= n_embd || t >= n_tok) return;
    const size_t base = (size_t)t * hc * n_embd + i;
    float s = 0.f;
    for (uint32_t c = 0; c < hc; c++) {
        const size_t o = base + (size_t)c * n_embd;
        s += a[o] / (1.f + expf(-g[o]));
    }
    mixed[(size_t)t * n_embd + i] = s / (float)hc;
}
/* hc up GEMV + mulsig_mean in one launch (Q4_HCUP_FUSE, default on): a
 * warp's four 8-lane row groups take the hc rows j, j + n_embd, ... of one
 * mixed column j, so the gate never leaves registers.  Each row's dot is
 * wgvn_bf16v_t<8>'s and the stream sum keeps mulsig_mean_n_k's order:
 * bit-identical to the two-kernel path. */
template <uint32_t NT>
__global__ static void hcup_mix_t(float *mixed, const uint8_t *w,
                                  const float *x, const float *a,
                                  uint32_t in_dim, uint32_t n_embd,
                                  float scale) {
    constexpr uint32_t G = 8, HC = 4;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (j >= n_embd) return;
    const uint32_t hcI = lane / G, gl = lane % G;
    const uint32_t row = hcI * n_embd + j;
    const uint4 *w4 = (const uint4 *)(w + (size_t)row * in_dim * 2u);
    const uint32_t n8 = in_dim >> 3;
    float acc[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    for (uint32_t i = gl; i < n8; i += G) {
        const uint4 p = w4[i];
        float a0, b0, a1, b1, a2, b2, a3, b3;
        bf16x2_f(p.x, &a0, &b0);
        bf16x2_f(p.y, &a1, &b1);
        bf16x2_f(p.z, &a2, &b2);
        bf16x2_f(p.w, &a3, &b3);
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            const float4 *x4 = (const float4 *)(x + (size_t)t * in_dim);
            const float4 xa = x4[2u * i], xb = x4[2u * i + 1u];
            acc[t] = fmaf(a0, xa.x, fmaf(b0, xa.y, acc[t]));
            acc[t] = fmaf(a1, xa.z, fmaf(b1, xa.w, acc[t]));
            acc[t] = fmaf(a2, xb.x, fmaf(b2, xb.y, acc[t]));
            acc[t] = fmaf(a3, xb.z, fmaf(b3, xb.w, acc[t]));
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        float v = acc[t];
        for (uint32_t o = G >> 1; o; o >>= 1)
            v += __shfl_xor_sync(0xffffffffull, v, o);
        v *= scale;
        float gv[HC];
#pragma unroll
        for (uint32_t c = 0; c < HC; c++) gv[c] = __shfl_sync(0xffffffffull, v, c * G);
        if (lane == 0) {
            const size_t base = (size_t)t * HC * n_embd + j;
            float s = 0.f;
#pragma unroll
            for (uint32_t c = 0; c < HC; c++) {
                const size_t o = base + (size_t)c * n_embd;
                s += a[o] / (1.f + expf(-gv[c]));
            }
            mixed[(size_t)t * n_embd + j] = s / (float)HC;
        }
    }
}

static int hcup_fuse_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_HCUP_FUSE");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* mixed = mean_c(a_c * sigmoid(W_up x)_c), BF16 W_up [hc * n_embd][in_dim];
 * false when the fused shape does not apply (caller: GEMV + mulsig). */
extern "C" bool q4_hip_hcup_mix_n(uint32_t ty, const uint8_t *d_w,
                                  const float *d_x, const float *d_a,
                                  float *d_mixed, uint32_t in_dim,
                                  uint32_t n_embd, uint32_t hc,
                                  uint32_t n_tok) {
    if (!g_ok || !hcup_fuse_on() || ty != Q4_T_BF16 || hc != 4 ||
        n_tok < 2 || n_tok > 8 || in_dim > 512u || (in_dim & 7u) ||
        ((uintptr_t)d_w & 15u) || ((uintptr_t)d_x & 15u) || !wgv_on() ||
        !wgvn_on())
        return false;
    const dim3 gg((n_embd * 32u + 255u) / 256u), bb(256);
    switch (n_tok) {
#define HCUP(N) case N: hcup_mix_t<N><<<gg, bb, 0, g_str>>>(d_mixed, d_w, d_x, d_a, in_dim, n_embd, 1.f); break;
    HCUP(2) HCUP(3) HCUP(4) HCUP(5) HCUP(6) HCUP(7) HCUP(8)
#undef HCUP
    default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_mulsig_mean_n(const float *a, const float *g,
                                     float *mixed, uint32_t n_embd,
                                     uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !a || !g || !mixed || !n_embd || !hc || !n_tok) return false;
    mulsig_mean_n_k<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, g_str>>>(
        a, g, mixed, n_embd, hc, n_tok);
    return hipGetLastError() == hipSuccess;
}

__global__ static void scale_silu_k(float *x, float s, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] * s;
        x[i] = v / (1.f + expf(-v));
    }
}
extern "C" bool q4_hip_scale_silu(float *x, float s, uint64_t n) {
    if (!g_ok || !x || !n) return false;
    scale_silu_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(x, s, n);
    return hipGetLastError() == hipSuccess;
}
__global__ static void silu_mul_k(float *g, const float *u, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        g[i] = v / (1.f + expf(-v)) * u[i];
    }
}
extern "C" bool q4_hip_silu_mul(float *g, const float *u, uint64_t n) {
    if (!g_ok || !g || !u || !n) return false;
    silu_mul_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(g, u, n);
    return hipGetLastError() == hipSuccess;
}
__global__ static void dot_sig_k(const float *a, const float *b, float *out,
                                 uint32_t n) {
    __shared__ float sh[256];
    float acc = 0.f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) acc += a[i] * b[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = 1.f / (1.f + expf(-sh[0]));
}
extern "C" bool q4_hip_dot_sigmoid(const float *a, const float *b, float *out,
                                   uint32_t n) {
    if (!g_ok || !a || !b || !out) return false;
    dot_sig_k<<<1, 256, 0, g_str>>>(a, b, out, n);
    return hipGetLastError() == hipSuccess;
}

static int32_t *g_es_d, *g_es_h; /* expert slot + weight staging (≤64 each) */
static float *g_ew_d, *g_ew_h;
static int32_t *g_en_d, *g_en_h; /* device-visible hit count for graphable exec */
static bool moe_stage_bufs(void) {
    if (!g_es_d) {
        /* Mapped host memory doubles as the device buffers: the exec kernels
         * read slot ids + weights over PCIe (≤512 B, L2-cached), which drops
         * two H2D copy nodes per MoE layer out of every decode graph. */
        if (hipHostMalloc((void **)&g_es_h, 64 * sizeof(int32_t),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostMalloc((void **)&g_ew_h, 64 * sizeof(float),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostMalloc((void **)&g_en_h, 64 * sizeof(int32_t),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostGetDevicePointer((void **)&g_es_d, g_es_h, 0) !=
                hipSuccess ||
            hipHostGetDevicePointer((void **)&g_ew_d, g_ew_h, 0) !=
                hipSuccess ||
            hipHostGetDevicePointer((void **)&g_en_d, g_en_h, 0) !=
                hipSuccess) {
            g_es_d = nullptr;
            return false;
        }
        *g_en_h = 0;
    }
    (void)xq8_bufs(); /* int8-activation staging — outside any capture */
    return true;
}

static bool moe_ty_ok(uint32_t ty, uint32_t in_dim) {
    switch (ty) {
    case Q4_T_IQ2_XXS:
    case Q4_T_IQ2_XS:
    case Q4_T_IQ2_S:
    case Q4_T_IQ3_S:
    case Q4_T_IQ3_XXS:
    case Q4_T_IQ4_XS:
        return in_dim % 256u == 0;
    case Q4_T_Q2_0:
        return in_dim % 64u == 0;
    case Q4_T_IQ4_NL:
    case Q4_T_Q4_0:
    case Q4_T_Q5_0:
        return in_dim % 32u == 0;
    default:
        return false;
    }
}

/* Warps per row for the routed-expert projections.  Decode's grids are
 * short — 4 hit experts x 640 rows is 320 blocks, 3.3 per CU over 96 CUs —
 * and the gemvbench occ sweep shows the warp-per-row kernels gaining 2-2.5x
 * past ~1024 blocks, so the row split tries to buy that by multiplying the
 * block count instead of the work.  Measured on the deployed GSQ-RCO IQ3_S
 * decode (decode-probe, 127 tok, interleaved): it LOSES, 44-47 t/s at
 * Q4_WGV_SPLIT=0 against 28-33 at 2 and 4.  The sweep's small-grid deficit
 * does not reproduce inside the graphed decode (where the replay already
 * keeps the machine fed and there is no drain to shorten), and four times
 * the warps pay four times the loop setup, the merge syncs and the warp
 * scheduling.  Off by default; the env knob stays for shapes where the
 * short grid is not hidden by a graph (and gemvbench mparity guards it).
 * Q4_WGV_SPLIT=2/4 force a split, 0 (the default) is the plain path. */
static inline uint32_t wgv_nsp(uint32_t out_dim, uint32_t gy) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("Q4_WGV_SPLIT");
        mode = e ? atoi(e) : 0;
        if (mode < 0) mode = 0;
        if (mode > 8) mode = 8;
    }
    if (mode == 0) return 1u;
    uint32_t rows = (out_dim + 7u) >> 3;
    if (rows * gy >= 1024u) return 1u;
    uint32_t want = mode == 1u ? 4u : (uint32_t)mode;
    for (uint32_t n = want; n > 1u; n >>= 1) {
        if (n > Q4_WGV_SP_MAX) continue;
        if (out_dim % (8u / n)) continue;
        static int logged = 0;
        if (!logged) {
            logged = 1;
            fprintf(stderr,
                    "q4: expert gemv row split x%u (out_dim %u gy %u, grid "
                    "%u -> %u blocks)\n",
                    n, out_dim, gy, rows * gy, rows * n * gy);
        }
        return n;
    }
    return 1u;
}

static bool moe_rows(uint32_t ty, const uint8_t *base, uint64_t stride,
                     const int32_t *slots, const float *x, uint32_t xs,
                     uint32_t in_dim, uint32_t out_dim, float *out, uint32_t ne,
                     const int32_t *n_d) {
    if (!in_dim || !out_dim || (!ne && !n_d)) return false;
    moe_count(ty, out_dim, in_dim, n_d ? 1u : ne, 1);
    if (wgv_on()) {
        dim3 wgrd((out_dim + 7u) / 8u, n_d ? WMOE_GY : ne), wblk(256);
        const uint32_t nsp = wgv_nsp(out_dim, n_d ? WMOE_GY : ne);
        const uint32_t q4_rpb = 8u / nsp;
        dim3 wgrd_sp((out_dim + q4_rpb - 1u) / q4_rpb, n_d ? WMOE_GY : ne);
        switch (ty) {
        case Q4_T_IQ2_XXS:
            if (in_dim % 256u) return false;
            wmoe_iq2xxs_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots,
                                                    x, xs, in_dim, out_dim,
                                                    n_d, nsp);
            break;
        case Q4_T_IQ2_XS:
            if (in_dim % 256u) return false;
            wmoe_iq2xs_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots,
                                                   x, xs, in_dim, out_dim,
                                                   n_d, nsp);
            break;
        case Q4_T_IQ2_S:
            if (in_dim % 256u) return false;
            wmoe_iq2_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                  xs, in_dim, out_dim, n_d, nsp);
            break;
        case Q4_T_IQ3_S:
            if (in_dim % 256u) return false;
            wmoe_iq3_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                  xs, in_dim, out_dim, n_d, nsp);
            break;
        case Q4_T_IQ3_XXS:
            if (in_dim % 256u) return false;
            wmoe_iq3xxs_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots,
                                                     x, xs, in_dim, out_dim,
                                                     n_d, nsp);
            break;
        case Q4_T_IQ4_XS:
            if (in_dim % 256u) return false;
            wmoe_iq4xs_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots,
                                                    x, xs, in_dim, out_dim,
                                                    n_d, nsp);
            break;
        case Q4_T_IQ4_NL:
            if (in_dim % 32u) return false;
            wmoe_iq4_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                  xs, in_dim, out_dim, n_d, nsp);
            break;
        case Q4_T_Q4_0:
            if (in_dim % 32u) return false;
            wmoe_q40_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                 xs, in_dim, out_dim, n_d, nsp);
            break;
        case Q4_T_Q5_0:
            if (in_dim % 32u) return false;
            wmoe_q50_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                 xs, in_dim, out_dim, n_d, nsp);
            break;
        case Q4_T_Q2_0:
            if (in_dim % 64u) return false;
            wmoe_q20_bk<<<wgrd_sp, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                 xs, in_dim, out_dim, n_d, nsp);
            break;
        default:
            return false;
        }
        return hipGetLastError() == hipSuccess;
    }
    if (n_d) return false;   /* legacy row-per-block kernels take host counts */
    dim3 grd(out_dim, ne), blk(64);
    switch (ty) {
    case Q4_T_IQ2_XS:
        if (in_dim % 256u) return false;
        moe_iq2xs_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                             in_dim, out_dim);
        break;
    case Q4_T_IQ2_S:
        if (in_dim % 256u) return false;
        moe_iq2_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_IQ3_S:
        if (in_dim % 256u) return false;
        moe_iq3_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_IQ3_XXS:
        if (in_dim % 256u) return false;
        moe_iq3xxs_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                              in_dim, out_dim);
        break;
    case Q4_T_IQ4_XS:
        if (in_dim % 256u) return false;
        moe_iq4xs_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                             in_dim, out_dim);
        break;
    case Q4_T_IQ4_NL:
        if (in_dim % 32u) return false;
        moe_iq4_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_Q4_0:
        if (in_dim % 32u) return false;
        moe_q40_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_Q5_0:
        if (in_dim % 32u) return false;
        moe_q50_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_Q2_0:
        if (in_dim % 64u) return false;
        moe_q20_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* ---- int8 activation staging for the *_q8 moe kernels ------------------
 * xq8_k quantizes each 32-float group to i8x32 + one fp32 scale, one warp
 * per group.  Buffers cover 64 expert slices of up to 16384 floats. */
#define XQ8_GPS_MAX 512u /* groups per slice: slen/32, slen <= 16384 */
#define XQ8_NS_MAX 64u
static int8_t *g_xq8;
static float *g_xq8d;
/* Tracks which x pointer g_xq8 slice-0 currently holds (host-side launch
 * bookkeeping only — stream ordering makes the device data consistent). */
static const float *g_xq8_src;
static bool xq8_bufs(void) {
    if (!g_xq8) {
        if (hipMalloc((void **)&g_xq8, XQ8_NS_MAX * XQ8_GPS_MAX * 32u) !=
                hipSuccess ||
            hipMalloc((void **)&g_xq8d, XQ8_NS_MAX * XQ8_GPS_MAX * 4u) !=
                hipSuccess) {
            g_xq8 = nullptr;
            return false;
        }
    }
    return true;
}

static int wq8_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_WGQ8") ? atoi(getenv("Q4_WGQ8")) : 1;
    return v;
}

static int wq8_iq4ya_on(void) {
    static int v = -1;
    if (v < 0)
        v = wq8_on() && getenv("Q4_WGQ8_IQ4YA") &&
            atoi(getenv("Q4_WGQ8_IQ4YA"));
    return v;
}

__global__ static void xq8_k(const float *x, uint32_t stride, int8_t *q8,
                             float *d8, uint32_t gps, const int32_t *n_d) {
    const uint32_t g = blockIdx.x, lane = threadIdx.x & 31u;
    const uint32_t e = g / gps, gl = g - e * gps;
    if (n_d && e >= (uint32_t)*n_d) return;
    const float *xs = x + (size_t)e * stride + gl * 32u;
    const float v = xs[lane];
    float amax = fabsf(v);
    for (int o = 16; o; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffull, amax, o));
    q8[(size_t)g * 32 + lane] = (int8_t)__float2int_rn(v * (amax ? 127.f / amax : 0.f));
    if (!lane) d8[g] = amax * (1.f / 127.f);
}

/* Quantize `ns` x-slices of `slen` floats (slice e at x + e*xs) on g_str.
 * n_d != NULL: device count bounds the slices actually written. */
static bool xq8_launch(const float *x, uint32_t xs, uint32_t slen,
                       uint32_t ns, const int32_t *n_d) {
    if (!slen || (slen & 31u) || (slen >> 5) > XQ8_GPS_MAX) return false;
    if (!xq8_bufs()) return false;
    const uint32_t gps = slen >> 5;
    const uint32_t ng = (n_d ? XQ8_NS_MAX : (xs ? ns : 1u)) * gps;
    xq8_k<<<ng, 32, 0, g_str>>>(x, xs, g_xq8, g_xq8d, gps, n_d);
    /* Multi-slice / strided launches overwrite slice 0 — drop the
     * pack-fused provenance tag. */
    g_xq8_src = (xs == 0 && !n_d && ns <= 1u) ? x : NULL;
    return hipGetLastError() == hipSuccess;
}

/* Fused gate+up: one launch when gt == ut (the common case). */
static bool moe_rows_gu(uint32_t ty, const uint8_t *bg, const uint8_t *bu,
                        uint64_t stride, const int32_t *slots, const float *x,
                        uint32_t xs, uint32_t in_dim, uint32_t out_dim,
                        float *og, float *ou, uint32_t ne,
                        const int32_t *n_d, uint32_t act) {
    if (!wgv_on() || !in_dim || !out_dim || (!ne && !n_d)) return false;
    moe_count(ty, out_dim, in_dim, n_d ? 1u : ne, 2);
    dim3 wgrd((out_dim + 7u) / 8u, n_d ? WMOE_GY : ne), wblk(256);
    const uint32_t nsp = wgv_nsp(out_dim, n_d ? WMOE_GY : ne);
    const uint32_t q4_rpb = 8u / nsp;
    dim3 wgrd_sp((out_dim + q4_rpb - 1u) / q4_rpb, n_d ? WMOE_GY : ne);
    /* xs==0 means all experts share one activation — quantize it once into
     * g_xq8 and take the i8-dot kernels. */
    bool q8ok = wq8_on() && xs == 0 && (in_dim % 256u) == 0 &&
                (ty == Q4_T_IQ2_XXS || ty == Q4_T_IQ2_XS ||
                 ty == Q4_T_IQ2_S || ty == Q4_T_IQ3_S ||
                 ty == Q4_T_IQ3_XXS || ty == Q4_T_IQ4_XS);
    if (q8ok) {
        if (g_xq8_src == x) {
            g_xq8_src = NULL;      /* consume-once: later x reuse must
                                    * re-quantize, never read stale data */
        } else {
            q8ok = xq8_launch(x, 0, in_dim, 1, NULL);
        }
    }
    switch (ty) {
    case Q4_T_IQ2_XXS:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq2xxs_gu_q8<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d, act, nsp);
            break;
        }
        wmoe_iq2xxs_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                                                   slots, x, xs, in_dim,
                                                   out_dim, n_d, act, nsp);
        break;
    case Q4_T_IQ2_XS:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq2xs_gu_q8<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d, act, nsp);
            break;
        }
        wmoe_iq2xs_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                                                  slots, x, xs, in_dim,
                                                  out_dim, n_d, act, nsp);
        break;
    case Q4_T_IQ2_S:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq2_gu_q8<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d, act, nsp);
            break;
        }
        wmoe_iq2_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                 x, xs, in_dim, out_dim, n_d, act, nsp);
        break;
    case Q4_T_IQ3_S:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq3_gu_q8<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d, act, nsp);
            break;
        }
        wmoe_iq3_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                 x, xs, in_dim, out_dim, n_d, act, nsp);
        break;
    case Q4_T_IQ3_XXS:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq3xxs_gu_q8<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d, act, nsp);
            break;
        }
        wmoe_iq3xxs_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                                                    slots, x, xs, in_dim,
                                                    out_dim, n_d, act, nsp);
        break;
    case Q4_T_IQ4_XS:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq4xs_gu_q8<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d, act, nsp);
            break;
        }
        wmoe_iq4xs_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                                                   slots, x, xs, in_dim,
                                                   out_dim, n_d, act, nsp);
        break;
    case Q4_T_IQ4_NL:
        if (in_dim % 32u) return false;
        wmoe_iq4_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                 x, xs, in_dim, out_dim, n_d, act, nsp);
        break;
    case Q4_T_Q4_0:
        if (in_dim % 32u) return false;
        wmoe_q40_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                x, xs, in_dim, out_dim, n_d, act, nsp);
        break;
    case Q4_T_Q5_0:
        if (in_dim % 32u) return false;
        wmoe_q50_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                x, xs, in_dim, out_dim, n_d, act, nsp);
        break;
    case Q4_T_Q2_0:
        if (in_dim % 64u) return false;
        wmoe_q20_gu_bk<<<wgrd_sp, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                x, xs, in_dim, out_dim, n_d, act, nsp);
        break;
    default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* Down proj + fused weighted accumulate into y (atomic epilogue). */
static bool moe_rows_ya(uint32_t ty, const uint8_t *base, uint64_t stride,
                        const int32_t *slots, const float *wts,
                        const float *x, uint32_t xs, uint32_t in_dim,
                        uint32_t out_dim, float *y, uint32_t ne,
                        const int32_t *n_d) {
    if (!wgv_on() || !in_dim || !out_dim || (!ne && !n_d) || !wts)
        return false;
    moe_count(ty, out_dim, in_dim, n_d ? 1u : ne, 1);
    dim3 wgrd((out_dim + 7u) / 8u, n_d ? WMOE_GY : ne), wblk(256);
    const uint32_t nsp = wgv_nsp(out_dim, n_d ? WMOE_GY : ne);
    const uint32_t q4_rpb = 8u / nsp;
    dim3 wgrd_sp((out_dim + q4_rpb - 1u) / q4_rpb, n_d ? WMOE_GY : ne);
    switch (ty) {
    case Q4_T_IQ2_XXS:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq2xxs_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_iq2xxs_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
            wts, x, xs, in_dim, out_dim, n_d, nsp);
        break;
    case Q4_T_IQ2_XS:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq2xs_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_iq2xs_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
            wts, x, xs, in_dim, out_dim, n_d, nsp);
        break;
    case Q4_T_IQ2_S:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq2_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_iq2_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                 x, xs, in_dim, out_dim, n_d, nsp);
        break;
    case Q4_T_IQ3_S:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq3_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_iq3_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                 x, xs, in_dim, out_dim, n_d, nsp);
        break;
    case Q4_T_IQ3_XXS:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq3xxs_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_iq3xxs_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                                                    wts, x, xs, in_dim,
                                                    out_dim, n_d, nsp);
        break;
    case Q4_T_IQ4_XS:
        if (in_dim % 256u) return false;
        /* Same lightweight-LUT class as iq4_nl: bandwidth-bound down rows,
         * marginal isolated win — keep float unless opted in. */
        if (wq8_iq4ya_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq4xs_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_iq4xs_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                   x, xs, in_dim, out_dim,
                                                   n_d, nsp);
        break;
    case Q4_T_IQ4_NL:
        if (in_dim % 32u) return false;
        /* Measured slower in-model (+10% kernel time) despite winning in the
         * isolated bench: at n_ff=640 the row is only 20 blocks, so the
         * extra xd8/xq8 dependent loads sit on the acc chain while weight
         * loads from L1 slots already bound the loop.  Keep float. */
        if (wq8_iq4ya_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq4_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_iq4_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                 x, xs, in_dim, out_dim, n_d, nsp);
        break;
    case Q4_T_Q4_0:
        if (in_dim % 32u) return false;
        wmoe_q40_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                x, xs, in_dim, out_dim, n_d, nsp);
        break;
    case Q4_T_Q5_0:
        if (in_dim % 32u) return false;
        wmoe_q50_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                x, xs, in_dim, out_dim, n_d, nsp);
        break;
    case Q4_T_Q2_0:
        if (in_dim % 64u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_q20_ya_q8<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d, nsp);
            break;
        }
        wmoe_q20_ya_bk<<<wgrd_sp, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                x, xs, in_dim, out_dim, n_d, nsp);
        break;
    default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* All n hit-experts in three launches when gt == ut (fused gate+up,
 * silu*mul, down with atomic weighted accumulate into d_y); falls back to
 * the five-launch chain for mixed types. Caller guarantees slots' data
 * already landed (upload watermark or copy-stream drain). */
extern "C" bool q4_hip_moe_exec(const uint8_t *base, uint64_t stride,
                                const int32_t *slots, const float *wts,
                                uint32_t n, uint32_t gt, uint64_t og,
                                uint32_t ut, uint64_t ou, uint32_t dt,
                                uint64_t od, const float *d_x, float *d_h,
                                float *d_h2, float *d_down, float *d_y,
                                uint32_t n_ff, uint32_t in_dim,
                                uint32_t n_embd) {
    if (!g_ok || !base || !slots || !wts || !n || n > 64) return false;
    /* Reject unsupported types BEFORE any launch so the caller's fallback
     * never double-accumulates work we already queued. */
    if (!moe_ty_ok(gt, in_dim) || !moe_ty_ok(ut, in_dim) ||
        !moe_ty_ok(dt, n_ff))
        return false;
    if (!moe_stage_bufs()) return false;
    memcpy(g_es_h, slots, n * sizeof(int32_t));
    memcpy(g_ew_h, wts, n * sizeof(float));
    /* slots/wts live in mapped host memory; the kernels below fetch them
     * over PCIe themselves — no copy nodes on the stream. */
    bool act_done = false;
    if (gt == ut &&
        moe_rows_gu(gt, base + og, base + ou, stride, g_es_d, d_x, 0, in_dim,
                    n_ff, d_h, d_h2, n, NULL, 1)) {
        act_done = true;
    } else if (!moe_rows(gt, base + og, stride, g_es_d, d_x, 0, in_dim, n_ff,
                         d_h, n, NULL) ||
               !moe_rows(ut, base + ou, stride, g_es_d, d_x, 0, in_dim, n_ff,
                         d_h2, n, NULL)) {
        return false;
    }
    uint64_t na = (uint64_t)n * n_ff;
    if (!act_done)
        moe_act_k<<<dim3((uint32_t)((na + 255) / 256)), 256, 0, g_str>>>(
            d_h, d_h2, na);
    /* All types were vetted above, so a ya failure here is a launch error —
     * do NOT fall back or the down contribution would be applied twice. */
    if (wgv_on())
        return moe_rows_ya(dt, base + od, stride, g_es_d, g_ew_d, d_h, n_ff,
                           n_ff, n_embd, d_y, n, NULL);
    if (!moe_rows(dt, base + od, stride, g_es_d, d_h, n_ff, n_ff, n_embd,
                  d_down, n, NULL))
        return false;
    moe_wsum_k<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(d_y, d_down,
                                                            g_ew_d, n, n_embd);
    return hipGetLastError() == hipSuccess;
}

/* Resident Q2_0 chain. Slots and weights are already device memory written
 * earlier on g_str, so this records cleanly inside a capture. ne is the
 * host count: the grid y is ne and n_d stays null. */
extern "C" bool q4_hip_moe_q2(const uint8_t *base, uint64_t stride,
                              const int32_t *d_slots, const float *d_wts,
                              uint32_t ne, uint64_t og, uint64_t ou,
                              uint64_t od, const float *d_x, float *d_h,
                              float *d_h2, float *d_down, float *d_y,
                              uint32_t n_ff, uint32_t in_dim,
                              uint32_t n_embd) {
    if (!g_ok || !base || !d_slots || !d_wts || !ne || ne > 64 || !d_x ||
        !d_h || !d_y)
        return false;
    if (!moe_ty_ok(Q4_T_Q2_0, in_dim) || !moe_ty_ok(Q4_T_Q2_0, n_ff))
        return false;
    if (wgv_on()) {
        if (!moe_rows_gu(Q4_T_Q2_0, base + og, base + ou, stride, d_slots, d_x,
                         0, in_dim, n_ff, d_h, d_h2, ne, NULL, 1))
            return false;
        return moe_rows_ya(Q4_T_Q2_0, base + od, stride, d_slots, d_wts, d_h,
                           n_ff, n_ff, n_embd, d_y, ne, NULL);
    }
    if (!d_h2 || !d_down) return false;
    if (!moe_rows(Q4_T_Q2_0, base + og, stride, d_slots, d_x, 0, in_dim, n_ff,
                  d_h, ne, NULL) ||
        !moe_rows(Q4_T_Q2_0, base + ou, stride, d_slots, d_x, 0, in_dim, n_ff,
                  d_h2, ne, NULL))
        return false;
    uint64_t na = (uint64_t)ne * n_ff;
    moe_act_k<<<dim3((uint32_t)((na + 255) / 256)), 256, 0, g_str>>>(d_h, d_h2,
                                                                     na);
    if (!moe_rows(Q4_T_Q2_0, base + od, stride, d_slots, d_h, n_ff, n_ff,
                  n_embd, d_down, ne, NULL))
        return false;
    moe_wsum_k<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(d_y, d_down,
                                                             d_wts, ne,
                                                             n_embd);
    return hipGetLastError() == hipSuccess;
}

/* ---- batched routed MoE over a resident Q2_0 pack ----
 * One launch per projection covers every (expert, token-row) pair of a
 * batch. slots[i] is pair i's expert, tok[i] the token row it belongs to,
 * wts[i] its routing weight: tok[] indexes d_x on the way in and receives
 * the weighted result on the way out, so a 64-row prefill group is two
 * launches instead of ~90 per token. Same warp-per-row geometry and the
 * same lane/byte mapping as wmoe_q20_gu_bk / wmoe_q20_ya_bk — only the row
 * indirection through tok[] is new. The pairs stay token-major (no
 * compaction step and no host round-trip); the L2 absorbs the repeated
 * expert rows. */
__global__ static void wmoe_q20_gu_tok_bk(float *og, const uint8_t *bg,
                                          const uint8_t *bu, uint64_t stride,
                                          const int32_t *slots,
                                          const int32_t *tok, const float *x,
                                          uint32_t xs, uint32_t in_dim,
                                          uint32_t out_dim) {
    const uint32_t nb = in_dim >> 6;
    Q4_WGV_ROW();
    const uint32_t e = blockIdx.y;
    const uint8_t *wg = bg + (size_t)slots[e] * stride +
                        (size_t)row * nb * 18ull;
    const uint8_t *wu = bu + (size_t)slots[e] * stride +
                        (size_t)row * nb * 18ull;
    const float *xt = x + (size_t)tok[e] * xs;
    const uint32_t b = lane >> 4, byte = lane & 15u;
    float ag = 0.f, au = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
        uint32_t bb = b0 + b;
        if (bb < nb) {
            ag += wgd_q20(wg + (size_t)bb * 18ull,
                          xt + bb * 64u + byte * 4u, byte);
            au += wgd_q20(wu + (size_t)bb * 18ull,
                          xt + bb * 64u + byte * 4u, byte);
        }
    }
    ag = wgv_shfl_red(ag);
    au = wgv_shfl_red(au);
    if (lane == 0)
        og[(size_t)e * out_dim + row] = ag / (1.f + expf(-ag)) * au;
}

__global__ static void wmoe_q20_ya_tok_bk(float *y, const uint8_t *base,
                                          uint64_t stride, const int32_t *slots,
                                          const int32_t *tok, const float *wts,
                                          const float *x, uint32_t xs,
                                          uint32_t in_dim, uint32_t out_dim) {
    const uint32_t nb = in_dim >> 6;
    Q4_WGV_ROW();
    const uint32_t e = blockIdx.y;
    const uint8_t *wr = base + (size_t)slots[e] * stride +
                        (size_t)row * nb * 18ull;
    const float *xt = x + (size_t)e * xs;
    const uint32_t b = lane >> 4, byte = lane & 15u;
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
        uint32_t bb = b0 + b;
        if (bb < nb)
            acc += wgd_q20(wr + (size_t)bb * 18ull,
                           xt + bb * 64u + byte * 4u, byte);
    }
    acc = wgv_shfl_red(acc);
    /* k pairs per token land on the same rows: plain atomicAdd, the same
     * contention the per-token ya kernel has. */
    if (lane == 0) atomicAdd(y + (size_t)tok[e] * out_dim + row, acc * wts[e]);
}

/* og/ou/od are the byte offsets of the gate / up / down slices inside one
 * expert blob. d_h must hold nr * n_ff floats. nr is the total number of
 * (expert, token-row) pairs, at most gridDim.y limits. */
extern "C" bool q4_hip_moe_q2_win(const uint8_t *base, uint64_t stride,
                                  const int32_t *d_slots, const int32_t *d_tok,
                                  const float *d_wts, uint32_t nr, uint64_t og,
                                  uint64_t ou, uint64_t od, const float *d_x,
                                  float *d_h, float *d_y, uint32_t n_ff,
                                  uint32_t in_dim, uint32_t n_embd) {
    if (!g_ok || !base || !d_slots || !d_tok || !d_wts || !nr || !d_x ||
        !d_h || !d_y)
        return false;
    if (!wgv_on()) return false;
    if (in_dim % 64u || n_ff % 64u) return false;
    dim3 blk(256);
    dim3 ggrd((n_ff + 7u) / 8u, nr), ygrd((n_embd + 7u) / 8u, nr);
    wmoe_q20_gu_tok_bk<<<ggrd, blk, 0, g_str>>>(d_h, base + og, base + ou,
                                               stride, d_slots, d_tok, d_x,
                                               in_dim, in_dim, n_ff);
    if (hipGetLastError() != hipSuccess) return false;
    wmoe_q20_ya_tok_bk<<<ygrd, blk, 0, g_str>>>(d_y, base + od, stride, d_slots,
                                               d_tok, d_wts, d_h, n_ff, n_ff,
                                               n_embd);
    return hipGetLastError() == hipSuccess;
}

/* Preflight for the per-token small-batch exec: same type/shape rule as
 * q4_hip_moe_exec so callers can decide the fast path before any launch. */
extern "C" bool q4_hip_moe_tok_ok(uint32_t gt, uint32_t ut, uint32_t dt,
                                  uint32_t in_dim, uint32_t n_ff) {
    return g_ok && wgv_on() && moe_ty_ok(gt, in_dim) &&
           moe_ty_ok(ut, in_dim) && moe_ty_ok(dt, n_ff);
}

/* Small-batch MoE hits (2..8 tokens, MTP verify / short prefill): slots
 * and weights live in DEVICE arrays laid out token-major with h_off
 * prefix offsets, so back-to-back calls never touch the mapped staging
 * arrays q4_hip_moe_exec uses (those would race with queued kernels).
 * Per token: fused gate+up rows into d_h/d_h2 at the token's hit offset,
 * one fused silu*mul over the whole hit count, then down rows with the
 * fused weighted atomicAdd into that token's d_y. */
extern "C" bool q4_hip_moe_exec_tok(const uint8_t *base, uint64_t stride,
                                      const int32_t *d_slots,
                                      const float *d_wts,
                                      const uint32_t *h_off, uint32_t n_tok,
                                      uint32_t gt, uint64_t og, uint32_t ut,
                                      uint64_t ou, uint32_t dt, uint64_t od,
                                      const float *d_x, float *d_h,
                                      float *d_h2, float *d_y, uint32_t n_ff,
                                      uint32_t in_dim, uint32_t n_embd) {
    if (!g_ok || !base || !d_slots || !d_wts || !h_off || !n_tok)
        return false;
    /* Preflight everything before the first launch: a false return means
     * nothing was queued and the caller may take its old path. */
    if (!wgv_on() || !moe_ty_ok(gt, in_dim) || !moe_ty_ok(ut, in_dim) ||
        !moe_ty_ok(dt, n_ff))
        return false;
    uint32_t tot = h_off[n_tok];
    if (!tot) return true;
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t cnt = h_off[t + 1] - h_off[t];
        if (!cnt) continue;
        if (gt != ut ||
            !moe_rows_gu(gt, base + og, base + ou, stride,
                         d_slots + h_off[t], d_x + (size_t)t * in_dim, 0,
                         in_dim, n_ff, d_h + (size_t)h_off[t] * n_ff,
                         d_h2 + (size_t)h_off[t] * n_ff, cnt, NULL, 1))
            return false;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t cnt = h_off[t + 1] - h_off[t];
        if (!cnt) continue;
        if (!moe_rows_ya(dt, base + od, stride, d_slots + h_off[t],
                         d_wts + h_off[t], d_h + (size_t)h_off[t] * n_ff,
                         n_ff, n_ff, n_embd, d_y + (size_t)t * n_embd, cnt,
                         NULL))
            return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* ---- device-side routing (graph-resident MoE layer) --------------------
 * route_dev_k runs the router softmax+top-k entirely on the GPU, resolves
 * each pick to an L1 slot through the device-visible residency table
 * (published by the copy stream), splits into GPU-hit / CPU-miss lists and
 * publishes misses to a pinned mailbox consumed by the host dispatcher.
 * One block; thread 0 does the scan — 512 experts × insertion-sort top-k is
 * only a few µs and stays bit-identical to the host q4_topk order. */
/* Device-side staging mailbox.  The kernel writes this layout in device
 * memory and returns — the publish is then carried to the pinned host
 * mailbox by an in-stream D2H memcpy (SDMA).  Previously the kernel wrote
 * host memory directly, and the dispatch completion had to drain the PCIe
 * posted writes — ~60us of hidden wall per call under load (the device's
 * own realtime counter shows ~9us of real work vs ~72us of trace
 * duration).  seq sits LAST: the ascending-address copy lands it after
 * the payload, so seq remains the host's "publish complete" flag and the
 * torn-read recheck stays meaningful.  The host-only fields (done/err)
 * live past the copied prefix. */
typedef struct {
    int32_t ids[64];                   /* ids+wts contiguous: warp-burst fill */
    float wts[64];
    uint32_t n;                        /* miss count */
    /* Batched publish (MTP verify): one mailbox entry carries every
     * token's CPU-miss segment.  ntok==0 means a legacy single-token
     * publish (the miss list is ids[0..n)); ntok>1 segments ids/wts by
     * tn[t]. */
    uint32_t ntok;
    uint32_t tn[8];
    /* Batched publish only: token t's L1 hit slots hs[t*16 .. +hn[t]), so
     * the host can refresh their recency (GPU hits never pass the host
     * otherwise).  The single-token kernels leave hn zero. */
    int32_t hs[8 * 16];
    uint32_t hn[8];
    /* Batched publish only: each miss's rank in its token's top-k (0 =
     * the heaviest pick), for the host's Q4_TAIL_SKIP. */
    uint8_t rk[64];
    /* Fire-and-forget routing profile channel: every call publishes the
     * full top-k list + an rseq stamp.  The host may drop entries (it's a
     * histogram, not a protocol) and never stamps anything back, so the
     * GPU never waits on it.  Routed reads are torn-tolerant by design. */
    volatile int32_t routed[16];
    unsigned long long rseq;
    unsigned long long seq;            /* LAST — doorbell; arrives last */
} q4_mail_t;

/* Warp-cooperative router: lanes split the 512-wide max/exp scans, probs
 * land in shared memory, then top-k is extracted by packed-key warp max
 * reductions ((float_bits<<32)|~idx — descending value, ascending index on
 * ties, bit-identical ordering vs q4_topk's serial insertion).  Lane 0
 * keeps the serial renormalizing sum so the add order is unchanged. */
__global__ static void route_dev_k(
        const float *d_rlog, const int32_t *devtab,
        int32_t *d_es, float *d_ew, int32_t *d_nhit,
        const int32_t *restab,
        int32_t *d_res_s, float *d_res_w, int32_t *d_res_n,
        uint32_t res_cap,
        q4_mail_t *mail, unsigned long long *d_seq,
        unsigned long long *d_rseq,
        unsigned long long *d_prof,
        uint32_t n_exp, uint32_t topk) {
    __shared__ float pr[1024];
    __shared__ unsigned long long s_pay[64]; /* ids[64] then wts[64] */
    const uint32_t lane = threadIdx.x;
    /* clock64() on gfx11 is a ~2^20-cycle shader counter that wraps inside
     * a long kernel section — deltas come out negative and wrap the u64
     * accumulator.  s_sendmsg_rtnl(131)=MSG_RTN_GET_REALTIME returns the
     * fixed-rate (100 MHz) global realtime counter, 64 bits; the 32-bit
     * s_sendmsg_rtn wraps every ~43 s. */
    unsigned long long t0 = 0, t1 = 0, t2 = 0;
    __asm__ volatile("" ::: "memory");
    t0 = __builtin_amdgcn_s_sendmsg_rtnl(131);
    __asm__ volatile("" ::: "memory");
    float mx = -INFINITY;
    for (uint32_t i = lane; i < n_exp; i += 32u) mx = fmaxf(mx, d_rlog[i]);
    for (int o = 16; o; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, o));
    float ps = 0.f;
    for (uint32_t i = lane; i < n_exp; i += 32u) {
        float e = expf(d_rlog[i] - mx);
        pr[i] = e;
        ps += e;
    }
    for (int o = 16; o; o >>= 1)
        ps += __shfl_xor_sync(0xffffffffull, ps, o);
    const float inv = ps > 0.f ? 1.f / ps : 0.f;
    for (uint32_t i = lane; i < n_exp; i += 32u) pr[i] *= inv;
    __asm__ volatile("" ::: "memory");
    t1 = __builtin_amdgcn_s_sendmsg_rtnl(131);
    __asm__ volatile("" ::: "memory");
    __shared__ uint32_t s_nm;
    /* Warp top-k by extraction: each round warp-max-reduces a packed key
     * (float_bits<<32 | ~idx) — descending value, ties to the lowest index,
     * exactly the serial scan's ordering — and the winner's owner lane
     * clears its pr[] slot.  A single lane's 512-step serial insertion
     * measured ~60us of stream time here; this is ~1-2us.  NaN inputs are
     * the one ordering divergence vs the host: pr[i] > -INF drops them
     * here while the host scan would insert NaN into an empty slot —
     * unreachable for real router outputs. */
    int32_t *sids = (int32_t *)s_pay;
    float *swts = (float *)(s_pay + 32);
    for (uint32_t k = 0; k < topk; k++) {
        unsigned long long best = 0;
        for (uint32_t i = lane; i < n_exp; i += 32u) {
            const float v = pr[i];
            if (v > -INFINITY) {
                const unsigned long long key =
                    ((unsigned long long)__float_as_uint(v) << 32) |
                    (unsigned long long)(~i);
                if (key > best) best = key;
            }
        }
        for (int o = 16; o; o >>= 1) {
            const unsigned long long t =
                __shfl_xor_sync(0xffffffffull, best, o);
            if (t > best) best = t;
        }
        if (lane == 0) {
            if (best) {
                const uint32_t wi = ~(uint32_t)(best & 0xffffffffull);
                sids[k] = (int32_t)wi;
                swts[k] = __uint_as_float((uint32_t)(best >> 32));
            } else {
                sids[k] = -1;
                swts[k] = -INFINITY;
            }
        }
        if (best) {
            const uint32_t wi = ~(uint32_t)(best & 0xffffffffull);
            if (lane == (wi & 31u)) pr[wi] = -INFINITY;
        }
        __syncwarp();
    }
    /* Renormalizing sum stays on lane 0 — its add order is bit-identical
     * to the serial version. */
    __shared__ float s_rinv;
    if (lane == 0) {
        float s = 0.f;
        for (uint32_t i = 0; i < topk; i++) s += swts[i];
        s_rinv = s > 0.f ? 1.f / s : 0.f;
    }
    __syncwarp();
    /* Warp-parallel hit/miss split: one ballot round per 32 picks (topk
     * can reach 64 — a single warp mask would silently drop picks 32..63).
     * Within each half lane k owns pick base+k; cumulative offsets carry
     * the compacted positions so both output lists keep top-k order.
     * Misses compact into the low end of the same s_pay ids/wts the
     * publish burst copies out. */
    {
        /* The miss compaction below rewrites s_pay in place — for
         * topk > 32 that clobbers the not-yet-read picks 32..63 (mwts
         * aliases sids[32+]), so each lane registers its pick(s) up
         * front before any shared writes. */
        int32_t pk_id[2];
        float pk_w[2];
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t pk = lane + 32u * h;
            pk_id[h] = (pk < topk) ? sids[pk] : -1;
            pk_w[h] = (pk < topk) ? swts[pk] * s_rinv : 0.f;
            if (d_rseq && pk < topk && pk < 16) mail->routed[pk] = pk_id[h];
        }
        __syncwarp();
        uint32_t nh = 0, nr = 0, nm = 0;
        for (uint32_t base = 0; base < topk; base += 32u) {
            const uint32_t pk = base + lane;
            const uint32_t h = base >> 5;
            const uint32_t km =
                (topk - base < 32u) ? ((1u << (topk - base)) - 1u)
                                    : 0xffffffffu;
            const int32_t tid = pk_id[h];
            const float wq = pk_w[h];
            const int32_t slot = (tid >= 0) ? devtab[tid] : -1;
            const uint32_t hitm = __ballot_sync(0xffffffffull, slot >= 0);
            const uint32_t hits = hitm & km;
            const uint32_t miss = ~hitm & km;
            /* L1 miss consults the static resident table: a nonzero entry
             * means the expert's bytes sit in the host-registered image
             * and the GPU reads them over PCIe — only non-resident misses
             * go to the CPU mailbox.  Ballot keeps top-k order within each
             * list. */
            const int32_t rk =
                (slot < 0 && tid >= 0 && restab) ? restab[tid] : 0;
            const uint32_t resm = __ballot_sync(0xffffffffull, rk > 0) & km;
            const uint32_t rr = nr + __popc(resm & ((1u << lane) - 1u));
            /* Partial split: only the first res_cap resident misses
             * (highest weights first, cumulative across halves) take the
             * PCIe path — the rest stay on the mailbox so the two
             * bandwidth pools overlap instead of serializing. */
            const int32_t take = (rk > 0 && rr < res_cap);
            const uint32_t res_t = __ballot_sync(0xffffffffull, take) & km;
            const uint32_t mm = miss & ~res_t;
            const uint32_t hr = nh + __popc(hits & ((1u << lane) - 1u));
            const uint32_t mr = nm + __popc(mm & ((1u << lane) - 1u));
            __syncwarp(); /* sids/swts captured before s_pay compaction */
            if (pk < topk) {
                if (slot >= 0) {
                    d_es[hr] = slot;
                    d_ew[hr] = wq;
                } else if (take) {
                    d_res_s[rr] = rk - 1;
                    d_res_w[rr] = wq;
                } else {
                    int32_t *mids = (int32_t *)s_pay;
                    float *mwts = (float *)(s_pay + 32);
                    mids[mr] = tid;
                    mwts[mr] = wq;
                }
            }
            nh += __popc(hits);
            nr += __popc(res_t);
            nm += __popc(mm);
            __syncwarp();
        }
        if (lane == 0) {
            *d_nhit = (int32_t)nh;
            if (d_res_n) *d_res_n = (int32_t)nr;
            s_nm = nm;
        }
    }
    __asm__ volatile("" ::: "memory");
    t2 = __builtin_amdgcn_s_sendmsg_rtnl(131);
    __asm__ volatile("" ::: "memory");
    __syncwarp();
    /* Stage the publish in device memory: the caller's in-stream D2H copy
     * carries it to the pinned host mailbox with payload-before-seq
     * ordering, so no fences (and no PCIe traffic) live in this kernel at
     * all.  The stamp still comes from the device-side per-layer counter
     * so stamping never needs a host round trip. */
    if (s_nm) {
        /* Element-typed copy: the miss payload was stored as int32/float
         * halves of s_pay, so a u64 publish load could be hoisted above
         * those stores (strict aliasing).  Same-type reads can't. */
        int32_t *did = mail->ids;
        float *dwt = mail->wts;
        const int32_t *mids = (const int32_t *)s_pay;
        const float *mwts = (const float *)(s_pay + 32);
        for (uint32_t i = lane; i < 64; i += 32u) {
            did[i] = mids[i];
            dwt[i] = mwts[i];
        }
        if (lane == 0) {
            mail->n = s_nm;
            mail->ntok = 0; /* legacy single-token publish */
        }
        __syncwarp();
        if (lane == 0) mail->seq = ++(*d_seq);
    }
    /* Profile stamp: published even when no CPU misses so the host
     * histogram sees hit-only tokens too.  No wait protocol — the
     * dispatcher just counts what it sees before the next bump. */
    if (d_rseq) {
        __syncwarp();
        if (lane == 0) mail->rseq = ++(*d_rseq);
    }
    if (lane == 0 && d_prof) {
        __asm__ volatile("" ::: "memory");
        const unsigned long long t3 = __builtin_amdgcn_s_sendmsg_rtnl(131);
        d_prof[0] += t3 - t0;
        d_prof[1] += 1;
        d_prof[2] += t1 - t0; /* softmax scan */
        d_prof[3] += t2 - t1; /* serial top-k + hit/miss */
        d_prof[4] += t3 - t2; /* staging publish */
    }
}

/* Batched variant for MTP verify / small-batch MoE: one block walks the
 * n_tok router rows, runs the same warp-scan + packed-key extraction +
 * ballot hit/miss split per token, and publishes ALL tokens' misses in
 * one mailbox entry.
 * Hits land in fixed-stride per-token segments (d_es + t*64) with the
 * per-token hit count in d_nh[t]; misses are the flat ids/wts payload
 * segmented by mail->tn[t].  Host gates n_tok*topk <= 64 so the shared
 * payload never overflows. */
__global__ static void route_dev_n_k(
        const float *d_rlog, const int32_t *devtab,
        int32_t *d_es, float *d_ew, int32_t *d_nh,
        q4_mail_t *mail, unsigned long long *d_seq,
        unsigned long long *d_rseq,
        unsigned long long *d_prof,
        uint32_t n_exp, uint32_t topk, uint32_t n_tok) {
    __shared__ float pr[1024];
    __shared__ unsigned long long s_pay[64]; /* ids[64] then wts[64] */
    __shared__ int32_t s_routed[16];
    __shared__ uint32_t s_tn[8];
    __shared__ uint32_t s_nm;
    __shared__ int32_t s_tid[16];
    __shared__ float s_twt[16];
    __shared__ float s_rinv;
    const uint32_t lane = threadIdx.x;
    const unsigned long long t0 = __builtin_amdgcn_s_sendmsg_rtnl(131);
    if (lane == 0) {
        s_nm = 0;
        for (uint32_t i = 0; i < 8; i++) s_tn[i] = 0;
    }
    __syncwarp();
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *rl = d_rlog + (size_t)t * n_exp;
        float mx = -INFINITY;
        for (uint32_t i = lane; i < n_exp; i += 32u)
            mx = fmaxf(mx, rl[i]);
        for (int o = 16; o; o >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, o));
        float ps = 0.f;
        for (uint32_t i = lane; i < n_exp; i += 32u) {
            float e = expf(rl[i] - mx);
            pr[i] = e;
            ps += e;
        }
        for (int o = 16; o; o >>= 1)
            ps += __shfl_xor_sync(0xffffffffull, ps, o);
        const float inv = ps > 0.f ? 1.f / ps : 0.f;
        for (uint32_t i = lane; i < n_exp; i += 32u) pr[i] *= inv;
        __syncwarp();
        /* Warp top-k by extraction — packed (float_bits<<32 | ~idx) max
         * reduce gives descending value, lowest-index tiebreak, identical
         * to the serial insertion order it replaces.  The old per-token
         * register arrays spilled to local scratch on every dynamic
         * index; s_tid/s_twt are shared scratch instead.  NaN logits are
         * dropped here vs inserted-into-empty-slot in q4_topk — corrupt
         * input only. */
        const uint32_t tk = topk > 16 ? 16 : topk;
        for (uint32_t k = 0; k < tk; k++) {
            unsigned long long best = 0;
            for (uint32_t i = lane; i < n_exp; i += 32u) {
                const float v = pr[i];
                if (v > -INFINITY) {
                    const unsigned long long key =
                        ((unsigned long long)__float_as_uint(v) << 32) |
                        (unsigned long long)(~i);
                    if (key > best) best = key;
                }
            }
            for (int o = 16; o; o >>= 1) {
                const unsigned long long t2 =
                    __shfl_xor_sync(0xffffffffull, best, o);
                if (t2 > best) best = t2;
            }
            if (lane == 0) {
                if (best) {
                    const uint32_t wi =
                        ~(uint32_t)(best & 0xffffffffull);
                    s_tid[k] = (int32_t)wi;
                    s_twt[k] = __uint_as_float((uint32_t)(best >> 32));
                } else {
                    s_tid[k] = -1;
                    s_twt[k] = -INFINITY;
                }
            }
            if (best) {
                const uint32_t wi =
                    ~(uint32_t)(best & 0xffffffffull);
                if (lane == (wi & 31u)) pr[wi] = -INFINITY;
            }
            __syncwarp();
        }
        /* Renormalizing sum stays on lane 0 — its add order is bit-identical
         * to the serial version. */
        if (lane == 0) {
            float s = 0.f;
            for (uint32_t i = 0; i < tk; i++) s += s_twt[i];
            s_rinv = s > 0.f ? 1.f / s : 0.f;
        }
        __syncwarp();
        /* Same ballot+popc hit/miss split as route_dev_k: lane k owns pick
         * k, so both output lists keep top-k order; misses append at the
         * running base so the flat payload stays token-segmented. */
        {
            const uint32_t km =
                (tk < 32u) ? ((1u << tk) - 1u) : 0xffffffffu;
            const int32_t tid = (lane < tk) ? s_tid[lane] : -1;
            const float wq = (lane < tk) ? s_twt[lane] * s_rinv : 0.f;
            if (t == 0 && d_rseq && lane < tk && lane < 16)
                s_routed[lane] = tid;
            const int32_t slot = (tid >= 0) ? devtab[tid] : -1;
            const uint32_t hitm = __ballot_sync(0xffffffffull, slot >= 0);
            const uint32_t hits = hitm & km;
            const uint32_t miss = ~hitm & km;
            const uint32_t hr = __popc(hits & ((1u << lane) - 1u));
            const uint32_t mr = __popc(miss & ((1u << lane) - 1u));
            const uint32_t base = s_nm;
            if (lane < tk) {
                int32_t *mids = (int32_t *)s_pay;
                float *mwts = (float *)(s_pay + 32);
                if (slot >= 0) {
                    d_es[t * 64 + hr] = slot;
                    d_ew[t * 64 + hr] = wq;
                } else {
                    mids[base + mr] = tid;
                    mwts[base + mr] = wq;
                    mail->rk[base + mr] = (uint8_t)lane;
                }
            }
            if (lane == 0) {
                d_nh[t] = (int32_t)__popc(hits);
                s_tn[t] = __popc(miss);
                s_nm += __popc(miss);
            }
        }
        __syncwarp();
    }
    /* Publish unconditionally to the device staging slot: the verify-side
     * in-stream wait keys on seq for EVERY layer call, and the caller's
     * D2H copy delivers payload-before-seq ordering.  No PCIe writes or
     * system fences in-kernel. */
    if (lane == 0) {
        for (uint32_t i = 0; i < 8; i++) mail->tn[i] = s_tn[i];
        for (uint32_t i = 0; i < 8; i++) mail->hn[i] = 0u;
        mail->ntok = n_tok;
        mail->n = s_nm;
    }
    __syncwarp();
    if (s_nm) {
        /* Same-type element copy (see route_dev_k): a u64 burst could be
         * reordered above the int32/float miss stores. */
        int32_t *did = mail->ids;
        float *dwt = mail->wts;
        const int32_t *mids = (const int32_t *)s_pay;
        const float *mwts = (const float *)(s_pay + 32);
        for (uint32_t i = lane; i < 64; i += 32u) {
            did[i] = mids[i];
            dwt[i] = mwts[i];
        }
    }
    __syncwarp();
    if (lane == 0) mail->seq = ++(*d_seq);
    if (d_rseq) {
        __syncwarp();
        if (lane == 0) {
            for (uint32_t k = 0; k < 16; k++) mail->routed[k] = s_routed[k];
            mail->rseq = ++(*d_rseq);
        }
    }
    if (lane == 0 && d_prof) {
        __asm__ volatile("" ::: "memory");
        const unsigned long long t3 = __builtin_amdgcn_s_sendmsg_rtnl(131);
        d_prof[0] += t3 - t0;
        d_prof[1] += 1;
    }
}

/* route_dev_n_k walked the n_tok router rows with one warp, ~14 us per
 * token (rocprof 2026-10-12: 55 us per verify layer, 2.7 ms per cycle) —
 * and the CPU-miss pool cannot start until it has published.  This runs
 * one warp per token with the router row in registers (n_exp <= 1024:
 * 32 values per lane), the same per-lane sum order, shuffle pattern,
 * packed-key extraction and lane-0 renormalize, so every token's picks
 * and weights are bit-identical; warp 0 then lays the misses out in token
 * order exactly as the serial walk did.  Q4_ROUTE_NPAR=0 keeps the
 * one-warp kernel. */
__global__ static void route_dev_np_k(
        const float *d_rlog, const int32_t *devtab,
        int32_t *d_es, float *d_ew, int32_t *d_nh,
        q4_mail_t *mail, unsigned long long *d_seq,
        unsigned long long *d_rseq, unsigned long long *d_prof,
        uint32_t n_exp, uint32_t topk, uint32_t n_tok) {
    const unsigned long long t0 = __builtin_amdgcn_s_sendmsg_rtnl(131);
    __shared__ int32_t s_mid[8][16];
    __shared__ float s_mwt[8][16];
    __shared__ uint8_t s_mrk[8][16];
    __shared__ uint32_t s_tn[8];
    __shared__ int32_t s_routed[16];
    const uint32_t lane = threadIdx.x & 31u, t = threadIdx.x >> 5;
    const uint32_t tk = topk > 16 ? 16 : topk;
    if (t < n_tok) {
        const float *rl = d_rlog + (size_t)t * n_exp;
        float v[32];
        float mx = -INFINITY;
#pragma unroll
        for (uint32_t j = 0; j < 32; j++) {
            const uint32_t i = lane + 32u * j;
            v[j] = i < n_exp ? rl[i] : -INFINITY;
            if (i < n_exp) mx = fmaxf(mx, v[j]);
        }
        for (int o = 16; o; o >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, o));
        float ps = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 32; j++) {
            if (lane + 32u * j < n_exp) {
                const float e = expf(v[j] - mx);
                v[j] = e;
                ps += e;
            }
        }
        for (int o = 16; o; o >>= 1)
            ps += __shfl_xor_sync(0xffffffffull, ps, o);
        const float inv = ps > 0.f ? 1.f / ps : 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 32; j++)
            if (lane + 32u * j < n_exp) v[j] *= inv;
        int32_t my_id = -1;   /* lane k keeps pick k */
        float my_wt = -INFINITY;
        for (uint32_t k = 0; k < tk; k++) {
            unsigned long long best = 0;
#pragma unroll
            for (uint32_t j = 0; j < 32; j++) {
                const uint32_t i = lane + 32u * j;
                if (i < n_exp && v[j] > -INFINITY) {
                    const unsigned long long key =
                        ((unsigned long long)__float_as_uint(v[j]) << 32) |
                        (unsigned long long)(~i);
                    if (key > best) best = key;
                }
            }
            for (int o = 16; o; o >>= 1) {
                const unsigned long long t2 =
                    __shfl_xor_sync(0xffffffffull, best, o);
                if (t2 > best) best = t2;
            }
            if (lane == k) {
                if (best) {
                    my_id = (int32_t)~(uint32_t)(best & 0xffffffffull);
                    my_wt = __uint_as_float((uint32_t)(best >> 32));
                } else {
                    my_id = -1;
                    my_wt = -INFINITY;
                }
            }
            if (best) {
                const uint32_t wi = ~(uint32_t)(best & 0xffffffffull);
                if (lane == (wi & 31u)) {
#pragma unroll
                    for (uint32_t j = 0; j < 32; j++)
                        if (j == (wi >> 5)) v[j] = -INFINITY;
                }
            }
        }
        /* lane-0 renormalize in pick order, like the serial version */
        float s = 0.f;
        for (uint32_t i = 0; i < tk; i++)
            s += __shfl_sync(0xffffffffull, my_wt, i);
        const float rinv = s > 0.f ? 1.f / s : 0.f;
        const uint32_t km = (tk < 32u) ? ((1u << tk) - 1u) : 0xffffffffu;
        const int32_t tid = (lane < tk) ? my_id : -1;
        const float wq = (lane < tk) ? my_wt * rinv : 0.f;
        if (t == 0 && d_rseq && lane < tk && lane < 16) s_routed[lane] = tid;
        const int32_t slot = (tid >= 0) ? devtab[tid] : -1;
        const uint32_t hitm = __ballot_sync(0xffffffffull, slot >= 0);
        const uint32_t hits = hitm & km;
        const uint32_t miss = ~hitm & km;
        const uint32_t hr = __popc(hits & ((1u << lane) - 1u));
        const uint32_t mr = __popc(miss & ((1u << lane) - 1u));
        if (lane < tk) {
            if (slot >= 0) {
                d_es[t * 64 + hr] = slot;
                d_ew[t * 64 + hr] = wq;
                mail->hs[t * 16 + hr] = slot;
            } else {
                s_mid[t][mr] = tid;
                s_mwt[t][mr] = wq;
                s_mrk[t][mr] = (uint8_t)lane;
            }
        }
        if (lane == 0) {
            d_nh[t] = (int32_t)__popc(hits);
            s_tn[t] = __popc(miss);
        }
    }
    __syncthreads();
    if (t != 0) return;
    uint32_t nm = 0;
    for (uint32_t u = 0; u < n_tok; u++) {
        const uint32_t c = s_tn[u];
        if (lane < c) {
            mail->ids[nm + lane] = s_mid[u][lane];
            mail->wts[nm + lane] = s_mwt[u][lane];
            mail->rk[nm + lane] = s_mrk[u][lane];
        }
        nm += c;
    }
    if (lane == 0) {
        for (uint32_t i = 0; i < 8; i++) mail->tn[i] = i < n_tok ? s_tn[i] : 0u;
        for (uint32_t i = 0; i < 8; i++)
            mail->hn[i] = i < n_tok ? tk - s_tn[i] : 0u;
        mail->ntok = n_tok;
        mail->n = nm;
    }
    __syncwarp();
    if (lane == 0) mail->seq = ++(*d_seq);
    if (d_rseq) {
        __syncwarp();
        if (lane == 0) {
            for (uint32_t k = 0; k < 16; k++)
                mail->routed[k] = k < tk ? s_routed[k] : 0;
            mail->rseq = ++(*d_rseq);
        }
    }
    if (lane == 0 && d_prof) {
        __asm__ volatile("" ::: "memory");
        const unsigned long long t3 = __builtin_amdgcn_s_sendmsg_rtnl(131);
        d_prof[0] += t3 - t0;
        d_prof[1] += 1;
    }
}

static int g_route_npar = -1;
static int route_npar_on(void) {
    if (g_route_npar < 0) {
        const char *e = getenv("Q4_ROUTE_NPAR");
        g_route_npar = !(e && e[0] == '0');
    }
    return g_route_npar;
}

/* Test hook: 0 = the one-warp kernel, 1 = warp per token. */
extern "C" void q4_hip_route_npar_set(int on) { g_route_npar = on ? 1 : 0; }

extern "C" bool q4_hip_route_dev_n(const float *d_rlog, const int32_t *devtab,
                                   int32_t *d_es, float *d_ew, int32_t *d_nh,
                                   void *mail, unsigned long long *d_seq,
                                   unsigned long long *d_rseq,
                                   unsigned long long *d_prof,
                                   uint32_t n_exp, uint32_t topk,
                                   uint32_t n_tok) {
    if (!g_ok || !d_rlog || !devtab || !d_es || !d_ew || !d_nh || !mail ||
        !d_seq || n_exp > 1024 || !n_tok || n_tok > 8 || !topk ||
        topk > 16 || n_tok * topk > 64)
        return false;
    if (route_npar_on())
        route_dev_np_k<<<dim3(1), dim3(32 * n_tok), 0, g_str>>>(
            d_rlog, devtab, d_es, d_ew, d_nh, (q4_mail_t *)mail, d_seq,
            d_rseq, d_prof, n_exp, topk, n_tok);
    else
        route_dev_n_k<<<dim3(1), dim3(32), 0, g_str>>>(
            d_rlog, devtab, d_es, d_ew, d_nh, (q4_mail_t *)mail, d_seq,
            d_rseq, d_prof, n_exp, topk, n_tok);
    return hipGetLastError() == hipSuccess;
}

/* Decode prefetch prediction: layer T's router logits computed on layer
 * T-1's MoE input (one warp per token: softmax), each expert scored by its
 * best probability over the tokens; the experts T's residency table does
 * not hold, at score >= pmin, are ranked (count of better-scored ones, ties
 * by id) and the best kmax go out with the publishing layer's seq. */
struct q4_pred_t {
    unsigned long long seq;    /* per-layer record counter */
    unsigned long long tseq;   /* target layer's route seq when predicted */
    int32_t n;
    int32_t pad;
    int32_t ids[16];
    float sc[16];
};
__global__ static void __launch_bounds__(512)
pred_sel_k(const float *d_plog, const int32_t *devtab,
           const unsigned long long *d_tseq, unsigned long long *d_cnt,
           q4_pred_t *out, uint32_t n_exp, uint32_t n_tok, uint32_t kmax,
           float pmin) {
    __shared__ float s_mx[8], s_inv[8];
    __shared__ float c_sc[512];
    __shared__ int32_t c_id[512];
    __shared__ int32_t s_n;
    const uint32_t w = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t e = threadIdx.x;
    if (threadIdx.x == 0) s_n = 0;
    /* warp t < n_tok: token t's softmax max and 1/sum */
    if (w < n_tok) {
        const float *rl = d_plog + (size_t)w * n_exp;
        float mx = -INFINITY;
        for (uint32_t i = lane; i < n_exp; i += 32u) mx = fmaxf(mx, rl[i]);
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
        float sum = 0.f;
        for (uint32_t i = lane; i < n_exp; i += 32u) sum += expf(rl[i] - mx);
        for (int o = 16; o; o >>= 1) sum += __shfl_xor(sum, o, 32);
        if (lane == 0) {
            s_mx[w] = mx;
            s_inv[w] = 1.f / sum;
        }
    }
    __syncthreads();
    /* thread e: best probability over the tokens; candidates (not in L1,
     * >= pmin) compact into c_* in any order */
    if (e < n_exp && devtab[e] < 0) {
        float b = 0.f;
        for (uint32_t t = 0; t < n_tok; t++)
            b = fmaxf(b, expf(d_plog[(size_t)t * n_exp + e] - s_mx[t]) *
                             s_inv[t]);
        if (b >= pmin) {
            const int32_t k = atomicAdd(&s_n, 1);
            c_sc[k] = b;
            c_id[k] = (int32_t)e;
        }
    }
    __syncthreads();
    const int32_t m = s_n;
    /* rank within the candidates: better score first, ties by id */
    if ((int32_t)threadIdx.x < m) {
        const float v = c_sc[threadIdx.x];
        const int32_t id = c_id[threadIdx.x];
        uint32_t r = 0;
        for (int32_t j = 0; j < m; j++) {
            const float u = c_sc[j];
            r += (u > v) || (u == v && c_id[j] < id);
        }
        if (r < kmax) {
            out->ids[r] = id;
            out->sc[r] = v;
        }
    }
    if (threadIdx.x == 0) {
        out->n = m < (int32_t)kmax ? m : (int32_t)kmax;
        out->tseq = *d_tseq;
        out->seq = ++(*d_cnt);
    }
}

extern "C" size_t q4_hip_pred_size(void) { return sizeof(q4_pred_t); }

extern "C" bool q4_hip_pred_sel(const float *d_plog, const int32_t *devtab,
                                const unsigned long long *d_tseq,
                                unsigned long long *d_cnt, void *stg,
                                uint32_t n_exp, uint32_t n_tok, uint32_t kmax,
                                float pmin) {
    if (!g_ok || !d_plog || !devtab || !d_tseq || !d_cnt || !stg ||
        n_exp > 512 || !n_tok || n_tok > 8 || !kmax || kmax > 16)
        return false;
    pred_sel_k<<<dim3(1), dim3(512), 0, g_str>>>(
        d_plog, devtab, d_tseq, d_cnt, (q4_pred_t *)stg, n_exp, n_tok, kmax,
        pmin);
    return hipGetLastError() == hipSuccess;
}

/* Prefill router: one warp per token, same softmax + packed-key top-k
 * extraction + lane-0 renormalize as route_dev_k, writing the raw
 * (id, renormalized-weight) lists.  Replaces the n_tok*512 logit D2H and
 * the per-token host softmax/topk loops in q4_moe_layer_dev_n.  n_exp is
 * caller-gated to <=512 (static shared prob table), topk <= 64. */
__global__ static void route_pf_k(const float *d_rlog, int32_t *d_ids,
                                  float *d_wts, uint32_t n_tok,
                                  uint32_t n_exp, uint32_t topk) {
    __shared__ float pr[4][512];
    __shared__ int32_t s_tid[4][64];
    __shared__ float s_twt[4][64];
    __shared__ float s_rinv[4];
    __shared__ float s_pinv[4];
    const uint32_t w = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t t = blockIdx.x * 4 + w;
    if (t >= n_tok) return;
    float *prw = pr[w];
    const float *rl = d_rlog + (size_t)t * n_exp;
    float mx = -INFINITY;
    for (uint32_t i = lane; i < n_exp; i += 32u) mx = fmaxf(mx, rl[i]);
    for (int o = 16; o; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, o));
    for (uint32_t i = lane; i < n_exp; i += 32u)
        prw[i] = expf(rl[i] - mx);
    __syncwarp();
    /* Serial in-order sum: the host q4_softmax sums x[0..n) sequentially,
     * and bitwise-equal probs keep topk boundary picks identical. */
    if (lane == 0) {
        float ps = 0.f;
        for (uint32_t i = 0; i < n_exp; i++) ps += prw[i];
        s_pinv[w] = ps > 0.f ? 1.f / ps : 0.f;
    }
    __syncwarp();
    const float inv = s_pinv[w];
    for (uint32_t i = lane; i < n_exp; i += 32u) prw[i] *= inv;
    __syncwarp();
    const uint32_t tk = topk > 64 ? 64 : topk;
    for (uint32_t k = 0; k < tk; k++) {
        unsigned long long best = 0;
        for (uint32_t i = lane; i < n_exp; i += 32u) {
            /* Pick order on RAW logits: softmax is monotonic so this ranks
             * identically to the host's prob-order topk, and it is immune
             * to device-vs-host expf last-ulp differences.  prw serves as
             * the aliveness mask (cleared entries lose to the 0 key).  The
             * sign-flip maps float order onto uint order including the
             * negatives a raw logit can take (probs were always >= 0).
             * Accepted divergence: when two probs are bitwise equal but
             * the logits differ, the host's prob+index key picks the lower
             * index while we pick the higher logit — a semantically fine
             * but different pick (Q4_PF_ROUTE_CHECK counts these as nset;
             * measured 0 across 48 layers x 512 tok on the target model).
             * NaN logits diverge too: dropped here (prw[i] > -INF fails),
             * while q4_topk would insert one into an empty slot — only
             * reachable on corrupt router input. */
            if (prw[i] > -INFINITY) {
                const uint32_t b = __float_as_uint(rl[i]);
                const uint32_t key32 =
                    (b & 0x80000000u) ? ~b : (b | 0x80000000u);
                const unsigned long long key =
                    ((unsigned long long)key32 << 32) |
                    (unsigned long long)(~i);
                if (key > best) best = key;
            }
        }
        for (int o = 16; o; o >>= 1) {
            const unsigned long long t2 =
                __shfl_xor_sync(0xffffffffull, best, o);
            if (t2 > best) best = t2;
        }
        if (best) {
            const uint32_t wi = ~(uint32_t)(best & 0xffffffffull);
            if (lane == 0) {
                s_tid[w][k] = (int32_t)wi;
                s_twt[w][k] = prw[wi];
            }
            __syncwarp();
            if (lane == (wi & 31u)) prw[wi] = -INFINITY;
        } else if (lane == 0) {
            s_tid[w][k] = -1;
            s_twt[w][k] = -INFINITY;
        }
        __syncwarp();
    }
    if (lane == 0) {
        float s = 0.f;
        for (uint32_t i = 0; i < tk; i++) s += s_twt[w][i];
        s_rinv[w] = s > 0.f ? 1.f / s : 0.f;
    }
    __syncwarp();
    /* Strided writeback: s_tid/s_twt hold 64 entries but a warp has 32
     * lanes — lane<tk alone would leave picks 32..63 uninitialized. */
    for (uint32_t i = lane; i < tk; i += 32u) {
        d_ids[(size_t)t * topk + i] = s_tid[w][i];
        d_wts[(size_t)t * topk + i] = s_twt[w][i] * s_rinv[w];
    }
}

/* Prefill assignment: one block per <=gsz-token group.  Phase 1 histograms
 * picks via shared-memory atomics; a serial thread-0 pass then produces the
 * ebase prefix and the used-expert list; phase 2 writes the assignment
 * rows (a_tok/rsrc/rw) with atomic per-expert cursors.  Row order within
 * an expert is atomic order rather than scan order — downstream GEMM rows
 * are independent so outputs are bit-identical either way.  d_ek/d_stg/
 * d_stgn are read back by the host for staging + meta build. */
__global__ static void assign_pf_k(
        const int32_t *d_ids, const float *d_wts,
        int32_t *d_atok, int32_t *d_rsrc, float *d_rw,
        int32_t *d_stg, uint32_t *d_stgn, uint32_t *d_ek,
        uint32_t gsz, uint32_t topk, uint32_t n_exp, uint32_t n_tok) {
    __shared__ uint32_t sh_ek[512];
    __shared__ uint32_t sh_cur[512];
    __shared__ uint32_t sh_stg[512];
    __shared__ uint32_t sh_nstg;
    const uint32_t g = blockIdx.x;
    const uint32_t t0 = g * gsz;
    const uint32_t t1 = (t0 + gsz < n_tok) ? t0 + gsz : n_tok;
    const uint32_t tid = threadIdx.x;
    for (uint32_t i = tid; i < n_exp; i += blockDim.x) sh_ek[i] = 0;
    if (tid == 0) sh_nstg = 0;
    __syncthreads();
    const uint32_t seg = g * gsz * topk;
    const uint32_t rows = (t1 - t0) * topk;
    for (uint32_t i = tid; i < rows; i += blockDim.x) {
        const uint32_t t = t0 + i / topk, k = i % topk;
        const int32_t e = d_ids[(size_t)t * topk + k];
        if (e >= 0 && (uint32_t)e < n_exp)
            atomicAdd(&sh_ek[e], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t b = 0;
        for (uint32_t e = 0; e < n_exp; e++) {
            const uint32_t n = sh_ek[e];
            sh_cur[e] = b;
            if (n) {
                b += n;
                sh_stg[sh_nstg++] = e;
            }
        }
        d_stgn[g] = sh_nstg;
    }
    __syncthreads();
    for (uint32_t i = tid; i < rows; i += blockDim.x) {
        const uint32_t t = t0 + i / topk, k = i % topk;
        const int32_t e = d_ids[(size_t)t * topk + k];
        if (e >= 0 && (uint32_t)e < n_exp) {
            const uint32_t p = atomicAdd(&sh_cur[e], 1u);
            d_atok[seg + p] = (int32_t)t;
            d_rsrc[(size_t)t * topk + k] = (int32_t)p;
            d_rw[(size_t)t * topk + k] = d_wts[(size_t)t * topk + k];
        } else {
            d_rsrc[(size_t)t * topk + k] = -1;
            d_rw[(size_t)t * topk + k] = 0.f;
        }
    }
    __syncthreads();
    for (uint32_t i = tid; i < n_exp; i += blockDim.x)
        d_ek[g * 512 + i] = sh_ek[i];
    for (uint32_t i = tid; i < sh_nstg; i += blockDim.x)
        d_stg[g * 512 + i] = (int32_t)sh_stg[i];
}

extern "C" bool q4_hip_route_pf(const float *d_rlog, int32_t *d_ids,
                                float *d_wts, uint32_t n_tok,
                                uint32_t n_exp, uint32_t topk) {
    if (!g_ok || !d_rlog || !d_ids || !d_wts || !n_tok || n_exp > 512u ||
        !topk || topk > 64u)
        return false;
    route_pf_k<<<dim3((n_tok + 3u) / 4u), dim3(128), 0, g_str>>>(
        d_rlog, d_ids, d_wts, n_tok, n_exp, topk);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_assign_pf(
        const int32_t *d_ids, const float *d_wts,
        int32_t *d_atok, int32_t *d_rsrc, float *d_rw,
        int32_t *d_stg, uint32_t *d_stgn, uint32_t *d_ek,
        uint32_t gsz, uint32_t topk, uint32_t n_exp, uint32_t n_tok) {
    if (!g_ok || !d_ids || !d_wts || !d_atok || !d_rsrc || !d_rw || !d_stg ||
        !d_stgn || !d_ek || !n_tok || !gsz || !topk || n_exp > 512u ||
        topk > 64u)
        return false;
    const uint32_t n_g = (n_tok + gsz - 1u) / gsz;
    assign_pf_k<<<dim3(n_g), dim3(256), 0, g_str>>>(
        d_ids, d_wts, d_atok, d_rsrc, d_rw, d_stg, d_stgn, d_ek, gsz, topk,
        n_exp, n_tok);
    return hipGetLastError() == hipSuccess;
}

/* Device-array MoE exec: slots/wts/count all live in device memory written
 * by route_dev_k earlier in the same stream — graphable, zero host staging. */
extern "C" bool q4_hip_moe_exec_dev(const uint8_t *base, uint64_t stride,
                                    const int32_t *d_es, const float *d_ew,
                                    const int32_t *d_nhit, uint32_t gt,
                                    uint64_t og, uint32_t ut, uint64_t ou,
                                    uint32_t dt, uint64_t od,
                                    const float *d_x, float *d_h, float *d_h2,
                                    float *d_down, float *d_y, uint32_t n_ff,
                                    uint32_t in_dim, uint32_t n_embd) {
    if (!g_ok || !base || !d_es || !d_ew || !d_nhit) return false;
    if (!moe_ty_ok(gt, in_dim) || !moe_ty_ok(ut, in_dim) ||
        !moe_ty_ok(dt, n_ff))
        return false;
    bool act_done = false;
    if (gt == ut &&
        moe_rows_gu(gt, base + og, base + ou, stride, d_es, d_x, 0, in_dim,
                    n_ff, d_h, d_h2, 0, d_nhit, 1)) {
        act_done = true;
    } else if (!moe_rows(gt, base + og, stride, d_es, d_x, 0, in_dim, n_ff,
                         d_h, 0, d_nhit) ||
               !moe_rows(ut, base + ou, stride, d_es, d_x, 0, in_dim, n_ff,
                         d_h2, 0, d_nhit)) {
        return false;
    }
    if (!act_done)
        moe_act_dk<<<dim3((uint32_t)((64ull * n_ff + 255) / 256)), 256, 0,
                   g_str>>>(d_h, d_h2, n_ff, d_nhit);
    /* Types vetted above — a ya failure is a launch error; never fall back
     * (the down contribution would land twice). */
    if (wgv_on())
        return moe_rows_ya(dt, base + od, stride, d_es, d_ew, d_h, n_ff, n_ff,
                           n_embd, d_y, 0, d_nhit);
    if (!moe_rows(dt, base + od, stride, d_es, d_h, n_ff, n_ff, n_embd,
                  d_down, 0, d_nhit))
        return false;
    moe_wsum_dk<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(d_y, d_down,
                                                             d_ew, d_nhit,
                                                             n_embd);
    return hipGetLastError() == hipSuccess;
}

/* Verify window (2..8 tokens).  route_dev_n_k leaves hits in per-token
 * stride-64 lists.  A pack kernel unions those into unique experts.
 * blockIdx.y is one expert (grid y is the 64-slot cap); each warp stages
 * that expert's weight row once and re-dots every token that selected it.
 * Gate/up stay on the int8-activation octs.
 * IQ4_NL down stays float; Q2_0 down requantizes the hidden rows.  Other
 * types return win_ok false and the caller keeps the per-token exec. */
struct q4_win_plan {
    int32_t nuniq;
    int32_t nhit;
    int32_t slot[64];
    uint32_t tcnt[64];
    uint32_t tok[512]; /* pair index = expert * 8 + ti; value is the token */
    uint32_t hrow[512];
    float wt[512];
};

static q4_win_plan *g_win;
static int g_win_dead;

__device__ static void win_ld(uint8_t *dst, const uint8_t *src, uint32_t n,
                              uint32_t lane) {
    const uint32_t n4 = n >> 2;
    for (uint32_t i = lane; i < n4; i += 32u) {
        uint32_t v;
        memcpy(&v, src + (size_t)i * 4u, 4);
        memcpy(dst + (size_t)i * 4u, &v, 4);
    }
    if (lane == 0)
        for (uint32_t i = n4 * 4u; i < n; i++) dst[i] = src[i];
}

__device__ static uint8_t *win_smem(uint8_t *raw, uint32_t warp,
                                    uint32_t stride, uint32_t rows) {
    const uintptr_t a = ((uintptr_t)raw + 15u) & ~(uintptr_t)15u;
    return (uint8_t *)a + (size_t)warp * rows * stride;
}

__global__ static void win_pack_k(const int32_t *d_es, const float *d_ew,
                                  const int32_t *d_nh, uint32_t n_tok,
                                  q4_win_plan *p) {
    if (threadIdx.x || blockIdx.x) return;
    int32_t nuniq = 0, nhit = 0;
    for (uint32_t t = 0; t < n_tok && t < 8u; t++) {
        int32_t nh = d_nh[t];
        if (nh < 0) nh = 0;
        if (nh > 16) nh = 16;
        for (int32_t i = 0; i < nh; i++) {
            const int32_t slot = d_es[(size_t)t * 64u + (uint32_t)i];
            if (slot < 0) continue;
            int32_t u = 0;
            for (; u < nuniq; u++)
                if (p->slot[u] == slot) break;
            if (u == nuniq) {
                if (nuniq >= 64) continue;
                p->slot[u] = slot;
                p->tcnt[u] = 0;
                nuniq++;
            }
            const uint32_t c = p->tcnt[u];
            if (c >= 8u || nhit >= 64) continue;
            const uint32_t pi = (uint32_t)u * 8u + c;
            p->tok[pi] = t;
            p->hrow[pi] = (uint32_t)nhit;
            p->wt[pi] = d_ew[(size_t)t * 64u + (uint32_t)i];
            p->tcnt[u] = c + 1u;
            nhit++;
        }
    }
    p->nuniq = nuniq;
    p->nhit = nhit;
}

/* One warp, one output row of one expert (blockIdx.y).  The y grid is
 * the plan cap (64); blocks past nuniq return.  Weight bytes are staged
 * once; the oct loop matches Q4_WMOE_GU8 so each token's dot is the same
 * int8 product. */
#define Q4_WMOE_WIN_GU(name, octfn, bb)                                        \
    __global__ static void name(float *h, const uint8_t *bg,                   \
                                const uint8_t *bu, uint64_t stride,            \
                                const q4_win_plan *plan, const int8_t *xq8,    \
                                const float *xd8, uint32_t in_dim,             \
                                uint32_t out_dim) {                            \
        extern __shared__ uint8_t wsm_raw[];                                   \
        const int32_t e = (int32_t)blockIdx.y;                                 \
        if (e >= plan->nuniq) return;                                          \
        const uint32_t tc = plan->tcnt[e];                                     \
        if (!tc) return;                                                       \
        const uint32_t nb = in_dim >> 8;                                       \
        const uint32_t nbytes = nb * (uint32_t)(bb);                           \
        const uint32_t rstride = (nbytes + 15u) & ~15u;                        \
        Q4_WGV_ROW();                                                          \
        uint8_t *sg = win_smem(wsm_raw, threadIdx.x >> 5, rstride, 2u);        \
        uint8_t *su = sg + rstride;                                            \
        const int32_t slot = plan->slot[e];                                    \
            const uint8_t *wg = bg + (size_t)slot * stride +                   \
                                (size_t)row * nbytes;                          \
            const uint8_t *wu = bu + (size_t)slot * stride +                   \
                                (size_t)row * nbytes;                          \
            win_ld(sg, wg, nbytes, lane);                                      \
            win_ld(su, wu, nbytes, lane);                                      \
            __syncwarp();                                                      \
            for (uint32_t ti = 0; ti < tc; ti++) {                             \
                const uint32_t pi = (uint32_t)e * 8u + ti;                     \
                const uint32_t tok = plan->tok[pi];                            \
                const uint32_t hr = plan->hrow[pi];                            \
                const int8_t *xq = xq8 + (size_t)tok * in_dim;                 \
                const float *xd = xd8 + (size_t)tok * (in_dim >> 5);           \
                float ag = 0.f, au = 0.f;                                      \
                for (uint32_t b0 = 0; b0 < nb; b0 += 4u) {                      \
                    const uint32_t b1 = b0 + 1u, b2 = b0 + 2u, b3 = b0 + 3u;    \
                    if (b0 < nb) {                                             \
                        float dg, du;                                          \
                        int32_t sg8 = octfn(sg + (size_t)b0 * (bb),            \
                                            xq + b0 * 256u, lane, &dg);        \
                        int32_t su8 = octfn(su + (size_t)b0 * (bb),            \
                                            xq + b0 * 256u, lane, &du);        \
                        const float dx = xd[b0 * 8u + (lane >> 2)];            \
                        ag += dg * dx * (float)sg8;                            \
                        au += du * dx * (float)su8;                            \
                    }                                                          \
                    if (b1 < nb) {                                             \
                        float dg, du;                                          \
                        int32_t sg8 = octfn(sg + (size_t)b1 * (bb),            \
                                            xq + b1 * 256u, lane, &dg);        \
                        int32_t su8 = octfn(su + (size_t)b1 * (bb),            \
                                            xq + b1 * 256u, lane, &du);        \
                        const float dx = xd[b1 * 8u + (lane >> 2)];            \
                        ag += dg * dx * (float)sg8;                            \
                        au += du * dx * (float)su8;                            \
                    }                                                          \
                    if (b2 < nb) {                                             \
                        float dg, du;                                          \
                        int32_t sg8 = octfn(sg + (size_t)b2 * (bb),            \
                                            xq + b2 * 256u, lane, &dg);        \
                        int32_t su8 = octfn(su + (size_t)b2 * (bb),            \
                                            xq + b2 * 256u, lane, &du);        \
                        const float dx = xd[b2 * 8u + (lane >> 2)];            \
                        ag += dg * dx * (float)sg8;                            \
                        au += du * dx * (float)su8;                            \
                    }                                                          \
                    if (b3 < nb) {                                             \
                        float dg, du;                                          \
                        int32_t sg8 = octfn(sg + (size_t)b3 * (bb),            \
                                            xq + b3 * 256u, lane, &dg);        \
                        int32_t su8 = octfn(su + (size_t)b3 * (bb),            \
                                            xq + b3 * 256u, lane, &du);        \
                        const float dx = xd[b3 * 8u + (lane >> 2)];            \
                        ag += dg * dx * (float)sg8;                            \
                        au += du * dx * (float)su8;                            \
                    }                                                          \
                }                                                              \
                ag = wgv_shfl_red(ag);                                         \
                au = wgv_shfl_red(au);                                         \
                if (lane == 0)                                                 \
                    h[(size_t)hr * out_dim + row] =                            \
                        ag / (1.f + expf(-ag)) * au;                           \
            }                                                                  \
    }
Q4_WMOE_WIN_GU(wmoe_iq2xxs_gu_win, wgd_iq2xxs_i8, 66ull)
Q4_WMOE_WIN_GU(wmoe_iq2xs_gu_win, wgd_iq2xs_i8, 74ull)
Q4_WMOE_WIN_GU(wmoe_iq2_gu_win, wgd_iq2s_i8, 82ull)
Q4_WMOE_WIN_GU(wmoe_iq3_gu_win, wgd_iq3s_i8, 110ull)
Q4_WMOE_WIN_GU(wmoe_iq3xxs_gu_win, wgd_iq3xxs_i8, 98ull)
Q4_WMOE_WIN_GU(wmoe_iq4xs_gu_win, wgd_iq4xs_i8, 136ull)

/* IQ4_NL down, float activation.  Same block walk as wmoe_iq4_ya_bk; the
 * weight row is the staged copy. */
__global__ static void wmoe_iq4_ya_win(float *y, const uint8_t *base,
                                       uint64_t stride, const q4_win_plan *plan,
                                       const float *h, uint32_t n_ff,
                                       uint32_t out_dim) {
    extern __shared__ uint8_t wsm_raw[];
    const int32_t e = (int32_t)blockIdx.y;
    if (e >= plan->nuniq) return;
    const uint32_t tc = plan->tcnt[e];
    if (!tc) return;
    const uint32_t nb = n_ff >> 5;
    const uint32_t nbytes = nb * 18u;
    const uint32_t rstride = (nbytes + 15u) & ~15u;
    Q4_WGV_ROW();
    uint8_t *sr = win_smem(wsm_raw, threadIdx.x >> 5, rstride, 1u);
    const uint32_t b = lane >> 2;
    const uint8_t *wr = base + (size_t)plan->slot[e] * stride +
                        (size_t)row * nbytes;
        win_ld(sr, wr, nbytes, lane);
        __syncwarp();
        for (uint32_t ti = 0; ti < tc; ti++) {
            const uint32_t pi = (uint32_t)e * 8u + ti;
            const uint32_t tok = plan->tok[pi];
            const uint32_t hr = plan->hrow[pi];
            const float wt = plan->wt[pi];
            const float *xt = h + (size_t)hr * n_ff;
            float acc = 0.f;
            for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
                const uint32_t bi = b0 + b;
                if (bi < nb)
                    acc += wgd_iq4nl(sr + (size_t)bi * 18ull,
                                     xt + bi * 32u + (lane & 3u) * 8u, lane);
            }
            acc = wgv_shfl_red(acc);
            if (lane == 0)
                atomicAdd(y + (size_t)tok * out_dim + row, acc * wt);
        }
}

/* Q2_0 down against hidden rows quantized by xq8_k.  Group index matches
 * wmoe_q20_ya_q8: block bi covers q8 groups 2*bi and 2*bi+1. */
__global__ static void wmoe_q20_ya_win(float *y, const uint8_t *base,
                                       uint64_t stride, const q4_win_plan *plan,
                                       const int8_t *xq8, const float *xd8,
                                       uint32_t n_ff, uint32_t out_dim) {
    extern __shared__ uint8_t wsm_raw[];
    const int32_t e = (int32_t)blockIdx.y;
    if (e >= plan->nuniq) return;
    const uint32_t tc = plan->tcnt[e];
    if (!tc) return;
    const uint32_t nb = n_ff >> 6;
    const uint32_t nbytes = nb * 18u;
    const uint32_t rstride = (nbytes + 15u) & ~15u;
    Q4_WGV_ROW();
    uint8_t *sr = win_smem(wsm_raw, threadIdx.x >> 5, rstride, 1u);
    const uint32_t b = lane >> 4, byte = lane & 15u;
    const uint8_t *wr = base + (size_t)plan->slot[e] * stride +
                        (size_t)row * nbytes;
    win_ld(sr, wr, nbytes, lane);
        __syncwarp();
        for (uint32_t ti = 0; ti < tc; ti++) {
            const uint32_t pi = (uint32_t)e * 8u + ti;
            const uint32_t tok = plan->tok[pi];
            const uint32_t hr = plan->hrow[pi];
            const float wt = plan->wt[pi];
            const int8_t *xq = xq8 + (size_t)hr * n_ff;
            const float *xd = xd8 + (size_t)hr * (n_ff >> 5);
            float acc = 0.f;
            for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
                const uint32_t bi = b0 + b;
                if (bi < nb) {
                    const uint32_t grp = bi * 2u + (byte >> 3);
                    float dw;
                    uint32_t xu;
                    memcpy(&xu, xq + grp * 32u + (byte & 7u) * 4u, 4);
                    int32_t s = wgd_q20_i8(sr + (size_t)bi * 18ull, xu, byte,
                                           &dw);
                    acc += dw * xd[grp] * (float)s;
                }
            }
            acc = wgv_shfl_red(acc);
            if (lane == 0)
                atomicAdd(y + (size_t)tok * out_dim + row, acc * wt);
        }
}

/* Tiled window down (Q4_WIN_TILE, default on): a block stages 32
 * consecutive rows of one expert (rows are contiguous inside an expert's
 * down tensor) with 16-byte loads all issued together, then each warp
 * walks 4 of the staged rows exactly like the warp-per-row kernels above
 * (same lane map and sum order per row).  The warp-per-row form issued
 * one 360-byte row per warp and spent its time in load latency: 108 us
 * for ~23 MB at 4 verify rows.  Blocks loop over (expert, row group)
 * items, so the fixed captured grid needs no plan-cap y dimension (the
 * y = 64 grid launched ~40 empty blocks per live one). */
#define WIN_TR 32u
__device__ static void win_tile_ld(uint8_t *sm, const uint8_t *src,
                                   uint32_t n16) {
    const uint4 *s4 = (const uint4 *)src;
    uint4 *d4 = (uint4 *)sm;
    uint32_t i = threadIdx.x;
    for (; i + 3u * blockDim.x < n16; i += 4u * blockDim.x) {
        const uint4 a = s4[i], b = s4[i + blockDim.x],
                    c = s4[i + 2u * blockDim.x], d = s4[i + 3u * blockDim.x];
        d4[i] = a;
        d4[i + blockDim.x] = b;
        d4[i + 2u * blockDim.x] = c;
        d4[i + 3u * blockDim.x] = d;
    }
    for (; i < n16; i += blockDim.x) d4[i] = s4[i];
}

__global__ static void wmoe_iq4_ya_wint(float *y, const uint8_t *base,
                                        uint64_t stride,
                                        const q4_win_plan *plan,
                                        const float *h, uint32_t n_ff,
                                        uint32_t out_dim) {
    extern __shared__ uint8_t wsm_raw[];
    uint8_t *sm = (uint8_t *)(((uintptr_t)wsm_raw + 15u) & ~(uintptr_t)15u);
    const uint32_t nb = n_ff >> 5;
    const uint32_t nbytes = nb * 18u;
    const uint32_t ngrp = out_dim / WIN_TR;
    const uint32_t nitem = (uint32_t)plan->nuniq * ngrp;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t b = lane >> 2;
    const uint32_t rpw = WIN_TR / (blockDim.x >> 5);
    for (uint32_t it = blockIdx.x; it < nitem; it += gridDim.x) {
        const uint32_t e = it / ngrp, g = it % ngrp;
        __syncthreads();
        win_tile_ld(sm,
                    base + (size_t)plan->slot[e] * stride +
                        (size_t)g * WIN_TR * nbytes,
                    WIN_TR * nbytes / 16u);
        __syncthreads();
        const uint32_t tc = plan->tcnt[e];
        for (uint32_t r = 0; r < rpw; r++) {
            const uint32_t lr = warp * rpw + r;
            const uint32_t row = g * WIN_TR + lr;
            const uint8_t *sr = sm + (size_t)lr * nbytes;
            for (uint32_t ti = 0; ti < tc; ti++) {
                const uint32_t pi = e * 8u + ti;
                const uint32_t tok = plan->tok[pi];
                const float *xt = h + (size_t)plan->hrow[pi] * n_ff;
                float acc = 0.f;
                for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
                    const uint32_t bi = b0 + b;
                    if (bi < nb)
                        acc += wgd_iq4nl(sr + (size_t)bi * 18ull,
                                         xt + bi * 32u + (lane & 3u) * 8u,
                                         lane);
                }
                acc = wgv_shfl_red(acc);
                if (lane == 0)
                    atomicAdd(y + (size_t)tok * out_dim + row,
                              acc * plan->wt[pi]);
            }
        }
    }
}

__global__ static void wmoe_q20_ya_wint(float *y, const uint8_t *base,
                                        uint64_t stride,
                                        const q4_win_plan *plan,
                                        const int8_t *xq8, const float *xd8,
                                        uint32_t n_ff, uint32_t out_dim) {
    extern __shared__ uint8_t wsm_raw[];
    uint8_t *sm = (uint8_t *)(((uintptr_t)wsm_raw + 15u) & ~(uintptr_t)15u);
    const uint32_t nb = n_ff >> 6;
    const uint32_t nbytes = nb * 18u;
    const uint32_t ngrp = out_dim / WIN_TR;
    const uint32_t nitem = (uint32_t)plan->nuniq * ngrp;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t b = lane >> 4, byte = lane & 15u;
    const uint32_t rpw = WIN_TR / (blockDim.x >> 5);
    for (uint32_t it = blockIdx.x; it < nitem; it += gridDim.x) {
        const uint32_t e = it / ngrp, g = it % ngrp;
        __syncthreads();
        win_tile_ld(sm,
                    base + (size_t)plan->slot[e] * stride +
                        (size_t)g * WIN_TR * nbytes,
                    WIN_TR * nbytes / 16u);
        __syncthreads();
        const uint32_t tc = plan->tcnt[e];
        for (uint32_t r = 0; r < rpw; r++) {
            const uint32_t lr = warp * rpw + r;
            const uint32_t row = g * WIN_TR + lr;
            const uint8_t *sr = sm + (size_t)lr * nbytes;
            for (uint32_t ti = 0; ti < tc; ti++) {
                const uint32_t pi = e * 8u + ti;
                const uint32_t tok = plan->tok[pi];
                const uint32_t hr = plan->hrow[pi];
                const int8_t *xq = xq8 + (size_t)hr * n_ff;
                const float *xd = xd8 + (size_t)hr * (n_ff >> 5);
                float acc = 0.f;
                for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
                    const uint32_t bi = b0 + b;
                    if (bi < nb) {
                        const uint32_t grp = bi * 2u + (byte >> 3);
                        float dw;
                        uint32_t xu;
                        memcpy(&xu, xq + grp * 32u + (byte & 7u) * 4u, 4);
                        int32_t sq = wgd_q20_i8(sr + (size_t)bi * 18ull, xu,
                                                byte, &dw);
                        acc += dw * xd[grp] * (float)sq;
                    }
                }
                acc = wgv_shfl_red(acc);
                if (lane == 0)
                    atomicAdd(y + (size_t)tok * out_dim + row,
                              acc * plan->wt[pi]);
            }
        }
    }
}

/* IQ4_NL down, four lanes per row (Q4_WIN_TILE=2, default): a block
 * stages 64 rows of one expert (16-byte loads, issued together); lane
 * part p of a row decodes blocks p, p+4, ... once into registers and
 * dots them with every token of the expert, then two xor shuffles sum the
 * four parts.  The warp-per-row form spent each row's work on a 32-lane
 * reduction per token and re-read the hidden row per output row (77K
 * row-token dots per verify layer at ~4 rows: 103 us). */
#define WIN_LR 64u
__global__ static void __launch_bounds__(256)
wmoe_iq4_ya_lpr(float *__restrict__ y, const uint8_t *__restrict__ base,
                uint64_t stride, const q4_win_plan *__restrict__ plan,
                const float *__restrict__ h, uint32_t n_ff, uint32_t out_dim) {
    extern __shared__ uint8_t wsm_raw[];
    uint8_t *sm = (uint8_t *)(((uintptr_t)wsm_raw + 15u) & ~(uintptr_t)15u);
    const uint32_t nb = n_ff >> 5;
    const uint32_t nbytes = nb * 18u;
    const uint32_t ngrp = out_dim / WIN_LR;
    const uint32_t nitem = (uint32_t)plan->nuniq * ngrp;
    const uint32_t lr = threadIdx.x >> 2, part = threadIdx.x & 3u;
    for (uint32_t it = blockIdx.x; it < nitem; it += gridDim.x) {
        const uint32_t e = it / ngrp, g = it % ngrp;
        __syncthreads();
        win_tile_ld(sm,
                    base + (size_t)plan->slot[e] * stride +
                        (size_t)g * WIN_LR * nbytes,
                    WIN_LR * nbytes / 16u);
        __syncthreads();
        const uint32_t tc = plan->tcnt[e] < 8u ? plan->tcnt[e] : 8u;
        const uint8_t *sr = sm + (size_t)lr * nbytes;
        float acc[8];
#pragma unroll
        for (int t = 0; t < 8; t++) acc[t] = 0.f;
        for (uint32_t bi = part; bi < nb; bi += 4u) {
            const uint8_t *bp = sr + (size_t)bi * 18u;
            const float d = f16d(bp);
            uint32_t q[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint16_t lo = *(const uint16_t *)(bp + 2 + 4 * k);
                const uint16_t hi = *(const uint16_t *)(bp + 4 + 4 * k);
                q[k] = (uint32_t)lo | ((uint32_t)hi << 16);
            }
            float v[32];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t a = kv_iq4nl_lut4(q[k] & 0x0f0f0f0fu);
                const uint32_t b = kv_iq4nl_lut4((q[k] >> 4) & 0x0f0f0f0fu);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    v[4 * k + j] = (float)(int8_t)(a >> (8 * j));
                    v[16 + 4 * k + j] = (float)(int8_t)(b >> (8 * j));
                }
            }
#pragma unroll
            for (int t = 0; t < 8; t++) {
                if ((uint32_t)t >= tc) break;
                const float4 *x4 = (const float4 *)(
                    h + (size_t)plan->hrow[e * 8u + (uint32_t)t] * n_ff +
                    bi * 32u);
                float sum = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float4 xv = x4[j];
                    sum += v[4 * j] * xv.x + v[4 * j + 1] * xv.y +
                           v[4 * j + 2] * xv.z + v[4 * j + 3] * xv.w;
                }
                acc[t] += d * sum;
            }
        }
        const uint32_t row = g * WIN_LR + lr;
#pragma unroll
        for (int t = 0; t < 8; t++) {
            if ((uint32_t)t >= tc) break;
            float a = acc[t];
            a += __shfl_xor(a, 1, 32);
            a += __shfl_xor(a, 2, 32);
            if (part == 0) {
                const uint32_t pi = e * 8u + (uint32_t)t;
                atomicAdd(y + (size_t)plan->tok[pi] * out_dim + row,
                          a * plan->wt[pi]);
            }
        }
    }
}

static int g_win_tile = -1;
static int win_tile_on(void) {
    if (g_win_tile < 0) {
        const char *e = getenv("Q4_WIN_TILE");
        g_win_tile = e ? atoi(e) : 2;
    }
    return g_win_tile;
}
extern "C" void q4_hip_win_tile_set(int v) { g_win_tile = v; }

/* Persistent grid for the tiled kernels: a few blocks per CU, each
 * looping over (expert, 32-row group) items. */
static uint32_t win_tile_grid(void) {
    static uint32_t gsz;
    if (!gsz) {
        int cu = 0;
        if (hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount,
                                  0) != hipSuccess || cu <= 0)
            cu = 96;
        const char *e = getenv("Q4_WIN_TILE_BPC");
        int bpc = e ? atoi(e) : 6;
        if (bpc < 1) bpc = 1;
        gsz = (uint32_t)(cu * bpc);
    }
    return gsz;
}

/* One warp builds the plan: entry f (token-major, valid slots only) is an
 * expert's first use iff no earlier entry has its slot; ranks come from
 * ballots, so the order (unique experts by first use, pairs token-major,
 * hrow = entry rank) is the serial loop's.  More than 64 valid entries
 * (cannot happen at <= 8 tokens x top-8, kept for safety) falls back to
 * the serial walk in lane 0. */
__global__ static void win_pack_w(const int32_t *d_es, const float *d_ew,
                                  const int32_t *d_nh, uint32_t n_tok,
                                  q4_win_plan *p) {
    __shared__ int32_t ss[128];
    __shared__ uint32_t st[128];
    __shared__ float sw[128];
    __shared__ int32_t s_n;
    __shared__ uint32_t s_cnt[4];
    const uint32_t lane = threadIdx.x;
    /* gather: thread f = t * 16 + i owns hit i of token t; valid entries
     * keep the token-major order through a ballot prefix */
    {
        const uint32_t f = threadIdx.x, t = f >> 4, i = f & 15u;
        int32_t slot = -1;
        if (t < n_tok && t < 8u) {
            int32_t nh = d_nh[t];
            if (nh > 16) nh = 16;
            if ((int32_t)i < nh) slot = d_es[(size_t)t * 64u + i];
        }
        const bool v = slot >= 0;
        const unsigned long long bal = __ballot(v);
        const uint32_t w = f >> 5, wl = f & 31u;
        if (wl == 0) s_cnt[w] = (uint32_t)__popcll(bal);
        __syncthreads();
        uint32_t base = 0;
        for (uint32_t k = 0; k < w; k++) base += s_cnt[k];
        if (v) {
            const uint32_t r =
                base + (uint32_t)__popcll(bal & ((1ull << wl) - 1ull));
            ss[r] = slot;
            st[r] = t;
            sw[r] = d_ew[(size_t)t * 64u + i];
        }
        if (f == 0) s_n = (int32_t)(s_cnt[0] + s_cnt[1] + s_cnt[2] + s_cnt[3]);
        __syncthreads();
    }
    const int32_t n = s_n;
    if (n > 64) {
        if (lane) return;
        int32_t nuniq = 0, nhit = 0;
        for (int32_t f = 0; f < n; f++) {
            const int32_t slot = ss[f];
            int32_t u = 0;
            for (; u < nuniq; u++)
                if (p->slot[u] == slot) break;
            if (u == nuniq) {
                if (nuniq >= 64) continue;
                p->slot[u] = slot;
                p->tcnt[u] = 0;
                nuniq++;
            }
            const uint32_t c = p->tcnt[u];
            if (c >= 8u || nhit >= 64) continue;
            const uint32_t pi = (uint32_t)u * 8u + c;
            p->tok[pi] = st[f];
            p->hrow[pi] = (uint32_t)nhit;
            p->wt[pi] = sw[f];
            p->tcnt[u] = c + 1u;
            nhit++;
        }
        p->nuniq = nuniq;
        p->nhit = nhit;
        return;
    }
    __shared__ int32_t s_first[64];   /* entry -> its expert's first entry */
    __shared__ int32_t s_u[64];       /* first entry -> unique index */
    for (int32_t f = (int32_t)lane; f < n; f += (int32_t)blockDim.x) {
        int32_t fi = f;
        for (int32_t j = 0; j < f; j++)
            if (ss[j] == ss[f]) { fi = j; break; }
        s_first[f] = fi;
    }
    __syncthreads();
    if (lane == 0) {
        int32_t u = 0;
        for (int32_t f = 0; f < n; f++)
            if (s_first[f] == f) s_u[f] = u++;
        p->nuniq = u;
        p->nhit = n;
    }
    __syncthreads();
    for (int32_t f = (int32_t)lane; f < n; f += (int32_t)blockDim.x) {
        const int32_t fi = s_first[f];
        const int32_t u = s_u[fi];
        uint32_t c = 0;
        for (int32_t j = fi; j < f; j++) c += (s_first[j] == fi);
        if (fi == f) {
            p->slot[u] = ss[f];
            int32_t cnt = 0;
            for (int32_t j = f; j < n; j++) cnt += (s_first[j] == f);
            p->tcnt[u] = (uint32_t)cnt;
        }
        const uint32_t pi = (uint32_t)u * 8u + c;
        p->tok[pi] = st[f];
        p->hrow[pi] = (uint32_t)f;
        p->wt[pi] = sw[f];
    }
}

/* Decode-once window gate/up (Q4_WIN_TILE >= 2): one warp per output row
 * over persistent (expert, 8-row group) items, weights read in place (no
 * LDS row copy).  Each lane decodes its oct of a super-block once into the
 * sign-split words dot8_sg uses and dots every token of the expert with
 * them, so a token's sum is the same int8 products in the same order as
 * Q4_WMOE_WIN_GU's.  The old kernel re-decoded the row per token and
 * staged it through LDS first (65-89 us per verify layer). */
struct win_oct {
    uint32_t p0, p1, n0, n1;
    float d;
};
__device__ static inline void win_oct_set(win_oct *o, uint64_t gv,
                                          uint32_t sg, float d) {
    uint32_t m0, m1;
    sg_masks(sg, &m0, &m1);
    const uint32_t w0 = (uint32_t)gv, w1 = (uint32_t)(gv >> 32);
    o->p0 = w0 & ~m0;
    o->p1 = w1 & ~m1;
    o->n0 = w0 & m0;
    o->n1 = w1 & m1;
    o->d = d;
}
__device__ static inline int32_t win_oct_dot(const win_oct &o,
                                             const int8_t *xq) {
    uint32_t xl, xh;
    memcpy(&xl, xq, 4);
    memcpy(&xh, xq + 4, 4);
    int32_t p = 0, n = 0;
    p = __builtin_amdgcn_sudot4(false, (int)o.p0, true, (int)xl, p, false);
    p = __builtin_amdgcn_sudot4(false, (int)o.p1, true, (int)xh, p, false);
    n = __builtin_amdgcn_sudot4(false, (int)o.n0, true, (int)xl, n, false);
    n = __builtin_amdgcn_sudot4(false, (int)o.n1, true, (int)xh, n, false);
    return p - n;
}
/* decoders: the bodies of wgd_iq2s_i8 / wgd_iq3s_i8 / wgd_iq3xxs_i8 */
__device__ static inline void wdec_iq2s(const uint8_t *blk, uint32_t lane,
                                        win_oct *o) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    const float sc = (l < 2) ? (float)(blk[74 + ib] & 0xf)
                             : (float)(blk[74 + ib] >> 4);
    const int idx = blk[2 + ib * 4 + l] |
                    ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    win_oct_set(o, q4_iq2s_grid[idx], blk[2 + 32 + ib * 4 + l],
                d * (0.5f + sc) * 0.25f);
}
__device__ static inline void wdec_iq3s(const uint8_t *blk, uint32_t lane,
                                        win_oct *o) {
    const float d = f16d(blk);
    const uint32_t p = lane >> 3, half = (lane >> 2) & 1u, l = lane & 3u;
    const float sc = (half == 0) ? (float)(blk[106 + p] & 0xf)
                                 : (float)(blk[106 + p] >> 4);
    const uint8_t qhb = blk[66 + p * 2 + half];
    const int i1 = blk[2 + p * 16 + half * 8 + 2 * l] |
                   ((qhb << (8 - 2 * l)) & 256);
    const int i2 = blk[2 + p * 16 + half * 8 + 2 * l + 1] |
                   ((qhb << (7 - 2 * l)) & 256);
    win_oct_set(o, (uint64_t)q4_iq3s_grid[i2] << 32 | q4_iq3s_grid[i1],
                blk[74 + p * 8 + half * 4 + l], d * (1.f + 2.f * sc));
}
__device__ static inline void wdec_iq3xxs(const uint8_t *blk, uint32_t lane,
                                          win_oct *o) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    const uint8_t *q3 = blk + 2 + ib * 8;
    win_oct_set(o,
                (uint64_t)q4_iq3xxs_grid[q3[2 * l + 1]] << 32 |
                    q4_iq3xxs_grid[q3[2 * l]],
                q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u],
                d * (0.5f + (float)(aux32 >> 28)) * 0.5f);
}

#define Q4_WMOE_WIN_GU2(name, decfn, bb)                                       \
    __global__ static void __launch_bounds__(256)                              \
    name(float *h, const uint8_t *bg, const uint8_t *bu, uint64_t stride,      \
         const q4_win_plan *plan, const int8_t *xq8, const float *xd8,         \
         uint32_t in_dim, uint32_t out_dim) {                                  \
        const uint32_t nb = in_dim >> 8;                                       \
        const uint32_t nbytes = nb * (uint32_t)(bb);                           \
        const uint32_t nrw = blockDim.x >> 5;                                  \
        const uint32_t ngrp = (out_dim + nrw - 1u) / nrw;                      \
        const uint32_t nitem = (uint32_t)plan->nuniq * ngrp;                   \
        const uint32_t lane = threadIdx.x & 31u;                               \
        for (uint32_t it = blockIdx.x; it < nitem; it += gridDim.x) {          \
            const uint32_t e = it / ngrp;                                      \
            const uint32_t row = (it % ngrp) * nrw + (threadIdx.x >> 5);       \
            if (row >= out_dim) continue;                                      \
            const uint32_t tc = plan->tcnt[e] < 8u ? plan->tcnt[e] : 8u;       \
            const size_t so = (size_t)plan->slot[e] * stride +                 \
                              (size_t)row * nbytes;                            \
            const uint8_t *wg = bg + so, *wu = bu + so;                        \
            float ag[8], au[8];                                                \
            _Pragma("unroll") for (int t = 0; t < 8; t++) ag[t] = au[t] = 0.f; \
            for (uint32_t b = 0; b < nb; b++) {                                \
                win_oct og, ou;                                                \
                decfn(wg + (size_t)b * (bb), lane, &og);                       \
                decfn(wu + (size_t)b * (bb), lane, &ou);                       \
                _Pragma("unroll") for (int t = 0; t < 8; t++) {                \
                    if ((uint32_t)t >= tc) break;                              \
                    const uint32_t tok = plan->tok[e * 8u + (uint32_t)t];      \
                    const int8_t *xq = xq8 + (size_t)tok * in_dim + b * 256u + \
                                       lane * 8u;                              \
                    const float dx = xd8[(size_t)tok * (in_dim >> 5) +         \
                                         b * 8u + (lane >> 2)];                \
                    ag[t] += og.d * dx * (float)win_oct_dot(og, xq);           \
                    au[t] += ou.d * dx * (float)win_oct_dot(ou, xq);           \
                }                                                              \
            }                                                                  \
            _Pragma("unroll") for (int t = 0; t < 8; t++) {                    \
                if ((uint32_t)t >= tc) break;                                  \
                const float g = wgv_shfl_red(ag[t]);                           \
                const float u = wgv_shfl_red(au[t]);                           \
                if (lane == 0)                                                 \
                    h[(size_t)plan->hrow[e * 8u + (uint32_t)t] * out_dim +     \
                      row] = g / (1.f + expf(-g)) * u;                         \
            }                                                                  \
        }                                                                      \
    }
Q4_WMOE_WIN_GU2(wmoe_iq2_gu_wd, wdec_iq2s, 82ull)
Q4_WMOE_WIN_GU2(wmoe_iq3_gu_wd, wdec_iq3s, 110ull)
Q4_WMOE_WIN_GU2(wmoe_iq3xxs_gu_wd, wdec_iq3xxs, 98ull)

static uint32_t win_iq_bb(uint32_t ty) {
    switch (ty) {
    case Q4_T_IQ2_XXS: return 66u;
    case Q4_T_IQ2_XS:  return 74u;
    case Q4_T_IQ2_S:   return 82u;
    case Q4_T_IQ3_XXS: return 98u;
    case Q4_T_IQ3_S:   return 110u;
    case Q4_T_IQ4_XS:  return 136u;
    default:           return 0u;
    }
}

static int win_smem_fit(size_t smem) {
    static int maxb = -2;
    if (maxb == -2) {
        int v = 0;
        maxb = (g_ok && hipDeviceGetAttribute(
                            &v, hipDeviceAttributeMaxSharedMemoryPerBlock,
                            0) == hipSuccess)
                   ? v
                   : 0;
    }
    return maxb > 0 && smem <= (size_t)maxb;
}

static size_t win_gu_smem(uint32_t bb, uint32_t in_dim) {
    const uint32_t nb = in_dim >> 8;
    const uint32_t rstride = ((nb * bb) + 15u) & ~15u;
    return (size_t)8u * 2u * rstride + 16u;
}

static size_t win_dn_smem(uint32_t nbytes_row) {
    const uint32_t rstride = (nbytes_row + 15u) & ~15u;
    return (size_t)8u * rstride + 16u;
}

extern "C" bool q4_hip_moe_win_ok(uint32_t gt, uint32_t ut, uint32_t dt,
                                  uint32_t in_dim, uint32_t n_ff,
                                  uint32_t n_tok) {
    if (g_win_dead || !wgv_on() || !wq8_on()) return false;
    if (n_tok < 2u || n_tok > 8u || gt != ut || (in_dim & 255u)) return false;
    const uint32_t bb = win_iq_bb(gt);
    if (!bb) return false;
    size_t need = win_gu_smem(bb, in_dim);
    size_t dn = 0;
    if (dt == Q4_T_IQ4_NL) {
        if (n_ff & 31u) return false;
        dn = win_dn_smem((n_ff >> 5) * 18u);
    } else if (dt == Q4_T_Q2_0) {
        if ((n_ff & 63u) || (n_ff & 31u)) return false;
        dn = win_dn_smem((n_ff >> 6) * 18u);
    } else {
        return false;
    }
    if (dn > need) need = dn;
    return win_smem_fit(need);
}

static bool win_gu_launch(uint32_t ty, float *h, const uint8_t *bg,
                          const uint8_t *bu, uint64_t stride, uint32_t in_dim,
                          uint32_t n_ff) {
    const uint32_t bb = win_iq_bb(ty);
    if (!bb) return false;
    if (win_tile_on() >= 2 &&
        (ty == Q4_T_IQ2_S || ty == Q4_T_IQ3_S || ty == Q4_T_IQ3_XXS)) {
        dim3 g2(win_tile_grid()), b2(256);
        if (ty == Q4_T_IQ2_S)
            wmoe_iq2_gu_wd<<<g2, b2, 0, g_str>>>(h, bg, bu, stride, g_win,
                                                 g_xq8, g_xq8d, in_dim, n_ff);
        else if (ty == Q4_T_IQ3_S)
            wmoe_iq3_gu_wd<<<g2, b2, 0, g_str>>>(h, bg, bu, stride, g_win,
                                                 g_xq8, g_xq8d, in_dim, n_ff);
        else
            wmoe_iq3xxs_gu_wd<<<g2, b2, 0, g_str>>>(
                h, bg, bu, stride, g_win, g_xq8, g_xq8d, in_dim, n_ff);
        return hipGetLastError() == hipSuccess;
    }
    const size_t smem = win_gu_smem(bb, in_dim);
    /* y = plan slot cap.  Blocks past nuniq return before touching weights. */
    dim3 grd((n_ff + 7u) / 8u, 64u), blk(256);
    switch (ty) {
    case Q4_T_IQ2_XXS:
        wmoe_iq2xxs_gu_win<<<grd, blk, smem, g_str>>>(
            h, bg, bu, stride, g_win, g_xq8, g_xq8d, in_dim, n_ff);
        break;
    case Q4_T_IQ2_XS:
        wmoe_iq2xs_gu_win<<<grd, blk, smem, g_str>>>(
            h, bg, bu, stride, g_win, g_xq8, g_xq8d, in_dim, n_ff);
        break;
    case Q4_T_IQ2_S:
        wmoe_iq2_gu_win<<<grd, blk, smem, g_str>>>(
            h, bg, bu, stride, g_win, g_xq8, g_xq8d, in_dim, n_ff);
        break;
    case Q4_T_IQ3_S:
        wmoe_iq3_gu_win<<<grd, blk, smem, g_str>>>(
            h, bg, bu, stride, g_win, g_xq8, g_xq8d, in_dim, n_ff);
        break;
    case Q4_T_IQ3_XXS:
        wmoe_iq3xxs_gu_win<<<grd, blk, smem, g_str>>>(
            h, bg, bu, stride, g_win, g_xq8, g_xq8d, in_dim, n_ff);
        break;
    case Q4_T_IQ4_XS:
        wmoe_iq4xs_gu_win<<<grd, blk, smem, g_str>>>(
            h, bg, bu, stride, g_win, g_xq8, g_xq8d, in_dim, n_ff);
        break;
    default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_moe_exec_win(const uint8_t *base, uint64_t stride,
                                    const int32_t *d_es, const float *d_ew,
                                    const int32_t *d_nh, uint32_t n_tok,
                                    uint32_t gt, uint64_t og, uint32_t ut,
                                    uint64_t ou, uint32_t dt, uint64_t od,
                                    const float *d_x, float *d_h, float *d_y,
                                    uint32_t n_ff, uint32_t in_dim,
                                    uint32_t n_embd) {
    if (g_win_dead || !g_ok || !base || !d_es || !d_ew || !d_nh || !d_x ||
        !d_h || !d_y)
        return false;
    if (!q4_hip_moe_win_ok(gt, ut, dt, in_dim, n_ff, n_tok)) return false;
    if (!g_win) {
        if (hipMalloc((void **)&g_win, sizeof(q4_win_plan)) != hipSuccess) {
            g_win = nullptr;
            g_win_dead = 1;
            fprintf(stderr, "q4 moe: verify window disabled (no plan buffer)\n");
            return false;
        }
    }
    /* Token activations, then the unique-expert plan, then gate/up.
     * d_h holds pair-major hidden rows (at most 64).  A Q2_0 down
     * requantizes those rows after gate/up is queued; stream order keeps
     * the second quant from clobbering the gate/up read of g_xq8. */
    if (!xq8_launch(d_x, in_dim, in_dim, n_tok, NULL)) {
        g_win_dead = 1;
        fprintf(stderr, "q4 moe: verify window disabled (activation quant)\n");
        return false;
    }
    if (win_tile_on())
        win_pack_w<<<dim3(1), dim3(128), 0, g_str>>>(d_es, d_ew, d_nh, n_tok,
                                                    g_win);
    else
        win_pack_k<<<dim3(1), dim3(1), 0, g_str>>>(d_es, d_ew, d_nh, n_tok,
                                                   g_win);
    if (hipGetLastError() != hipSuccess ||
        !win_gu_launch(gt, d_h, base + og, base + ou, stride, in_dim, n_ff)) {
        g_win_dead = 1;
        fprintf(stderr, "q4 moe: verify window disabled (gate/up launch)\n");
        return false;
    }
    dim3 grd((n_embd + 7u) / 8u, 64u), blk(256);
    /* tiled down: rows of 16-byte aligned groups, out_dim a multiple of 32 */
    const bool tile = win_tile_on() && !(n_embd % WIN_TR) &&
                      !(((uintptr_t)(base + od) | (uintptr_t)stride) & 15u);
    if (tile) {
        const uint32_t nbr = dt == Q4_T_Q2_0 ? (n_ff >> 6) * 18u
                                             : (n_ff >> 5) * 18u;
        const size_t smem = (size_t)WIN_TR * nbr + 16u;
        if (dt == Q4_T_Q2_0) {
            if (!xq8_launch(d_h, n_ff, n_ff, 0, &g_win->nhit)) {
                g_win_dead = 1;
                fprintf(stderr,
                        "q4 moe: verify window disabled (hidden quant)\n");
                return false;
            }
            wmoe_q20_ya_wint<<<dim3(win_tile_grid()), blk, smem, g_str>>>(
                d_y, base + od, stride, g_win, g_xq8, g_xq8d, n_ff, n_embd);
        } else if (win_tile_on() >= 2 && !(n_embd % WIN_LR)) {
            wmoe_iq4_ya_lpr<<<dim3(win_tile_grid()), blk,
                              (size_t)WIN_LR * nbr + 16u, g_str>>>(
                d_y, base + od, stride, g_win, d_h, n_ff, n_embd);
        } else {
            wmoe_iq4_ya_wint<<<dim3(win_tile_grid()), blk, smem, g_str>>>(
                d_y, base + od, stride, g_win, d_h, n_ff, n_embd);
        }
    } else if (dt == Q4_T_Q2_0) {
        if (!xq8_launch(d_h, n_ff, n_ff, 0, &g_win->nhit)) {
            g_win_dead = 1;
            fprintf(stderr,
                    "q4 moe: verify window disabled (hidden quant)\n");
            return false;
        }
        const size_t smem = win_dn_smem((n_ff >> 6) * 18u);
        wmoe_q20_ya_win<<<grd, blk, smem, g_str>>>(
            d_y, base + od, stride, g_win, g_xq8, g_xq8d, n_ff, n_embd);
    } else {
        const size_t smem = win_dn_smem((n_ff >> 5) * 18u);
        wmoe_iq4_ya_win<<<grd, blk, smem, g_str>>>(
            d_y, base + od, stride, g_win, d_h, n_ff, n_embd);
    }
    if (hipGetLastError() != hipSuccess) {
        g_win_dead = 1;
        fprintf(stderr, "q4 moe: verify window disabled (down launch)\n");
        return false;
    }
    return true;
}

extern "C" void q4_hip_moe_scratch_warm(void) {
    if (g_ok) q4_mmvq_warm();
    if (!g_ok || g_win_dead) return;
    (void)xq8_bufs();
    if (!g_win) {
        if (hipMalloc((void **)&g_win, sizeof(q4_win_plan)) != hipSuccess)
            g_win = nullptr;
    }
}

/* Host-staged hit-exec for the hybrid path.  q4_hip_moe_stage writes the
 * slot/weight/count words into mapped host memory (safe to call between gA
 * and gB every token); q4_hip_moe_exec_staged launches the device-count
 * kernels reading those words — identical inside a graph capture, which is
 * what lets the hit-exec join gB instead of being a host-launch seam. */
extern "C" bool q4_hip_moe_stage(const int32_t *slots, const float *wts,
                                 uint32_t n) {
    if (!moe_stage_bufs() || n > 64) return false;
    memcpy(g_es_h, slots, n * sizeof(int32_t));
    memcpy(g_ew_h, wts, n * sizeof(float));
    *g_en_h = (int32_t)n;
    g_moe_nhits_hint = n;   /* profiling: the count the replay will read */
    return true;
}

extern "C" bool q4_hip_moe_exec_staged(
        const uint8_t *base, uint64_t stride, uint32_t gt, uint64_t og,
        uint32_t ut, uint64_t ou, uint32_t dt, uint64_t od, const float *d_x,
        float *d_h, float *d_h2, float *d_down, float *d_y, uint32_t n_ff,
        uint32_t in_dim, uint32_t n_embd) {
    if (!g_es_d) return false;
    return q4_hip_moe_exec_dev(base, stride, g_es_d, g_ew_d, g_en_d, gt, og,
                               ut, ou, dt, od, d_x, d_h, d_h2, d_down, d_y,
                               n_ff, in_dim, n_embd);
}

extern "C" bool q4_hip_route_dev(const float *d_rlog, const int32_t *devtab,
                                 int32_t *d_es, float *d_ew, int32_t *d_nhit,
                                 const int32_t *restab,
                                 int32_t *d_res_s, float *d_res_w,
                                 int32_t *d_res_n, uint32_t res_cap,
                                 void *mail, unsigned long long *d_seq,
                                 unsigned long long *d_rseq,
                                 unsigned long long *d_prof,
                                 uint32_t n_exp, uint32_t topk) {
    /* route_dev_k stages picks into 64-deep shared arrays (s_pay/sids/
     * swts) — topk > 64 would write out of bounds. */
    if (!g_ok || !d_rlog || !devtab || !d_es || !d_ew || !d_nhit || !mail ||
        !d_seq || n_exp > 1024 || !topk || topk > 64u)
        return false;
    route_dev_k<<<dim3(1), dim3(32), 0, g_str>>>(
        d_rlog, devtab, d_es, d_ew, d_nhit, restab, d_res_s, d_res_w,
        d_res_n, res_cap, (q4_mail_t *)mail, d_seq, d_rseq, d_prof, n_exp,
        topk);
    return hipGetLastError() == hipSuccess;
}

/* Enqueue a 4-byte write on the dedicated copy stream — used to publish or
 * clear device residency-table entries so the writes stay ordered with the
 * expert payload uploads on that stream.  Falls back to a one-thread kernel
 * when stream write-value ops are unavailable. */
__global__ static void set_i32_k(int32_t *dst, int32_t v) { *dst = v; }

/* Batched devtab writes: one tiny kernel applies up to 64 (index, value)
 * pairs in order (a later pair for the same index wins), instead of one
 * stream write-value op per entry.  Each of those is a separate dispatch
 * on the copy queue (~940 per MTP verify cycle), which lengthened the
 * compute queue's inter-kernel gaps. */
#define Q4_SCATTER_MAX 64
struct q4_scatter_args {
    uint32_t n;
    uint32_t idx[Q4_SCATTER_MAX];
    int32_t val[Q4_SCATTER_MAX];
};
__global__ static void scatter_i32_k(int32_t *base, q4_scatter_args a) {
    for (uint32_t i = 0; i < a.n; i++) base[a.idx[i]] = a.val[i];
}

extern "C" bool q4_hip_copy_scatter_i32(int32_t *base, const uint32_t *idx,
                                        const int32_t *val, uint32_t n) {
    if (!base || (n && (!idx || !val))) return false;
    hipStream_t st = g_copy ? g_copy : 0;
    while (n) {
        q4_scatter_args a;
        a.n = n < Q4_SCATTER_MAX ? n : Q4_SCATTER_MAX;
        memcpy(a.idx, idx, a.n * sizeof(uint32_t));
        memcpy(a.val, val, a.n * sizeof(int32_t));
        scatter_i32_k<<<dim3(1), dim3(1), 0, st>>>(base, a);
        if (hipGetLastError() != hipSuccess) return false;
        idx += a.n;
        val += a.n;
        n -= a.n;
    }
    return true;
}

extern "C" bool q4_hip_copy_i32(int32_t *dst, int32_t v) {
    if (!dst) return false;
    if (g_copy)
        return hipStreamWriteValue32(g_copy, dst, (uint32_t)v, 0) ==
               hipSuccess;
    set_i32_k<<<dim3(1), dim3(32), 0, 0>>>(dst, v);
    return hipGetLastError() == hipSuccess;
}

/* memset/memcpy on the copy stream — keeps device-table writes ordered with
 * expert payload uploads (which also ride g_copy). */
extern "C" bool q4_hip_copy_memset(void *dst, int v, size_t n) {
    if (!dst || !n) return false;
    hipStream_t st = g_copy ? g_copy : 0;
    return hipMemsetAsync(dst, v, n, st) == hipSuccess;
}

extern "C" bool q4_hip_copy_h2d(void *dst, const void *src, size_t n) {
    if (!dst || !src || !n) return false;
    hipStream_t st = g_copy ? g_copy : 0;
    return hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, st) ==
           hipSuccess;
}

/* Dequant one IQ block into plain floats so a prefill row can be reused
 * across every token that routed here. */
__device__ static void deq_iq2s_blk(const uint8_t *blk, float *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *scales = blk + 74;
    const uint8_t *signs = qs + 32;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        const float db0 = d * (0.5f + (float)(scales[ib] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const float dl = l < 2 ? db0 : db1;
            const int idx = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
            const uint64_t gv = q4_iq2s_grid[idx];
            const uint8_t sg = signs[l];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float s = (sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f;
                dst[xi++] = dl * iq_u8(gv, j) * s;
            }
        }
        qs += 4;
        signs += 4;
    }
}

__device__ static void deq_iq3s_blk(const uint8_t *blk, float *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *signs = blk + 74;
    const uint8_t *scales = blk + 106;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib += 2) {
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
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    dst[xi + j] = db * iq_u8(g1, j) *
                                  ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f);
                    dst[xi + 4 + j] = db * iq_u8(g2, j) *
                                      ((sg & q4_kmask_iq2xs[j + 4]) ? -1.f
                                                                   : 1.f);
                }
                xi += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

__device__ static void deq_iq4nl_blk(const uint8_t *blk, float *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    for (int j = 0; j < 16; j++) {
        const uint8_t q = qs[j];
        dst[j] = d * (float)q4_kvalues_iq4nl[q & 0xf];
        dst[j + 16] = d * (float)q4_kvalues_iq4nl[q >> 4];
    }
}

/* Per-element dequant: inverse map of deq_*_blk so the shared-memory fill can
 * fan out over all threads instead of one thread per quant block. */
__device__ static float deq_iq2s_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, l = (r & 31u) >> 3, j = r & 7u;
    const float sc = (l < 2) ? (float)(blk[74 + ib] & 0xf)
                             : (float)(blk[74 + ib] >> 4);
    const float dl = d * (0.5f + sc) * 0.25f;
    const int idx = blk[2 + ib * 4 + l] | ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    const float s = (blk[2 + 32 + ib * 4 + l] & (1u << j)) ? -1.f : 1.f;
    return dl * iq_u8(q4_iq2s_grid[idx], j) * s;
}

__device__ static float deq_iq3s_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t p = r >> 6, half = (r >> 5) & 1u, rr = r & 31u;
    const uint32_t l = rr >> 3, jj = rr & 7u;
    const float sc = (half == 0) ? (float)(blk[106 + p] & 0xf)
                                 : (float)(blk[106 + p] >> 4);
    const float db = d * (1.f + 2.f * sc);
    const uint8_t qhb = blk[66 + p * 2 + half];
    const int i1 = blk[2 + p * 16 + half * 8 + 2 * l] |
                   ((qhb << (8 - 2 * l)) & 256);
    const int i2 = blk[2 + p * 16 + half * 8 + 2 * l + 1] |
                   ((qhb << (7 - 2 * l)) & 256);
    const uint8_t sg = blk[74 + p * 8 + half * 4 + l];
    const float s = (sg & q4_kmask_iq2xs[jj]) ? -1.f : 1.f;
    if (jj < 4u) return db * iq_u8(q4_iq3s_grid[i1], jj) * s;
    return db * iq_u8(q4_iq3s_grid[i2], jj - 4u) * s;
}

__device__ static float deq_iq4nl_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint8_t q = blk[2 + (r & 15u)];
    const uint8_t v = (r < 16u) ? (q & 0xf) : (q >> 4);
    return d * (float)q4_kvalues_iq4nl[v];
}

/* Batched dequant: 32 contiguous elements at once (one 32-run within the
 * quantization block; e0 is the element offset, multiple of 32).  Scales,
 * grid bytes and signs are fetched once per 8/32-element run instead of
 * once per element — ~10x fewer instructions in the WMMA tile loads.
 * dst is float or __half; T picks the store conversion. */
__device__ static inline void deq_st(float v, float *d, uint32_t i) {
    d[i] = v;
}
__device__ static inline void deq_st(float v, __half *d, uint32_t i) {
    d[i] = __float2half(v);
}

template<typename T>
__device__ static void deq_iq2s_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    const uint8_t *qs = blk + 2 + ib * 4;
    const uint8_t *sg = blk + 34 + ib * 4;
    const uint32_t qh = blk[66 + ib];
    const uint32_t sc = blk[74 + ib];
    const float dl0 = d * (0.5f + (float)(sc & 0xf)) * 0.25f;
    const float dl1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const float dl = l < 2 ? dl0 : dl1;
        const uint32_t idx = qs[l] | ((qh << (8 - 2 * l)) & 0x300u);
        const uint64_t gv = q4_iq2s_grid[idx];
        const uint32_t s = sg[l];
#pragma unroll
        for (int j = 0; j < 8; j++)
            deq_st(dl * iq_u8(gv, j) * ((s & (1u << j)) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + j));
    }
}

__device__ static float deq_iq2xxs_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, l = (r & 31u) >> 3, j = r & 7u;
    uint32_t aux32[2];
    memcpy(aux32, blk + 2 + ib * 8, 8);
    const float sc = (float)(aux32[1] >> 28);
    const float dl = d * 0.125f * (2.f * sc + 1.f);
    const uint64_t gv = q4_iq2xxs_grid[(aux32[0] >> (8 * l)) & 0xffu];
    const uint8_t sg = q4_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
    return dl * iq_u8(gv, j) * ((sg & (1u << j)) ? -1.f : 1.f);
}

template<typename T>
__device__ static void deq_iq2xxs_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    uint32_t aux32[2];
    memcpy(aux32, blk + 2 + ib * 8, 8);
    const uint8_t *aux8 = (const uint8_t *)aux32;
    const float dl = d * 0.125f * (float)(2 * (int)(aux32[1] >> 28) + 1);
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const uint64_t gv = q4_iq2xxs_grid[aux8[l]];
        const uint32_t s = q4_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
#pragma unroll
        for (int j = 0; j < 8; j++)
            deq_st(dl * iq_u8(gv, j) * ((s & (1u << j)) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + j));
    }
}

__device__ static float deq_iq2xs_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, l = (r & 31u) >> 3, j = r & 7u;
    const uint8_t sb = blk[66 + ib];
    const float sc = (l < 2) ? (float)(sb & 0xf) : (float)(sb >> 4);
    const float dl = d * 0.125f * (2.f * sc + 1.f);
    uint16_t v;
    memcpy(&v, blk + 2 + ib * 8 + 2 * l, 2);
    const float s = (q4_ksigns_iq2xs[v >> 9] & (1u << j)) ? -1.f : 1.f;
    return dl * iq_u8(q4_iq2xs_grid[v & 511u], j) * s;
}

template<typename T>
__device__ static void deq_iq2xs_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    const uint8_t sb = blk[66 + ib];
    const float dl0 = d * 0.125f * (float)(2 * (sb & 0xf) + 1);
    const float dl1 = d * 0.125f * (float)(2 * (sb >> 4) + 1);
    uint16_t v[4];
    memcpy(v, blk + 2 + ib * 8, 8);
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const float dl = l < 2 ? dl0 : dl1;
        const uint64_t gv = q4_iq2xs_grid[v[l] & 511u];
        const uint32_t s = q4_ksigns_iq2xs[v[l] >> 9];
#pragma unroll
        for (int j = 0; j < 8; j++)
            deq_st(dl * iq_u8(gv, j) * ((s & (1u << j)) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + j));
    }
}

template<typename T>
__device__ static void deq_iq3s_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t p = e0 >> 6, half = (e0 >> 5) & 1u;
    const uint32_t sc = blk[106 + p];
    const float db =
        d * (1.f + 2.f * (float)(half ? (sc >> 4) : (sc & 0xf)));
    const uint32_t qhb = blk[66 + p * 2 + half];
    const uint8_t *qs = blk + 2 + p * 16 + half * 8;
    const uint8_t *sg = blk + 74 + p * 8 + half * 4;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const uint32_t i1 = qs[2 * l] | ((qhb << (8 - 2 * l)) & 256u);
        const uint32_t i2 = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256u);
        const uint32_t g1 = q4_iq3s_grid[i1];
        const uint32_t g2 = q4_iq3s_grid[i2];
        const uint32_t s = sg[l];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            deq_st(db * iq_u8(g1, m) *
                       ((s & q4_kmask_iq2xs[m]) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + m));
            deq_st(db * iq_u8(g2, m) *
                       ((s & q4_kmask_iq2xs[m + 4]) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + 4 + m));
        }
    }
}

template<typename T>
__device__ static void deq_iq4nl_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t q = qs[j];
        deq_st(d * (float)q4_kvalues_iq4nl[q & 0xf], dst, (uint32_t)j);
        deq_st(d * (float)q4_kvalues_iq4nl[q >> 4], dst, (uint32_t)(j + 16));
    }
}

/* iq3_xxs: 256-elem block {f16 d, qs[96]}; qs[0..63] grid indices,
 * qs[64..95] = 8 aux32 sign/scale words. */
__device__ static float deq_iq3xxs_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, rr = r & 31u;
    const uint32_t l = rr >> 3, jj = rr & 7u;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
    const uint32_t gv = q4_iq3xxs_grid[blk[2 + ib * 8 + 2 * l + (jj >> 2)]];
    const float s = (sg & (1u << jj)) ? -1.f : 1.f;
    return db * iq_u8(gv, jj & 3u) * s;
}

template<typename T>
__device__ static void deq_iq3xxs_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t *q3 = blk + 2 + ib * 8;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const uint8_t s = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
        const uint32_t g1 = q4_iq3xxs_grid[q3[2 * l]];
        const uint32_t g2 = q4_iq3xxs_grid[q3[2 * l + 1]];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            deq_st(db * iq_u8(g1, m) * ((s & (1u << m)) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + m));
            deq_st(db * iq_u8(g2, m) * ((s & (1u << (m + 4))) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + 4 + m));
        }
    }
}

/* iq4_xs: 256-elem block {f16 d, u16 scales_h, u8 scales_l[4], qs[128]}. */
__device__ static float deq_iq4xs_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, rr = r & 31u;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)(ls - 32);
    const uint8_t q = blk[8 + ib * 16 + (rr & 15u)];
    const uint8_t v = (rr < 16u) ? (q & 0xf) : (q >> 4);
    return dl * (float)q4_kvalues_iq4nl[v];
}

template<typename T>
__device__ static void deq_iq4xs_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)(ls - 32);
    const uint8_t *qs = blk + 8 + ib * 16;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        deq_st(dl * (float)q4_kvalues_iq4nl[qs[j] & 0xf], dst, (uint32_t)j);
        deq_st(dl * (float)q4_kvalues_iq4nl[qs[j] >> 4], dst,
               (uint32_t)(j + 16));
    }
}

__device__ static float deq_q8_0_elem(const uint8_t *blk, uint32_t r) {
    return f16d(blk) * (float)(const int8_t)blk[2 + r];
}

template<typename T>
__device__ static void deq_q8_0_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
#pragma unroll
    for (int j = 0; j < 32; j++)
        deq_st(d * (float)(const int8_t)blk[2 + j], dst, (uint32_t)j);
}

/* q4_0: 32-value block = fp16 d + 16 packed nibbles, lo nibbles are elements
 * 0..15, hi 16..31. */
__device__ static float deq_q40_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint8_t q = blk[2 + (r & 15u)];
    const int v = (r < 16u) ? (q & 0xf) : (q >> 4);
    return d * (float)(v - 8);
}

template<typename T>
__device__ static void deq_q40_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t q = qs[j];
        deq_st(d * (float)((int)(q & 0xf) - 8), dst, (uint32_t)j);
        deq_st(d * (float)((int)(q >> 4) - 8), dst, (uint32_t)(j + 16));
    }
}

/* q5_0: q4_0 nibble order + high bits packed as a u32 at blk+2 (bit i =
 * element i), offset -16. */
__device__ static float deq_q50_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint8_t q = blk[6 + (r & 15u)];
    const int v = (int)(((r < 16u) ? (q & 0xf) : (q >> 4)) |
                        (((qh >> r) & 1u) << 4)) - 16;
    return d * (float)v;
}

template<typename T>
__device__ static void deq_q50_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint8_t *qs = blk + 6;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t q = qs[j];
        deq_st(d * (float)((int)((q & 0xf) | (((qh >> j) & 1u) << 4)) - 16),
               dst, (uint32_t)j);
        deq_st(d * (float)((int)((q >> 4) |
                                 (((qh >> (j + 16)) & 1u) << 4)) - 16),
               dst, (uint32_t)(j + 16));
    }
}

/* q2_0: 64-value block = fp16 d + 16 bytes of 2-bit quads, (q - 1) * d.
 * x32 covers half a block: e0 is 0 or 32. */
__device__ static float deq_q20_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    return d * (float)((int)((blk[2 + (r >> 2)] >> ((r & 3u) * 2)) & 3u) - 1);
}

template<typename T>
__device__ static void deq_q20_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2 + (e0 >> 2);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint32_t qv = qs[j];
#pragma unroll
        for (int m = 0; m < 4; m++)
            deq_st(d * (float)((int)((qv >> (m * 2)) & 3u) - 1),
                   dst, (uint32_t)(j * 4 + m));
    }
}

/* q6_k: 256-value superblock, 210 bytes (ql[128] lo nibbles, qh[64] hi
 * bit-pairs, sc[16] int8 per-16 scales, fp16 d at +208). */
__device__ static float deq_q6_k_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk + 208u);
    const uint32_t half = r >> 7, rr = r & 127u;
    const uint32_t g = rr >> 5, l = rr & 31u;
    const uint32_t ql = blk[half * 64u + l + (g & 1u) * 32u];
    const uint32_t qh = blk[128u + half * 32u + l];
    const uint32_t q =
        ((g & 2u) ? (ql >> 4) : (ql & 0xFu)) | (((qh >> (2u * g)) & 3u) << 4);
    const int32_t sc =
        (int32_t)(const int8_t)blk[192u + half * 8u + g * 2u + (l >> 4)];
    return d * (float)sc * (float)((int32_t)q - 32);
}

template<typename T>
__device__ static void deq_q6_k_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk + 208u);
    const uint32_t half = e0 >> 7, g = (e0 & 127u) >> 5;
    const uint8_t *ql = blk + half * 64u + (g & 1u) * 32u;
    const uint8_t *qh = blk + 128u + half * 32u;
    const int8_t *sc = (const int8_t *)(blk + 192u + half * 8u + g * 2u);
    const uint32_t qsh = 2u * g;
    const uint32_t hi = g & 2u;
    const float d0 = d * (float)sc[0], d1 = d * (float)sc[1];
#pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t q = (hi ? (uint32_t)(ql[j] >> 4)
                               : (uint32_t)(ql[j] & 0xFu)) |
                           ((((uint32_t)qh[j]) >> qsh) & 3u) << 4;
        deq_st((j < 16u ? d0 : d1) * (float)((int32_t)q - 32), dst, j);
    }
}

/* q4_k / q5_k: 256-value superblock. d fp16@0, dmin fp16@2, 12 packed
 * 6-bit scale/min bytes @4 (8 sub-blocks of 32 elems via scale_min_k4
 * indexing), then q4_k: qs[128]@16; q5_k adds qh[32]@16, ql[128]@48.
 * Element e: j64 = e>>6, l = e&31, half = (e>>5)&1, is = j64*2+half. */
__device__ static void scale_min_k4_dev(int is, const uint8_t *sc,
                                        float *d1, float *m1) {
    uint8_t s, m;
    if (is < 4) {
        s = sc[is] & 63;
        m = sc[is + 4] & 63;
    } else {
        s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
        m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
    }
    *d1 = f16d(sc - 4) * (float)s;
    *m1 = f16d(sc - 2) * (float)m;
}

__device__ static float deq_q4k_elem(const uint8_t *blk, uint32_t r) {
    const uint32_t j64 = r >> 6, l = r & 31u, half = (r >> 5) & 1u;
    float d1, m1;
    scale_min_k4_dev((int)(j64 * 2u + half), blk + 4u, &d1, &m1);
    const uint32_t q = blk[16u + j64 * 32u + l];
    return d1 * (float)(half ? (q >> 4) : (q & 0xFu)) - m1;
}

template<typename T>
__device__ static void deq_q4k_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const uint32_t j64 = e0 >> 6, half = (e0 >> 5) & 1u;
    float d1, m1;
    scale_min_k4_dev((int)(j64 * 2u + half), blk + 4u, &d1, &m1);
    const uint8_t *qs = blk + 16u + j64 * 32u;
#pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t q = half ? (uint32_t)(qs[j] >> 4)
                                : (uint32_t)(qs[j] & 0xFu);
        deq_st(d1 * (float)q - m1, dst, j);
    }
}

__device__ static float deq_q5k_elem(const uint8_t *blk, uint32_t r) {
    const uint32_t j64 = r >> 6, l = r & 31u, half = (r >> 5) & 1u;
    const uint32_t is = j64 * 2u + half;
    float d1, m1;
    scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
    const uint32_t q = blk[48u + j64 * 32u + l];
    const uint32_t q5 = (half ? (q >> 4) : (q & 0xFu)) |
                        (((blk[16u + l] >> is) & 1u) << 4);
    return d1 * (float)q5 - m1;
}

template<typename T>
__device__ static void deq_q5k_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const uint32_t j64 = e0 >> 6, half = (e0 >> 5) & 1u;
    const uint32_t is = j64 * 2u + half;
    float d1, m1;
    scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
    const uint8_t *ql = blk + 48u + j64 * 32u;
    const uint8_t *qh = blk + 16u;
#pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t q = half ? (uint32_t)(ql[j] >> 4)
                                : (uint32_t)(ql[j] & 0xFu);
        deq_st(d1 * (float)(q | (((qh[j] >> is) & 1u) << 4)) - m1, dst, j);
    }
}

/* 32-element tile fill for the WMMA A-loads: covers MOEW_KC columns of one
 * row via KC/32 runs; falls back to scalar deq at the kn/nrows fringe so
 * tail chunks and clipped rows still produce zeros where needed. */
#define Q4_DEQ_TILE(As, wp, r0, r, k0, sb, kn, nrows, nb, BLK, BBYTES,       \
                    deq_elem, deq32)                                        \
    do {                                                                   \
        const uint32_t e2_ = (k0) + (sb) * 32u;                             \
        const uint8_t *b_ = (wp) + (size_t)((r0) + (r)) * (nb) * (BBYTES) + \
                            (size_t)(e2_ / (BLK)) * (BBYTES);               \
        if ((sb) * 32u + 32u <= (kn) && (r0) + (r) < (nrows)) {             \
            deq32(b_, e2_ % (BLK), &(As)[(r)][(sb) * 32u]);                 \
        } else {                                                           \
            for (uint32_t j = 0; j < 32u; j++) {                            \
                const uint32_t kc_ = (sb) * 32u + j;                        \
                (As)[(r)][kc_] = __float2half(                              \
                    kc_ < (kn) && (r0) + (r) < (nrows)                      \
                        ? deq_elem(b_, (e2_ % (BLK)) + j)                   \
                        : 0.f);                                             \
            }                                                              \
        }                                                                  \
    } while (0)

/* One row per block. All threads cooperatively dequant the row into shared
 * memory (element-indexed), then each warp dots one token: lanes split ncols
 * (coalesced x reads) and shfl-reduce. */
#define Q4_GEMM_REUSE(name, deq_elem, BLK, BBYTES)                             \
    __global__ static void name(const uint8_t *w, const float *x,              \
                                const int32_t *which, float *out,              \
                                uint32_t nrows, uint32_t ncols, float scale,   \
                                uint32_t n_batch) {                            \
        uint32_t row = blockIdx.x;                                             \
        if (row >= nrows) return;                                             \
        extern __shared__ float rowv[];                                       \
        uint32_t nb = ncols / (BLK);                                           \
        const uint8_t *wr = w + (size_t)row * nb * (BBYTES);                   \
        for (uint32_t e = threadIdx.x; e < ncols; e += blockDim.x)            \
            rowv[e] = deq_elem(wr + (size_t)(e / (BLK)) * (BBYTES),           \
                               e % (BLK));                                     \
        __syncthreads();                                                       \
        uint32_t lane = threadIdx.x & 31u;                                    \
        for (uint32_t tok = threadIdx.x >> 5; tok < n_batch;                  \
             tok += blockDim.x >> 5) {                                        \
            const float *xt =                                                 \
                x + (size_t)(which ? which[tok] : (int32_t)tok) * ncols;      \
            float acc = 0.f;                                                   \
            for (uint32_t i = lane; i < ncols; i += 32)                       \
                acc += rowv[i] * xt[i];                                       \
            for (uint32_t o = 16; o; o >>= 1)                                 \
                acc += __shfl_down(acc, o);                                   \
            if (lane == 0)                                                    \
                out[(size_t)tok * nrows + row] = acc * scale;                 \
        }                                                                      \
    }

Q4_GEMM_REUSE(gemm_iq2xxs_reuse, deq_iq2xxs_elem, 256u, 66u)
Q4_GEMM_REUSE(gemm_iq2xs_reuse, deq_iq2xs_elem, 256u, 74u)
Q4_GEMM_REUSE(gemm_iq2_reuse, deq_iq2s_elem, 256u, 82u)
Q4_GEMM_REUSE(gemm_iq3_reuse, deq_iq3s_elem, 256u, 110u)
Q4_GEMM_REUSE(gemm_iq4_reuse, deq_iq4nl_elem, 32u, 18u)
Q4_GEMM_REUSE(gemm_iq3xxs_reuse, deq_iq3xxs_elem, 256u, 98u)
Q4_GEMM_REUSE(gemm_iq4xs_reuse, deq_iq4xs_elem, 256u, 136u)
Q4_GEMM_REUSE(gemm_q40_reuse, deq_q40_elem, 32u, 18u)
Q4_GEMM_REUSE(gemm_q50_reuse, deq_q50_elem, 32u, 22u)
Q4_GEMM_REUSE(gemm_q20_reuse, deq_q20_elem, 64u, 18u)

/* Batched prefill expert GEMM: grid (row, expert-entry).  Entry e has ec[e]
 * assigned tokens, listed at atok[eb[e]..eb[e]+ec[e]) (global token ids into
 * x), or — when atok == NULL — assignment-indexed rows of x directly.
 * Weights live in the L1 slot arena: base + slots[e]*stride + woff.
 * One launch replaces n_experts small GEMM launches. */
#define Q4_MOE_PF(name, deq_elem, BLK, BBYTES)                                 \
    __global__ static void name(float *out, const uint8_t *wbase,              \
                                uint64_t stride, uint64_t woff,                \
                                const int32_t *slots, const uint32_t *eb,      \
                                const uint32_t *ec, const int32_t *atok,       \
                                const float *x, uint32_t nrows,                \
                                uint32_t ncols, float scale) {                 \
        /* two rows per block: one x load feeds two dots */                   \
        const uint32_t R = 2u;                                                \
        uint32_t r0 = blockIdx.x * R, e = blockIdx.y;                         \
        uint32_t K = ec[e];                                                   \
        if (!K) return;                                                       \
        extern __shared__ float rowv[]; /* [R][ncols] */                      \
        uint32_t nb = ncols / (BLK);                                          \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride + woff;         \
        for (uint32_t t = threadIdx.x; t < ncols * R; t += blockDim.x) {      \
            uint32_t rr = t / ncols, i = t - rr * ncols;                      \
            uint32_t row = r0 + rr < nrows ? r0 + rr : r0;                    \
            rowv[t] = deq_elem(wp + (size_t)row * nb * (BBYTES) +             \
                               (size_t)(i / (BLK)) * (BBYTES), i % (BLK));    \
        }                                                                     \
        __syncthreads();                                                       \
        uint32_t lane = threadIdx.x & 31u, nw = blockDim.x >> 5;              \
        uint32_t a0 = eb[e];                                                  \
        for (uint32_t k = threadIdx.x >> 5; k < K; k += nw) {                 \
            const float *xt =                                                 \
                x + (size_t)(atok ? atok[a0 + k] : (int32_t)(a0 + k)) *       \
                    ncols;                                                    \
            float acc[R];                                                     \
            for (uint32_t r = 0; r < R; r++) acc[r] = 0.f;                    \
            for (uint32_t i = lane; i < ncols; i += 32) {                     \
                float xv = xt[i];                                             \
                for (uint32_t r = 0; r < R; r++)                              \
                    acc[r] += rowv[r * ncols + i] * xv;                       \
            }                                                                 \
            for (uint32_t r = 0; r < R; r++) {                                \
                float a = acc[r];                                             \
                for (uint32_t o = 16; o; o >>= 1)                             \
                    a += __shfl_down(a, o);                                   \
                if (lane == 0 && r0 + r < nrows)                              \
                    out[(size_t)(a0 + k) * nrows + r0 + r] = a * scale;       \
            }                                                                 \
        }                                                                      \
    }

Q4_MOE_PF(moepf_iq2xxs_k, deq_iq2xxs_elem, 256u, 66u)
Q4_MOE_PF(moepf_iq2xs_k, deq_iq2xs_elem, 256u, 74u)
Q4_MOE_PF(moepf_iq2_k, deq_iq2s_elem, 256u, 82u)
Q4_MOE_PF(moepf_iq3_k, deq_iq3s_elem, 256u, 110u)
Q4_MOE_PF(moepf_iq4_k, deq_iq4nl_elem, 32u, 18u)
Q4_MOE_PF(moepf_iq3xxs_k, deq_iq3xxs_elem, 256u, 98u)
Q4_MOE_PF(moepf_iq4xs_k, deq_iq4xs_elem, 256u, 136u)
Q4_MOE_PF(moepf_q40_k, deq_q40_elem, 32u, 18u)
Q4_MOE_PF(moepf_q50_k, deq_q50_elem, 32u, 22u)
Q4_MOE_PF(moepf_q20_k, deq_q20_elem, 64u, 18u)

/* Fused gate+up variant: same shape/type, same token list — one smem pair of
 * rows (gate row r, up row r), one launch, half the x traffic. */
#define Q4_MOE_PF2(name, deq_elem, BLK, BBYTES)                                \
    __global__ static void name(float *out0, float *out1,                      \
                                const uint8_t *wbase, uint64_t stride,         \
                                uint64_t woff0, uint64_t woff1,                \
                                const int32_t *slots, const uint32_t *eb,      \
                                const uint32_t *ec, const int32_t *atok,       \
                                const float *x, uint32_t nrows,                \
                                uint32_t ncols, float scale) {                 \
        uint32_t row = blockIdx.x, e = blockIdx.y;                            \
        uint32_t K = ec[e];                                                   \
        if (!K) return;                                                       \
        extern __shared__ float rowv[]; /* [0]=gate row, [1]=up row */        \
        uint32_t nb = ncols / (BLK);                                          \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride;                \
        for (uint32_t i = threadIdx.x; i < ncols; i += blockDim.x) {          \
            uint32_t bl = i / (BLK), rr = i % (BLK);                          \
            rowv[i] = deq_elem(wp + woff0 + (size_t)row * nb * (BBYTES) +     \
                               (size_t)bl * (BBYTES), rr);                    \
            rowv[ncols + i] =                                                 \
                deq_elem(wp + woff1 + (size_t)row * nb * (BBYTES) +           \
                         (size_t)bl * (BBYTES), rr);                          \
        }                                                                     \
        __syncthreads();                                                       \
        uint32_t lane = threadIdx.x & 31u, nw = blockDim.x >> 5;              \
        uint32_t a0 = eb[e];                                                  \
        for (uint32_t k = threadIdx.x >> 5; k < K; k += nw) {                 \
            const float *xt =                                                 \
                x + (size_t)(atok ? atok[a0 + k] : (int32_t)(a0 + k)) *       \
                    ncols;                                                    \
            float acc = 0.f, acc1 = 0.f;                                      \
            for (uint32_t i = lane; i < ncols; i += 32) {                     \
                float xv = xt[i];                                             \
                acc += rowv[i] * xv;                                          \
                acc1 += rowv[ncols + i] * xv;                                 \
            }                                                                 \
            for (uint32_t o = 16; o; o >>= 1) {                               \
                acc += __shfl_down(acc, o);                                   \
                acc1 += __shfl_down(acc1, o);                                 \
            }                                                                 \
            if (lane == 0) {                                                  \
                out0[(size_t)(a0 + k) * nrows + row] = acc * scale;           \
                out1[(size_t)(a0 + k) * nrows + row] = acc1 * scale;          \
            }                                                                 \
        }                                                                      \
    }

Q4_MOE_PF2(moepf2_iq2xxs_k, deq_iq2xxs_elem, 256u, 66u)
Q4_MOE_PF2(moepf2_iq2xs_k, deq_iq2xs_elem, 256u, 74u)
Q4_MOE_PF2(moepf2_iq2_k, deq_iq2s_elem, 256u, 82u)
Q4_MOE_PF2(moepf2_iq3_k, deq_iq3s_elem, 256u, 110u)
Q4_MOE_PF2(moepf2_iq4_k, deq_iq4nl_elem, 32u, 18u)
Q4_MOE_PF2(moepf2_iq3xxs_k, deq_iq3xxs_elem, 256u, 98u)
Q4_MOE_PF2(moepf2_iq4xs_k, deq_iq4xs_elem, 256u, 136u)
Q4_MOE_PF2(moepf2_q40_k, deq_q40_elem, 32u, 18u)
Q4_MOE_PF2(moepf2_q50_k, deq_q50_elem, 32u, 22u)
Q4_MOE_PF2(moepf2_q20_k, deq_q20_elem, 64u, 18u)

/* RDNA3 wave32 WMMA (16x16x16 f16 in, f32 acc): lane holds one A row
 * (all 16 k), one B column (all 16 k), and output column lane%16 of rows
 * 2i+lane/16 of its wave's tile (verified empirically).  Block = 4 waves x
 * 32 = 128 threads covering a 64-row tile; up to 64 tokens per pass in four
 * 16-column WMMA tiles.  Weight tiles are dequantized into LDS as f16 once
 * per token pass; X is staged to f16 LDS per K-chunk. */
typedef _Float16 wmma_a16 __attribute__((ext_vector_type(16)));
typedef float wmma_c8 __attribute__((ext_vector_type(8)));
#define MOEW_ROWS 64
#define MOEW_TOKS 64
#define MOEW_KC 64
#define MOEW_PAD 2

/* Fused gate+up + silu(g)*u: writes act output to out[a0+t][row]. */
#define Q4_MOEW2(name, deq_elem, deq32, BLK, BBYTES)                                 \
    __global__ static void name(float *out, const uint8_t *wbase,             \
            uint64_t stride, uint64_t woff0, uint64_t woff1,                  \
            const int32_t *slots, const uint32_t *eb, const uint32_t *ec,     \
            const int32_t *atok, const float *x, uint32_t nrows,              \
            uint32_t ncols) {                                                \
        const uint32_t r0 = blockIdx.x * MOEW_ROWS, e = blockIdx.y;           \
        const uint32_t K = ec[e], a0 = eb[e];                                 \
        if (!K) return;                                                      \
        __shared__ __half As[2][MOEW_ROWS][MOEW_KC + MOEW_PAD];               \
        __shared__ __half Xs[MOEW_KC][MOEW_TOKS + MOEW_PAD];                  \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride;                \
        const uint32_t nb = ncols / (BLK);                                    \
        const int wave = threadIdx.x >> 5, l = threadIdx.x & 31;              \
        for (uint32_t t0 = 0; t0 < K; t0 += MOEW_TOKS) {                      \
            const uint32_t tn = K - t0 < MOEW_TOKS ? K - t0 : MOEW_TOKS;      \
            wmma_c8 ag[MOEW_TOKS / 16], au[MOEW_TOKS / 16];                   \
            for (uint32_t nt = 0; nt < MOEW_TOKS / 16; nt++) {                \
                ag[nt] = (wmma_c8){};                                         \
                au[nt] = (wmma_c8){};                                         \
            }                                                                 \
            for (uint32_t k0 = 0; k0 < ncols; k0 += MOEW_KC) {                \
                const uint32_t kn = ncols - k0 < MOEW_KC ? ncols - k0         \
                                                         : MOEW_KC;           \
                for (uint32_t i = threadIdx.x;                                \
                     i < 2u * MOEW_ROWS * (MOEW_KC / 32u);                    \
                     i += blockDim.x) {                                       \
                    const uint32_t t = i / (MOEW_ROWS * (MOEW_KC / 32u));     \
                    const uint32_t rr = i % (MOEW_ROWS * (MOEW_KC / 32u));    \
                    const uint32_t r = rr / (MOEW_KC / 32u);                  \
                    const uint32_t sb = rr % (MOEW_KC / 32u);                 \
                    Q4_DEQ_TILE(As[t], wp + (t ? woff1 : woff0), r0, r, k0,   \
                                sb, kn, nrows, nb, BLK, BBYTES, deq_elem,     \
                                deq32);                                       \
                }                                                             \
                for (uint32_t i = threadIdx.x; i < MOEW_KC * MOEW_TOKS;       \
                     i += blockDim.x) {                                       \
                    uint32_t tt = i / MOEW_KC, kc = i % MOEW_KC;              \
                    float v = 0.f;                                            \
                    if (kc < kn && tt < tn) {                                 \
                        int32_t tok =                                         \
                            atok ? atok[a0 + t0 + tt]                         \
                                 : (int32_t)(a0 + t0 + tt);                   \
                        v = x[(size_t)tok * ncols + k0 + kc];                 \
                    }                                                         \
                    Xs[kc][tt] = __float2half(v);                             \
                }                                                             \
                __syncthreads();                                              \
                for (uint32_t ks = 0; ks < kn; ks += 16) {                    \
                    wmma_a16 ag16, au16;                                      \
                    const uint32_t rw = wave * 16 + (l & 15);                 \
                    for (uint32_t i = 0; i < 16; i++) {                       \
                        ag16[i] = __half2float(As[0][rw][ks + i]);            \
                        au16[i] = __half2float(As[1][rw][ks + i]);            \
                    }                                                         \
                    for (uint32_t nt = 0; nt * 16 < tn; nt++) {               \
                        wmma_a16 b16;                                         \
                        for (uint32_t i = 0; i < 16; i++)                     \
                            b16[i] = __half2float(                            \
                                Xs[ks + i][nt * 16 + (l & 15)]);              \
                        ag[nt] =                                              \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                                ag16, b16, ag[nt]);                           \
                        au[nt] =                                              \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                                au16, b16, au[nt]);                           \
                    }                                                         \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            for (uint32_t nt = 0; nt * 16 < tn; nt++)                         \
                for (uint32_t i = 0; i < 8; i++) {                            \
                    uint32_t r = r0 + wave * 16 + 2 * i + (l >> 4);           \
                    uint32_t tt = nt * 16 + (l & 15);                         \
                    if (r < nrows && tt < tn) {                               \
                        float g = ag[nt][i], u = au[nt][i];                   \
                        out[(size_t)(a0 + t0 + tt) * nrows + r] =             \
                            g / (1.f + expf(-g)) * u;                         \
                    }                                                         \
                }                                                             \
        }                                                                     \
    }

/* Single tensor (down): plain GEMM, out[a0+t][row] = W . x[tok]. */
#define Q4_MOEW(name, deq_elem, deq32, BLK, BBYTES)                                  \
    __global__ static void name(float *out, const uint8_t *wbase,             \
            uint64_t stride, uint64_t woff,                                   \
            const int32_t *slots, const uint32_t *eb, const uint32_t *ec,     \
            const int32_t *atok, const float *x, uint32_t nrows,              \
            uint32_t ncols) {                                                \
        const uint32_t r0 = blockIdx.x * MOEW_ROWS, e = blockIdx.y;           \
        const uint32_t K = ec[e], a0 = eb[e];                                 \
        if (!K) return;                                                      \
        __shared__ __half As[MOEW_ROWS][MOEW_KC + MOEW_PAD];                  \
        __shared__ __half Xs[MOEW_KC][MOEW_TOKS + MOEW_PAD];                  \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride + woff;         \
        const uint32_t nb = ncols / (BLK);                                    \
        const int wave = threadIdx.x >> 5, l = threadIdx.x & 31;              \
        for (uint32_t t0 = 0; t0 < K; t0 += MOEW_TOKS) {                      \
            const uint32_t tn = K - t0 < MOEW_TOKS ? K - t0 : MOEW_TOKS;      \
            wmma_c8 acc[MOEW_TOKS / 16];                                      \
            for (uint32_t nt = 0; nt < MOEW_TOKS / 16; nt++)                  \
                acc[nt] = (wmma_c8){};                                        \
            for (uint32_t k0 = 0; k0 < ncols; k0 += MOEW_KC) {                \
                const uint32_t kn = ncols - k0 < MOEW_KC ? ncols - k0         \
                                                         : MOEW_KC;           \
                for (uint32_t i = threadIdx.x;                                \
                     i < MOEW_ROWS * (MOEW_KC / 32u); i += blockDim.x) {       \
                    const uint32_t r = i / (MOEW_KC / 32u);                   \
                    const uint32_t sb = i % (MOEW_KC / 32u);                  \
                    Q4_DEQ_TILE(As, wp, r0, r, k0, sb, kn, nrows, nb, BLK,    \
                                BBYTES, deq_elem, deq32);                     \
                }                                                             \
                for (uint32_t i = threadIdx.x; i < MOEW_KC * MOEW_TOKS;       \
                     i += blockDim.x) {                                       \
                    uint32_t tt = i / MOEW_KC, kc = i % MOEW_KC;              \
                    float v = 0.f;                                            \
                    if (kc < kn && tt < tn) {                                 \
                        int32_t tok =                                         \
                            atok ? atok[a0 + t0 + tt]                         \
                                 : (int32_t)(a0 + t0 + tt);                   \
                        v = x[(size_t)tok * ncols + k0 + kc];                 \
                    }                                                         \
                    Xs[kc][tt] = __float2half(v);                             \
                }                                                             \
                __syncthreads();                                              \
                for (uint32_t ks = 0; ks < kn; ks += 16) {                    \
                    wmma_a16 a16;                                             \
                    const uint32_t rw = wave * 16 + (l & 15);                 \
                    for (uint32_t i = 0; i < 16; i++)                         \
                        a16[i] = __half2float(As[rw][ks + i]);                \
                    for (uint32_t nt = 0; nt * 16 < tn; nt++) {               \
                        wmma_a16 b16;                                         \
                        for (uint32_t i = 0; i < 16; i++)                     \
                            b16[i] = __half2float(                            \
                                Xs[ks + i][nt * 16 + (l & 15)]);              \
                        acc[nt] =                                             \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                                a16, b16, acc[nt]);                           \
                    }                                                         \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            for (uint32_t nt = 0; nt * 16 < tn; nt++)                         \
                for (uint32_t i = 0; i < 8; i++) {                            \
                    uint32_t r = r0 + wave * 16 + 2 * i + (l >> 4);           \
                    uint32_t tt = nt * 16 + (l & 15);                         \
                    if (r < nrows && tt < tn)                                 \
                        out[(size_t)(a0 + t0 + tt) * nrows + r] =             \
                            acc[nt][i];                                       \
                }                                                             \
        }                                                                     \
    }

/* Dense many-token GEMM on WMMA: same 64x64x64 tiling as the expert
 * kernels (128-token tiles regressed: 8 c8 accumulators spill). BLK/BBYTES
 * are the quant superblock's value/byte size. */
#define Q8W_TOKS 64u
#define Q4_DGEMW(name, BLK, BBYTES, deq_elem, deq32)                         \
__global__ static void name(float *out, const uint8_t *w, const float *x,   \
                                  uint32_t in_dim, uint32_t out_dim,        \
                                  uint32_t n_batch, float scale) {          \
    const uint32_t r0 = blockIdx.x * MOEW_ROWS, t0 = blockIdx.y * Q8W_TOKS; \
    __shared__ __half As[MOEW_ROWS][MOEW_KC + MOEW_PAD];                    \
    __shared__ __half Xs[MOEW_KC][Q8W_TOKS + MOEW_PAD];                     \
    const uint32_t nb = in_dim / (BLK);                                     \
    const int wave = threadIdx.x >> 5, l = threadIdx.x & 31;                \
    const uint32_t tn = n_batch - t0 < Q8W_TOKS ? n_batch - t0 : Q8W_TOKS;  \
    wmma_c8 acc[Q8W_TOKS / 16];                                             \
    for (uint32_t nt = 0; nt < Q8W_TOKS / 16; nt++) acc[nt] = (wmma_c8){};  \
    for (uint32_t k0 = 0; k0 < in_dim; k0 += MOEW_KC) {                     \
        const uint32_t kn = in_dim - k0 < MOEW_KC ? in_dim - k0 : MOEW_KC;  \
        for (uint32_t i = threadIdx.x; i < MOEW_ROWS * (MOEW_KC / 32u);     \
             i += blockDim.x) {                                             \
            const uint32_t r = i / (MOEW_KC / 32u);                         \
            const uint32_t sb = i % (MOEW_KC / 32u);                        \
            Q4_DEQ_TILE(As, w, r0, r, k0, sb, kn, out_dim, nb, BLK,         \
                        BBYTES, deq_elem, deq32);                           \
        }                                                                 \
        for (uint32_t i = threadIdx.x; i < MOEW_KC * Q8W_TOKS;              \
             i += blockDim.x) {                                             \
            uint32_t tt = i / MOEW_KC, kc = i % MOEW_KC;                    \
            float v = 0.f;                                                  \
            if (kc < kn && tt < tn)                                         \
                v = x[(size_t)(t0 + tt) * in_dim + k0 + kc];                \
            Xs[kc][tt] = __float2half(v);                                   \
        }                                                                 \
        __syncthreads();                                                    \
        for (uint32_t ks = 0; ks < kn; ks += 16) {                          \
            wmma_a16 a16;                                                   \
            const uint32_t rw = wave * 16 + (l & 15);                       \
            for (uint32_t i = 0; i < 16; i++)                               \
                a16[i] = __half2float(As[rw][ks + i]);                      \
            for (uint32_t nt = 0; nt * 16 < tn; nt++) {                     \
                wmma_a16 b16;                                               \
                for (uint32_t i = 0; i < 16; i++)                           \
                    b16[i] = __half2float(Xs[ks + i][nt * 16 + (l & 15)]);  \
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                    a16, b16, acc[nt]);                                     \
            }                                                             \
        }                                                                 \
        __syncthreads();                                                    \
    }                                                                     \
    for (uint32_t nt = 0; nt * 16 < tn; nt++)                               \
        for (uint32_t i = 0; i < 8; i++) {                                  \
            uint32_t r = r0 + wave * 16 + 2 * i + (l >> 4);                 \
            uint32_t tt = nt * 16 + (l & 15);                               \
            if (r < out_dim && tt < tn)                                     \
                out[(size_t)(t0 + tt) * out_dim + r] = acc[nt][i] * scale;  \
        }                                                                 \
}

Q4_DGEMW(gemm_q8w_k, 32u, 34u, deq_q8_0_elem, deq_q8_0_x32)
Q4_DGEMW(gemm_q6w_k, 256u, 210u, deq_q6_k_elem, deq_q6_k_x32)
Q4_DGEMW(gemm_q4kw_k, 256u, 144u, deq_q4k_elem, deq_q4k_x32)
Q4_DGEMW(gemm_q5kw_k, 256u, 176u, deq_q5k_elem, deq_q5k_x32)
Q4_DGEMW(gemm_q40w_k, 32u, 18u, deq_q40_elem, deq_q40_x32)
Q4_DGEMW(gemm_q50w_k, 32u, 22u, deq_q50_elem, deq_q50_x32)
Q4_DGEMW(gemm_q20w_k, 64u, 18u, deq_q20_elem, deq_q20_x32)
Q4_DGEMW(gemm_iq4xsw_k, 256u, 136u, deq_iq4xs_elem, deq_iq4xs_x32)

/* BF16 dense batched GEMM on tensor cores: same MOEW tiling as Q4_DGEMW but
 * weights arrive already-bf16 (no dequant); x converts f32->bf16 with
 * round-to-nearest-even — numerics class matches the usual bf16 GEMM
 * engines rather than the fp32 tiled path it replaces.  Off by default:
 * measured slower than gemm_tiled_k on the target model (see dispatch).
 * Q4_GEMM_BF16W=1 opts in. */
typedef short wmma_b16 __attribute__((ext_vector_type(16)));

static __device__ __forceinline__ uint16_t f32_to_bf16(float v) {
    uint32_t u = __float_as_uint(v);
    /* round-to-nearest-even on the dropped low 16 bits */
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__global__ static void gemm_wmma_bf16_k(float *out, const uint16_t *w,
                                      const float *x, uint32_t in_dim,
                                      uint32_t out_dim, float scale,
                                      uint32_t n_batch, const int32_t *which) {
    const uint32_t r0 = blockIdx.x * MOEW_ROWS, t0 = blockIdx.y * Q8W_TOKS;
    __shared__ uint16_t As[MOEW_ROWS][MOEW_KC + MOEW_PAD];
    __shared__ uint16_t Xs[MOEW_KC][Q8W_TOKS + MOEW_PAD];
    const int wave = threadIdx.x >> 5, l = threadIdx.x & 31;
    const uint32_t tn = n_batch - t0 < Q8W_TOKS ? n_batch - t0 : Q8W_TOKS;
    wmma_c8 acc[Q8W_TOKS / 16];
    for (uint32_t nt = 0; nt < Q8W_TOKS / 16; nt++) acc[nt] = (wmma_c8){};
    for (uint32_t k0 = 0; k0 < in_dim; k0 += MOEW_KC) {
        const uint32_t kn = in_dim - k0 < MOEW_KC ? in_dim - k0 : MOEW_KC;
        for (uint32_t i = threadIdx.x; i < MOEW_ROWS * MOEW_KC;
             i += blockDim.x) {
            const uint32_t r = i / MOEW_KC, kc = i % MOEW_KC;
            As[r][kc] = (r0 + r < out_dim && kc < kn)
                            ? w[(size_t)(r0 + r) * in_dim + k0 + kc]
                            : (uint16_t)0;
        }
        for (uint32_t i = threadIdx.x; i < MOEW_KC * Q8W_TOKS;
             i += blockDim.x) {
            uint32_t tt = i / MOEW_KC, kc = i % MOEW_KC;
            float v = 0.f;
            if (kc < kn && tt < tn)
                v = x[(size_t)(which ? which[t0 + tt] : (int32_t)(t0 + tt)) *
                             in_dim + k0 + kc];
            Xs[kc][tt] = f32_to_bf16(v);
        }
        __syncthreads();
        for (uint32_t ks = 0; ks < kn; ks += 16) {
            wmma_b16 a16;
            const uint32_t rw = wave * 16 + (l & 15);
            for (uint32_t i = 0; i < 16; i++)
                a16[i] = (short)As[rw][ks + i];
            for (uint32_t nt = 0; nt * 16 < tn; nt++) {
                wmma_b16 b16;
                for (uint32_t i = 0; i < 16; i++)
                    b16[i] = (short)Xs[ks + i][nt * 16 + (l & 15)];
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    a16, b16, acc[nt]);
            }
        }
        __syncthreads();
    }
    for (uint32_t nt = 0; nt * 16 < tn; nt++)
        for (uint32_t i = 0; i < 8; i++) {
            uint32_t r = r0 + wave * 16 + 2 * i + (l >> 4);
            uint32_t tt = nt * 16 + (l & 15);
            if (r < out_dim && tt < tn)
                out[(size_t)(t0 + tt) * out_dim + r] = acc[nt][i] * scale;
        }
}


Q4_MOEW2(moew2_iq2xxs_k, deq_iq2xxs_elem, deq_iq2xxs_x32, 256u, 66u)
Q4_MOEW2(moew2_iq2xs_k, deq_iq2xs_elem, deq_iq2xs_x32, 256u, 74u)
Q4_MOEW2(moew2_iq2_k, deq_iq2s_elem, deq_iq2s_x32, 256u, 82u)
Q4_MOEW2(moew2_iq3_k, deq_iq3s_elem, deq_iq3s_x32, 256u, 110u)
Q4_MOEW2(moew2_iq4_k, deq_iq4nl_elem, deq_iq4nl_x32, 32u, 18u)
Q4_MOEW2(moew2_iq3xxs_k, deq_iq3xxs_elem, deq_iq3xxs_x32, 256u, 98u)
Q4_MOEW2(moew2_iq4xs_k, deq_iq4xs_elem, deq_iq4xs_x32, 256u, 136u)
Q4_MOEW2(moew2_q40_k, deq_q40_elem, deq_q40_x32, 32u, 18u)
Q4_MOEW2(moew2_q50_k, deq_q50_elem, deq_q50_x32, 32u, 22u)
Q4_MOEW2(moew2_q20_k, deq_q20_elem, deq_q20_x32, 64u, 18u)
Q4_MOEW(moew_iq2xxs_k, deq_iq2xxs_elem, deq_iq2xxs_x32, 256u, 66u)
Q4_MOEW(moew_iq2xs_k, deq_iq2xs_elem, deq_iq2xs_x32, 256u, 74u)
Q4_MOEW(moew_iq2_k, deq_iq2s_elem, deq_iq2s_x32, 256u, 82u)
Q4_MOEW(moew_iq3_k, deq_iq3s_elem, deq_iq3s_x32, 256u, 110u)
Q4_MOEW(moew_iq4_k, deq_iq4nl_elem, deq_iq4nl_x32, 32u, 18u)
Q4_MOEW(moew_iq3xxs_k, deq_iq3xxs_elem, deq_iq3xxs_x32, 256u, 98u)
Q4_MOEW(moew_iq4xs_k, deq_iq4xs_elem, deq_iq4xs_x32, 256u, 136u)
Q4_MOEW(moew_q40_k, deq_q40_elem, deq_q40_x32, 32u, 18u)
Q4_MOEW(moew_q50_k, deq_q50_elem, deq_q50_x32, 32u, 22u)
Q4_MOEW(moew_q20_k, deq_q20_elem, deq_q20_x32, 64u, 18u)

/* Q4_GEMV_PROF=1: per-type call/byte counters. Free (no events), works inside
 * graph capture too since it is pure host-side accounting.
 * Q4_GEMV_PROF=2: additionally bracket each launch with stream events and
 * accumulate per-type GPU intervals at report time. Events are recorded on
 * g_str so no host sync is needed; intervals include in-stream gaps.
 * Both are phase-tagged (q4_hip_gemv_prof_phase): the MTP cycle splits a
 * decode round into draft / verify / logits / commit / repair so a weight
 * read can be charged to the phase that caused it. */

extern "C" void q4_hip_gemv_prof_reset(void) {
    memset(g_gemv_calls, 0, sizeof(g_gemv_calls));
    memset(g_gemv_bytes, 0, sizeof(g_gemv_bytes));
    memset(g_gemv_ms, 0, sizeof(g_gemv_ms));
    g_gev_n = 0;
}

/* Charge every counted launch from here on to phase `p` (see QG_PH_*). */
extern "C" void q4_hip_gemv_prof_phase(int p) {
    if (p < 0) p = QG_PH_NONE;
    if (p >= QG_PHASES) p = QG_PHASES - 1;
    g_gemv_phase = p;
}

/* Copy the per-type call/byte counters out (no reset, no GPU sync). */
extern "C" void q4_hip_gemv_prof_snap(uint64_t *calls, uint64_t *bytes) {
    for (int p = 0; p < QG_PHASES; p++)
        for (int t = 0; t < 64; t++) {
            if (calls) calls[p * 64 + t] = g_gemv_calls[p][t];
            if (bytes) bytes[p * 64 + t] = g_gemv_bytes[p][t];
        }
}

extern "C" void q4_hip_gemv_prof_report(void) {
    const char *tn[64] = {0};
    tn[Q4_T_F32] = "f32"; tn[Q4_T_F16] = "f16"; tn[Q4_T_Q5_1] = "q5_1";
    tn[Q4_T_Q8_0] = "q8_0"; tn[Q4_T_Q4_K] = "q4_k"; tn[Q4_T_Q5_K] = "q5_k";
    tn[Q4_T_Q6_K] = "q6_k"; tn[Q4_T_IQ4_NL] = "iq4_nl";
    tn[Q4_T_IQ3_S] = "iq3_s"; tn[Q4_T_IQ2_S] = "iq2_s"; tn[Q4_T_BF16] = "bf16";
    tn[Q4_T_IQ2_XXS] = "iq2_xxs";
    tn[Q4_T_IQ2_XS] = "iq2_xs";
    tn[Q4_T_IQ3_XXS] = "iq3_xxs"; tn[Q4_T_IQ4_XS] = "iq4_xs";
    tn[Q4_T_Q4_0] = "q4_0"; tn[Q4_T_Q5_0] = "q5_0"; tn[Q4_T_Q2_0] = "q2_0";
    /* Resolve recorded event pairs into per-type GPU ms (drain-free: events
     * already complete by report time). */
    if (g_gev_n) {
        (void)hipDeviceSynchronize();
        for (uint32_t i = 0; i < g_gev_n; i++) {
            float ms = 0.f;
            if (hipEventElapsedTime(&ms, g_gev_a[i], g_gev_b[i]) == hipSuccess)
                g_gemv_ms[g_gev_ty[i] >> 6][g_gev_ty[i] & 63] += ms;
        }
    }
    for (int p = 0; p < QG_PHASES; p++) {
        for (uint32_t t = 0; t < 64; t++) {
            if (!g_gemv_calls[p][t]) continue;
            const char *nm = tn[t] ? tn[t] : "?";
            fprintf(stderr,
                    "gemv-prof %-7s %-8s calls %llu  bytes %.2f GiB  gpu %.0f ms\n",
                    qg_ph_name[p], nm,
                    (unsigned long long)g_gemv_calls[p][t],
                    (double)g_gemv_bytes[p][t] / (1 << 30), g_gemv_ms[p][t]);
        }
    }
}

extern "C" bool q4_hip_moe_pf2(uint32_t type, const uint8_t *wbase,
                               uint64_t stride, uint64_t woff0, uint64_t woff1,
                               const int32_t *d_slots, const uint32_t *d_eb,
                               const uint32_t *d_ec, const int32_t *d_atok,
                               const float *d_x, float *d_out0, float *d_out1,
                               uint32_t nrows, uint32_t ncols, uint32_t ne) {
    if (!g_ok || !wbase || !d_slots || !d_eb || !d_ec || !d_x || !d_out0 ||
        !d_out1 || !ne || !nrows || !ncols || ncols > 4096u ||
        ncols * 2u * sizeof(float) > 48u * 1024u)
        return false;
    size_t shmem = (size_t)ncols * 2u * sizeof(float);
    dim3 grid(nrows, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[g_gemv_phase][type] += 2 * ne;
        g_gemv_bytes[g_gemv_phase][type] += rb * nrows * 2 * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_XXS:
        if (ncols % 256u) return false;
        moepf2_iq2xxs_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase,
                                                    stride, woff0, woff1,
                                                    d_slots, d_eb, d_ec,
                                                    d_atok, d_x, nrows, ncols,
                                                    1.f);
        break;
    case Q4_T_IQ2_XS:
        if (ncols % 256u) return false;
        moepf2_iq2xs_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase,
                                                   stride, woff0, woff1,
                                                   d_slots, d_eb, d_ec,
                                                   d_atok, d_x, nrows, ncols,
                                                   1.f);
        break;
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moepf2_iq2_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moepf2_iq3_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moepf2_iq3xxs_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase,
                                                    stride, woff0, woff1,
                                                    d_slots, d_eb, d_ec,
                                                    d_atok, d_x, nrows, ncols,
                                                    1.f);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moepf2_iq4xs_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase,
                                                   stride, woff0, woff1,
                                                   d_slots, d_eb, d_ec,
                                                   d_atok, d_x, nrows, ncols,
                                                   1.f);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moepf2_iq4_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moepf2_q40_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moepf2_q50_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moepf2_q20_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_moe_pf(uint32_t type, const uint8_t *wbase,
                              uint64_t stride, uint64_t woff,
                              const int32_t *d_slots, const uint32_t *d_eb,
                              const uint32_t *d_ec, const int32_t *d_atok,
                              const float *d_x, float *d_out, uint32_t nrows,
                              uint32_t ncols, uint32_t ne) {
    if (!g_ok || !wbase || !d_slots || !d_eb || !d_ec || !d_x || !d_out ||
        !ne || !nrows || !ncols || ncols > 4096u ||
        ncols * 2u * sizeof(float) > 48u * 1024u)
        return false;
    size_t shmem = (size_t)ncols * 2u * sizeof(float);
    dim3 grid((nrows + 1u) / 2u, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[g_gemv_phase][type] += ne;
        g_gemv_bytes[g_gemv_phase][type] += rb * nrows * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_XXS:
        if (ncols % 256u) return false;
        moepf_iq2xxs_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                                   d_slots, d_eb, d_ec,
                                                   d_atok, d_x, nrows, ncols,
                                                   1.f);
        break;
    case Q4_T_IQ2_XS:
        if (ncols % 256u) return false;
        moepf_iq2xs_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                                  d_slots, d_eb, d_ec,
                                                  d_atok, d_x, nrows, ncols,
                                                  1.f);
        break;
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moepf_iq2_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moepf_iq3_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moepf_iq3xxs_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                                   d_slots, d_eb, d_ec,
                                                   d_atok, d_x, nrows, ncols,
                                                   1.f);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moepf_iq4xs_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                                  d_slots, d_eb, d_ec,
                                                  d_atok, d_x, nrows, ncols,
                                                  1.f);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moepf_iq4_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moepf_q40_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moepf_q50_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moepf_q20_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}


static int wmma_off(void) {
    static int v = -1;
    if (v < 0)
        v = getenv("Q4_WMMA") && getenv("Q4_WMMA")[0] == '0';
    return v;
}

/* Fused gate+up+silu WMMA prefill path; returns false for unsupported types
 * or when Q4_WMMA=0 so the caller falls back to q4_hip_moe_pf2 + moe_act. */
extern "C" bool q4_hip_moe_wmma2(uint32_t type, const uint8_t *wbase,
                                 uint64_t stride, uint64_t woff0,
                                 uint64_t woff1, const int32_t *d_slots,
                                 const uint32_t *d_eb, const uint32_t *d_ec,
                                 const int32_t *d_atok, const float *d_x,
                                 float *d_out, uint32_t nrows, uint32_t ncols,
                                 uint32_t ne) {
    if (!g_ok || wmma_off() || !wbase || !d_slots || !d_eb || !d_ec || !d_x ||
        !d_out || !ne || !nrows || !ncols)
        return false;
    dim3 grid((nrows + MOEW_ROWS - 1) / MOEW_ROWS, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[g_gemv_phase][type] += 2 * ne;
        g_gemv_bytes[g_gemv_phase][type] += rb * nrows * 2 * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_XXS:
        if (ncols % 256u) return false;
        moew2_iq2xxs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                                woff1, d_slots, d_eb, d_ec,
                                                d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ2_XS:
        if (ncols % 256u) return false;
        moew2_iq2xs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                               woff1, d_slots, d_eb, d_ec,
                                               d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moew2_iq2_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                             woff1, d_slots, d_eb, d_ec,
                                             d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moew2_iq3_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                             woff1, d_slots, d_eb, d_ec,
                                             d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moew2_iq3xxs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                                woff1, d_slots, d_eb, d_ec,
                                                d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moew2_iq4xs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                               woff1, d_slots, d_eb, d_ec,
                                               d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moew2_iq4_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                             woff1, d_slots, d_eb, d_ec,
                                             d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moew2_q40_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                            woff1, d_slots, d_eb, d_ec,
                                            d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moew2_q50_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                            woff1, d_slots, d_eb, d_ec,
                                            d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moew2_q20_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                            woff1, d_slots, d_eb, d_ec,
                                            d_atok, d_x, nrows, ncols);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_moe_wmma(uint32_t type, const uint8_t *wbase,
                                uint64_t stride, uint64_t woff,
                                const int32_t *d_slots, const uint32_t *d_eb,
                                const uint32_t *d_ec, const int32_t *d_atok,
                                const float *d_x, float *d_out,
                                uint32_t nrows, uint32_t ncols, uint32_t ne) {
    if (!g_ok || wmma_off() || !wbase || !d_slots || !d_eb || !d_ec || !d_x ||
        !d_out || !ne || !nrows || !ncols)
        return false;
    dim3 grid((nrows + MOEW_ROWS - 1) / MOEW_ROWS, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[g_gemv_phase][type] += ne;
        g_gemv_bytes[g_gemv_phase][type] += rb * nrows * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_XXS:
        if (ncols % 256u) return false;
        moew_iq2xxs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols);
        break;
    case Q4_T_IQ2_XS:
        if (ncols % 256u) return false;
        moew_iq2xs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                              d_slots, d_eb, d_ec, d_atok,
                                              d_x, nrows, ncols);
        break;
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moew_iq2_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moew_iq3_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moew_iq3xxs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moew_iq4xs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                              d_slots, d_eb, d_ec, d_atok,
                                              d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moew_iq4_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moew_q40_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moew_q50_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moew_q20_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                           d_slots, d_eb, d_ec, d_atok, d_x,
                                           nrows, ncols);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}

/* silu(g)*u in place on g, over n contiguous floats — same math as
 * q4_hip_silu + q4_hip_mul but one launch. */
extern "C" bool q4_hip_moe_act(float *d_g, const float *d_u, uint64_t n) {
    if (!g_ok || !d_g || !d_u || !n) return false;
    moe_act_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(d_g, d_u,
                                                                  n);
    return hipGetLastError() == hipSuccess;
}

/* ---- device argmax + max-prob (temp-0 draft sampling) ------------------
 * Two kernels, wave32-safe (no wave64 assumptions): block partials
 * (max, first-index argmax, sum exp(x-blockmax)) then a one-block combine.
 * Tie-break everywhere: the smaller element index wins, matching the host
 * argmax() walk exactly. */

#define AMX_MAXB 1024u
typedef struct {
    float mx;
    uint32_t am;
    float se;
} amx_part_t;
static amx_part_t *g_amx_part; /* device */
static uint32_t *g_amx_h, *g_amx_d; /* mapped host [0]=id [1]=f32-bits prob */

static bool amx_bufs(void) {
    if (g_amx_part) return true;
    if (hipMalloc((void **)&g_amx_part, AMX_MAXB * sizeof(amx_part_t)) !=
        hipSuccess)
        return false;
    if (hipHostMalloc((void **)&g_amx_h, 2 * sizeof(uint32_t),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer((void **)&g_amx_d, g_amx_h, 0) !=
            hipSuccess) {
        (void)hipFree(g_amx_part);
        g_amx_part = nullptr;
        return false;
    }
    return true;
}

/* Pairwise (max, first-index) combine used by both reduction stages. */
__device__ static inline void amx_pick(float vm, uint32_t va, float *mx,
                                       uint32_t *am) {
    if (vm > *mx || (vm == *mx && va < *am)) {
        *mx = vm;
        *am = va;
    }
}

__global__ static void amx_part_k(const float *x, uint32_t n, amx_part_t *p) {
    const uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stp = gridDim.x * blockDim.x;
    float mx = -INFINITY;
    uint32_t am = 0xffffffffu;
    for (uint32_t i = tid; i < n; i += stp) {
        const float v = x[i];
        amx_pick(v, i, &mx, &am);
    }
    for (uint32_t o = 16; o; o >>= 1)
        amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
    __shared__ float sm[8];
    __shared__ uint32_t sa[8];
    __shared__ float s_bmx;
    if (lane == 0) {
        sm[w] = mx;
        sa[w] = am;
    }
    __syncthreads();
    if (w == 0) {
        mx = lane < 8 ? sm[lane] : -INFINITY;
        am = lane < 8 ? sa[lane] : 0xffffffffu;
        for (uint32_t o = 4; o; o >>= 1)
            amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
        if (lane == 0) {
            s_bmx = mx;
            p[blockIdx.x].am = am;
        }
    }
    __syncthreads();
    const float bmx = s_bmx;
    float se = 0.f;
    for (uint32_t i = tid; i < n; i += stp) {
        const float d = x[i] - bmx;
        se += (d == d) ? __expf(d) : 0.f; /* -inf - -inf = NaN */
    }
    for (uint32_t o = 16; o; o >>= 1) se += __shfl_down(se, o);
    __shared__ float ss[8];
    if (lane == 0) ss[w] = se;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) t += ss[i];
        p[blockIdx.x].mx = bmx;
        p[blockIdx.x].se = t;
    }
}

__global__ static void amx_comb_k(const amx_part_t *p, uint32_t nb,
                                  uint32_t *out) {
    const uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    float mx = -INFINITY;
    uint32_t am = 0xffffffffu;
    for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x)
        amx_pick(p[i].mx, p[i].am, &mx, &am);
    for (uint32_t o = 16; o; o >>= 1)
        amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
    __shared__ float sm[8];
    __shared__ uint32_t sa[8];
    __shared__ float s_gmx;
    __shared__ uint32_t s_gam;
    if (lane == 0) {
        sm[w] = mx;
        sa[w] = am;
    }
    __syncthreads();
    if (w == 0) {
        mx = lane < 8 ? sm[lane] : -INFINITY;
        am = lane < 8 ? sa[lane] : 0xffffffffu;
        for (uint32_t o = 4; o; o >>= 1)
            amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
        if (lane == 0) {
            s_gmx = mx;
            s_gam = am;
        }
    }
    __syncthreads();
    const float gmx = s_gmx;
    float se = 0.f;
    for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
        const float w = __expf(p[i].mx - gmx);
        se += (w == w) ? w * p[i].se : 0.f;
    }
    for (uint32_t o = 16; o; o >>= 1) se += __shfl_down(se, o);
    __shared__ float ss[8];
    if (lane == 0) ss[w] = se;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) t += ss[i];
        out[0] = s_gam;
        float pr = (t > 0.f) ? 1.f / t : 0.f;
        memcpy(out + 1, &pr, 4);
    }
}

/* Device argmax of d_x[0..n) plus the softmax probability of the max
 * (p_max = 1 / sum_i exp(x_i - max)). The launch writes 8 bytes into mapped
 * memory; the syncing wrapper is what a host read waits on. */
extern "C" bool q4_hip_argmax_launch(const float *d_x, uint32_t n) {
    if (!g_ok || !d_x || !n) return false;
    if (!amx_bufs()) return false;
    uint32_t nb = (n + 2047u) / 2048u;
    if (nb > AMX_MAXB) nb = AMX_MAXB;
    if (nb < 1) nb = 1;
    amx_part_k<<<nb, 256, 0, g_str>>>(d_x, n, g_amx_part);
    amx_comb_k<<<1, 256, 0, g_str>>>(g_amx_part, nb, g_amx_d);
    return hipGetLastError() == hipSuccess;
}

extern "C" void q4_hip_argmax_read(int32_t *h_id, float *h_p) {
    if (!g_amx_h) return;
    if (h_id) *h_id = (int32_t)g_amx_h[0];
    if (h_p) {
        float p;
        memcpy(&p, g_amx_h + 1, 4);
        *h_p = p;
    }
}

extern "C" bool q4_hip_argmax_prob(const float *d_x, uint32_t n, int32_t *h_id,
                                   float *h_p) {
    if (!h_id || !h_p) return false;
    if (!q4_hip_argmax_launch(d_x, n)) return false;
    if (hipStreamSynchronize(g_str) != hipSuccess) return false;
    q4_hip_argmax_read(h_id, h_p);
    return true;
}

/* Top-k over a logit row. Each block keeps the top-k of a contiguous span
 * with the same insertion rule as q4_topk (strict >, earlier index stays).
 * The merge walks blocks in index order, and within a block inserts in
 * index order, so ties match the host scan. k <= 128. */
#define TK_MAX 128u
#define TK_BLOCKS 128u
static float *g_tk_bv;
static int32_t *g_tk_bi;
static int32_t *g_tk_hids, *g_tk_dids;
static float *g_tk_hvs, *g_tk_dvs;
static uint32_t g_tk_k;

static bool tk_bufs(void) {
    if (g_tk_bv) return true;
    size_t n = (size_t)TK_BLOCKS * TK_MAX;
    if (hipMalloc((void **)&g_tk_bv, n * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc((void **)&g_tk_bi, n * sizeof(int32_t)) != hipSuccess)
        return false;
    if (hipHostMalloc((void **)&g_tk_hids, TK_MAX * sizeof(int32_t),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer((void **)&g_tk_dids, g_tk_hids, 0) !=
            hipSuccess ||
        hipHostMalloc((void **)&g_tk_hvs, TK_MAX * sizeof(float),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer((void **)&g_tk_dvs, g_tk_hvs, 0) != hipSuccess) {
        return false;
    }
    return true;
}

__device__ static void tk_consider(float *bv, int32_t *bi, uint32_t k, float x,
                                   int32_t ix) {
    uint32_t p = k;
    while (p > 0 && (bi[p - 1] < 0 || x > bv[p - 1])) p--;
    if (p == k) return;
    for (uint32_t j = k - 1; j > p; j--) {
        bv[j] = bv[j - 1];
        bi[j] = bi[j - 1];
    }
    bv[p] = x;
    bi[p] = ix;
}

__global__ static void tk_block_k(const float *x, uint32_t n, uint32_t k,
                                  float *ov, int32_t *oi, int32_t id_base) {
    if (threadIdx.x != 0) return;
    uint32_t span = (n + gridDim.x - 1) / gridDim.x;
    uint32_t lo = blockIdx.x * span;
    if (lo > n) lo = n;
    uint32_t hi = lo + span < n ? lo + span : n;
    float bv[TK_MAX];
    int32_t bi[TK_MAX];
    for (uint32_t i = 0; i < k; i++) {
        bv[i] = -INFINITY;
        bi[i] = -1;
    }
    for (uint32_t i = lo; i < hi; i++)
        tk_consider(bv, bi, k, x[i], id_base + (int32_t)i);
    for (uint32_t i = 0; i < k; i++) {
        ov[(size_t)blockIdx.x * k + i] = bv[i];
        oi[(size_t)blockIdx.x * k + i] = bi[i];
    }}

__global__ static void tk_merge_k(const float *ov, const int32_t *oi,
                                  uint32_t nb, uint32_t k, int32_t *ids,
                                  float *vals) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float bv[TK_MAX];
    int32_t bi[TK_MAX];
    for (uint32_t i = 0; i < k; i++) {
        bv[i] = -INFINITY;
        bi[i] = -1;
    }
    for (uint32_t b = 0; b < nb; b++) {
        int32_t id[TK_MAX];
        float v[TK_MAX];
        uint32_t m = 0;
        for (uint32_t i = 0; i < k; i++) {
            int32_t ix = oi[(size_t)b * k + i];
            if (ix < 0) continue;
            id[m] = ix;
            v[m] = ov[(size_t)b * k + i];
            m++;
        }
        for (uint32_t i = 1; i < m; i++) {
            int32_t ix = id[i];
            float vx = v[i];
            uint32_t j = i;
            while (j > 0 && id[j - 1] > ix) {
                id[j] = id[j - 1];
                v[j] = v[j - 1];
                j--;
            }
            id[j] = ix;
            v[j] = vx;
        }
        for (uint32_t i = 0; i < m; i++) tk_consider(bv, bi, k, v[i], id[i]);
    }
    for (uint32_t i = 0; i < k; i++) {
        ids[i] = bi[i];
        vals[i] = bv[i];
    }
}

static bool launch(uint32_t type, const uint8_t *d_w, uint64_t nrows,
                   uint64_t ncols, const float *d_x, float *d_y, float scale,
                   uint32_t n_batch, const int32_t *which);






/* ---- parallel top-k ----
 * The serial tk_block_k/tk_merge_k pair costs ~4 ms per call over the
 * 248320-row head (measured 2.7 ms in the merge alone, single-threaded
 * with local-memory indices). This pair does the same selection with all
 * 32 threads of a block working at once:
 *   tk_sel_k   each thread keeps a local top-k over a strided range in
 *              shared memory, then the warp cooperates on k selection
 *              passes over the 32*k candidates;
 *   tk_selm_k  one thread per input block folds its k candidates, then the
 *              same cooperative passes pick the winner.
 * Ordering is identical to the serial pair (value descending, ties to the
 * lowest id), so the sampled distribution is unchanged. */
#define TK_THREADS 32u

__global__ static void tk_sel_k(const float *x, uint32_t n, uint32_t k,
                                float *ov, int32_t *oi, int32_t id_base) {
    extern __shared__ float smem[];
    float *cv = smem;                 /* TK_THREADS*k candidates */
    int32_t *ci = (int32_t *)(smem + TK_THREADS * k);
    float *rv = (float *)(smem + 2u * TK_THREADS * k);
    int32_t *ri = (int32_t *)(rv + k);
    const uint32_t t = threadIdx.x;
    const uint32_t nc = TK_THREADS * k;
    float *cv_t = cv + (size_t)t * k;
    int32_t *ci_t = ci + (size_t)t * k;
    for (uint32_t i = 0; i < k; i++) {
        cv_t[i] = -INFINITY;
        ci_t[i] = -1;
    }
    uint32_t span = (n + gridDim.x - 1) / gridDim.x;
    uint32_t lo = blockIdx.x * span;
    if (lo > n) lo = n;
    uint32_t hi = lo + span < n ? lo + span : n;
    for (uint32_t i = lo + t; i < hi; i += TK_THREADS)
        tk_consider(cv_t, ci_t, k, x[i], id_base + (int32_t)i);
    __syncthreads();
    for (uint32_t p = 0; p < k; p++) {
        float bv = -INFINITY;
        int32_t bi = -1;
        for (uint32_t i = t; i < nc; i += TK_THREADS)
            if (cv[i] > bv) {
                bv = cv[i];
                bi = ci[i];
            }
        for (int s = 16; s > 0; s >>= 1) {
            float ov_v = __shfl_down_sync(0xffffffffffffffffULL, bv, s);
            int32_t ov_i = __shfl_down_sync(0xffffffffffffffffULL, bi, s);
            if (ov_v > bv || (ov_v == bv && ov_i >= 0 && (bi < 0 || ov_i < bi))) {
                bv = ov_v;
                bi = ov_i;
            }
        }
        if (t == 0) {
            rv[p] = bv;
            ri[p] = bi;
        }
        __syncthreads();
        /* Every thread must drop the WINNER from its own slice; only
         * thread 0 holds the warp-reduced best, so read it back from the
         * shared result slot. */
        {
            float wv = rv[p];
            int32_t wi = ri[p];
            for (uint32_t i = t; i < nc; i += TK_THREADS)
                if (ci[i] == wi && cv[i] == wv) cv[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (t == 0)
        for (uint32_t i = 0; i < k; i++) {
            ov[(size_t)blockIdx.x * k + i] = rv[i];
            oi[(size_t)blockIdx.x * k + i] = ri[i];
        }
}

/* Merge nb blocks' k candidates each (nb <= TK_THREADS): thread t folds
 * block t, then the same cooperative selection passes pick the winners. */
__global__ static void tk_selm_k(const float *cv, const int32_t *ci, uint32_t nb,
                                 uint32_t k, int32_t *ids, float *vals) {
    extern __shared__ float smem[];
    float *scv = smem;
    int32_t *sci = (int32_t *)(smem + TK_THREADS * k);
    float *rv = (float *)(smem + 2u * TK_THREADS * k);
    int32_t *ri = (int32_t *)(rv + k);
    const uint32_t t = threadIdx.x;
    /* Only 32 threads exist for up to TK_THREADS input blocks, so each
     * thread folds nb/TK_THREADS blocks into its own top-k list. */
    {
        float *cvt = scv + (size_t)t * k;
        int32_t *cit = sci + (size_t)t * k;
        for (uint32_t i = 0; i < k; i++) {
            cvt[i] = -INFINITY;
            cit[i] = -1;
        }
        for (uint32_t bl = t; bl < nb; bl += TK_THREADS)
            for (uint32_t i = 0; i < k; i++)
                tk_consider(cvt, cit, k, cv[(size_t)bl * k + i],
                            ci[(size_t)bl * k + i]);
    }
    __syncthreads();
    const uint32_t nc = TK_THREADS * k;
    for (uint32_t p = 0; p < k; p++) {
        float bv = -INFINITY;
        int32_t bi = -1;
        for (uint32_t i = t; i < nc; i += TK_THREADS)
            if (scv[i] > bv) {
                bv = scv[i];
                bi = sci[i];
            }
        for (int s = 16; s > 0; s >>= 1) {
            float ov_v = __shfl_down_sync(0xffffffffffffffffULL, bv, s);
            int32_t ov_i = __shfl_down_sync(0xffffffffffffffffULL, bi, s);
            if (ov_v > bv || (ov_v == bv && ov_i >= 0 && (bi < 0 || ov_i < bi))) {
                bv = ov_v;
                bi = ov_i;
            }
        }
        if (t == 0) {
            rv[p] = bv;
            ri[p] = bi;
        }
        __syncthreads();
        {
            float wv = rv[p];
            int32_t wi = ri[p];
            for (uint32_t i = t; i < nc; i += TK_THREADS)
                if (sci[i] == wi && scv[i] == wv) scv[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (t == 0)
        for (uint32_t i = 0; i < k; i++) {
            ids[i] = ri[i];
            vals[i] = rv[i];
        }
}

/* tk_sel_k always keeps TK_THREADS local top-k lists in shared memory
 * (one per thread); tk_selm_k keeps nb lists (one per input block). */
static size_t tk_sel_smem(uint32_t k) {
    return (size_t)TK_THREADS * k * 2u * sizeof(float) + (size_t)k * 8u + 64u;
}

static size_t tk_selm_smem(uint32_t k) {
    return (size_t)TK_THREADS * k * 2u * sizeof(float) + (size_t)k * 8u + 64u;
}

/* ---- wide top-k ----
 * The slab path gave each 32768-row slab of the draft head 4 blocks of 32
 * threads (tk_selm_k's TK_THREADS cap): rocprof 2026-10-11 put tk_sel_k at
 * 8 x ~430 us = 3.4 ms of a 4.65 ms draft step. tkw_k selects k from a
 * 2048-element span per block: 256 threads hold 8 values each in
 * registers, each pass is one block max and retires the winner. A launch
 * turns n inputs into ceil(n/2048)*k candidates; tkw_run repeats it until
 * one block is left. Order is the same as tk_sel_k/tk_selm_k: value
 * descending, ties to the lowest id, NaN ranks as -inf, missing slots are
 * (-inf, -1). Q4_TOPK_WIDE=0 restores the slab/32-block kernels. */
#define TKW_T 256u
#define TKW_E 8u
#define TKW_SPAN (TKW_T * TKW_E)
#define TKW_CAND 65536u          /* first-pass candidates: 2^20 rows at k=128 */
static float *g_tkw_v[2];
static int32_t *g_tkw_i[2];

static int tkw_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_TOPK_WIDE");
        v = !(e && e[0] == '0');
    }
    return v;
}

static bool tkw_bufs(void) {
    if (g_tkw_v[1]) return true;
    for (int b = 0; b < 2; b++)
        if (hipMalloc((void **)&g_tkw_v[b], TKW_CAND * sizeof(float)) !=
                hipSuccess ||
            hipMalloc((void **)&g_tkw_i[b], TKW_CAND * sizeof(int32_t)) !=
                hipSuccess)
            return false;
    return true;
}

__device__ static inline void tkw_pick(float v, int32_t i, float *bv,
                                       int32_t *bi) {
    if (i < 0) return;
    if (*bi < 0 || v > *bv || (v == *bv && i < *bi)) {
        *bv = v;
        *bi = i;
    }
}

/* xi == NULL: ids are the input index. map != NULL (last pass only):
 * written ids go through map (subset row -> token id). */
__global__ static void tkw_k(const float *xv, const int32_t *xi, uint32_t n,
                             uint32_t k, float *ov, int32_t *oi,
                             const int32_t *map, uint32_t sx = 0,
                             uint32_t so = 0) {
    /* blockIdx.y: row of a batched call — input row stride sx, output row
     * stride so (0 / 0 for a single row). */
    xv += (size_t)blockIdx.y * sx;
    if (xi) xi += (size_t)blockIdx.y * sx;
    ov += (size_t)blockIdx.y * so;
    oi += (size_t)blockIdx.y * so;
    const uint32_t t = threadIdx.x, lane = t & 31u, w = t >> 5;
    const uint32_t lo = blockIdx.x * TKW_SPAN;
    float v[TKW_E];
    int32_t id[TKW_E];
#pragma unroll
    for (uint32_t e = 0; e < TKW_E; e++) {
        const uint32_t i = lo + e * TKW_T + t;
        if (i < n) {
            const float x = xv[i];
            v[e] = x == x ? x : -INFINITY;
            id[e] = xi ? xi[i] : (int32_t)i;
        } else {
            v[e] = -INFINITY;
            id[e] = -1;
        }
    }
    __shared__ float sv[TKW_T / 32u];
    __shared__ int32_t si[TKW_T / 32u];
    __shared__ int32_t s_win;
    for (uint32_t p = 0; p < k; p++) {
        float bv = -INFINITY;
        int32_t bi = -1;
#pragma unroll
        for (uint32_t e = 0; e < TKW_E; e++) tkw_pick(v[e], id[e], &bv, &bi);
        for (uint32_t o = 16; o; o >>= 1) {
            const float vo = __shfl_down(bv, o);
            const int32_t io = __shfl_down(bi, o);
            tkw_pick(vo, io, &bv, &bi);
        }
        if (lane == 0) {
            sv[w] = bv;
            si[w] = bi;
        }
        __syncthreads();
        if (t == 0) {
            float gv = sv[0];
            int32_t gi = si[0];
            for (uint32_t j = 1; j < TKW_T / 32u; j++)
                tkw_pick(sv[j], si[j], &gv, &gi);
            s_win = gi;
            const size_t o = (size_t)blockIdx.x * k + p;
            ov[o] = gi >= 0 ? gv : -INFINITY;
            oi[o] = (gi >= 0 && map) ? map[gi] : gi;
        }
        __syncthreads();
        const int32_t win = s_win;
        if (win >= 0)
#pragma unroll
            for (uint32_t e = 0; e < TKW_E; e++)
                if (id[e] == win) {
                    id[e] = -1;
                    v[e] = -INFINITY;
                }
    }
}

/* Top-k of d_x[0..n) into (out_v, out_i); NULL = the mapped top-k result
 * (q4_hip_topk_read). */
static bool tkw_run_to(const float *d_x, uint32_t n, uint32_t k,
                       const int32_t *map, float *out_v, int32_t *out_i) {
    if (!tkw_bufs() || !tk_bufs()) return false;
    if (k > n) k = n;
    const float *xv = d_x;
    const int32_t *xi = NULL;
    for (int b = 0;; b ^= 1) {
        const uint32_t nb = (n + TKW_SPAN - 1u) / TKW_SPAN;
        const bool last = nb == 1u;
        if (!last && (size_t)nb * k > TKW_CAND) return false;
        float *ov = last ? (out_v ? out_v : g_tk_dvs) : g_tkw_v[b];
        int32_t *oi = last ? (out_i ? out_i : g_tk_dids) : g_tkw_i[b];
        tkw_k<<<nb, TKW_T, 0, g_str>>>(xv, xi, n, k, ov, oi,
                                       last ? map : NULL);
        if (last) break;
        xv = ov;
        xi = oi;
        n = nb * k;
    }
    if (!out_v) g_tk_k = k;
    return hipGetLastError() == hipSuccess;
}

static bool tkw_run(const float *d_x, uint32_t n, uint32_t k,
                    const int32_t *map) {
    return tkw_run_to(d_x, n, k, map, NULL, NULL);
}

/* Per-row top-k of a [rows][n] block (the MTP verify rows' logits) into
 * mapped memory: rows x k pairs, read with q4_hip_topk_rows_read after a
 * stream sync. rows <= TKR_ROWS, k <= TK_MAX. */
#define TKR_ROWS 8u
static int32_t *g_tkr_hids, *g_tkr_dids;
static float *g_tkr_hvs, *g_tkr_dvs;
static uint32_t g_tkr_k;

extern "C" bool q4_hip_topk_rows_launch(const float *d_x, uint32_t n,
                                        uint32_t rows, uint32_t k) {
    if (!g_ok || !d_x || !n || !rows || rows > TKR_ROWS || !k || k > TK_MAX)
        return false;
    if (!g_tkr_hids) {
        const size_t c = (size_t)TKR_ROWS * TK_MAX;
        if (hipHostMalloc((void **)&g_tkr_hids, c * sizeof(int32_t),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostGetDevicePointer((void **)&g_tkr_dids, g_tkr_hids, 0) !=
                hipSuccess ||
            hipHostMalloc((void **)&g_tkr_hvs, c * sizeof(float),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostGetDevicePointer((void **)&g_tkr_dvs, g_tkr_hvs, 0) !=
                hipSuccess) {
            g_tkr_hids = NULL;
            return false;
        }
    }
    if (k > n) k = n;
    /* All rows per pass in one launch (grid.y = row) while the candidate
     * buffers hold rows x spans x k; 12 launches -> 3 for a 4-row verify.
     * Same per-row selection, so the result is unchanged. */
    const uint32_t nb0 = (n + TKW_SPAN - 1u) / TKW_SPAN;
    if (tkw_bufs() && tk_bufs() && nb0 > 1u &&
        (size_t)rows * nb0 * k <= TKW_CAND &&
        !(getenv("Q4_TKR_BATCH") && getenv("Q4_TKR_BATCH")[0] == '0')) {
        const float *xv = d_x;
        const int32_t *xi = NULL;
        uint32_t nn = n, sx = n;
        for (int b = 0;; b ^= 1) {
            const uint32_t nb = (nn + TKW_SPAN - 1u) / TKW_SPAN;
            const bool last = nb == 1u;
            float *ov = last ? g_tkr_dvs : g_tkw_v[b];
            int32_t *oi = last ? g_tkr_dids : g_tkw_i[b];
            const uint32_t so = last ? k : nb * k;
            tkw_k<<<dim3(nb, rows), TKW_T, 0, g_str>>>(xv, xi, nn, k, ov, oi,
                                                       NULL, sx, so);
            if (last) break;
            xv = ov;
            xi = oi;
            nn = nb * k;
            sx = so;
        }
        g_tkr_k = k;
        return hipGetLastError() == hipSuccess;
    }
    for (uint32_t r = 0; r < rows; r++)
        if (!tkw_run_to(d_x + (size_t)r * n, n, k, NULL,
                        g_tkr_dvs + (size_t)r * k, g_tkr_dids + (size_t)r * k))
            return false;
    g_tkr_k = k;
    return true;
}

extern "C" void q4_hip_topk_rows_read(uint32_t row, int32_t *ids, float *vals,
                                      uint32_t k) {
    if (!g_tkr_hids || row >= TKR_ROWS || !ids || !vals) return;
    if (k > g_tkr_k) k = g_tkr_k;
    memcpy(ids, g_tkr_hids + (size_t)row * g_tkr_k, (size_t)k * sizeof(int32_t));
    memcpy(vals, g_tkr_hvs + (size_t)row * g_tkr_k, (size_t)k * sizeof(float));
}


/* ---- slab head top-k: top-k over a tall projection, no full logits
 * vector. Each slab of rows is one GEMV into a small buffer, reduced to
 * k (id,val) pairs by TK_BLOCKS blocks plus one merge (the same pair of
 * kernels the full top-k uses), and the per-slab results merge at the
 * end. The result lands in the shared pinned top-k buffers; read it with
 * q4_hip_topk_read after a stream sync. A full-vocab GEMV would write
 * nrows*4 bytes (995 MiB for the draft head) per step; this writes ~30 KB
 * and reads the same weight bytes. */
#define HK_SLAB 32768u
#define HK_MAXSLAB 64u
static float *g_hk_slab;   /* HK_SLAB floats */
static float *g_hk_fv;     /* HK_MAXSLAB * TK_MAX */
static int32_t *g_hk_fi;
static float *g_hk_rows;   /* wide path: HK_MAXSLAB * HK_SLAB floats (8 MiB) */

static bool hk_bufs(void) {
    if (tkw_on()) {
        if (!g_hk_rows &&
            hipMalloc((void **)&g_hk_rows,
                      (size_t)HK_MAXSLAB * HK_SLAB * sizeof(float)) !=
                hipSuccess) {
            g_hk_rows = NULL;
            return false;
        }
        return tkw_bufs();
    }
    if (g_hk_slab) return true;
    if (hipMalloc((void **)&g_hk_slab, HK_SLAB * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc((void **)&g_hk_fv,
                  (size_t)HK_MAXSLAB * TK_MAX * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc((void **)&g_hk_fi,
                  (size_t)HK_MAXSLAB * TK_MAX * sizeof(int32_t)) != hipSuccess)
        return false;
    return true;
}

/* Pre-allocate the slab buffers: illegal inside a graph capture. */
extern "C" void q4_hip_head_topk_warm(void) {
    (void)hk_bufs();
    (void)tk_bufs();
    (void)amx_bufs();
}

__global__ static void gather_rows_k(uint32_t *dst, const uint32_t *src,
                                     uint32_t row_words, const int32_t *ids) {
    const uint64_t r = blockIdx.y;
    const uint64_t s = (uint64_t)ids[r];
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < row_words;
         i += gridDim.x * blockDim.x)
        dst[r * row_words + i] = src[s * row_words + i];
}

extern "C" bool q4_hip_gather_rows(uint8_t *dst, const uint8_t *src,
                                   uint64_t row_bytes, const int32_t *d_ids,
                                   uint32_t n) {
    if (!g_ok || !dst || !src || !d_ids || !n || (row_bytes & 3u) ||
        n > 65535u * 64u)
        return false;
    const uint32_t rw = (uint32_t)(row_bytes / 4u);
    /* gridDim.y caps at 65535: split the rows into launches. */
    for (uint32_t r0 = 0; r0 < n; r0 += 65535u) {
        const uint32_t nr = n - r0 < 65535u ? n - r0 : 65535u;
        gather_rows_k<<<dim3((rw + 255u) / 256u, nr), 256, 0, g_str>>>(
            (uint32_t *)(dst + (uint64_t)r0 * row_bytes), (const uint32_t *)src,
            rw, d_ids + r0);
    }
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_head_topk(uint32_t ggml_type, const uint8_t *d_w,
                                 uint64_t ncols, uint64_t nrows,
                                 const float *d_x, uint32_t k,
                                 const int32_t *d_map) {
    if (d_map && !tkw_on()) return false;
    if (!g_ok || !d_w || !d_x || !ncols || !nrows || !k || k > TK_MAX)
        return false;
    if (nrows > (uint64_t)HK_MAXSLAB * HK_SLAB) return false;
    if (!hk_bufs() || !tk_bufs()) return false;
    uint64_t rb = q4_row_bytes(ggml_type, ncols);
    if (!rb) return false;
    if (k > nrows) k = (uint32_t)nrows;
    if (tkw_on()) {
        /* One GEMV into an nrows-float buffer (0.95 MiB at the 248320
         * vocab), then the wide selection. */
        if (!launch(ggml_type, d_w, nrows, ncols, d_x, g_hk_rows, 1.f, 1,
                    NULL))
            return false;
        /* k == 1 (greedy draft): also the softmax p of the max, for the
         * confidence gate — q4_hip_argmax_read's prob (its id is the
         * unmapped row; take the id from the top-k). */
        if (k == 1u && !q4_hip_argmax_launch(g_hk_rows, (uint32_t)nrows))
            return false;
        return tkw_run(g_hk_rows, (uint32_t)nrows, k, d_map);
    }
    uint32_t ns = (uint32_t)((nrows + HK_SLAB - 1) / HK_SLAB);
    /* One block group per slab; total groups must fit the final merge's
     * 32 threads (tk_selm_k holds nb*k candidates in shared memory). */
    uint32_t nbs = TK_THREADS / ns;
    if (nbs < 1u) nbs = 1u;
    size_t smb = tk_sel_smem(k);
    size_t smm = tk_selm_smem(k);
    for (uint32_t s = 0; s < ns; s++) {
        uint64_t rows = nrows - (uint64_t)s * HK_SLAB;
        if (rows > HK_SLAB) rows = HK_SLAB;
        uint32_t nr = (uint32_t)rows;
        if (!launch(ggml_type, d_w + (size_t)s * HK_SLAB * rb, nr, ncols, d_x,
                    g_hk_slab, 1.f, 1, NULL))
            return false;
        tk_sel_k<<<nbs, TK_THREADS, smb, g_str>>>(
            g_hk_slab, nr, k, g_hk_fv + (size_t)s * nbs * k,
            g_hk_fi + (size_t)s * nbs * k, (int32_t)((uint64_t)s * HK_SLAB));
    }
    g_tk_k = k;
    tk_selm_k<<<1, TK_THREADS, smm, g_str>>>(g_hk_fv, g_hk_fi, ns * nbs, k,
                                             g_tk_dids, g_tk_dvs);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_topk_launch(const float *d_x, uint32_t n, uint32_t k) {
    if (!g_ok || !d_x || !n || !k || k > TK_MAX) return false;
    if (tkw_on()) return tkw_run(d_x, n, k, NULL);
    if (!tk_bufs()) return false;
    if (k > n) k = n;
    uint32_t nb = n < TK_THREADS ? n : TK_THREADS;
    if (nb < 1) nb = 1;
    g_tk_k = k;
    tk_sel_k<<<nb, TK_THREADS, tk_sel_smem(k), g_str>>>(d_x, n, k, g_tk_bv,
                                                        g_tk_bi, 0);
    tk_selm_k<<<1, TK_THREADS, tk_selm_smem(k), g_str>>>(g_tk_bv, g_tk_bi, nb,
                                                            k, g_tk_dids,
                                                            g_tk_dvs);
    return hipGetLastError() == hipSuccess;
}

extern "C" void q4_hip_topk_read(int32_t *h_ids, float *h_vals, uint32_t k) {
    if (!g_tk_hids || !h_ids || !h_vals) return;
    if (k > g_tk_k) k = g_tk_k;
    if (k > TK_MAX) k = TK_MAX;
    memcpy(h_ids, g_tk_hids, (size_t)k * sizeof(int32_t));
    memcpy(h_vals, g_tk_hvs, (size_t)k * sizeof(float));
}

extern "C" bool q4_hip_topk(const float *d_x, uint32_t n, uint32_t k,
                            int32_t *h_ids, float *h_vals) {
    if (!h_ids || !h_vals) return false;
    if (!q4_hip_topk_launch(d_x, n, k)) return false;
    if (!g_str || hipStreamSynchronize(g_str) != hipSuccess) return false;
    uint32_t kk = k > n ? n : k;
    q4_hip_topk_read(h_ids, h_vals, kk);
    return true;
}

__device__ static inline void scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                           uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ static void gemv_q4_k_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * (size_t)nb * 144ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
            const uint8_t *blk = wr + i * 144u;
            float d = f16d(blk);
            float minv = f16d(blk + 2);
            const uint8_t *sc = blk + 4;
            const uint8_t *q = blk + 16;
            const float *xr = xt + i * 256u;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is + 0, sc, &s1, &m1);
                scale_min_k4(is + 1, sc, &s2, &m2);
                float d1 = d * s1, mm1 = minv * m1;
                float d2 = d * s2, mm2 = minv * m2;
                for (int l = 0; l < 32; l++)
                    acc += (d1 * (q[l] & 0xF) - mm1) * xr[j + l];
                for (int l = 0; l < 32; l++)
                    acc += (d2 * (q[l] >> 4) - mm2) * xr[j + 32 + l];
                q += 32;
                is += 2;
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__global__ static void gemv_q5_1_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 32u;
    const uint8_t *wr = w + (size_t)row * (size_t)nb * 24ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
            const uint8_t *blk = wr + (size_t)b * 24u;
            float d = f16d(blk);
            float m = f16d(blk + 2);
            uint32_t qh;
            memcpy(&qh, blk + 4, 4);
            const uint8_t *qs = blk + 8;
            const float *xr = xt + b * 32u;
            for (int j = 0; j < 16; j++) {
                uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
                uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                int x0 = (qs[j] & 0x0F) | xh0;
                int x1 = (qs[j] >> 4) | xh1;
                acc += (x0 * d + m) * xr[j];
                acc += (x1 * d + m) * xr[j + 16];
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__global__ static void gemv_q5_k_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * (size_t)nb * 176ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
            const uint8_t *blk = wr + i * 176u;
            float d = f16d(blk);
            float minv = f16d(blk + 2);
            const uint8_t *sc = blk + 4;
            const uint8_t *qh = blk + 16;
            const uint8_t *ql = blk + 48;
            const float *xr = xt + i * 256u;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is + 0, sc, &s1, &m1);
                scale_min_k4(is + 1, sc, &s2, &m2);
                float d1 = d * s1, mm1 = minv * m1;
                float d2 = d * s2, mm2 = minv * m2;
                for (int l = 0; l < 32; l++)
                    acc += (d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - mm1) *
                           xr[j + l];
                for (int l = 0; l < 32; l++)
                    acc += (d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - mm2) *
                           xr[j + 32 + l];
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

/* Tiled Q8_0 x F32 GEMM for batched prefill projections.
 * W Q8_0 [nrows x ncols], X F32 [n_batch x ncols], Y F32 [n_batch x nrows].
 * Block: 256 threads, tile 32 rows x 32 tokens x 32 cols (one Q8_0 block). */
__global__ static void gemm_q8_0_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim,
                                   uint32_t n_batch, float scale) {
    __shared__ float ws[32][33];
    __shared__ float xs[32][33];
    const uint32_t r0 = blockIdx.x * 32u;
    const uint32_t t0 = blockIdx.y * 32u;
    const uint32_t nb = in_dim / 32u;
    const uint32_t wr = threadIdx.x / 8u;      /* 0..31: row stripe */
    const uint32_t kk = (threadIdx.x % 8u) * 4u; /* 0,4,...,28 */
    const uint32_t warp = threadIdx.x / 32u;   /* 0..7 */
    const uint32_t lane = threadIdx.x % 32u;   /* 0..31 */
    /* each thread owns exactly 1 row x 4 tokens of the 32x32 tile */
    const uint32_t tr = warp * 4u + lane / 8u; /* 0..31 */
    const uint32_t tt = (lane % 8u) * 4u;      /* 0,4,...,28 */
    float acc[4] = {};
    for (uint32_t kb = 0; kb < nb; kb++) {
        /* load W tile: rows r0+0..31, q8 block kb */
        {
            uint32_t r = r0 + wr;
            float d = 0.f;
            int8_t qv[4] = {0, 0, 0, 0};
            if (r < out_dim) {
                const uint8_t *blk = w + ((size_t)r * nb + kb) * 34ull;
                __half dh;
                memcpy(&dh, blk, 2);
                d = __half2float(dh);
                memcpy(qv, blk + 2 + kk, 4);
            }
            ws[wr][kk + 0] = d * (float)qv[0];
            ws[wr][kk + 1] = d * (float)qv[1];
            ws[wr][kk + 2] = d * (float)qv[2];
            ws[wr][kk + 3] = d * (float)qv[3];
        }
        /* load X tile: toks t0+0..31, cols kb*32+0..31 */
        {
            uint32_t t = t0 + wr;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (t < n_batch) {
                const float *xr = x + (size_t)t * in_dim + kb * 32u + kk;
                v[0] = xr[0];
                v[1] = xr[1];
                v[2] = xr[2];
                v[3] = xr[3];
            }
            xs[wr][kk + 0] = v[0];
            xs[wr][kk + 1] = v[1];
            xs[wr][kk + 2] = v[2];
            xs[wr][kk + 3] = v[3];
        }
        __syncthreads();
#pragma unroll 8
        for (uint32_t k = 0; k < 32u; k++) {
            float wv = ws[tr][k];
#pragma unroll
            for (int j = 0; j < 4; j++) acc[j] += wv * xs[tt + j][k];
        }
        __syncthreads();
    }
    {
        uint32_t r = r0 + tr;
        if (r < out_dim) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t t = t0 + tt + (uint32_t)j;
                if (t < n_batch) out[(size_t)t * out_dim + r] = acc[j] * scale;
            }
        }
    }
}

/* Split-K gemv for tiny out_dim: grid (out_dim, S), partials to scratch,
 * deterministic ascending sum in the second kernel. */
__global__ static void gemv_f32_sk(float *part, const float *w, const float *x,
                                   uint32_t in_dim, uint32_t S) {
    uint32_t row = blockIdx.x, s = blockIdx.y;
    uint32_t per = (in_dim + S - 1) / S;
    uint32_t i0 = s * per, i1 = i0 + per < in_dim ? i0 + per : in_dim;
    const float *wr = w + (size_t)row * in_dim;
    float acc = 0.f;
    for (uint32_t i = i0 + threadIdx.x; i < i1; i += blockDim.x)
        acc += wr[i] * x[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t sp = blockDim.x >> 1; sp > 0; sp >>= 1) {
        if (threadIdx.x < sp) sh[threadIdx.x] += sh[threadIdx.x + sp];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[(size_t)row * S + s] = sh[0];
}

__global__ static void gemv_sk_sum(float *out, const float *part,
                                   uint32_t out_dim, uint32_t S, float scale) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    float acc = 0.f;
    for (uint32_t s = 0; s < S; s++) acc += part[(size_t)row * S + s];
    out[row] = acc * scale;
}

#define GEMV_SK_MAXROWS 256u
#define GEMV_SK_S 64u

/* Q4_BF16N=0 keeps the warp-per-row batch kernel. */
static int bf16n_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_BF16N");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* Shape split mirrors the n_batch==1 bf16 dispatch.  act != 0 folds
 * scale+silu onto each row (hc down).  Returns false without launching
 * when the aligned path does not apply. */
static bool bf16n_launch(float *d_y, const uint8_t *d_w, const float *d_x,
                         uint32_t ncols, uint32_t nrows, float scale,
                         float act, uint32_t nt) {
    if (!bf16n_on() || nt < 2 || nt > WGVN_MAX || !nrows || !ncols) return false;
    if ((ncols & 7u) || ((uintptr_t)d_w & 15u) || ((uintptr_t)d_x & 15u))
        return false;
#define BF16N_ONE(NT)                                                          \
    do {                                                                       \
        if (nrows <= 1024u && ncols >= 2048u)                                  \
            gemv_bf16_bn_t<NT><<<nrows, 256, 0, g_str>>>(                      \
                d_y, d_w, d_x, ncols, nrows, scale, act);                      \
        else if (ncols <= 512u) {                                              \
            dim3 gg((nrows * 8u + 255u) / 256u);                               \
            wgvn_bf16v_t<8, NT><<<gg, 256, 0, g_str>>>(                        \
                d_y, d_w, d_x, ncols, nrows, scale, act);                      \
        } else if (ncols <= 1024u) {                                           \
            dim3 gg((nrows * 16u + 255u) / 256u);                              \
            wgvn_bf16v_t<16, NT><<<gg, 256, 0, g_str>>>(                       \
                d_y, d_w, d_x, ncols, nrows, scale, act);                      \
        } else {                                                               \
            dim3 wg((nrows + 7u) / 8u);                                        \
            wgvn_bf16v_t<32, NT><<<wg, 256, 0, g_str>>>(                       \
                d_y, d_w, d_x, ncols, nrows, scale, act);                      \
        }                                                                      \
    } while (0)
    switch (nt) {
    case 2: BF16N_ONE(2); break;
    case 3: BF16N_ONE(3); break;
    case 4: BF16N_ONE(4); break;
    case 5: BF16N_ONE(5); break;
    case 6: BF16N_ONE(6); break;
    case 7: BF16N_ONE(7); break;
    case 8: BF16N_ONE(8); break;
    default: return false;
    }
#undef BF16N_ONE
    return hipGetLastError() == hipSuccess;
}

static bool launch(uint32_t type, const uint8_t *d_w, uint64_t nrows,
                   uint64_t ncols, const float *d_x, float *d_y, float scale,
                   uint32_t n_batch, const int32_t *which) {
    if (n_batch == 0) n_batch = 1;
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[g_gemv_phase][type]++;
        g_gemv_bytes[g_gemv_phase][type] += rb * nrows;
    }
    static int shapes = -1;
    if (shapes < 0) shapes = getenv("Q4_SHAPES") != NULL;
    static int bigsh = -1;
    if (bigsh < 0) bigsh = getenv("Q4_BIGSHAPE") != NULL;
    if (bigsh && nrows > 100000u)
        fprintf(stderr, "BIGSHAPE ty=%u r=%llu c=%llu nb=%llu\n", type,
                (unsigned long long)nrows, (unsigned long long)ncols,
                (unsigned long long)n_batch);
    if (shapes) {
        bool dup = false;
        for (int i = 0; i < g_shape_n; i++)
            if (g_shapes[i].t == type && g_shapes[i].r == nrows &&
                g_shapes[i].c == ncols && g_shapes[i].b == n_batch) {
                g_shapes[i].n++;
                dup = true;
            }
        if (!dup && g_shape_n < 256) {
            g_shapes[g_shape_n].t = type;
            g_shapes[g_shape_n].r = nrows;
            g_shapes[g_shape_n].c = ncols;
            g_shapes[g_shape_n].b = n_batch;
            g_shapes[g_shape_n].n = 1;
            g_shape_n++;
        }
    }
#define GEMV_DONE() return true
    /* Library dense path: for large-token batches dequant the weight to
     * fp16 (bf16 weights used directly) and run hipBLASLt/hipBLAS —
     * measured ~5-10x the warp-per-row kernels at T=8192.  Q4_HGEMM=0
     * restores the kernels below bit-identically. */
    if (n_batch >= 64 && !which && !g_capturing && q4_hgemm_on() &&
        q4_hgemm_try(type, d_w, nrows, ncols, d_x, d_y, scale, n_batch))
        GEMV_DONE();
    /* Decode GEMV (single token, no which remap): warp-per-row kernels.  The
     * fp sum order differs from the 64-thread path — bounded reorder noise,
     * same class as the f32-sk path.  Q4_WARPGV=0 falls back to the old
     * kernels bit-identically. */
    if (n_batch == 1 && !which && wgv_on()) {
        const dim3 wg((uint32_t)((nrows + 7u) / 8u));
        switch (type) {
        case Q4_T_Q8_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_q8_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale, 0.f);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_XXS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq2xxs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_XS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq2xs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq2_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq3_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_XXS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq3xxs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_XS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            if (!kq16_launch(type, d_y, d_w, d_x, (uint32_t)ncols,
                             (uint32_t)nrows, scale, 1))
            wgv_iq4xs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_NL:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_iq4_nl_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q4_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_q4_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_q5_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q2_0:
            if (ncols % 64u) break;
            GEV_PRE(type);
            wgv_q2_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_F32:
            /* keep the split-K path for tiny outputs: it wins there */
            if ((nrows <= GEMV_SK_MAXROWS && nrows < 96u && ncols >= 1024 &&
                 g_sk_part) ||
                ncols % 4u)
                break;
            GEV_PRE(type);
            wgv_f32_k<<<wg, 256, 0, g_str>>>(d_y, (const float *)d_w, d_x,
                                             (uint32_t)ncols, (uint32_t)nrows,
                                             scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_BF16:
            if (!(ncols % 8u) && !((uintptr_t)d_w & 15u) &&
                !((uintptr_t)d_x & 15u)) {
                GEV_PRE(type);
                if (nrows <= 1024u && ncols >= 2048u) {
                    /* narrow output, wide input: one block per row keeps
                     * every CU busy (hc_*_down, ssm_alpha/beta, indexer_k) */
                    gemv_bf16_bk<<<(uint32_t)nrows, 256, 0, g_str>>>(
                        d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                        scale, 0.f);
                } else if (ncols <= 1024u) {
                    /* short rows (hc_*_up in=320/512): sub-warp groups so a
                     * warp retires several rows per reduce */
                    const uint32_t G = ncols <= 512u ? 8u : 16u;
                    const dim3 gg((nrows * G + 255u) / 256u);
                    if (G == 8u)
                        wgv_bf16v_k<8><<<gg, 256, 0, g_str>>>(
                            d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                            scale, 0.f);
                    else
                        wgv_bf16v_k<16><<<gg, 256, 0, g_str>>>(
                            d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                            scale, 0.f);
                } else {
                    wgv_bf16v_k<32><<<wg, 256, 0, g_str>>>(
                        d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                        scale, 0.f);
                }
                GEV_POST();
                Q4_HIP_CHECK(hipGetLastError());
                GEMV_DONE();
            }
            GEV_PRE(type);
            wgv_bf16_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q6_K:
            /* float4 x loads need 16B; fall through to the block kernel
             * when a caller hands an unaligned activation. */
            if (ncols % 256u || ((uintptr_t)d_x & 15u)) break;
            GEV_PRE(type);
            if (!(q6t_mode() == 2 &&
                  q6t_launch(d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                             scale, 1)))
            wgv_q6_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q4_K:
            if (ncols % 256u || ((uintptr_t)d_x & 15u)) break;
            GEV_PRE(type);
            if (!kq16_launch(type, d_y, d_w, d_x, (uint32_t)ncols,
                             (uint32_t)nrows, scale, 1))
            wgv_q4_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_K:
            if (ncols % 256u || ((uintptr_t)d_x & 15u)) break;
            GEV_PRE(type);
            if (!kq16_launch(type, d_y, d_w, d_x, (uint32_t)ncols,
                             (uint32_t)nrows, scale, 1))
            wgv_q5_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        default:
            break;
        }
    }
    /* Small multi-token batches (MTP verify, short prefill): warp-per-row
     * with one accumulator per token, same per-token add order as wgv —
     * each row is bit-identical to the single-token call.  Q4_WGVN=0
     * restores the old paths. */
    if (n_batch >= 2 && n_batch <= WGVN_MAX && !which && wgv_on() &&
        wgvn_on() &&
        wgvt_launch(type, d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                    scale, n_batch)) {
        Q4_HIP_CHECK(hipGetLastError());
        GEMV_DONE();
    }
    if (n_batch >= 2 && n_batch <= WGVN_MAX && !which && wgv_on() &&
        wgvn_on()) {
        const dim3 wg((uint32_t)((nrows + 7u) / 8u));
        switch (type) {
        case Q4_T_Q8_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_q8_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_XXS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq2xxs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                 (uint32_t)nrows, scale,
                                                 n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_XS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq2xs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale,
                                                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq2_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale,
                                                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq3_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale,
                                                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_XXS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq3xxs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                 (uint32_t)nrows, scale,
                                                 n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_XS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq4xs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale,
                                                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_NL:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_iq4_nl_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                 (uint32_t)nrows, scale,
                                                 n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q4_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_q4_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_q5_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q2_0:
            if (ncols % 64u) break;
            GEV_PRE(type);
            wgvn_q2_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_F32:
            if (ncols % 4u) break;
            GEV_PRE(type);
            wgvn_f32_k<<<wg, 256, 0, g_str>>>(d_y, (const float *)d_w, d_x,
                                              (uint32_t)ncols, (uint32_t)nrows,
                                              scale, n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_BF16:
            GEV_PRE(type);
            if (!bf16n_launch(d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                              scale, 0.f, n_batch))
                wgvn_bf16_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x,
                                                   (uint32_t)ncols,
                                                   (uint32_t)nrows, scale,
                                                   n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q6_K:
            /* Per token, wdot_q6k — same row as the single-token warp
             * kernel.  Unaligned x stays on the block-per-token path. */
            if (ncols % 256u || ((uintptr_t)d_x & 15u)) break;
            GEV_PRE(type);
            wgvn_q6_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                             (uint32_t)nrows, scale, n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q4_K:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_q4_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_K:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_q5_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        default:
            break;
        }
    }
    /* New-type dense batches: WMMA beats the reuse kernel once enough tokens
     * amortize the tile setup. which-gathers still take the reuse path. */
    if ((type == Q4_T_Q4_0 || type == Q4_T_Q5_0 || type == Q4_T_Q2_0 ||
         type == Q4_T_IQ4_XS) &&
        n_batch >= 16 && !wmma_off() && !which &&
        (type == Q4_T_IQ4_XS ? ncols % 256u == 0 : ncols % 64u == 0)) {
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
               (n_batch + 63u) / 64u);
        if (type == Q4_T_Q4_0)
            gemm_q40w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        else if (type == Q4_T_Q5_0)
            gemm_q50w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        else if (type == Q4_T_IQ4_XS)
            gemm_iq4xsw_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, n_batch,
                                                scale);
        else
            gemm_q20w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    /* Several tokens share an expert. Dequant the row once (element-parallel),
     * then warp-per-token dots — faster than the per-thread serial-dot GEMV
     * even for a single token. */
    if (ncols <= 4096u && ncols * sizeof(float) <= 48u * 1024u) {
        size_t shmem = (size_t)ncols * sizeof(float);
        if (type == Q4_T_IQ2_XXS && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq2xxs_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ2_XS && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq2xs_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ2_S && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq2_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ3_S && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq3_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ4_NL && ncols % 32u == 0) {
            GEV_PRE(type);
            gemm_iq4_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ3_XXS && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq3xxs_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ4_XS && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq4xs_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_Q4_0 && ncols % 32u == 0) {
            GEV_PRE(type);
            gemm_q40_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_Q5_0 && ncols % 32u == 0) {
            GEV_PRE(type);
            gemm_q50_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_Q2_0 && ncols % 64u == 0) {
            GEV_PRE(type);
            gemm_q20_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
    }
    /* gemm_q8_* reads x[0..n_batch) in order. A which-gather (expert rows
     * inside a prefill group) must not take it — that silently dots the
     * wrong tokens once an expert is shared by >=16 rows. */
    if (type == Q4_T_Q8_0 && n_batch >= 16 && ncols % 32u == 0 && !which) {
        if (!wmma_off() && ncols % 16u == 0) {
            GEV_PRE(type);
            dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
                   (n_batch + 63u) / 64u);
            gemm_q8w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                           (uint32_t)nrows, n_batch, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            return true;
        }
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + 31u) / 32u), (n_batch + 31u) / 32u);
        gemm_q8_0_k<<<g, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                                n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    if (type == Q4_T_Q6_K && n_batch >= 16 && ncols % 256u == 0 &&
        !wmma_off() && !which) {
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
               (n_batch + 63u) / 64u);
        gemm_q6w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                         (uint32_t)nrows, n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    if ((type == Q4_T_Q4_K || type == Q4_T_Q5_K) && n_batch >= 16 &&
        ncols % 256u == 0 && !wmma_off() && !which) {
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
               (n_batch + 63u) / 64u);
        if (type == Q4_T_Q4_K)
            gemm_q4kw_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        else
            gemm_q5kw_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    uint32_t gy = n_batch < 64u ? n_batch : 64u;
    dim3 grid((uint32_t)nrows, gy);
    dim3 block(64);
    GEV_PRE(type);
    switch (type) {
    case Q4_T_Q8_0:
        gemv_q8_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q4_K:
        gemv_q4_k_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q5_1:
        gemv_q5_1_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q5_K:
        gemv_q5_k_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q6_K:
        if (ncols % 256u) return false;
        gemv_q6_k_k<<<grid, dim3(128), 0, g_str>>>(d_y, d_w, d_x,
                                     (uint32_t)ncols, (uint32_t)nrows, scale,
                                     n_batch, which);
        break;
    case Q4_T_IQ2_XXS:
        if (ncols % 256u) return false;
        gemv_iq2xxs_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ2_XS:
        if (ncols % 256u) return false;
        gemv_iq2xs_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                      (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        gemv_iq2_s_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                      (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        gemv_iq3_s_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                      (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        gemv_iq3xxs_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        gemv_iq4xs_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                      (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        gemv_iq4_nl_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        gemv_q4_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        gemv_q5_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        gemv_q2_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_F32:
        static int f32pf = -1;
        if (f32pf < 0)
            f32pf = !(getenv("Q4_F32_PF") && getenv("Q4_F32_PF")[0] == '0');
        if (n_batch > 1 && f32pf) {
            dim3 g32((unsigned)((nrows + 31u) / 32u),
                     (unsigned)((n_batch + 31u) / 32u));
            gemm_tiled_k<float><<<g32, 256, 0, g_str>>>(
                d_y, (const float *)d_w, d_x, (uint32_t)ncols,
                (uint32_t)nrows, scale, n_batch, which);
        } else if (n_batch == 1 && !which && nrows <= GEMV_SK_MAXROWS &&
                   nrows < 96u && ncols >= 1024 && g_sk_part) {
            dim3 gs((uint32_t)nrows, GEMV_SK_S);
            gemv_f32_sk<<<gs, 256, 0, g_str>>>(g_sk_part,
                                               (const float *)d_w, d_x,
                                               (uint32_t)ncols, GEMV_SK_S);
            gemv_sk_sum<<<(uint32_t)nrows, 32, 0, g_str>>>(d_y, g_sk_part,
                                                         (uint32_t)nrows,
                                                         GEMV_SK_S, scale);
        } else {
            gemv_f32_k<<<grid, block, 0, g_str>>>(d_y, (const float *)d_w, d_x,
                                        (uint32_t)ncols, (uint32_t)nrows, scale,
                                        n_batch, which);
        }
        break;
    case Q4_T_BF16:
        if (n_batch > 1) {
            /* Opt-in only: measured 457.1 vs 440.8-443.7 tok/s on a
             * 2048-tok prefill — these bf16 projections are
             * bandwidth-bound, so the WMMA path nets a small loss.
             * Q4_GEMM_BF16W=1 enables it for revisiting. */
            static int bf16w = -1;
            if (bf16w < 0)
                bf16w = (getenv("Q4_GEMM_BF16W") &&
                         getenv("Q4_GEMM_BF16W")[0] == '1');
            if (bf16w && ncols % 16u == 0) {
                dim3 g((unsigned)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
                       (unsigned)((n_batch + Q8W_TOKS - 1) / Q8W_TOKS));
                gemm_wmma_bf16_k<<<g, 128, 0, g_str>>>(
                    d_y, (const uint16_t *)d_w, d_x, (uint32_t)ncols,
                    (uint32_t)nrows, scale, n_batch, which);
                break;
            }
            dim3 g32((unsigned)((nrows + 31u) / 32u),
                     (unsigned)((n_batch + 31u) / 32u));
            gemm_tiled_k<uint16_t><<<g32, 256, 0, g_str>>>(
                d_y, (const uint16_t *)d_w, d_x, (uint32_t)ncols,
                (uint32_t)nrows, scale, n_batch, which);
        } else {
            gemv_bf16_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x,
                                                   (uint32_t)ncols,
                                                   (uint32_t)nrows, scale,
                                                   n_batch, which);
        }
        break;
    default:
        return false;
    }
    GEV_POST();
    Q4_HIP_CHECK(hipGetLastError());
    GEMV_DONE();
}

extern "C" bool q4_hip_gemv_dev(uint32_t ggml_type, const uint8_t *d_w,
                                uint64_t nrows, uint64_t ncols, const float *x,
                                float *y, float scale) {
    if (!g_ok || !d_w || !x || !y || nrows == 0 || ncols == 0) return false;
    if (!grow((void **)&g_dx, &g_dx_n, ncols * sizeof(float))) return false;
    if (!grow((void **)&g_dy, &g_dy_n, nrows * sizeof(float))) return false;
    if (!sync_copy(g_dx, x, ncols * sizeof(float), hipMemcpyHostToDevice))
        return false;
    if (!launch(ggml_type, d_w, nrows, ncols, g_dx, g_dy, scale, 1, NULL)) return false;
    return sync_copy(y, g_dy, nrows * sizeof(float), hipMemcpyDeviceToHost);
}

extern "C" bool q4_hip_gemv(uint32_t ggml_type, const uint8_t *w, uint64_t nrows,
                            uint64_t ncols, const float *x, float *y,
                            float scale) {
    if (!g_ok || !w) return false;
    uint64_t rb = q4_row_bytes(ggml_type, ncols);
    if (rb == 0) return false;
    size_t wn = (size_t)rb * (size_t)nrows;
    if (!grow((void **)&g_dw, &g_dw_n, wn)) return false;
    if (!sync_copy(g_dw, w, wn, hipMemcpyHostToDevice)) return false;
    return q4_hip_gemv_dev(ggml_type, g_dw, nrows, ncols, x, y, scale);
}

extern "C" bool q4_hip_gemv_dd(uint32_t ggml_type, const uint8_t *d_w,
                               uint64_t nrows, uint64_t ncols, const float *d_x,
                               float *d_y, float scale) {
    if (!g_ok || !d_w || !d_x || !d_y || nrows == 0 || ncols == 0) return false;
    return launch(ggml_type, d_w, nrows, ncols, d_x, d_y, scale, 1, NULL);
}

/* Warp GEMV with a fused epilogue: out = silu(dot * scale * act) when
 * act != 0 — absorbs the scale_silu elementwise pass between the two
 * hc_mix GEMVs.  Only the q8_0 warp path supports it; callers fall back
 * to gemv + scale_silu. */
extern "C" bool q4_hip_gemv_dd_act(uint32_t ggml_type, const uint8_t *d_w,
                                   uint64_t nrows, uint64_t ncols,
                                   const float *d_x, float *d_y, float scale,
                                   float act) {
    if (!g_ok || !d_w || !d_x || !d_y || nrows == 0 || ncols == 0) return false;
    if (!wgv_on() || act == 0.f) return false;
    if (ggml_type == Q4_T_Q8_0 && ncols % 32u == 0) {
        const dim3 wg((uint32_t)((nrows + 7u) / 8u));
        wgv_q8_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                          (uint32_t)nrows, scale, act);
        return hipGetLastError() == hipSuccess;
    }
    /* bf16 with the silu epilogue: mirrors the launch() dispatch — one
     * kernel instead of gemv + scale_silu (hc_*_down in the decode head
     * and hc_mix paths). */
    if (ggml_type == Q4_T_BF16 && !(ncols % 8u) &&
        !((uintptr_t)d_w & 15u) && !((uintptr_t)d_x & 15u)) {
        if (nrows <= 1024u && ncols >= 2048u) {
            gemv_bf16_bk<<<(uint32_t)nrows, 256, 0, g_str>>>(
                d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows, scale, act);
        } else if (ncols <= 1024u) {
            const uint32_t G = ncols <= 512u ? 8u : 16u;
            const dim3 gg((nrows * G + 255u) / 256u);
            if (G == 8u)
                wgv_bf16v_k<8><<<gg, 256, 0, g_str>>>(
                    d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows, scale,
                    act);
            else
                wgv_bf16v_k<16><<<gg, 256, 0, g_str>>>(
                    d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows, scale,
                    act);
        } else {
            const dim3 wg((uint32_t)((nrows + 7u) / 8u));
            wgv_bf16v_k<32><<<wg, 256, 0, g_str>>>(
                d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows, scale, act);
        }
        return hipGetLastError() == hipSuccess;
    }
    return false;
}

/* n_batch 2..8 bf16 with the same silu epilogue.  Other types and wider
 * batches return false so the caller keeps gemv + scale_silu. */
extern "C" bool q4_hip_gemv_dd_n_act(uint32_t ggml_type, const uint8_t *d_w,
                                     uint64_t nrows, uint64_t ncols,
                                     const float *d_x, float *d_y, float scale,
                                     float act, uint32_t n_batch) {
    if (!g_ok || !d_w || !d_x || !d_y || !nrows || !ncols || !n_batch ||
        act == 0.f)
        return false;
    if (n_batch == 1)
        return q4_hip_gemv_dd_act(ggml_type, d_w, nrows, ncols, d_x, d_y,
                                  scale, act);
    if (n_batch > WGVN_MAX || ggml_type != Q4_T_BF16 || !wgv_on() ||
        !wgvn_on())
        return false;
    return bf16n_launch(d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                        scale, act, n_batch);
}

/* Q4_BF16_PAIR=0 keeps two launches.  Off while GEMV profiling counts
 * per-call bytes (the callers fall back to the two single calls). */
static int g_bf16_pair = -1;
extern "C" void q4_hip_bf16_pair_set(int v) { g_bf16_pair = v; }
extern "C" bool q4_hip_gemv_bf16_n2(const uint8_t *w0, float *y0, uint32_t r0,
                                    float act0, const uint8_t *w1, float *y1,
                                    uint32_t r1, float act1, const float *d_x,
                                    uint32_t ncols, uint32_t nt) {
    if (g_bf16_pair < 0) {
        const char *e = getenv("Q4_BF16_PAIR");
        g_bf16_pair = !(e && e[0] == '0');
    }
    if (!g_ok || !g_bf16_pair || gemv_mode() || !w0 || !w1 || !y0 || !y1 ||
        !d_x || !r0 || !r1 || nt < 2 || nt > WGVN_MAX || !wgv_on() ||
        !wgvn_on() || !bf16n_on())
        return false;
    /* bf16n_launch's gemv_bf16_bn_t shape, both tensors */
    if (r0 > 1024u || r1 > 1024u || ncols < 2048u || (ncols & 7u) ||
        ((uintptr_t)w0 & 15u) || ((uintptr_t)w1 & 15u) ||
        ((uintptr_t)d_x & 15u))
        return false;
#define BF16N2_ONE(NT)                                                         \
    gemv_bf16_bn2_t<NT><<<r0 + r1, 256, 0, g_str>>>(y0, w0, r0, act0, y1, w1,  \
                                                    r1, act1, d_x, ncols)
    switch (nt) {
    case 2: BF16N2_ONE(2); break;
    case 3: BF16N2_ONE(3); break;
    case 4: BF16N2_ONE(4); break;
    case 5: BF16N2_ONE(5); break;
    case 6: BF16N2_ONE(6); break;
    case 7: BF16N2_ONE(7); break;
    case 8: BF16N2_ONE(8); break;
    default: return false;
    }
#undef BF16N2_ONE
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_gemv_dd_n(uint32_t ggml_type, const uint8_t *d_w,
                                 uint64_t nrows, uint64_t ncols, const float *d_x,
                                 float *d_y, float scale, uint32_t n_batch) {
    return q4_hip_gemv_dd_n_which(ggml_type, d_w, nrows, ncols, d_x, d_y, scale,
                                  n_batch, NULL);
}

extern "C" bool q4_hip_gemv_dd_n_which(uint32_t ggml_type, const uint8_t *d_w,
                                       uint64_t nrows, uint64_t ncols,
                                       const float *d_x, float *d_y, float scale,
                                       uint32_t n_batch, const int32_t *which) {
    if (!g_ok || !d_w || !d_x || !d_y || nrows == 0 || ncols == 0) return false;
    if (!which && n_batch >= 1 && n_batch <= 8 &&
        q4_mmvq_n(ggml_type, d_w, nrows, ncols, d_x, d_y, scale, n_batch))
        return true;
    return launch(ggml_type, d_w, nrows, ncols, d_x, d_y, scale,
                  n_batch ? n_batch : 1, which);
}

/* Q8_0 row reused across a gathered token list. The weight row is dequanted
 * into shared memory once, then every token dots that copy. A PCIe-resident
 * expert otherwise re-reads the same row from host memory once per token
 * (gemv_q8_0_k). which == NULL means x is already token-major contiguous.
 * Addition order differs from gemv_q8_0_k by a few ulps; callers that need
 * the bit-identical GEMV stay on launch(). */
__global__ static void q8_reuse_k(float *out, const uint8_t *w, const float *x,
                                  const int32_t *which, uint32_t in_dim,
                                  uint32_t out_dim, float scale,
                                  uint32_t n_batch) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    extern __shared__ float sh[];
    float *rowf = sh;
    float *red = sh + in_dim;
    const uint32_t nb = in_dim >> 5;
    const uint8_t *wr = w + ((size_t)row * nb) * 34ull;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const uint32_t b = i >> 5;
        const uint32_t j = i & 31u;
        const uint8_t *blk = wr + (size_t)b * 34ull;
        __half dh;
        memcpy(&dh, blk, 2);
        const float d = __half2float(dh);
        rowf[i] = d * (float)(int8_t)blk[2u + j];
    }
    __syncthreads();
    for (uint32_t tok = 0; tok < n_batch; tok++) {
        const int32_t src = which ? which[tok] : (int32_t)tok;
        const float *xt = x + (size_t)src * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            acc += rowf[i] * xt[i];
        red[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = red[0] * scale;
        __syncthreads();
    }
}

extern "C" bool q4_hip_q8_reuse(const uint8_t *d_w, uint64_t nrows,
                                uint64_t ncols, const float *d_x,
                                const int32_t *which, float *d_y, float scale,
                                uint32_t n_batch) {
    if (!g_ok || !d_w || !d_x || !d_y || !nrows || !ncols || !n_batch)
        return false;
    if ((ncols & 31u) || ncols > 4096u) return false;
    /* row dequant plus a 256-wide reduction scratch */
    size_t shmem = ((size_t)ncols + 256u) * sizeof(float);
    if (shmem > 48u * 1024u) return false;
    q8_reuse_k<<<(uint32_t)nrows, 256, shmem, g_str>>>(
        d_y, d_w, d_x, which, (uint32_t)ncols, (uint32_t)nrows, scale, n_batch);
    return hipGetLastError() == hipSuccess;
}

__global__ static void unary_silu_k(float *d, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = d[i];
        d[i] = v / (1.f + expf(-v));
    }
}
__global__ static void unary_sig_k(float *d, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 1.f / (1.f + expf(-d[i]));
}
__global__ static void unary_scale_k(float *d, float s, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= s;
}
__global__ static void mul_k(float *y, const float *a, const float *b, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}
__global__ static void add_k(float *y, const float *a, const float *b, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ static void axpy_k(float *y, const float *x, float a, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
__global__ static void fill_k(float *d, float v, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}
__global__ static void copy_k(float *d, const float *s, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}

static bool launch1(uint64_t n, dim3 *g, dim3 *b) {
    *b = dim3(256);
    *g = dim3((uint32_t)((n + 255) / 256));
    return n > 0;
}

extern "C" bool q4_hip_silu(float *d, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !d || !launch1(n, &g, &b)) return false;
    unary_silu_k<<<g, b, 0, g_str>>>(d, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_sigmoid(float *d, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !d || !launch1(n, &g, &b)) return false;
    unary_sig_k<<<g, b, 0, g_str>>>(d, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_scale(float *d, float s, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !d || !launch1(n, &g, &b)) return false;
    unary_scale_k<<<g, b, 0, g_str>>>(d, s, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_mul(float *y, const float *a, const float *b, uint64_t n) {
    dim3 g, bl;
    if (!g_ok || !y || !a || !b || !launch1(n, &g, &bl)) return false;
    mul_k<<<g, bl, 0, g_str>>>(y, a, b, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_add(float *y, const float *a, const float *b, uint64_t n) {
    dim3 g, bl;
    if (!g_ok || !y || !a || !b || !launch1(n, &g, &bl)) return false;
    add_k<<<g, bl, 0, g_str>>>(y, a, b, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_axpy(float *y, const float *x, float a, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !y || !x || !launch1(n, &g, &b)) return false;
    axpy_k<<<g, b, 0, g_str>>>(y, x, a, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_fill(float *d, float v, uint64_t n) {
    if (!g_ok || !d || n == 0) return false;
    if (v == 0.f && !g_capturing && !g_str) {
        Q4_HIP_CHECK(hipMemset(d, 0, n * sizeof(float)));
        return true;
    }
    dim3 g, b;
    launch1(n, &g, &b);
    fill_k<<<g, b, 0, g_str>>>(d, v, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_copy(float *d, const float *s, uint64_t n) {
    if (!g_ok || !d || !s || n == 0) return false;
    return sync_copy(d, s, n * sizeof(float), hipMemcpyDeviceToDevice);
}

/* Q4_COPY_SYNC=1 keeps the draining copy for a bisect. Default queues the
 * D2D and returns; the caller syncs at the next real host read. */
/* Up to 8 independent float copies in one launch (the verify's per-layer
 * GDN stash: four D2D copies that were four blit dispatches). */
#define Q4_MCOPY_MAX 8
struct q4_mcopy_args {
    uint32_t n;
    float *d[Q4_MCOPY_MAX];
    const float *s[Q4_MCOPY_MAX];
    uint64_t len[Q4_MCOPY_MAX];
};
__global__ static void mcopy_k(q4_mcopy_args a) {
    const uint32_t seg = blockIdx.y;
    if (seg >= a.n) return;
    const uint64_t len = a.len[seg];
    float *d = a.d[seg];
    const float *s = a.s[seg];
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < len;
         i += (uint64_t)gridDim.x * blockDim.x)
        d[i] = s[i];
}

extern "C" bool q4_hip_copy_multi(uint32_t n, float *const *d,
                                  const float *const *s, const uint64_t *len) {
    if (!g_ok || !n || n > Q4_MCOPY_MAX || !d || !s || !len) return false;
    q4_mcopy_args a;
    memset(&a, 0, sizeof(a));
    a.n = n;
    uint64_t mx = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!d[i] || !s[i]) return false;
        a.d[i] = d[i];
        a.s[i] = s[i];
        a.len[i] = len[i];
        if (len[i] > mx) mx = len[i];
    }
    if (!mx) return true;
    uint64_t bx = (mx + 255) / 256;
    if (bx > 256) bx = 256;
    mcopy_k<<<dim3((unsigned)bx, n), 256, 0, g_str>>>(a);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_copy_async(float *d, const float *s, uint64_t n) {
    if (!g_ok || !d || !s || n == 0) return false;
    static int drain = -1;
    if (drain < 0) {
        const char *e = getenv("Q4_COPY_SYNC");
        drain = e && e[0] == '1';
    }
    if (drain || !g_str)
        return sync_copy(d, s, n * sizeof(float), hipMemcpyDeviceToDevice);
    return hipMemcpyAsync(d, s, n * sizeof(float), hipMemcpyDeviceToDevice,
                          g_str) == hipSuccess;
}

__global__ static void grouped_rms_k(const float *x, const float *w, float *y,
                                     uint32_t n_embd, uint32_t hc, float eps) {
    uint32_t c = blockIdx.x;
    uint32_t tok = blockIdx.y;
    const float *xc = x + ((size_t)tok * hc + c) * n_embd;
    float *yc = y + ((size_t)tok * hc + c) * n_embd;
    const float *wc = w ? w + (size_t)c * n_embd : nullptr;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        ss += xc[i] * xc[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)n_embd + eps);
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        yc[i] = xc[i] * scale * (wc ? wc[i] : 1.f);
}

extern "C" bool q4_hip_grouped_rms_n(const float *x, const float *w, float *y,
                                     uint32_t n_embd, uint32_t hc, float eps,
                                     uint32_t n_tok) {
    if (!g_ok || !x || !y || n_embd == 0 || hc == 0 || n_tok == 0) return false;
    grouped_rms_k<<<dim3(hc, n_tok), 256, 0, g_str>>>(x, w, y, n_embd, hc, eps);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_grouped_rms(const float *x, const float *w, float *y,
                                   uint32_t n_embd, uint32_t hc, float eps) {
    return q4_hip_grouped_rms_n(x, w, y, n_embd, hc, eps, 1);
}

__global__ static void mean_hc_k(const float *gated, float *mixed, uint32_t n_embd,
                                 uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tok = blockIdx.y;
    if (i >= n_embd) return;
    const float *g = gated + (size_t)tok * hc * n_embd;
    float s = 0.f;
    for (uint32_t c = 0; c < hc; c++) s += g[(size_t)c * n_embd + i];
    mixed[(size_t)tok * n_embd + i] = s / (float)hc;
}
extern "C" bool q4_hip_mean_hc_n(const float *gated, float *mixed, uint32_t n_embd,
                                 uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !gated || !mixed) return false;
    dim3 b(256), g((n_embd + 255) / 256, n_tok ? n_tok : 1);
    mean_hc_k<<<g, b, 0, g_str>>>(gated, mixed, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_mean_hc(const float *gated, float *mixed, uint32_t n_embd,
                               uint32_t hc) {
    return q4_hip_mean_hc_n(gated, mixed, n_embd, hc, 1);
}

__global__ static void hc_combine_k(float *res, const float *block,
                                    const float *inject, uint32_t n_embd,
                                    uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tok = blockIdx.y;
    if (i >= n_embd) return;
    float inv = 1.f / (float)hc;
    float *rt = res + (size_t)tok * hc * n_embd;
    const float *bt = block + (size_t)tok * n_embd;
    const float *it = inject + (size_t)tok * hc;
    for (uint32_t c = 0; c < hc; c++) {
        float w = 2.f / (1.f + expf(-it[c] * inv));
        rt[(size_t)c * n_embd + i] += bt[i] * w;
    }
}
extern "C" bool q4_hip_hc_combine_n(float *res, const float *block,
                                    const float *inject, uint32_t n_embd,
                                    uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !res || !block || !inject) return false;
    dim3 b(256), g((n_embd + 255) / 256, n_tok ? n_tok : 1);
    hc_combine_k<<<g, b, 0, g_str>>>(res, block, inject, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_hc_combine(float *res, const float *block,
                                  const float *inject, uint32_t n_embd,
                                  uint32_t hc) {
    return q4_hip_hc_combine_n(res, block, inject, n_embd, hc, 1);
}

/* hc_combine + grouped_rms in one launch: one block per residual group
 * applies res[c] += block*w_c, then normalizes the updated group into y.
 * Bit-identical to the two separate kernels (same per-element math, same
 * strided reduction order); the re-read of rc[] after the reduction is an
 * L2 hit. */
__global__ static void hc_comb_rms_k(float *res, const float *block,
                                     const float *inject, const float *wn,
                                     float *y, uint32_t n_embd, uint32_t hc,
                                     float eps, const float *hy) {
    const uint32_t c = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    const size_t base = ((size_t)tok * hc + c) * n_embd;
    float *rc = res + base;
    const float *bt = block + (size_t)tok * n_embd;
    /* hy: the MoE's CPU-miss rows, merged here instead of by an axpy
     * into block first (same sum: block + 1.f * hy). */
    const float *ht = hy ? hy + (size_t)tok * n_embd : nullptr;
    const float w = 2.f / (1.f + expf(-inject[(size_t)tok * hc + c] *
                                      (1.f / (float)hc)));
    __shared__ float sh[256];
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float b = ht ? fmaf(1.f, ht[i], bt[i]) : bt[i];
        const float v = rc[i] + b * w;
        rc[i] = v;
        ss += v * v;
    }
    if (!y) return;
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(sh[0] / (float)n_embd + eps);
    const float *wc = wn ? wn + (size_t)c * n_embd : nullptr;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        y[base + i] = rc[i] * scale * (wc ? wc[i] : 1.f);
}
extern "C" bool q4_hip_hc_comb_rms(float *res, const float *block,
                                   const float *inject, const float *wn,
                                   float *y, uint32_t n_embd, uint32_t hc,
                                   float eps) {
    if (!g_ok || !res || !block || !inject || !y) return false;
    hc_comb_rms_k<<<dim3(hc, 1), 256, 0, g_str>>>(res, block, inject, wn, y,
                                                  n_embd, hc, eps, nullptr);
    return hipGetLastError() == hipSuccess;
}
/* hc_comb_rms_k with one float4 per thread (blockDim = n_embd/4): the
 * 256-thread version walks each group in two strided passes with a load
 * round trip per step — ~44 us for a 4-row verify on 16 blocks.  Here a
 * thread loads its 4 values once and keeps them in registers.  The sum of
 * squares reduces in a different order than grouped_rms_k (float-level
 * differences, not bit-identical). */
__global__ static void hc_comb_rms4_k(float *res, const float *block,
                                      const float *inject, const float *wn,
                                      float *y, uint32_t n_embd, uint32_t hc,
                                      float eps, const float *hy) {
    const uint32_t c = blockIdx.x, tok = blockIdx.y, i = threadIdx.x;
    const size_t base = ((size_t)tok * hc + c) * n_embd;
    float4 *rc = (float4 *)(res + base);
    const float4 b = ((const float4 *)(block + (size_t)tok * n_embd))[i];
    float4 h = make_float4(0.f, 0.f, 0.f, 0.f);
    if (hy) h = ((const float4 *)(hy + (size_t)tok * n_embd))[i];
    const float w = 2.f / (1.f + expf(-inject[(size_t)tok * hc + c] *
                                      (1.f / (float)hc)));
    float4 v = rc[i];
    v.x += (hy ? fmaf(1.f, h.x, b.x) : b.x) * w;
    v.y += (hy ? fmaf(1.f, h.y, b.y) : b.y) * w;
    v.z += (hy ? fmaf(1.f, h.z, b.z) : b.z) * w;
    v.w += (hy ? fmaf(1.f, h.w, b.w) : b.w) * w;
    rc[i] = v;
    if (!y) return;
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    for (uint32_t o = 16; o; o >>= 1) ss += __shfl_xor(ss, o);
    __shared__ float sh[32];
    if ((i & 31u) == 0) sh[i >> 5] = ss;
    __syncthreads();
    if (i < 32) {
        float t = i < (blockDim.x >> 5) ? sh[i] : 0.f;
        for (uint32_t o = 16; o; o >>= 1) t += __shfl_xor(t, o);
        if (i == 0) sh[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(sh[0] / (float)n_embd + eps);
    float4 o4 = make_float4(v.x * scale, v.y * scale, v.z * scale,
                            v.w * scale);
    if (wn) {
        const float4 wc = ((const float4 *)(wn + (size_t)c * n_embd))[i];
        o4.x *= wc.x; o4.y *= wc.y; o4.z *= wc.z; o4.w *= wc.w;
    }
    ((float4 *)(y + base))[i] = o4;
}

/* n_tok rows; y == NULL: combine only (+hy merge), no normalization. */
extern "C" bool q4_hip_hc_comb_rms_n(float *res, const float *block,
                                     const float *hy, const float *inject,
                                     const float *wn, float *y,
                                     uint32_t n_embd, uint32_t hc, float eps,
                                     uint32_t n_tok) {
    if (!g_ok || !res || !block || !inject || !n_tok) return false;
    static int v4 = -1;
    if (v4 < 0) v4 = !(getenv("Q4_COMB_V4") && getenv("Q4_COMB_V4")[0] == '0');
    if (v4 && (n_embd & 127u) == 0 && n_embd / 4u <= 1024u &&
        !((uintptr_t)res & 15u) && !((uintptr_t)block & 15u) &&
        !((uintptr_t)hy & 15u) && !((uintptr_t)y & 15u) &&
        !((uintptr_t)wn & 15u)) {
        hc_comb_rms4_k<<<dim3(hc, n_tok), n_embd / 4u, 0, g_str>>>(
            res, block, inject, wn, y, n_embd, hc, eps, hy);
        return hipGetLastError() == hipSuccess;
    }
    hc_comb_rms_k<<<dim3(hc, n_tok), 256, 0, g_str>>>(res, block, inject, wn,
                                                      y, n_embd, hc, eps, hy);
    return hipGetLastError() == hipSuccess;
}

__global__ static void repeat_hc_k(const float *emb, float *res, uint32_t n_embd,
                                   uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float v = emb[i];
    for (uint32_t c = 0; c < hc; c++) res[(size_t)c * n_embd + i] = v;
}
extern "C" bool q4_hip_repeat_hc(const float *emb, float *res, uint32_t n_embd,
                                 uint32_t hc) {
    if (!g_ok || !emb || !res) return false;
    dim3 b(256), g((n_embd + 255) / 256);
    repeat_hc_k<<<g, b, 0, g_str>>>(emb, res, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

/* emb is [n_tok, n_embd]; res is [n_tok, hc, n_embd]. Same per-token copy
 * as repeat_hc_k. */
__global__ static void repeat_hc_n_k(const float *emb, float *res,
                                     uint32_t n_embd, uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tok = blockIdx.y;
    if (i >= n_embd) return;
    float v = emb[(size_t)tok * n_embd + i];
    float *rt = res + ((size_t)tok * hc) * n_embd;
    for (uint32_t c = 0; c < hc; c++) rt[(size_t)c * n_embd + i] = v;
}
extern "C" bool q4_hip_repeat_hc_n(const float *emb, float *res, uint32_t n_embd,
                                   uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !emb || !res || !n_embd || !hc || !n_tok) return false;
    dim3 b(256), g((n_embd + 255) / 256, n_tok);
    repeat_hc_n_k<<<g, b, 0, g_str>>>(emb, res, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

/* cat row r = [e[r] | h[r]], r in [0, n_tok*hc). */
__global__ static void eh_cat_k(float *cat, const float *e, const float *h,
                                uint32_t n_embd) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (i >= n_embd) return;
    float *dst = cat + (size_t)row * (n_embd * 2u);
    dst[i] = e[(size_t)row * n_embd + i];
    dst[n_embd + i] = h[(size_t)row * n_embd + i];
}
extern "C" bool q4_hip_eh_cat(float *cat, const float *e, const float *h,
                              uint32_t n_embd, uint32_t n_rows) {
    if (!g_ok || !cat || !e || !h || !n_embd || !n_rows) return false;
    dim3 b(256), g((n_embd + 255) / 256, n_rows);
    eh_cat_k<<<g, b, 0, g_str>>>(cat, e, h, n_embd);
    return hipGetLastError() == hipSuccess;
}

__global__ static void dot_k(const float *a, const float *b, float *out, uint32_t n) {
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += a[i] * b[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = sh[0];
}
extern "C" bool q4_hip_dot(const float *a, const float *b, float *out, uint32_t n) {
    if (!g_ok || !a || !b || !out) return false;
    dot_k<<<1, 256, 0, g_str>>>(a, b, out, n);
    return hipGetLastError() == hipSuccess;
}

__global__ static void gdn_conv_k(const float *hist, const float *qkv,
                                  const float *w, float *out, uint32_t ch,
                                  uint32_t ksz) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float acc = 0.f;
    for (uint32_t k = 0; k + 1 < ksz; k++)
        acc += w[k + c * ksz] * hist[k * ch + c];
    acc += w[(ksz - 1) + c * ksz] * qkv[c];
    out[c] = acc / (1.f + expf(-acc));
}

/* Fused conv + history shift: one thread owns channel c end-to-end so the
 * read-then-update of hist needs no extra kernel (saves 1 launch/layer). */
__global__ static void gdn_convshift_k(float *hist, const float *qkv,
                                       const float *w, float *out, uint32_t ch,
                                       uint32_t ksz) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float acc = 0.f;
    for (uint32_t k = 0; k + 1 < ksz; k++)
        acc += w[k + c * ksz] * hist[k * ch + c];
    acc += w[(ksz - 1) + c * ksz] * qkv[c];
    out[c] = acc / (1.f + expf(-acc));
    for (uint32_t k = 0; k + 2 < ksz; k++)
        hist[k * ch + c] = hist[(k + 1) * ch + c];
    hist[(ksz - 2) * ch + c] = qkv[c];
}
extern "C" bool q4_hip_gdn_convshift(float *hist, const float *qkv,
                                     const float *w, float *out, uint32_t ch,
                                     uint32_t ksz) {
    if (!g_ok || !hist || !qkv || !w || !out || ksz < 2) return false;
    dim3 b(256), g((ch + 255) / 256);
    gdn_convshift_k<<<g, b, 0, g_str>>>(hist, qkv, w, out, ch, ksz);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_gdn_conv(const float *hist, const float *qkv, const float *w,
                                float *out, uint32_t ch, uint32_t ksz) {
    if (!g_ok || !hist || !qkv || !w || !out) return false;
    dim3 b(256), g((ch + 255) / 256);
    gdn_conv_k<<<g, b, 0, g_str>>>(hist, qkv, w, out, ch, ksz);
    return hipGetLastError() == hipSuccess;
}

__global__ static void gdn_shift_k(float *hist, const float *qkv, uint32_t ch,
                                   uint32_t ksz) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    for (uint32_t k = 0; k + 2 < ksz; k++)
        hist[k * ch + c] = hist[(k + 1) * ch + c];
    hist[(ksz - 2) * ch + c] = qkv[c];
}
extern "C" bool q4_hip_gdn_shift_hist(float *hist, const float *qkv, uint32_t ch,
                                      uint32_t ksz) {
    if (!g_ok || !hist || !qkv || ksz < 2) return false;
    dim3 b(256), g((ch + 255) / 256);
    gdn_shift_k<<<g, b, 0, g_str>>>(hist, qkv, ch, ksz);
    return hipGetLastError() == hipSuccess;
}

__global__ static void l2_heads_k(float *x, uint32_t d, float eps) {
    uint32_t h = blockIdx.x;
    float *v = x + (size_t)h * d;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += v[i] * v[i];
    __shared__ float sh[128];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] + eps);
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) v[i] *= scale;
}
extern "C" bool q4_hip_l2_heads(float *x, uint32_t n_h, uint32_t d, float eps) {
    if (!g_ok || !x || n_h == 0) return false;
    l2_heads_k<<<n_h, 128, 0, g_str>>>(x, d, eps);
    return hipGetLastError() == hipSuccess;
}

__device__ static inline float softplus_d(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return expf(x);
    return logf(1.f + expf(x));
}

__global__ static void gdn_step_k(float *S, const float *q, const float *k,
                                  const float *v, const float *beta,
                                  const float *alpha, const float *A,
                                  const float *dt, float *y, uint32_t n_kh,
                                  uint32_t d, float qscale) {
    uint32_t vh = blockIdx.x;
    uint32_t col = threadIdx.x;
    if (col >= d) return;
    /* GGUF V heads are tiled: pair V[vh] with K[vh % n_k]. */
    uint32_t kh = n_kh ? (vh % n_kh) : 0;
    const float *qh = q + kh * d;
    const float *khv = k + kh * d;
    float gv = expf(A[vh] * softplus_d(alpha[vh] + dt[vh]));
    float b = 1.f / (1.f + expf(-beta[vh])); /* sigmoid folded: raw logits in */
    float *Sh = S + (size_t)vh * d * d;
    float kv = 0.f;
    for (uint32_t i = 0; i < d; i++) kv += Sh[i * d + col] * khv[i];
    float delta = (v[vh * d + col] - gv * kv) * b;
    float att = 0.f;
    for (uint32_t i = 0; i < d; i++) {
        float s = gv * Sh[i * d + col] + khv[i] * delta;
        Sh[i * d + col] = s;
        att += s * qh[i];
    }
    y[vh * d + col] = att * qscale;
}
extern "C" bool q4_hip_gdn_step(float *S, const float *q, const float *k,
                                const float *v, const float *beta,
                                const float *alpha, const float *A,
                                const float *dt, float *y, uint32_t n_vh,
                                uint32_t n_kh, uint32_t d, float qscale) {
    if (!g_ok || !S || !q || !k || !v || !beta || !alpha || !A || !dt || !y)
        return false;
    gdn_step_k<<<n_vh, d, 0, g_str>>>(S, q, k, v, beta, alpha, A, dt, y, n_kh, d, qscale);
    return hipGetLastError() == hipSuccess;
}

/* Generation-count grid barrier: safe here because nblocks (≤48) is far
 * below the resident limit, so every block coexists on the device.
 * g_bar is declared near the top of this file. */
__device__ static void gbar(uint32_t *cnt, volatile uint32_t *fl,
                            uint32_t nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t gen = *fl;
        if (atomicAdd(cnt, 1u) + 1 == nb) {
            *cnt = 0;
            atomicExch((uint32_t *)fl, gen + 1);
        } else {
            while (*fl == gen) __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
}

/* Whole GDN tail in ONE launch: conv+hist-shift, q/k l2 norms, gated
 * delta-net step, rms+sigmoid(z) gate — all per-value-head work lives inside
 * a single block so no inter-block sync is needed. Requires d_k == d_v.
 * The hist update waits on a grid barrier: q/k channel ranges are shared by
 * several vh blocks, so every block must finish reading before anyone
 * writes. */
__global__ static void gdn_tail_k(float *hist, const float *qkv,
                                  const float *cw, uint32_t ch, uint32_t ksz,
                                  float *S, const float *beta,
                                  const float *alpha, const float *A,
                                  const float *dt, const float *nw,
                                  const float *z, float *core, uint32_t n_kh,
                                  uint32_t d, float eps, float qscale,
                                  uint32_t *bar) {
    uint32_t vh = blockIdx.x;
    uint32_t col = threadIdx.x;
    if (col >= d) return;
    uint32_t kh = n_kh ? (vh % n_kh) : 0;
    __shared__ float qs[128], ks[128], vs[128], sh[128];
    const uint32_t base[3] = { kh * d, d * n_kh + kh * d,
                               2 * d * n_kh + vh * d };
    float *dst[3] = { qs, ks, vs };
    /* conv + silu for this block's q,k,v channel slices (read-only pass) */
    for (int s = 0; s < 3; s++) {
        uint32_t c = base[s] + col;
        float acc = 0.f;
        for (uint32_t j = 0; j + 1 < ksz; j++)
            acc += cw[j + c * ksz] * hist[j * ch + c];
        acc += cw[(ksz - 1) + c * ksz] * qkv[c];
        dst[s][col] = acc / (1.f + expf(-acc));
    }
    /* all conv reads done everywhere → hist updates are now race-free */
    gbar(bar, bar + 1, gridDim.x);
    /* q/k channel updates: only the first block of each kh group writes
     * (the three vh blocks sharing a kh would otherwise triple-write). */
    if (vh < n_kh) {
        for (int s = 0; s < 2; s++) {
            uint32_t c = base[s] + col;
            for (uint32_t j = 0; j + 2 < ksz; j++)
                hist[j * ch + c] = hist[(j + 1) * ch + c];
            hist[(ksz - 2) * ch + c] = qkv[c];
        }
    }
    /* v channels are block-private: every block shifts its own slice */
    {
        uint32_t c = base[2] + col;
        for (uint32_t j = 0; j + 2 < ksz; j++)
            hist[j * ch + c] = hist[(j + 1) * ch + c];
        hist[(ksz - 2) * ch + c] = qkv[c];
    }
    __syncthreads();
    /* l2-normalize q and k head slices (v stays raw) */
    for (int s = 0; s < 2; s++) {
        float *v = dst[s];
        sh[col] = v[col] * v[col];
        __syncthreads();
        for (uint32_t t = d >> 1; t > 0; t >>= 1) {
            if (col < t) sh[col] += sh[col + t];
            __syncthreads();
        }
        float sc = rsqrtf(sh[0] + eps);
        __syncthreads();
        v[col] *= sc;
        __syncthreads();
    }
    /* gated delta-net step: beta sigmoid is applied here (raw logits in) */
    float gv = expf(A[vh] * softplus_d(alpha[vh] + dt[vh]));
    float b = 1.f / (1.f + expf(-beta[vh]));
    float *Sh = S + (size_t)vh * d * d;
    float kv = 0.f;
    for (uint32_t i = 0; i < d; i++) kv += Sh[i * d + col] * ks[i];
    float delta = (vs[col] - gv * kv) * b;
    float att = 0.f;
    for (uint32_t i = 0; i < d; i++) {
        float s = gv * Sh[i * d + col] + ks[i] * delta;
        Sh[i * d + col] = s;
        att += s * qs[i];
    }
    float y = att * qscale;
    /* rms + sigmoid(z) gate epilogue, written straight to core */
    sh[col] = y * y;
    __syncthreads();
    for (uint32_t t = d >> 1; t > 0; t >>= 1) {
        if (col < t) sh[col] += sh[col + t];
        __syncthreads();
    }
    float sc = rsqrtf(sh[0] / (float)d + eps);
    float zz = z[(size_t)vh * d + col];
    core[(size_t)vh * d + col] =
        (y * sc * (nw ? nw[col] : 1.f)) / (1.f + expf(-zz));
}
extern "C" bool q4_hip_gdn_tail(float *hist, const float *qkv, const float *cw,
                                uint32_t ch, uint32_t ksz, float *S,
                                const float *beta, const float *alpha,
                                const float *A, const float *dt,
                                const float *nw, const float *z, float *core,
                                uint32_t n_vh, uint32_t n_kh, uint32_t d,
                                float eps, float qscale) {
    if (!g_ok || !g_bar || !hist || !qkv || !cw || !S || !beta || !alpha ||
        !A || !dt || !z || !core || !n_vh || !n_kh || n_vh % n_kh ||
        d != 128 || ksz < 2)
        return false;
    gdn_tail_k<<<n_vh, d, 0, g_str>>>(hist, qkv, cw, ch, ksz, S, beta, alpha,
                                      A, dt, nw, z, core, n_kh, d, eps,
                                      qscale, g_bar);
    return hipGetLastError() == hipSuccess;
}

/* Whole-chunk GDN scan (prefill): one launch per layer. Block vh keeps its
 * column of the delta-net state S in registers across all T tokens and a
 * private conv-history slice in shared, so the per-token launches and the
 * grid barrier of gdn_tail_k disappear. Math is op-for-op identical:
 * conv+silu -> hist shift -> l2(q,k) -> delta step -> rms+sigmoid(z) gate. */
template <bool W>
__global__ static void gdn_scan_k(float *hist, const float *qkv,
                                  const float *cw, uint32_t ch, uint32_t ksz,
                                  float *S, const float *beta,
                                  const float *alpha, const float *A,
                                  const float *dt, const float *nw,
                                  const float *z, float *core, uint32_t n_tok,
                                  uint32_t n_kh, uint32_t d, float eps,
                                  float qscale, float *qkh,
                                  float *Ssnap = nullptr,
                                  float *hsnap = nullptr,
                                  float *stash = nullptr,
                                  uint32_t stash_cap = 0) {
    constexpr uint32_t D = 128, KMAX = 8, QS = D / 4;
    uint32_t vh = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t col = tid >> 2;        /* 4 threads share each column */
    uint32_t qt = tid & 3;          /* this thread's row quarter */
    if (col >= d || ksz > KMAX) return;
    uint32_t n_vh = gridDim.x;
    uint32_t kh = n_kh ? (vh % n_kh) : 0;
    const uint32_t base[3] = { kh * d, d * n_kh + kh * d,
                               2 * d * n_kh + vh * d };
    /* MTP verify stash of the token inputs (q4_hip_gdn_scan_stash): each
     * block copies its v slice, z slice and beta/alpha scalars; block kh
     * also its q/k slices (shared by the blocks of that kh).  Layout as
     * the caller's copy_multi: qkv rows, z rows, beta, alpha, each
     * stash_cap rows apart. */
    if (stash) {
        float *sq = stash, *sz = stash + (size_t)stash_cap * ch;
        float *sb = sz + (size_t)stash_cap * n_vh * d;
        float *sa = sb + (size_t)stash_cap * n_vh;
        for (uint32_t e = tid; e < n_tok * d; e += blockDim.x) {
            const uint32_t t = e / d, i = e - t * d;
            const size_t r = (size_t)t * ch;
            sq[r + base[2] + i] = qkv[r + base[2] + i];
            if (vh < n_kh) {
                sq[r + base[0] + i] = qkv[r + base[0] + i];
                sq[r + base[1] + i] = qkv[r + base[1] + i];
            }
            const size_t zr = (size_t)t * n_vh * d + (size_t)vh * d + i;
            sz[zr] = z[zr];
        }
        if (tid < n_tok) {
            sb[(size_t)tid * n_vh + vh] = beta[(size_t)tid * n_vh + vh];
            sa[(size_t)tid * n_vh + vh] = alpha[(size_t)tid * n_vh + vh];
        }
    }
    __shared__ float qs[D], ks[D], vs[D], sh[2 * D];
    __shared__ float h[KMAX][3 * D]; /* private hist rows for q,k,v slices */
    /* Load this block's hist slices (q,k of its kh; v of its vh). */
    for (uint32_t e = tid; e < 3 * D; e += blockDim.x) {
        uint32_t s = e / D, cc = e - s * D;
        for (uint32_t j = 0; j + 1 < ksz; j++) {
            const float hv = hist[(size_t)j * ch + base[s] + cc];
            h[j][e] = hv;
            /* snapshot of the incoming conv state (q/k slices are shared
             * by several blocks: identical values) */
            if (hsnap) hsnap[(size_t)j * ch + base[s] + cc] = hv;
        }
    }
    float *Sh = S + (size_t)vh * d * d;
    float scol[QS];
    /* W: S (and its snapshot) move as 32-row x 128-column chunks through
     * LDS with 16-byte coalesced accesses; a thread's quarter-column (rows
     * qt*32.., column col) is one chunk.  The direct per-thread loads hit 4
     * rows x 8 columns per wave instruction.  W also reduces the l2 / rms
     * sums per wave (2 barriers per token instead of ~20).  Verify, 4
     * rows: ~29 us per layer -> see AGENTS.md.  Q4_GDN_SCAN_W=0: old. */
    __shared__ float4 sc4[QS * D / 4];
    if (W) {
        for (uint32_t k = 0; k < 4; k++) {
            const float4 *src = (const float4 *)(Sh + (size_t)k * QS * d);
            for (uint32_t e = tid; e < QS * D / 4; e += blockDim.x) {
                const float4 v4 = src[e];
                sc4[e] = v4;
                if (Ssnap)
                    ((float4 *)(Ssnap + (size_t)vh * d * d +
                                (size_t)k * QS * d))[e] = v4;
            }
            __syncthreads();
            if (qt == k) {
                const float *sf = (const float *)sc4;
#pragma unroll
                for (uint32_t i = 0; i < QS; i++) scol[i] = sf[i * D + col];
            }
            __syncthreads();
        }
    } else {
#pragma unroll
        for (uint32_t i = 0; i < QS; i++) {
            scol[i] = Sh[(size_t)(qt * QS + i) * d + col];
            if (Ssnap) Ssnap[(size_t)vh * d * d + (size_t)(qt * QS + i) * d + col] = scol[i];
        }
    }
    __syncthreads();
    __shared__ float wq[16], wk[16], wy[16];
    const uint32_t wv = tid >> 5;
    /* W: this thread's conv taps and its channel of the (<= 8) token rows
     * are loaded once, ahead of the dependent per-token steps. */
    constexpr uint32_t TPF = 8;
    float cwr[KMAX] = {}, xpf[TPF] = {};
    const uint32_t e0 = tid < 3 * D ? tid : 0;
    const uint32_t c0 = base[e0 / D] + (e0 - (e0 / D) * D);
    if (W && blockDim.x >= 3 * D && tid < 3 * D) {
#pragma unroll
        for (uint32_t j = 0; j < KMAX; j++)
            if (j < ksz) cwr[j] = cw[j + (size_t)c0 * ksz];
#pragma unroll
        for (uint32_t t = 0; t < TPF; t++)
            if (t < n_tok) xpf[t] = qkv[(size_t)t * ch + c0];
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *x = qkv + (size_t)t * ch;
        /* conv + silu on the private slices, then private shift+append */
        float pq = 0.f, pk = 0.f;
        for (uint32_t e = tid; e < 3 * D; e += blockDim.x) {
            uint32_t s = e / D, cc = e - s * D;
            uint32_t c = base[s] + cc;
            const bool pf = W && blockDim.x >= 3 * D && t < TPF;
            float acc = 0.f;
            float xc;
            if (pf) {
                float xt = 0.f;
#pragma unroll
                for (uint32_t u = 0; u < TPF; u++)
                    if (u == t) xt = xpf[u];
                xc = xt;
                float cl = 0.f;
#pragma unroll
                for (uint32_t j = 0; j < KMAX; j++) {
                    if (j + 1 < ksz) acc += cwr[j] * h[j][e];
                    if (j == ksz - 1) cl = cwr[j];
                }
                acc += cl * xc;
            } else {
                for (uint32_t j = 0; j + 1 < ksz; j++)
                    acc += cw[j + (size_t)c * ksz] * h[j][e];
                xc = x[c];
                acc += cw[(ksz - 1) + (size_t)c * ksz] * xc;
            }
            float v = acc / (1.f + expf(-acc));
            (s == 0 ? qs : s == 1 ? ks : vs)[cc] = v;
            if (s == 0) pq = v * v;
            else if (s == 1) pk = v * v;
            for (uint32_t j = 0; j + 2 < ksz; j++)
                h[j][e] = h[j + 1][e];
            h[ksz - 2][e] = xc;
        }
        float scq = 1.f, sck = 1.f;
        if (W) {
            for (uint32_t o = 16; o; o >>= 1) {
                pq += __shfl_xor(pq, o);
                pk += __shfl_xor(pk, o);
            }
            if ((tid & 31) == 0 && wv < 16) {
                wq[wv] = pq;
                wk[wv] = pk;
            }
            __syncthreads();
            float sq = 0.f, sk = 0.f;
#pragma unroll
            for (uint32_t i = 0; i < 16; i++) {
                sq += wq[i];
                sk += wk[i];
            }
            scq = rsqrtf(sq + eps);
            sck = rsqrtf(sk + eps);
        } else {
            __syncthreads();
            /* l2-normalize q and k in shared (v stays raw): both sums in
             * one reduction pass. */
            if (tid < D) {
                sh[tid] = qs[tid] * qs[tid];
                sh[D + tid] = ks[tid] * ks[tid];
            }
            __syncthreads();
            for (uint32_t r = D >> 1; r > 0; r >>= 1) {
                if (tid < r) {
                    sh[tid] += sh[tid + r];
                    sh[D + tid] += sh[D + tid + r];
                }
                __syncthreads();
            }
            float s1 = rsqrtf(sh[0] + eps), s2 = rsqrtf(sh[D] + eps);
            __syncthreads();
            if (tid < D) {
                qs[tid] *= s1;
                ks[tid] *= s2;
            }
            __syncthreads();
        }
        /* gated delta-net step on the register quarter-column */
        float gv = expf(A[vh] * softplus_d(alpha[(size_t)t * n_vh + vh] + dt[vh]));
        float b = 1.f / (1.f + expf(-beta[(size_t)t * n_vh + vh]));
        float kv = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < QS; i++)
            kv += scol[i] * (ks[qt * QS + i] * sck);
        kv += __shfl_xor(kv, 1);
        kv += __shfl_xor(kv, 2);
        float delta = (vs[col] - gv * kv) * b;
        float att = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < QS; i++) {
            float s = gv * scol[i] + (ks[qt * QS + i] * sck) * delta;
            scol[i] = s;
            att += s * (qs[qt * QS + i] * scq);
        }
        att += __shfl_xor(att, 1);
        att += __shfl_xor(att, 2);
        float y = att * qscale;
        /* rms + sigmoid(z) gate epilogue straight to core */
        float sc;
        if (W) {
            float py = qt == 0 ? y * y : 0.f;
            for (uint32_t o = 16; o; o >>= 1) py += __shfl_xor(py, o);
            if ((tid & 31) == 0 && wv < 16) wy[wv] = py;
            __syncthreads();
            float sy = 0.f;
#pragma unroll
            for (uint32_t i = 0; i < 16; i++) sy += wy[i];
            sc = rsqrtf(sy / (float)d + eps);
        } else {
            if (qt == 0) sh[col] = y * y;
            __syncthreads();
            for (uint32_t r = D >> 1; r > 0; r >>= 1) {
                if (tid < r) sh[tid] += sh[tid + r];
                __syncthreads();
            }
            sc = rsqrtf(sh[0] / (float)d + eps);
        }
        if (qt == 0) {
            float zz = z[(size_t)t * n_vh * d + (size_t)vh * d + col];
            core[(size_t)t * n_vh * d + (size_t)vh * d + col] =
                (y * sc * (nw ? nw[col] : 1.f)) / (1.f + expf(-zz));
        }
        if (!W) __syncthreads();
    }
    /* Write back the final state: S quarter-column and the private hist rows.
     * The q/k rows are shared across kh groups and read at kernel start by
     * every block, so they must not be overwritten in place — they go to the
     * qkh scratch and are copied back by the host after this launch. The v
     * rows are block-private and stay in place. */
    if (W) {
        for (uint32_t k = 0; k < 4; k++) {
            __syncthreads();
            if (qt == k) {
                float *sf = (float *)sc4;
#pragma unroll
                for (uint32_t i = 0; i < QS; i++) sf[i * D + col] = scol[i];
            }
            __syncthreads();
            float4 *dst = (float4 *)(Sh + (size_t)k * QS * d);
            for (uint32_t e = tid; e < QS * D / 4; e += blockDim.x)
                dst[e] = sc4[e];
        }
    } else {
#pragma unroll
        for (uint32_t i = 0; i < QS; i++)
            Sh[(size_t)(qt * QS + i) * d + col] = scol[i];
    }
    for (uint32_t e = tid; e < 3 * D; e += blockDim.x) {
        uint32_t s = e / D, cc = e - s * D;
        if (s < 2 && vh >= n_kh) continue;
        float *dst = s < 2 ? qkh : hist;
        for (uint32_t j = 0; j + 1 < ksz; j++)
            dst[(size_t)j * ch + base[s] + cc] = h[j][e];
    }
}
/* Pipelined GDN scan (Q4_GDN_REC, default on, n_tok >= 64): same math as
 * gdn_scan_k, restructured (MIT; see README References).  Four stages:
 * per-channel causal conv tiled over tokens (the conv reads inputs, so the
 * 64-token tiles are independent and the hist update is its own pass),
 * q/k l2 norms over (t, slice) blocks, the delta-net recurrence split by
 * 4 column blocks per value head (192 blocks instead of 48, LDS-staged q/k
 * with the next token's inputs fetched while this one computes), and a
 * per-(t, head) output norm — the only step coupling a head's columns. */
constexpr uint32_t GDN_TILE = 64;
__global__ static void gdn_gates2_k(const float *alpha, const float *beta_in,
                                    const float *A, const float *dt,
                                    float *gate, float *betag, uint32_t n_vh,
                                    uint32_t n_tok) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_tok * n_vh) return;
    const uint32_t h = (uint32_t)(i % n_vh);
    const float v = alpha[i] + dt[h];
    gate[i] = A[h] * (v > 20.f ? v : log1pf(expf(v)));
    betag[i] = 1.f / (1.f + expf(-beta_in[i]));
}
__global__ static void gdn_conv_tile_k(const float *hist, const float *qkv,
                                       const float *cw, float *h, uint32_t ch,
                                       uint32_t ksz, uint32_t n_tok) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch || ksz > 8u) return;
    const uint32_t t0 = blockIdx.y * GDN_TILE;
    if (t0 >= n_tok) return;
    const uint32_t t1 = t0 + GDN_TILE < n_tok ? t0 + GDN_TILE : n_tok;
    const uint32_t km1 = ksz - 1;
    float v[7];
    for (uint32_t j = 0; j < km1; j++) {
        const int64_t u = (int64_t)t0 - (int64_t)km1 + j;
        v[j] = u >= 0 ? qkv[(size_t)u * ch + c]
                      : hist[(size_t)(u + km1) * ch + c];
    }
    for (uint32_t t = t0; t < t1; t++) {
        const float x = qkv[(size_t)t * ch + c];
        float acc = 0.f;
        for (uint32_t j = 0; j < km1; j++)
            acc += cw[j + (size_t)c * ksz] * v[j];
        acc += cw[km1 + (size_t)c * ksz] * x;
        h[(size_t)t * ch + c] = acc / (1.f + expf(-acc));
        for (uint32_t j = 0; j + 1 < km1; j++) v[j] = v[j + 1];
        v[km1 - 1] = x;
    }
}
__global__ static void gdn_conv_hist_k(float *hist, const float *qkv,
                                       uint32_t ch, uint32_t ksz,
                                       uint32_t n_tok) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch || ksz > 8u) return;
    const uint32_t km1 = ksz - 1;
    float v[7];
    for (uint32_t j = 0; j < km1; j++) {
        const int64_t u = (int64_t)n_tok - (int64_t)km1 + j;
        v[j] = u >= 0 ? qkv[(size_t)u * ch + c]
                      : hist[(size_t)(u + km1) * ch + c];
    }
    for (uint32_t j = 0; j < km1; j++)
        hist[(size_t)j * ch + c] = v[j];
}
__global__ static void gdn_l2_k(float *h, uint32_t ch, float eps) {
    float *x = h + (size_t)blockIdx.y * ch + (size_t)blockIdx.x * 128;
    const float v = x[threadIdx.x];
    float sq = v * v;
    sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8); sq += __shfl_xor(sq, 4);
    sq += __shfl_xor(sq, 2);  sq += __shfl_xor(sq, 1);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    x[threadIdx.x] = v * rsqrtf(part[0] + part[1] + part[2] + part[3] + eps);
}
constexpr uint32_t GDN_S = 128, GDN_RG = 4, GDN_RPG = GDN_S / GDN_RG,
                   GDN_CB = 32, GDN_NCB = GDN_S / GDN_CB;
__global__ static void __launch_bounds__(GDN_CB * GDN_RG)
    gdn_rec_pipe_k(float *S, const float *h, const float *gate,
                   const float *betag, float *oc, uint32_t ch, uint32_t n_vh,
                   uint32_t n_kh, float qscale, uint32_t n_tok) {
    constexpr uint32_t NT = GDN_CB * GDN_RG, LPT = GDN_S / NT;
    __shared__ float sk[GDN_S], sq[GDN_S], red[GDN_RG][GDN_CB];
    const uint32_t head = blockIdx.x / GDN_NCB, cb = blockIdx.x % GDN_NCB;
    const uint32_t c = threadIdx.x, rg = threadIdx.y,
                   tid = rg * GDN_CB + c, col = cb * GDN_CB + c;
    const uint32_t qh = n_kh ? head % n_kh : 0;
    float s[GDN_RPG];
    float *base = S + (size_t)head * GDN_S * GDN_S +
                  (size_t)rg * GDN_RPG * GDN_S + col;
#pragma unroll
    for (uint32_t r = 0; r < GDN_RPG; ++r) s[r] = base[r * GDN_S];
    float nq[LPT], nk[LPT], nv = 0.f, ng = 0.f, nb = 0.f;
    auto fetch = [&](uint32_t t) {
        const float *ht = h + (size_t)t * ch;
#pragma unroll
        for (uint32_t u = 0; u < LPT; ++u) {
            nq[u] = ht[(size_t)qh * GDN_S + tid + u * NT];
            nk[u] = ht[((size_t)n_kh + qh) * GDN_S + tid + u * NT];
        }
        nv = ht[(size_t)2 * n_kh * GDN_S + (size_t)head * GDN_S + col];
        ng = gate[(size_t)t * n_vh + head];
        nb = betag[(size_t)t * n_vh + head];
    };
    fetch(0);
    for (uint32_t t = 0; t < n_tok; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (uint32_t u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        __syncthreads();
#pragma unroll
        for (uint32_t u = 0; u < LPT; ++u) {
            sq[tid + u * NT] = cq[u];
            sk[tid + u * NT] = ck[u];
        }
        __syncthreads();
        if (t + 1 < n_tok) fetch(t + 1);
        const float g = expf(cg);
        float kv = 0.f;
#pragma unroll
        for (uint32_t r = 0; r < GDN_RPG; ++r)
            kv = fmaf(s[r], sk[rg * GDN_RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.f;
#pragma unroll
        for (uint32_t r = 0; r < GDN_RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * GDN_RPG + r] * delta);
            o = fmaf(s[r], sq[rg * GDN_RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0)
            oc[(size_t)t * n_vh * GDN_S + (size_t)head * GDN_S + col] =
                (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * qscale;
    }
#pragma unroll
    for (uint32_t r = 0; r < GDN_RPG; ++r) base[r * GDN_S] = s[r];
}
__global__ static void __launch_bounds__(GDN_S)
    gdn_out_norm_k(const float *oc, const float *z, const float *nw,
                   float *core, uint32_t n_vh, float eps) {
    __shared__ float wsum[4];
    const uint32_t head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t)blockIdx.x * n_vh * GDN_S +
                      (size_t)head * GDN_S + col;
    const float y = oc[at];
    float sp = y * y;
    sp += __shfl_xor(sp, 16); sp += __shfl_xor(sp, 8); sp += __shfl_xor(sp, 4);
    sp += __shfl_xor(sp, 2);  sp += __shfl_xor(sp, 1);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float v = y * rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) /
                                   (float)GDN_S + eps) *
                    (nw ? nw[col] : 1.f) / (1.f + expf(-z[at]));
    core[at] = v;
}

/* Bytes of the rec-path scan scratch for n_tok rows: h, oc, gates. */
extern "C" void q4_hip_gdn_scan_need(uint32_t n_tok, uint32_t ch,
                                     uint32_t n_vh, uint32_t d,
                                     uint64_t need[3]) {
    need[0] = (uint64_t)n_tok * ch * sizeof(float);
    need[1] = (uint64_t)n_tok * n_vh * d * sizeof(float);
    need[2] = 2ull * n_tok * n_vh * sizeof(float);
}

/* Point the rec-path scratch at borrowed memory (p non-NULL: three
 * buffers of the q4_hip_gdn_scan_need sizes) or drop it (NULL).  The
 * first call frees any grown buffers and disables growth for good. */
extern "C" void q4_hip_gdn_scan_lend(void *const p[3], const uint64_t n[3]) {
    if (!g_scan_lend) {
        if (g_scanh) (void)hipFree(g_scanh);
        if (g_scanoc) (void)hipFree(g_scanoc);
        if (g_scangb) (void)hipFree(g_scangb);
        g_scan_lend = 1;
    }
    g_scanh = p ? (float *)p[0] : nullptr;
    g_scanoc = p ? (float *)p[1] : nullptr;
    g_scangb = p ? (float *)p[2] : nullptr;
    g_scanh_n = p ? (size_t)n[0] : 0;
    g_scanoc_n = p ? (size_t)n[1] : 0;
    g_scangb_n = p ? (size_t)n[2] : 0;
}

/* One-shot: the next q4_hip_gdn_scan (n_tok < 64 kernel) also writes the
 * incoming S / conv state to these (the MTP verify's rollback snapshot,
 * in place of a separate 113 MB copy).  Returns false when that call
 * would not take the single-kernel path. */
static float *g_scan_Ssnap, *g_scan_hsnap;
static float *g_scan_stash;
static uint32_t g_scan_stash_cap;
/* One-shot like gdn_scan_snap: the next single-kernel scan also copies
 * its token inputs (qkv / z / beta / alpha) to the MTP stash at sb (rows
 * cap apart).  Returns false when that scan would not take the
 * single-kernel path — the caller copies instead. */
extern "C" bool q4_hip_gdn_scan_stash(float *sb, uint32_t cap,
                                      uint32_t n_tok) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("Q4_STASH_SCAN");
        on = !(e && e[0] == '0');
    }
    if (!on || n_tok >= 64 || !sb || n_tok > cap) return false;
    g_scan_stash = sb;
    g_scan_stash_cap = cap;
    return true;
}
extern "C" bool q4_hip_gdn_scan_snap(float *Ssnap, float *hsnap,
                                     uint32_t n_tok) {
    if (n_tok >= 64 || !Ssnap || !hsnap) return false;
    g_scan_Ssnap = Ssnap;
    g_scan_hsnap = hsnap;
    return true;
}
extern "C" bool q4_hip_gdn_scan(float *hist, const float *qkv, const float *cw,
                                uint32_t ch, uint32_t ksz, float *S,
                                const float *beta, const float *alpha,
                                const float *A, const float *dt,
                                const float *nw, const float *z, float *core,
                                uint32_t n_tok, uint32_t n_vh, uint32_t n_kh,
                                uint32_t d, float eps, float qscale) {
    /* the stash request is one-shot whichever path runs */
    float *const stash = g_scan_stash;
    const uint32_t stash_cap = g_scan_stash_cap;
    g_scan_stash = nullptr;
    if (!g_ok || !hist || !qkv || !cw || !S || !beta || !alpha || !A || !dt ||
        !z || !core || !n_tok || !n_vh || !n_kh || n_vh % n_kh || d != 128 ||
        ksz < 2 || ksz > 8)
        return false;
    if (stash && n_tok >= 64) return false;
    const char *gr = getenv("Q4_GDN_REC");
    const int gdnrec = !gr || gr[0] != '0';
    if (gdnrec && n_tok >= 64 && g_scan_lend &&
        (!g_scanh || g_scanh_n < (size_t)n_tok * ch * sizeof(float) ||
         g_scanoc_n < (size_t)n_tok * n_vh * d * sizeof(float) ||
         g_scangb_n < 2 * (size_t)n_tok * n_vh * sizeof(float)))
        return false; /* no borrowed scratch: caller's per-token path */
    if (gdnrec && n_tok >= 64) {
        if (!grow((void **)&g_scanh, &g_scanh_n,
                  (size_t)n_tok * ch * sizeof(float)) ||
            !grow((void **)&g_scanoc, &g_scanoc_n,
                  (size_t)n_tok * n_vh * d * sizeof(float)) ||
            !grow((void **)&g_scangb, &g_scangb_n,
                  2 * (size_t)n_tok * n_vh * sizeof(float)))
            return false;
        float *gate = g_scangb, *betag = g_scangb + (size_t)n_tok * n_vh;
        gdn_gates2_k<<<(unsigned)((n_tok * n_vh + 255u) / 256u), 256, 0,
                       g_str>>>(alpha, beta, A, dt, gate, betag, n_vh, n_tok);
        const dim3 cgrid((ch + 127u) / 128u,
                         (n_tok + GDN_TILE - 1u) / GDN_TILE);
        gdn_conv_tile_k<<<cgrid, 128, 0, g_str>>>(hist, qkv, cw, g_scanh, ch,
                                                ksz, n_tok);
        gdn_conv_hist_k<<<(ch + 127u) / 128u, 128, 0, g_str>>>(hist, qkv, ch,
                                                             ksz, n_tok);
        gdn_l2_k<<<dim3(2 * n_kh, n_tok), 128, 0, g_str>>>(g_scanh, ch, eps);
        gdn_rec_pipe_k<<<n_vh * GDN_NCB, dim3(GDN_CB, GDN_RG), 0, g_str>>>(
            S, g_scanh, gate, betag, g_scanoc, ch, n_vh, n_kh, qscale, n_tok);
        gdn_out_norm_k<<<dim3(n_tok, n_vh), 128, 0, g_str>>>(g_scanoc, z, nw,
                                                           core, n_vh, eps);
        return hipGetLastError() == hipSuccess;
    }
    if (!grow((void **)&g_scanqk, &g_scanqk_n,
              (size_t)(ksz - 1) * ch * sizeof(float)))
        return false;
    static int scan_w = -1;
    if (scan_w < 0) {
        const char *e = getenv("Q4_GDN_SCAN_W");
        scan_w = !(e && e[0] == '0');
    }
    if (scan_w && d == 128)
        gdn_scan_k<true><<<n_vh, 512, 0, g_str>>>(hist, qkv, cw, ch, ksz, S, beta, alpha,
                                      A, dt, nw, z, core, n_tok, n_kh, d, eps,
                                      qscale, g_scanqk, g_scan_Ssnap,
                                      g_scan_hsnap, stash, stash_cap);
    else
        gdn_scan_k<false><<<n_vh, 512, 0, g_str>>>(hist, qkv, cw, ch, ksz, S, beta, alpha,
                                      A, dt, nw, z, core, n_tok, n_kh, d, eps,
                                      qscale, g_scanqk, g_scan_Ssnap,
                                      g_scan_hsnap, stash, stash_cap);
    g_scan_Ssnap = g_scan_hsnap = nullptr;
    if (hipGetLastError() != hipSuccess) return false;
    /* Stream-ordered copy-back of the shared q/k channel rows [0, 2*d*n_kh)
     * of every history row: one strided copy, not ksz-1 dispatches. */
    return hipMemcpy2DAsync(hist, (size_t)ch * sizeof(float), g_scanqk,
                            (size_t)ch * sizeof(float),
                            (size_t)2 * n_kh * d * sizeof(float), ksz - 1,
                            hipMemcpyDeviceToDevice, g_str) == hipSuccess;
}


__global__ static void rms_gate_heads_k(float *h, const float *w, const float *z,
                                        uint32_t d, float eps) {
    uint32_t vh = blockIdx.x;
    float *hv = h + (size_t)vh * d;
    const float *zv = z + (size_t)vh * d;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += hv[i] * hv[i];
    __shared__ float sh[128];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)d + eps);
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        float nrm = hv[i] * scale * (w ? w[i] : 1.f);
        hv[i] = nrm / (1.f + expf(-zv[i]));
    }
}
/* Publish the decode position for graph-captured kernels. Stream-ordered so
 * replays launched afterwards read the new value.  A stream write-value op
 * (not an async memcpy through g_pos_h) so a late-pushed value can't leak
 * into a still-pending earlier copy.  A failed write would leave graphed
 * layers decoding at a stale position — fall back to a kernel store and
 * report once rather than corrupting output silently. */
extern "C" void q4_hip_pos_push(int32_t pos) {
    if (!g_pos_d || !g_str) return;
    if (hipStreamWriteValue32(g_str, g_pos_d, (uint32_t)pos, 0) !=
        hipSuccess) {
        set_i32_k<<<dim3(1), dim3(32), 0, g_str>>>(g_pos_d, pos);
        if (hipGetLastError() != hipSuccess) {
            static int once = 0;
            if (!once++)
                fprintf(stderr, "q4: pos_push write failed — "
                                "graphed layers may see a stale position\n");
        }
    }
}
extern "C" void q4_hip_kv_push(int32_t kv) {
    if (!g_kv_d || !g_str) return;
    if (hipStreamWriteValue32(g_str, g_kv_d, (uint32_t)kv, 0) != hipSuccess) {
        set_i32_k<<<dim3(1), dim3(32), 0, g_str>>>(g_kv_d, kv);
        (void)hipGetLastError();
    }
}
extern "C" int q4_hip_kv_dev_ok(void) { return g_kv_d != nullptr; }
extern "C" int q4_hip_capturing(void) { return g_capturing; }
extern "C" int q4_hip_pos_dev_ok(void) {
    return g_pos_d != nullptr;
}

/* Image (M-RoPE) positions.  qwen4exp rotates with interleaved M-RoPE
 * (rope.dimension_sections 11/11/10): rotary pair p takes its position
 * from stream p % 3 (time, height, width).  Text has t = h = w = its cell
 * index, which is all the rope kernels knew; an image does not (mtmd: an
 * image starting at text position p puts its cell (x, y) at t = p,
 * h = p + y, w = p + x, and the text after it continues at
 * p + max(nx, ny)).  Callers keep passing cell indices and the kernels
 * look the rotary position up: g_mrope_tab [n][3] covers the prompt's
 * cells, a cell past it sits at cell - g_mrope_shift (decode, verify,
 * drafts).  Device symbols, not kernel arguments, so graphs captured
 * before an image request read the current table.  Null (the default and
 * every request without an image): the position is the cell, as before. */
__device__ const int32_t *g_mrope_tab;
__device__ int32_t g_mrope_n;
__device__ int32_t g_mrope_shift;
static int32_t *g_mrope_buf;
static size_t g_mrope_cap;

__device__ __forceinline__ static int32_t mrope_pos(int32_t cell, uint32_t pair) {
    const int32_t *t = g_mrope_tab;
    if (!t) return cell;
    if (cell >= 0 && cell < g_mrope_n) return t[(size_t)cell * 3 + pair % 3];
    return cell - g_mrope_shift;
}

/* tab NULL: back to the identity.  Waits for the device: no kernel may be
 * reading the table it replaces. */
/* Rotary pair layout.  qwen4exp is a NEOX-style rope (llama.cpp
 * LLAMA_ROPE_TYPE_IMROPE): rotary pair p is elements (p, p + n_rot/2).
 * q4 used to rotate (2p, 2p + 1) — the same frequencies on the wrong
 * element pairs, for the attention q/k, the indexer q and the pooled
 * indexer keys alike.  At short range the error hides (small angles); at
 * depth the attention and the indexer's top-k lose their sense of
 * distance (same text: nll 0.6-1.8 at the start of a document, 3-6 past
 * ~20K tokens).  Q4_ROPE_NEOX=0 restores the old pairing for A/Bs. */
__device__ __forceinline__ static void rope_pair(uint32_t p, uint32_t np,
                                                 uint32_t &i0, uint32_t &i1) {
    if (g_rope_neox) {
        i0 = p;
        i1 = p + np;
    } else {
        i0 = 2 * p;
        i1 = 2 * p + 1;
    }
}

extern "C" bool q4_hip_mrope_set(const int32_t *tab, uint32_t n, int32_t shift) {
    if (!g_ok || hipDeviceSynchronize() != hipSuccess) return false;
    const int32_t *dp = nullptr;
    if (tab && n) {
        const size_t need = (size_t)n * 3 * sizeof(int32_t);
        if (need > g_mrope_cap) {
            if (g_mrope_buf) (void)hipFree(g_mrope_buf);
            g_mrope_buf = nullptr;
            g_mrope_cap = 0;
            if (hipMalloc((void **)&g_mrope_buf, need) != hipSuccess) {
                g_mrope_buf = nullptr;
                return false;
            }
            g_mrope_cap = need;
        }
        if (hipMemcpy(g_mrope_buf, tab, need, hipMemcpyHostToDevice) != hipSuccess)
            return false;
        dp = g_mrope_buf;
    } else {
        n = 0;
        shift = 0;
    }
    const int32_t nn = (int32_t)n;
    return hipMemcpyToSymbol(HIP_SYMBOL(g_mrope_n), &nn, sizeof(nn)) == hipSuccess &&
           hipMemcpyToSymbol(HIP_SYMBOL(g_mrope_shift), &shift, sizeof(shift)) ==
               hipSuccess &&
           hipMemcpyToSymbol(HIP_SYMBOL(g_mrope_tab), &dp, sizeof(dp)) == hipSuccess &&
           hipDeviceSynchronize() == hipSuccess;
}

__global__ static void qsa_q_prep_k(const float *qg, const float *wn, float *q,
                                    float *g, uint32_t hd, uint32_t n_rot,
                                    int32_t pos0, float base, float eps,
                                    uint32_t n_head, const int32_t *pp) {
    uint32_t h = blockIdx.x;
    uint32_t t = blockIdx.y;
    int32_t pos = (pp ? *pp : pos0) + (int32_t)t;
    const float *src = qg + ((size_t)t * n_head + h) * hd * 2;
    float *qd = q + ((size_t)t * n_head + h) * hd;
    float *gd = g + ((size_t)t * n_head + h) * hd;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) gd[i] = src[hd + i];
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) ss += src[i] * src[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)hd + eps);
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x)
        qd[i] = src[i] * scale * (wn ? wn[i] : 1.f);
    __syncthreads();
    uint32_t np = n_rot / 2;
    for (uint32_t p = threadIdx.x; p < np; p += blockDim.x) {
        float freq = powf(base, -(float)p / (float)np);
        float ang = (float)mrope_pos(pos, p) * freq;
        float c = cosf(ang), si = sinf(ang);
        uint32_t i0, i1;
        rope_pair(p, np, i0, i1);
        float x0 = qd[i0], x1 = qd[i1];
        qd[i0] = x0 * c - x1 * si;
        qd[i1] = x0 * si + x1 * c;
    }
}

__global__ static void qsa_k_prep_k(float *k, const float *wn, uint32_t hd,
                                    uint32_t n_rot, int32_t pos0, float base,
                                    float eps, uint32_t n_kvh,
                                    const int32_t *pp) {
    uint32_t h = blockIdx.x;
    uint32_t t = blockIdx.y;
    int32_t pos = (pp ? *pp : pos0) + (int32_t)t;
    float *kh = k + ((size_t)t * n_kvh + h) * hd;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) ss += kh[i] * kh[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)hd + eps);
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x)
        kh[i] = kh[i] * scale * (wn ? wn[i] : 1.f);
    __syncthreads();
    uint32_t np = n_rot / 2;
    for (uint32_t p = threadIdx.x; p < np; p += blockDim.x) {
        float freq = powf(base, -(float)p / (float)np);
        float ang = (float)mrope_pos(pos, p) * freq;
        float c = cosf(ang), si = sinf(ang);
        uint32_t i0, i1;
        rope_pair(p, np, i0, i1);
        float x0 = kh[i0], x1 = kh[i1];
        kh[i0] = x0 * c - x1 * si;
        kh[i1] = x0 * si + x1 * c;
    }
}

extern "C" bool q4_hip_qsa_q_prep_n(const float *d_qg, const float *d_wn, float *d_q,
                                    float *d_g, uint32_t n_head, uint32_t hd,
                                    uint32_t n_rot, int32_t pos0, float base,
                                    float eps, uint32_t n_tok) {
    if (!g_ok || !d_qg || !d_q || !d_g || n_head == 0 || n_tok == 0) return false;
    qsa_q_prep_k<<<dim3(n_head, n_tok), 256, 0, g_str>>>(d_qg, d_wn, d_q, d_g, hd, n_rot, pos0,
                                               base, eps, n_head, nullptr);
    return hipGetLastError() == hipSuccess;
}
/* Device-position variant for graph capture: pos comes from g_pos_d. */
extern "C" bool q4_hip_qsa_q_prep_d(const float *d_qg, const float *d_wn,
                                    float *d_q, float *d_g, uint32_t n_head,
                                    uint32_t hd, uint32_t n_rot, float base,
                                    float eps) {
    if (!g_ok || !g_pos_d || !d_qg || !d_q || !d_g || n_head == 0) return false;
    qsa_q_prep_k<<<dim3(n_head, 1), 256, 0, g_str>>>(d_qg, d_wn, d_q, d_g, hd,
                                                   n_rot, 0, base, eps, n_head,
                                                   g_pos_d);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_qsa_q_prep(const float *d_qg, const float *d_wn, float *d_q,
                                  float *d_g, uint32_t n_head, uint32_t hd,
                                  uint32_t n_rot, int32_t pos, float base,
                                  float eps) {
    return q4_hip_qsa_q_prep_n(d_qg, d_wn, d_q, d_g, n_head, hd, n_rot, pos, base,
                               eps, 1);
}
extern "C" bool q4_hip_qsa_k_prep_n(float *d_k, const float *d_wn, uint32_t n_kvh,
                                    uint32_t hd, uint32_t n_rot, int32_t pos0,
                                    float base, float eps, uint32_t n_tok) {
    if (!g_ok || !d_k || n_kvh == 0 || n_tok == 0) return false;
    qsa_k_prep_k<<<dim3(n_kvh, n_tok), 256, 0, g_str>>>(d_k, d_wn, hd, n_rot, pos0, base, eps,
                                              n_kvh, nullptr);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_qsa_k_prep_d(float *d_k, const float *d_wn,
                                    uint32_t n_kvh, uint32_t hd, uint32_t n_rot,
                                    float base, float eps) {
    if (!g_ok || !g_pos_d || !d_k || n_kvh == 0) return false;
    qsa_k_prep_k<<<dim3(n_kvh, 1), 256, 0, g_str>>>(d_k, d_wn, hd, n_rot, 0,
                                                  base, eps, n_kvh, g_pos_d);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_qsa_k_prep(float *d_k, const float *d_wn, uint32_t n_kvh,
                                  uint32_t hd, uint32_t n_rot, int32_t pos,
                                  float base, float eps) {
    return q4_hip_qsa_k_prep_n(d_k, d_wn, n_kvh, hd, n_rot, pos, base, eps, 1);
}

/* One-head rms+rope shared by the standalone prep kernels and the fused
 * prep3 launch: dst[i] = src[i]*rms(src)*wn[i], then rope rotate pairs.
 * src==dst is allowed (in-place K path). */
__device__ static void rmsrope1(const float *src, const float *wn, float *dst,
                                uint32_t hd, uint32_t n_rot, int32_t pos,
                                float base, float eps, float *sh) {
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x)
        ss += src[i] * src[i];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(sh[0] / (float)hd + eps);
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x)
        dst[i] = src[i] * scale * (wn ? wn[i] : 1.f);
    __syncthreads();
    const uint32_t np = n_rot / 2;
    for (uint32_t p = threadIdx.x; p < np; p += blockDim.x) {
        const float freq = powf(base, -(float)p / (float)np);
        const float ang = (float)mrope_pos(pos, p) * freq;
        const float c = cosf(ang), si = sinf(ang);
        uint32_t i0, i1;
        rope_pair(p, np, i0, i1);
        const float x0 = dst[i0], x1 = dst[i1];
        dst[i0] = x0 * c - x1 * si;
        dst[i1] = x0 * si + x1 * c;
    }
}

/* Fused QSA prep: q norm+rope+gate-split (blocks [0,n_head)), k norm+rope
 * ([n_head, n_head+n_kvh)), optional indexer-q norm+rope on its own head
 * dim (remaining blocks).  Bit-identical to the three separate launches —
 * the heads were independent anyway. */
__global__ static void qsa_prep3_k(const float *qg, const float *qnw,
                                   float *q, float *g, uint32_t n_head,
                                   float *k, const float *knw, uint32_t n_kvh,
                                   float *ik, const float *inw, uint32_t n_ih,
                                   uint32_t hd, uint32_t ihd, uint32_t n_rot,
                                   float base, float eps, const int32_t *pp) {
    __shared__ float sh[256];
    const uint32_t h = blockIdx.x;
    const int32_t pos = *pp;
    if (h < n_head) {
        const float *src = qg + (size_t)h * hd * 2;
        float *gd = g + (size_t)h * hd;
        for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x)
            gd[i] = src[hd + i];
        rmsrope1(src, qnw, q + (size_t)h * hd, hd, n_rot, pos, base, eps, sh);
    } else if (h < n_head + n_kvh) {
        float *kh = k + (size_t)(h - n_head) * hd;
        rmsrope1(kh, knw, kh, hd, n_rot, pos, base, eps, sh);
    } else {
        float *kh = ik + (size_t)(h - n_head - n_kvh) * ihd;
        rmsrope1(kh, inw, kh, ihd, n_rot, pos, base, eps, sh);
    }
}
extern "C" bool q4_hip_qsa_prep3_d(const float *d_qg, const float *d_qnw,
                                   float *d_q, float *d_g, uint32_t n_head,
                                   float *d_k, const float *d_knw,
                                   uint32_t n_kvh, float *d_ik,
                                   const float *d_inw, uint32_t n_ih,
                                   uint32_t hd, uint32_t ihd, uint32_t n_rot,
                                   float base, float eps) {
    if (!g_ok || !g_pos_d || !d_qg || !d_q || !d_g || !d_k ||
        !n_head || !n_kvh || (n_ih && !d_ik))
        return false;
    qsa_prep3_k<<<n_head + n_kvh + n_ih, 256, 0, g_str>>>(
        d_qg, d_qnw, d_q, d_g, n_head, d_k, d_knw, n_kvh,
        d_ik, d_inw, n_ih, hd, ihd, n_rot, base, eps, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_rms_gate_heads(float *h, const float *w, const float *z,
                                      uint32_t n_h, uint32_t d, float eps) {
    if (!g_ok || !h || !z || n_h == 0) return false;
    rms_gate_heads_k<<<n_h, 128, 0, g_str>>>(h, w, z, d, eps);
    return hipGetLastError() == hipSuccess;
}

__global__ static void softmax_topk_k(const float *x, int32_t *ids, float *wts,
                                      int n, int k) {
    __shared__ float buf[512];
    int t = (int)threadIdx.x;
    buf[t] = (t < n) ? x[t] : -INFINITY;
    __syncthreads();
    float m = -INFINITY;
    for (int i = t; i < n; i += (int)blockDim.x)
        if (buf[i] > m) m = buf[i];
    __shared__ float sh[512];
    sh[t] = m;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s && sh[t + s] > sh[t]) sh[t] = sh[t + s];
        __syncthreads();
    }
    m = sh[0];
    float e = (t < n) ? expf(buf[t] - m) : 0.f;
    buf[t] = e;
    __syncthreads();
    float sum = 0.f;
    for (int i = t; i < n; i += (int)blockDim.x) sum += buf[i];
    sh[t] = sum;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    float inv = sh[0] > 0.f ? 1.f / sh[0] : 0.f;
    if (t < n) buf[t] *= inv;
    __syncthreads();
    /* Top-k by warp 0: per pick a lane-strided max of (value, ~index) keys
     * — the serial thread-0 scan's order (largest value, lowest index on a
     * tie; probabilities are >= 0 so their bits order as unsigned), ~70 us
     * of each draft step before (512 x k shared reads on one thread). */
    if (t < 32 && k <= 32) {
        int myid = 0;
        float myw = 0.f;
        for (int j = 0; j < k; j++) {
            unsigned long long best = 0;
            for (int i = t; i < n; i += 32) {
                const float v = buf[i];
                if (v >= 0.f) {
                    const unsigned long long key =
                        ((unsigned long long)__float_as_uint(v) << 32) |
                        (unsigned long long)(~(uint32_t)i);
                    if (key > best) best = key;
                }
            }
            for (int o = 16; o; o >>= 1) {
                const unsigned long long b2 = __shfl_xor(best, o);
                if (b2 > best) best = b2;
            }
            const int bi = best ? (int)(~(uint32_t)(best & 0xffffffffull)) : 0;
            const float bv = best ? __uint_as_float((uint32_t)(best >> 32)) : -1.f;
            if (t == j) { myid = bi; myw = bv; }
            if (t == 0) buf[bi] = -1.f;
            __syncwarp();
        }
        float sm = 0.f;
        for (int j = 0; j < k; j++) sm += __shfl(myw, j);
        if (t < k) {
            ids[t] = myid;
            wts[t] = sm > 0.f ? myw * (1.f / sm) : myw;
        }
    } else if (t == 0 && k > 32) {
        for (int j = 0; j < k; j++) {
            int bi = 0;
            float bv = -1.f;
            for (int i = 0; i < n; i++) {
                if (buf[i] > bv) {
                    bv = buf[i];
                    bi = i;
                }
            }
            ids[j] = bi;
            wts[j] = bv;
            buf[bi] = -1.f;
        }
        float s = 0.f;
        for (int j = 0; j < k; j++) s += wts[j];
        if (s > 0.f) {
            float r = 1.f / s;
            for (int j = 0; j < k; j++) wts[j] *= r;
        }
    }
}

extern "C" bool q4_hip_softmax_topk(const float *d_logits, uint32_t n, uint32_t k,
                                    int32_t *h_ids, float *h_wts) {
    if (!g_ok || !d_logits || !h_ids || !h_wts || n == 0 || k == 0 || n > 512)
        return false;
    static int32_t *d_ids = nullptr;
    static float *d_wts = nullptr;
    if (!d_ids) {
        if (hipMalloc((void **)&d_ids, 64 * sizeof(int32_t)) != hipSuccess)
            return false;
        if (hipMalloc((void **)&d_wts, 64 * sizeof(float)) != hipSuccess)
            return false;
    }
    softmax_topk_k<<<1, 512, 0, g_str>>>(d_logits, d_ids, d_wts, (int)n, (int)k);
    if (hipGetLastError() != hipSuccess) return false;
    if (!sync_copy(h_ids, d_ids, k * sizeof(int32_t), hipMemcpyDeviceToHost))
        return false;
    return sync_copy(h_wts, d_wts, k * sizeof(float), hipMemcpyDeviceToHost);
}

extern "C" bool q4_hip_softmax_topk_launch(const float *d_logits, uint32_t n,
                                           uint32_t k, int32_t *d_ids,
                                           float *d_wts) {
    if (!g_ok || !d_logits || !d_ids || !d_wts || n == 0 || k == 0 || n > 512 ||
        k > 64)
        return false;
    softmax_topk_k<<<1, 512, 0, g_str>>>(d_logits, d_ids, d_wts, (int)n, (int)k);
    return hipGetLastError() == hipSuccess;
}

__device__ static float q4k_block_dot(const uint8_t *blk, const float *xr) {
    float d = f16d(blk);
    float minv = f16d(blk + 2);
    const uint8_t *sc = blk + 4;
    const uint8_t *q = blk + 16;
    float acc = 0.f;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s1, m1, s2, m2;
        scale_min_k4(is + 0, sc, &s1, &m1);
        scale_min_k4(is + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = minv * m1;
        float d2 = d * s2, mm2 = minv * m2;
        for (int l = 0; l < 32; l++)
            acc += (d1 * (q[l] & 0xF) - mm1) * xr[j + l];
        for (int l = 0; l < 32; l++)
            acc += (d2 * (q[l] >> 4) - mm2) * xr[j + 32 + l];
        q += 32;
        is += 2;
    }
    return acc;
}

__global__ static void gateup_q4k_k(const uint8_t *gate, const uint8_t *up,
                                    const float *x, float *h, uint32_t n_ff,
                                    uint32_t n_embd) {
    uint32_t row = blockIdx.x;
    uint32_t tok = blockIdx.y;
    if (row >= n_ff) return;
    x += (size_t)tok * n_embd;
    h += (size_t)tok * n_ff;
    uint32_t nb = n_embd / 256u;
    uint64_t rb = (uint64_t)nb * 144ull;
    const uint8_t *gr = gate + row * rb;
    const uint8_t *ur = up + row * rb;
    float acc_g = 0.f, acc_u = 0.f;
    for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
        acc_g += q4k_block_dot(gr + i * 144u, x + i * 256u);
        acc_u += q4k_block_dot(ur + i * 144u, x + i * 256u);
    }
    __shared__ float sg[64], su[64];
    sg[threadIdx.x] = acc_g;
    su[threadIdx.x] = acc_u;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sg[threadIdx.x] += sg[threadIdx.x + s];
            su[threadIdx.x] += su[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float g = sg[0];
        h[row] = (g / (1.f + expf(-g))) * su[0];
    }
}

__global__ static void gateup_q4k_n_k(const uint8_t *gate, const uint8_t *up,
                                      const float *x, float *h, uint32_t n_ff,
                                      uint32_t n_embd, uint32_t n_tok,
                                      const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= n_ff) return;
    uint32_t nb = n_embd / 256u;
    uint64_t rb = (uint64_t)nb * 144ull;
    const uint8_t *gr = gate + row * rb;
    const uint8_t *ur = up + row * rb;
    __shared__ float sg[64], su[64];
    for (uint32_t tok = blockIdx.y; tok < n_tok; tok += gridDim.y ? gridDim.y : 1) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * n_embd;
        float acc_g = 0.f, acc_u = 0.f;
        for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
            acc_g += q4k_block_dot(gr + i * 144u, xt + i * 256u);
            acc_u += q4k_block_dot(ur + i * 144u, xt + i * 256u);
        }
        sg[threadIdx.x] = acc_g;
        su[threadIdx.x] = acc_u;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sg[threadIdx.x] += sg[threadIdx.x + s];
                su[threadIdx.x] += su[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float g = sg[0];
            h[(size_t)tok * n_ff + row] = (g / (1.f + expf(-g))) * su[0];
        }
        __syncthreads();
    }
}

extern "C" bool q4_hip_gateup_q4k_n(const uint8_t *d_gate, const uint8_t *d_up,
                                    const float *d_x, float *d_h, uint32_t n_ff,
                                    uint32_t n_embd, uint32_t n_tok) {
    return q4_hip_gateup_q4k_n_which(d_gate, d_up, d_x, d_h, n_ff, n_embd, n_tok,
                                     NULL);
}

extern "C" bool q4_hip_gateup_q4k_n_which(const uint8_t *d_gate,
                                          const uint8_t *d_up, const float *d_x,
                                          float *d_h, uint32_t n_ff,
                                          uint32_t n_embd, uint32_t n_tok,
                                          const int32_t *which) {
    if (!g_ok || !d_gate || !d_up || !d_x || !d_h) return false;
    if (!n_tok) n_tok = 1;
    uint32_t gy = n_tok < 64u ? n_tok : 64u;
    gateup_q4k_n_k<<<dim3(n_ff, gy), 64, 0, g_str>>>(d_gate, d_up, d_x, d_h, n_ff, n_embd,
                                           n_tok, which);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_gateup_q4k(const uint8_t *d_gate, const uint8_t *d_up,
                                  const float *d_x, float *d_h, uint32_t n_ff,
                                  uint32_t n_embd) {
    return q4_hip_gateup_q4k_n(d_gate, d_up, d_x, d_h, n_ff, n_embd, 1);
}

__device__ static float kv_ld(const float *p) { return *p; }
__device__ static float kv_ld(const __half *p) { return __half2float(*p); }

/* One query (head, token). KV tile is staged in shared memory so the K
 * load is coalesced, and softmax max/sum run once per tile instead of
 * once per thread. kv_f16 stores the cache as fp16. */
template <int KIND, int INDEXED>
__global__ static void gqa_decode_fast(const float *q, const uint8_t *kcache,
                                       const uint8_t *vcache, const float *gate,
                                       float *out, uint32_t n_head, uint32_t n_kvh,
                                       uint32_t hd, uint32_t kv_base, uint32_t max_kv,
                                       float scale, const int32_t *idx,
                                       uint32_t width) {
    constexpr uint32_t TILE = 32;
    uint32_t h = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (h >= n_head || n_kvh == 0 || hd == 0 || hd > 256) return;
    uint32_t gqa = n_head / n_kvh;
    if (gqa == 0) return;
    uint32_t kh = h / gqa;
    uint32_t n_kv = kv_base + t + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    uint32_t n_pos = INDEXED ? width : n_kv;
    const float *qh = q + ((size_t)t * n_head + h) * hd;
    uint32_t i = threadIdx.x;

    __shared__ float qs[256];
    __shared__ float acc[256];
    __shared__ float tile[TILE * 256];
    __shared__ float wt[TILE];
    __shared__ float red[256];
    if (i < hd) {
        qs[i] = qh[i];
        acc[i] = 0.f;
    }
    __syncthreads();

    int have_m = 0;
    float m = 0.f, sum = 0.f;
    for (uint32_t base = 0; base < n_pos; base += TILE) {
        uint32_t lim = n_pos - base;
        if (lim > TILE) lim = TILE;
        uint32_t tile_n = TILE * hd;
        uint32_t q8b = (hd / 32u) * 34u;
        if constexpr (KIND == 2) {
            if (i < lim) {
                int p = INDEXED ? idx[(size_t)t * width + base + i]
                                : (int)(base + i);
                const uint8_t *src =
                    p >= 0 ? kcache + ((size_t)p * n_kvh + kh) * q8b : nullptr;
                float *dst = tile + (size_t)i * hd;
                if (!src) {
                    for (uint32_t d = 0; d < hd; d++) dst[d] = 0.f;
                } else {
                for (uint32_t b = 0; b < hd / 32u; b++) {
                    float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (int j = 0; j < 32; j++)
                        dst[b * 32u + j] = sc * (float)qs8[j];
                    src += 34;
                }
                }
            }
        } else {
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        kcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* 8 threads share one KV position so all 256 lanes do the dot. */
        uint32_t pos = i % TILE;
        uint32_t sub = i / TILE;
        uint32_t d0 = sub * (hd / 8u);
        uint32_t d1 = (sub == 7u) ? hd : d0 + (hd / 8u);
        float partial = 0.f;
        if (pos < lim) {
            const float *kk = tile + (size_t)pos * hd;
            for (uint32_t d = d0; d < d1; d++) partial += qs[d] * kk[d];
        }
        red[i] = partial;
        __syncthreads();
        float dot = -1.0e30f;
        if (i < TILE && i < lim) {
            int p = INDEXED ? idx[(size_t)t * width + base + i] : (int)(base + i);
            if (p >= 0) {
                dot = 0.f;
                for (uint32_t k = 0; k < 8; k++) dot += red[k * TILE + i];
                dot *= scale;
            }
        }
        __syncthreads();
        red[i] = (i < TILE) ? dot : -1.0e30f;
        __syncthreads();
        for (uint32_t s = 128; s > 0; s >>= 1) {
            if (i < s) red[i] = fmaxf(red[i], red[i + s]);
            __syncthreads();
        }
        float tm = red[0];
        int live = tm > -1.0e20f;
        float nm = m, alpha = 1.f;
        if (live) {
            if (!have_m) {
                nm = tm;
                alpha = 0.f;
                have_m = 1;
            } else if (tm > m) {
                nm = tm;
                alpha = expf(m - nm);
            } else {
                nm = m;
                alpha = 1.f;
            }
        }
        float w = 0.f;
        if (live && i < lim && dot > -1.0e20f) w = expf(dot - nm);
        if (i < TILE) wt[i] = w;
        __syncthreads();
        red[i] = (i < TILE) ? wt[i] : 0.f;
        __syncthreads();
        for (uint32_t s = 16; s > 0; s >>= 1) {
            if (i < s) red[i] += red[i + s];
            __syncthreads();
        }
        float tsum = red[0];
        if (i < hd) acc[i] *= alpha;
        sum *= alpha;
        __syncthreads();
        if constexpr (KIND == 2) {
            if (i < lim) {
                int p = INDEXED ? idx[(size_t)t * width + base + i]
                                : (int)(base + i);
                const uint8_t *src =
                    p >= 0 ? vcache + ((size_t)p * n_kvh + kh) * q8b : nullptr;
                float *dst = tile + (size_t)i * hd;
                if (!src) {
                    for (uint32_t d = 0; d < hd; d++) dst[d] = 0.f;
                } else {
                for (uint32_t b = 0; b < hd / 32u; b++) {
                    float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (int j = 0; j < 32; j++)
                        dst[b * 32u + j] = sc * (float)qs8[j];
                    src += 34;
                }
                }
            }
        } else {
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        vcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        if (i < hd) {
            float part = 0.f;
            for (uint32_t j = 0; j < lim; j++)
                part += wt[j] * tile[(size_t)j * hd + i];
            acc[i] += part;
        }
        sum += tsum;
        m = nm;
        __syncthreads();
    }
    float inv = sum > 0.f ? 1.f / sum : 0.f;
    if (i < hd) {
        float gv = 1.f / (1.f + expf(-gate[((size_t)t * n_head + h) * hd + i]));
        out[((size_t)t * n_head + h) * hd + i] = acc[i] * inv * gv;
    }
}

/* Decode attention, split-KV ("flash decoding") + GQA sharing: grid
 * (n_kvh, S). One workgroup runs ALL gqa query heads of its KV head over a
 * contiguous slice of the position list (indexed: slice of the sel list;
 * dense: slice of [0, n_kv)). The K/V tile is dequantized once per slice and
 * shared across heads. Per-head online-softmax partials (m, l, acc[hd]) go
 * to `part` for gqa_dec_comb_k to merge. n_tok == 1 only. */
#define GQA_DEC_SPLIT_S 128u
#define GQA_DEC_MAX_GQA 12u
/* Runtime split count for the decode path (env Q4_DEC_SPLIT_S, capped at
 * the compile-time partial buffers / combine shmem bound). */
static uint32_t g_dec_split_s(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("Q4_DEC_SPLIT_S");
        long v = e ? strtol(e, nullptr, 10) : (long)GQA_DEC_SPLIT_S;
        if (v < 1) v = 1;
        if (v > (long)GQA_DEC_SPLIT_S) v = GQA_DEC_SPLIT_S;
        s = (int)v;
    }
    return (uint32_t)s;
}
template <int KIND, bool INDEXED>
__global__ static void gqa_dec_split_k(const float *q, const uint8_t *kcache,
                                       const uint8_t *vcache, float *part,
                                       uint32_t n_head, uint32_t n_kvh,
                                       uint32_t hd, uint32_t kv_base,
                                       uint32_t max_kv, float scale,
                                       const int32_t *idx, uint32_t width,
                                       uint32_t S, const int32_t *pp) {
    constexpr uint32_t TILE = 32;
    uint32_t kh = blockIdx.x, s = blockIdx.y;
    uint32_t gqa = n_head / n_kvh;
    if (!gqa || !hd || hd > 256 || gqa > GQA_DEC_MAX_GQA) return;
    /* blockIdx.z: token row of a small batch (0 for decode) — its q row,
     * its sel row, one more causal position, its own partials block. */
    const uint32_t tz = blockIdx.z;
    q += (size_t)tz * n_head * hd;
    if (INDEXED) idx += (size_t)tz * width;
    part += (size_t)tz * n_kvh * S * gqa * (hd + 2u);
    uint32_t n_kv = (pp ? (uint32_t)*pp : kv_base) + tz + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    uint32_t n_pos = INDEXED ? width : n_kv;
    uint32_t per = (n_pos + S - 1) / S;
    uint32_t p0 = s * per;
    uint32_t p1 = p0 + per < n_pos ? p0 + per : n_pos;
    uint32_t i = threadIdx.x;
    float *pv = part + (((size_t)kh * S + s) * gqa) * (hd + 2u);

    __shared__ float qs[GQA_DEC_MAX_GQA * 256];
    __shared__ float acc[GQA_DEC_MAX_GQA * 256];
    __shared__ float tile[TILE * 256];
    __shared__ float wt[GQA_DEC_MAX_GQA * TILE];
    __shared__ float s_m[GQA_DEC_MAX_GQA];
    __shared__ float s_su[GQA_DEC_MAX_GQA];
    __shared__ float s_al[GQA_DEC_MAX_GQA];
    __shared__ uint32_t s_have[GQA_DEC_MAX_GQA];
    if (i < hd)
        for (uint32_t g = 0; g < gqa; g++) {
            qs[g * 256 + i] = q[((size_t)(kh * gqa + g)) * hd + i];
            acc[g * 256 + i] = 0.f;
        }
    if (i < GQA_DEC_MAX_GQA) {
        s_m[i] = 0.f;
        s_su[i] = 0.f;
        s_al[i] = 1.f;
        s_have[i] = 0;
    }
    __syncthreads();

    for (uint32_t base = p0; base < p1; base += TILE) {
        uint32_t lim = p1 - base;
        if (lim > TILE) lim = TILE;
        uint32_t q8b = (hd / 32u) * 34u;
        if constexpr (KIND == 2) {
            /* One 34-byte quant block per thread: TILE*(hd/32) = 256 work
             * items, so with TILE=32 and 256 threads each thread dequants a
             * single 32-value block instead of a whole row serially. */
            const uint32_t nb = hd / 32u;
            const uint32_t pp = i / nb, bb = i % nb;
            if (pp < lim) {
                int p = INDEXED ? idx[base + pp] : (int)(base + pp);
                float *dst = tile + (size_t)pp * hd + bb * 32u;
                if (p < 0) {
                    for (uint32_t j = 0; j < 32; j++) dst[j] = 0.f;
                } else {
                    const uint8_t *src =
                        kcache + ((size_t)p * n_kvh + kh) * q8b + bb * 34u;
                    const float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (uint32_t j = 0; j < 32; j++)
                        dst[j] = sc * (float)qs8[j];
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[base + pp] : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        kcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* Warp-per-position scores for all heads: 8 warps cover TILE=32
         * positions (4 each); every lane reduces its head dot by shuffle —
         * no block barriers inside the head loop. */
        uint32_t wid = i >> 5, lane = i & 31;
        for (uint32_t pos = wid; pos < TILE; pos += 8) {
            int p = -1;
            if (pos < lim)
                p = INDEXED ? idx[base + pos] : (int)(base + pos);
            const float *kk = tile + (size_t)pos * hd;
            for (uint32_t g = 0; g < gqa; g++) {
                float a = 0.f;
                if (p >= 0) {
                    const float *qh = qs + g * 256;
                    for (uint32_t d = lane; d < hd; d += 32)
                        a += qh[d] * kk[d];
                }
                for (uint32_t o = 16; o; o >>= 1)
                    a += __shfl_down_sync(~0ull, a, o);
                if (!lane) wt[g * TILE + pos] = p >= 0 ? a * scale : -1.0e30f;
            }
        }
        __syncthreads();
        /* Warp-per-head online softmax update over the tile scores. */
        for (uint32_t g = wid; g < gqa; g += 8) {
            float mx = -1.0e30f;
            for (uint32_t p = lane; p < lim; p += 32)
                mx = fmaxf(mx, wt[g * TILE + p]);
            for (uint32_t o = 16; o; o >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(~0ull, mx, o));
            mx = __shfl_sync(~0ull, mx, 0);
            int live = mx > -1.0e20f;
            float mo = s_m[g], nm = mo, alpha = 1.f;
            if (live) {
                if (!s_have[g]) {
                    nm = mx;
                    alpha = 0.f;
                } else if (mx > mo) {
                    nm = mx;
                    alpha = expf(mo - nm);
                }
            }
            float su = 0.f;
            for (uint32_t p = lane; p < lim; p += 32) {
                float w = (live && wt[g * TILE + p] > -1.0e20f)
                              ? expf(wt[g * TILE + p] - nm)
                              : 0.f;
                wt[g * TILE + p] = w;
                su += w;
            }
            for (uint32_t o = 16; o; o >>= 1)
                su += __shfl_down_sync(~0ull, su, o);
            if (!lane) {
                s_m[g] = nm;
                s_su[g] = s_su[g] * alpha + su;
                s_al[g] = alpha;
                if (live) s_have[g] = 1;
            }
        }
        __syncthreads();
        if (i < hd)
            for (uint32_t g = 0; g < gqa; g++)
                acc[g * 256 + i] *= s_al[g];
        __syncthreads();
        /* V tile + per-head accumulate (same one-block-per-thread split) */
        if constexpr (KIND == 2) {
            const uint32_t nb = hd / 32u;
            const uint32_t pp = i / nb, bb = i % nb;
            if (pp < lim) {
                int p = INDEXED ? idx[base + pp] : (int)(base + pp);
                float *dst = tile + (size_t)pp * hd + bb * 32u;
                if (p < 0) {
                    for (uint32_t j = 0; j < 32; j++) dst[j] = 0.f;
                } else {
                    const uint8_t *src =
                        vcache + ((size_t)p * n_kvh + kh) * q8b + bb * 34u;
                    const float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (uint32_t j = 0; j < 32; j++)
                        dst[j] = sc * (float)qs8[j];
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[base + pp] : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        vcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        if (i < hd)
            for (uint32_t g = 0; g < gqa; g++) {
                float a = 0.f;
                for (uint32_t j = 0; j < lim; j++)
                    a += wt[g * TILE + j] * tile[(size_t)j * hd + i];
                acc[g * 256 + i] += a;
            }
        __syncthreads();
    }
    for (uint32_t g = 0; g < gqa; g++) {
        if (i < hd) pv[g * (hd + 2u) + 2u + i] = acc[g * 256 + i];
        if (i == 0) {
            pv[g * (hd + 2u)] = s_m[g];
            pv[g * (hd + 2u) + 1u] = s_su[g];
        }
    }
}

/* Merge S partials per head; applies the sigmoid gate exactly like
 * gqa_decode_fast's epilogue. */
__global__ static void gqa_dec_comb_k(const float *part, const float *gate,
                                      float *out, uint32_t n_head,
                                      uint32_t n_kvh, uint32_t hd,
                                      uint32_t S) {
    uint32_t h = blockIdx.x;
    uint32_t gqa = n_head / n_kvh;
    if (!gqa || h >= n_head || hd > 256) return;
    /* blockIdx.y: token row of a small batch (0 for decode). */
    part += (size_t)blockIdx.y * n_kvh * S * gqa * (hd + 2u);
    gate += (size_t)blockIdx.y * n_head * hd;
    out += (size_t)blockIdx.y * n_head * hd;
    uint32_t kh = h / gqa, g = h % gqa;
    uint32_t i = threadIdx.x;
    __shared__ float ms[GQA_DEC_SPLIT_S], ls[GQA_DEC_SPLIT_S], sh[256];
    if (i < S) {
        const float *pv = part + (((size_t)kh * S + i) * gqa + g) * (hd + 2u);
        ms[i] = pv[0];
        ls[i] = pv[1];
    }
    __syncthreads();
    sh[i] = (i < S) ? ms[i] : -1.0e30f;
    __syncthreads();
    for (uint32_t sp = 128; sp > 0; sp >>= 1) {
        if (i < sp) sh[i] = fmaxf(sh[i], sh[i + sp]);
        __syncthreads();
    }
    float M = sh[0];
    float ws = (i < S && ls[i] > 0.f) ? expf(ms[i] - M) : 0.f;
    sh[i] = (i < S) ? ls[i] * ws : 0.f;
    __syncthreads();
    for (uint32_t sp = 128; sp > 0; sp >>= 1) {
        if (i < sp) sh[i] += sh[i + sp];
        __syncthreads();
    }
    float wsum = sh[0];
    if (i < S) ms[i] = ws;
    __syncthreads();
    float inv = wsum > 0.f ? 1.f / wsum : 0.f;
    if (i < hd) {
        float a = 0.f;
        for (uint32_t sp = 0; sp < S; sp++) {
            const float *pv =
                part + (((size_t)kh * S + sp) * gqa + g) * (hd + 2u);
            a += ms[sp] * pv[2u + i];
        }
        float gv = 1.f / (1.f + expf(-gate[(size_t)h * hd + i]));
        out[(size_t)h * hd + i] = a * inv * gv;
    }
}

/* Prefill attention: grid (n_kvh, n_tok), one block runs all gqa query heads
 * of one token against one KV head. The 32-position K/V tile is dequantized
 * once into shared and reused by every head (gqa_decode_fast re-reads it per
 * head). Per-head accumulators live in registers: thread i owns dim i of all
 * heads. Same online-softmax semantics as gqa_decode_fast. */
template <int KIND, bool INDEXED>
__global__ static void gqa_prefill_k(const float *q, const uint8_t *kcache,
                                     const uint8_t *vcache, const float *gate,
                                     float *out, uint32_t n_head,
                                     uint32_t n_kvh, uint32_t hd,
                                     uint32_t kv_base, uint32_t max_kv,
                                     float scale, const int32_t *idx,
                                     uint32_t width) {
    constexpr uint32_t TILE = 32, GQA = 12;
    uint32_t kh = blockIdx.x;
    uint32_t t = blockIdx.y;
    uint32_t gqa = n_head / n_kvh;
    if (kh >= n_kvh || gqa == 0 || gqa > GQA || hd == 0 || hd > 256) return;
    uint32_t n_kv = kv_base + t + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    uint32_t n_pos = INDEXED ? width : n_kv;
    uint32_t i = threadIdx.x;
    uint32_t w = i >> 5, lane = i & 31;

    __shared__ float qs[GQA * 256];
    __shared__ __half tile[TILE * 256];
    __shared__ float wt[GQA * TILE];
    __shared__ float dots[GQA * TILE];
    __shared__ float mm[GQA], ssum[GQA], al[GQA];

    for (uint32_t e = i; e < gqa * hd; e += blockDim.x) {
        uint32_t g = e / hd, d = e - g * hd;
        qs[e] = q[((size_t)t * n_head + kh * gqa + g) * hd + d];
    }
    if (i < GQA) { mm[i] = -1.0e30f; ssum[i] = 0.f; }
    float acc[GQA];
    for (uint32_t g = 0; g < GQA; g++) acc[g] = 0.f;
    __syncthreads();

    uint32_t q8b = (hd / 32u) * 34u;
    for (uint32_t base = 0; base < n_pos; base += TILE) {
        uint32_t lim = n_pos - base;
        if (lim > TILE) lim = TILE;
        if constexpr (KIND == 2) {
            /* warp per position: lanes split hd (one element per 34B
             * q8_0 block lane-step) — the old i<lim path left 224 threads
             * idle while each of 32 threads dequantized a whole row. */
            for (uint32_t pos = w; pos < lim; pos += blockDim.x >> 5) {
                int p = INDEXED ? idx[(size_t)t * width + base + pos]
                                : (int)(base + pos);
                const uint8_t *src =
                    p >= 0 ? kcache + ((size_t)p * n_kvh + kh) * q8b
                           : nullptr;
                __half *dst = tile + (size_t)pos * hd;
                for (uint32_t d = lane; d < hd; d += 32) {
                    float v = 0.f;
                    if (src) {
                        const uint8_t *b = src + (d >> 5) * 34u;
                        v = f16d(b) * (float)(const int8_t)b[2 + (d & 31u)];
                    }
                    dst[d] = __float2half(v);
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        kcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* 8 warps cover the gqa*32 (g,pos) dots; lanes split hd, shfl-reduce. */
        for (uint32_t pr = w; pr < gqa * TILE; pr += 8) {
            uint32_t g = pr / TILE, pos = pr - g * TILE;
            float d = -1.0e30f;
            if (pos < lim) {
                int p = INDEXED ? idx[(size_t)t * width + base + pos]
                                : (int)(base + pos);
                if (p >= 0) {
                    d = 0.f;
                    const __half *kk = tile + (size_t)pos * hd;
                    const float *qq = qs + (size_t)g * hd;
                    for (uint32_t dd = lane; dd < hd; dd += 32)
                        d += qq[dd] * __half2float(kk[dd]);
                    for (uint32_t o = 16; o; o >>= 1)
                        d += __shfl_down(d, o);
                    if (lane == 0) d *= scale;
                }
            }
            if (lane == 0) dots[pr] = d;
        }
        __syncthreads();
        /* One warp per head (12 heads over 8 warps): tile max, online update,
         * weights, weight sum. Scalars shared via mm/ssum/al. */
        for (uint32_t g = w; g < gqa; g += 8) {
            float dv = (lane < TILE && lane < lim) ? dots[g * TILE + lane]
                                                   : -1.0e30f;
            float tm = dv;
            for (uint32_t o = 16; o; o >>= 1)
                tm = fmaxf(tm, __shfl_down(tm, o));
            tm = __shfl(tm, 0);
            int live = tm > -1.0e20f;
            float m = mm[g], nm = m, alpha = 1.f;
            if (live) {
                if (tm > m) {
                    nm = tm;
                    alpha = expf(m - nm);
                }
            }
            float wv = (live && dv > -1.0e20f) ? expf(dv - nm) : 0.f;
            if (lane < TILE) wt[g * TILE + lane] = wv;
            float ts = wv;
            for (uint32_t o = 16; o; o >>= 1)
                ts += __shfl_down(ts, o);
            if (lane == 0) {
                mm[g] = nm;
                ssum[g] = ssum[g] * alpha + ts;
                al[g] = alpha;
            }
        }
        __syncthreads();
        if constexpr (KIND == 2) {
            for (uint32_t pos = w; pos < lim; pos += blockDim.x >> 5) {
                int p = INDEXED ? idx[(size_t)t * width + base + pos]
                                : (int)(base + pos);
                const uint8_t *src =
                    p >= 0 ? vcache + ((size_t)p * n_kvh + kh) * q8b
                           : nullptr;
                __half *dst = tile + (size_t)pos * hd;
                for (uint32_t d = lane; d < hd; d += 32) {
                    float v = 0.f;
                    if (src) {
                        const uint8_t *b = src + (d >> 5) * 34u;
                        v = f16d(b) * (float)(const int8_t)b[2 + (d & 31u)];
                    }
                    dst[d] = __float2half(v);
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        vcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* acc[g] = acc[g]*alpha[g] + sum_j wt[g][j]*tile[j][i] for dim i. */
        if (i < hd) {
            for (uint32_t g = 0; g < gqa; g++) {
                float a = acc[g] * al[g];
                const float *wrow = wt + g * TILE;
                for (uint32_t j = 0; j < lim; j++)
                    a += wrow[j] * __half2float(tile[(size_t)j * hd + i]);
                acc[g] = a;
            }
        }
        __syncthreads();
    }
    if (i < hd) {
        for (uint32_t g = 0; g < gqa; g++) {
            uint32_t h = kh * gqa + g;
            float inv = ssum[g] > 0.f ? 1.f / ssum[g] : 0.f;
            float gv = 1.f / (1.f + expf(-gate[((size_t)t * n_head + h) * hd + i]));
            out[((size_t)t * n_head + h) * hd + i] = acc[g] * inv * gv;
        }
    }
}

/* WMMA flash-attention prefill for QSA (Q4_QSA_WMMA, default on, KIND 2
 * only): same grid (kv head, token) and online-softmax semantics as
 * gqa_prefill_k — out = acc/ssum * sigmoid(gate).  One token's ≤12 query
 * heads are the WMMA rows (16-row pad); each 64-position tile: cooperative
 * q8_0 dequant to fp16 LDS, Q·Kᵀ split across the 8 waves by (j-tile ×
 * k-half) with a shared partial-sum combine, warp-per-head softmax into an
 * fp16 P tile, then P·V with the O accumulators in registers (8 waves × 2
 * col-tiles of 16), rescaled by alpha each tile.  ~54 KiB LDS. */
/* One q8_0 KV row (hd 256 = 8 blocks of 34 B) -> 256 fp16, lane l
 * covering values 8l..8l+7: four u16 loads into one 16-byte register
 * (same values as the per-element f16d(b) * q path); the caller stores
 * it to LDS — split so V's global loads can be issued early. */
__device__ static inline uint4 q8row_ld(const uint8_t *src, uint32_t lane) {
    union { __half h[8]; uint4 u; } o;
    if (src) {
        const uint8_t *b = src + (lane >> 2) * 34u;
        const float sc = f16d(b);
        const uint16_t *q16 = (const uint16_t *)(b + 2 + (lane & 3u) * 8u);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint16_t v = q16[k];
            o.h[2 * k] = __float2half(sc * (float)(int8_t)(v & 0xffu));
            o.h[2 * k + 1] = __float2half(sc * (float)(int8_t)(v >> 8));
        }
    } else {
        o.u = make_uint4(0u, 0u, 0u, 0u);
    }
    return o.u;
}

/* f32 KV row (hd 256) -> lane l's 8 values as fp16, like q8row_ld. */
__device__ static inline uint4 f32row_ld(const float *src, uint32_t lane) {
    union { __half h[8]; uint4 u; } o;
    if (src) {
        const float4 a = *(const float4 *)(src + 8u * lane);
        const float4 c = *(const float4 *)(src + 8u * lane + 4u);
        o.h[0] = __float2half(a.x); o.h[1] = __float2half(a.y);
        o.h[2] = __float2half(a.z); o.h[3] = __float2half(a.w);
        o.h[4] = __float2half(c.x); o.h[5] = __float2half(c.y);
        o.h[6] = __float2half(c.z); o.h[7] = __float2half(c.w);
    } else {
        o.u = make_uint4(0u, 0u, 0u, 0u);
    }
    return o.u;
}

template <bool INDEXED>
__global__ static void gqa_prefill_wmma_k(const float *q, const uint8_t *kcache,
                                        const uint8_t *vcache,
                                        const float *gate, float *out,
                                        uint32_t n_head, uint32_t n_kvh,
                                        uint32_t hd, uint32_t kv_base,
                                        uint32_t max_kv, float scale,
                                        const int32_t *idx, uint32_t width) {
    constexpr uint32_t TILE = 64, GQAM = 16;
    const uint32_t kh = blockIdx.x, t = blockIdx.y;
    const uint32_t gqa = n_head / n_kvh;
    if (gqa == 0 || gqa > GQAM || hd != 256) return;
    uint32_t n_kv = kv_base + t + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    const uint32_t n_pos = INDEXED ? width : n_kv;
    const uint32_t i = threadIdx.x, w = i >> 5, lane = i & 31;

    __shared__ __half qsf16[GQAM][256];
    __shared__ __half tile[TILE][256];
    __shared__ __half pf16[GQAM][TILE];
    __shared__ float spart[2][GQAM][TILE];
    __shared__ float mm[GQAM], ssum[GQAM], al[GQAM];

    for (uint32_t e = i; e < GQAM * 256; e += blockDim.x) {
        const uint32_t g = e / 256, d = e - g * 256;
        qsf16[g][d] =
            __float2half(g < gqa
                             ? q[((size_t)t * n_head + kh * gqa + g) * 256 +
                                 d] *
                                   scale
                             : 0.f);
    }
    if (i < GQAM) { mm[i] = -1.0e30f; ssum[i] = 0.f; }

    /* O accumulators: 2 col-tiles per wave, 8 fp32 each. */
    wmma_c8 oacc[2];
    oacc[0] = (wmma_c8){};
    oacc[1] = (wmma_c8){};

    uint32_t q8b = (256 / 32u) * 34u;
    for (uint32_t base = 0; base < n_pos; base += TILE) {
        uint32_t lim = n_pos - base;
        if (lim > TILE) lim = TILE;
        __syncthreads();
        /* dequant K tile, warp per position, 8 values per lane; V rows go
         * to registers now and into the tile after the softmax (same
         * values as per-element loads; V latency under Q*K^T). */
        uint4 vr[TILE / 8];
#pragma unroll
        for (uint32_t r = 0; r < TILE / 8; r++) {
            const uint32_t pos = w + 8 * r;
            int p = -1;
            if (pos < lim)
                p = INDEXED ? idx[(size_t)t * width + base + pos]
                            : (int)(base + pos);
            const size_t ro = ((size_t)(p < 0 ? 0 : p) * n_kvh + kh) * q8b;
            *(uint4 *)(tile[pos] + 8u * lane) =
                q8row_ld(p >= 0 ? kcache + ro : nullptr, lane);
            vr[r] = q8row_ld(p >= 0 ? vcache + ro : nullptr, lane);
        }
        __syncthreads();
        /* S[16][64] = Q·Kᵀ: wave w covers j-tile w&3, k-half w>>2 (128 of
         * hd each, 8 wmma steps). */
        {
            const uint32_t jt = w & 3, kh2 = w >> 2;
            wmma_c8 sc = (wmma_c8){};
            for (uint32_t ks = 0; ks < 128; ks += 16) {
                wmma_a16 a16, b16;
                const uint32_t k0 = kh2 * 128 + ks;
#pragma unroll
                for (uint32_t u = 0; u < 16; u++) {
                    a16[u] = __half2float(qsf16[lane & 15][k0 + u]);
                    b16[u] = __half2float(tile[jt * 16 + (lane & 15)][k0 + u]);
                }
                sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a16, b16, sc);
            }
#pragma unroll
            for (uint32_t u = 0; u < 8; u++)
                spart[kh2][2 * u + (lane >> 4)][jt * 16 + (lane & 15)] =
                    sc[u];
        }
        __syncthreads();
        /* warp per head: combine the two k-half partials, mask invalid
         * positions, online softmax, P in fp16.  32 lanes, 2 positions each
         * (lane, lane+32). */
        for (uint32_t g = w; g < gqa; g += 8) {
            const uint32_t l2 = lane + 32;
            int p0 = -1, p1 = -1;
            if (lane < lim)
                p0 = INDEXED ? idx[(size_t)t * width + base + lane]
                             : (int)(base + lane);
            if (l2 < lim)
                p1 = INDEXED ? idx[(size_t)t * width + base + l2]
                             : (int)(base + l2);
            const float dv0 = (lane < lim && p0 >= 0)
                                  ? spart[0][g][lane] + spart[1][g][lane]
                                  : -1.0e30f;
            const float dv1 = (l2 < lim && p1 >= 0)
                                  ? spart[0][g][l2] + spart[1][g][l2]
                                  : -1.0e30f;
            float tm = fmaxf(dv0, dv1);
            for (uint32_t o = 16; o; o >>= 1)
                tm = fmaxf(tm, __shfl_down(tm, o));
            tm = __shfl(tm, 0);
            const int live = tm > -1.0e20f;
            float m = mm[g], nm = m, alpha = 1.f;
            if (live && tm > m) {
                nm = tm;
                alpha = expf(m - nm);
            }
            const float wv0 = (live && dv0 > -1.0e20f) ? expf(dv0 - nm) : 0.f;
            const float wv1 = (live && dv1 > -1.0e20f) ? expf(dv1 - nm) : 0.f;
            pf16[g][lane] = __float2half(wv0);
            pf16[g][l2] = __float2half(wv1);
            float ts = wv0 + wv1;
            for (uint32_t o = 16; o; o >>= 1)
                ts += __shfl_down(ts, o);
            if (lane == 0) {
                mm[g] = nm;
                ssum[g] = ssum[g] * alpha + ts;
                al[g] = alpha;
            }
        }
        __syncthreads();
        /* rescale O by this tile's alpha, then dequant V and accumulate. */
#pragma unroll
        for (uint32_t ct = 0; ct < 2; ct++)
#pragma unroll
            for (uint32_t u = 0; u < 8; u++)
                oacc[ct][u] *= al[2 * u + (lane >> 4)];
#pragma unroll
        for (uint32_t r = 0; r < TILE / 8; r++)
            *(uint4 *)(tile[w + 8 * r] + 8u * lane) = vr[r];
        __syncthreads();
        for (uint32_t ct = 0; ct < 2; ct++) {
            const uint32_t c0 = (w * 2 + ct) * 16;
            for (uint32_t ks = 0; ks < TILE; ks += 16) {
                wmma_a16 a16, b16;
#pragma unroll
                for (uint32_t u = 0; u < 16; u++) {
                    a16[u] = __half2float(pf16[lane & 15][ks + u]);
                    b16[u] =
                        __half2float(tile[ks + u][c0 + (lane & 15)]);
                }
                oacc[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a16, b16, oacc[ct]);
            }
        }
    }
    /* out[t][kh*gqa+g][d] = acc/ssum * sigmoid(gate) */
    for (uint32_t ct = 0; ct < 2; ct++) {
        const uint32_t c0 = (w * 2 + ct) * 16;
#pragma unroll
        for (uint32_t u = 0; u < 8; u++) {
            const uint32_t g = 2 * u + (lane >> 4);
            if (g >= gqa) continue;
            const uint32_t d = c0 + (lane & 15);
            const uint32_t h = kh * gqa + g;
            const float inv = ssum[g] > 0.f ? 1.f / ssum[g] : 0.f;
            const float gv =
                1.f / (1.f + expf(-gate[((size_t)t * n_head + h) * 256 + d]));
            out[((size_t)t * n_head + h) * 256 + d] = oacc[ct][u] * inv * gv;
        }
    }
}

/* Prompt QSA attention over the union of T consecutive tokens' selections.
 * Neighbouring tokens pick overlapping 4-position blocks (union / one
 * selection ~1.4 at 4K context, ~2.2 at 24K+ for T = 4), so one block per
 * (kv head, T tokens) loads each K/V row once for all T*gqa query rows and
 * masks per (token, position).  qsa_union_k builds the union: ublk[g][u]
 * ascending block ids (position >> 2), umask[g][u] bit k*4 + (p & 3) when
 * token k of the group selects position p, ucnt[g] = U <= umax =
 * min(T * width, blocks).  Positions must be unique within a token's
 * selection (the topk output is); the per-token kernels would count a
 * repeated one twice.  Engine prompt (Q4_QSA_UNION=4 vs 0): QSA section
 * 0.90 -> 0.56 s per 5.5K chunk at 34K context, 34K prompt 1610 -> 1800,
 * 10K 1596 -> 1743 t/s. */
template <uint32_t T>
__global__ static void qsa_union_k(const int32_t *sel, uint32_t width,
                                   uint32_t n_tok, uint32_t nbw, uint32_t umax,
                                   int32_t *ublk, uint32_t *umask,
                                   uint32_t *ucnt) {
    extern __shared__ uint32_t qsa_un_sh[];
    uint32_t *bm = qsa_un_sh, *pre = bm + nbw, *um = pre + nbw;
    __shared__ uint32_t part[256];
    const uint32_t g = blockIdx.x, tx = threadIdx.x, nth = blockDim.x;
    const uint32_t t0 = g * T;
    const uint32_t nk = n_tok - t0 < T ? n_tok - t0 : T;
    const int32_t *sg = sel + (size_t)t0 * width;
    for (uint32_t i = tx; i < nbw; i += nth) bm[i] = 0u;
    for (uint32_t i = tx; i < umax; i += nth) um[i] = 0u;
    __syncthreads();
    for (uint32_t e = tx; e < nk * width; e += nth) {
        const int32_t p = sg[e];
        if (p >= 0) {
            const uint32_t b = (uint32_t)p >> 2;
            if ((b >> 5) < nbw) atomicOr(&bm[b >> 5], 1u << (b & 31u));
        }
    }
    __syncthreads();
    /* exclusive popcount scan over the bitmap words */
    const uint32_t chunk = (nbw + nth - 1u) / nth;
    const uint32_t w0 = tx * chunk;
    const uint32_t w1 = w0 + chunk < nbw ? w0 + chunk : nbw;
    uint32_t c = 0;
    for (uint32_t w = w0; w < w1; w++) c += __popc(bm[w]);
    part[tx] = c;
    __syncthreads();
    if (tx == 0) {
        uint32_t acc = 0;
        for (uint32_t i = 0; i < nth; i++) {
            const uint32_t v = part[i];
            part[i] = acc;
            acc += v;
        }
        ucnt[g] = acc < umax ? acc : umax;
    }
    __syncthreads();
    c = part[tx];
    for (uint32_t w = w0; w < w1; w++) {
        pre[w] = c;
        c += __popc(bm[w]);
    }
    __syncthreads();
    int32_t *ub = ublk + (size_t)g * umax;
    for (uint32_t w = tx; w < nbw; w += nth) {
        uint32_t bits = bm[w], idx = pre[w];
        while (bits) {
            const uint32_t b = __ffs(bits) - 1u;
            if (idx < umax) ub[idx] = (int32_t)(w * 32u + b);
            idx++;
            bits &= bits - 1u;
        }
    }
    for (uint32_t e = tx; e < nk * width; e += nth) {
        const int32_t p = sg[e];
        if (p < 0) continue;
        const uint32_t b = (uint32_t)p >> 2, w = b >> 5;
        if (w >= nbw) continue;
        const uint32_t idx =
            pre[w] + __popc(bm[w] & ((1u << (b & 31u)) - 1u));
        if (idx < umax)
            atomicOr(&um[idx], 1u << ((e / width) * 4u + ((uint32_t)p & 3u)));
    }
    __syncthreads();
    uint32_t *umg = umask + (size_t)g * umax;
    const uint32_t U = ucnt[g];
    for (uint32_t i = tx; i < U; i += nth) umg[i] = um[i];
}

/* Workgroup barrier for LDS-only exchange: waits for this wave's LDS ops,
 * not its global loads (__syncthreads() also waits vmcnt(0)). */
__device__ static inline void lds_sync() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

/* xor-lane exchange within 16-lane rows (DPP row_xmask, gfx10+): one VALU
 * op instead of a ds_bpermute round trip. */
template <int M>
__device__ static inline float dpp_xmask(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x160 + M, 0xF, 0xF, true));
}

/* grid (kv head, group of T tokens), 8 waves; gqa 12, q8 KV, hd 256.  Rows
 * r = k * 12 + head over MT 16-row WMMA tiles; 32-position tiles (8 union
 * blocks).  Per tile:
 *  - K rows dequantized into `tile` (warp per row), V into registers as one
 *    q8 block per lane (lane = position, wave = 32 values);
 *  - S = Q*K^T by waves that keep their (row tile, k half) of Q in
 *    registers for the whole loop (WPS waves per pair split the columns);
 *  - softmax with 8 lanes per row, one union block per lane (its mask
 *    nibble), reductions by DPP row_xmask (the per-row lane shuffles were
 *    ~2K clocks: 850 of 3300 us on a 512-token, 28K-context layer);
 *  - V written transposed (`vt`, aliasing the K tile) so the P*V B operand
 *    is two 16-byte LDS reads instead of 16 u16 column reads. */
template <uint32_t T>
__global__ static __launch_bounds__(256) void gqa_union_wmma_k(
    const float *q, const uint8_t *kcache, const uint8_t *vcache,
    const float *gate, float *out, uint32_t n_head, uint32_t n_kvh,
    uint32_t n_tok, float scale, const int32_t *ublk, const uint32_t *umask,
    const uint32_t *ucnt, uint32_t umax) {
    constexpr uint32_t GQA = 12, ROWS = T * GQA, MT = (ROWS + 15u) / 16u;
    constexpr uint32_t TILE = 32, TB = TILE / 4, NJ = TILE / 16, NR = TILE / 8;
    constexpr uint32_t LD = 256 + 8, LDV = TILE + 8;
    constexpr uint32_t WPS = 8u / (2u * MT);
    static_assert(WPS >= 1 && NJ % WPS == 0, "S wave split");
    constexpr uint32_t KV_H = TILE * LD > 256 * LDV ? TILE * LD : 256 * LDV;
    const uint32_t kh = blockIdx.x, g = blockIdx.y;
    if (n_head != n_kvh * GQA) return;
    const uint32_t t0 = g * T;
    const uint32_t i = threadIdx.x, w = i >> 5, lane = i & 31;
    const uint32_t U = ucnt[g];
    const int32_t *ub = ublk + (size_t)g * umax;
    const uint32_t *umk = umask + (size_t)g * umax;

    __shared__ __attribute__((aligned(16))) __half kvbuf[KV_H];
    __half (*tile)[LD] = (__half (*)[LD])kvbuf;
    __half (*vt)[LDV] = (__half (*)[LDV])kvbuf;
    __shared__ __attribute__((aligned(16))) float spart[2][MT * 16][TILE];
    __shared__ __attribute__((aligned(16))) __half pf16[MT * 16][TILE];
    __shared__ float mm[MT * 16], ssum[MT * 16], al[MT * 16];
    __shared__ uint32_t tmask[TB];

    const bool sw = w < 2 * MT * WPS;
    const uint32_t smt = w / (2 * WPS), kh2 = (w / WPS) & 1u, jg = w % WPS;
    wmma_a16 qa[8];
    {
        const uint32_t r = smt * 16 + (lane & 15u);
        const uint32_t k = r / GQA, hh = r - k * GQA;
        const bool ok = sw && r < ROWS && t0 + k < n_tok;
        const float *qr =
            ok ? q + ((size_t)(t0 + k) * n_head + kh * GQA + hh) * 256 +
                     kh2 * 128
               : nullptr;
#pragma unroll
        for (uint32_t s = 0; s < 8; s++)
#pragma unroll
            for (uint32_t u = 0; u < 16; u++)
                qa[s][u] = qr ? (_Float16)__float2half(qr[s * 16 + u] * scale)
                              : (_Float16)0.f;
    }
    if (i < MT * 16) { mm[i] = -1.0e30f; ssum[i] = 0.f; }

    wmma_c8 oacc[MT][2];
#pragma unroll
    for (uint32_t m = 0; m < MT; m++) {
        oacc[m][0] = (wmma_c8){};
        oacc[m][1] = (wmma_c8){};
    }

    const uint32_t q8b = (256 / 32u) * 34u;
    for (uint32_t u0 = 0; u0 < U; u0 += TB) {
        lds_sync();
        uint4 vq[2];
        float vs = 0.f;
        /* a position of a union block that no token of the group selected
         * stays zero: it may lie past the last written cell (the block of
         * the prompt's final position), whose bytes are not a KV row */
        constexpr uint32_t ANY = T == 4 ? 0x1111u : 0x11u;
        {
            const uint32_t u = u0 + (lane >> 2);
            const int p = u < U && ((umk[u] >> (lane & 3u)) & ANY)
                              ? ub[u] * 4 + (int)(lane & 3u)
                              : -1;
            vq[0] = vq[1] = make_uint4(0u, 0u, 0u, 0u);
            if (p >= 0) {
                const uint8_t *bv =
                    vcache + ((size_t)p * n_kvh + kh) * q8b + w * 34u;
                vs = f16d(bv);
                __builtin_memcpy(&vq[0], bv + 2, 32);
            }
        }
#pragma unroll
        for (uint32_t r = 0; r < NR; r++) {
            const uint32_t j = w + 8 * r, u = u0 + (j >> 2);
            const int p = u < U && ((umk[u] >> (j & 3u)) & ANY)
                              ? ub[u] * 4 + (int)(j & 3u)
                              : -1;
            const size_t ro = ((size_t)(p < 0 ? 0 : p) * n_kvh + kh) * q8b;
            *(uint4 *)(&tile[j][8u * lane]) =
                q8row_ld(p >= 0 ? kcache + ro : nullptr, lane);
        }
        if (i < TB) tmask[i] = u0 + i < U ? umk[u0 + i] : 0u;
        lds_sync();
        if (sw) {
#pragma unroll
            for (uint32_t jj = 0; jj < NJ / WPS; jj++) {
                const uint32_t jt = jg * (NJ / WPS) + jj;
                wmma_c8 sc = (wmma_c8){};
#pragma unroll
                for (uint32_t s = 0; s < 8; s++) {
                    const wmma_a16 b16 = *(const wmma_a16 *)(
                        &tile[jt * 16 + (lane & 15u)][kh2 * 128 + s * 16]);
                    sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[s], b16,
                                                                     sc);
                }
#pragma unroll
                for (uint32_t u = 0; u < 8; u++)
                    spart[kh2][smt * 16 + 2 * u + (lane >> 4)]
                         [jt * 16 + (lane & 15u)] = sc[u];
            }
        }
        lds_sync();
        {
            const uint32_t sub = lane & 7u, j = sub * 4;
#pragma unroll
            for (uint32_t r0 = 0; r0 < MT * 16; r0 += 32) {
                const uint32_t r = r0 + w * 4 + (lane >> 3);
                if (r0 + w * 4 >= MT * 16) continue;
                const uint32_t k = r / GQA;
                const bool rv = r < ROWS && t0 + k < n_tok;
                const uint32_t nib = rv ? (tmask[sub] >> (k * 4u)) & 15u : 0u;
                const float4 s0 = *(const float4 *)&spart[0][r][j];
                const float4 s1 = *(const float4 *)&spart[1][r][j];
                float dv[4] = {s0.x + s1.x, s0.y + s1.y, s0.z + s1.z,
                               s0.w + s1.w};
                float tm = -1.0e30f;
#pragma unroll
                for (uint32_t e = 0; e < 4; e++) {
                    if (!((nib >> e) & 1u)) dv[e] = -1.0e30f;
                    tm = fmaxf(tm, dv[e]);
                }
                tm = fmaxf(tm, dpp_xmask<1>(tm));
                tm = fmaxf(tm, dpp_xmask<2>(tm));
                tm = fmaxf(tm, dpp_xmask<4>(tm));
                const int live = tm > -1.0e20f;
                const float m = mm[r];
                float nm = m, alpha = 1.f;
                if (live && tm > m) {
                    nm = tm;
                    alpha = expf(m - nm);
                }
                union { __half h[4]; uint2 u; } pk;
                float ts = 0.f;
#pragma unroll
                for (uint32_t e = 0; e < 4; e++) {
                    const float wv =
                        (live && dv[e] > -1.0e20f) ? expf(dv[e] - nm) : 0.f;
                    pk.h[e] = __float2half(wv);
                    ts += wv;
                }
                *(uint2 *)&pf16[r][j] = pk.u;
                ts += dpp_xmask<1>(ts);
                ts += dpp_xmask<2>(ts);
                ts += dpp_xmask<4>(ts);
                if (sub == 0) {
                    mm[r] = nm;
                    ssum[r] = ssum[r] * alpha + ts;
                    al[r] = alpha;
                }
            }
        }
        lds_sync();
#pragma unroll
        for (uint32_t m = 0; m < MT; m++)
#pragma unroll
            for (uint32_t ct = 0; ct < 2; ct++)
#pragma unroll
                for (uint32_t u = 0; u < 8; u++)
                    oacc[m][ct][u] *= al[m * 16 + 2 * u + (lane >> 4)];
        {
            const int8_t *qv = (const int8_t *)&vq[0];
#pragma unroll
            for (uint32_t e = 0; e < 32; e++)
                vt[w * 32 + e][lane] = __float2half(vs * (float)qv[e]);
        }
        lds_sync();
#pragma unroll
        for (uint32_t ks = 0; ks < TILE; ks += 16) {
            wmma_a16 bv[2];
#pragma unroll
            for (uint32_t ct = 0; ct < 2; ct++)
                bv[ct] = *(const wmma_a16 *)(
                    &vt[(w * 2 + ct) * 16 + (lane & 15u)][ks]);
#pragma unroll
            for (uint32_t m = 0; m < MT; m++) {
                const wmma_a16 a16 =
                    *(const wmma_a16 *)(&pf16[m * 16 + (lane & 15u)][ks]);
#pragma unroll
                for (uint32_t ct = 0; ct < 2; ct++)
                    oacc[m][ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        a16, bv[ct], oacc[m][ct]);
            }
        }
    }
    lds_sync();
#pragma unroll
    for (uint32_t m = 0; m < MT; m++)
#pragma unroll
        for (uint32_t ct = 0; ct < 2; ct++) {
            const uint32_t d = (w * 2 + ct) * 16 + (lane & 15u);
#pragma unroll
            for (uint32_t u = 0; u < 8; u++) {
                const uint32_t r = m * 16 + 2 * u + (lane >> 4);
                const uint32_t k = r / GQA, hh = r - k * GQA;
                if (r >= ROWS || t0 + k >= n_tok) continue;
                const size_t o =
                    ((size_t)(t0 + k) * n_head + kh * GQA + hh) * 256 + d;
                const float inv = ssum[r] > 0.f ? 1.f / ssum[r] : 0.f;
                const float gv = 1.f / (1.f + expf(-gate[o]));
                out[o] = oacc[m][ct][u] * inv * gv;
            }
        }
}

/* T = 2 or 4 (Q4_QSA_UNION, default 4, 0 = per-token gqa_prefill_wmma_k);
 * false when the shape does not fit (caller falls back). */
static bool qsa_union_launch(const float *d_q, const void *d_kcache,
                             const void *d_vcache, const float *d_gate,
                             float *d_out, const int32_t *idx, uint32_t width,
                             uint32_t n_head, uint32_t n_kvh, uint32_t n_tok,
                             uint32_t max_kv, float scale, uint32_t T) {
    static size_t bc, mc, cc;
    if (n_head != n_kvh * 12u || (T != 2 && T != 4)) return false;
    const uint32_t ng = (n_tok + T - 1) / T;
    const uint32_t nbw = ((max_kv + 3u) / 4u + 31u) / 32u;
    /* each token's width positions touch at most width blocks */
    const uint32_t umax = T * width < nbw * 32u ? T * width : nbw * 32u;
    const size_t shb = (2u * (size_t)nbw + umax) * 4u;
    if (shb > 60u * 1024u) return false;
    if (!grow((void **)&g_un_blk, &bc, (size_t)ng * umax * 4) ||
        !grow((void **)&g_un_mask, &mc, (size_t)ng * umax * 4) ||
        !grow((void **)&g_un_cnt, &cc, (size_t)ng * 4))
        return false;
    dim3 ga(n_kvh, ng);
#define UNION_T(TT)                                                           \
    do {                                                                      \
        qsa_union_k<TT><<<ng, 256, shb, g_str>>>(idx, width, n_tok, nbw,      \
                                                 umax, g_un_blk, g_un_mask,   \
                                                 g_un_cnt);                   \
        gqa_union_wmma_k<TT><<<ga, 256, 0, g_str>>>(                          \
            d_q, (const uint8_t *)d_kcache, (const uint8_t *)d_vcache,        \
            d_gate, d_out, n_head, n_kvh, n_tok, scale, g_un_blk, g_un_mask,  \
            g_un_cnt, umax);                                                  \
    } while (0)
    if (T == 4) UNION_T(4);
    else UNION_T(2);
#undef UNION_T
    return hipGetLastError() == hipSuccess;
}

/* Split-KV variant of gqa_prefill_wmma_k for small batches (MTP verify,
 * 1..8 rows): grid (kv head, split, token row); each split runs `tps`
 * 64-position tiles of its row's positions and leaves (m, l, acc) partials
 * for gqa_dec_comb_k, like gqa_dec_split_k.  The scalar split kernel spent
 * ~110 us on one row of a 2K selection (~20 GB/s of KV); the WMMA tiles
 * take the Q*K^T and P*V products off the shared-memory FMA loops. */
template <bool INDEXED, int KIND = 2>
__global__ static void gqa_split_wmma_k(const float *q, const uint8_t *kcache,
                                        const uint8_t *vcache,
                                        float *part,
                                        uint32_t n_head, uint32_t n_kvh,
                                        uint32_t hd, uint32_t kv_base,
                                        uint32_t max_kv, float scale,
                                        const int32_t *idx, uint32_t width,
                                        uint32_t S, uint32_t tps,
                                        const int32_t *pp = nullptr) {
    constexpr uint32_t TILE = 64, GQAM = 16;
    const uint32_t kh = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const uint32_t gqa = n_head / n_kvh;
    if (gqa == 0 || gqa > GQAM || hd != 256) return;
    /* pp: cache length from the device counter (graph-captured decode);
     * tps == 0: tiles per split derived from it at run time. */
    uint32_t n_kv = (pp ? (uint32_t)*pp : kv_base) + t + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    const uint32_t n_all = INDEXED ? width : n_kv;
    if (!tps) tps = ((n_all + TILE - 1u) / TILE + S - 1u) / S;
    /* this split's tiles: [p0, n_pos) */
    const uint32_t p0 = sp * tps * TILE;
    const uint32_t n_pos =
        p0 + tps * TILE < n_all ? p0 + tps * TILE : n_all;
    float *pv = part + (size_t)t * n_kvh * S * gqa * (hd + 2u) +
                ((size_t)kh * S + sp) * gqa * (hd + 2u);
    const uint32_t i = threadIdx.x, w = i >> 5, lane = i & 31;

    __shared__ __half qsf16[GQAM][256];
    __shared__ __half tile[TILE][256];
    __shared__ __half pf16[GQAM][TILE];
    __shared__ float spart[2][GQAM][TILE];
    __shared__ float mm[GQAM], ssum[GQAM], al[GQAM];

    for (uint32_t e = i; e < GQAM * 256; e += blockDim.x) {
        const uint32_t g = e / 256, d = e - g * 256;
        qsf16[g][d] =
            __float2half(g < gqa
                             ? q[((size_t)t * n_head + kh * gqa + g) * 256 +
                                 d] *
                                   scale
                             : 0.f);
    }
    if (i < GQAM) { mm[i] = -1.0e30f; ssum[i] = 0.f; }

    /* O accumulators: 2 col-tiles per wave, 8 fp32 each. */
    wmma_c8 oacc[2];
    oacc[0] = (wmma_c8){};
    oacc[1] = (wmma_c8){};

    uint32_t q8b = (256 / 32u) * 34u;
    for (uint32_t base = p0; base < n_pos; base += TILE) {
        uint32_t lim = n_pos - base;
        if (lim > TILE) lim = TILE;
        __syncthreads();
        /* dequant K tile, warp per position; V rows go to registers now
         * and into the tile after the softmax (latency under Q*K^T). */
        uint4 vr[TILE / 8];
#pragma unroll
        for (uint32_t r = 0; r < TILE / 8; r++) {
            const uint32_t pos = w + 8 * r;
            int p = -1;
            if (pos < lim)
                p = INDEXED ? idx[(size_t)t * width + base + pos]
                            : (int)(base + pos);
            const size_t ro = ((size_t)(p < 0 ? 0 : p) * n_kvh + kh) * q8b;
            if (KIND == 2) {
                *(uint4 *)(tile[pos] + 8u * lane) =
                    q8row_ld(p >= 0 ? kcache + ro : nullptr, lane);
                vr[r] = q8row_ld(p >= 0 ? vcache + ro : nullptr, lane);
            } else {
                const size_t fo = ((size_t)(p < 0 ? 0 : p) * n_kvh + kh) * 256u;
                *(uint4 *)(tile[pos] + 8u * lane) = f32row_ld(
                    p >= 0 ? (const float *)kcache + fo : nullptr, lane);
                vr[r] = f32row_ld(p >= 0 ? (const float *)vcache + fo : nullptr,
                                  lane);
            }
        }
        __syncthreads();
        /* S[16][64] = Q·Kᵀ: wave w covers j-tile w&3, k-half w>>2 (128 of
         * hd each, 8 wmma steps). */
        {
            const uint32_t jt = w & 3, kh2 = w >> 2;
            wmma_c8 sc = (wmma_c8){};
            for (uint32_t ks = 0; ks < 128; ks += 16) {
                wmma_a16 a16, b16;
                const uint32_t k0 = kh2 * 128 + ks;
#pragma unroll
                for (uint32_t u = 0; u < 16; u++) {
                    a16[u] = __half2float(qsf16[lane & 15][k0 + u]);
                    b16[u] = __half2float(tile[jt * 16 + (lane & 15)][k0 + u]);
                }
                sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a16, b16, sc);
            }
#pragma unroll
            for (uint32_t u = 0; u < 8; u++)
                spart[kh2][2 * u + (lane >> 4)][jt * 16 + (lane & 15)] =
                    sc[u];
        }
        __syncthreads();
        /* warp per head: combine the two k-half partials, mask invalid
         * positions, online softmax, P in fp16.  32 lanes, 2 positions each
         * (lane, lane+32). */
        for (uint32_t g = w; g < gqa; g += 8) {
            const uint32_t l2 = lane + 32;
            int p0 = -1, p1 = -1;
            if (lane < lim)
                p0 = INDEXED ? idx[(size_t)t * width + base + lane]
                             : (int)(base + lane);
            if (l2 < lim)
                p1 = INDEXED ? idx[(size_t)t * width + base + l2]
                             : (int)(base + l2);
            const float dv0 = (lane < lim && p0 >= 0)
                                  ? spart[0][g][lane] + spart[1][g][lane]
                                  : -1.0e30f;
            const float dv1 = (l2 < lim && p1 >= 0)
                                  ? spart[0][g][l2] + spart[1][g][l2]
                                  : -1.0e30f;
            float tm = fmaxf(dv0, dv1);
            for (uint32_t o = 16; o; o >>= 1)
                tm = fmaxf(tm, __shfl_down(tm, o));
            tm = __shfl(tm, 0);
            const int live = tm > -1.0e20f;
            float m = mm[g], nm = m, alpha = 1.f;
            if (live && tm > m) {
                nm = tm;
                alpha = expf(m - nm);
            }
            const float wv0 = (live && dv0 > -1.0e20f) ? expf(dv0 - nm) : 0.f;
            const float wv1 = (live && dv1 > -1.0e20f) ? expf(dv1 - nm) : 0.f;
            pf16[g][lane] = __float2half(wv0);
            pf16[g][l2] = __float2half(wv1);
            float ts = wv0 + wv1;
            for (uint32_t o = 16; o; o >>= 1)
                ts += __shfl_down(ts, o);
            if (lane == 0) {
                mm[g] = nm;
                ssum[g] = ssum[g] * alpha + ts;
                al[g] = alpha;
            }
        }
        __syncthreads();
        /* rescale O by this tile's alpha, then dequant V and accumulate. */
#pragma unroll
        for (uint32_t ct = 0; ct < 2; ct++)
#pragma unroll
            for (uint32_t u = 0; u < 8; u++)
                oacc[ct][u] *= al[2 * u + (lane >> 4)];
#pragma unroll
        for (uint32_t r = 0; r < TILE / 8; r++)
            *(uint4 *)(tile[w + 8 * r] + 8u * lane) = vr[r];
        __syncthreads();
        for (uint32_t ct = 0; ct < 2; ct++) {
            const uint32_t c0 = (w * 2 + ct) * 16;
            for (uint32_t ks = 0; ks < TILE; ks += 16) {
                wmma_a16 a16, b16;
#pragma unroll
                for (uint32_t u = 0; u < 16; u++) {
                    a16[u] = __half2float(pf16[lane & 15][ks + u]);
                    b16[u] =
                        __half2float(tile[ks + u][c0 + (lane & 15)]);
                }
                oacc[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a16, b16, oacc[ct]);
            }
        }
    }
    /* partials in gqa_dec_split_k's layout: m, l, unnormalized acc */
    for (uint32_t ct = 0; ct < 2; ct++) {
        const uint32_t c0 = (w * 2 + ct) * 16;
#pragma unroll
        for (uint32_t u = 0; u < 8; u++) {
            const uint32_t g = 2 * u + (lane >> 4);
            if (g >= gqa) continue;
            pv[g * (hd + 2u) + 2u + c0 + (lane & 15)] = oacc[ct][u];
        }
    }
    if (i < gqa) {
        pv[i * (hd + 2u)] = mm[i];
        pv[i * (hd + 2u) + 1u] = ssum[i];
    }
}

__global__ static void f32_to_f16_k(const float *src, __half *dst, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

extern "C" bool q4_hip_f32_to_f16(void *dst_f16, const float *src, uint64_t n) {
    if (!g_ok || !dst_f16 || !src || !n) return false;
    uint32_t block = 256;
    uint32_t grid = (uint32_t)((n + block - 1) / block);
    f32_to_f16_k<<<grid, block, 0, g_str>>>(src, (__half *)dst_f16, n);
    return hipGetLastError() == hipSuccess;
}

__global__ static void f32_to_q8_k(const float *src, uint8_t *dst, uint64_t nblocks) {
    uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float *x = src + b * 32u;
    float amax = 0.f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    uint8_t *o = dst + b * 34u;
    __half h = __float2half(d);
    memcpy(o, &h, 2);
    for (int i = 0; i < 32; i++) {
        int q = (int)rintf(x[i] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        o[2 + i] = (uint8_t)(int8_t)q;
    }
}

extern "C" bool q4_hip_f32_to_q8(void *dst_q8, const float *src, uint64_t n) {
    if (!g_ok || !dst_q8 || !src || n == 0 || (n % 32u) != 0) return false;
    uint64_t nb = n / 32u;
    uint32_t block = 256;
    uint32_t grid = (uint32_t)((nb + block - 1) / block);
    f32_to_q8_k<<<grid, block, 0, g_str>>>(src, (uint8_t *)dst_q8, nb);
    return hipGetLastError() == hipSuccess;
}

/* Small batches (2..8 rows: MTP verify, short tails): one split-KV decode
 * per token.  The prefill kernels run one block per (kv head, token) — 8
 * blocks for a 4-row verify — and took ~750 us per QSA layer at a 2.4K
 * context; split-K spreads each row over n_kvh x S blocks (each row's
 * math is the single-row decode's at this S).  Q4_VER_SPLIT=0 keeps the
 * prefill kernels. */
static bool qsa_small_split(const float *d_q, const void *d_kcache,
                            const void *d_vcache, const float *d_gate,
                            float *d_out, const int32_t *idx, uint32_t width,
                            uint32_t n_head, uint32_t n_kvh, uint32_t hd,
                            uint32_t n_tok, uint32_t kv_base, uint32_t max_kv,
                            float scale, int kv_f16) {
    static int on = -1;
    if (on < 0) on = !(getenv("Q4_VER_SPLIT") && getenv("Q4_VER_SPLIT")[0] == '0');
    if (!on || n_tok < 2 || n_tok > 8 || !g_attn_part || n_kvh > 4 ||
        hd > 256 || n_head / n_kvh > GQA_DEC_MAX_GQA ||
        (getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0'))
        return false;
    /* All rows in one launch (grid.z = token).  The partials buffer holds
     * 4 x GQA_DEC_SPLIT_S (kv head, split) blocks, so S shrinks with the
     * row count; at least one 32-position tile per split. */
    uint32_t S = g_dec_split_s();
    const uint32_t cap = 4u * GQA_DEC_SPLIT_S / (n_kvh * n_tok);
    if (S > cap) S = cap;
    const uint32_t n_pos = idx ? width : kv_base + n_tok;
    const uint32_t tiles = (n_pos + 31u) / 32u;
    if (S > tiles) S = tiles;
    if (S < 1) S = 1;
    const dim3 g2(n_kvh, S, n_tok);
#define SMALL_SPLIT(K)                                                          \
    do {                                                                        \
        if (idx)                                                                \
            gqa_dec_split_k<K, true><<<g2, 256, 0, g_str>>>(                    \
                d_q, (const uint8_t *)d_kcache, (const uint8_t *)d_vcache,      \
                g_attn_part, n_head, n_kvh, hd, 0, max_kv, scale, idx, width,   \
                S, nullptr);                                                    \
        else                                                                    \
            gqa_dec_split_k<K, false><<<g2, 256, 0, g_str>>>(                   \
                d_q, (const uint8_t *)d_kcache, (const uint8_t *)d_vcache,      \
                g_attn_part, n_head, n_kvh, hd, kv_base, max_kv, scale,         \
                nullptr, 0, S, nullptr);                                        \
    } while (0)
    static int swm = -1;
    if (swm < 0) swm = !(getenv("Q4_QSA_SWMMA") && getenv("Q4_QSA_SWMMA")[0] == '0');
    if (swm && kv_f16 == 2 && hd == 256 && n_head / n_kvh <= 16) {
        /* tiles per split: enough splits to fill the GPU, partials capped */
        static int tps_env = -1;
        if (tps_env < 0) {
            const char *e = getenv("Q4_SWMMA_TPS");
            tps_env = e ? atoi(e) : 0;
        }
        const uint32_t t64 = (n_pos + 63u) / 64u;
        uint32_t tps = tps_env > 0 ? (uint32_t)tps_env : 1u;
        uint32_t S2 = (t64 + tps - 1) / tps;
        if (S2 > cap) {
            S2 = cap;
            tps = (t64 + S2 - 1) / S2;
            S2 = (t64 + tps - 1) / tps;
        }
        if (S2 < 1) S2 = 1;
        S = S2;
        const dim3 gw(n_kvh, S, n_tok);
        if (idx)
            gqa_split_wmma_k<true><<<gw, 256, 0, g_str>>>(
                d_q, (const uint8_t *)d_kcache, (const uint8_t *)d_vcache,
                g_attn_part, n_head, n_kvh, hd, 0, max_kv, scale, idx, width,
                S, tps);
        else
            gqa_split_wmma_k<false><<<gw, 256, 0, g_str>>>(
                d_q, (const uint8_t *)d_kcache, (const uint8_t *)d_vcache,
                g_attn_part, n_head, n_kvh, hd, kv_base, max_kv, scale,
                nullptr, 0, S, tps);
    } else if (kv_f16 == 2) SMALL_SPLIT(2);
    else if (kv_f16 == 1) SMALL_SPLIT(1);
    else SMALL_SPLIT(0);
#undef SMALL_SPLIT
    gqa_dec_comb_k<<<dim3(n_head, n_tok), 256, 0, g_str>>>(
        g_attn_part, d_gate, d_out, n_head, n_kvh, hd, S);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_qsa_decode_n(const float *d_q, const void *d_kcache,
                                    const void *d_vcache, const float *d_gate,
                                    float *d_out, uint32_t n_head, uint32_t n_kvh,
                                    uint32_t hd, uint32_t n_tok, uint32_t kv_base,
                                    uint32_t max_kv, float scale, int kv_f16) {
    if (!g_ok || !d_q || !d_kcache || !d_vcache || !d_gate || !d_out) return false;
    if (n_tok == 0 || hd > 256 || hd == 0 || n_head == 0 || n_kvh == 0) return false;
    (void)hipGetLastError();
    if (qsa_small_split(d_q, d_kcache, d_vcache, d_gate, d_out, nullptr, 0,
                        n_head, n_kvh, hd, n_tok, kv_base, max_kv, scale,
                        kv_f16))
        return true;
    /* n_tok > 1 is prefill: one block per (kv head, token) shares each K/V
     * tile across all its query heads. n_tok == 1 (decode, incl. the graph)
     * keeps gqa_decode_fast. */
    static int qsa_pf = -1;
    if (qsa_pf < 0)
        qsa_pf = !(getenv("Q4_QSA_PF") && getenv("Q4_QSA_PF")[0] == '0');
    int prefill = n_tok > 1 && n_head / n_kvh <= 12 && qsa_pf;
    if (!prefill && n_tok == 1 && g_attn_part && n_kvh <= 4 && hd <= 256 &&
        n_head / n_kvh <= GQA_DEC_MAX_GQA &&
        !(getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0')) {
        const uint32_t S = g_dec_split_s();
        dim3 g2(n_kvh, S);
#define DEC_SPLIT(K)                                                          \
        do {                                                                  \
            gqa_dec_split_k<K, false><<<g2, 256, 0, g_str>>>(                   \
                (const float *)d_q, (const uint8_t *)d_kcache,                \
                (const uint8_t *)d_vcache, g_attn_part, n_head, n_kvh, hd,    \
                kv_base, max_kv, scale, nullptr, 0, S, nullptr);              \
            gqa_dec_comb_k<<<n_head, 256, 0, g_str>>>(g_attn_part, d_gate,    \
                                                    d_out, n_head, n_kvh,   \
                                                    hd, S);                 \
        } while (0)
        if (kv_f16 == 2) DEC_SPLIT(2);
        else if (kv_f16 == 1) DEC_SPLIT(1);
        else DEC_SPLIT(0);
#undef DEC_SPLIT
        if (hipGetLastError() != hipSuccess) return false;
        return true;
    }
    dim3 g(prefill ? n_kvh : n_head, n_tok);
    const char *qw = getenv("Q4_QSA_WMMA");
    const int wmma_ok = prefill && kv_f16 == 2 && hd == 256 &&
                      (!qw || qw[0] != '0');
    if (kv_f16 == 2)
        (wmma_ok ? gqa_prefill_wmma_k<false>
                 : (prefill ? gqa_prefill_k<2, 0> : gqa_decode_fast<2, 0>))<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out, n_head,
                                          n_kvh, hd, kv_base, max_kv, scale, nullptr, 0);
    else if (kv_f16 == 1)
        (prefill ? gqa_prefill_k<1, 0> : gqa_decode_fast<1, 0>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out, n_head,
                                          n_kvh, hd, kv_base, max_kv, scale, nullptr, 0);
    else
        (prefill ? gqa_prefill_k<0, 0> : gqa_decode_fast<0, 0>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out, n_head,
                                          n_kvh, hd, kv_base, max_kv, scale, nullptr, 0);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "q4 hip: qsa launch %s\n", hipGetErrorString(e));
        return false;
    }
    return true;
}

/* Whether the device-pos dense decode path can run — lets the host decide
 * QSA graphability before capture instead of failing inside it. */
extern "C" bool q4_hip_qsa_dec_d_ok(uint32_t n_head, uint32_t n_kvh,
                                    uint32_t hd) {
    return g_ok && g_pos_d && g_attn_part && n_kvh > 0 && n_kvh <= 4 &&
           hd > 0 && hd <= 256 && n_head / n_kvh <= GQA_DEC_MAX_GQA &&
           !(getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0');
}

/* Dense single-token decode whose kv length comes from the pushed device
 * position — graphable. Split path only; falls back cleanly when its guards
 * do not hold. */
extern "C" bool q4_hip_qsa_decode_d(const float *d_q, const void *d_kcache,
                                    const void *d_vcache, const float *d_gate,
                                    float *d_out, uint32_t n_head,
                                    uint32_t n_kvh, uint32_t hd,
                                    uint32_t max_kv, float scale, int kv_f16) {
    if (!g_ok || !g_pos_d || !d_q || !d_kcache || !d_vcache || !d_gate ||
        !d_out)
        return false;
    if (hd == 0 || hd > 256 || n_head == 0 || n_kvh == 0) return false;
    (void)hipGetLastError();
    if (!g_attn_part || n_kvh > 4 ||
        n_head / n_kvh > GQA_DEC_MAX_GQA ||
        (getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0'))
        return false;
    const uint32_t S = g_dec_split_s();
    dim3 g2(n_kvh, S);
#define DEC_SPLITD(K)                                                         \
    do {                                                                      \
        gqa_dec_split_k<K, false><<<g2, 256, 0, g_str>>>(                       \
            (const float *)d_q, (const uint8_t *)d_kcache,                    \
            (const uint8_t *)d_vcache, g_attn_part, n_head, n_kvh, hd,        \
            0, max_kv, scale, nullptr, 0, S, g_pos_d);                        \
        gqa_dec_comb_k<<<n_head, 256, 0, g_str>>>(g_attn_part, d_gate,        \
                                                d_out, n_head, n_kvh,       \
                                                hd, S);                     \
    } while (0)
    if (kv_f16 == 2) DEC_SPLITD(2);
    else if (kv_f16 == 1) DEC_SPLITD(1);
    else DEC_SPLITD(0);
#undef DEC_SPLITD
    return hipGetLastError() == hipSuccess;
}

/* Cache length from the kv counter. Rope stays on g_pos_d (the prep kernels). */
extern "C" bool q4_hip_qsa_decode_kv(const float *d_q, const void *d_kcache,
                                     const void *d_vcache, const float *d_gate,
                                     float *d_out, uint32_t n_head,
                                     uint32_t n_kvh, uint32_t hd,
                                     uint32_t max_kv, float scale, int kv_f16) {
    if (!g_ok || !g_kv_d || !d_q || !d_kcache || !d_vcache || !d_gate || !d_out)
        return false;
    if (hd == 0 || hd > 256 || n_head == 0 || n_kvh == 0) return false;
    (void)hipGetLastError();
    if (!g_attn_part || n_kvh > 4 || n_head / n_kvh > GQA_DEC_MAX_GQA ||
        (getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0'))
        return false;
    /* WMMA split over the device-counted cache (Q4_KV_SWMMA=0: scalar):
     * f32 or q8 KV, hd 256; S splits with tiles per split set in-kernel. */
    static int kvw = -1;
    if (kvw < 0) kvw = !(getenv("Q4_KV_SWMMA") && getenv("Q4_KV_SWMMA")[0] == '0');
    if (kvw && hd == 256 && n_head / n_kvh <= 16 && (kv_f16 == 0 || kv_f16 == 2) &&
        !((uintptr_t)d_kcache & 15u) && !((uintptr_t)d_vcache & 15u)) {
        const uint32_t Sw = 32u;
        const dim3 gw(n_kvh, Sw, 1);
        if (kv_f16 == 2)
            gqa_split_wmma_k<false, 2><<<gw, 256, 0, g_str>>>(
                d_q, (const uint8_t *)d_kcache, (const uint8_t *)d_vcache,
                g_attn_part, n_head, n_kvh, hd, 0, max_kv, scale, nullptr, 0,
                Sw, 0u, g_kv_d);
        else
            gqa_split_wmma_k<false, 0><<<gw, 256, 0, g_str>>>(
                d_q, (const uint8_t *)d_kcache, (const uint8_t *)d_vcache,
                g_attn_part, n_head, n_kvh, hd, 0, max_kv, scale, nullptr, 0,
                Sw, 0u, g_kv_d);
        gqa_dec_comb_k<<<n_head, 256, 0, g_str>>>(g_attn_part, d_gate, d_out,
                                                  n_head, n_kvh, hd, Sw);
        return hipGetLastError() == hipSuccess;
    }
    const uint32_t S = g_dec_split_s();
    dim3 g2(n_kvh, S);
#define DEC_SPLITK(K)                                                         \
    do {                                                                      \
        gqa_dec_split_k<K, false><<<g2, 256, 0, g_str>>>(                     \
            (const float *)d_q, (const uint8_t *)d_kcache,                    \
            (const uint8_t *)d_vcache, g_attn_part, n_head, n_kvh, hd, 0,     \
            max_kv, scale, nullptr, 0, S, g_kv_d);                            \
        gqa_dec_comb_k<<<n_head, 256, 0, g_str>>>(g_attn_part, d_gate, d_out, \
                                                n_head, n_kvh, hd, S);      \
    } while (0)
    if (kv_f16 == 2) DEC_SPLITK(2);
    else if (kv_f16 == 1) DEC_SPLITK(1);
    else DEC_SPLITK(0);
#undef DEC_SPLITK
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_qsa_decode_idx(const float *d_q, const void *d_kcache,
                                      const void *d_vcache, const float *d_gate,
                                      float *d_out, const int32_t *idx,
                                      uint32_t width, uint32_t n_head,
                                      uint32_t n_kvh, uint32_t hd, uint32_t n_tok,
                                      uint32_t max_kv, float scale, int kv_f16) {
    if (!g_ok || !d_q || !d_kcache || !d_vcache || !d_gate || !d_out || !idx)
        return false;
    if (n_tok == 0 || width == 0 || hd > 256 || hd == 0) return false;
    (void)hipGetLastError();
    if (qsa_small_split(d_q, d_kcache, d_vcache, d_gate, d_out, idx, width,
                        n_head, n_kvh, hd, n_tok, 0, max_kv, scale, kv_f16))
        return true;
    static int qsa_pf = -1;
    if (qsa_pf < 0)
        qsa_pf = !(getenv("Q4_QSA_PF") && getenv("Q4_QSA_PF")[0] == '0');
    int prefill = n_tok > 1 && n_head / n_kvh <= 12 && qsa_pf;
    if (!prefill && n_tok == 1 && g_attn_part && n_kvh <= 4 && hd <= 256 &&
        n_head / n_kvh <= GQA_DEC_MAX_GQA &&
        !(getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0')) {
        const uint32_t S = g_dec_split_s();
        dim3 g2(n_kvh, S);
#define DEC_SPLITI(K)                                                         \
        do {                                                                  \
            gqa_dec_split_k<K, true><<<g2, 256, 0, g_str>>>(                    \
                (const float *)d_q, (const uint8_t *)d_kcache,                \
                (const uint8_t *)d_vcache, g_attn_part, n_head, n_kvh, hd,    \
                0, max_kv, scale, idx, width, S, nullptr);                    \
            gqa_dec_comb_k<<<n_head, 256, 0, g_str>>>(g_attn_part, d_gate,    \
                                                    d_out, n_head, n_kvh,   \
                                                    hd, S);                 \
        } while (0)
        if (kv_f16 == 2) DEC_SPLITI(2);
        else if (kv_f16 == 1) DEC_SPLITI(1);
        else DEC_SPLITI(0);
#undef DEC_SPLITI
        if (hipGetLastError() != hipSuccess) return false;
        return true;
    }
    dim3 g(prefill ? n_kvh : n_head, n_tok);
    const char *qw = getenv("Q4_QSA_WMMA");
    const int wmma_ok = prefill && kv_f16 == 2 && hd == 256 &&
                      (!qw || qw[0] != '0');
    const char *qu = getenv("Q4_QSA_UNION");
    const int qsa_union = qu && qu[0] ? atoi(qu) : 4;
    if (wmma_ok && qsa_union && n_tok >= 16 &&
        qsa_union_launch(d_q, d_kcache, d_vcache, d_gate, d_out, idx, width,
                         n_head, n_kvh, n_tok, max_kv, scale,
                         (uint32_t)qsa_union))
        return true;
    if (kv_f16 == 2)
        (wmma_ok ? gqa_prefill_wmma_k<true>
                 : (prefill ? gqa_prefill_k<2, 1> : gqa_decode_fast<2, 1>))<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out,
                                          n_head, n_kvh, hd, 0, max_kv, scale, idx,
                                          width);
    else if (kv_f16 == 1)
        (prefill ? gqa_prefill_k<1, 1> : gqa_decode_fast<1, 1>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out,
                                          n_head, n_kvh, hd, 0, max_kv, scale, idx,
                                          width);
    else
        (prefill ? gqa_prefill_k<0, 1> : gqa_decode_fast<0, 1>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out,
                                          n_head, n_kvh, hd, 0, max_kv, scale, idx,
                                          width);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "q4 hip: qsa idx launch %s\n", hipGetErrorString(e));
        return false;
    }
    return true;
}

/* Mean of `ratio` raw fp16 keys, RMSNorm, RoPE at the block's first position. */
__global__ static void qsa_pool_k(const __half *raw, float *pool, const float *wn,
                                  uint32_t hd, uint32_t ratio, uint32_t b0,
                                  uint32_t n_rot, float rope_base, float eps) {
    uint32_t b = b0 + blockIdx.x;
    uint32_t d = threadIdx.x;
    if (hd != 128 || d >= hd) return;
    __shared__ float v[128];
    __shared__ float red[128];
    float s = 0.f;
    const __half *src = raw + (size_t)b * ratio * hd;
    for (uint32_t j = 0; j < ratio; j++)
        s += __half2float(src[(size_t)j * hd + d]);
    v[d] = s / (float)ratio;
    __syncthreads();
    red[d] = v[d] * v[d];
    __syncthreads();
    for (uint32_t st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)hd + eps);
    v[d] = v[d] * inv * (wn ? wn[d] : 1.f);
    __syncthreads();
    if (g_rope_neox ? d < n_rot / 2u : ((d & 1u) == 0 && d < n_rot)) {
        uint32_t p = g_rope_neox ? d : d >> 1;
        float freq = powf(rope_base, -(float)p / (float)(n_rot / 2u));
        float ang = (float)mrope_pos((int32_t)(b * ratio), p) * freq;
        float c = cosf(ang), si = sinf(ang);
        uint32_t i0, i1;
        rope_pair(p, n_rot / 2u, i0, i1);
        float x0 = v[i0], x1 = v[i1];
        v[i0] = x0 * c - x1 * si;
        v[i1] = x0 * si + x1 * c;
    }
    __syncthreads();
    pool[(size_t)b * hd + d] = v[d];
}

extern "C" bool q4_hip_qsa_pool(const void *raw_f16, float *pool, const float *wn,
                                uint32_t hd, uint32_t ratio, uint32_t b0,
                                uint32_t b1, uint32_t n_rot, float rope_base,
                                float eps) {
    if (!g_ok || !raw_f16 || !pool || b1 <= b0 || hd != 128 || ratio == 0)
        return false;
    qsa_pool_k<<<b1 - b0, 128, 0, g_str>>>((const __half *)raw_f16, pool, wn, hd, ratio, b0,
                                 n_rot, rope_base, eps);
    return hipGetLastError() == hipSuccess;
}

/* One query per block. Q stays in shared memory; each thread scores a stride of blocks. */
__global__ static void qsa_score_k(const float *q, const float *pool, float *score,
                                   uint32_t n_blocks, uint32_t stride, uint32_t n_head,
                                   uint32_t hd, uint32_t pos0, uint32_t ratio,
                                   const int32_t *pp) {
    uint32_t t = blockIdx.x;
    uint32_t pos = (pp ? (uint32_t)*pp : pos0) + t;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    if (pp) n_blocks = n_vis; /* device pos: bound derived on-device */
    uint32_t d = threadIdx.x;
    __shared__ float qh[4][128];
    if (d < hd && n_head <= 4) {
        for (uint32_t h = 0; h < n_head; h++)
            qh[h][d] = q[((size_t)t * n_head + h) * hd + d];
    }
    __syncthreads();
    for (uint32_t b = d; b < n_blocks; b += blockDim.x) {
        float s = -1.0e30f;
        if (b < n_vis) {
            s = 0.f;
            const float *kb = pool + (size_t)b * hd;
            for (uint32_t h = 0; h < n_head; h++) {
                float dot = 0.f;
                const float *qq = qh[h];
                for (uint32_t i = 0; i < hd; i++) dot += qq[i] * kb[i];
                s += fmaxf(dot, 0.f);
            }
        }
        score[(size_t)t * stride + b] = s;
    }
}

/* Small batches (2..8 rows): warp per pooled block, every row at once.
 * qsa_score_k gives each row ONE 128-thread block whose threads walk
 * their own pooled rows serially (uncoalesced): ~100 us per QSA layer at
 * a 2.4K context for a 4-row verify.  Here each lane holds 4 of the
 * block's 128 values (one coalesced float4) and the dot is a shuffle
 * reduction, so the sum order differs from qsa_score_k's serial loop. */
__global__ static void qsa_score_nw_k(const float *q, const float *pool,
                                      float *score, uint32_t n_tok,
                                      uint32_t n_blocks, uint32_t stride,
                                      uint32_t n_head, uint32_t pos0,
                                      uint32_t ratio) {
    __shared__ float qs[8 * 4 * 128];
    for (uint32_t i = threadIdx.x; i < n_tok * n_head * 128u; i += blockDim.x)
        qs[i] = q[i];
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b >= n_blocks) return;
    const float4 k = *(const float4 *)(pool + (size_t)b * 128u + lane * 4u);
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t n_vis = ratio ? (pos0 + t + 1) / ratio : 0;
        float sc = -1.0e30f;
        if (b < n_vis) {
            sc = 0.f;
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qq = qs + ((size_t)t * n_head + h) * 128u + lane * 4u;
                float d = qq[0] * k.x + qq[1] * k.y + qq[2] * k.z + qq[3] * k.w;
                for (uint32_t o = 16; o; o >>= 1) d += __shfl_xor(d, o);
                sc += fmaxf(d, 0.f);
            }
        }
        if (lane == 0) score[(size_t)t * stride + b] = sc;
    }
}

/* Prefill indexer score as a WMMA tile GEMM: score[t][b] = sum_h
 * relu(q_h(t) . pool(b)), -1e30 past the token's causal block count.
 * qsa_score_k gave each token one 128-thread block walking its pooled
 * rows serially in fp32 (~1.3 TFLOPS: 0.44 s of a 5.5K chunk at a 34K
 * context); here 32 tokens x 64 pooled blocks per workgroup, 8 waves of
 * 16x16 tiles, fp16 operands (the prompt attention's precision). */
__global__ static void qsa_score_wmma_k(const float *q, const float *pool,
                                        float *score, uint32_t n_tok,
                                        uint32_t n_blocks, uint32_t stride,
                                        uint32_t n_head, uint32_t pos0,
                                        uint32_t ratio) {
    constexpr uint32_t TT = 32, TB = 64, HD = 128, PADH = HD + 8;
    __shared__ __half qs[4][TT][PADH];
    __shared__ __half ks[TB][PADH];
    const uint32_t t0 = blockIdx.y * TT, b0 = blockIdx.x * TB;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, w = tid >> 5;
    /* the whole tile is past every row's causal bound: masked writes only */
    const uint32_t t_last = (t0 + TT < n_tok ? t0 + TT : n_tok) - 1u;
    const uint32_t vis_max = ratio ? (pos0 + t_last + 1u) / ratio : 0u;
    if (b0 >= vis_max) {
        for (uint32_t e = tid; e < TT * TB; e += blockDim.x) {
            const uint32_t t = t0 + e / TB, b = b0 + e % TB;
            if (t < n_tok && b < n_blocks)
                score[(size_t)t * stride + b] = -1.0e30f;
        }
        return;
    }
    for (uint32_t e = tid; e < 4u * TT * HD; e += blockDim.x) {
        const uint32_t h = e / (TT * HD), r = e % (TT * HD);
        const uint32_t t = r / HD, d = r % HD;
        float v = 0.f;
        if (h < n_head && t0 + t < n_tok)
            v = q[((size_t)(t0 + t) * n_head + h) * HD + d];
        qs[h][t][d] = __float2half(v);
    }
    for (uint32_t e = tid; e < TB * HD; e += blockDim.x) {
        const uint32_t b = e / HD, d = e % HD;
        ks[b][d] = __float2half(b0 + b < n_blocks
                                    ? pool[(size_t)(b0 + b) * HD + d] : 0.f);
    }
    __syncthreads();
    const uint32_t ts = w >> 2, bs = w & 3u;
    wmma_c8 tot = (wmma_c8){};
    for (uint32_t h = 0; h < n_head; h++) {
        wmma_c8 acc = (wmma_c8){};
#pragma unroll
        for (uint32_t k0 = 0; k0 < HD; k0 += 16) {
            wmma_a16 a16, b16;
#pragma unroll
            for (uint32_t u = 0; u < 16; u++) {
                a16[u] = qs[h][ts * 16 + (lane & 15u)][k0 + u];
                b16[u] = ks[bs * 16 + (lane & 15u)][k0 + u];
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a16, b16, acc);
        }
#pragma unroll
        for (uint32_t u = 0; u < 8; u++) tot[u] += fmaxf(acc[u], 0.f);
    }
#pragma unroll
    for (uint32_t u = 0; u < 8; u++) {
        const uint32_t t = t0 + ts * 16 + 2 * u + (lane >> 4);
        const uint32_t b = b0 + bs * 16 + (lane & 15u);
        if (t < n_tok && b < n_blocks) {
            const uint32_t n_vis = ratio ? (pos0 + t + 1u) / ratio : 0u;
            score[(size_t)t * stride + b] = b < n_vis ? tot[u] : -1.0e30f;
        }
    }
}

extern "C" bool q4_hip_qsa_score(const float *q, const float *pool, float *score,
                                 uint32_t n_tok, uint32_t n_blocks, uint32_t stride,
                                 uint32_t n_head, uint32_t hd, uint32_t pos0,
                                 uint32_t ratio) {
    if (!g_ok || !q || !pool || !score || n_tok == 0 || hd != 128 || n_head == 0 ||
        n_head > 4)
        return false;
    static int nw = -1;
    if (nw < 0) nw = !(getenv("Q4_QSA_SCORE_NW") && getenv("Q4_QSA_SCORE_NW")[0] == '0');
    if (nw && n_tok >= 2 && n_tok <= 8 && n_blocks && !((uintptr_t)pool & 15u)) {
        qsa_score_nw_k<<<(n_blocks + 7u) / 8u, 256, 0, g_str>>>(
            q, pool, score, n_tok, n_blocks, stride, n_head, pos0, ratio);
        return hipGetLastError() == hipSuccess;
    }
    static int sw = -1;
    if (sw < 0) sw = !(getenv("Q4_QSA_SCORE_WMMA") && getenv("Q4_QSA_SCORE_WMMA")[0] == '0');
    if (sw && n_tok >= 16 && n_blocks) {
        qsa_score_wmma_k<<<dim3((n_blocks + 63u) / 64u, (n_tok + 31u) / 32u), 256,
                           0, g_str>>>(q, pool, score, n_tok, n_blocks, stride,
                                       n_head, pos0, ratio);
        return hipGetLastError() == hipSuccess;
    }
    qsa_score_k<<<n_tok, 128, 0, g_str>>>(q, pool, score, n_blocks, stride, n_head, hd, pos0,
                                ratio, nullptr);
    return hipGetLastError() == hipSuccess;
}
/* Device-position single-token variant: n_blocks derived inside the kernel. */
/* Decode (device-pos, graph-captured) sparse indexer. Fixed grids only:
 * n_vis comes from *pp at replay time, so workgroup counts and scratch
 * sizes must not depend on it. */

__device__ static void qsa_write_sel(int32_t *sel, const int *chosen,
                                     int nchosen, uint32_t t, uint32_t pos,
                                     uint32_t n_vis, uint32_t ratio,
                                     uint32_t width);

#define QSA_SD_G 128   /* score workgroups */
#define QSA_SD_TILE 64 /* pooled blocks staged per workgroup pass (32 KiB) */
#define QSA_TK_G 64    /* topk workgroups (contiguous slices for ordering) */

__device__ static uint32_t qsa_tk_xkey(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ static void qsa_score_dk(const float *q, const float *pool,
                                    float *score, uint32_t n_head, uint32_t hd,
                                    uint32_t ratio, const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    __shared__ float qh[4][128];
    __shared__ float kb[QSA_SD_TILE][128];
    uint32_t d = threadIdx.x;
    if (d < hd)
        for (uint32_t h = 0; h < n_head; h++) qh[h][d] = q[h * hd + d];
    for (uint32_t t0 = blockIdx.x * QSA_SD_TILE; t0 < n_vis;
         t0 += gridDim.x * QSA_SD_TILE) {
        __syncthreads();
        uint32_t nt = n_vis - t0;
        if (nt > QSA_SD_TILE) nt = QSA_SD_TILE;
        for (uint32_t b2 = 0; b2 < nt; b2++)
            for (uint32_t i = d; i < hd; i += blockDim.x)
                kb[b2][i] = pool[((size_t)t0 + b2) * hd + i];
        __syncthreads();
        if (d < nt) {
            float s = 0.f;
            const float *k = kb[d];
            for (uint32_t h = 0; h < n_head; h++) {
                float dot = 0.f;
                const float *qq = qh[h];
                for (uint32_t i = 0; i < hd; i++) dot += qq[i] * k[i];
                s += fmaxf(dot, 0.f);
            }
            score[t0 + d] = s;
        }
    }
}

/* Exact top-need by (key desc, index asc): 4 radix passes over the 8 MSB
 * digits of the order-preserving uint key, then ordered compaction. st:
 * [0] prefix bits above the current digit, [1] (unused), [2] hi_cnt,
 * [3] threshold key, [4] n_gt total. */

__global__ static void qsa_tk_init_k(uint32_t *hist, uint32_t *st) {
    uint32_t d = threadIdx.x;
    if (d < 256) hist[d] = 0;
    if (d == 0) { st[0] = 0; st[2] = 0; }
}

__global__ static void qsa_tk_hist_k(const float *score, const uint32_t *st,
                                     uint32_t *hist, uint32_t ratio,
                                     uint32_t k_blk, const int32_t *pp,
                                     uint32_t pass) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    if (n_vis <= k_blk) return;
    __shared__ uint32_t h[256];
    uint32_t d = threadIdx.x;
    h[d] = 0;
    __syncthreads();
    uint32_t slice = (n_vis + QSA_TK_G - 1) / QSA_TK_G;
    uint32_t b0 = blockIdx.x * slice;
    uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
    uint32_t sh = 24 - pass * 8;
    uint32_t pre = st[0];
    for (uint32_t b = b0 + d; b < b1; b += blockDim.x) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (pass == 0 || (u >> (sh + 8)) == pre)
            atomicAdd(&h[(u >> sh) & 255], 1u);
    }
    __syncthreads();
    if (h[d]) atomicAdd(&hist[d], h[d]);
}

__global__ static void qsa_tk_pick_k(uint32_t *hist, uint32_t *st,
                                     uint32_t ratio, uint32_t k_blk,
                                     const int32_t *pp, uint32_t last) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    uint32_t d = threadIdx.x;
    if (n_vis <= k_blk) {
        if (d < 256) hist[d] = 0;
        return;
    }
    __shared__ int cnt[256];
    cnt[d] = (int)hist[d];
    __syncthreads();
    if (d == 0) {
        uint32_t target = need - st[2];
        int acc = 0;
        uint32_t dg = 0;
        for (int i = 255; i >= 0; i--) {
            if ((uint32_t)(acc + cnt[i]) >= target) {
                dg = (uint32_t)i;
                break;
            }
            acc += cnt[i];
        }
        st[2] += (uint32_t)acc;
        st[0] = (st[0] << 8) | dg;
        if (last) {
            st[3] = st[0];
            st[4] = st[2];
        }
    }
    __syncthreads();
    hist[d] = 0;
}

__global__ static void qsa_tk_cnt_k(const float *score, const uint32_t *st,
                                    uint32_t *cnt, uint32_t ratio,
                                    uint32_t k_blk, const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    if (n_vis <= k_blk) return;
    __shared__ uint32_t cg, ce;
    uint32_t d = threadIdx.x;
    if (d == 0) { cg = 0; ce = 0; }
    __syncthreads();
    uint32_t thr = st[3];
    uint32_t slice = (n_vis + QSA_TK_G - 1) / QSA_TK_G;
    uint32_t b0 = blockIdx.x * slice;
    uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
    for (uint32_t b = b0 + d; b < b1; b += blockDim.x) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (u > thr) atomicAdd(&cg, 1u);
        else if (u == thr) atomicAdd(&ce, 1u);
    }
    __syncthreads();
    if (d == 0) {
        cnt[2 * blockIdx.x] = cg;
        cnt[2 * blockIdx.x + 1] = ce;
    }
}

__global__ static void qsa_tk_scan_k(const uint32_t *cnt, uint32_t *off) {
    if (threadIdx.x == 0) {
        uint32_t ag = 0, ae = 0;
        for (uint32_t i = 0; i < QSA_TK_G; i++) {
            off[2 * i] = ag;
            off[2 * i + 1] = ae;
            ag += cnt[2 * i];
            ae += cnt[2 * i + 1];
        }
    }
}

__global__ static void qsa_tk_emit_k(const float *score, const uint32_t *st,
                                     const uint32_t *off, int32_t *chosen,
                                     uint32_t ratio, uint32_t k_blk,
                                     const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    if (n_vis <= k_blk) return;
    __shared__ int cg[256], ce[256];
    uint32_t d = threadIdx.x;
    uint32_t thr = st[3];
    uint32_t rem = need - st[4]; /* ==thr slots still free */
    uint32_t slice = (n_vis + QSA_TK_G - 1) / QSA_TK_G;
    uint32_t b0 = blockIdx.x * slice;
    uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
    uint32_t sub = (b1 - b0 + blockDim.x - 1) / blockDim.x;
    uint32_t c0 = b0 + d * sub;
    uint32_t c1 = c0 + sub < b1 ? c0 + sub : b1;
    int ng = 0, ne = 0;
    for (uint32_t b = c0; b < c1; b++) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (u > thr) ng++;
        else if (u == thr) ne++;
    }
    cg[d] = ng;
    ce[d] = ne;
    __syncthreads();
    if (d == 0) {
        int ag = 0, ae = 0;
        for (uint32_t i = 0; i < blockDim.x; i++) {
            int tg = cg[i], te = ce[i];
            cg[i] = ag; ce[i] = ae;
            ag += tg; ae += te;
        }
    }
    __syncthreads();
    uint32_t wg = off[2 * blockIdx.x] + (uint32_t)cg[d];
    uint32_t we = off[2 * blockIdx.x + 1] + (uint32_t)ce[d];
    for (uint32_t b = c0; b < c1; b++) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (u > thr)
            chosen[wg++] = (int32_t)b;
        else if (u == thr) {
            if (we < rem) chosen[st[4] + we] = (int32_t)b;
            we++;
        }
    }
}

__global__ static void qsa_tk_sel_k(const uint32_t *st, const int32_t *chosen,
                                    int32_t *sel, uint32_t ratio,
                                    uint32_t k_blk, uint32_t width,
                                    const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t d = threadIdx.x;
    if (n_vis <= k_blk) {
        for (uint32_t w = d; w < width; w += blockDim.x)
            sel[w] = w <= pos ? (int32_t)w : -1;
        return;
    }
    /* Parallel expansion of qsa_write_sel: entry w<need*ratio maps to
     * chosen[w/ratio]*ratio + w%ratio; the tail appends the newest partial
     * block positions n_vis*ratio..pos; the rest is -1 padding. */
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    uint32_t ncs = need * ratio;
    for (uint32_t w = d; w < width; w += blockDim.x) {
        int32_t v;
        if (w < ncs) {
            uint32_t i = w / ratio, j = w - i * ratio;
            v = (int32_t)((uint32_t)chosen[i] * ratio + j);
        } else {
            uint32_t tail = n_vis * ratio + (w - ncs);
            v = tail <= pos ? (int32_t)tail : -1;
        }
        sel[w] = v;
    }
}

extern "C" bool q4_hip_qsa_topk_alloc(void) {
    static size_t hc, sc, cc, oc, xc, pc, skc;
    return grow((void **)&g_tk_hist, &hc, 256 * 4) &&
           grow((void **)&g_tk_st, &sc, 8 * 4) &&
           grow((void **)&g_tk_cnt, &cc, 2 * QSA_TK_G * 4) &&
           grow((void **)&g_tk_off, &oc, 2 * QSA_TK_G * 4) &&
           grow((void **)&g_tk_chosen, &xc, 512 * 4) &&
           grow((void **)&g_attn_part, &pc,
                4 * GQA_DEC_SPLIT_S * GQA_DEC_MAX_GQA * (256 + 2) * 4) &&
           grow((void **)&g_sk_part, &skc, GEMV_SK_MAXROWS * GEMV_SK_S * 4);
}

extern "C" bool q4_hip_qsa_score_d(const float *q, const float *pool,
                                   float *score, uint32_t stride,
                                   uint32_t n_head, uint32_t hd,
                                   uint32_t ratio) {
    if (!g_ok || !g_pos_d || !q || !pool || !score || hd != 128 ||
        n_head == 0 || n_head > 4)
        return false;
    qsa_score_dk<<<QSA_SD_G, 128, 0, g_str>>>(q, pool, score, n_head, hd,
                                            ratio, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

__device__ static void qsa_write_sel(int32_t *sel, const int *chosen, int nchosen,
                                      uint32_t t, uint32_t pos, uint32_t n_vis,
                                      uint32_t ratio, uint32_t width) {
    uint32_t w = 0;
    for (int i = 0; i < nchosen && w < width; i++) {
        uint32_t b = (uint32_t)chosen[i];
        for (uint32_t j = 0; j < ratio && w < width; j++)
            sel[(size_t)t * width + w++] = (int32_t)(b * ratio + j);
    }
    uint32_t tail = n_vis * ratio;
    while (tail <= pos && w < width) sel[(size_t)t * width + w++] = (int32_t)tail++;
    while (w < width) sel[(size_t)t * width + w++] = -1;
}

/* Threshold search, then one pass that keeps the blocks at or above it. */
__global__ static void qsa_topk_k(const float *score, int32_t *sel, uint32_t stride,
                                  uint32_t k_blk, uint32_t ratio, uint32_t pos0,
                                  uint32_t width, const int32_t *pp) {
    uint32_t t = blockIdx.x;
    uint32_t tx = threadIdx.x;
    uint32_t pos = (pp ? (uint32_t)*pp : pos0) + t;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    const float *row = score + (size_t)t * stride;
    __shared__ float red[128];
    __shared__ int cred[128];
    __shared__ int chosen[512];
    __shared__ int nch;
    __shared__ float bound[2];

    if (need == n_vis) {
        /* Every complete block fits, so the selection is the dense prefix. */
        for (uint32_t w = tx; w < width; w += blockDim.x)
            sel[(size_t)t * width + w] = w <= pos ? (int32_t)w : -1;
        return;
    }

    /* Exact k-th key via 4 passes of 8-bit radix on the order-preserving
     * uint transform; then ordered compaction of u>thr ascending plus
     * u==thr ascending until `need` is full. */
    __shared__ uint32_t hist[256];
    __shared__ uint32_t st[2]; /* [0] prefix [1] hi_cnt */
    if (tx == 0) { st[0] = 0; st[1] = 0; }
    for (int p = 0; p < 4; p++) {
        for (uint32_t i = tx; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        uint32_t sh = 24 - p * 8;
        uint32_t pre = st[0];
        for (uint32_t b = tx; b < n_vis; b += blockDim.x) {
            uint32_t u = qsa_tk_xkey(row[b]);
            if (p == 0 || (u >> (sh + 8)) == pre)
                atomicAdd(&hist[(u >> sh) & 255], 1u);
        }
        __syncthreads();
        if (tx == 0) {
            uint32_t target = need - st[1];
            uint32_t acc = 0, dg = 0;
            for (int i = 255; i >= 0; i--) {
                if (acc + hist[i] >= target) { dg = (uint32_t)i; break; }
                acc += hist[i];
            }
            st[1] += acc;
            st[0] = (st[0] << 8) | dg;
        }
        __syncthreads();
    }
    uint32_t thr = st[0];
    int cgt = 0;
    for (uint32_t b = tx; b < n_vis; b += blockDim.x)
        if (qsa_tk_xkey(row[b]) > thr) cgt++;
    cred[tx] = cgt;
    __syncthreads();
    for (uint32_t stp = blockDim.x >> 1; stp > 0; stp >>= 1) {
        if (tx < stp) cred[tx] += cred[tx + stp];
        __syncthreads();
    }
    int n_gt = cred[0];
    if (tx == 0) nch = 0;
    __syncthreads();
    /* Exact threshold: n_gt < need and n_gt + n_eq >= need always. The
     * emit order matches the pre-radix kernel: blocks above the threshold
     * ascending, then threshold ties ascending. */
    {
        uint32_t slice = (n_vis + blockDim.x - 1) / blockDim.x;
        uint32_t b0 = tx * slice;
        uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
        int c = 0;
        for (uint32_t b = b0; b < b1; b++)
            if (qsa_tk_xkey(row[b]) > thr) c++;
        cred[tx] = c;
        __syncthreads();
        if (tx == 0) {
            int acc = 0;
            for (uint32_t i = 0; i < blockDim.x; i++) {
                int t = cred[i]; cred[i] = acc; acc += t;
            }
            nch = acc;
        }
        __syncthreads();
        int off = cred[tx];
        for (uint32_t b = b0; b < b1 && off < (int)k_blk; b++)
            if (qsa_tk_xkey(row[b]) > thr) chosen[off++] = (int)b;
        __syncthreads();
        if (tx == 0) {
            for (uint32_t b = 0; b < n_vis && nch < (int)need; b++)
                if (qsa_tk_xkey(row[b]) == thr) chosen[nch++] = (int)b;
            nch = (int)(nch < (int)need ? nch : need);
        }
        __syncthreads();
        {
            uint32_t ncs = (uint32_t)nch * ratio;
            for (uint32_t w = tx; w < width; w += blockDim.x) {
                int32_t v;
                if (w < ncs) {
                    uint32_t i = w / ratio, j = w - i * ratio;
                    v = (int32_t)((uint32_t)chosen[i] * ratio + j);
                } else {
                    uint32_t tail = n_vis * ratio + (w - ncs);
                    v = tail <= pos ? (int32_t)tail : -1;
                }
                sel[(size_t)t * width + w] = v;
            }
        }
    }
}

/* qsa_topk_k with every step parallel (Q4_QSA_TOPK_PAR, default on).  The
 * old kernel appended the threshold ties with one thread scanning the row
 * from block 0 — ties are usually a single block, so it walked about the
 * whole row serially: 43 us per row at a 2.5K context, 307 at 34K.  Same
 * selection, same order: the radix digit comes from a suffix scan of the
 * histogram, the count above the threshold is the radix's running count,
 * and one coalesced pass compacts "above" and "tie" blocks in ascending
 * order through ballot prefixes (wave32, 256 threads). */
#define QSA_TKP_T 256
__global__ static void __launch_bounds__(QSA_TKP_T)
qsa_topk_p_k(const float *score, int32_t *sel, uint32_t stride,
             uint32_t k_blk, uint32_t ratio, uint32_t pos0, uint32_t width) {
    const uint32_t t = blockIdx.x, tx = threadIdx.x;
    const uint32_t pos = pos0 + t;
    const uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    const uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    const float *row = score + (size_t)t * stride;
    __shared__ uint32_t hist[256], sfx[256];
    __shared__ uint32_t st[3]; /* prefix, count above, digit */
    __shared__ uint32_t wgt[QSA_TKP_T / 32], weq[QSA_TKP_T / 32];
    __shared__ int chosen[512];

    if (need == n_vis) {
        for (uint32_t w = tx; w < width; w += QSA_TKP_T)
            sel[(size_t)t * width + w] = w <= pos ? (int32_t)w : -1;
        return;
    }
    if (tx == 0) { st[0] = 0; st[1] = 0; }
    for (int p = 0; p < 4; p++) {
        hist[tx] = 0;
        if (tx == 0) st[2] = 0xffffffffu;
        __syncthreads();
        const uint32_t sh = 24 - p * 8;
        const uint32_t pre = st[0];
        for (uint32_t b = tx; b < n_vis; b += QSA_TKP_T) {
            uint32_t u = qsa_tk_xkey(row[b]);
            if (p == 0 || (u >> (sh + 8)) == pre)
                atomicAdd(&hist[(u >> sh) & 255], 1u);
        }
        __syncthreads();
        /* sfx[i] = sum of hist[j >= i] */
        sfx[tx] = hist[tx];
        __syncthreads();
        for (uint32_t o = 1; o < 256; o <<= 1) {
            uint32_t v = tx + o < 256 ? sfx[tx + o] : 0;
            __syncthreads();
            sfx[tx] += v;
            __syncthreads();
        }
        /* the serial walk's digit: the largest i with sfx[i] >= target */
        const uint32_t target = need - st[1];
        const uint32_t nx = tx + 1 < 256 ? sfx[tx + 1] : 0;
        if (sfx[tx] >= target && nx < target) st[2] = tx;
        __syncthreads();
        if (tx == 0) {
            uint32_t dg = st[2] == 0xffffffffu ? 0 : st[2];
            st[1] += st[2] == 0xffffffffu ? sfx[0] : (dg + 1 < 256 ? sfx[dg + 1] : 0);
            st[0] = (st[0] << 8) | dg;
        }
        __syncthreads();
    }
    const uint32_t thr = st[0];
    const uint32_t n_gt = st[1];
    const uint32_t w = tx >> 5, wl = tx & 31u;
    const unsigned long long lm = (1ull << wl) - 1ull;
    uint32_t ogt = 0, oeq = 0; /* running counts before this chunk */
    for (uint32_t c0 = 0; c0 < n_vis; c0 += QSA_TKP_T) {
        const uint32_t b = c0 + tx;
        const uint32_t u = b < n_vis ? qsa_tk_xkey(row[b]) : 0u;
        const bool gt = b < n_vis && u > thr;
        const bool eq = b < n_vis && u == thr;
        const unsigned long long bg = __ballot(gt), be = __ballot(eq);
        if (wl == 0) { wgt[w] = (uint32_t)__popcll(bg); weq[w] = (uint32_t)__popcll(be); }
        __syncthreads();
        uint32_t bgt = ogt, beq = oeq, tgt = 0, teq = 0;
        for (uint32_t k = 0; k < QSA_TKP_T / 32; k++) {
            if (k < w) { bgt += wgt[k]; beq += weq[k]; }
            tgt += wgt[k]; teq += weq[k];
        }
        if (gt) {
            uint32_t r = bgt + (uint32_t)__popcll(bg & lm);
            if (r < k_blk) chosen[r] = (int)b;
        }
        if (eq) {
            uint32_t r = n_gt + beq + (uint32_t)__popcll(be & lm);
            if (r < need) chosen[r] = (int)b;
        }
        ogt += tgt; oeq += teq;
        __syncthreads();
        if (n_gt + oeq >= need && ogt >= n_gt) break;
    }
    __syncthreads();
    const uint32_t nch = need;
    const uint32_t ncs = nch * ratio;
    for (uint32_t x = tx; x < width; x += QSA_TKP_T) {
        int32_t v;
        if (x < ncs) {
            uint32_t i = x / ratio, j = x - i * ratio;
            v = (int32_t)((uint32_t)chosen[i] * ratio + j);
        } else {
            uint32_t tail = n_vis * ratio + (x - ncs);
            v = tail <= pos ? (int32_t)tail : -1;
        }
        sel[(size_t)t * width + x] = v;
    }
}

static int g_qsa_topk_par = -1;
static bool qsa_topk_par(void) {
    if (g_qsa_topk_par < 0) {
        const char *e = getenv("Q4_QSA_TOPK_PAR");
        g_qsa_topk_par = !(e && e[0] == '0');
    }
    return g_qsa_topk_par;
}
/* q4-test: force the kernel (0 old, 1 parallel, -1 back to the env) */
extern "C" void q4_qsa_topk_par_set(int v) { g_qsa_topk_par = v; }

extern "C" bool q4_hip_qsa_topk(const float *score, int32_t *sel, uint32_t n_tok,
                                uint32_t n_blocks, uint32_t stride, uint32_t k_blk,
                                uint32_t ratio, uint32_t pos0, uint32_t width) {
    (void)n_blocks;
    if (!g_ok || !score || !sel || n_tok == 0 || k_blk == 0 || k_blk > 512 ||
        width == 0)
        return false;
    if (qsa_topk_par())
        qsa_topk_p_k<<<n_tok, QSA_TKP_T, 0, g_str>>>(score, sel, stride, k_blk,
                                                    ratio, pos0, width);
    else
        qsa_topk_k<<<n_tok, 128, 0, g_str>>>(score, sel, stride, k_blk, ratio,
                                             pos0, width, nullptr);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "q4 hip: qsa topk %s\n", hipGetErrorString(e));
        return false;
    }
    return true;
}
extern "C" bool q4_hip_qsa_topk_d(const float *score, int32_t *sel,
                                  uint32_t stride, uint32_t k_blk,
                                  uint32_t ratio, uint32_t width) {
    if (!g_ok || !g_pos_d || !score || !sel || k_blk == 0 || k_blk > 512 ||
        width == 0)
        return false;
    if (!g_tk_hist)
        return false;
    qsa_tk_init_k<<<1, 256, 0, g_str>>>(g_tk_hist, g_tk_st);
    for (uint32_t p = 0; p < 4; p++) {
        qsa_tk_hist_k<<<QSA_TK_G, 256, 0, g_str>>>(score, g_tk_st, g_tk_hist,
                                                 ratio, k_blk, g_pos_d, p);
        qsa_tk_pick_k<<<1, 256, 0, g_str>>>(g_tk_hist, g_tk_st, ratio, k_blk,
                                            g_pos_d, p == 3);
    }
    qsa_tk_cnt_k<<<QSA_TK_G, 256, 0, g_str>>>(score, g_tk_st, g_tk_cnt, ratio,
                                            k_blk, g_pos_d);
    qsa_tk_scan_k<<<1, 128, 0, g_str>>>(g_tk_cnt, g_tk_off);
    qsa_tk_emit_k<<<QSA_TK_G, 256, 0, g_str>>>(score, g_tk_st, g_tk_off,
                                             g_tk_chosen, ratio, k_blk,
                                             g_pos_d);
    qsa_tk_sel_k<<<1, 128, 0, g_str>>>(g_tk_st, g_tk_chosen, sel, ratio,
                                       k_blk, width, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

/* Pool one indexer block with bounds computed from the device position.
 * Always launched in graphs; self-gates when no block completed. */
__global__ static void qsa_pool_dk(const __half *raw, float *pool,
                                   const float *wn, uint32_t hd,
                                   uint32_t ratio, uint32_t n_rot,
                                   float rope_base, float eps,
                                   const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t b0 = pos / ratio;
    uint32_t b1 = (pos + 1) / ratio;
    if (b1 <= b0 || hd != 128 || threadIdx.x >= hd) return;
    uint32_t b = b0;
    uint32_t d = threadIdx.x;
    __shared__ float v[128];
    __shared__ float red[128];
    float s = 0.f;
    const __half *src = raw + (size_t)b * ratio * hd;
    for (uint32_t j = 0; j < ratio; j++)
        s += __half2float(src[(size_t)j * hd + d]);
    v[d] = s / (float)ratio;
    __syncthreads();
    red[d] = v[d] * v[d];
    __syncthreads();
    for (uint32_t st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)hd + eps);
    v[d] = v[d] * inv * (wn ? wn[d] : 1.f);
    __syncthreads();
    if (g_rope_neox ? d < n_rot / 2u : ((d & 1u) == 0 && d < n_rot)) {
        uint32_t p = g_rope_neox ? d : d >> 1;
        float freq = powf(rope_base, -(float)p / (float)(n_rot / 2u));
        float ang = (float)mrope_pos((int32_t)(b * ratio), p) * freq;
        float c = cosf(ang), si = sinf(ang);
        uint32_t i0, i1;
        rope_pair(p, n_rot / 2u, i0, i1);
        float x0 = v[i0], x1 = v[i1];
        v[i0] = x0 * c - x1 * si;
        v[i1] = x0 * si + x1 * c;
    }
    __syncthreads();
    pool[(size_t)b * hd + d] = v[d];
}
extern "C" bool q4_hip_qsa_pool_d(const void *raw_f16, float *pool,
                                  const float *wn, uint32_t hd, uint32_t ratio,
                                  uint32_t n_rot, float rope_base, float eps) {
    if (!g_ok || !g_pos_d || !raw_f16 || !pool || hd != 128 || ratio == 0)
        return false;
    qsa_pool_dk<<<1, 128, 0, g_str>>>((const __half *)raw_f16, pool, wn, hd,
                                      ratio, n_rot, rope_base, eps, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

/* Store raw indexer key at the device position's slot. */
__global__ static void f32_to_f16_at_k(const float *src, __half *dst0,
                                       uint32_t stride, uint64_t n,
                                       const int32_t *pp) {
    __half *dst = dst0 + (size_t)*pp * stride;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
extern "C" bool q4_hip_f32_to_f16_d(void *dst_base, uint32_t stride_elems,
                                    const float *src, uint64_t n) {
    if (!g_ok || !g_pos_d || !dst_base || !src || !n) return false;
    uint32_t block = 256;
    uint32_t grid = (uint32_t)((n + block - 1) / block);
    f32_to_f16_at_k<<<grid, block, 0, g_str>>>(src, (__half *)dst_base,
                                             stride_elems, n, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

/* Append K and V (n floats each) into the caches at the device position.
 * kind: 0 f32, 1 f16, 2 q8-per-32 — mirrors kv_write on the host path. */
__global__ static void kv_append_k(const float *k, const float *v, uint8_t *kc,
                                   uint8_t *vc, uint32_t n, uint32_t bpt,
                                   int kind, const int32_t *pp) {
    size_t off = (size_t)(uint32_t)*pp * bpt;
    uint8_t *dk = kc + off, *dv = vc + off;
    if (kind == 2) {
        uint32_t nb = n / 32u;
        uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= 2 * nb) return;
        uint32_t i = b < nb ? b : b - nb;
        const float *x = (b < nb ? k : v) + (size_t)i * 32u;
        uint8_t *o = (b < nb ? dk : dv) + (size_t)i * 34u;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        float d = amax / 127.f;
        float id = d > 0.f ? 1.f / d : 0.f;
        __half h = __float2half(d);
        memcpy(o, &h, 2);
        for (int j = 0; j < 32; j++) {
            int q = (int)rintf(x[j] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            o[2 + j] = (uint8_t)(int8_t)q;
        }
    } else {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= 2 * n) return;
        const float *x = i < n ? k : v;
        uint32_t j = i < n ? i : i - n;
        uint8_t *o = (i < n ? dk : dv) + (size_t)j * (kind == 1 ? 2u : 4u);
        if (kind == 1) {
            __half h = __float2half(x[j]);
            memcpy(o, &h, 2);
        } else {
            memcpy(o, &x[j], 4);
        }
    }
}
extern "C" bool q4_hip_kv_append_d(void *kcache, void *vcache, const float *k,
                                   const float *v, uint32_t n, uint32_t bpt,
                                   int kind) {
    if (!g_ok || !g_pos_d || !kcache || !vcache || !k || !v || !n || !bpt)
        return false;
    uint32_t unit = kind == 2 ? n / 32u : n;
    if (kind == 2 && (n % 32u)) return false;
    uint32_t grid = (2 * unit + 255) / 256;
    kv_append_k<<<grid, 256, 0, g_str>>>(k, v, (uint8_t *)kcache,
                                         (uint8_t *)vcache, n, bpt, kind,
                                         g_pos_d);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_kv_append_kv(void *kcache, void *vcache, const float *k,
                                    const float *v, uint32_t n, uint32_t bpt,
                                    int kind) {
    if (!g_ok || !g_kv_d || !kcache || !vcache || !k || !v || !n || !bpt)
        return false;
    uint32_t unit = kind == 2 ? n / 32u : n;
    if (kind == 2 && (n % 32u)) return false;
    uint32_t grid = (2 * unit + 255) / 256;
    kv_append_k<<<grid, 256, 0, g_str>>>(k, v, (uint8_t *)kcache,
                                         (uint8_t *)vcache, n, bpt, kind,
                                         g_kv_d);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_qsa_decode(const float *d_q, const void *d_kcache,
                                  const void *d_vcache, const float *d_gate,
                                  float *d_out, uint32_t n_head, uint32_t n_kvh,
                                  uint32_t hd, uint32_t n_kv, uint32_t max_kv,
                                  float scale, int kv_f16) {
    if (n_kv == 0) return false;
    return q4_hip_qsa_decode_n(d_q, d_kcache, d_vcache, d_gate, d_out, n_head,
                               n_kvh, hd, 1, n_kv - 1, max_kv, scale, kv_f16);
}

/* PLE batched over T tokens (prefill). Per-(token,stream) gate:
 * dot = sum(kn_c*qn_c)/sqrt(n_embd); gate = sigmoid(copysign(sqrt(max(|dot|,
 * 1e-6)), dot)); kn[t,c,i] <- val[t,i] * gate (kn is dead after the dot). */
__global__ static void ple_gate_k(float *kn, const float *qn, const float *val,
                                  uint32_t n_embd, uint32_t hc) {
    uint32_t c = blockIdx.x, t = blockIdx.y;
    float *k = kn + ((size_t)t * hc + c) * n_embd;
    const float *qv = qn + ((size_t)t * hc + c) * n_embd;
    float d = 0.f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        d += k[i] * qv[i];
    __shared__ float sh[256], sg;
    sh[threadIdx.x] = d;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float dot = sh[0] * (1.0f / sqrtf((float)n_embd));
        float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
        sg = 1.f / (1.f + expf(-copysignf(mag, dot)));
    }
    __syncthreads();
    float gate = sg;
    const float *v = val + (size_t)t * n_embd;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        k[i] = v[i] * gate;
}

extern "C" bool q4_hip_ple_gate(float *kn, const float *qn, const float *val,
                                uint32_t n_embd, uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !kn || !qn || !val || !n_embd || !hc || !n_tok) return false;
    ple_gate_k<<<dim3(hc, n_tok), 256, 0, g_str>>>(kn, qn, val, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

/* Causal dilated depthwise conv over the batch: token t taps conv-norm rows
 * t - (kern-1-k)*dil, reading the (hist+1)-row state for offsets < 0.
 * co[t,c] = silu(sum_k kw[k + c*kern] * src[t-(kern-1-k)*dil, c]). */
__global__ static void ple_conv_k(float *co, const float *cn,
                                  const float *state, const float *kw,
                                  uint32_t hc_dim, uint32_t kern, uint32_t dil,
                                  uint32_t hist) {
    uint32_t t = blockIdx.x;
    uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= hc_dim) return;
    float acc = 0.f;
    for (uint32_t k = 0; k < kern; k++) {
        int64_t o = (int64_t)t - (int64_t)(kern - 1 - k) * (int64_t)dil;
        const float *src = o >= 0 ? cn + (size_t)o * hc_dim
                                  : state + ((size_t)o + hist + 1) * hc_dim;
        acc += kw[k + (size_t)c * kern] * src[c];
    }
    co[(size_t)t * hc_dim + c] = acc / (1.f + expf(-acc));
}

extern "C" bool q4_hip_ple_conv(float *co, const float *cn, const float *state,
                                const float *kw, uint32_t hc_dim, uint32_t kern,
                                uint32_t dil, uint32_t hist, uint32_t n_tok) {
    if (!g_ok || !co || !cn || !state || !kw || !hc_dim || !kern || !dil ||
        !n_tok)
        return false;
    dim3 g(n_tok, (hc_dim + 255) / 256);
    ple_conv_k<<<g, 256, 0, g_str>>>(co, cn, state, kw, hc_dim, kern, dil, hist);
    return hipGetLastError() == hipSuccess;
}

/* New conv state = last H conv-norm rows; positions before the batch come
 * from the tail of the old state. dst is the ping-pong buffer (no overlap). */
__global__ static void ple_state_k(float *dst, const float *old,
                                   const float *cn, uint32_t T, uint32_t hc_dim,
                                   uint32_t H) {
    uint64_t e = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (uint64_t)H * hc_dim) return;
    uint32_t r = (uint32_t)(e / hc_dim), c = (uint32_t)(e % hc_dim);
    int64_t idx = (int64_t)T - (int64_t)H + (int64_t)r;
    dst[e] = idx >= 0 ? cn[(size_t)idx * hc_dim + c]
                      : old[((size_t)T + r) * hc_dim + c];
}

extern "C" bool q4_hip_ple_state(float *dst, const float *old, const float *cn,
                                 uint32_t n_tok, uint32_t hc_dim, uint32_t H) {
    if (!g_ok || !dst || !old || !cn || !hc_dim || !H) return false;
    uint64_t n = (uint64_t)H * hc_dim;
    ple_state_k<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(dst, old, cn,
                                                              n_tok, hc_dim, H);
    return hipGetLastError() == hipSuccess;
}

/* Deterministic per-token reduce of expert outputs:
 * dy[tok] += sum_k rw[tok*topk+k] * pd[rsrc[tok*topk+k]]  (rsrc < 0 skipped). */
__global__ static void reduce_expert_k(float *dy, const float *pd,
                                       const int32_t *rsrc, const float *rw,
                                       uint32_t topk, uint32_t n_embd) {
    uint32_t tok = blockIdx.y;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0.f;
    for (uint32_t k = 0; k < topk; k++) {
        int32_t src = rsrc[tok * topk + k];
        if (src >= 0) acc += rw[tok * topk + k] * pd[(size_t)src * n_embd + i];
    }
    dy[(size_t)tok * n_embd + i] += acc;
}

extern "C" bool q4_hip_expert_reduce(float *d_y, const float *d_pd,
                                     const int32_t *d_rsrc, const float *d_rw,
                                     uint32_t n_tok, uint32_t topk,
                                     uint32_t n_embd) {
    if (!g_ok || !d_y || !d_pd || !d_rsrc || !d_rw || !n_tok || !topk) return false;
    dim3 b(256), g((n_embd + 255u) / 256u, n_tok);
    reduce_expert_k<<<g, b, 0, g_str>>>(d_y, d_pd, d_rsrc, d_rw, topk, n_embd);
    return hipGetLastError() == hipSuccess;
}

/* dy[t*n+i] += sg[t] * dx[t*n+i] */
__global__ static void axpy_rows_k(float *dy, const float *dx, const float *sg,
                                   uint32_t n) {
    uint32_t t = blockIdx.y;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dy[(size_t)t * n + i] += sg[t] * dx[(size_t)t * n + i];
}

/* Shared-expert gate + add in one launch (Q4_SGATE_FUSE, default on):
 * every block recomputes its token's sigmoid(w . x[t]) (n <= ~16K floats,
 * L2-resident) and adds its slice of sg * sh.  Replaces a one-row GEMV,
 * a sigmoid and the axpy (three launches, ~12 us per verify layer). */
__global__ static void sgate_axpy_k(float *dy, const float *dsh,
                                    const float *w, const float *x,
                                    uint32_t n) {
    __shared__ float red[8];
    const uint32_t t = blockIdx.y;
    const float *xt = x + (size_t)t * n;
    float a = 0.f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) a += w[i] * xt[i];
    for (int o = 16; o; o >>= 1) a += __shfl_xor(a, o, 32);
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    float d = 0.f;
    for (uint32_t k = 0; k < (blockDim.x >> 5); k++) d += red[k];
    const float sg = 1.f / (1.f + expf(-d));
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dy[(size_t)t * n + i] += sg * dsh[(size_t)t * n + i];
}

extern "C" bool q4_hip_sgate_axpy(float *d_y, const float *d_sh,
                                  const float *d_w, const float *d_x,
                                  uint32_t n, uint32_t n_tok) {
    if (!g_ok || !d_y || !d_sh || !d_w || !d_x || !n || !n_tok) return false;
    dim3 b(256), g((n + 255u) / 256u, n_tok);
    sgate_axpy_k<<<g, b, 0, g_str>>>(d_y, d_sh, d_w, d_x, n);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_axpy_rows(float *d_y, const float *d_x, const float *d_sg,
                                 uint32_t n, uint32_t n_tok) {
    if (!g_ok || !d_y || !d_x || !d_sg || !n || !n_tok) return false;
    dim3 b(256), g((n + 255u) / 256u, n_tok);
    axpy_rows_k<<<g, b, 0, g_str>>>(d_y, d_x, d_sg, n);
    return hipGetLastError() == hipSuccess;
}

/* ---- decode-graph support -------------------------------------------------
 * Stream capture on g_str. The decode layer body is a fixed launch chain per
 * layer; capturing it once and replaying removes ~40 launch submissions per
 * layer (GPU was ~10% busy: submission-bound).
 * -------------------------------------------------------------------------*/

extern "C" int q4_hip_graph_begin(void) {
    if (!g_ok || !g_str || g_capturing) return 0;
    /* ThreadLocal: io threads keep enqueuing async H2D on g_copy while we
     * capture g_str; Global mode would abort the capture on their calls. */
    hipError_t e =
        hipStreamBeginCapture(g_str, hipStreamCaptureModeThreadLocal);
    if (e != hipSuccess) {
        static int once = 0;
        if (!once++) fprintf(stderr, "q4: graph begin: %s\n",
                             hipGetErrorString(e));
        return 0;
    }
    g_capturing = 1;
    return 1;
}

extern "C" void *q4_hip_graph_end(void) {
    g_capturing = 0;
    hipGraph_t g = nullptr;
    hipError_t e = hipStreamEndCapture(g_str, &g);
    if (e != hipSuccess || !g) {
        static int once = 0;
        if (!once++) fprintf(stderr, "q4: graph end: %s\n",
                             hipGetErrorString(e));
        /* A capture-invalidating call leaves a sticky last-error on this
         * thread; drain it or the first hipGetLastError check in the
         * eager re-run reports a phantom failure. Worse, on ROCm a failed
         * EndCapture can leave g_str permanently in capture state, making
         * every later launch fail — recreate the stream when that happens. */
        (void)hipGetLastError();
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(g_str, &cs) == hipSuccess &&
            cs != hipStreamCaptureStatusNone) {
            hipStream_t ns = nullptr;
            (void)hipStreamDestroy(g_str);
            if (hipStreamCreateWithFlags(&ns, hipStreamNonBlocking) ==
                hipSuccess) {
                g_str = ns;
            } else {
                g_str = nullptr;
                (void)hipGetLastError();
            }
            static int once2 = 0;
            if (!once2++)
                fprintf(stderr, "q4: graph end: stream stuck capturing — "
                                "recreated (%s)\n", g_str ? "ok" : "failed");
            (void)hipGetLastError(); /* the query itself set last-error */
        }
        return nullptr;
    }
    size_t nn = 0;
    (void)hipGraphGetNodes(g, nullptr, &nn);
    g_last_graph_nodes = (uint32_t)nn;
    hipGraphExec_t ex = nullptr;
    e = hipGraphInstantiate(&ex, g, nullptr, nullptr, 0);
    (void)hipGraphDestroy(g);
    if (e != hipSuccess) {
        static int once = 0;
        if (!once++) fprintf(stderr, "q4: graph instantiate: %s\n",
                             hipGetErrorString(e));
        return nullptr;
    }
    /* Pre-upload node schedules to the device so replay skips most of the
     * per-launch host work. */
    (void)hipGraphUpload(ex, g_str);
    return (void *)ex;
}

extern "C" uint32_t q4_hip_graph_nodes(void *exec) {
    (void)exec;
    return g_last_graph_nodes;
}

extern "C" bool q4_hip_graph_launch(void *exec) {
    if (!exec) return false;
    return hipGraphLaunch((hipGraphExec_t)exec, g_str) == hipSuccess;
}

extern "C" void q4_hip_graph_free(void *exec) {
    if (exec) (void)hipGraphExecDestroy((hipGraphExec_t)exec);
}

/* Pinned-memory sync between the compute stream and the host/CPU expert
 * pool.  Both sides use monotonically increasing counters so nothing is
 * baked into launch arguments: graph replays keep working untouched. */
__global__ static void flag_bump_k(volatile uint32_t *p) { ++*p; }

/* Stream-ordered "routing data is in host memory" marker: enqueued right
 * after the router/x D2H, so the host may consume them while later kernels
 * (shared expert) still run.  Host spins on the flag, one slot per layer. */
extern "C" bool q4_hip_flag_bump(void *flag_d) {
    if (!g_ok || !flag_d) return false;
    flag_bump_k<<<1, 1, 0, g_str>>>((volatile uint32_t *)flag_d);
    return hipGetLastError() == hipSuccess;
}

/* One graph node replacing fill(d_y) + dot_sigmoid + flag_bump in the MoE
 * decode path: zero the routed-merge accumulator, compute the shared-gate
 * scalar sigmoid(sgate . x), then bump the per-layer "routing is in host
 * memory" flag.  Runs right after the router/x D2H copies on the same
 * stream, so the flag still marks copy completion.  d_sgate/flag may be
 * NULL (no sgate tensor / no flag scheme). */
__global__ static void moe_pack_k(float *d_y, const float *d_x,
                                  const float *d_sgate, float *d_dot,
                                  volatile uint32_t *flag, uint32_t n_embd,
                                  int8_t *q8, float *d8,
                                  float *sm, const float *sm2,
                                  uint32_t n_sm) {
    __shared__ float red[8];
    const uint32_t i = threadIdx.x;
    float acc = 0.f;
    for (uint32_t j = i; j < n_embd; j += 256u) {
        d_y[j] = 0.f;
        if (d_sgate) acc += d_x[j] * d_sgate[j];
    }
    /* Shared-expert silu*mul rides the same launch when the gate/up
     * projections already ran (fused_sh). */
    for (uint32_t j = i; j < n_sm; j += 256u) {
        const float v = sm[j];
        sm[j] = v / (1.f + expf(-v)) * sm2[j];
    }
    if (q8) {
        /* quantize d_x into slice 0 — identical math to xq8_k, 8 warps
         * covering n_embd/32 groups.  Lets moe_rows_gu skip its separate
         * xq8 launch. */
        const uint32_t gps = n_embd >> 5;
        const uint32_t warp = i >> 5, lane = i & 31u;
        for (uint32_t g = warp; g < gps; g += 8u) {
            const float *xs = d_x + (size_t)g * 32u;
            const float v = xs[lane];
            float amax = fabsf(v);
            for (int o = 16; o; o >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffull, amax, o));
            q8[(size_t)g * 32 + lane] =
                (int8_t)__float2int_rn(v * (amax ? 127.f / amax : 0.f));
            if (!lane) d8[g] = amax * (1.f / 127.f);
        }
    }
    if (d_sgate) {
        for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o);
        if ((i & 31u) == 0) red[i >> 5] = acc;
        __syncthreads();
        if (i < 8) {
            acc = red[i];
            for (int o = 4; o; o >>= 1) acc += __shfl_down(acc, o);
            if (i == 0) *d_dot = 1.f / (1.f + expf(-acc));
        }
    }
    __syncthreads();
    if (i == 0 && flag) ++*flag;
}
extern "C" bool q4_hip_moe_pack(float *d_y, const float *d_x,
                                const float *d_sgate, float *d_dot,
                                void *flag_d, uint32_t n_embd, bool qx,
                                float *sm, const float *sm2, uint32_t n_sm) {
    if (!g_ok || !d_y || !d_x || !n_embd) return false;
    int8_t *q8 = NULL;
    float *d8 = NULL;
    if (qx && !(n_embd & 31u) && (n_embd >> 5) <= XQ8_GPS_MAX &&
        xq8_bufs()) {
        q8 = g_xq8;
        d8 = g_xq8d;
        g_xq8_src = d_x;
    }
    moe_pack_k<<<1, 256, 0, g_str>>>(d_y, d_x, d_sgate, d_dot,
                                     (volatile uint32_t *)flag_d, n_embd,
                                     q8, d8, sm, sm2, n_sm);
    return hipGetLastError() == hipSuccess;
}

/* Fused MoE tail for decode: y += hy; y += sh * sig(dot); then the
 * hyper-connection residual combine — one node for three.  dot already
 * holds the sigmoid scalar (moe_pack / dot_sigmoid); NULL means no shared
 * gate (plain add). */
__global__ static void moe_final_k(float *res, const float *y, const float *hy,
                                   const float *sh, const float *dot,
                                   const float *inj, uint32_t n_embd,
                                   uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float v = y[i] + hy[i] + sh[i] * (dot ? *dot : 1.f);
    float inv = 1.f / (float)hc;
    for (uint32_t c = 0; c < hc; c++) {
        float w = 2.f / (1.f + expf(-inj[c] * inv));
        res[(size_t)c * n_embd + i] += v * w;
    }
}
extern "C" bool q4_hip_moe_final(float *res, const float *y, const float *hy,
                                 const float *sh, const float *dot,
                                 const float *inj, uint32_t n_embd,
                                 uint32_t hc) {
    if (!g_ok || !res || !y || !hy || !sh || !inj || !n_embd || !hc)
        return false;
    moe_final_k<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(
        res, y, hy, sh, dot, inj, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

/* The CPU-miss wait runs as a HOST callback node inside the stream (the
 * verify path now uses cpx_spin_k below, on coherent memory): a GPU
 * kernel cannot spin on host-registered memory — device reads of mapped
 * sysmem are L2-cached and host writes do not snoop GPU caches (measured:
 * a polling kernel saw done=46 for 45+ s after the host wrote 48, while a
 * just-launched read of expect=48 was fresh).  hipLaunchHostFunc inside a
 * captured graph replays fine, the callback reads pinned host memory with
 * full coherence, and stream order is preserved: downstream merge kernels
 * do not run until it returns. */

typedef struct {
    volatile unsigned long long *done;
    volatile unsigned long long *expect;
    volatile unsigned long long *progress; /* pool-wide monotone floor */
} cpx_wait_ctx;

#define CPX_WAITQ 1024
static cpx_wait_ctx g_cpxq[CPX_WAITQ];
static uint32_t g_cpxq_i;
/* Capture-owned ctxs: inside a graph the host-node arg is BAKED IN and
 * re-read at every replay, so it cannot come from the recycled eager
 * ring — ~1024 eager registrations later a live graph's ctx would alias
 * an unrelated wait (wrong done/expect pair).  Captured ctxs are
 * malloc'd, chained here, and freed at shutdown; graphs never outlive
 * the context.  Bounded: a handful of captures x ~48 layers x 48 B. */
typedef struct cpx_dyn_s {
    cpx_wait_ctx w;
    struct cpx_dyn_s *next;
} cpx_wait_dyn;
static cpx_wait_dyn *g_cpx_dyn;
static void cpx_dyn_free_all(void) {
    while (g_cpx_dyn) {
        cpx_wait_dyn *n = g_cpx_dyn->next;
        free(g_cpx_dyn);
        g_cpx_dyn = n;
    }
}
/* Host-side prof: sum µs, calls, timeouts, (last_seen<<20|expect) */
static unsigned long long g_cpx_us, g_cpx_n, g_cpx_to, g_cpx_ls;

static void cpx_wait_hostfn(void *arg) {
    cpx_wait_ctx *w = (cpx_wait_ctx *)arg;
    struct timespec ts0, tprog, ts;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    tprog = ts0;
    unsigned long long hb = w->progress ? *w->progress : 0;
    unsigned long long last = 0;
    for (;;) {
        /* Re-read expect each pass: the eager path can raise it after this
         * node is queued, and the callback must track the latest target. */
        const unsigned long long e =
            __atomic_load_n(w->expect, __ATOMIC_ACQUIRE);
        last = __atomic_load_n(w->done, __ATOMIC_ACQUIRE);
        if (last >= e) break;
        if (w->progress) {
            unsigned long long now =
                __atomic_load_n(w->progress, __ATOMIC_ACQUIRE);
            if (now != hb) { hb = now; clock_gettime(CLOCK_MONOTONIC, &tprog); }
        }
        clock_gettime(CLOCK_MONOTONIC, &ts);
        /* ~45 s of pool-wide silence = wedged pipeline; still under the
         * host watchdog's 180 s stall trip. */
        if (ts.tv_sec - tprog.tv_sec > 45) {
            __atomic_add_fetch(&g_cpx_to, 1, __ATOMIC_RELAXED);
            __atomic_store_n(&g_cpx_ls, (last << 20) | (e & 0xfffffull),
                             __ATOMIC_RELAXED);
            break;
        }
        _mm_pause();
    }
    clock_gettime(CLOCK_MONOTONIC, &ts);
    __atomic_add_fetch(&g_cpx_n, 1, __ATOMIC_RELAXED);
    /* tv_nsec can be < ts0's (borrowed into tv_sec): normalize before the
     * unsigned cast or a negative delta wraps to ~1.8e19 µs. */
    long long dns = (long long)(ts.tv_nsec - ts0.tv_nsec);
    long long dus = (long long)(ts.tv_sec - ts0.tv_sec) * 1000000ll +
                    dns / 1000ll;
    if (dus > 0)
        __atomic_add_fetch(&g_cpx_us, (unsigned long long)dus,
                           __ATOMIC_RELAXED);
}

extern "C" bool q4_hip_cpx_wait(void *done_h, void *expect_h,
                                void *progress_h) {
    if (!g_ok || !done_h || !expect_h) return false;
    cpx_wait_ctx *w;
    if (g_capturing) {
        cpx_wait_dyn *d = (cpx_wait_dyn *)malloc(sizeof(*d));
        if (!d) return false;
        d->next = g_cpx_dyn;
        g_cpx_dyn = d;
        w = &d->w;
    } else {
        uint32_t i = __atomic_fetch_add(&g_cpxq_i, 1, __ATOMIC_RELAXED) &
                     (CPX_WAITQ - 1);
        w = &g_cpxq[i];
    }
    w->done = (volatile unsigned long long *)done_h;
    w->expect = (volatile unsigned long long *)expect_h;
    w->progress = (volatile unsigned long long *)progress_h;
    return hipLaunchHostFunc(g_str, cpx_wait_hostfn, w) == hipSuccess;
}

/* Q4_VTL: verify-timeline stamps.  A 1-thread kernel writes the GPU wall
 * clock into a device slot; the host reads the slots after the pass and
 * maps them onto CLOCK_MONOTONIC with q4_hip_vtl_calib's offset.
 * Diagnostic only — each stamp is one more launch. */
#define VTL_SLOTS 1024u
static unsigned long long *g_vtl_d;
static double g_vtl_ns_per_tick;
__global__ void vtl_stamp_k(unsigned long long *p) { *p = wall_clock64(); }
__global__ void vtl_cal_k(volatile unsigned long long *hp) {
    *hp = wall_clock64();
    __threadfence_system();
}
extern "C" bool q4_hip_vtl_init(void) {
    if (!g_ok) return false;
    if (g_vtl_d) return true;
    int khz = 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0) !=
            hipSuccess || khz <= 0)
        return false;
    g_vtl_ns_per_tick = 1e6 / (double)khz;
    if (hipMalloc((void **)&g_vtl_d, VTL_SLOTS * 8) != hipSuccess) {
        g_vtl_d = nullptr;
        return false;
    }
    return hipMemset(g_vtl_d, 0, VTL_SLOTS * 8) == hipSuccess;
}
extern "C" bool q4_hip_vtl_stamp(uint32_t slot) {
    if (!g_vtl_d || slot >= VTL_SLOTS) return false;
    vtl_stamp_k<<<1, 1, 0, g_str>>>(g_vtl_d + slot);
    return hipGetLastError() == hipSuccess;
}
/* Sync read of n slots, converted to host CLOCK_MONOTONIC seconds (0 for
 * a slot never written). */
extern "C" bool q4_hip_vtl_read(double *out, uint32_t n, double off_ns) {
    if (!g_vtl_d || n > VTL_SLOTS) return false;
    unsigned long long h[VTL_SLOTS];
    if (hipStreamSynchronize(g_str) != hipSuccess ||
        hipMemcpy(h, g_vtl_d, n * 8, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    for (uint32_t i = 0; i < n; i++)
        out[i] = h[i] ? ((double)h[i] * g_vtl_ns_per_tick + off_ns) * 1e-9
                      : 0.0;
    return true;
}
/* host_ns - gpu_ns, min over a few PCIe round trips (the host sees the
 * GPU's write ~1 us late, so the offset errs late by about that). */
extern "C" double q4_hip_vtl_calib(void) {
    if (!g_vtl_d) return 0.0;
    unsigned long long *hp = nullptr;
    if (hipHostMalloc((void **)&hp, 64,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess)
        return 0.0;
    unsigned long long *dp = nullptr;
    (void)hipHostGetDevicePointer((void **)&dp, hp, 0);
    double best = 1e30;
    for (int r = 0; r < 8; r++) {
        __atomic_store_n(hp, 0ull, __ATOMIC_RELEASE);
        hipStreamSynchronize(g_str);
        vtl_cal_k<<<1, 1, 0, g_str>>>(dp);
        unsigned long long v;
        while (!(v = __atomic_load_n((volatile unsigned long long *)hp,
                                     __ATOMIC_ACQUIRE)))
            _mm_pause();
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double hn = (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
        double d = hn - (double)v * g_vtl_ns_per_tick;
        if (d < best) best = d;
    }
    hipStreamSynchronize(g_str);
    hipHostFree(hp);
    return best;
}

/* Q4_CPX_SPIN (default on): the verify MoE waits for its CPU-miss job in a
 * one-thread GPU kernel instead of the host callback above.  The kernel
 * polls a done word in hipHostMallocCoherent memory — uncached on the GPU,
 * unlike the host-registered mailbox the comment above measured — against
 * the layer's device seq counter (written by the route kernel, so a graph
 * replay compares with the current publish).  The CPU rows the merge reads
 * afterwards must live in coherent memory too.  seambench: the callback
 * costs ~39 us per seam with nothing to wait for, the spin ~12. */
static unsigned long long *g_spin_to; /* coherent: timeouts */
__global__ static void cpx_spin_k(const unsigned long long *want_d,
                                  const unsigned long long *done_d,
                                  unsigned long long *to) {
    if (threadIdx.x) return;
    const unsigned long long want = *want_d;
    const unsigned long long t0 = __builtin_amdgcn_s_sendmsg_rtnl(131);
    for (;;) {
        const unsigned long long d = __hip_atomic_load(
            done_d, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (d >= want) break;
        /* 100 MHz: give up after 60 s (the host watchdog trips at 180).
         * The 64-bit read matters: with the 32-bit s_sendmsg_rtn the
         * counter wrapped every ~43 s, a spin across the wrap saw a huge
         * difference and "timed out" at once (~7e-6 per wait: the merge
         * read unfinished CPU rows and the request aborted as wedged —
         * 3 of 4 long 4-row ppl runs). */
        if (__builtin_amdgcn_s_sendmsg_rtnl(131) - t0 > 6000000000ull) {
            __hip_atomic_fetch_add(to, 1ull, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
            break;
        }
        __builtin_amdgcn_s_sleep(2);
    }
}

extern "C" void *q4_hip_host_alloc_coh(size_t n) {
    void *p = nullptr;
    if (!g_ok || !n) return nullptr;
    if (hipHostMalloc(&p, n, hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    return p;
}
extern "C" void q4_hip_host_free_coh(void *p) {
    if (p) (void)hipHostFree(p);
}

/* Allocates the timeout counter; call outside any stream capture. */
extern "C" bool q4_hip_cpx_spin_init(void) {
    if (!g_spin_to) {
        g_spin_to = (unsigned long long *)q4_hip_host_alloc_coh(64);
        if (g_spin_to) *g_spin_to = 0;
    }
    return g_spin_to != nullptr;
}

extern "C" bool q4_hip_cpx_spin(const unsigned long long *want_d,
                                const unsigned long long *done_d) {
    if (!g_ok || !g_spin_to || !want_d || !done_d) return false;
    cpx_spin_k<<<1, 32, 0, g_str>>>(want_d, done_d, g_spin_to);
    return hipGetLastError() == hipSuccess;
}

/* Avg wait µs/call (resets cycle/count on read) or -1; calls via *calls. */
extern "C" double q4_hip_cpx_wait_prof(unsigned long long *calls) {
    unsigned long long us = __atomic_exchange_n(&g_cpx_us, 0,
                                                __ATOMIC_RELAXED);
    unsigned long long n = __atomic_exchange_n(&g_cpx_n, 0,
                                               __ATOMIC_RELAXED);
    if (calls) *calls = n;
    if (!n) return -1.0;
    return (double)us / (double)n;
}

/* Cumulative count of cpx wait timeouts (survives profiling reads). */
extern "C" unsigned long long q4_hip_cpx_wait_timeouts(void) {
    return __atomic_load_n(&g_cpx_to, __ATOMIC_RELAXED) +
           (g_spin_to ? __atomic_load_n(g_spin_to, __ATOMIC_RELAXED) : 0ull);
}

/* Last timeout's diagnostic: (seen done)<<20 | (expect & 2^20-1). */
extern "C" unsigned long long q4_hip_cpx_wait_lastseen(void) {
    return __atomic_load_n(&g_cpx_ls, __ATOMIC_RELAXED);
}

/* Session teardown: clear the cumulative timeout state so a re-inited
 * devroute doesn't inherit a poisoned counter. */
extern "C" void q4_hip_cpx_wait_reset(void) {
    __atomic_store_n(&g_cpx_to, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_cpx_ls, 0, __ATOMIC_RELAXED);
    if (g_spin_to) __atomic_store_n(g_spin_to, 0, __ATOMIC_RELAXED);
}

/* Copies on the compute stream: inside capture they become memcpy nodes;
 * outside capture they behave like any async enqueue (caller must sync). */
extern "C" bool q4_hip_d2h_async(void *h, const void *d, uint64_t n) {
    if (!g_ok || !h || !d || !n) return false;
    return hipMemcpyAsync(h, d, (size_t)n, hipMemcpyDeviceToHost, g_str) ==
           hipSuccess;
}
extern "C" bool q4_hip_h2d_async(void *d, const void *h, uint64_t n) {
    if (!g_ok || !d || !h || !n) return false;
    return hipMemcpyAsync(d, h, (size_t)n, hipMemcpyHostToDevice, g_str) ==
           hipSuccess;
}
extern "C" bool q4_hip_stream_sync(void) {
    if (!g_ok) return false;
    return (g_str ? hipStreamSynchronize(g_str) : hipDeviceSynchronize()) ==
           hipSuccess;
}

/* GPU-side timing: events recorded on g_str bracket a phase; q4_hip_mark_ms
 * drains slot and returns elapsed ms (GPU busy + in-stream idle). Slots:
 * 0 = whole decode window, 1 = graph A, 2 = graph B, 3 = head graph,
 * 4 = eager attn, 5 = eager moe-a, 6 = eager moe-b. */
/* Boundary marker: records an event on g_str. Safe inside graph capture (it
 * becomes an event-record node timestamped at replay). No-op unless
 * Q4_STEP_PROF=1 so call sites need no env plumbing. */
static int mk_env(void) {
    static int en = -1;
    if (en < 0) {
        const char *e = getenv("Q4_STEP_PROF");
        en = e && e[0] == '1';
        /* Pre-create all marker events now: hipEventCreate during stream
         * capture would invalidate the capture. */
        if (en && g_ok && g_str)
            for (int i = 0; i < Q4_MK_SLOTS; i++) {
                (void)hipEventCreate(&g_mk0[i]);
                (void)hipEventCreate(&g_mk1[i]);
            }
    }
    return en;
}
extern "C" void q4_hip_mark_pt(int slot) {
    if (!mk_env() || !g_ok || !g_str || slot < 0 || slot >= Q4_MK_SLOTS)
        return;
    if (!g_mk0[slot]) return;
    hipError_t rc = hipEventRecord(g_mk0[slot], g_str);
    static int dbg = 0;
    if (slot >= 12 && dbg < 8) {
        dbg++;
        fprintf(stderr, "q4: mark_pt slot %d rc=%d cap=%d\n", slot, (int)rc,
                g_capturing);
    }
    if (rc == hipSuccess)
        g_mk_armed[slot] = 2; /* 2 = boundary recorded */
}
/* Elapsed ms between two boundary events; syncs sb first (caller must be
 * outside capture). */
extern "C" double q4_hip_mark_elapsed(int sa, int sb) {
    if (sa < 0 || sb < 0 || sa >= Q4_MK_SLOTS || sb >= Q4_MK_SLOTS) return -1.0;
    if (!g_mk0[sa] || !g_mk0[sb]) return -1.0;
    /* Never-recorded events would make hipEventElapsedTime fail and leave a
     * sticky error that poisons subsequent hipGetLastError checks. */
    if (g_mk_armed[sa] != 2 || g_mk_armed[sb] != 2) {
        static int d0 = 0;
        if (sb >= 12 && d0++ < 6)
            fprintf(stderr, "q4: mark_elapsed %d->%d armed %d/%d\n",
                    sa, sb, g_mk_armed[sa], g_mk_armed[sb]);
        return -1.0;
    }
    hipError_t sr = hipEventSynchronize(g_mk0[sb]);
    if (sr != hipSuccess) {
        static int d1 = 0;
        if (sb >= 12 && d1++ < 6)
            fprintf(stderr, "q4: mark_elapsed %d->%d sync rc=%d\n",
                    sa, sb, (int)sr);
        (void)hipGetLastError(); /* keep profiler errors off the work path */
        return -1.0;
    }
    float ms = 0.f;
    hipError_t er = hipEventElapsedTime(&ms, g_mk0[sa], g_mk0[sb]);
    if (er != hipSuccess) {
        static int d2 = 0;
        if (sb >= 12 && d2++ < 6)
            fprintf(stderr, "q4: mark_elapsed %d->%d elapsed rc=%d\n",
                    sa, sb, (int)er);
        (void)hipGetLastError();
        return -1.0;
    }
    return (double)ms;
}
extern "C" void q4_hip_mark_begin_slot(int slot) {
    if (!g_ok || !g_str || g_capturing || slot < 0 || slot >= Q4_MK_SLOTS)
        return;
    if (!g_mk0[slot]) {
        if (hipEventCreate(&g_mk0[slot]) != hipSuccess ||
            hipEventCreate(&g_mk1[slot]) != hipSuccess) {
            g_mk0[slot] = g_mk1[slot] = nullptr;
            return;
        }
    }
    if (hipEventRecord(g_mk0[slot], g_str) == hipSuccess)
        g_mk_armed[slot] = 1;
    else
        (void)hipGetLastError(); /* profiler must not poison the work path */
}
extern "C" double q4_hip_mark_ms_slot(int slot) {
    if (!g_mk0[slot] || !g_str || g_mk_armed[slot] != 1) return -1.0;
    g_mk_armed[slot] = 0;
    float ms = 0.f;
    if (hipEventRecord(g_mk1[slot], g_str) != hipSuccess ||
        hipEventSynchronize(g_mk1[slot]) != hipSuccess ||
        hipEventElapsedTime(&ms, g_mk0[slot], g_mk1[slot]) != hipSuccess) {
        (void)hipGetLastError();
        return -1.0;
    }
    return (double)ms;
}
extern "C" void q4_hip_mark_begin(void) { q4_hip_mark_begin_slot(0); }
extern "C" double q4_hip_mark_ms(void) { return q4_hip_mark_ms_slot(0); }

__global__ static void axpy_da_k(float *y, const float *x, const float *a,
                                 uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[0] * x[i];
}

/* y += (*d_alpha) * x  — alpha lives on device so a graph can be replayed
 * without patching kernel args. */
extern "C" bool q4_hip_axpy_dalpha(float *y, const float *x,
                                   const float *d_alpha, uint64_t n) {
    if (!g_ok || !y || !x || !d_alpha || !n) return false;
    dim3 b(256), g((uint32_t)((n + 255) / 256));
    axpy_da_k<<<g, b, 0, g_str>>>(y, x, d_alpha, n);
    return hipGetLastError() == hipSuccess;
}
