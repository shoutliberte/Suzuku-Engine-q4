#include "q4.h"

#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double q4_prof_moe_io;
double q4_prof_moe_d2h;
double q4_prof_moe_gemm;
double q4_prof_moe_cw;
double q4_prof_moe_xd2h;   /* d_x D2H for CPU experts */
double q4_prof_moe_topk;   /* router topk D2H */
double q4_prof_moe_sg;     /* sgate dot + D2H */
double q4_prof_moe_join;   /* cpuex join */
double q4_prof_moe_merg;   /* h_y H2D + axpy */
double q4_prof_moe_ng;     /* routed uses executed on GPU L1 (decode) */
double q4_prof_moe_nc;     /* routed uses dispatched to the CPU pool */
double q4_prof_moe_ncall;  /* hostpart calls (decode layers x tokens) */
double q4_prof_moe_rt;     /* prefill: router gemv launch */
double q4_prof_moe_pf;     /* prefill: pf_join + sticky swap */
double q4_prof_moe_sh;     /* prefill: shared-expert tail */

static double moe_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Q4_VTL=1: verify timeline.  GPU stamps per layer (slot L*8+k): 0 layer
 * start, 1 MoE entry, 2 router done, 3 publish copied, 4 hits done,
 * 5 CPU wait released, 6 merge done, 7 layer end.  Host stamps from the
 * dispatcher: seen, dispatched, job finished, done stamped. */
static int moe_vtl = -1;
static double moe_vtl_h[Q4_MAX_LAYER][4];
static int moe_vtl_on(void) {
    if (moe_vtl < 0) {
        const char *e = getenv("Q4_VTL");
        moe_vtl = (e && e[0] == '1') ? 1 : 0;
        if (moe_vtl && !q4_hip_vtl_init()) moe_vtl = 0;
    }
    return moe_vtl;
}
void q4_moe_vtl_stamp(int layer, int k) {
    if (moe_vtl_on() && layer >= 0 && layer < 128)
        (void)q4_hip_vtl_stamp((uint32_t)layer * 8u + (uint32_t)k);
}
static int moe_cpx_repeat, moe_stage_late;
static double moe_rep_us1, moe_rep_us2;
static unsigned long long moe_rep_n;
#define VTL_H(L, k) do { if (moe_vtl > 0) moe_vtl_h[L][k] = moe_now(); } while (0)

typedef struct {
    int32_t layer;
    uint8_t *router;    /* F32 [n_embd, n_expert] */
    uint8_t *sgate;     /* F32 [n_embd] */
    uint8_t *gate_s;
    uint8_t *up_s;
    uint8_t *down_s;
    const q4_tensor *t_router, *t_sgate, *t_gate_s, *t_up_s, *t_down_s;
} q4_moe_dense;

static q4_moe_dense g_dense = { .layer = -1 };

static void dense_free(q4_moe_dense *d) {
    free(d->router);
    free(d->sgate);
    free(d->gate_s);
    free(d->up_s);
    free(d->down_s);
    memset(d, 0, sizeof(*d));
    d->layer = -1;
}

static uint8_t *load_t(const q4_gguf *g, const q4_tensor *t) {
    if (!t) return NULL;
    uint8_t *p = malloc((size_t)t->nbytes);
    if (!p) return NULL;
    if (!q4_tensor_read(g, t, p, t->nbytes)) {
        free(p);
        return NULL;
    }
    return p;
}

static bool dense_load(const q4_gguf *g, int32_t layer) {
    if (g_dense.layer == layer) return true;
    dense_free(&g_dense);
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp.weight", layer);
    g_dense.t_router = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    g_dense.t_sgate = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_shexp.weight", layer);
    g_dense.t_gate_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_up_shexp.weight", layer);
    g_dense.t_up_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_down_shexp.weight", layer);
    g_dense.t_down_s = q4_find_tensor(g, n);
    if (!g_dense.t_router || !g_dense.t_gate_s || !g_dense.t_up_s ||
        !g_dense.t_down_s)
        return false;
    g_dense.router = load_t(g, g_dense.t_router);
    g_dense.gate_s = load_t(g, g_dense.t_gate_s);
    g_dense.up_s = load_t(g, g_dense.t_up_s);
    g_dense.down_s = load_t(g, g_dense.t_down_s);
    if (g_dense.t_sgate) g_dense.sgate = load_t(g, g_dense.t_sgate);
    if (!g_dense.router || !g_dense.gate_s || !g_dense.up_s || !g_dense.down_s)
        return false;
    g_dense.layer = layer;
    return true;
}

static bool gemv_t(const q4_tensor *t, const uint8_t *w, const uint8_t *d_w,
                   const float *x, float *y) {
    if (!t || t->n_dims < 2) return false;
    uint64_t ncols = t->ne[0];
    uint64_t nrows = t->ne[1];
    if (q4_hip_ok()) {
        if (d_w) return q4_hip_gemv_dev(t->ggml_type, d_w, nrows, ncols, x, y, 1.f);
        if (w) return q4_hip_gemv(t->ggml_type, w, nrows, ncols, x, y, 1.f);
    }
    if (!w) return false;
    memset(y, 0, (size_t)nrows * sizeof(float));
    return q4_gemv(t->ggml_type, w, nrows, ncols, x, y, 1.f);
}

/* sgate (ffn_gate_inp_shexp) is F32 in some builds and BF16 in others;
 * every consumer wants f32.  The device variant dequantizes the tensor
 * from the gguf and uploads once, cached by tensor pointer. */
static const float *moe_sgate_f32d(const q4_gguf *g, const q4_store *st,
                                   const q4_tensor *t) {
    static const q4_tensor *sk[64];
    static float *sv[64];
    static int sn;
    if (!t || !g || !st) return NULL;
    if (t->ggml_type == Q4_T_F32)
        return (const float *)q4_store_dev(st, t);
    for (int i = 0; i < sn; i++)
        if (sk[i] == t) return sv[i];
    if (sn >= 64) return NULL;
    const uint64_t n = t->ne[0];
    uint8_t *raw = (uint8_t *)malloc((size_t)t->nbytes);
    float *h = (float *)malloc(n * sizeof(float));
    float *d = NULL;
    if (raw && h && q4_tensor_read(g, t, raw, t->nbytes) &&
        q4_dequant_row(t->ggml_type, raw, n, h)) {
        d = q4_hip_malloc(n * sizeof(float));
        if (!d || !q4_hip_h2d(d, h, n * sizeof(float))) d = NULL;
    }
    free(raw);
    free(h);
    if (!d) return NULL;
    sk[sn] = t;
    sv[sn] = d;
    sn++;
    return d;
}

/* Host-side counterpart for the CPU moe_body path. */
static const float *moe_sgate_f32h(const uint8_t *raw, const q4_tensor *t) {
    static const q4_tensor *sk[64];
    static float *sv[64];
    static int sn;
    if (!t || !raw) return NULL;
    if (t->ggml_type == Q4_T_F32) return (const float *)raw;
    for (int i = 0; i < sn; i++)
        if (sk[i] == t) return sv[i];
    if (sn >= 64) return NULL;
    float *h = (float *)malloc(t->ne[0] * sizeof(float));
    if (!h || !q4_dequant_row(t->ggml_type, raw, t->ne[0], h)) {
        free(h);
        return NULL;
    }
    sk[sn] = t;
    sv[sn] = h;
    sn++;
    return h;
}

static bool moe_body(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     const float *x, float *y, const uint8_t *router,
                     const q4_tensor *t_router, const uint8_t *d_router,
                     const uint8_t *sgate, const q4_tensor *t_sgate,
                     const uint8_t *gate_s, const q4_tensor *t_gate_s,
                     const uint8_t *d_gate_s, const uint8_t *up_s,
                     const q4_tensor *t_up_s, const uint8_t *d_up_s,
                     const uint8_t *down_s, const q4_tensor *t_down_s,
                     const uint8_t *d_down_s) {
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        return false;

    float *logits = malloc((size_t)n_exp * sizeof(float));
    float *h = malloc((size_t)n_ff * sizeof(float));
    float *h2 = malloc((size_t)n_ff * sizeof(float));
    float *down_buf = malloc((size_t)n_embd * sizeof(float));
    float *shared = malloc((size_t)n_embd * sizeof(float));
    if (!logits || !h || !h2 || !down_buf || !shared) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    if (!gemv_t(t_router, router, d_router, x, logits)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    q4_softmax(logits, n_exp);
    int32_t ids[64];
    float wts[64];
    if (topk > 64) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    q4_topk(logits, n_exp, topk, ids, wts, true);
    q4_expert_route_note(c, layer, ids, (int)topk);

    memset(y, 0, (size_t)n_embd * sizeof(float));
    for (uint32_t k = 0; k < topk; k++) {
        /* Partial-resident tail experts are not in the image: touch reports
         * -1, so bounce them straight off the GGUF instead of failing. */
        uint8_t *pk = NULL;
        int64_t nread = q4_expert_cache_touch(c, layer, ids[k]);
        if (nread < 0) {
            pk = malloc((size_t)tot);
            if (!pk || q4_expert_pread(c, layer, ids[k], pk) < 0) {
                free(pk);
                free(logits); free(h); free(h2); free(down_buf); free(shared);
                return false;
            }
            nread = (int64_t)tot;
        }
        const uint8_t *pack = pk ? pk : q4_expert_cache_data(c, layer, ids[k]);
        int32_t slot = q4_expert_cache_slot(c, layer, ids[k]);
        const uint8_t *dpack = NULL;
        if (pack && slot >= 0 && !pk && q4_hip_ok() && q4_hip_experts_dev(0)) {
            if (nread > 0) q4_hip_experts_put((uint32_t)slot, pack, (uint64_t)nread);
            dpack = q4_hip_experts_dev((uint32_t)slot);
        }
        if (!pack) {
            /* Static-L1 experts hold no image copy — the bytes only exist
             * on the GPU, so this host eval reads them from the GGUF. */
            pk = malloc((size_t)tot);
            if (!pk || q4_expert_pread(c, layer, ids[k], pk) < 0) {
                free(pk);
                free(logits); free(h); free(h2); free(down_buf);
                free(shared);
                return false;
            }
            pack = pk;
        }
        if (!gemv_t(gate, pack + og, dpack ? dpack + og : NULL, x, h)) {
            free(pk);
            free(logits); free(h); free(h2); free(down_buf); free(shared);
            return false;
        }
        q4_silu(h, n_ff);
        if (!gemv_t(up, pack + ou, dpack ? dpack + ou : NULL, x, h2)) {
            free(pk);
            free(logits); free(h); free(h2); free(down_buf); free(shared);
            return false;
        }
        for (uint32_t i = 0; i < n_ff; i++) h[i] *= h2[i];
        if (!gemv_t(down, pack + od, dpack ? dpack + od : NULL, h, down_buf)) {
            free(pk);
            free(logits); free(h); free(h2); free(down_buf); free(shared);
            return false;
        }
        const float w = wts[k];
        for (uint32_t i = 0; i < n_embd; i++) y[i] += w * down_buf[i];
        free(pk);
    }

    if (!gemv_t(t_gate_s, gate_s, d_gate_s, x, h)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    q4_silu(h, n_ff);
    if (!gemv_t(t_up_s, up_s, d_up_s, x, h2)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    for (uint32_t i = 0; i < n_ff; i++) h[i] *= h2[i];
    if (!gemv_t(t_down_s, down_s, d_down_s, h, shared)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    float sg = 1.0f;
    if (sgate && t_sgate && t_sgate->ne[0] == n_embd) {
        const float *gw = moe_sgate_f32h(sgate, t_sgate);
        if (gw) {
            float acc = 0;
            for (uint32_t i = 0; i < n_embd; i++) acc += gw[i] * x[i];
            sg = 1.0f / (1.0f + expf(-acc));
        }
    }
    for (uint32_t i = 0; i < n_embd; i++) y[i] += sg * shared[i];
    free(logits); free(h); free(h2); free(down_buf); free(shared);
    return true;
}

bool q4_moe_layer(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                  const float *x, float *y) {
    if (!g || !c || !x || !y || layer < 0) return false;
    if (!g->n_embd || !g->n_expert || !g->n_expert_used) return false;
    if (!dense_load(g, layer)) return false;
    return moe_body(g, c, layer, x, y, g_dense.router, g_dense.t_router, NULL,
                    g_dense.sgate, g_dense.t_sgate, g_dense.gate_s,
                    g_dense.t_gate_s, NULL, g_dense.up_s, g_dense.t_up_s, NULL,
                    g_dense.down_s, g_dense.t_down_s, NULL);
}

/* ---- CPU-expert decode path, split so the two GPU halves can be captured
 * into hipGraphs (decode is launch-bound: GPU ~10% busy).
 *   A:  router gemv + shared expert + sgate dot + async D2H of logits/x
 *   host: stream sync, softmax+topk, CPU dispatch, CPU join
 *   B:  expert-merge axpy + sgate axpy (+ hc_combine by the caller)
 * The pinned host buffers are shared file state; one session at a time. */

static float *moe_hx, *moe_hy, *moe_hh, *moe_hrlog;
static float *moe_dhy;                 /* device alias of moe_hy */
static uint32_t moe_hcap;
/* Pinned sync block: bytes [0,256) hold per-layer "routing ready" bump
 * flags written by a GPU kernel after the D2H; bytes 256/264 hold the
 * expect/done seq words published by the CPU expert pool for the in-stream
 * wait kernel; bytes 272/280 hold the devroute want/done job counters
 * (want is bumped by route_dev_k on the GPU, done by the dispatcher
 * thread after join).  All counters are monotonic → graph-replay safe. */
#define MOE_SYNC_BYTES 288
static volatile uint32_t *moe_route_h;   /* u32[64] */
static uint32_t *moe_route_d;
static volatile unsigned long long *moe_cpx_h; /* {expect, done} */
static unsigned long long *moe_cpx_d;
static volatile unsigned long long *moe_dwant_h; /* devroute {want, done} */
static unsigned long long *moe_dwant_d;
static uint32_t moe_route_last[64];
static uint64_t moe_eager_tag;  /* dispatch tag of the in-flight eager job */
static int32_t moe_ids[64];
static float moe_wts[64];
static float *moe_d_sh, *moe_d_dot, *moe_d_down, *moe_dy;
static float *moe_dx, *moe_dh, *moe_dh2;
static const uint8_t *moe_d_sgate;
static const q4_tensor *moe_t_sgate;
static uint32_t moe_nembd;
static int moe_want_cpu = -1;

/* ---- device-resident routing ------------------------------------------
 * The whole MoE layer — router gemv, top-k, hit/miss split, hit-expert
 * exec, CPU-miss wait and the merged tail — is graphable.  route_dev_k
 * publishes CPU misses through a pinned per-layer mailbox; a dispatcher
 * thread feeds them to the CPU pool and bumps the done counter the in-graph
 * wait kernel spins on.  Decoding never leaves the stream. */
/* Layout: the prefix up to and including seq matches q4_mail_t in
 * q4_rocm.cu (the device staging slot the route kernel fills).  The
 * publish is an in-stream D2H copy of that prefix — seq sits LAST so the
 * ascending-address DMA lands it after the payload and this host's seq
 * check keeps its "publish complete" meaning.  done/err are host-only:
 * they live past the copied prefix and are never overwritten by a
 * publish. */
typedef struct {
    int32_t ids[64];
    float wts[64];
    uint32_t n;                        /* miss count */
    /* Batched publish (verify): one entry carries every token's miss
     * segment.  ntok==0 marks a legacy single-token publish (ids[0..n));
     * ntok>1 segments ids/wts by tn[t]. */
    volatile uint32_t ntok;
    volatile uint32_t tn[8];
    volatile int32_t hs[8 * 16];       /* batched: token t's L1 hit slots */
    volatile uint32_t hn[8];
    volatile uint8_t rk[64];           /* batched: each miss's top-k rank */
    volatile int32_t routed[16];       /* full top-k for the hit histogram */
    volatile unsigned long long rseq;  /* profile stamp (fire-and-forget) */
    volatile unsigned long long seq;   /* bump per job — DMA tail byte */
    volatile unsigned long long done;  /* dispatcher's completion stamp */
    volatile uint32_t err;             /* overwrite/error flag (host-checked) */
} moe_mail_t;

static moe_mail_t *moe_mail;        /* pinned[Q4_MAX_LAYER] */
static void *moe_mail_d;            /* device alias (legacy; kernel now
                                     * writes the moe_mstg staging ring) */
/* Per-layer device staging slots the route kernels fill; an in-stream
 * D2H copy publishes each slot to the pinned mailbox.  Stride rounds the
 * staged struct up from 1176B. */
#define MOE_MSTG_STRIDE 1280u
static uint8_t *moe_mstg;           /* device[Q4_MAX_LAYER * 640] */
#define MOE_MAIL_DMA ((size_t)offsetof(moe_mail_t, done))
_Static_assert(offsetof(moe_mail_t, done) <= MOE_MSTG_STRIDE,
               "mailbox prefix must fit its device staging slot");
#define MOE_DEVN_MAXTOK 8u          /* verify batch bound (MTP_MAX_DRAFT+1) */
static float *moe_hyL;              /* pinned per-layer CPU output */
/* Q4_CPX_SPIN (default on): the verify wait is a GPU spin on moe_gdone
 * (coherent host memory, one 64-byte line per layer) and hyL comes from
 * coherent memory too — host-registered pages are L2-cached on the GPU, so
 * a kernel could read stale rows (q4_rocm.cu, cpx_spin_k). */
static unsigned long long *moe_gdone;
static int moe_hyL_coh;
static int moe_spin = -1;
static int moe_spin_on(void) {
    if (moe_spin < 0) {
        const char *e = getenv("Q4_CPX_SPIN");
        moe_spin = !(e && e[0] == '0');
    }
    return moe_spin;
}
static void moe_hyL_free(void) {
    if (!moe_hyL) return;
    if (moe_hyL_coh) {
        q4_hip_host_free_coh(moe_hyL);
    } else {
        q4_hip_host_unregister(moe_hyL);
        free(moe_hyL);
    }
    moe_hyL = NULL;
    moe_hyL_coh = 0;
}
static void moe_done_store(int L, unsigned long long seq) {
    __atomic_store_n(&moe_mail[L].done, seq, __ATOMIC_RELEASE);
    if (moe_gdone)
        __atomic_store_n(&moe_gdone[(size_t)L * 8], seq, __ATOMIC_RELEASE);
}
static float *moe_dhyL;             /* device alias */
static int32_t *moe_des;            /* device hit slots [64] */
static float *moe_dew;              /* device hit weights [64] */
static int32_t *moe_dnh;            /* device hit count */
static int32_t *moe_res_s;          /* device resident-miss slots [64] */
static float *moe_res_w;            /* device resident-miss weights [64] */
static int32_t *moe_res_n;          /* device resident-miss count */
static unsigned long long *moe_dseq; /* per-layer route sequence counters */
static unsigned long long *moe_drseq; /* per-layer profile stamps */
static unsigned long long *moe_dprof; /* route_dev cycles/calls counters */
static pthread_t moe_disp_th;
static _Atomic int moe_disp_on;
static int moe_disp_started;
static const q4_gguf *moe_disp_g;
static q4_expert_cache *moe_disp_c;
static int moe_disp_nl;
static _Atomic int moe_dev_err;
static _Atomic unsigned long long moe_dev_jobs, moe_dev_nmiss;
static _Atomic unsigned long long moe_dev_lat_us;  /* sum of job latency */

void q4_moe_dev_stats(unsigned long long *jobs, unsigned long long *miss,
                      unsigned long long *lat_us) {
    *jobs = atomic_load(&moe_dev_jobs);
    *miss = atomic_load(&moe_dev_nmiss);
    *lat_us = atomic_load(&moe_dev_lat_us);
}

/* Average on-GPU duration of route_dev_k (µs) since the last call.  Reads
 * back the kernel's clock64 counters — call only with the stream idle.
 * `sub` (optional, 4 doubles) gets per-section averages: softmax scan,
 * serial top-k/split, mailbox publish, everything else. */
double q4_moe_route_prof(double *sub) {
    if (!moe_dprof) return -1.0;
    unsigned long long h[8] = { 0 };
    if (!q4_hip_d2h(h, moe_dprof, sizeof(h)) || !h[1]) return -1.0;
    if (!q4_hip_copy_memset(moe_dprof, 0, sizeof(h))) return -1.0;
    q4_hip_copy_wait();
    if (sub) {
        /* counters are nanoseconds (MSG_RTN_GET_REALTIME) */
        sub[0] = (double)h[2] / (double)h[1] * 1e-3;
        sub[1] = (double)h[3] / (double)h[1] * 1e-3;
        sub[2] = (double)h[4] / (double)h[1] * 1e-3;
        sub[3] = ((double)h[0] - (double)h[2] - (double)h[3] -
                  (double)h[4]) / (double)h[1] * 1e-3;
    }
    return (double)h[0] / (double)h[1] * 1e-3;
}

static int moe_cpu_bufs(uint32_t n_embd, uint32_t n_exp, uint32_t topk,
                        uint32_t n_ff);
static int moe_cpu_want(void);
static int moe_verify_reuse(void);

/* Verify publishes carry each token's L1 hit slots.  Q4_HIT_TOUCH=0
 * leaves them unread (hits never refreshed the staging LRU before: its
 * victim was the oldest-bound slot).  Q4_ROUTE_TRACE=<file> appends every
 * verify layer's routing (hit occupants + misses) plus a slot-table
 * snapshot at each request start, for offline cache-policy replay. */
/* Decode prefetch (Q4_DPF): each verify layer L also runs layer L+1's
 * router on its MoE input; the best-scored experts L+1's devtab lacks
 * come back through a small pinned mailbox and the dispatcher uploads
 * them into L1 before L+1 routes (q4_expert_dprefetch).  Q4_DPF_N caps
 * the candidates per layer, Q4_DPF_PMIN the best-token probability. */
typedef struct {
    volatile unsigned long long seq;
    volatile unsigned long long tseq;
    volatile int32_t n, pad;
    volatile int32_t ids[16];
    volatile float sc[16];
} moe_pred_t;
static moe_pred_t *moe_pred;           /* host, pinned, per layer */
static uint8_t *moe_pstg;              /* device staging, per layer */
static float *moe_dplog;               /* device, 8 x n_exp logits */
static unsigned long long *moe_dpcnt;  /* device, per-layer record counter */
static int moe_dpf = -1;
static uint32_t moe_dpf_n;
static float moe_dpf_pmin;
static int moe_dpf_on(void) {
    if (moe_dpf < 0) {
        const char *e = getenv("Q4_DPF");
        moe_dpf = e && e[0] == '1';
        const char *n = getenv("Q4_DPF_N");
        moe_dpf_n = n ? (uint32_t)atoi(n) : 4u;
        if (moe_dpf_n < 1) moe_dpf_n = 1;
        if (moe_dpf_n > 16) moe_dpf_n = 16;
        const char *pm = getenv("Q4_DPF_PMIN");
        moe_dpf_pmin = pm ? (float)atof(pm) : 0.01f;
    }
    return moe_dpf;
}

static int moe_hit_touch = -1;
static float moe_freq_decay;        /* 0: Q4_STAGE_ADMIT=0, no freq table */
static FILE *moe_rtrace;
static double moe_rtrace_last;
/* Q4_TAIL_SKIP=r (opt-in, off by default; Strata's
 * STRATA_ROUTE_TAIL_SKIP): in a verify window, a missed expert that every
 * token routing it ranks r-th or lower in its top-k (1-based; 7 = the four
 * lightest of ten) is neither computed on the CPU nor staged; it adds zero
 * and the other experts keep their routed weights (no renormalisation).
 * An approximation: the output changes. */
static int moe_tail_rank = -1;
static _Atomic unsigned long long moe_tail_skipped, moe_tail_seen;
static int moe_tail_on(void) {
    if (moe_tail_rank < 0) {
        const char *e = getenv("Q4_TAIL_SKIP");
        int r = e ? atoi(e) : 0;
        moe_tail_rank = r >= 1 && r <= 16 ? r : 0;
        if (moe_tail_rank)
            fprintf(stderr, "q4 moe: Q4_TAIL_SKIP=%d — missed experts every "
                            "window token ranks %d-th or lower are skipped "
                            "(approximation)\n", moe_tail_rank, moe_tail_rank);
    }
    return moe_tail_rank;
}
/* Drops the skipped pairs from a window's token-segmented miss list;
 * returns the new pair count. */
static uint32_t moe_tail_skip(int32_t *ids, float *wts, const uint8_t *rk,
                              uint32_t *tn, uint32_t nt, uint32_t n) {
    const uint32_t r0 = (uint32_t)moe_tail_rank - 1;
    uint8_t keep[64];
    for (uint32_t i = 0; i < n; i++) {
        uint32_t best = rk[i];
        for (uint32_t j = 0; j < n; j++)
            if (ids[j] == ids[i] && rk[j] < best) best = rk[j];
        keep[i] = best < r0;
    }
    uint32_t o = 0, w = 0;
    for (uint32_t t = 0; t < nt; t++) {
        uint32_t c = 0;
        for (uint32_t k = 0; k < tn[t] && o + k < n; k++) {
            const uint32_t i = o + k;
            if (!keep[i]) continue;
            ids[w] = ids[i];
            wts[w] = wts[i];
            w++;
            c++;
        }
        o += tn[t];
        tn[t] = c;
    }
    atomic_fetch_add(&moe_tail_seen, n);
    atomic_fetch_add(&moe_tail_skipped, n - w);
    return w;
}
void q4_moe_tail_skip_stats(unsigned long long *skipped,
                            unsigned long long *seen) {
    if (skipped) *skipped = atomic_load(&moe_tail_skipped);
    if (seen) *seen = atomic_load(&moe_tail_seen);
}

static void moe_hit_note(q4_expert_cache *c, int32_t L, uint32_t nt,
                         const uint32_t *hn, const int32_t *hs,
                         const uint32_t *tn, const int32_t *ids) {
    if (moe_hit_touch < 0) {
        const char *e = getenv("Q4_HIT_TOUCH");
        moe_hit_touch = !(e && e[0] == '0');
        const char *tf = getenv("Q4_ROUTE_TRACE");
        if (tf && tf[0]) moe_rtrace = fopen(tf, "ab");
        const char *ad = getenv("Q4_STAGE_ADMIT");
        const char *fd = getenv("Q4_FREQ_DECAY");
        moe_freq_decay = (ad && ad[0] == '0') ? 0.f
                         : fd ? (float)atof(fd) : 0.97f;
    }
    int32_t hsl[128];
    uint32_t nh = 0;
    for (uint32_t t = 0; t < nt && t < 8; t++)
        for (uint32_t i = 0; i < hn[t] && i < 16; i++)
            hsl[nh++] = hs[t * 16 + i];
    if (moe_hit_touch) q4_expert_slots_touch(c, hsl, nh);
    if (moe_freq_decay > 0.f) {
        /* every expert this layer routed (hit occupants + misses), once */
        /* Q4_FREQ_TOK=1: weight each expert by the tokens that routed it
         * (default: once per publish) */
        static int ftok = -1;
        if (ftok < 0) ftok = getenv("Q4_FREQ_TOK") && getenv("Q4_FREQ_TOK")[0] == '1';
        int32_t ue[128];
        float uw[128];
        uint32_t nu = 0, nm = 0;
        for (uint32_t t = 0; t < nt && t < 8; t++) nm += tn[t];
        if (nm > 64) nm = 64;
        for (uint32_t i = 0; i < nh + nm; i++) {
            int32_t e;
            if (i < nh) {
                int32_t sl = -1;
                e = q4_expert_slot_eid(c, hsl[i], &sl);
                if (sl != L) continue;   /* rebound since the route */
            } else {
                e = ids[i - nh];
            }
            uint32_t u = 0;
            while (u < nu && ue[u] != e) u++;
            if (u == nu) {
                ue[nu] = e;
                uw[nu++] = 1.f;
            } else if (ftok) {
                uw[u] += 1.f;
            }
        }
        q4_expert_freq_note(c, L, ue, uw, nu, moe_freq_decay);
    }
    if (!moe_rtrace) return;
    const double now = moe_now();
    if (now - moe_rtrace_last > 0.5) {
        static int16_t snap[3 * 16384];
        uint32_t n = q4_expert_slot_dump(c, snap, 16384);
        int32_t hd[2] = {-1, (int32_t)n};
        fwrite(hd, sizeof(hd), 1, moe_rtrace);
        fwrite(snap, 6, n, moe_rtrace);
    }
    moe_rtrace_last = now;
    int16_t rec[2 + 8 * 2 + 8 * 32];
    uint32_t k = 0, o = 0;
    rec[k++] = (int16_t)L;
    rec[k++] = (int16_t)nt;
    for (uint32_t t = 0; t < nt && t < 8; t++) {
        uint32_t h = hn[t] > 16 ? 16 : hn[t];
        rec[k++] = (int16_t)h;
        rec[k++] = (int16_t)tn[t];
        for (uint32_t i = 0; i < h; i++)
            rec[k++] = (int16_t)q4_expert_slot_eid(c, hs[t * 16 + i], NULL);
        for (uint32_t i = 0; i < tn[t] && o < 64; i++)
            rec[k++] = (int16_t)ids[o++];
    }
    int32_t hd[2] = {L, (int32_t)k};
    fwrite(hd, sizeof(hd), 1, moe_rtrace);
    fwrite(rec, 2, k, moe_rtrace);
    if (L == 0 || L + 1 == moe_disp_nl) fflush(moe_rtrace);
    /* Q4_ROUTE_TRACE_X=1: also the layer's router input rows (moe_hx holds
     * them until the next layer's publish), for predictor studies. */
    static int tx = -1;
    if (tx < 0) tx = getenv("Q4_ROUTE_TRACE_X") &&
                     getenv("Q4_ROUTE_TRACE_X")[0] == '1';
    if (tx && moe_hx && moe_nembd) {
        int32_t hx[2] = {-2, (int32_t)(nt * moe_nembd)};
        fwrite(hx, sizeof(hx), 1, moe_rtrace);
        fwrite(moe_hx, sizeof(float), (size_t)nt * moe_nembd, moe_rtrace);
    }
}

static void *moe_dispatch_loop(void *arg) {
    (void)arg;
    const q4_gguf *g = moe_disp_g;
    q4_expert_cache *c = moe_disp_c;
    unsigned long long last[Q4_MAX_LAYER];
    memset(last, 0, sizeof(last));
    unsigned long long last_r[Q4_MAX_LAYER];
    memset(last_r, 0, sizeof(last_r));
    const uint32_t disp_topk = g->n_expert_used > 16 ? 16 :
                               (g->n_expert_used ? g->n_expert_used : 10);
    /* In-flight publish per layer: a batched (verify) entry carries up to
     * MOE_DEVN_MAXTOK token segments.  With the window on, one CPU job
     * covers every token (each miss expert dequantized once).  Otherwise
     * each token is its own job.  The layer's done stamp lands only when
     * the whole entry resolves. */
    typedef struct {
        unsigned long long seq;   /* 0 = idle */
        uint32_t ntok;
        uint32_t tn[8];
        uint32_t off[8];          /* segment start inside ids[]/wts[] */
        int32_t ids[64];
        float wts[64];
        uint32_t next;            /* next token segment to dispatch */
        int slot[8];
        uint64_t tag[8];          /* poll by tag: slots can be recycled */
        uint8_t fin[8];
        uint32_t joined;
        double t0;
        int win_state;            /* 0 per-token, 1 try window, 2 in flight,
                                   * 3 joined */
        int win_slot;
        uint64_t win_tag;
        int rep;
        double t_rep;
    } disp_pend_t;
    static disp_pend_t pend_buf[Q4_MAX_LAYER];
    int32_t ids[64];
    float wts[64];
    /* Deferred warm-up staging: the scan pushes (layer, ids) here and the
     * ring drains while the CPU pool crunches — staging I/O never delays
     * the next dispatch. */
    int32_t sq_layer[128], sq_ids[128][64];
    uint32_t sq_n[128], sq_r = 0, sq_w = 0;
    /* Q4_DISP_STAGE=0: skip the miss warm-up staging (diagnostic — its
     * H2D DMA reads DRAM alongside the CPU jobs). */
    const char *dse = getenv("Q4_DISP_STAGE");
    const int disp_stage = !(dse && dse[0] == '0');
    moe_cpx_repeat = getenv("Q4_CPX_REPEAT") && getenv("Q4_CPX_REPEAT")[0] == '1';
    moe_stage_late = getenv("Q4_STAGE_LATE") && getenv("Q4_STAGE_LATE")[0] == '1';
    unsigned long long last_p[Q4_MAX_LAYER];
    memset(last_p, 0, sizeof(last_p));
    /* Q4_DPF_AT=1 (default): hold a layer's guess until that layer's CPU
     * job has joined, so the uploads read DRAM while the GPU runs layer
     * L+1's pre-route work instead of beside the job; 0 uploads on
     * arrival. */
    const char *dat = getenv("Q4_DPF_AT");
    const int dpf_at = dat ? atoi(dat) : 1;
    while (moe_disp_on) {
        /* Decode prefetch first: its uploads race layer L+1's route. */
        for (int L = 0; moe_pred && L + 1 < moe_disp_nl && L < Q4_MAX_LAYER;
             L++) {
            const unsigned long long ps = moe_pred[L].seq;
            if (ps <= last_p[L]) continue;
            if (dpf_at && (pend_buf[L].seq || last[L] <= moe_pred[L].tseq))
                continue;   /* layer L's publish unread or its job running */
            /* L+1 already routed this pass: too late for this guess */
            if (moe_mail[L + 1].seq > moe_pred[L].tseq) {
                last_p[L] = ps;
                continue;
            }
            int32_t pid[16];
            int32_t pn = moe_pred[L].n;
            if (pn < 0) pn = 0;
            if (pn > 16) pn = 16;
            memcpy(pid, (const void *)moe_pred[L].ids, (size_t)pn * 4);
            if (moe_pred[L].seq != ps) continue;   /* torn */
            if (!pn ||
                q4_expert_dprefetch(c, L, L + 1, pid, (uint32_t)pn) != -2)
                last_p[L] = ps;
        }
        while (sq_r != sq_w) {
            uint32_t i = sq_r & 127;
            /* Never block the drain on a busy io pool: the completion pass
             * below stamps 'done' for the in-graph cpx_wait — parking here
             * starves it until the wait kernel's timeout aborts decode. */
            if (q4_expert_stage_batch_try(c, sq_layer[i], sq_ids[i],
                                          sq_n[i], 1) == -2)
                break;
            sq_r++;
        }
        int did = 0;
        /* In-flight publishes: a window entry is one job; otherwise keep
         * dispatching token segments and join finished jobs.  The done
         * stamp lands once the whole entry resolves — the in-stream wait
         * key is per-layer so interleaved dispatches can't falsely pass. */
        for (int L = 0; L < moe_disp_nl && L < Q4_MAX_LAYER; L++) {
            disp_pend_t *P = pend_buf + L;
            if (!P->seq) continue;
            if (P->win_state == 2) {
                if (P->win_slot >= 0 &&
                    q4_cpuex_tag_done(P->win_slot, P->win_tag)) {
                    VTL_H(L, 2);
                    (void)q4_cpuex_join_tag(P->win_slot, P->win_tag);
                    /* Q4_CPX_REPEAT=1 (diagnostic): run the same job a
                     * second time back to back — its cost is the job's
                     * floor with every expert already in cache. */
                    if (moe_cpx_repeat && !P->rep) {
                        P->rep = 1;
                        P->t_rep = moe_now();
                        int s2 = q4_cpuex_dispatch_win(
                            g, c, L, moe_hx, P->ntok, P->ids, P->wts, P->tn,
                            moe_hyL + (size_t)L * MOE_DEVN_MAXTOK * moe_nembd,
                            &P->win_tag);
                        if (s2 >= 0) {
                            P->win_slot = s2;
                            moe_rep_us1 += (P->t_rep - moe_vtl_h[L][1]) * 1e6;
                            did = 1;
                            continue;
                        }
                    }
                    if (P->rep) {
                        moe_rep_us2 += (moe_now() - P->t_rep) * 1e6;
                        moe_rep_n++;
                    }
                    if (disp_stage && moe_stage_late && sq_w - sq_r < 128) {
                        int32_t uids[64];
                        uint32_t uk = 0, o = 0;
                        for (uint32_t t = 0; t < P->ntok && uk < 64; t++) {
                            for (uint32_t i = 0; i < P->tn[t] && uk < 64; i++) {
                                int32_t id = P->ids[o + i];
                                uint32_t u = 0;
                                for (; u < uk; u++)
                                    if (uids[u] == id) break;
                                if (u == uk) uids[uk++] = id;
                            }
                            o += P->tn[t];
                        }
                        if (uk) {
                            uint32_t wi = sq_w & 127;
                            sq_layer[wi] = L;
                            sq_n[wi] = uk;
                            memcpy(sq_ids[wi], uids, uk * sizeof(int32_t));
                            sq_w++;
                        }
                    }
                    for (uint32_t t = 0; t < P->ntok; t++) P->fin[t] = 1;
                    P->joined = P->ntok;
                    P->next = P->ntok;
                    P->win_state = 3;
                    did = 1;
                }
            } else if (P->win_state == 1) {
                int slot = q4_cpuex_dispatch_win(
                    g, c, L, moe_hx, P->ntok, P->ids, P->wts, P->tn,
                    moe_hyL + (size_t)L * MOE_DEVN_MAXTOK * moe_nembd,
                    &P->win_tag);
                if (slot == -1) {
                    /* pool busy: retry this entry next pass */
                } else if (slot < 0) {
                    P->win_state = 0;
                } else {
                    VTL_H(L, 1);
                    P->win_slot = slot;
                    P->win_state = 2;
                    P->next = P->ntok;
                    atomic_fetch_add(&moe_dev_jobs, 1);
                    uint32_t nmiss = 0;
                    for (uint32_t t = 0; t < P->ntok; t++) nmiss += P->tn[t];
                    atomic_fetch_add(&moe_dev_nmiss, nmiss);
                    /* Q4_STAGE_LATE=1: queue the warm-up staging when the
                     * job joins instead (its DMA then reads DRAM after the
                     * CPU did, not alongside it). */
                    if (disp_stage && !moe_stage_late && sq_w - sq_r < 128) {
                        int32_t uids[64];
                        uint32_t uk = 0, o = 0;
                        for (uint32_t t = 0; t < P->ntok && uk < 64; t++) {
                            for (uint32_t i = 0; i < P->tn[t] && uk < 64; i++) {
                                int32_t id = P->ids[o + i];
                                uint32_t u = 0;
                                for (; u < uk; u++)
                                    if (uids[u] == id) break;
                                if (u == uk) uids[uk++] = id;
                            }
                            o += P->tn[t];
                        }
                        if (uk) {
                            uint32_t wi = sq_w & 127;
                            sq_layer[wi] = L;
                            sq_n[wi] = uk;
                            memcpy(sq_ids[wi], uids, uk * sizeof(int32_t));
                            sq_w++;
                        }
                    }
                    did = 1;
                }
            }
            if (P->win_state == 0) {
                for (uint32_t t = 0; t < P->ntok; t++) {
                    if (P->fin[t] || P->slot[t] < 0) continue;
                    if (!q4_cpuex_tag_done(P->slot[t], P->tag[t])) continue;
                    (void)q4_cpuex_join_tag(P->slot[t], P->tag[t]);
                    P->fin[t] = 1;
                    P->joined++;
                    did = 1;
                }
                while (P->next < P->ntok) {
                    uint32_t t = P->next;
                    if (!P->tn[t]) { P->next++; continue; }
                    int slot = q4_cpuex_dispatch2(
                        g, c, L, moe_hx + (size_t)t * moe_nembd,
                        P->ids + P->off[t], P->wts + P->off[t], P->tn[t],
                        moe_hyL + ((size_t)L * MOE_DEVN_MAXTOK + t) * moe_nembd,
                        &P->tag[t]);
                    if (slot < 0) {
                        if (slot == -2)
                            atomic_store_explicit(&moe_dev_err, 1,
                                                  memory_order_relaxed);
                        break;   /* pool busy: resume next pass */
                    }
                    P->slot[t] = slot;
                    P->next++;
                    atomic_fetch_add(&moe_dev_jobs, 1);
                    atomic_fetch_add(&moe_dev_nmiss, P->tn[t]);
                    if (disp_stage && sq_w - sq_r < 128) {
                        uint32_t wi = sq_w & 127;
                        sq_layer[wi] = L;
                        sq_n[wi] = P->tn[t];
                        memcpy(sq_ids[wi], P->ids + P->off[t],
                               P->tn[t] * sizeof(int32_t));
                        sq_w++;
                    }
                    did = 1;
                }
            }
            if (P->next == P->ntok && P->joined == P->ntok) {
                atomic_fetch_add(&moe_dev_lat_us,
                                 (unsigned long long)(
                                     (moe_now() - P->t0) * 1e6));
                moe_done_store(L, P->seq);
                VTL_H(L, 3);
                P->seq = 0;
                did = 1;
            }
        }
        (void)q4_cpuex_retire();
        for (int L = 0; L < moe_disp_nl && L < Q4_MAX_LAYER; L++) {
            /* Profile channel: count the full top-k whenever the kernel
             * publishes a new routed list.  Fire-and-forget — never waited
             * on by either side, so a missed bump only thins the histogram. */
            unsigned long long rs = moe_mail[L].rseq;
            if (rs != last_r[L]) {
                int32_t rid[16];
                for (uint32_t k = 0; k < disp_topk; k++)
                    rid[k] = moe_mail[L].routed[k];
                if (moe_mail[L].rseq == rs) {   /* torn: skip this sample */
                    q4_expert_route_note(c, L, rid, (int)disp_topk);
                    last_r[L] = rs;
                }
            }
            if (pend_buf[L].seq) continue;   /* entry still resolving */
            unsigned long long s0 = moe_mail[L].seq;
            if (s0 <= last[L]) continue;   /* consumed, stale or torn-below */
            /* A stable forward jump is a real overrun: a publish was
             * consumed before this pass read it, so its CPU misses never
             * dispatched.  The DMA writes seq as two dwords — a mid-copy
             * read at a carry boundary (255->256, 511->512) transiently
             * shows 0/256/767 and settles on the real seq; re-read and
             * only flag when it is stable. */
            if (last[L] && s0 > last[L] + 1 &&
                moe_mail[L].seq == s0)
                atomic_store_explicit(&moe_dev_err, 1,
                                      memory_order_relaxed);
            uint32_t nt = moe_mail[L].ntok;
            if (!nt) nt = 1;
            if (nt > MOE_DEVN_MAXTOK) nt = MOE_DEVN_MAXTOK;
            uint32_t n = moe_mail[L].n;
            if (n > 64) n = 64;
            memcpy(ids, (const void *)moe_mail[L].ids, n * sizeof(int32_t));
            memcpy(wts, (const void *)moe_mail[L].wts, n * sizeof(float));
            uint32_t tn[8];
            uint32_t hn[8];
            int32_t hs[8 * 16];
            uint8_t rk[64];
            if (nt > 1) {
                memcpy(tn, (const void *)moe_mail[L].tn, sizeof(tn));
                memcpy(hn, (const void *)moe_mail[L].hn, sizeof(hn));
                memcpy(hs, (const void *)moe_mail[L].hs, sizeof(hs));
                if (moe_tail_on())
                    memcpy(rk, (const void *)moe_mail[L].rk, n);
            } else
                tn[0] = n;
            if (moe_mail[L].seq != s0) continue;   /* torn: retry next pass */
            last[L] = s0;
            did = 1;
            if (nt > 1) moe_hit_note(c, L, nt, hn, hs, tn, ids);
            if (nt > 1 && n && moe_tail_on()) {
                uint32_t sum = 0;
                for (uint32_t t = 0; t < nt; t++) sum += tn[t];
                if (sum == n) n = moe_tail_skip(ids, wts, rk, tn, nt, n);
            }
            VTL_H(L, 0);
            if (!n) {
                /* Zero-miss publish (any ntok): the in-stream wait keys on
                 * seq for every layer call, and every token's hy row must
                 * read zeros at merge. */
                for (uint32_t t = 0; t < nt; t++)
                    memset(moe_hyL +
                               ((size_t)L * MOE_DEVN_MAXTOK + t) *
                                   moe_nembd,
                           0, (size_t)moe_nembd * sizeof(float));
                moe_done_store(L, s0);
                if (moe_vtl > 0)
                    moe_vtl_h[L][1] = moe_vtl_h[L][2] = moe_vtl_h[L][3] =
                        moe_vtl_h[L][0];
                continue;
            }
            disp_pend_t *P = pend_buf + L;
            memset(P, 0, sizeof(*P));
            P->seq = s0;
            P->ntok = nt;
            memcpy(P->tn, tn, sizeof(tn));
            memcpy(P->ids, ids, n * sizeof(int32_t));
            memcpy(P->wts, wts, n * sizeof(float));
            uint32_t o = 0;
            for (uint32_t t = 0; t < nt; t++) {
                P->off[t] = o;
                P->slot[t] = -1;
                o += P->tn[t];
                if (o > 64) {   /* tn/n inconsistent — torn read or a
                                 * kernel-side overflow: drop the publish;
                                 * the in-stream wait will time out and
                                 * flag the pipeline wedged. */
                    P->seq = 0;
                    atomic_store_explicit(&moe_dev_err, 1,
                                          memory_order_relaxed);
                    break;
                }
                if (!P->tn[t]) {
                    /* No misses: zero the out row so the merged axpy
                     * reads zeros, and count the token resolved. */
                    memset(moe_hyL +
                               ((size_t)L * MOE_DEVN_MAXTOK + t) *
                                   moe_nembd,
                           0, (size_t)moe_nembd * sizeof(float));
                    P->fin[t] = 1;
                    P->joined++;
                }
            }
            P->t0 = moe_now();
            P->win_slot = -1;
            if (P->seq && nt > 1 && moe_verify_reuse())
                P->win_state = 1;
        }
        if (!did && sq_r == sq_w) _mm_pause();
    }
    return NULL;
}

/* Devroute init: device residency table, mailbox, per-layer CPU output
 * staging and the dispatcher thread.  Returns 1 when the whole-layer
 * graphable path may be used. */
static int moe_route_prof; /* Q4_ROUTE_PROF set -> kernel publishes routed[] */
static int moe_dev_tried;      /* q4_moe_dev_free resets it for re-init */
static int moe_dev_init(const q4_gguf *g, q4_expert_cache *c) {
    if (moe_dev_tried) return moe_disp_started && moe_mail != NULL;
    moe_dev_tried = 1;
    {
        const char *rp = getenv("Q4_ROUTE_PROF");
        moe_route_prof = (rp && rp[0]) ? 1 : 0;
    }
    if (!moe_cpu_bufs(g->n_embd, g->n_expert, g->n_expert_used,
                      g->n_ff_exp ? g->n_ff_exp : 640))
        return 0;
    if (!q4_expert_dev_enable(c)) return 0;
    (void)moe_vtl_on();   /* allocates outside any capture */
    size_t mail_n = sizeof(moe_mail_t) * Q4_MAX_LAYER;
    moe_mail = malloc(mail_n);
    /* hyL holds MOE_DEVN_MAXTOK rows per layer: verify publishes every
     * token's CPU-miss job in one mailbox entry, each with its own out
     * row. */
    {
        const size_t hyb = (size_t)Q4_MAX_LAYER * MOE_DEVN_MAXTOK *
                           g->n_embd * sizeof(float);
        moe_hyL_coh = 0;
        if (moe_spin_on() && q4_hip_cpx_spin_init()) {
            moe_hyL = q4_hip_host_alloc_coh(hyb);
            if (moe_hyL) {
                moe_gdone = q4_hip_host_alloc_coh((size_t)Q4_MAX_LAYER * 64);
                if (moe_gdone) {
                    memset(moe_gdone, 0, (size_t)Q4_MAX_LAYER * 64);
                    moe_hyL_coh = 1;
                } else {
                    q4_hip_host_free_coh(moe_hyL);
                    moe_hyL = NULL;
                }
            }
        }
        if (!moe_hyL_coh) moe_hyL = malloc(hyb);
    }
    if (!moe_mail || !moe_hyL) goto fail;
    memset(moe_mail, 0, mail_n);
    memset(moe_hyL, 0,
           (size_t)Q4_MAX_LAYER * MOE_DEVN_MAXTOK * g->n_embd *
               sizeof(float));
    q4_hip_host_register(moe_mail, mail_n);
    if (!moe_hyL_coh)
        q4_hip_host_register(moe_hyL,
                             (size_t)Q4_MAX_LAYER * MOE_DEVN_MAXTOK *
                                 g->n_embd * sizeof(float));
    moe_mail_d = q4_hip_host_devptr(moe_mail);
    moe_dhyL = (float *)q4_hip_host_devptr(moe_hyL);
    /* Route kernels stage each publish here; an in-stream D2H copy lands
     * it in the pinned mailbox — no PCIe writes in compute kernels. */
    moe_mstg = (uint8_t *)q4_hip_malloc(
        (size_t)Q4_MAX_LAYER * MOE_MSTG_STRIDE);
    /* Device block: per-layer seq counters, then the hit arrays the batched
     * route kernel fills (8 tokens x 64 slots), then dnh/prof/drseq.  The
     * drseq tail must be its own MAX_LAYER slice — the old layout let
     * profile stamps run past the allocation. */
    /* Tail after drseq: the resident-miss work list the route kernel fills
     * for GPU-PCIe execution (single-token path only — verify batches keep
     * all misses on the mailbox). */
    moe_dseq = (unsigned long long *)q4_hip_malloc(
        (size_t)Q4_MAX_LAYER * sizeof(uint64_t) +
        8 * 64 * sizeof(int32_t) + 8 * 64 * sizeof(float) +
        8 * sizeof(int32_t) + 64 +
        (size_t)Q4_MAX_LAYER * sizeof(uint64_t) +
        64 * sizeof(int32_t) + 64 * sizeof(float) + 16);
    if (!moe_mail_d || !moe_dhyL || !moe_dseq || !moe_dwant_d ||
        !moe_mstg)
        goto fail;
    if (!q4_hip_copy_memset(moe_mstg, 0,
                            (size_t)Q4_MAX_LAYER * MOE_MSTG_STRIDE))
        goto fail;
    moe_des = (int32_t *)(moe_dseq + Q4_MAX_LAYER);
    moe_dew = (float *)(moe_des + 8 * 64);
    moe_dnh = moe_des + 2 * 8 * 64;
    moe_dprof = (unsigned long long *)(moe_dnh + 8);
    moe_drseq = moe_dprof + 8;
    moe_res_s = (int32_t *)(moe_drseq + Q4_MAX_LAYER);
    moe_res_w = (float *)(moe_res_s + 64);
    moe_res_n = (int32_t *)(moe_res_w + 64);
    if (((uintptr_t)moe_dprof & 7) || ((uintptr_t)moe_drseq & 7) ||
        !q4_hip_copy_memset(moe_dseq, 0,
                            Q4_MAX_LAYER * sizeof(uint64_t)) ||
        !q4_hip_copy_memset(moe_dprof, 0, 64) ||
        !q4_hip_copy_memset(moe_drseq, 0,
                            Q4_MAX_LAYER * sizeof(uint64_t)))
        goto fail;
    if (moe_dpf_on() && g->n_expert <= 512 &&
        q4_hip_pred_size() == sizeof(moe_pred_t)) {
        const size_t pn = sizeof(moe_pred_t) * Q4_MAX_LAYER;
        moe_pred = malloc(pn);
        moe_pstg = (uint8_t *)q4_hip_malloc(pn);
        moe_dplog = (float *)q4_hip_malloc((size_t)8 * 512 * sizeof(float) +
                                           Q4_MAX_LAYER * sizeof(uint64_t));
        moe_dpcnt = moe_dplog ? (unsigned long long *)(moe_dplog + 8 * 512)
                              : NULL;
        if (moe_pred) {
            memset((void *)moe_pred, 0, pn);
            q4_hip_host_register((void *)moe_pred, pn);
        }
        if (!moe_pred || !moe_pstg || !moe_dplog ||
            !q4_hip_copy_memset(moe_pstg, 0, pn) ||
            !q4_hip_copy_memset(moe_dpcnt, 0,
                                Q4_MAX_LAYER * sizeof(uint64_t))) {
            fprintf(stderr, "q4: decode prefetch off (buffers)\n");
            moe_dpf = 0;
        } else {
            fprintf(stderr, "q4: decode prefetch on (%u per layer, "
                            "pmin %.3f)\n", moe_dpf_n, moe_dpf_pmin);
        }
    }
    q4_hip_copy_wait();
    moe_disp_g = g;
    moe_disp_c = c;
    moe_disp_nl = (int)g->n_layer;
    moe_nembd = g->n_embd;
    moe_disp_on = 1;
    if (pthread_create(&moe_disp_th, NULL, moe_dispatch_loop, NULL) != 0) {
        moe_disp_on = 0;
        goto fail;
    }
    moe_disp_started = 1;
    fprintf(stderr, "q4: devroute live (mailbox %d slots, devtab %ux%u)\n",
            moe_disp_nl, (unsigned)g->n_layer, (unsigned)g->n_expert);
    return 1;
fail:
    moe_disp_on = 0;
    if (moe_mstg) { q4_hip_free(moe_mstg); moe_mstg = NULL; }
    if (moe_dseq) { q4_hip_free(moe_dseq); moe_dseq = NULL; }
    /* Registered before any goto fail can fire — unregister first. */
    if (moe_mail) q4_hip_host_unregister(moe_mail);
    free(moe_mail); moe_mail = NULL; moe_mail_d = NULL;
    moe_hyL_free(); moe_dhyL = NULL;
    q4_hip_host_free_coh(moe_gdone); moe_gdone = NULL;
    return 0;
}

int q4_moe_dev_active(void) {
    return moe_disp_started && moe_mail != NULL;
}

/* Stop and join the dispatcher.  Required before the expert cache it reads
 * (moe_disp_c) is freed — a live dispatcher on a freed cache segfaulted
 * teardown after an aborted decode.  A dispatcher blocked inside
 * stage_batch still finishes the in-flight job before noticing the flag. */
void q4_moe_dev_stop(void) {
    if (!moe_disp_started) return;
    moe_disp_on = 0;
    pthread_join(moe_disp_th, NULL);
    moe_disp_started = 0;
}

/* Full devroute teardown: stop the dispatcher, drop every device
 * allocation (mstg ring + the carved dseq block holding des/dew/dnh/
 * dprof/drseq/res_s/res_w/res_n) and unregister+free the pinned host
 * buffers.  Safe to call with devroute never started.  Callers: invoked
 * from q4_expert_cache_close — devroute state is GLOBAL, so this asserts
 * no other cache (or session with live graphs referencing mailboxes) may
 * still be active; teardown order must stay dispatcher join -> session
 * graph destroy -> this. */
/* Verify MoE graphs, one per (layer, n_tok in 2..4). Baked against the
 * session buffers captured after the prompt prefill. */
static void *vg[Q4_MAX_LAYER][5];
static float *vg_x, *vg_y, *vg_ws;
static int vg_off;
static uint32_t vg_n;

static void verify_graphs_free(void) {
    for (int L = 0; L < Q4_MAX_LAYER; L++) {
        for (int n = 0; n < 5; n++) {
            if (vg[L][n]) q4_hip_graph_free(vg[L][n]);
            vg[L][n] = NULL;
        }
    }
    vg_x = vg_y = vg_ws = NULL;
    vg_n = 0;
}

void q4_moe_dev_free(void) {
    q4_moe_dev_stop();
    verify_graphs_free();
    vg_off = 0;
    if (moe_mstg || moe_dseq) q4_hip_sync();
    if (moe_mstg) { q4_hip_free(moe_mstg); moe_mstg = NULL; }
    if (moe_dseq) { q4_hip_free(moe_dseq); moe_dseq = NULL; }
    if (moe_pstg) { q4_hip_free(moe_pstg); moe_pstg = NULL; }
    if (moe_dplog) { q4_hip_free(moe_dplog); moe_dplog = NULL; moe_dpcnt = NULL; }
    if (moe_pred) {
        q4_hip_host_unregister((void *)moe_pred);
        free((void *)moe_pred);
        moe_pred = NULL;
    }
    moe_des = NULL; moe_dew = NULL; moe_dnh = NULL;
    moe_res_s = NULL; moe_res_w = NULL; moe_res_n = NULL;
    moe_dprof = NULL; moe_drseq = NULL;
    if (moe_mail) { q4_hip_host_unregister(moe_mail); free(moe_mail); moe_mail = NULL; }
    moe_hyL_free();
    q4_hip_host_free_coh(moe_gdone); moe_gdone = NULL;
    if (moe_hx) { q4_hip_host_unregister(moe_hx); free(moe_hx); moe_hx = NULL; }
    if (moe_hy) { q4_hip_host_unregister(moe_hy); free(moe_hy); moe_hy = NULL; }
    if (moe_hh) { free(moe_hh); moe_hh = NULL; }
    if (moe_hrlog) { q4_hip_host_unregister(moe_hrlog); free(moe_hrlog); moe_hrlog = NULL; }
    if (moe_route_h) {
        /* syncb block — also owns the moe_cpx_ and moe_dwant_ aliases. */
        q4_hip_host_unregister((void *)moe_route_h);
        free((void *)moe_route_h);
        moe_route_h = NULL;
    }
    moe_mail_d = NULL; moe_dhyL = NULL; moe_dhy = NULL;
    moe_route_d = NULL; moe_cpx_h = NULL; moe_cpx_d = NULL;
    moe_dwant_h = NULL; moe_dwant_d = NULL;
    moe_hcap = 0;
    moe_disp_nl = 0; moe_disp_g = NULL; moe_disp_c = NULL;
    /* Re-init must be able to start clean: arm the once-guard and drop
     * diagnostics a new session would otherwise inherit (a stale
     * moe_dev_err or cpx timeout trips q4_moe_dev_check immediately). */
    moe_dev_tried = 0;
    moe_dev_err = 0;
    moe_dev_jobs = 0; moe_dev_nmiss = 0; moe_dev_lat_us = 0;
    moe_eager_tag = 0;
    memset(moe_route_last, 0, sizeof(moe_route_last));
    q4_hip_cpx_wait_reset();
}

void q4_moe_dev_warm(const q4_gguf *g, const q4_store *st, int32_t layer) {
    if (!g || !st || layer < 0) return;
    char nm[Q4_MAX_NAME];
    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp_shexp.weight", (int)layer);
    const q4_tensor *t = q4_find_tensor(g, nm);
    if (t && t->ggml_type != Q4_T_F32 && t->ne[0] == g->n_embd)
        (void)moe_sgate_f32d(g, st, t);
}

static int moe_devn_on(void);   /* defined with the batched route below */

int q4_moe_dev_route(const q4_gguf *g, q4_expert_cache *c) {
    if (!moe_cpu_want() || !g || !c || !g->n_embd || !q4_hip_ok()) return 0;
    const char *e = getenv("Q4_DEVROUTE");
    int want1 = !(e && e[0] == '0');
    /* The dispatcher is shared with the batched verify path (devroute_n):
     * bring it up whenever either consumer is enabled, but keep the
     * decode-side return honoring Q4_DEVROUTE alone. */
    if (!want1 && !moe_devn_on()) return 0;
    int ok = moe_dev_init(g, c);
    return want1 ? ok : 0;
}

static int moe_cpu_want(void) {
    if (moe_want_cpu < 0) {
        const char *ce = getenv("Q4_CPU_EXPERTS");
        moe_want_cpu = !(ce && ce[0] == '0');
    }
    return moe_want_cpu;
}

static int moe_cpu_bufs(uint32_t n_embd, uint32_t n_exp, uint32_t topk,
                        uint32_t n_ff) {
    if (moe_hcap >= n_embd && moe_hh) return 1;
    if (moe_hx) q4_hip_host_unregister(moe_hx);
    if (moe_hy) q4_hip_host_unregister(moe_hy);
    if (moe_hrlog) q4_hip_host_unregister(moe_hrlog);
    if (moe_route_h) q4_hip_host_unregister((void *)moe_route_h);
    if (moe_mail) q4_hip_host_unregister(moe_mail);
    free(moe_hx); free(moe_hy); free(moe_hh); free(moe_hrlog);
    free((void *)moe_route_h);
    free(moe_mail); moe_hyL_free();
    q4_hip_host_free_coh(moe_gdone); moe_gdone = NULL;
    moe_mail = NULL; moe_mail_d = NULL; moe_hyL = NULL; moe_dhyL = NULL;
    moe_route_h = NULL; moe_route_d = NULL;
    moe_cpx_h = NULL; moe_cpx_d = NULL;
    moe_dwant_h = NULL; moe_dwant_d = NULL;
    /* moe_hx carries up to MOE_DEVN_MAXTOK token rows: batched verify
     * publishes read every row, single-token decode uses row 0. */
    moe_hx = malloc((size_t)n_embd * MOE_DEVN_MAXTOK * 4);
    moe_hy = malloc((size_t)n_embd * 4);
    moe_hh = malloc((size_t)topk * n_ff * 4);
    moe_hrlog = malloc((size_t)n_exp * 4);
    uint8_t *syncb = malloc(MOE_SYNC_BYTES);
    moe_hcap = 0;
    moe_dhy = NULL;
    if (moe_hx && moe_hy && moe_hh && moe_hrlog) {
        q4_hip_host_register(moe_hx,
                             (size_t)n_embd * MOE_DEVN_MAXTOK * 4);
        q4_hip_host_register(moe_hy, (size_t)n_embd * 4);
        q4_hip_host_register(moe_hrlog, (size_t)n_exp * 4);
        moe_dhy = (float *)q4_hip_host_devptr(moe_hy);
        if (moe_dhy) moe_hcap = n_embd;
    }
    if (syncb) {
        memset(syncb, 0, MOE_SYNC_BYTES);
        q4_hip_host_register(syncb, MOE_SYNC_BYTES);
        uint8_t *sd = q4_hip_host_devptr(syncb);
        if (sd) {
            moe_route_h = (volatile uint32_t *)syncb;
            moe_route_d = (uint32_t *)sd;
            moe_cpx_h = (volatile unsigned long long *)(syncb + 256);
            moe_cpx_d = (unsigned long long *)(sd + 256);
            moe_dwant_h = (volatile unsigned long long *)(syncb + 272);
            moe_dwant_d = (unsigned long long *)(sd + 272);
            memset(moe_route_last, 0, sizeof(moe_route_last));
            q4_cpuex_syncpin((unsigned long long *)moe_cpx_h,
                             (unsigned long long *)moe_cpx_h + 1);
        } else {
            free(syncb);
        }
    }
    return moe_hcap >= n_embd && moe_dhy != NULL;
}

/* 1 when the CPU-expert path is live (env on + buffers registered); decode
 * uses this to pick the graphable flow. */
int q4_moe_cpu_active(const q4_gguf *g) {
    if (!moe_cpu_want() || !g || !g->n_embd) return 0;
    return moe_cpu_bufs(g->n_embd, g->n_expert, g->n_expert_used,
                        g->n_ff_exp ? g->n_ff_exp : 640);
}

static const uint8_t *moe_devw(const q4_gguf *g, const q4_store *st,
                               int32_t layer, const char *pat,
                               const q4_tensor **t_out) {
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), pat, layer);
    const q4_tensor *t = q4_find_tensor(g, n);
    if (t_out) *t_out = t;
    return t ? q4_store_dev(st, t) : NULL;
}


/* GPU part 1 (graphable): y=0, router gemv, shared expert, sgate dot,
 * async D2H of router logits and d_x into pinned host buffers. */
bool q4_moe_gpu_a(const q4_gguf *g, const q4_store *st, int32_t layer,
                  float *d_x, float *d_y, float *d_ws) {
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const q4_tensor *t_router, *t_gate_s, *t_up_s, *t_down_s;
    const uint8_t *d_router = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_inp.weight", &t_router);
    const uint8_t *d_gate_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_shexp.weight", &t_gate_s);
    const uint8_t *d_up_s = moe_devw(g, st, layer,
                                   "blk.%d.ffn_up_shexp.weight", &t_up_s);
    const uint8_t *d_down_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_down_shexp.weight", &t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    const uint8_t *d_sgate = moe_devw(g, st, layer,
                                    "blk.%d.ffn_gate_inp_shexp.weight",
                                    &moe_t_sgate);
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp;
    float *d_h2 = d_h + n_ff;
    float *d_down = d_h2 + n_ff;
    float *d_sh = d_down + n_embd;
    float *d_dot = d_sh + n_embd;
    /* Batched routed-expert scratch (per-expert rows, ≤64 experts). */
    float *d_hall = d_dot + 16;
    float *d_h2all = d_hall + 64 * n_ff;
    float *d_dall = d_h2all + 64 * n_ff;
    if (!q4_hip_gemv_dd(t_router->ggml_type, d_router, t_router->ne[1],
                        t_router->ne[0], d_x, d_rlog, 1.f))
        return false;
    /* Routing D2H first, then a bump flag: the host only waits for these
     * copies, so the shared expert below stays overlapped with topk,
     * CPU dispatch and the hit-expert launches.  The pack kernel also
     * fills d_y and computes the sgate scalar, saving two more nodes. */
    if (!q4_hip_d2h_async(moe_hrlog, d_rlog, (size_t)n_exp * sizeof(float)) ||
        !q4_hip_d2h_async(moe_hx, d_x, (size_t)n_embd * sizeof(float)))
        return false;
    /* shared expert (GPU) while the CPU will do the routed ones: gate+up
     * fused into one multi-seg GEMV; when it fires, silu*mul rides inside
     * moe_pack instead of a separate launch. */
    q4_seg_t sh_segs[2] = {
        { d_gate_s, d_h, t_gate_s->ne[1], t_gate_s->ggml_type , 0.f },
        { d_up_s, d_h2, t_up_s->ne[1], t_up_s->ggml_type , 0.f },
    };
    const bool fused_sh = q4_hip_gemv_m(sh_segs, 2, d_x, n_embd);
    {
        const float *sg = (d_sgate && moe_t_sgate &&
                           moe_t_sgate->ne[0] == n_embd)
                              ? moe_sgate_f32d(g, st, moe_t_sgate)
                              : NULL;
        if (!q4_hip_moe_pack(d_y, d_x, sg, d_dot,
                             (moe_route_d && layer >= 0 && layer < 64)
                                 ? (void *)(moe_route_d + layer)
                                 : NULL,
                             n_embd, true,
                             fused_sh ? d_h : NULL, d_h2,
                             fused_sh ? n_ff : 0))
            return false;
        moe_d_dot = sg ? d_dot : NULL;
    }
    if (!fused_sh) {
        if (!q4_hip_gemv_dd(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                            t_gate_s->ne[0], d_x, d_h, 1.f))
            return false;
        if (!q4_hip_gemv_dd(t_up_s->ggml_type, d_up_s, t_up_s->ne[1],
                            t_up_s->ne[0], d_x, d_h2, 1.f))
            return false;
        if (!q4_hip_silu_mul(d_h, d_h2, n_ff)) return false;
    }
    if (!q4_hip_gemv_dd(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                        t_down_s->ne[0], d_h, d_sh, 1.f))
        return false;
    moe_d_sgate = d_sgate;
    moe_d_sh = d_sh;
    moe_d_down = d_dall;
    moe_dy = d_y;
    moe_dx = d_x;
    moe_dh = d_hall;
    moe_dh2 = d_h2all;
    moe_nembd = n_embd;
    q4_hip_mark_pt(11); /* gA end boundary (Q4_STEP_PROF only) */
    return true;
}

/* Whole-layer graphable MoE (device-resident routing).  Runs entirely on
 * the compute stream: router gemv → x D2H → device top-k+slot split →
 * shared expert → hit-expert exec → in-stream wait for CPU misses → merge.
 * The dispatcher thread handles the mailboxed misses.  Requires
 * q4_moe_dev_route() == 1; the caller owns the fallback when this fails. */
bool q4_moe_gpu_dev(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                    int32_t layer, float *d_res, const float *d_inj,
                    uint32_t hc, float *d_x, float *d_y, float *d_ws) {
    if (!moe_disp_started || !moe_mail_d || !moe_dhyL || !moe_dwant_d)
        return false;
    if (layer < 0 || layer >= moe_disp_nl || layer >= Q4_MAX_LAYER)
        return false;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (!n_exp || !topk || topk > 64) return false;
    const q4_tensor *t_router, *t_gate_s, *t_up_s, *t_down_s;
    const uint8_t *d_router = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_inp.weight", &t_router);
    const uint8_t *d_gate_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_shexp.weight", &t_gate_s);
    const uint8_t *d_up_s = moe_devw(g, st, layer,
                                   "blk.%d.ffn_up_shexp.weight", &t_up_s);
    const uint8_t *d_down_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_down_shexp.weight", &t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    const uint8_t *d_sgate = moe_devw(g, st, layer,
                                    "blk.%d.ffn_gate_inp_shexp.weight",
                                    &moe_t_sgate);
    int32_t *devtab = q4_expert_dev_row(c, layer);
    const q4_tensor *tg, *tu, *td;
    uint64_t og, ou, od, tot;
    if (!devtab ||
        !q4_expert_parts(g, layer, &tg, &tu, &td, &og, &ou, &od, &tot))
        return false;
    int arena = q4_expert_cache_arena(c);
    const uint8_t *base = q4_hip_experts_base_a(arena);
    uint64_t stride = q4_hip_experts_stride_a(arena);
    if (!base || !stride) return false;
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp;
    float *d_h2 = d_h + n_ff;
    float *d_down = d_h2 + n_ff;
    float *d_sh = d_down + n_embd;
    float *d_dot = d_sh + n_embd;
    float *d_hall = d_dot + 16;
    float *d_h2all = d_hall + 64 * n_ff;
    float *d_dall = d_h2all + 64 * n_ff;
    /* router + shared gate/up all consume d_x: one multi-segment launch
     * (the gate/up results are only needed after moe_pack, so computing
     * them early is harmless). */
    q4_seg_t sh_segs[3] = {
        { d_router, d_rlog, t_router->ne[1], t_router->ggml_type , 0.f },
        { d_gate_s, d_h,    t_gate_s->ne[1], t_gate_s->ggml_type , 0.f },
        { d_up_s,   d_h2,   t_up_s->ne[1],   t_up_s->ggml_type   , 0.f },
    };
    const bool fused_sh = q4_hip_gemv_m(sh_segs, 3, d_x, n_embd);
    if (!fused_sh &&
        !q4_hip_gemv_dd(t_router->ggml_type, d_router, t_router->ne[1],
                        t_router->ne[0], d_x, d_rlog, 1.f))
        return false;
    /* x D2H before the mailbox publish: the seq bump implies x landed. */
    if (!q4_hip_d2h_async(moe_hx, d_x, (size_t)n_embd * sizeof(float)))
        return false;
    /* Stage the publish on-device; SDMA carries the slot to the pinned
     * mailbox in-stream so the route kernel never writes host memory
     * (posted-write drains were costing ~60us of hidden wall per call). */
    uint8_t *mstg = moe_mstg + (size_t)layer * MOE_MSTG_STRIDE;
    /* Q4_RES_GPU=N routes the N highest-weight L1-miss resident experts to
     * the GPU PCIe read path; the rest stay on the CPU mailbox so the DRAM
     * pool and gen4 x16 overlap.  0 (default) keeps everything on the CPU —
     * a full takeover measured slower than mailbox on this box. */
    static int moe_res_gpu = -1;
    if (moe_res_gpu < 0) {
        const char *e = getenv("Q4_RES_GPU");
        moe_res_gpu = e ? atoi(e) : 0;
        if (moe_res_gpu < 0) moe_res_gpu = 0;
        if (moe_res_gpu > 64) moe_res_gpu = 64;
    }
    const int32_t *restab =
        moe_res_gpu ? q4_expert_res_devrow(c, layer) : NULL;
    /* Hit-only tokens publish nothing (s_nm==0 skips the mailbox), so the
     * seq wait below passes instantly — zero this layer's hy row in-stream
     * first or moe_final would merge the previous miss token's CPU output
     * again.  Ordered after the previous token's cpx_wait (same stream) and
     * before this token's publish D2H, so it can neither clobber an
     * in-flight job nor race the one this layer is about to post. */
    if (!q4_hip_fill(moe_dhyL +
                         (size_t)layer * MOE_DEVN_MAXTOK * n_embd,
                     0.f, n_embd))
        return false;
    if (!q4_hip_route_dev(d_rlog, devtab, moe_des, moe_dew, moe_dnh,
                          restab, moe_res_s, moe_res_w, moe_res_n,
                          (uint32_t)moe_res_gpu, mstg, moe_dseq + layer,
                          moe_route_prof ? moe_drseq + layer : NULL,
                          moe_dprof, n_exp, topk))
        return false;
    if (!q4_hip_d2h_async(moe_mail + layer, mstg, MOE_MAIL_DMA))
        return false;
    {
        const float *sg = (d_sgate && moe_t_sgate &&
                           moe_t_sgate->ne[0] == n_embd)
                              ? moe_sgate_f32d(g, st, moe_t_sgate)
                              : NULL;
        if (!q4_hip_moe_pack(d_y, d_x, sg, d_dot, NULL, n_embd, true,
                             fused_sh ? d_h : NULL, d_h2,
                             fused_sh ? n_ff : 0))
            return false;
        moe_d_dot = sg ? d_dot : NULL;
    }
    if (!fused_sh) {
        if (!q4_hip_gemv_dd(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                            t_gate_s->ne[0], d_x, d_h, 1.f))
            return false;
        if (!q4_hip_gemv_dd(t_up_s->ggml_type, d_up_s, t_up_s->ne[1],
                            t_up_s->ne[0], d_x, d_h2, 1.f))
            return false;
        if (!q4_hip_silu_mul(d_h, d_h2, n_ff)) return false;
    }
    if (!q4_hip_gemv_dd(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                        t_down_s->ne[0], d_h, d_sh, 1.f))
        return false;
    /* GPU-resident top-k experts: slots/wts/count live on the device. */
    if (!q4_hip_moe_exec_dev(base, stride, moe_des, moe_dew, moe_dnh,
                             tg->ggml_type, og, tu->ggml_type, ou,
                             td->ggml_type, od, d_x, d_hall, d_h2all, d_dall,
                             d_y, n_ff, tg->ne[0], n_embd))
        return false;
    /* Resident-image experts the route kernel classified as GPU-PCIe: same
     * device-count exec over the registered host arena (base = devbase for
     * this layer, stride = per-expert total).  Stream ordering makes the
     * d_hall/d_h2all/d_dall scratch reuse safe; both lists atomically
     * accumulate into d_y.  No entries ⇒ res_n=0 ⇒ the kernels early-out. */
    const uint8_t *rbase = q4_expert_res_devbase(c, layer);
    const uint64_t rstride = q4_expert_res_stride(c, layer);
    if (restab && rbase && rstride &&
        !q4_hip_moe_exec_dev(rbase, rstride, moe_res_s, moe_res_w,
                             moe_res_n, tg->ggml_type, og, tu->ggml_type, ou,
                             td->ggml_type, od, d_x, d_hall, d_h2all, d_dall,
                             d_y, n_ff, tg->ne[0], n_embd))
        return false;
    /* In-stream wait for THIS layer's CPU misses only: a host-callback node
     * spins on the pinned mailbox (device-side polling of host memory is
     * not coherent on gfx1100).  Progress word = the cpuex completion
     * floor: the wait stays alive while ANY job finishes. */
    moe_mail_t *hmail = moe_mail + layer;
    if (!q4_hip_cpx_wait((void *)&hmail->done, (void *)&hmail->seq,
                         moe_cpx_h ? (void *)(moe_cpx_h + 1)
                                   : (void *)&hmail->done))
        return false;
    return q4_hip_moe_final(d_res, d_y,
                          moe_dhyL + (size_t)layer * MOE_DEVN_MAXTOK * n_embd,
                          d_sh, moe_d_dot, d_inj, n_embd, hc);
}

/* Report any wedged pipeline / mailbox overrun; returns -1 when the
 * session must abort.  Split from the tick so a check can run between
 * head-logit readback and token emission: a cpx_wait that hit its 45 s
 * timeout let the stream merge a possibly-incomplete CPU output — the
 * token is only trustworthy if no timeout was recorded. */
int q4_moe_dev_check(void) {
    /* Covers every path that can enqueue a cpx wait: devroute per-layer
     * mailboxes (moe_disp_started) AND the legacy hybrid pipeline
     * (moe_cpx_h) — a wedged CPU job times out on either and the merge
     * downstream would otherwise combine an incomplete output. */
    if (!moe_disp_started && !moe_cpx_h) return 0;
    if (q4_hip_cpx_wait_timeouts()) {
        /* Self-diagnose the wedge: did the GPU publish seqs, did the
         * dispatcher see them, did cpuex slots ever finish? */
        int stuck = -1;
        if (moe_mail)
            for (int L = 0; L < moe_disp_nl; L++)
                if (moe_mail[L].seq > moe_mail[L].done) { stuck = L; break; }
        unsigned long long ls = q4_hip_cpx_wait_lastseen();
        fprintf(stderr, "q4: cpx_wait timed out — CPU-miss pipeline wedged; "
                        "aborting decode (jobs=%llu miss=%llu err=%d "
                        "stuck-layer=%d seq=%llu done=%llu cpx exp=%llu "
                        "done=%llu kseen=%llu kexp=%llu)\n",
                (unsigned long long)atomic_load(&moe_dev_jobs),
                (unsigned long long)atomic_load(&moe_dev_nmiss),
                atomic_load_explicit(&moe_dev_err, memory_order_relaxed),
                stuck,
                stuck >= 0 ? (unsigned long long)moe_mail[stuck].seq : 0,
                stuck >= 0 ? (unsigned long long)moe_mail[stuck].done : 0,
                moe_cpx_h ? (unsigned long long)moe_cpx_h[0] : 0,
                moe_cpx_h ? (unsigned long long)moe_cpx_h[1] : 0,
                ls >> 20, ls & 0xfffffull);
        q4_cpuex_diag();
        return -1;
    }
    if (atomic_load_explicit(&moe_dev_err, memory_order_relaxed)) {
        fprintf(stderr, "q4: devroute mailbox overrun — aborting decode\n");
        return -1;
    }
    return 0;
}

/* Per-request reset of the sticky fault state: a mailbox overrun or a
 * wedged cpx_wait aborts that decode, but the server must answer the next
 * request — clear both latches at session reset. */
void q4_moe_dev_reset(void) {
    atomic_store_explicit(&moe_dev_err, 0, memory_order_relaxed);
    q4_hip_cpx_wait_reset();
}

/* Token-boundary hook: bumps the expert tick (deferred evictions refill)
 * and reports dispatcher errors.  Returns 0 ok, -1 on mailbox overrun. */
int q4_moe_dev_tick(q4_expert_cache *c) {
    if (!moe_disp_started) return 0;
    if (q4_moe_dev_check() < 0) return -1;
    q4_expert_dev_tick(c);
    return 0;
}

static bool moe_gpu_chain(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const float *d_x, float *d_y, const int32_t *ids,
                          const float *wts, uint32_t topk, float *d_h,
                          float *d_h2, float *d_down, uint32_t n_ff,
                          uint32_t n_embd);
static bool moe_gpu_exec(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                         const float *d_x, float *d_y, const int32_t *ids,
                         const float *wts, uint32_t n, float *d_h,
                         float *d_h2, float *d_down, uint32_t n_ff,
                         uint32_t n_embd);
static bool moe_gpu_stage(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const int32_t *ids, const float *wts, uint32_t n);
static bool moe_gpu_exec_staged(const q4_gguf *g, q4_expert_cache *c,
                                int32_t layer, const float *d_x, float *d_y,
                                float *d_h, float *d_h2, float *d_down,
                                uint32_t n_ff, uint32_t n_embd);

/* Staged hit-exec state.  moe_staged_exec: -1 undecided, 0 = staging broken
 * so hits run eagerly in hostpart, 1 = hits published through the mapped
 * arrays and executed by the per-layer gB graph (or gpu_b on the eager
 * tail). */
static int32_t moe_hslots[64];
static float moe_hwts[64];
static uint32_t moe_nhits;
static int moe_staged_exec = -1;

/* Host middle: drain the stream, softmax+topk, dispatch routed experts on
 * the CPU pool, wait for them.  Returns 1 when the CPU path produced h_y;
 * 0 means "dispatch refused" and the caller must run the eager GPU chain;
 * -1 is a hard error. */
/* Wait for the "routing ready" bump (stream-ordered after the two D2H
 * copies), NOT for the whole stream: the shared expert keeps running while
 * the host does topk and dispatches the CPU pool.  Falls back to a full
 * drain if the flag mechanism is unavailable or stalls. */
#define MOE_ROUTE_SPIN 20000000ull /* ~100 ms of pause, then drain */
static int moe_route_wait(int32_t layer) {
    if (!moe_route_h || layer < 0 || layer >= 64)
        return q4_hip_stream_sync() ? 0 : -1;
    volatile uint32_t *f = moe_route_h + layer;
    uint64_t spins = 0;
    while (*f == moe_route_last[layer]) {
        if (++spins >= MOE_ROUTE_SPIN) {
            if (!q4_hip_stream_sync()) return -1;
            break;
        }
        _mm_pause();
    }
    moe_route_last[layer] = *f;
    return 0;
}

int q4_moe_hostpart(const q4_gguf *g, q4_expert_cache *c, int32_t layer) {
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (moe_route_wait(layer) < 0) return -1;
    /* The previous layer's job may still be live on the pool (its join was
     * deferred to the in-stream wait kernel).  Reap before touching the
     * shared host buffers; normally already done so this is instant. */
    if (!q4_cpuex_reap()) return -1;
    double t0 = moe_now();
    q4_softmax(moe_hrlog, n_exp);
    q4_topk(moe_hrlog, n_exp, topk, moe_ids, moe_wts, true);
    q4_expert_route_note(c, layer, moe_ids, (int)topk);
    q4_prof_moe_topk += moe_now() - t0;
    /* Drain async expert H2D enqueued by earlier layers, then split the top-k:
     * experts already resident in the GPU L1 arena run there (GPU is ~90%
     * idle), misses go to the CPU pool reading the DRAM resident image. */
    int32_t gids[64], cids[64];
    float gwt[64], cwt[64];
    uint32_t ng = 0, nc = 0;
    int hybrid = q4_hip_ok() && q4_expert_cache_arena(c) >= 0 &&
                 moe_dh && moe_dh2;
    if (hybrid) {
        double tcw = moe_now();
        q4_hip_copy_wait();
        q4_prof_moe_cw += moe_now() - tcw;
        q4_expert_uploads_mark(c);
        /* The async stager mutates the L1 map under the io mutex; join it
         * before the lock-free gpu_slot reads below.  Its job was posted a
         * full layer ago, so this is normally a no-op. */
        q4_expert_stage_join(c);
        int arena = q4_expert_cache_arena(c);
        for (uint32_t k = 0; k < topk; k++) {
            int32_t s = q4_expert_gpu_slot(c, layer, moe_ids[k]);
            if (s >= 0 && (uint32_t)s < q4_hip_experts_n_a(arena)) {
                gids[ng] = moe_ids[k];
                gwt[ng++] = moe_wts[k];
            } else {
                cids[nc] = moe_ids[k];
                cwt[nc++] = moe_wts[k];
            }
        }
    } else {
        for (uint32_t k = 0; k < topk; k++) {
            cids[nc] = moe_ids[k];
            cwt[nc++] = moe_wts[k];
        }
    }
    q4_prof_moe_ng += ng;
    q4_prof_moe_nc += nc;
    q4_prof_moe_ncall += 1;
    int disp = 1;
    int slot = -1;
    if (nc) {
        /* -1 means all ring slots are busy (e.g. devroute jobs in flight):
         * retire completions and retry briefly before falling back. */
        for (int t = 0; t < 400; t++) {
            slot = q4_cpuex_dispatch2(g, c, layer, moe_hx, cids, cwt, nc,
                                      moe_hy, &moe_eager_tag);
            if (slot != -1) break;
            (void)q4_cpuex_retire();
            _mm_pause();
        }
        disp = slot >= 0;
    } else {
        memset(moe_hy, 0, (size_t)moe_nembd * sizeof(float));
    }
    if (disp && hybrid) {
        /* Warm the CPU-routed experts into GPU L1 for later tokens while the
         * CPU pool crunches. Hit slots are protected from eviction during
         * the batch; warm puts land via the copy stream. */
        double t1 = moe_now();
        if (ng) q4_expert_protect(c, layer, gids, ng, 1);
        /* Q4_MISS_STAGE=0 skips uploading CPU-routed misses into GPU L1 —
         * with a rotating expert tail the binds churn LRU and burn PCIe
         * without ever re-hitting.  Only valid when L2 is fully resident —
         * otherwise the touch is what pages the expert into DRAM. */
        static int ms = -1;
        if (ms < 0) {
            const char *e = getenv("Q4_MISS_STAGE");
            ms = (!e || e[0] != '0') || !q4_expert_is_resident(c);
        }
        /* Hand the bind batch to the stager thread: it finishes the
         * synchronous stage off the critical path and releases the hit
         * protection itself.  The fallback keeps the old inline order. */
        int posted = 0;
        if (nc && ms)
            posted = q4_expert_stage_async(c, layer, cids, nc, gids, ng);
        if (ng && !posted) q4_expert_protect(c, layer, gids, ng, 0);
        q4_prof_moe_io += moe_now() - t1;
        t1 = moe_now();
        /* Publish hits through the mapped arrays; the exec kernels live in
         * the per-layer gB graph (captured or launched by the caller), which
         * removes ~4 host launches per MoE layer from the decode seam.  If
         * staging can't work at all (no arena, alloc failure on the first
         * token) fall back to eager launches permanently. */
        if (moe_staged_exec != 0) {
            if (moe_gpu_stage(g, c, layer, gids, gwt, ng))
                moe_staged_exec = 1;
            else if (moe_staged_exec > 0)
                return -1;      /* a gB graph already contains the exec */
            else
                moe_staged_exec = 0;
        }
        if (moe_staged_exec == 0 &&
            ng && !moe_gpu_exec(g, c, layer, moe_dx, moe_dy, gids, gwt, ng,
                                moe_dh, moe_dh2, moe_d_down, n_ff, moe_nembd))
            return -1;
        q4_prof_moe_gemm += moe_now() - t1;
    }
    if (!disp) {
        /* Dispatch refused mid-flight: hit-expert work is only staged (or,
         * on the permanent eager path, already launched validly); the tail
         * recomputes all routed experts over the same d_y. */
        return 0;
    }
    t0 = moe_now();
    if (nc && !moe_cpx_d && !q4_cpuex_join_tag(slot, moe_eager_tag))
        return -1;
    /* With the pinned seq block the pool join is an in-stream wait kernel
     * inside gB (or queued by gpu_b on the eager tail) — nothing to do here. */
    q4_prof_moe_join += moe_now() - t0;
    return 1;
}

/* GPU part 2 + hit-exec + pool join + residual combine (decode graph gB).
 * Same function serves graph capture and the eager tail — it only issues
 * launches, and every input arrives through mapped/device words. */
bool q4_moe_gpu_b_hc(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     float *d_res, const float *d_inj, uint32_t hc) {
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (moe_staged_exec == 1 &&
        !moe_gpu_exec_staged(g, c, layer, moe_dx, moe_dy, moe_dh, moe_dh2,
                             moe_d_down, n_ff, moe_nembd))
        return false;
    if (moe_cpx_h &&
        !q4_hip_cpx_wait((void *)(moe_cpx_h + 1), (void *)moe_cpx_h,
                         (void *)(moe_cpx_h + 1)))
        return false;
    return q4_hip_moe_final(d_res, moe_dy, moe_dhy, moe_d_sh, moe_d_dot,
                            d_inj, moe_nembd, hc);
}

/* GPU part 2 (eager tail of the graphed path): staged hit-exec, pool join,
 * merge CPU expert output, apply shared expert. */
bool q4_moe_gpu_b(const q4_gguf *g, q4_expert_cache *c, int32_t layer) {
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (moe_staged_exec == 1 &&
        !moe_gpu_exec_staged(g, c, layer, moe_dx, moe_dy, moe_dh, moe_dh2,
                             moe_d_down, n_ff, moe_nembd))
        return false;
    if (moe_cpx_h &&
        !q4_hip_cpx_wait((void *)(moe_cpx_h + 1), (void *)moe_cpx_h,
                         (void *)(moe_cpx_h + 1)))
        return false;
    const uint32_t n_embd = moe_nembd;
    double t0 = moe_now();
    /* GPU axpy reads the pinned host buffer straight over PCIe; skips the
     * explicit H2D copy and its stream sync. */
    if (!q4_hip_axpy(moe_dy, moe_dhy, 1.f, n_embd)) return false;
    q4_prof_moe_merg += moe_now() - t0;
    if (moe_d_dot)
        return q4_hip_axpy_dalpha(moe_dy, moe_d_sh, moe_d_dot, n_embd);
    return q4_hip_axpy(moe_dy, moe_d_sh, 1.f, n_embd);
}

/* Shared-expert tail only (routed experts were merged by the caller). */
static bool moe_shared_merge(void) {
    if (moe_d_dot)
        return q4_hip_axpy_dalpha(moe_dy, moe_d_sh, moe_d_dot, moe_nembd);
    return q4_hip_axpy(moe_dy, moe_d_sh, 1.f, moe_nembd);
}

/* After a "dispatch refused" hostpart: run the routed experts on GPU using
 * the ids/wts already computed, then merge the shared expert. Never
 * captures; called from graphed decode's fallback and from _dev. */
bool q4_moe_gpu_tail(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     float *d_x, float *d_y, float *d_ws) {
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp + 16;         /* ≤64 experts batched */
    float *d_h2 = d_h + 64 * n_ff;
    float *d_down = d_h2 + 64 * n_ff;
    if (!moe_gpu_chain(g, c, layer, d_x, d_y, moe_ids, moe_wts, topk, d_h,
                       d_h2, d_down, n_ff, n_embd))
        return false;
    return moe_shared_merge();
}

/* Routed-expert GEMV chain for ids whose L1 slots already contain the data.
 * Launches on the compute stream; overlaps CPU expert work. Batched: gate,
 * up, fused activation, down and weighted-sum run as five launches for the
 * whole hit set (before: ~4 small launches per expert). */
static bool moe_gpu_exec(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                         const float *d_x, float *d_y, const int32_t *ids,
                         const float *wts, uint32_t n, float *d_h,
                         float *d_h2, float *d_down, uint32_t n_ff,
                         uint32_t n_embd) {
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        return false;
    int arena = q4_expert_cache_arena(c);
    const uint8_t *base = q4_hip_experts_base_a(arena);
    uint64_t stride = q4_hip_experts_stride_a(arena);
    uint32_t na = q4_hip_experts_n_a(arena);
    if (!base || !stride) return false;
    int32_t slots[64];
    float wt[64];
    for (uint32_t k = 0; k < n; k++) {
        int32_t slot = q4_expert_cache_slot(c, layer, ids[k]);
        if (slot < 0 || (uint32_t)slot >= na) return false;
        slots[k] = slot;
        wt[k] = wts[k];
    }
    if (q4_hip_moe_exec(base, stride, slots, wt, n, gate->ggml_type, og,
                        up->ggml_type, ou, down->ggml_type, od, d_x, d_h,
                        d_h2, d_down, d_y, n_ff, gate->ne[0], n_embd))
        return true;
    /* Unsupported weight type: per-expert fallback. */
    const uint8_t *dpacks[64];
    for (uint32_t k = 0; k < n; k++)
        dpacks[k] = base + (size_t)slots[k] * stride;
    int use_fuse = (gate->ggml_type == Q4_T_Q4_K &&
                    up->ggml_type == Q4_T_Q4_K);
    for (uint32_t k = 0; k < n; k++) {
        const uint8_t *dpack = dpacks[k];
        if (use_fuse) {
            if (!q4_hip_gateup_q4k(dpack + og, dpack + ou, d_x, d_h, n_ff,
                                   n_embd))
                return false;
        } else {
            if (!q4_hip_gemv_dd(gate->ggml_type, dpack + og, gate->ne[1],
                                gate->ne[0], d_x, d_h, 1.f))
                return false;
            if (!q4_hip_gemv_dd(up->ggml_type, dpack + ou, up->ne[1],
                                up->ne[0], d_x, d_h2, 1.f))
                return false;
            if (!q4_hip_silu_mul(d_h, d_h2, n_ff)) return false;
        }
        if (!q4_hip_gemv_dd(down->ggml_type, dpack + od, down->ne[1],
                            down->ne[0], d_h, d_down, 1.f))
            return false;
        if (!q4_hip_axpy(d_y, (float *)d_down, wts[k], n_embd)) return false;
    }
    return true;
}

/* Resolve ids→L1 slots and publish slots/weights/count through the mapped
 * staging arrays.  The hit-exec kernels read those words at exec time, so
 * once captured into gB the same launches replay correctly every token. */
static bool moe_gpu_stage(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const int32_t *ids, const float *wts, uint32_t n) {
    (void)g;
    int arena = q4_expert_cache_arena(c);
    if (arena < 0) return false;
    uint32_t na = q4_hip_experts_n_a(arena);
    for (uint32_t k = 0; k < n; k++) {
        int32_t slot = q4_expert_cache_slot(c, layer, ids[k]);
        if (slot < 0 || (uint32_t)slot >= na) return false;
        moe_hslots[k] = slot;
        moe_hwts[k] = wts[k];
    }
    moe_nhits = n;
    return q4_hip_moe_stage(moe_hslots, moe_hwts, n);
}

/* Launch hit-exec reading the staged arrays — identical op sequence whether
 * called inside a gB capture or eagerly on the tail path. */
static bool moe_gpu_exec_staged(const q4_gguf *g, q4_expert_cache *c,
                                int32_t layer, const float *d_x, float *d_y,
                                float *d_h, float *d_h2, float *d_down,
                                uint32_t n_ff, uint32_t n_embd) {
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        return false;
    int arena = q4_expert_cache_arena(c);
    const uint8_t *base = q4_hip_experts_base_a(arena);
    uint64_t stride = q4_hip_experts_stride_a(arena);
    if (!base || !stride) return false;
    if (q4_hip_moe_exec_staged(base, stride, gate->ggml_type, og,
                               up->ggml_type, ou, down->ggml_type, od, d_x,
                               d_h, d_h2, d_down, d_y, n_ff, gate->ne[0],
                               n_embd))
        return true;
    /* Unsupported weight type: per-expert fallback over the staged set. */
    const uint8_t *dpacks[64];
    for (uint32_t k = 0; k < moe_nhits; k++)
        dpacks[k] = base + (size_t)moe_hslots[k] * stride;
    int use_fuse = (gate->ggml_type == Q4_T_Q4_K &&
                    up->ggml_type == Q4_T_Q4_K);
    for (uint32_t k = 0; k < moe_nhits; k++) {
        const uint8_t *dpack = dpacks[k];
        if (use_fuse) {
            if (!q4_hip_gateup_q4k(dpack + og, dpack + ou, d_x, d_h, n_ff,
                                   n_embd))
                return false;
        } else {
            if (!q4_hip_gemv_dd(gate->ggml_type, dpack + og, gate->ne[1],
                                gate->ne[0], d_x, d_h, 1.f))
                return false;
            if (!q4_hip_gemv_dd(up->ggml_type, dpack + ou, up->ne[1],
                                up->ne[0], d_x, d_h2, 1.f))
                return false;
            if (!q4_hip_silu_mul(d_h, d_h2, n_ff)) return false;
        }
        if (!q4_hip_gemv_dd(down->ggml_type, dpack + od, down->ne[1],
                            down->ne[0], d_h, d_down, 1.f))
            return false;
        if (!q4_hip_axpy(d_y, (float *)d_down, moe_hwts[k], n_embd))
            return false;
    }
    return true;
}

/* Eager GPU routed-expert chain: stage misses to L1, run the GEMV chain.
 * Used when the CPU path is off or refused dispatch mid-flight. */
static bool moe_gpu_chain(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const float *d_x, float *d_y, const int32_t *ids,
                          const float *wts, uint32_t topk, float *d_h,
                          float *d_h2, float *d_down, uint32_t n_ff,
                          uint32_t n_embd) {
    double t0 = moe_now();
    if (q4_expert_stage_batch_sync(c, layer, ids, topk, 1) != 0)
        return false;
    q4_hip_copy_wait();
    q4_prof_moe_io += moe_now() - t0;
    t0 = moe_now();
    bool ok = moe_gpu_exec(g, c, layer, d_x, d_y, ids, wts, topk, d_h, d_h2,
                           d_down, n_ff, n_embd);
    q4_prof_moe_gemm += moe_now() - t0;
    return ok;
}

bool q4_moe_layer_dev(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                      int32_t layer, float *d_x, float *d_y, float *d_ws) {
    /* Decode after prefill: a leftover prefetch must not own the io pool.
     * No-op when none is running. */
    q4_expert_pf_join(c);
    if (!g || !st || !c || !d_x || !d_y || !d_ws || !q4_hip_ok()) return false;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (n_exp > 512 || topk > 64) return false;
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp + 16;         /* ≤64 experts batched */
    float *d_h2 = d_h + 64 * n_ff;
    float *d_down = d_h2 + 64 * n_ff;

    if (q4_moe_cpu_active(g)) {
        if (!q4_moe_gpu_a(g, st, layer, d_x, d_y, d_ws)) return false;
        int rc = q4_moe_hostpart(g, c, layer);
        if (rc < 0) return false;
        if (rc > 0) return q4_moe_gpu_b(g, c, layer);
        /* dispatch refused: finish on the GPU with the ids we already have */
        return q4_moe_gpu_tail(g, c, layer, d_x, d_y, d_ws);
    }

    /* ---- original all-GPU path (Q4_CPU_EXPERTS=0 / buffer alloc failed) -- */
    const q4_tensor *t_router, *t_gate_s, *t_up_s, *t_down_s, *t_sgate;
    const uint8_t *d_router = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_inp.weight", &t_router);
    const uint8_t *d_gate_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_shexp.weight", &t_gate_s);
    const uint8_t *d_up_s = moe_devw(g, st, layer,
                                   "blk.%d.ffn_up_shexp.weight", &t_up_s);
    const uint8_t *d_down_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_down_shexp.weight",
                                     &t_down_s);
    (void)moe_devw(g, st, layer, "blk.%d.ffn_gate_inp_shexp.weight",
                   &t_sgate);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    float *d_sh = d_down + n_embd;
    float *d_dot = d_sh + n_embd;
    int32_t ids[64];
    float wts[64];
    if (!q4_hip_gemv_dd(t_router->ggml_type, d_router, t_router->ne[1],
                        t_router->ne[0], d_x, d_rlog, 1.f))
        return false;
    if (!q4_hip_fill(d_y, 0.f, n_embd)) return false;
    {
        double t0 = moe_now();
        if (!q4_hip_softmax_topk(d_rlog, n_exp, topk, ids, wts)) return false;
        q4_prof_moe_d2h += moe_now() - t0;
        q4_prof_moe_topk += moe_now() - t0;
    }
    if (!moe_gpu_chain(g, c, layer, d_x, d_y, ids, wts, topk, d_h, d_h2,
                       d_down, n_ff, n_embd))
        return false;
    if (!q4_hip_gemv_dd(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                        t_gate_s->ne[0], d_x, d_h, 1.f))
        return false;
    if (!q4_hip_gemv_dd(t_up_s->ggml_type, d_up_s, t_up_s->ne[1], t_up_s->ne[0],
                        d_x, d_h2, 1.f))
        return false;
    if (!q4_hip_silu_mul(d_h, d_h2, n_ff)) return false;
    if (!q4_hip_gemv_dd(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                        t_down_s->ne[0], d_h, d_sh, 1.f))
        return false;
    float sg = 1.f;
    int sg_dev = 0;
    const float *d_sgf = (t_sgate && t_sgate->ne[0] == n_embd)
                             ? moe_sgate_f32d(g, st, t_sgate)
                             : NULL;
    if (d_sgf) {
        double t0 = moe_now();
        if (!q4_hip_dot(d_sgf, d_x, d_dot, n_embd))
            return false;
        if (!q4_hip_sigmoid(d_dot, 1)) return false;
        sg_dev = 1;
        q4_prof_moe_sg += moe_now() - t0;
    }
    if (sg_dev)
        return q4_hip_axpy_dalpha(d_y, d_sh, d_dot, n_embd);
    return q4_hip_axpy(d_y, d_sh, sg, n_embd);
}

/* Shared-expert + sgate tail for the n-token device path: gate_s -> silu,
 * up_s -> mul, down_s -> d_sh, then sgate-scaled or plain add into d_y.
 * Extracted so the small-batch hybrid path computes the shared
 * contribution identically. */
/* Q4_SGATE_FUSE (default on): the verify shared-expert gate dot,
 * sigmoid and add run as one kernel (q4_hip_sgate_axpy). */
static int moe_sgate_fuse(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_SGATE_FUSE");
        v = !(e && e[0] == '0');
    }
    return v;
}

static bool moe_dev_n_shared(const q4_gguf *g, const q4_store *st,
                             const q4_tensor *t_gate_s,
                             const q4_tensor *t_up_s,
                             const q4_tensor *t_down_s,
                             const q4_tensor *t_sgate,
                             const uint8_t *d_gate_s, const uint8_t *d_up_s,
                             const uint8_t *d_down_s, const float *d_x,
                             float *d_h, float *d_h2, float *d_sh, float *d_sg,
                             float *d_y, uint32_t n_ff, uint32_t n_embd,
                             uint32_t n_tok) {
    q4_mmvq_hold(d_x);   /* gate and up share one activation quantization */
    const bool gu_ok =
        q4_hip_gemv_dd_n(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                         t_gate_s->ne[0], d_x, d_h, 1.f, n_tok) &&
        q4_hip_gemv_dd_n(t_up_s->ggml_type, d_up_s, t_up_s->ne[1],
                         t_up_s->ne[0], d_x, d_h2, 1.f, n_tok);
    q4_mmvq_release();
    if (!gu_ok ||
        !q4_hip_silu_mul(d_h, d_h2, (uint64_t)n_tok * n_ff) ||
        !q4_hip_gemv_dd_n(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                          t_down_s->ne[0], d_h, d_sh, 1.f, n_tok))
        return false;
    const float *d_sgf = (t_sgate && t_sgate->ne[0] == n_embd)
                             ? moe_sgate_f32d(g, st, t_sgate)
                             : NULL;
    if (d_sgf && moe_sgate_fuse())
        return q4_hip_sgate_axpy(d_y, d_sh, d_sgf, d_x, n_embd, n_tok);
    if (d_sgf) {
        /* one batched dot per token: sg[t] = sigmoid(sgate . x[t]) */
        return q4_hip_gemv_dd_n(Q4_T_F32, (const uint8_t *)d_sgf, 1, n_embd,
                                d_x, d_sg, 1.f, n_tok) &&
               q4_hip_sigmoid(d_sg, n_tok) &&
               q4_hip_axpy_rows(d_y, d_sh, d_sg, n_embd, n_tok);
    }
    return q4_hip_add(d_y, d_y, d_sh, (uint64_t)n_tok * n_embd);
}

/* Small-batch hybrid MoE (2..8 tokens, MTP verify + short prefill): top-k
 * hits run on GPU through the wmoe rows (per-token slices of a device
 * slot/weight array), misses are dispatched to the CPU pool one job per
 * token — instead of the prefill path's synchronous PCIe staging of every
 * miss.  Host scratch is grow-once; device scratch is carved from d_ws. */
static int moe_sb_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MOE_SB");
        v = !(e && e[0] == '0');
    }
    return v;
}

#define MOE_SB_MAXTOK 8u
#define MOE_SB_MAXK 64u

static float *sb_hx, *sb_hy, *sb_hlog;
static int32_t *sb_ids, *sb_cids, *sb_hslots, *sb_mslots, *sb_heids;
static float *sb_wts, *sb_cwt, *sb_hwts, *sb_mwts;
static uint32_t *sb_nc;
static uint32_t sb_off[MOE_SB_MAXTOK + 1], sb_moff[MOE_SB_MAXTOK + 1];
static int sb_jobs[MOE_SB_MAXTOK];
static uint64_t sb_tags[MOE_SB_MAXTOK];

static bool sb_grow(void) {
    static int done;
    if (done) return true;
    sb_hx = malloc(MOE_SB_MAXTOK * 16384 * sizeof(float));
    sb_hy = malloc(MOE_SB_MAXTOK * 16384 * sizeof(float));
    sb_hlog = malloc(MOE_SB_MAXTOK * 512 * sizeof(float));
    sb_ids = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_cids = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_hslots = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_mslots = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_heids = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_wts = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(float));
    sb_cwt = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(float));
    sb_hwts = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(float));
    sb_mwts = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(float));
    sb_nc = malloc(MOE_SB_MAXTOK * sizeof(uint32_t));
    done = sb_hx && sb_hy && sb_hlog && sb_ids && sb_cids && sb_hslots &&
           sb_mslots && sb_heids && sb_wts && sb_cwt && sb_hwts && sb_mwts &&
           sb_nc;
    return done;
}

/* Compute misses where they live: a resident miss maps to
 * a slot in the DMA-registered host image, so the wmoe rows read it over
 * PCIe directly — zero copies, no CPU-pool join, no x D2H, no result H2D.
 * Measured SLOWER than the CPU pool on gfx1100 (17.1 vs 24.2 t/s): the
 * fine-grained dequant reads don't coalesce into efficient PCIe traffic.
 * Two variants of the idea both lost (this and a copy-to-VRAM scratch
 * arena), so it stays opt-in: Q4_SB_MISSGPU=1 enables it. */
static int moe_sb_missgpu(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_SB_MISSGPU");
        v = e && e[0] == '1';
    }
    return v;
}

/* Hit-slot protection: while this call's exec kernels are queued, the io
 * stager must not rebind the L1 slots they read.  We protect the hit ids
 * (filling=1) and release at the next call's boundary — by then the rlog
 * D2H has drained the whole compute stream.  Releasing late is safe
 * (pinned-ish slots for a layer or two), releasing early is not. */
static q4_expert_cache *sb_prot_c;
static int32_t sb_prot_layer = -1;
static int32_t sb_prot_ids[MOE_SB_MAXTOK * MOE_SB_MAXK];
static uint32_t sb_prot_n;

static void sb_prot_release(void) {
    if (sb_prot_c)
        q4_expert_protect(sb_prot_c, sb_prot_layer, sb_prot_ids, sb_prot_n,
                          0);
    sb_prot_c = NULL;
    sb_prot_layer = -1;
    sb_prot_n = 0;
}

static int sb_prof(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_PROFILE") && getenv("Q4_PROFILE")[0] != '0';
    return v;
}
static double sb_p_route, sb_p_topk, sb_p_bound, sb_p_disp, sb_p_enq,
              sb_p_join, sb_p_tail, sb_p_stage;
static uint64_t sb_p_calls, sb_p_tok, sb_p_hit, sb_p_hu, sb_p_miss, sb_p_mu;
#define SB_PT(acc)                                                           \
    do {                                                                     \
        if (sbp) {                                                           \
            double _n = moe_now();                                           \
            acc += _n - _tt;                                                 \
            _tt = _n;                                                        \
        }                                                                    \
    } while (0)

/* Join every outstanding CPU job from the sb dispatch loop (result
 * ignored — used on late-failure paths so no worker still writes sb_hy or
 * holds a pool slot when the caller falls back). */
static void sb_join_all(uint32_t n_tok) {
    for (uint32_t t = 0; t < n_tok; t++)
        if (sb_jobs[t] >= 0) {
            q4_cpuex_join_tag(sb_jobs[t], sb_tags[t]);
            sb_jobs[t] = -1;
        }
}

void q4_moe_sb_prof_print(FILE *f) {
    if (!f || !sb_p_calls) return;
    double c = (double)sb_p_calls;
    fprintf(f, "q4 sb prof: calls %llu  tok %llu  hit %llu (uniq %llu)  "
               "miss %llu (uniq %llu)  | route %.2f  topk %.2f  "
               "boundary %.2f  cpu-disp %.2f  gpu-enq %.2f  join %.2f  "
               "tail %.2f  stage %.2f ms/call\n",
            (unsigned long long)sb_p_calls, (unsigned long long)sb_p_tok,
            (unsigned long long)sb_p_hit, (unsigned long long)sb_p_hu,
            (unsigned long long)sb_p_miss, (unsigned long long)sb_p_mu,
            sb_p_route * 1e3 / c, sb_p_topk * 1e3 / c, sb_p_bound * 1e3 / c,
            sb_p_disp * 1e3 / c, sb_p_enq * 1e3 / c, sb_p_join * 1e3 / c,
            sb_p_tail * 1e3 / c, sb_p_stage * 1e3 / c);
    sb_p_route = sb_p_topk = sb_p_bound = sb_p_disp = sb_p_enq = 0;
    sb_p_join = sb_p_tail = sb_p_stage = 0;
    sb_p_calls = sb_p_tok = sb_p_hit = sb_p_hu = sb_p_miss = sb_p_mu = 0;
}

/* Batched device-routed MoE (2..8 tokens): the whole layer stays on the
 * compute stream — batched router gemv → x D2H (so the dispatcher can
 * feed CPU-miss jobs) → route_dev_n_k does per-token topk + L1 slot
 * split + one mailbox publish → shared expert → one window launch over
 * the hit experts (or the per-token exec when the window is off) →
 * in-stream hostfunc wait for the published seq → hy rows merge.
 * No host reads GPU memory anywhere in the layer, so back-to-back verify
 * layers pipeline instead of draining on the router-logit D2H. */
static int moe_verify_reuse(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_VERIFY_REUSE");
        v = !(e && e[0] == '0');
    }
    return v;
}

static int moe_devn_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_DEVROUTE_N");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* Q4_MERGE_FUSE (default on): the verify MoE leaves its CPU-miss rows
 * for the caller's hc combine to add (one launch less per layer on the
 * post-wait critical path).  Graphs are captured either way, so this is
 * fixed for the process. */
static int moe_merge_defer(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MERGE_FUSE");
        v = !(e && e[0] == '0');
    }
    return v;
}
static const float *moe_hy_pend;
const float *q4_moe_take_hy(void) {
    const float *p = moe_hy_pend;
    moe_hy_pend = NULL;
    return p;
}

static bool moe_dev_n_route(const q4_gguf *g, const q4_store *st,
                            q4_expert_cache *c, int32_t layer, float *d_x,
                            float *d_y, float *d_ws, uint32_t n_tok,
                            uint64_t ws_floats) {
#define DRN_FAIL(x) do { \
        static int _seen[32]; \
        if (getenv("Q4_MTP_DEBUG") && (x) < 32 && !_seen[x]++) \
            fprintf(stderr, "q4: devroute_n fail #%d layer %d ntok %u\n", \
                    (x), (int)layer, n_tok); \
        return false; \
    } while (0)
    if (!moe_devn_on()) DRN_FAIL(0);
    if (!moe_disp_started || !moe_mail_d || !moe_dhyL || !moe_dwant_d)
        DRN_FAIL(1);
    if (c != moe_disp_c) DRN_FAIL(2);
    if (layer < 0 || layer >= moe_disp_nl || layer >= Q4_MAX_LAYER)
        DRN_FAIL(3);
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (!n_exp || !topk || topk > 16 || n_tok > MOE_DEVN_MAXTOK ||
        n_tok * topk > 64 || n_exp > 1024 || n_embd > 16384)
        DRN_FAIL(4);
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, NULL))
        DRN_FAIL(5);
    const uint32_t in_dim = gate->ne[0];
    if (gate->ggml_type != up->ggml_type || in_dim != up->ne[0] ||
        gate->ne[1] != up->ne[1] || in_dim != n_embd ||
        !q4_hip_moe_tok_ok(gate->ggml_type, up->ggml_type, down->ggml_type,
                           in_dim, n_ff))
        DRN_FAIL(6);
    const q4_tensor *t_router, *t_gate_s, *t_up_s, *t_down_s;
    const uint8_t *d_router = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_inp.weight", &t_router);
    const uint8_t *d_gate_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_shexp.weight", &t_gate_s);
    const uint8_t *d_up_s = moe_devw(g, st, layer,
                                   "blk.%d.ffn_up_shexp.weight", &t_up_s);
    const uint8_t *d_down_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_down_shexp.weight", &t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) DRN_FAIL(7);
    (void)moe_devw(g, st, layer, "blk.%d.ffn_gate_inp_shexp.weight",
                   &moe_t_sgate);
    int32_t *devtab = q4_expert_dev_row(c, layer);
    if (!devtab) DRN_FAIL(8);
    int arena = q4_expert_cache_arena(c);
    const uint8_t *base = q4_hip_experts_base_a(arena);
    uint64_t stride = q4_hip_experts_stride_a(arena);
    if (!base || !stride) DRN_FAIL(9);

    /* d_ws floats: rlog | he | he2 (per-token hit scratch, 64 rows each) |
     * dd (wsum fallback, unreachable while tok_ok gates wgv) | h | h2 |
     * sh | sg for the shared expert. */
    float *d_rlog = d_ws;
    float *d_he = d_rlog + (size_t)n_tok * n_exp;
    float *d_he2 = d_he + (size_t)n_tok * 64 * n_ff;
    float *d_dd = d_he2 + (size_t)n_tok * 64 * n_ff;
    float *d_h = d_dd + 64 * n_embd;
    float *d_h2 = d_h + (size_t)n_tok * n_ff;
    float *d_sh = d_h2 + (size_t)n_tok * n_ff;
    float *d_sg = d_sh + (size_t)n_tok * n_embd;
    if ((size_t)(d_sg + n_tok - d_ws) > ws_floats) DRN_FAIL(10);

    /* From here on everything is enqueued; a false return below means an
     * enqueue failed — the caller's fallback recomputes the layer (its own
     * fill zeroes d_y before any accumulation). */
    if (!q4_hip_gemv_dd_n(t_router->ggml_type, d_router, t_router->ne[1],
                          t_router->ne[0], d_x, d_rlog, 1.f, n_tok))
        DRN_FAIL(11);
    q4_moe_vtl_stamp(layer, 2);
    /* The mailbox publish is stream-ordered AFTER this copy — when the
     * dispatcher sees the seq bump every token row has landed. */
    if (!q4_hip_d2h_async(moe_hx, d_x, (size_t)n_tok * n_embd * sizeof(float)))
        DRN_FAIL(12);
    uint8_t *mstg = moe_mstg + (size_t)layer * MOE_MSTG_STRIDE;
    if (!q4_hip_route_dev_n(d_rlog, devtab, moe_des, moe_dew, moe_dnh, mstg,
                            moe_dseq + layer,
                            moe_route_prof ? moe_drseq + layer : NULL,
                            moe_dprof, n_exp, topk, n_tok))
        DRN_FAIL(13);
    if (!q4_hip_d2h_async(moe_mail + layer, mstg, MOE_MAIL_DMA))
        DRN_FAIL(14);
    q4_moe_vtl_stamp(layer, 3);
    if (!q4_hip_fill(d_y, 0.f, (uint64_t)n_tok * n_embd))
        DRN_FAIL(14);
    if (!moe_dev_n_shared(g, st, t_gate_s, t_up_s, t_down_s, moe_t_sgate,
                          d_gate_s, d_up_s, d_down_s, d_x, d_h, d_h2, d_sh,
                          d_sg, d_y, n_ff, n_embd, n_tok))
        DRN_FAIL(15);
    /* Hit exec.  The window stages each expert row once into d_he's first
     * 64 rows (the per-token scratch is n_tok*64, and shared already
     * consumed d_h).  A launch error fails the route so the small-batch
     * path recomputes the layer; a type the window does not cover stays
     * on the per-token exec inside this route. */
    if (n_tok > 1 && moe_verify_reuse() &&
        q4_hip_moe_win_ok(gate->ggml_type, up->ggml_type, down->ggml_type,
                          in_dim, n_ff, n_tok)) {
        static int once;
        if (!once) {
            once = 1;
            fprintf(stderr, "q4 moe: verify window reuses each expert "
                            "across its tokens\n");
        }
        if (!q4_hip_moe_exec_win(base, stride, moe_des, moe_dew, moe_dnh,
                                 n_tok, gate->ggml_type, og, up->ggml_type,
                                 ou, down->ggml_type, od, d_x, d_he, d_y,
                                 n_ff, in_dim, n_embd))
            DRN_FAIL(16);
    } else {
        for (uint32_t t = 0; t < n_tok; t++) {
            if (!q4_hip_moe_exec_dev(base, stride, moe_des + t * 64,
                                     moe_dew + t * 64, moe_dnh + t,
                                     gate->ggml_type, og, up->ggml_type, ou,
                                     down->ggml_type, od,
                                     d_x + (size_t)t * in_dim,
                                     d_he + (size_t)t * 64 * n_ff,
                                     d_he2 + (size_t)t * 64 * n_ff, d_dd,
                                     d_y + (size_t)t * n_embd, n_ff, in_dim,
                                     n_embd))
                DRN_FAIL(16);
        }
    }
    /* Decode prefetch: layer+1's router on this layer's input, after the
     * hits — in a layer whose CPU job is the longer side the GPU would sit
     * in the wait below anyway.  The record carries layer+1's route seq of
     * the previous pass (tseq) so the dispatcher can tell a guess that
     * arrives after layer+1 routed.  Best effort: a failed enqueue only
     * loses this layer's guess. */
    if (moe_dpf > 0 && moe_pred && layer + 1 < moe_disp_nl) {
        const q4_tensor *t_r1;
        const uint8_t *d_r1 = moe_devw(g, st, layer + 1,
                                       "blk.%d.ffn_gate_inp.weight", &t_r1);
        int32_t *dt1 = q4_expert_dev_row(c, layer + 1);
        uint8_t *stg = moe_pstg + (size_t)layer * sizeof(moe_pred_t);
        if (d_r1 && dt1 && t_r1->ne[1] == n_exp &&
            q4_hip_gemv_dd_n(t_r1->ggml_type, d_r1, t_r1->ne[1],
                             t_r1->ne[0], d_x, moe_dplog, 1.f, n_tok) &&
            q4_hip_pred_sel(moe_dplog, dt1, moe_dseq + layer + 1,
                            moe_dpcnt + layer, stg, n_exp, n_tok, moe_dpf_n,
                            moe_dpf_pmin))
            (void)q4_hip_d2h_async((void *)(moe_pred + layer), stg,
                                   sizeof(moe_pred_t));
    }
    moe_mail_t *hmail = moe_mail + layer;
    q4_moe_vtl_stamp(layer, 4);
    if (moe_gdone && moe_hyL_coh) {
        if (!q4_hip_cpx_spin(moe_dseq + layer, &moe_gdone[(size_t)layer * 8]))
            DRN_FAIL(17);
    } else if (!q4_hip_cpx_wait((void *)&hmail->done, (void *)&hmail->seq,
                                moe_cpx_h ? (void *)(moe_cpx_h + 1)
                                          : (void *)&hmail->done))
        DRN_FAIL(17);
    q4_moe_vtl_stamp(layer, 5);
    /* CPU-miss merge: hy rows for tokens with no misses were zeroed by the
     * dispatcher, so a flat add is exact.  Both sides are n_tok contiguous
     * rows — one launch, not one per token (this sits on the critical
     * path, after the CPU wait).  With the merge deferred the caller's
     * hc combine adds the rows instead (q4_moe_take_hy). */
    if (moe_merge_defer())
        moe_hy_pend = moe_dhyL + (size_t)layer * MOE_DEVN_MAXTOK * n_embd;
    else if (!q4_hip_axpy(d_y,
                          moe_dhyL + (size_t)layer * MOE_DEVN_MAXTOK * n_embd,
                          1.f, (uint64_t)n_tok * n_embd))
        DRN_FAIL(18);
    q4_moe_vtl_stamp(layer, 6);
    return true;
#undef DRN_FAIL
}

static bool moe_dev_n_sb(const q4_gguf *g, const q4_store *st,
                         q4_expert_cache *c, int32_t layer, float *d_x,
                         float *d_y, float *d_ws, uint32_t n_tok,
                         uint64_t ws_floats) {
    const int arena = q4_expert_cache_arena(c);
    if (arena < 0 || !sb_grow()) return false;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (!n_exp || !topk || topk > MOE_SB_MAXK || n_exp > 512 ||
        n_embd > 16384)
        return false;
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, NULL))
        return false;
    if (gate->ggml_type != up->ggml_type || gate->ne[0] != up->ne[0] ||
        gate->ne[1] != up->ne[1] ||
        !q4_hip_moe_tok_ok(gate->ggml_type, up->ggml_type, down->ggml_type,
                           gate->ne[0], n_ff))
        return false;
    char nm[Q4_MAX_NAME];
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer);
    const q4_tensor *t_router = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", layer);
    const q4_tensor *t_gate_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", layer);
    const q4_tensor *t_up_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer);
    const q4_tensor *t_down_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    const q4_tensor *t_sgate = q4_find_tensor(g, nm);
    if (!t_router || !t_gate_s || !t_up_s || !t_down_s) return false;
    const uint8_t *d_router = q4_store_dev(st, t_router);
    const uint8_t *d_gate_s = q4_store_dev(st, t_gate_s);
    const uint8_t *d_up_s = q4_store_dev(st, t_up_s);
    const uint8_t *d_down_s = q4_store_dev(st, t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    const uint8_t *abase = q4_hip_experts_base_a(arena);
    uint64_t astride = q4_hip_experts_stride_a(arena);
    if (!abase || !astride) return false;

    /* d_ws layout: rlog | slots | wts | h | h2 | shared out | sg | hy |
     * mslots | mwts */
    float *d_rlog = d_ws;
    int32_t *d_slots = (int32_t *)(d_rlog + (size_t)n_tok * n_exp);
    float *d_wts = (float *)(d_slots + (size_t)n_tok * topk);
    float *d_h = d_wts + (size_t)n_tok * topk;
    float *d_h2 = d_h + (size_t)n_tok * topk * n_ff;
    float *d_sh = d_h2 + (size_t)n_tok * topk * n_ff;
    float *d_sg = d_sh + (size_t)n_tok * n_embd;
    float *d_hy = d_sg + n_tok;
    int32_t *d_mslots = (int32_t *)(d_hy + (size_t)n_tok * n_embd);
    float *d_mwts = (float *)(d_mslots + (size_t)n_tok * topk);
    size_t need = (size_t)(d_mwts + (size_t)n_tok * topk - d_ws);
    if (need > ws_floats) return false;

    int sbp = sb_prof();
    double _tt = sbp ? moe_now() : 0;
    if (!q4_hip_gemv_dd_n(t_router->ggml_type, d_router, t_router->ne[1],
                          t_router->ne[0], d_x, d_rlog, 1.f, n_tok))
        return false;
    {
        double t0 = moe_now();
        /* x is pulled lazily below only if a CPU job actually dispatches. */
        if (!q4_hip_d2h(sb_hlog, d_rlog, (size_t)n_tok * n_exp * sizeof(float)))
            return false;
        q4_prof_moe_d2h += moe_now() - t0;
    }
    SB_PT(sb_p_route);
    {
        double t0 = moe_now();
        for (uint32_t t = 0; t < n_tok; t++) {
            q4_softmax(sb_hlog + t * n_exp, n_exp);
            q4_topk(sb_hlog + t * n_exp, n_exp, topk, sb_ids + t * topk,
                    sb_wts + t * topk, true);
            q4_expert_route_note(c, layer, sb_ids + t * topk, (int)topk);
        }
        q4_prof_moe_topk += moe_now() - t0;
    }
    SB_PT(sb_p_topk);
    /* Boundary with prior work: reap an outstanding CPU job, drain async
     * H2D and the async stager, then the lock-free gpu_slot reads below are
     * safe. */
    if (!q4_cpuex_reap()) return false;
    {
        double t0 = moe_now();
        q4_hip_copy_wait();
        q4_prof_moe_cw += moe_now() - t0;
    }
    q4_expert_uploads_mark(c);
    q4_expert_stage_join(c);
    q4_expert_pf_join(c);
    q4_expert_clear_sticky(c);
    /* The rlog D2H above is a full compute-stream drain: the previous
     * call's exec kernels are done, so its hit-slot protections can go. */
    sb_prot_release();
    SB_PT(sb_p_bound);

    /* Split each token's top-k into GPU hits (token-major prefix) and CPU
     * misses, preserving top-k order within each list. */
    const uint32_t na = q4_hip_experts_n_a(arena);
    uint32_t nhit = 0, nmiss = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        sb_off[t] = nhit;
        uint32_t nc = 0;
        for (uint32_t k = 0; k < topk; k++) {
            int32_t e = sb_ids[t * topk + k];
            float w = sb_wts[t * topk + k];
            int32_t s = (e >= 0 && e < (int32_t)n_exp)
                            ? q4_expert_gpu_slot(c, layer, e)
                            : -1;
            if (s >= 0 && (uint32_t)s < na) {
                sb_hslots[nhit] = s;
                sb_heids[nhit] = e;
                sb_hwts[nhit] = w;
                nhit++;
            } else {
                sb_cids[t * MOE_SB_MAXK + nc] = e;
                sb_cwt[t * MOE_SB_MAXK + nc] = w;
                nc++;
            }
        }
        sb_nc[t] = nc;
        nmiss += nc;
    }
    sb_off[n_tok] = nhit;
    /* Protect this call's hit slots: the exec kernels below read them
     * while queued, so the stager must not rebind them until the stream
     * has provably drained (released at the next call's boundary). */
    if (nhit) {
        memcpy(sb_prot_ids, sb_heids, nhit * sizeof(int32_t));
        q4_expert_protect(c, layer, sb_prot_ids, nhit, 1);
        sb_prot_c = c;
        sb_prot_layer = layer;
        sb_prot_n = nhit;
    }
    q4_prof_moe_ng += nhit;
    q4_prof_moe_nc += nmiss;
    q4_prof_moe_ncall += n_tok;
    /* Distinct hit slots / miss ids for the call (miss union doubles as the
     * stage_async list below). */
    int32_t uni[MOE_SB_MAXTOK * MOE_SB_MAXK];
    uint32_t nu = 0;
    {
        uint32_t hu = 0;
        for (uint32_t i = 0; i < nhit; i++) {
            uint32_t u = 0;
            for (; u < i; u++) if (sb_hslots[u] == sb_hslots[i]) break;
            if (u == i) hu++;
        }
        for (uint32_t t = 0; t < n_tok; t++)
            for (uint32_t k = 0; k < sb_nc[t]; k++) {
                int32_t e = sb_cids[t * MOE_SB_MAXK + k];
                uint32_t u = 0;
                for (; u < nu; u++) if (uni[u] == e) break;
                if (u == nu) uni[nu++] = e;
            }
        if (sbp) {
            sb_p_calls++;
            sb_p_tok += n_tok;
            sb_p_hit += nhit;
            sb_p_hu += hu;
            sb_p_miss += nmiss;
            sb_p_mu += nu;
        }
    }

    /* GPU-miss resolution: the resident image is DMA-registered, so a
     * populated miss maps to a "slot" inside its layer's tight-packed
     * region — the exec kernels read the bytes over PCIe themselves, no
     * staging copy at all.  Unfilled/non-resident experts stay on the
     * CPU pool. */
    const uint8_t *mbase = NULL;
    uint64_t mstride = 0;
    if (nmiss && moe_sb_missgpu()) {
        mbase = q4_expert_res_devbase(c, layer);
        mstride = q4_expert_res_stride(c, layer);
        if (!mstride) mbase = NULL;
    }
    int32_t rslot[MOE_SB_MAXTOK * MOE_SB_MAXK];
    for (uint32_t i = 0; i < nu; i++)
        rslot[i] = mbase ? q4_expert_res_slot(c, layer, uni[i]) : -1;

    /* Split each token's CPU-miss list: entries whose expert lives in the
     * resident image become (resident slot, weight) pairs; the rest are
     * compacted in place for the CPU dispatch below. */
    int32_t upos[512];
    for (uint32_t i = 0; i < 512; i++) upos[i] = -1;
    for (uint32_t i = 0; i < nu; i++)
        if (rslot[i] >= 0) upos[uni[i]] = rslot[i];
    uint32_t ngm = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        sb_moff[t] = ngm;
        uint32_t ncc = 0;
        for (uint32_t k = 0; k < sb_nc[t]; k++) {
            int32_t e = sb_cids[t * MOE_SB_MAXK + k];
            float w = sb_cwt[t * MOE_SB_MAXK + k];
            if (e >= 0 && e < 512 && upos[e] >= 0) {
                sb_mslots[ngm] = upos[e];
                sb_mwts[ngm] = w;
                ngm++;
            } else {
                sb_cids[t * MOE_SB_MAXK + ncc] = e;
                sb_cwt[t * MOE_SB_MAXK + ncc] = w;
                ncc++;
            }
        }
        sb_nc[t] = ncc;
    }
    sb_moff[n_tok] = ngm;

    /* CPU jobs first so the pool crunches while GPU work queues. x is
     * fetched only when a job really dispatches. */
    int need_x = 0;
    for (uint32_t t = 0; t < n_tok; t++)
        if (sb_nc[t]) need_x = 1;
    if (need_x &&
        !q4_hip_d2h(sb_hx, d_x, (size_t)n_tok * n_embd * sizeof(float)))
        return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        sb_jobs[t] = -1;
        if (!sb_nc[t]) {
            memset(sb_hy + (size_t)t * n_embd, 0, n_embd * sizeof(float));
            continue;
        }
        int slot = -1;
        for (int r = 0; r < 400; r++) {
            slot = q4_cpuex_dispatch2(g, c, layer,
                                      sb_hx + (size_t)t * n_embd,
                                      sb_cids + (size_t)t * MOE_SB_MAXK,
                                      sb_cwt + (size_t)t * MOE_SB_MAXK,
                                      sb_nc[t], sb_hy + (size_t)t * n_embd,
                                      &sb_tags[t]);
            if (slot != -1) break;
            (void)q4_cpuex_retire();
            _mm_pause();
        }
        if (slot < 0) {
            /* -1 backpressure that never cleared, or -2 hard failure
             * (k>CPX_MAX_E, pool/malloc fail): without this abort the
             * un-dispatched row stayed stale in sb_hy and the H2D+axpy
             * below would merge garbage into d_y.  d_y is still
             * untouched, so the old path can run. */
            sb_join_all(n_tok);
            sb_prot_release();
            return false;
        }
        sb_jobs[t] = slot;
    }
    SB_PT(sb_p_disp);

    double t_gemm = moe_now();
    if (!q4_hip_fill(d_y, 0.f, (uint64_t)n_tok * n_embd)) {
        sb_join_all(n_tok);
        sb_prot_release();
        return false;
    }
    if (nhit) {
        if (!q4_hip_h2d(d_slots, sb_hslots, nhit * sizeof(int32_t)) ||
            !q4_hip_h2d(d_wts, sb_hwts, nhit * sizeof(float)) ||
            !q4_hip_moe_exec_tok(abase, astride, d_slots, d_wts, sb_off,
                                 n_tok, gate->ggml_type, og, up->ggml_type,
                                 ou, down->ggml_type, od, d_x, d_h, d_h2,
                                 d_y, n_ff, gate->ne[0], n_embd)) {
            sb_join_all(n_tok);
            sb_prot_release();
            return false;
        }
    }
    if (!moe_dev_n_shared(g, st, t_gate_s, t_up_s, t_down_s, t_sgate,
                          d_gate_s, d_up_s, d_down_s, d_x, d_h, d_h2, d_sh,
                          d_sg, d_y,
                          n_ff, n_embd, n_tok)) {
        sb_join_all(n_tok);
        sb_prot_release();
        return false;
    }
    if (ngm) {
        /* No ordering work at all: the exec kernels read the registered
         * host image over PCIe — nothing was enqueued on the copy stream. */
        if (!q4_hip_h2d(d_mslots, sb_mslots, ngm * sizeof(int32_t)) ||
            !q4_hip_h2d(d_mwts, sb_mwts, ngm * sizeof(float)) ||
            !q4_hip_moe_exec_tok(mbase, mstride, d_mslots, d_mwts, sb_moff,
                                 n_tok, gate->ggml_type, og, up->ggml_type,
                                 ou, down->ggml_type, od, d_x, d_h, d_h2,
                                 d_y, n_ff, gate->ne[0], n_embd)) {
            sb_join_all(n_tok);
            sb_prot_release();
            return false;
        }
    }
    q4_prof_moe_gemm += moe_now() - t_gemm;
    SB_PT(sb_p_enq);

    {
        double t0 = moe_now();
        for (uint32_t t = 0; t < n_tok; t++)
            if (sb_jobs[t] >= 0) {
                if (!q4_cpuex_join_tag(sb_jobs[t], sb_tags[t])) {
                    sb_join_all(n_tok);
                    sb_prot_release();
                    return false;
                }
                sb_jobs[t] = -1;
            }
        q4_prof_moe_join += moe_now() - t0;
    }
    SB_PT(sb_p_join);
    if (need_x &&
        (!q4_hip_h2d(d_hy, sb_hy, (size_t)n_tok * n_embd * sizeof(float)) ||
         !q4_hip_axpy(d_y, d_hy, 1.f, (uint64_t)n_tok * n_embd))) {
        sb_join_all(n_tok);
        sb_prot_release();
        return false;
    }

    SB_PT(sb_p_tail);
    /* With MTP on, main-model tokens go through verify instead of
     * hostpart — stage the misses here so the L1 LRU stays warm.  No
     * stream sync needed: the hit slots the queued kernels read are
     * protected, and everything else the stager may rebind is unread
     * by this call.  Protection releases at the next call's boundary,
     * after the rlog D2H has provably drained the compute stream. */
    if (nmiss) {
        static int ms = -1;
        if (ms < 0) {
            const char *e = getenv("Q4_MISS_STAGE");
            ms = (!e || e[0] != '0') || !q4_expert_is_resident(c);
        }
        if (ms) {
            double t0 = moe_now();
            if (nu) q4_expert_stage_async(c, layer, uni, nu, NULL, 0);
            q4_prof_moe_io += moe_now() - t0;
            if (sbp) { double _n = moe_now(); sb_p_stage += _n - _tt; _tt = _n; }
        }
    }
    return true;
}

static int verify_graph_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_VERIFY_GRAPH");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* Q4_CPX_JOBBENCH=1 (diagnostic): synthetic verify CPU-miss jobs on the
 * real pool and DRAM image, before the first verify.  Per layer: 12
 * distinct image-resident experts, 4 tokens x 4 picks (16 pairs).  Cold
 * = fresh random experts each job, warm = the same job again at once. */
static void moe_cpx_jobbench(const q4_gguf *g, q4_expert_cache *c) {
    const uint32_t nl = g->n_layer < Q4_MAX_LAYER ? g->n_layer : Q4_MAX_LAYER;
    const uint32_t n_embd = g->n_embd, n_exp = g->n_expert;
    uint64_t st = 0x2545f4914f6cdd1dull;
#define JB_RND() (st ^= st << 13, st ^= st >> 7, st ^= st << 17, st)
    for (uint32_t i = 0; i < 4 * n_embd; i++)
        moe_hx[i] = ((float)(int32_t)(JB_RND() >> 32) / 2147483648.f);
    double cold = 0, warm = 0;
    uint32_t nj = 0;
    for (int pass = 0; pass < 4; pass++) {
        for (uint32_t L = 0; L < nl; L++) {
            int32_t uid[12];
            uint32_t nu = 0;
            for (int tries = 0; nu < 12 && tries < 10000; tries++) {
                int32_t e = (int32_t)(JB_RND() % n_exp);
                if (!q4_expert_host_ptr(c, (int32_t)L, e)) continue;
                uint32_t k = 0;
                for (; k < nu; k++) if (uid[k] == e) break;
                if (k == nu) uid[nu++] = e;
            }
            if (nu < 12) continue;
            int32_t ids[16];
            float wts[16];
            uint32_t tn[8] = {4, 4, 4, 4};
            /* token t picks uid[3t..3t+3] and uid[(3t+3)%12]: 12 distinct,
             * 4 shared */
            for (uint32_t t = 0; t < 4; t++)
                for (uint32_t k = 0; k < 4; k++) {
                    ids[4 * t + k] = uid[(3 * t + k) % 12];
                    wts[4 * t + k] = 0.1f;
                }
            float *y = moe_hyL + (size_t)L * MOE_DEVN_MAXTOK * moe_nembd;
            for (int rep = 0; rep < 2; rep++) {
                uint64_t tag = 0;
                double t0 = moe_now();
                int slot;
                while ((slot = q4_cpuex_dispatch_win(g, c, (int32_t)L, moe_hx,
                                                     4, ids, wts, tn, y,
                                                     &tag)) == -1)
                    (void)q4_cpuex_retire();
                if (slot < 0) { fprintf(stderr, "q4 jobbench: dispatch %d\n", slot); return; }
                while (!q4_cpuex_tag_done(slot, tag)) _mm_pause();
                (void)q4_cpuex_join_tag(slot, tag);
                double dt = (moe_now() - t0) * 1e6;
                if (pass) { if (rep) warm += dt; else cold += dt; }
            }
            if (pass) nj++;
        }
    }
#undef JB_RND
    if (nj)
        fprintf(stderr, "q4 jobbench: %u jobs  cold %.0f us  warm %.0f us  "
                "(12 experts x 16 pairs)\n", nj, cold / nj, warm / nj);
}

static void verify_capture_fail(float *d_x, float *d_y, float *d_ws) {
    static int logged;
    vg_off = 1;
    vg_x = d_x;
    vg_y = d_y;
    vg_ws = d_ws;
    q4_hip_clear();
    if (!logged) {
        logged = 1;
        fprintf(stderr, "q4 moe: verify graph capture failed; eager\n");
    }
}

void q4_moe_verify_capture(const q4_gguf *g, const q4_store *st,
                           q4_expert_cache *c, float *d_x, float *d_y,
                           float *d_ws, uint64_t ws_floats) {
    if (!verify_graph_on() || vg_off) return;
    if (!g || !st || !c || !d_x || !d_y || !d_ws || !q4_hip_ok()) return;
    if (vg_x == d_x && vg_y == d_y && vg_ws == d_ws && vg_n) return;
    verify_graphs_free();
    if (!q4_moe_dev_active()) (void)q4_moe_dev_route(g, c);
    if (!q4_moe_dev_active()) return;
    q4_hip_moe_scratch_warm();
    uint32_t nl = g->n_layer;
    if (nl > Q4_MAX_LAYER) nl = Q4_MAX_LAYER;
    for (uint32_t L = 0; L < nl; L++) q4_moe_dev_warm(g, st, (int32_t)L);
    uint32_t got = 0;
    for (uint32_t n = 2; n <= 4; n++) {
        for (uint32_t L = 0; L < nl; L++) {
            q4_hip_clear();
            if (!q4_hip_graph_begin()) {
                verify_capture_fail(d_x, d_y, d_ws);
                return;
            }
            int ok = moe_dev_n_route(g, st, c, (int32_t)L, d_x, d_y, d_ws, n,
                                     ws_floats);
            void *ex = q4_hip_graph_end();
            if (!ok || !ex) {
                if (ex) q4_hip_graph_free(ex);
                verify_capture_fail(d_x, d_y, d_ws);
                return;
            }
            vg[L][n] = ex;
            got++;
        }
    }
    vg_x = d_x;
    vg_y = d_y;
    vg_ws = d_ws;
    vg_n = got;
    moe_hy_pend = NULL;   /* the capture pass merges nothing */
    static int announced;
    if (!announced && got) {
        announced = 1;
        fprintf(stderr, "q4 moe: verify graph live (%u)\n", got);
        if (getenv("Q4_CPX_JOBBENCH") && getenv("Q4_CPX_JOBBENCH")[0] == '1')
            moe_cpx_jobbench(g, c);
    }
}

bool q4_moe_layer_dev_n(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                        int32_t layer, float *d_x, float *d_y, float *d_ws,
                        uint32_t n_tok, uint64_t ws_floats) {
    if (n_tok >= 2 && n_tok <= 4 && layer >= 0 && layer < Q4_MAX_LAYER &&
        d_x && d_x == vg_x && d_y == vg_y && d_ws == vg_ws &&
        vg[layer][n_tok]) {
        if (!q4_hip_graph_launch(vg[layer][n_tok])) return false;
        if (moe_merge_defer())
            moe_hy_pend = moe_dhyL +
                          (size_t)layer * MOE_DEVN_MAXTOK * g->n_embd;
        return true;
    }
    if (n_tok <= 1)
        return q4_moe_layer_dev(g, st, c, layer, d_x, d_y, d_ws);
    if (n_tok <= MOE_SB_MAXTOK && n_tok > 1 && moe_devn_on() &&
        g && st && c && d_x && d_y && d_ws && q4_hip_ok() &&
        moe_dev_n_route(g, st, c, layer, d_x, d_y, d_ws, n_tok, ws_floats))
        return true;
    if (n_tok <= MOE_SB_MAXTOK && moe_sb_on() && g && st && c && d_x && d_y &&
        d_ws && q4_hip_ok() &&
        moe_dev_n_sb(g, st, c, layer, d_x, d_y, d_ws, n_tok, ws_floats))
        return true;
    if (!g || !st || !c || !d_x || !d_y || !d_ws || !q4_hip_ok()) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 1); return false; }
    const int arena = q4_expert_cache_arena(c);
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (n_exp > 512 || topk > 10) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 3); return false; }
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 4); return false; }
    char nm[Q4_MAX_NAME];
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer);
    const q4_tensor *t_router = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", layer);
    const q4_tensor *t_gate_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", layer);
    const q4_tensor *t_up_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer);
    const q4_tensor *t_down_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    const q4_tensor *t_sgate = q4_find_tensor(g, nm);
    if (!t_router || !t_gate_s || !t_up_s || !t_down_s) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 5); return false; }
    const uint8_t *d_router = q4_store_dev(st, t_router);
    const uint8_t *d_gate_s = q4_store_dev(st, t_gate_s);
    const uint8_t *d_up_s = q4_store_dev(st, t_up_s);
    const uint8_t *d_down_s = q4_store_dev(st, t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 6); return false; }

    /* Tokens are processed in groups of <=1024 rows (<=Q4_MMQ_GROUP on the
     * MMQ path): with the next layer's experts prefetching on the copy
     * stream, groups only bound the workspace (Gmax assignment rows ->
     * d_h/d_h2/d_pd) while every GEMM queues back to back on the compute
     * stream. */
    /* MMQ is the main model's big-batch path only: the shared wbuf is
     * sized to THIS model's strides, and a second cache (the MTP head's
     * Q8_0 experts) would grow it ~1.2 GiB. Head calls stay off MMQ
     * (arena != 0) so head_est needs no wbuf reservation. A head batch
     * wider than the L1 slot count reads the resident image instead
     * (see use_res below) — it does not grow the slot arena. */
    int mmq_layer = q4_mmq_on() && arena == 0 &&
                    gate->ggml_type == up->ggml_type &&
                    q4_mmq_supported(gate->ggml_type, 2ull * n_ff) &&
                    q4_mmq_supported(down->ggml_type, (uint64_t)n_embd);
    const uint32_t gcap = mmq_layer ? q4_mmq_group() : 1024;
    const uint32_t gsz = n_tok < gcap ? n_tok : gcap;
    const uint32_t n_g = (n_tok + gsz - 1) / gsz;
    const uint32_t Gmax = gsz * topk;
    /* d_ws layout (floats): router logits | ph/ph2 (Gmax*n_ff) | shared out |
     * expert out pd (Gmax*n_embd) | assignment tables | sgate | group meta |
     * device-route tail | MMQ tail (bounds|eid|slot|xq|hq) */
    uint64_t xq_f = 0, hq_f = 0;
    if (mmq_layer) {
        xq_f = (q4_mmq_q8_bytes(Gmax, n_embd) + 15) / 16 * 4;
        hq_f = (q4_mmq_q8_bytes(Gmax, n_ff) + 15) / 16 * 4;
    }
    float *d_rlog = d_ws;
    float *d_h = d_rlog + (size_t)n_tok * n_exp;
    float *d_h2 = d_h + (size_t)Gmax * n_ff;
    float *d_sh = d_h2 + (size_t)Gmax * n_ff;
    float *d_pd = d_sh + (size_t)n_tok * n_embd;
    int32_t *d_atok = (int32_t *)(d_pd + (size_t)Gmax * n_embd);
    int32_t *d_rsrc = d_atok + (size_t)n_tok * topk;
    float *d_rw = (float *)(d_rsrc + (size_t)n_tok * topk);
    float *d_sg = d_rw + (size_t)n_tok * topk;
    int32_t *d_meta = (int32_t *)(d_sg + n_tok);
    /* Device-route tail: ek|stg|stgn stay contiguous so one small D2H
     * lands everything the host still needs (staging lists + counts).
     * ids/wts live on-device; they come back only for the route-hit
     * profile. */
    uint32_t *d_ek = (uint32_t *)(d_meta + 3 * 512 * n_g);
    int32_t *d_stg_d = (int32_t *)(d_ek + (size_t)n_g * 512);
    uint32_t *d_stgn_d = (uint32_t *)(d_stg_d + (size_t)n_g * 512);
    int32_t *d_ids = (int32_t *)(d_stgn_d + n_g);
    float *d_wt = (float *)(d_ids + (size_t)n_tok * topk);
    /* MMQ tail (carved only when this layer can run MMQ; the planner
     * reserves it for every layer while Q4_MMQ is on). */
    int32_t *d_bounds = NULL, *d_eid = NULL, *d_slot = NULL;
    uint8_t *d_xq = NULL, *d_hq = NULL;
    const float *d_end = (const float *)(d_wt + (size_t)n_tok * topk);
    if (mmq_layer) {
        d_bounds = (int32_t *)(d_wt + (size_t)n_tok * topk); /* n_g*(n_exp+1) */
        d_eid = d_bounds + (size_t)n_g * (n_exp + 1);
        d_slot = d_eid + n_exp;
        d_xq = (uint8_t *)(d_slot + n_exp);
        d_hq = d_xq + xq_f * sizeof(float);
        d_end = (const float *)(d_hq + hq_f * sizeof(float));
    }
    size_t need = (size_t)(d_end - d_ws);
    if (need > ws_floats) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 7); return false; }
    double t_rt0 = moe_now();
    if (!q4_hip_gemv_dd_n(t_router->ggml_type, d_router, t_router->ne[1],
                          t_router->ne[0], d_x, d_rlog, 1.f, n_tok))
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 8); return false; }
    q4_prof_moe_rt += moe_now() - t_rt0;
    /* Host block: logits, topk, per-group histograms and assignment tables,
     * stage id lists, batched-GEMM metadata (chunk-sized -> heap).
     * u32d is the contiguous device-route readback (ek|stg|stgn). */
    size_t hfloats = (size_t)n_tok * n_exp + 2u * (size_t)n_tok * topk;
    /* Tail ints: u32d (n_g*1025) then the MMQ host lists
     * hbounds(n_g*(n_exp+1)) | ge(n_exp) | gsl(n_exp). */
    size_t hints = 3u * (size_t)n_tok * topk + 4u * (size_t)n_g * 512 +
                   (size_t)n_g + 3u * (size_t)n_g * 512 +
                   (size_t)n_g * 1025u + (size_t)n_g * (n_exp + 1) +
                   2u * n_exp;
    uint8_t *hbuf = malloc(hfloats * sizeof(float) + hints * sizeof(int32_t));
    if (!hbuf) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 9); return false; }
    float *hlog = (float *)hbuf;
    float *wts = hlog + (size_t)n_tok * n_exp;
    float *rw = wts + (size_t)n_tok * topk;
    int32_t *ids = (int32_t *)(rw + (size_t)n_tok * topk);
    int32_t *a_tok = ids + (size_t)n_tok * topk;
    int32_t *rsrc = a_tok + (size_t)n_tok * topk;
    uint32_t *ek = (uint32_t *)(rsrc + (size_t)n_tok * topk);
    uint32_t *ebase = ek + (size_t)n_g * 512;
    uint32_t *cur = ebase + (size_t)n_g * 512;
    int32_t *stg = (int32_t *)(cur + (size_t)n_g * 512);
    uint32_t *stgn = (uint32_t *)(stg + (size_t)n_g * 512);
    int32_t *meta = (int32_t *)(stgn + n_g);
    uint32_t *u32d = (uint32_t *)(meta + (size_t)n_g * 3 * 512);
    int32_t *hb = (int32_t *)(u32d + (size_t)n_g * 1025);
    int32_t *ge = hb + (size_t)n_g * (n_exp + 1);
    int32_t *gsl = ge + n_exp;
    memset(ek, 0, (size_t)n_g * 512 * sizeof(uint32_t));
    memset(stgn, 0, (size_t)n_g * sizeof(uint32_t));
    /* Device routing: the route/assign kernels write ek|stg|stgn into one
     * contiguous device block; a single sync D2H drains the kernels and
     * lands the ~KBs the host still needs (staging lists + GEMM meta).
     * Q4_PF_ROUTE=0 keeps the host path for bisecting. */
    static int pf_devroute = -1;
    if (pf_devroute < 0) {
        const char *e = getenv("Q4_PF_ROUTE");
        pf_devroute = !(e && e[0] == '0');
    }
    /* Q4_PF_ASSIGN=0 bisects the pair: device route_pf still computes
     * ids/wts, but the host rebuilds the assignment tables from a D2H of
     * d_ids — isolating assign_pf from route_pf on a mismatch. */
    static int pf_devassign = -1;
    if (pf_devassign < 0) {
        const char *e = getenv("Q4_PF_ASSIGN");
        pf_devassign = !(e && e[0] == '0');
    }
    int dev_route = pf_devroute &&
        q4_hip_route_pf(d_rlog, d_ids, d_wt, n_tok, n_exp, topk);
    int host_assign = 0;
    if (dev_route && !pf_devassign) {
        /* host assignment from device picks: ids D2H, then the same
         * serial histogram/cursor build the host path uses. */
        /* Q4_PF_WT_HOST=2 recomputes weights on the host (bit-identical
         * to the full-host path) while keeping device picks — isolates
         * the ~1ulp expf weight difference from pick-order effects. */
        static int pf_wth = -1;
        if (pf_wth < 0) {
            const char *e = getenv("Q4_PF_WT_HOST");
            pf_wth = e && e[0] == '2';
        }
        if (!q4_hip_d2h(ids, d_ids,
                        (size_t)n_tok * topk * sizeof(int32_t)) ||
            !(pf_wth || q4_hip_d2h(wts, d_wt,
                                   (size_t)n_tok * topk * sizeof(float))))
            dev_route = 0;
        else {
            if (pf_wth) {
                /* host-class weights: redo softmax+topk on the same
                 * logits so wts is bit-identical to the host route. */
                if (!q4_hip_d2h(hlog, d_rlog,
                                (size_t)n_tok * n_exp * sizeof(float))) {
                    dev_route = 0;
                    goto host_assign_done;
                }
                int32_t idummy[64];
                for (uint32_t t = 0; t < n_tok; t++) {
                    q4_softmax(hlog + t * n_exp, n_exp);
                    q4_topk(hlog + t * n_exp, n_exp, topk, idummy,
                            wts + t * topk, true);
                }
            }
            for (uint32_t gno = 0; gno < n_g; gno++)
                for (uint32_t e = 0; e < n_exp; e++)
                    ek[gno * 512 + e] = 0;
            for (uint32_t t = 0; t < n_tok; t++)
                for (uint32_t k = 0; k < topk; k++) {
                    int32_t e = ids[t * topk + k];
                    if (e >= 0) ek[(t / gsz) * 512 + e]++;
                }
            for (uint32_t gno = 0; gno < n_g; gno++) {
                uint32_t b = 0;
                for (uint32_t e = 0; e < n_exp; e++) {
                    ebase[gno * 512 + e] = b;
                    b += ek[gno * 512 + e];
                }
                memcpy(cur + gno * 512, ebase + gno * 512,
                       512 * sizeof(uint32_t));
            }
            for (uint32_t t = 0; t < n_tok; t++) {
                uint32_t gno = t / gsz, seg = gno * gsz * topk;
                for (uint32_t k = 0; k < topk; k++) {
                    int32_t e = ids[t * topk + k];
                    if (e < 0 || e >= (int32_t)n_exp) {
                        rsrc[t * topk + k] = -1;
                        rw[t * topk + k] = 0.f;
                        continue;
                    }
                    uint32_t p = cur[gno * 512 + e]++;
                    a_tok[seg + p] = (int32_t)t;
                    rsrc[t * topk + k] = (int32_t)p;
                    rw[t * topk + k] = wts[t * topk + k];
                }
            }
            for (uint32_t gno = 0; gno < n_g; gno++)
                for (uint32_t e = 0; e < n_exp; e++)
                    if (ek[gno * 512 + e])
                        stg[gno * 512 + stgn[gno]++] = (int32_t)e;
            if (!q4_hip_h2d(d_atok, a_tok,
                            (size_t)n_tok * topk * sizeof(int32_t)) ||
                !q4_hip_h2d(d_rsrc, rsrc,
                            (size_t)n_tok * topk * sizeof(int32_t)) ||
                !q4_hip_h2d(d_rw, rw, (size_t)n_tok * topk * sizeof(float)))
                dev_route = 0;
            else
                host_assign = 1; /* ek/stg/stgn/ebase already host-built */
host_assign_done:;
        }
    }
    if (dev_route && pf_devassign)
        dev_route = q4_hip_assign_pf(d_ids, d_wt, d_atok, d_rsrc, d_rw,
                                     d_stg_d, d_stgn_d, d_ek, gsz, topk,
                                     n_exp, n_tok);
    double t_hb0 = moe_now();
    if (dev_route) {
        if (!host_assign) {
            double t0 = moe_now();
            if (!q4_hip_d2h(u32d, d_ek,
                            (size_t)n_g * 1025u * sizeof(uint32_t))) {
                free(hbuf);
                { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 10); return false; }
            }
            q4_prof_moe_d2h += moe_now() - t0;
            memcpy(ek, u32d, (size_t)n_g * 512 * sizeof(uint32_t));
            memcpy(stg, u32d + (size_t)n_g * 512,
                   (size_t)n_g * 512 * sizeof(int32_t));
            memcpy(stgn, u32d + 2 * (size_t)n_g * 512,
                   (size_t)n_g * sizeof(uint32_t));
            for (uint32_t gno = 0; gno < n_g; gno++) {
                uint32_t b = 0;
                for (uint32_t e = 0; e < n_exp; e++) {
                    ebase[gno * 512 + e] = b;
                    b += ek[gno * 512 + e];
                }
            }
        }
        /* Route-hit profile parity: ids come back only when the
         * histogram is live (~80 KB per layer otherwise skipped). */
        if (q4_expert_route_prof_on(c)) {
            if (!host_assign &&
                !q4_hip_d2h(ids, d_ids,
                            (size_t)n_tok * topk * sizeof(int32_t))) {
                free(hbuf);
                return false;
            }
            for (uint32_t t = 0; t < n_tok; t++)
                q4_expert_route_note(c, layer, ids + t * topk, (int)topk);
        }
        /* Q4_PF_ROUTE_CHECK: replay the host route and diff ids/weights/
         * histograms — a debug A/B for device-vs-host route equivalence.
         * Boundary flips (equal-prob ties broken differently) count
         * separately from hard mismatches. */
        static int pf_chk = -1;
        if (pf_chk < 0) pf_chk = getenv("Q4_PF_ROUTE_CHECK") != NULL;
        if (pf_chk && !host_assign) {
            int32_t *h_ids = malloc((size_t)n_tok * topk * sizeof(int32_t) *
                                    2 + (size_t)n_tok * n_exp * 4 +
                                    (size_t)n_tok * topk * 4);
            float *h_lg = (float *)(h_ids + 2 * (size_t)n_tok * topk);
            float *h_wt = h_lg + (size_t)n_tok * n_exp;
            int32_t *d_ids_h = h_ids + (size_t)n_tok * topk;
            uint32_t nset = 0, nord = 0, nmeta = 0;
            double maxdw = 0.0;
            bool okc = h_ids &&
                q4_hip_d2h(d_ids_h, d_ids,
                           (size_t)n_tok * topk * sizeof(int32_t)) &&
                q4_hip_d2h(h_wt, d_wt,
                           (size_t)n_tok * topk * sizeof(float)) &&
                q4_hip_d2h(h_lg, d_rlog,
                           (size_t)n_tok * n_exp * sizeof(float));
            if (okc) {
                float *hwts_chk = malloc((size_t)topk * sizeof(float));
                for (uint32_t t = 0; okc && t < n_tok; t++) {
                    q4_softmax(h_lg + t * n_exp, n_exp);
                    q4_topk(h_lg + t * n_exp, n_exp, topk, h_ids + t * topk,
                            hwts_chk, true);
                    for (uint32_t k = 0; k < topk; k++) {
                        int32_t a = h_ids[t * topk + k],
                                b = d_ids_h[t * topk + k];
                        if (a != b) {
                            /* set-member check: a boundary flip swaps two
                             * near-equal picks rather than producing a
                             * genuinely different pick set */
                            bool inset = false;
                            for (uint32_t j = 0; j < topk; j++)
                                if (d_ids_h[t * topk + j] == a) inset = true;
                            if (inset) nord++; else nset++;
                        }
                        double dw = fabs((double)h_wt[t * topk + k] -
                                         (double)hwts_chk[k]);
                        if (dw > maxdw) maxdw = dw;
                    }
                }
                free(hwts_chk);
                /* host-topk ek histogram vs the device-readback ek in
                 * u32d (ek[] itself was filled from u32d already). */
                memset(ek, 0, (size_t)n_g * 512 * 4);
                for (uint32_t t = 0; t < n_tok; t++)
                    for (uint32_t k = 0; k < topk; k++) {
                        int32_t e = h_ids[t * topk + k];
                        if (e >= 0) ek[(t / gsz) * 512 + e]++;
                    }
                for (uint32_t i = 0; i < n_g * 512; i++)
                    if (ek[i] != ((uint32_t *)u32d)[i]) nmeta++;
                /* restore ek to the device values downstream needs */
                memcpy(ek, u32d, (size_t)n_g * 512 * sizeof(uint32_t));
                /* assignment rows: readback d_atok/d_rsrc/d_rw and check
                 * every pick lands on a row owned by its expert, the row
                 * maps back to the token, rows are hole-free, and rw
                 * carries the kernel weight. */
                {
                    const size_t nn = (size_t)n_tok * topk;
                    int32_t *h_at = malloc(nn * 4), *h_rs = malloc(nn * 4);
                    float *h_rw = malloc(nn * 4);
                    uint32_t *seen = calloc(n_g, 4 * gsz * topk / 32 +
                                            32);
                    uint32_t bad = 0;
                    bool ok2 = h_at && h_rs && h_rw && seen &&
                        q4_hip_d2h(h_at, d_atok, nn * 4) &&
                        q4_hip_d2h(h_rs, d_rsrc, nn * 4) &&
                        q4_hip_d2h(h_rw, d_rw, nn * 4);
                    if (ok2) {
                        for (uint32_t t = 0; t < n_tok; t++) {
                            const uint32_t gno = t / gsz;
                            const uint32_t seg = gno * gsz * topk;
                            for (uint32_t k = 0; k < topk; k++) {
                                const int32_t e = d_ids_h[t * topk + k];
                                const int32_t p = h_rs[t * topk + k];
                                if (e < 0) {
                                    if (p != -1 || h_rw[t * topk + k] != 0.f)
                                        bad++;
                                    continue;
                                }
                                uint32_t base = 0;
                                for (uint32_t ee = 0; ee < (uint32_t)e; ee++)
                                    base += ek[gno * 512 + ee];
                                if (p < (int32_t)base ||
                                    p >= (int32_t)(base +
                                                   ek[gno * 512 + e]) ||
                                    h_at[seg + p] != (int32_t)t ||
                                    h_rw[t * topk + k] !=
                                        h_wt[t * topk + k] ||
                                    (seen[(seg + p) >> 5] >>
                                         ((seg + p) & 31)) & 1u)
                                    bad++;
                                else
                                    seen[(seg + p) >> 5] |=
                                        1u << ((seg + p) & 31);
                            }
                        }
                    }
                    fprintf(stderr,
                            "pfchk L%d: tok %u flips set=%u ord=%u "
                            "ek-diff=%u maxdw=%.3g asgn-bad=%u\n",
                            layer, n_tok, nset, nord, nmeta, maxdw, bad);
                    free(h_at); free(h_rs); free(h_rw); free(seen);
                }
            }
            free(h_ids);
        }
    } else {
        double t0 = moe_now();
        if (!q4_hip_d2h(hlog, d_rlog, (size_t)n_tok * n_exp * sizeof(float))) {
            free(hbuf);
            { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 10); return false; }
        }
        q4_prof_moe_d2h += moe_now() - t0;
        for (uint32_t t = 0; t < n_tok; t++) {
            q4_softmax(hlog + t * n_exp, n_exp);
            q4_topk(hlog + t * n_exp, n_exp, topk, ids + t * topk,
                    wts + t * topk, true);
            q4_expert_route_note(c, layer, ids + t * topk, (int)topk);
        }
        /* Assignment rows are group-local so d_h/d_pd stay group-sized.
         * A group's segment in the concat arrays is g*gsz*topk; rsrc/rw
         * stay token-major. */
        for (uint32_t t = 0; t < n_tok; t++) {
            uint32_t gno = t / gsz;
            for (uint32_t k = 0; k < topk; k++) {
                int32_t e = ids[t * topk + k];
                if (e >= 0 && e < (int32_t)n_exp) ek[gno * 512 + e]++;
            }
        }
        for (uint32_t gno = 0; gno < n_g; gno++) {
            uint32_t b = 0;
            for (uint32_t e = 0; e < n_exp; e++) {
                ebase[gno * 512 + e] = b;
                b += ek[gno * 512 + e];
            }
            memcpy(cur + gno * 512, ebase + gno * 512,
                   512 * sizeof(uint32_t));
        }
        for (uint32_t t = 0; t < n_tok; t++) {
            uint32_t gno = t / gsz;
            uint32_t seg = gno * gsz * topk;
            for (uint32_t k = 0; k < topk; k++) {
                int32_t e = ids[t * topk + k];
                if (e < 0 || e >= (int32_t)n_exp) {
                    rsrc[t * topk + k] = -1;
                    rw[t * topk + k] = 0.f;
                    continue;
                }
                uint32_t p = cur[gno * 512 + e]++;
                a_tok[seg + p] = (int32_t)t;
                rsrc[t * topk + k] = (int32_t)p;
                rw[t * topk + k] = wts[t * topk + k];
            }
        }
        for (uint32_t gno = 0; gno < n_g; gno++) {
            for (uint32_t e = 0; e < n_exp; e++)
                if (ek[gno * 512 + e])
                    stg[gno * 512 + stgn[gno]++] = (int32_t)e;
        }
        if (!q4_hip_h2d(d_atok, a_tok,
                        (size_t)n_tok * topk * sizeof(int32_t)) ||
            !q4_hip_h2d(d_rsrc, rsrc,
                        (size_t)n_tok * topk * sizeof(int32_t)) ||
            !q4_hip_h2d(d_rw, rw, (size_t)n_tok * topk * sizeof(float))) {
            free(hbuf);
            { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 11); return false; }
        }
    }
    q4_prof_moe_topk += moe_now() - t_hb0;
    if (!q4_hip_fill(d_y, 0.f, (uint64_t)n_tok * n_embd)) {
        free(hbuf);
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 12); return false; }
    }

    /* Boundary with the previous layer's prefetch: it has finished enqueuing
     * once joined (its g_copy tail lands at copy_wait below). L-1's sticky
     * set is released here; L-1's expert GEMMs retired at its own sync. */
    double t_pf0 = moe_now();
    q4_expert_pf_join(c);
    q4_expert_clear_sticky(c);
    q4_prof_moe_pf += moe_now() - t_pf0;
    double t_gemm0 = moe_now();
    /* A group that does not fit in L1 thrashes the slot cache (stage,
     * sticky, then moe fail #14 once a sticky slot cannot be evicted).
     * When every expert in those groups is already in the DMA-registered
     * resident image, point the GEMM at that image and skip L1. A group
     * that fits stays on VRAM — the main model (hundreds of slots, one
     * layer's experts) never takes this path. A partial image falls back
     * to staging so an unfilled expert is never read. */
    int use_res = 0;
    const uint8_t *rbase = NULL;
    uint64_t rstride = 0;
    {
        uint32_t nslots = q4_expert_cache_n_slots(c);
        int overflow = 0;
        for (uint32_t gno = 0; gno < n_g; gno++)
            if (stgn[gno] > nslots) overflow = 1;
        if (overflow) {
            rbase = q4_expert_res_devbase(c, layer);
            rstride = q4_expert_res_stride(c, layer);
            if (rbase && rstride) {
                int all = 1;
                for (uint32_t gno = 0; all && gno < n_g; gno++)
                    for (uint32_t i = 0; i < stgn[gno]; i++)
                        if (q4_expert_res_slot(c, layer,
                                               stg[gno * 512 + i]) < 0)
                            all = 0;
                use_res = all;
            }
        }
        if (use_res) {
            static int logged = 0;
            if (!logged) {
                fprintf(stderr,
                        "q4 moe: L%d group exceeds %u L1 slots; "
                        "reading the resident image\n",
                        (int)layer, nslots);
                logged = 1;
            }
        }
    }
    if (!use_res) {
        double t0 = moe_now();
        for (uint32_t gno = 0; gno < n_g; gno++) {
            if (!stgn[gno]) continue;
            if (q4_expert_stage_batch_sync(c, layer, stg + gno * 512,
                                           stgn[gno], 1) != 0) {
                free(hbuf);
                { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 13); return false; }
            }
            /* Protect every already-staged group: a later group's misses
             * must not rebind slots whose meta is built below. The union
             * doubles as the prefetch-eviction shield once pf_begin runs. */
            q4_expert_sticky(c, layer, stg + gno * 512, stgn[gno]);
        }
        double t1 = moe_now();
        q4_hip_copy_wait();
        q4_prof_moe_cw += moe_now() - t1;
        q4_prof_moe_io += moe_now() - t0;
    }
    /* Slots are stable from here on (sticky + prefetch shield), so the
     * batched-GEMM metadata for every group uploads in one shot. */
    q4_expert_pf_hist_add(c, layer, ek, n_g, n_exp);
    memset(meta, 0, (size_t)n_g * 3 * 512 * sizeof(int32_t));
    int all_slotted = 1;
    for (uint32_t gno = 0; gno < n_g; gno++) {
        int32_t *h_sl = meta + gno * 3 * 512;
        int32_t *h_eb = h_sl + 512, *h_ec = h_sl + 1024;
        for (uint32_t i = 0; i < stgn[gno]; i++) {
            int32_t e = stg[gno * 512 + i];
            int32_t slot = use_res ? q4_expert_res_slot(c, layer, e)
                                   : q4_expert_cache_slot(c, layer, e);
            if (slot < 0) { all_slotted = 0; break; }
            h_sl[i] = slot;
            h_eb[i] = (int32_t)ebase[gno * 512 + e];
            h_ec[i] = (int32_t)ek[gno * 512 + e];
        }
        if (!all_slotted) break;
    }
    if (all_slotted &&
        !q4_hip_h2d(d_meta, meta, (size_t)n_g * 3 * 512 * sizeof(int32_t)))
        all_slotted = 0;

    /* MMQ expert bounds for every group in one upload (group-local row
     * ranges; bounds[e] = ebase[gno][e], bounds[n_exp] = group rows). */
    if (mmq_layer) {
        for (uint32_t gno = 0; gno < n_g; gno++) {
            int32_t *hbg = hb + (size_t)gno * (n_exp + 1);
            uint32_t rows = 0;
            for (uint32_t e = 0; e < n_exp; e++) {
                hbg[e] = (int32_t)ebase[(size_t)gno * 512 + e];
                rows += ek[(size_t)gno * 512 + e];
            }
            hbg[n_exp] = (int32_t)rows;
        }
        if (!q4_hip_h2d(d_bounds, hb,
                        (size_t)n_g * (n_exp + 1) * sizeof(int32_t)))
            mmq_layer = 0;
    }
    /* Layer L's experts are sticky; the next layer's set stages underneath
     * the GEMM launches below plus the following attention block. */
    int launched = n_tok >= 256 ? q4_expert_pf_begin(c, layer + 1) : 0;

    const uint8_t *abase = use_res ? rbase : q4_hip_experts_base_a(arena);
    uint64_t astride = use_res ? rstride : q4_hip_experts_stride_a(arena);
    int use_fuse = (gate->ggml_type == Q4_T_Q4_K && up->ggml_type == Q4_T_Q4_K);
    /* MMQ: once per layer, gather every used expert's contiguous
     * gate+up [og,od) and down [od,tot) slices out of their L1 slots into
     * expert-id-indexed buffers — MMQ addresses weights by expert id, not
     * slot.  Slots are stable here (sticky + prefetch shield). */
    uint8_t *d_gu = NULL, *d_dn = NULL;
    int mmq_ready = 0;
    /* Q4_XG: the dedicated fp16 WMMA expert GEMMs (src/q4_xg.cu).  h16
     * (Gmax x n_ff fp16) and the fp16 token rows (n_tok x n_embd) reuse
     * the d_h|d_h2 rows, the tile table the MMQ xq region. */
    int xg_ready = 0;
    uint32_t xg_t0[64], xg_tn[64];
    void *xg_h16 = d_h;
    void *xg_x16 = (uint8_t *)d_h + (size_t)Gmax * n_ff * 2;
    const int xg_layer =
        mmq_layer && q4_xg_on() && n_g <= 64 &&
        q4_xg_gu_type(gate->ggml_type) && q4_xg_dn_type(down->ggml_type) &&
        q4_xg_shape_ok(n_embd, n_ff) &&
        gate->ne[0] == n_embd && gate->ne[1] == n_ff && ou - og == od - ou &&
        (size_t)Gmax * n_ff * 2 + (size_t)n_tok * n_embd * 2 <=
            (size_t)2 * Gmax * n_ff * sizeof(float) &&
        (size_t)n_g * q4_xg_tile_cap(Gmax, n_exp) * 16 <=
            (size_t)xq_f * sizeof(float);
    const uint64_t gu_bytes = od - og, d_bytes = tot - od;
    uint64_t gu_st = gu_bytes, d_st = d_bytes;
    if (mmq_layer && !use_res && all_slotted && abase && astride) {
        uint32_t n_used = 0;
        int slots_ok = 1;
        for (uint32_t e = 0; e < n_exp && slots_ok; e++) {
            int used = 0;
            for (uint32_t gno = 0; gno < n_g && !used; gno++)
                used = ek[(size_t)gno * 512 + e] != 0;
            if (!used) continue;
            int32_t slot = q4_expert_cache_slot(c, layer, (int32_t)e);
            if (slot < 0) { slots_ok = 0; break; }
            ge[n_used] = (int32_t)e;
            gsl[n_used] = slot;
            n_used++;
        }
        if (slots_ok && n_used) {
            /* Buffers are sized at the cross-layer maxima (same helper the
             * planner uses): one allocation, no realloc mid-prefill. */
            uint64_t gu_max = 0, d_max = 0;
            if (getenv("Q4_MMQ_DEBUG")) {
                fprintf(stderr, "mmq L%d gather %u experts\n", (int)layer,
                        n_used);
                fflush(stderr);
            }
            /* MMQ's last k-iteration reads a full span past the row's
             * blocks; on the final weight row that lands past the expert's
             * region — stale bytes there have decoded to NaN fp16 scales.
             * Every gathered expert carries a zeroed pad row (rewritten by
             * the gather each call, so other layers' strides can't leave
             * data in it). */
            const uint64_t gu_rb =
                q4_row_bytes(gate->ggml_type, n_embd);
            const uint64_t d_rb =
                q4_row_bytes(down->ggml_type, n_ff);
            if (!gu_rb || !d_rb) goto skip_pad;
            gu_st = gu_bytes + q4_mmq_pad(gu_rb);
            d_st = d_bytes + q4_mmq_pad(d_rb);
        skip_pad:;

            q4_mmq_wbuf_size(g, &gu_max, &d_max);
            if (xg_layer) {
                /* q4_xg reads the slots directly: build the tile table
                 * (every group, one upload) instead of the gather. */
                int32_t smap[512];
                for (uint32_t e = 0; e < n_exp; e++) smap[e] = -1;
                for (uint32_t i = 0; i < n_used; i++) smap[ge[i]] = gsl[i];
                const uint32_t cap = q4_xg_tile_cap(Gmax, n_exp);
                int32_t *ht = malloc((size_t)n_g * cap * 16);
                uint32_t ntot = 0;
                xg_ready = ht != NULL;
                for (uint32_t gno = 0; gno < n_g && xg_ready; gno++) {
                    const uint32_t nt = q4_xg_tiles(
                        ek + (size_t)gno * 512, ebase + (size_t)gno * 512,
                        smap, n_exp, ht + (size_t)ntot * 4, cap);
                    xg_ready = nt || !stgn[gno];
                    xg_t0[gno] = ntot;
                    xg_tn[gno] = nt;
                    ntot += nt;
                }
                if (xg_ready && ntot)
                    xg_ready = q4_hip_h2d(d_xq, ht, (size_t)ntot * 16);
                free(ht);
                if (xg_ready)
                    xg_ready = q4_xg_cvt(d_x, xg_x16, (uint64_t)n_tok * n_embd);
            } else if (gu_max >= gu_st && d_max >= d_st &&
                q4_mmq_wbuf(gu_max, d_max, n_exp, &d_gu, &d_dn) &&
                q4_hip_h2d(d_eid, ge, (size_t)n_used * sizeof(int32_t)) &&
                q4_hip_h2d(d_slot, gsl, (size_t)n_used * sizeof(int32_t)) &&
                q4_mmq_gather(abase, astride, og, gu_bytes, od, d_bytes,
                              gu_st, d_st, d_eid, d_slot, n_used,
                              d_gu, d_dn))
                mmq_ready = 1;
        }
    }
    for (uint32_t gno = 0; gno < n_g; gno++) {
        uint32_t t0 = gno * gsz;
        uint32_t gtn = n_tok - t0 < gsz ? n_tok - t0 : gsz;
        uint32_t seg = gno * gsz * topk;
        if (!stgn[gno]) continue;
        /* Batched path: one kernel launch per tensor over the whole expert
         * group instead of ~5 launches per expert. */
        int batched = all_slotted && abase != NULL && astride != 0;
        if (xg_ready) {
            const int32_t *dt = (const int32_t *)d_xq + (size_t)xg_t0[gno] * 4;
            int ok = q4_xg_gu(gate->ggml_type, abase, astride, og, ou, n_embd,
                              n_ff, dt, xg_tn[gno], xg_x16, d_atok + seg,
                              xg_h16) &&
                     q4_xg_dn(down->ggml_type, abase, astride, od, n_ff,
                              n_embd, dt, xg_tn[gno], xg_h16, d_pd);
            if (!ok) {
                fprintf(stderr, "q4 moe: xg l%d g%u failed\n", (int)layer,
                        gno);
                free(hbuf);
                return false;
            }
        } else if (mmq_ready) {
            static int mmqd = -1;
            if (mmqd < 0) mmqd = getenv("Q4_MMQ_DEBUG") != NULL;
            const int32_t *dbnd = d_bounds + (size_t)gno * (n_exp + 1);
            uint32_t rows = 0, maxr = 0;
            for (uint32_t e = 0; e < n_exp; e++) {
                uint32_t k = ek[(size_t)gno * 512 + e];
                rows += k;
                if (k > maxr) maxr = k;
            }
            /* GU [1280,2560] output lands across the contiguous d_h|d_h2
             * region (ld 2*n_ff); swiglu folds it into H at the front of
             * d_pd, and the down GEMM then overwrites d_pd — all ordered
             * on the compute stream.  Failure drops to the old path for
             * this and later groups (per-expert rewrite of d_h/d_pd). */
            if (mmqd && layer == 0 && gno == 0) {
                fprintf(stderr,
                        "mmq ptr xq=%p+%zu hq=%p+%zu bnd=%p+%d pd=%p+%zu h=%p+%zu atok=%p\n",
                        (void *)d_xq, (size_t)xq_f * 4, (void *)d_hq,
                        (size_t)hq_f * 4,
                        (void *)d_bounds, (int)(n_g * (n_exp + 1) * 4),
                        (void *)d_pd, (size_t)Gmax * n_embd * 4,
                        (void *)d_h, (size_t)2 * Gmax * n_ff * 4,
                        (void *)d_atok);
                fflush(stderr);
            }
            if (mmqd) { fprintf(stderr, "mmq L%d G%u q-gu r%u\n", (int)layer, gno, rows); fflush(stderr); }
            int mok =
                q4_mmq_quantize(d_x, d_atok + seg, d_xq, gate->ggml_type,
                                n_embd, rows);
            if (mmqd) { fprintf(stderr, "mmq L%d G%u m-gu %s\n", (int)layer, gno, mok ? "ok" : "ERR"); fflush(stderr); }
            mok = mok &&
                q4_mmq_mul_id(d_gu, gate->ggml_type, 2ull * n_ff, n_embd,
                              gu_st, n_exp, d_xq, dbnd, maxr, rows,
                              d_h, 2ull * n_ff);
            if (mmqd) { fprintf(stderr, "mmq L%d G%u swiglu %s\n", (int)layer, gno, mok ? "ok" : "ERR"); fflush(stderr); }
            mok = mok &&
                q4_mmq_swiglu(d_h, d_pd, rows, n_ff);
            if (mmqd) { fprintf(stderr, "mmq L%d G%u q-h %s\n", (int)layer, gno, mok ? "ok" : "ERR"); fflush(stderr); }
            mok = mok &&
                q4_mmq_quantize(d_pd, NULL, d_hq, down->ggml_type, n_ff,
                                rows);
            if (mmqd) { fprintf(stderr, "mmq L%d G%u m-dn %s\n", (int)layer, gno, mok ? "ok" : "ERR"); fflush(stderr); }
            mok = mok &&
                q4_mmq_mul_id(d_dn, down->ggml_type, n_embd, n_ff, d_st,
                              n_exp, d_hq, dbnd, maxr, rows, d_pd, n_embd);
            if (!mok) {
                fprintf(stderr, "q4 moe: mmq l%d g%u failed, fallback\n",
                        (int)layer, gno);
                mmq_ready = 0;
            }
            batched = mok;
        } else if (batched) {
            const int32_t *gm = d_meta + gno * 3 * 512;
            const uint32_t *d_eb = (const uint32_t *)gm + 512;
            const uint32_t *d_ec = (const uint32_t *)gm + 1024;
            uint32_t n_st = stgn[gno];
            /* WMMA path first: fused gate+up+silu writing d_h directly,
             * then the same tiled kernel for down. Unsupported types (or
             * Q4_WMMA=0) make the externs return false -> moe_pf fallback. */
            int fused = 0;
            int ok = 0;
            if (n_tok >= 64 && gate->ggml_type == up->ggml_type &&
                gate->ne[0] == up->ne[0] && gate->ne[1] == up->ne[1])
                fused = ok = q4_hip_moe_wmma2(gate->ggml_type, abase, astride,
                                    og, ou, gm, d_eb, d_ec, d_atok + seg, d_x,
                                    d_h, gate->ne[1], gate->ne[0], n_st);
            if (!ok && gate->ggml_type == up->ggml_type &&
                gate->ne[0] == up->ne[0] && gate->ne[1] == up->ne[1])
                ok = q4_hip_moe_pf2(gate->ggml_type, abase, astride, og,
                                    ou, gm, d_eb, d_ec, d_atok + seg, d_x,
                                    d_h, d_h2, gate->ne[1], gate->ne[0],
                                    n_st);
            else if (!ok)
                ok = q4_hip_moe_pf(gate->ggml_type, abase, astride, og,
                                   gm, d_eb, d_ec, d_atok + seg, d_x, d_h,
                                   gate->ne[1], gate->ne[0], n_st) &&
                     q4_hip_moe_pf(up->ggml_type, abase, astride, ou,
                                   gm, d_eb, d_ec, d_atok + seg, d_x, d_h2,
                                   up->ne[1], up->ne[0], n_st);
            if (ok && !fused) {
                uint32_t A_g = ebase[gno * 512 + stg[gno * 512 + n_st - 1]] +
                               ek[gno * 512 + stg[gno * 512 + n_st - 1]];
                ok = q4_hip_moe_act(d_h, d_h2, (uint64_t)A_g * n_ff);
            }
            int down_w = 0;
            if (ok && n_tok >= 64)
                down_w = q4_hip_moe_wmma(down->ggml_type, abase, astride, od,
                                         gm, d_eb, d_ec, NULL, d_h, d_pd,
                                         down->ne[1], down->ne[0], n_st);
            if (ok && !down_w)
                ok = q4_hip_moe_pf(down->ggml_type, abase, astride, od,
                                   gm, d_eb, d_ec, NULL, d_h, d_pd,
                                   down->ne[1], down->ne[0], n_st);
            batched = ok;
        }
        if (!batched) {
            /* fall through to the per-expert loop on unsupported types */
            for (uint32_t e = 0; e < n_exp; e++) {
                uint32_t K = ek[gno * 512 + e];
                if (!K) continue;
                uint32_t base = ebase[gno * 512 + e];
                const uint8_t *dpack = NULL;
                if (use_res) {
                    int32_t rs = q4_expert_res_slot(c, layer, (int32_t)e);
                    if (rs >= 0)
                        dpack = rbase + (uint64_t)rs * rstride;
                } else {
                    int32_t slot = q4_expert_cache_slot(c, layer, (int32_t)e);
                    if (slot < 0) {
                        /* Lost a slot race (next-layer prefetch evicted a
                         * sticky bind, or a deferred devroute refill left the
                         * map unbound): re-stage this one expert and retry
                         * once instead of failing the whole prefill. */
                        int32_t one = (int32_t)e;
                        q4_expert_stage_batch_sync(c, layer, &one, 1, 1);
                        q4_hip_copy_wait();
                        slot = q4_expert_cache_slot(c, layer, (int32_t)e);
                    }
                    if (slot >= 0)
                        dpack = q4_hip_experts_dev_a(arena, (uint32_t)slot);
                }
                if (!dpack) {
                    free(hbuf);
                    { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 14); return false; }
                }
                float *ph = d_h + (size_t)base * n_ff;
                float *ph2 = d_h2 + (size_t)base * n_ff;
                float *pd = d_pd + (size_t)base * n_embd;
                const int32_t *which = d_atok + seg + base;
                /* Resident Q8 experts are read over PCIe. gemv re-fetches
                 * the weight row once per token; the reuse kernel dequants
                 * it once. VRAM-resident experts stay on the bit-stable
                 * gemv (use_res is off when the group fits in L1). */
                int q8res = use_res && gate->ggml_type == Q4_T_Q8_0 &&
                            up->ggml_type == Q4_T_Q8_0;
                if (use_fuse) {
                    if (!q4_hip_gateup_q4k_n_which(dpack + og, dpack + ou,
                                                   d_x, ph, n_ff, n_embd, K,
                                                   which)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 15); return false; }
                    }
                } else if (q8res) {
                    if (!q4_hip_q8_reuse(dpack + og, gate->ne[1], gate->ne[0],
                                         d_x, which, ph, 1.f, K) ||
                        !q4_hip_q8_reuse(dpack + ou, up->ne[1], up->ne[0],
                                         d_x, which, ph2, 1.f, K)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #16q\n"); return false; }
                    }
                    if (!q4_hip_silu_mul(ph, ph2, (uint64_t)K * n_ff)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #16d\n"); return false; }
                    }
                } else {
                    if (!q4_hip_gemv_dd_n_which(gate->ggml_type, dpack + og,
                                                gate->ne[1], gate->ne[0],
                                                d_x, ph, 1.f, K, which)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #16a ty=%u K=%u nr=%llu nc=%llu\n", gate->ggml_type, K, (unsigned long long)gate->ne[1], (unsigned long long)gate->ne[0]); return false; }
                    }
                    if (!q4_hip_gemv_dd_n_which(up->ggml_type, dpack + ou,
                                                up->ne[1], up->ne[0],
                                                d_x, ph2, 1.f, K, which)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #16c\n"); return false; }
                    }
                    if (!q4_hip_silu_mul(ph, ph2, (uint64_t)K * n_ff)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #16d\n"); return false; }
                    }
                }
                /* ph rows are assignment-indexed: plain batched GEMV to pd.
                 * Down has no gather, so which is NULL. */
                if (q8res && down->ggml_type == Q4_T_Q8_0) {
                    if (!q4_hip_q8_reuse(dpack + od, down->ne[1], down->ne[0],
                                         ph, NULL, pd, 1.f, K)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #17q\n"); return false; }
                    }
                } else if (!q4_hip_gemv_dd_n(down->ggml_type, dpack + od,
                                      down->ne[1], down->ne[0], ph, pd,
                                      1.f, K)) {
                    free(hbuf);
                    { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 17); return false; }
                }
            }
        }
        /* dy[t] += sum_k rw * pd[rsrc]: group-local rows, token-window view. */
        if (!q4_hip_expert_reduce(d_y + (size_t)t0 * n_embd, d_pd,
                                  d_rsrc + (size_t)t0 * topk,
                                  d_rw + (size_t)t0 * topk, gtn, topk,
                                  n_embd)) {
            free(hbuf);
            { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 18); return false; }
        }
        if (getenv("Q4_MMQ_DEBUG")) {
            fprintf(stderr, "mmq L%d G%u reduce done\n", (int)layer, gno);
            fflush(stderr);
        }
    }
    q4_hip_sync();
    q4_prof_moe_gemm += moe_now() - t_gemm0;
    if (!launched) {
        /* No prefetch to shield: restore the decode-time protection of just
         * the last token's experts (the all-used set must not linger). */
        q4_expert_clear_sticky(c);
        const int32_t *last_ids = ids + (size_t)(n_tok - 1) * topk;
        if (dev_route) {
            /* Host ids are only filled on the fallback path (or for the
             * route-hits profile) — pull the one row we need. */
            if (!q4_hip_d2h(ids + (size_t)(n_tok - 1) * topk,
                            d_ids + (size_t)(n_tok - 1) * topk,
                            (size_t)topk * sizeof(int32_t))) {
                free(hbuf);
                return false;
            }
        }
        q4_expert_sticky(c, layer, last_ids, topk);
    }
    /* With a prefetch in flight the all-used sticky stays: it is released
     * at the next layer's join, after these GEMMs have long retired. */
    double t_sh0 = moe_now();
    if (getenv("Q4_MMQ_DEBUG")) {
        fprintf(stderr, "mmq L%d pre-shared\n", (int)layer);
        fflush(stderr);
    }
    if (!moe_dev_n_shared(g, st, t_gate_s, t_up_s, t_down_s, t_sgate,
                          d_gate_s, d_up_s, d_down_s, d_x, d_h, d_h2, d_sh,
                          d_sg, d_y,
                          n_ff, n_embd, n_tok)) {
        free(hbuf);
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 19); return false; }
    }
    q4_prof_moe_sh += moe_now() - t_sh0;
    free(hbuf);
    return true;
}
bool q4_moe_layer_store(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                        int32_t layer, const float *x, float *y) {
    if (!g || !st || !c || !x || !y) return false;
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp.weight", layer);
    const q4_tensor *t_router = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    const q4_tensor *t_sgate = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_shexp.weight", layer);
    const q4_tensor *t_gate_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_up_shexp.weight", layer);
    const q4_tensor *t_up_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_down_shexp.weight", layer);
    const q4_tensor *t_down_s = q4_find_tensor(g, n);
    if (!t_router || !t_gate_s || !t_up_s || !t_down_s) return false;
    return moe_body(g, c, layer, x, y, q4_store_get(st, t_router), t_router,
                    q4_store_dev(st, t_router), q4_store_get(st, t_sgate),
                    t_sgate, q4_store_get(st, t_gate_s), t_gate_s,
                    q4_store_dev(st, t_gate_s), q4_store_get(st, t_up_s),
                    t_up_s, q4_store_dev(st, t_up_s), q4_store_get(st, t_down_s),
                    t_down_s, q4_store_dev(st, t_down_s));
}

/* Q4_VTL: after a verify pass has synced, fold this pass's stamps into
 * per-cycle sums (all layers).  Columns: see q4_moe_vtl_print. */
#define VTL_NCOL 13
static double vtl_sum[VTL_NCOL];
static unsigned long long vtl_n;
static double vtl_late[2];              /* CPU-late / GPU-late excess */
static unsigned long long vtl_late_n[2];
static uint32_t vtl_nt;                 /* rows of the pass being stamped */
static double vtl_nt_ms[9];
static unsigned long long vtl_nt_n[9], vtl_nt_miss[9];
static unsigned long long vtl_miss0;
void q4_moe_vtl_rows(uint32_t nt) { vtl_nt = nt; }
static unsigned long long vtl_req_pass;      /* pass index in the request */
static double vtl_ix_ms[3];
static unsigned long long vtl_ix_n[3], vtl_ix_miss[3], vtl_ix_rows[3];
void q4_moe_vtl_collect(int nl) {
    if (moe_vtl <= 0 || nl <= 0 || nl > Q4_MAX_LAYER) {
        static int warned;
        if (!warned++) fprintf(stderr, "q4 vtl: off (%d, nl %d)\n", moe_vtl, nl);
        return;
    }
    double off = q4_hip_vtl_calib();
    double g[Q4_MAX_LAYER * 8];
    if (!q4_hip_vtl_read(g, (uint32_t)nl * 8u, off)) return;
    double acc[VTL_NCOL] = {0};
    for (int L = 0; L < nl; L++) {
        const double *s = g + L * 8;
        const double *h = moe_vtl_h[L];
        if (!s[0] || !s[7] || !s[2] || !s[5] || !h[0]) {
            static int warned;
            if (!warned++)
                fprintf(stderr, "q4 vtl: layer %d missing stamp "
                        "(%g %g %g %g %g)\n", L, s[0], s[7], s[2], s[5], h[0]);
            return;
        }
        acc[0] += s[1] - s[0];                 /* pre: hc, attn, hcF */
        acc[1] += s[2] - s[1];                 /* router */
        acc[2] += s[3] - s[2];                 /* publish copies */
        acc[3] += h[0] - s[3];                 /* dispatcher notices */
        acc[4] += h[1] - h[0];                 /* dispatch */
        acc[5] += h[2] - h[1];                 /* CPU job */
        acc[6] += h[3] - h[2];                 /* join -> done stamp */
        acc[7] += s[4] - s[3];                 /* GPU hits (+shared) */
        double rel = h[3] > s[4] ? h[3] : s[4];
        acc[8] += s[5] - rel;                  /* wait release latency */
        acc[9] += s[6] - s[5];                 /* merge */
        acc[10] += s[7] - s[6];                /* combine */
        if (L + 1 < nl) acc[11] += g[(L + 1) * 8] - s[7]; /* gap */
        /* which side the wait release waited for: CPU job done (h[3])
         * vs the GPU hits (s[4]) — the CPU-late excess bounds what any
         * miss-side saving can buy before the hits become the wall */
        if (h[3] > s[4]) { vtl_late[0] += h[3] - s[4]; vtl_late_n[0]++; }
        else { vtl_late[1] += s[4] - h[3]; vtl_late_n[1]++; }
    }
    acc[12] = g[(nl - 1) * 8 + 7] - g[0];
    for (int i = 0; i < VTL_NCOL; i++) vtl_sum[i] += acc[i];
    vtl_n++;
    if (vtl_nt && vtl_nt <= 8) {
        unsigned long long m = atomic_load(&moe_dev_nmiss);
        {
            const unsigned long long pi = vtl_req_pass++;
            const int b = pi < 10 ? 0 : pi < 30 ? 1 : 2;
            vtl_ix_ms[b] += acc[12];
            vtl_ix_n[b]++;
            vtl_ix_miss[b] += m - vtl_miss0;
            vtl_ix_rows[b] += vtl_nt;
        }
        vtl_nt_ms[vtl_nt] += acc[12];
        vtl_nt_n[vtl_nt]++;
        vtl_nt_miss[vtl_nt] += m - vtl_miss0;
        vtl_miss0 = m;
    }
}
void q4_moe_vtl_print(FILE *f) {
    if (!vtl_n || !f) return;
    static const char *nm[VTL_NCOL] = {
        "pre", "router", "publish", "notice", "dispatch", "cpujob",
        "joindone", "gpuhits", "release", "merge", "combine", "gap",
        "total"};
    fprintf(f, "  vtl ms/pass (%llu):", vtl_n);
    for (int i = 0; i < VTL_NCOL; i++)
        fprintf(f, " %s %.2f", nm[i], vtl_sum[i] * 1e3 / (double)vtl_n);
    fprintf(f, "\n");
    fprintf(f, "  vtl wait side: cpu-late %.1f layers excess %.2f ms/pass, "
               "gpu-late %.1f layers excess %.2f ms/pass\n",
            (double)vtl_late_n[0] / (double)vtl_n,
            vtl_late[0] * 1e3 / (double)vtl_n,
            (double)vtl_late_n[1] / (double)vtl_n,
            vtl_late[1] * 1e3 / (double)vtl_n);
    memset(vtl_late, 0, sizeof(vtl_late));
    memset(vtl_late_n, 0, sizeof(vtl_late_n));
    for (int r = 1; r <= 8; r++)
        if (vtl_nt_n[r])
            fprintf(f, "  vtl rows %d: %llu passes  %.2f ms/pass  %.1f misses/pass\n",
                    r, vtl_nt_n[r], vtl_nt_ms[r] * 1e3 / (double)vtl_nt_n[r],
                    (double)vtl_nt_miss[r] / (double)vtl_nt_n[r]);
    static const char *ixn[3] = {"0-9", "10-29", "30+"};
    for (int b = 0; b < 3; b++)
        if (vtl_ix_n[b])
            fprintf(f, "  vtl cycles %s: %llu passes  %.2f ms/pass  %.1f misses/pass  %.2f rows\n",
                    ixn[b], vtl_ix_n[b], vtl_ix_ms[b] * 1e3 / (double)vtl_ix_n[b],
                    (double)vtl_ix_miss[b] / (double)vtl_ix_n[b],
                    (double)vtl_ix_rows[b] / (double)vtl_ix_n[b]);
    memset(vtl_ix_ms, 0, sizeof(vtl_ix_ms));
    memset(vtl_ix_n, 0, sizeof(vtl_ix_n));
    memset(vtl_ix_miss, 0, sizeof(vtl_ix_miss));
    memset(vtl_ix_rows, 0, sizeof(vtl_ix_rows));
    vtl_req_pass = 0;
    memset(vtl_nt_ms, 0, sizeof(vtl_nt_ms));
    memset(vtl_nt_n, 0, sizeof(vtl_nt_n));
    memset(vtl_nt_miss, 0, sizeof(vtl_nt_miss));
    if (moe_rep_n)
        fprintf(f, "  cpx repeat: first %.0f us  second %.0f us  (%llu jobs)\n",
                moe_rep_us1 / (double)moe_rep_n,
                moe_rep_us2 / (double)moe_rep_n, moe_rep_n);
    moe_rep_us1 = moe_rep_us2 = 0;
    moe_rep_n = 0;
    memset(vtl_sum, 0, sizeof(vtl_sum));
    vtl_n = 0;
}
