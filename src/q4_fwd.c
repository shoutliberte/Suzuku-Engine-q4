#include "q4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double q4_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int q4_prof(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_PROFILE") && getenv("Q4_PROFILE")[0] != '0';
    return v;
}

double q4_prof_sec[8]; /* 0 ple 1 hc_attn 2 qsa 3 comb1 4 hc_ffn 5 moe 6 comb2 7 gdn */
static void sec_add(int i, double t0) { q4_prof_sec[i] += q4_now() - t0; }
#define SEC_T0 double _t0 = q4_prof() ? q4_now() : 0
#define SEC_ADD(i) do { if (q4_prof()) { q4_hip_sync(); sec_add((i), _t0); } _t0 = q4_now(); } while (0)

struct q4_sess {
    const q4_gguf *g;
    q4_store *st;
    q4_expert_cache *ex;
    q4_ple *ple;
    float *res; /* hc_dim */
    float *gdn_S;
    float *gdn_conv;
    uint32_t n_gdn;
    uint32_t gdn_ix[Q4_MAX_LAYER];
    float *qsa_k, *qsa_v;
    uint32_t n_qsa;
    uint32_t qsa_ix[Q4_MAX_LAYER];
    uint32_t n_kv, max_kv;
    float *ple_conv;
    float *ple_kw;      /* dequantized ple_conv1d (any gguf type, once) */
    int32_t ple_hist[8];
    int32_t pos;
    /* Image cells (q4_sess_set_embd_rows): cell c < emb_rows_n with a
     * non-NULL emb_rows[c] embeds that row instead of its token. */
    const float *const *emb_rows;
    uint32_t emb_rows_n;
    float *ws;
    float *d_res, *d_ws, *d_gdn_S, *d_gdn_conv, *d_logits, *d_ple_conv, *d_emb;
    float *d_logits_n;   /* MTP verify tail: up to 8 rows of logits (lazy) */
    float *d_ple_kw;            /* device copy of ple_kw (any gguf type) */
    float *d_qsa_k, *d_qsa_v;
    /* Indexer raw keys (fp16) and pooled block keys (fp32, norm+RoPE). */
    void *d_idx_raw;
    float *d_idx_pool;
    float *d_idx_score;
    int32_t *d_idx_sel;
    float *d_idx_q;
    float *d_idx_k;
    uint32_t idx_dim, idx_ratio, idx_kblk, idx_width, max_blocks;
    int kv_kind; /* 0 f32, 1 f16, 2 q8 */
    uint32_t kv_bpt;
    /* KV paging (q4_sess_set_kv_paging): the QSA KV and indexer key caches
     * are VA-only regions mapped up to kv_cap positions; kv_donor asks the
     * engine for pool chunks (expert slots handed over) when they grow. */
    int kv_paged;
    uint32_t kv_cap;
    bool (*kv_donor)(void *ctx, uint32_t chunks);
    void *kv_donor_ctx;
    float *h_logits;
    float *d_pref, *d_mixb, *d_yb, *d_injb, *d_pws;
    float *d_ple_conv2;         /* ping-pong for the batched conv state */
    uint32_t pref_chunk;
    uint64_t pws_floats;
    /* L1 lending (q4_expert_lend_*): while a prompt runs, d_pref..d_pws
     * point into the borrowed arena tail (lent=1); otherwise at the
     * permanent 8-row verify set m_*.  lend_mode: this session never owns
     * d_pref..d_pws (close frees m_* instead). */
    int lend_mode, lent;
    float *m_pref, *m_mixb, *m_yb, *m_injb, *m_pws;
    uint64_t m_pws_floats;
    void (*pref_hook)(uint32_t layer, uint32_t n_layer, void *u);
    void *pref_hook_u;
    /* ---- MTP verify support (see q4_sess_prefill_tokens) ---- */
    int mtp_stash;      /* set while a verify batch runs */
    int mtp_n;          /* tokens in the current verify batch */
    double vprof_enq;   /* Q4_VPROF: when prefill_hip finished enqueueing */
    double vprof_tick;  /* Q4_VPROF: devroute tick (deferred refills) */
    int mtp_max;        /* stash capacity (n_max from q4_sess_mtp_prep) */
    uint32_t n_kv0;     /* n_kv before the verify batch */
    int32_t pos0;       /* pos before the verify batch */
    float *d_gdn_S_snap, *d_gdn_conv_snap; /* pre-batch GDN shadows (VRAM) */
    float **d_ple_snap; /* per-token PLE conv snapshots (VRAM) */
    int32_t *ple_hist_snap; /* per-token PLE n-gram history (host) */
    /* Per (gdn layer, token) qkv/z/beta/alpha kept ON DEVICE as D2D copies
     * (async, no host sync): [qkv max*conv_ch | z max*nvh_dv | beta | alpha]. */
    float *d_gdn_stash;
    uint32_t gdn_stash_row;
    float *d_gdn_replay; /* core scratch for the commit replay */
    /* PLE gather pipeline (Q4_PLE_AHEAD, default on): up to two outstanding
     * gather jobs — one launched by q4_sess_ple_ahead for the next chunk,
     * one at the top of prefill_hip for the current chunk.  ple_chunk_d
     * consumes a job only on an exact (toks content, n, hist) match; any
     * mismatch leaves the job live for a later chunk and falls back to the
     * synchronous gather, bit-identically. */
    struct ple_job {
        struct q4_sess *s;
        pthread_t th;
        int live;          /* spawned and not yet consumed */
        int ok;
        uint64_t seq;      /* eviction order for stale jobs */
        int n;
        const int32_t *toks;   /* hint: the caller's id pointer */
        int32_t *toks_copy;    /* job-owned copy (lifetime-safe) */
        int32_t hist[8];       /* ple_hist this gather assumed */
        int32_t *rows;
        float *emb;            /* host-pinned when possible */
        int emb_pinned;
        size_t cap;            /* token capacity of rows/emb/toks_copy */
    } ple_jobs[2];
    /* Decode graphs: per-layer hipGraphExec (void* to keep this file C).
     * QSA layers run dense until the sparse indexer locks in, then the
     * graph is recaptured in sparse mode (gA_sp). The PLE layer stays
     * eager (host round-trips). */
    void *gA[Q4_MAX_LAYER];  /* attn + moe_gpu_a (devroute: whole layer) */
    void *gB[Q4_MAX_LAYER];  /* moe_gpu_b + hc_combine (unused in devroute) */
    void *g_head;            /* head minus host argmax */
    int graph_off;           /* Q4_GRAPH=0 or capture failure */
    int devroute;            /* -1 untried, 0 unavailable, 1 live */
    uint8_t gA_sp[Q4_MAX_LAYER]; /* gA was captured with sparse QSA active */
};

void q4_sess_set_pref_hook(q4_sess *s,
                           void (*cb)(uint32_t layer, uint32_t n_layer, void *u),
                           void *u) {
    if (!s) return;
    s->pref_hook = cb;
    s->pref_hook_u = u;
}

static bool embed_tok(q4_sess *s, int32_t tok, float *emb);
static bool prefill_hip(q4_sess *s, const int32_t *toks, int n, q4_ahead *ahead);

static const q4_tensor *T(const q4_gguf *g, const char *fmt, int layer) {
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), fmt, layer);
    return q4_find_tensor(g, n);
}

static bool mm_d(const q4_store *st, const q4_tensor *t, const float *d_x,
                 float *d_y) {
    if (!t || t->n_dims < 2 || !d_x || !d_y) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    return q4_hip_gemv_dd(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y, 1.f);
}

static bool mm_d_n(const q4_store *st, const q4_tensor *t, const float *d_x,
                   float *d_y, uint32_t n) {
    if (!t || t->n_dims < 2 || !d_x || !d_y || n == 0) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    return q4_hip_gemv_dd_n(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y, 1.f, n);
}

/* mm_d with the silu epilogue fused into the GEMV (warp path); falls back
 * to gemv + scale_silu where unsupported. */
static bool mm_d_silu(const q4_store *st, const q4_tensor *t, const float *d_x,
                      float *d_y, float act) {
    if (!t || t->n_dims < 2 || !d_x || !d_y) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    if (q4_hip_gemv_dd_act(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y, 1.f,
                           act))
        return true;
    return mm_d(st, t, d_x, d_y) &&
           q4_hip_scale_silu(d_y, act, t->ne[1]);
}

/* Batched hc down: silu folded into the gemv when the narrow bf16 kernel
 * applies, otherwise the two-launch sequence. */
static bool mm_d_n_silu(const q4_store *st, const q4_tensor *t, const float *d_x,
                        float *d_y, float act, uint32_t n) {
    if (!t || t->n_dims < 2 || !d_x || !d_y || n == 0) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    if (q4_hip_gemv_dd_n_act(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y,
                             1.f, act, n))
        return true;
    return mm_d_n(st, t, d_x, d_y, n) &&
           q4_hip_scale_silu(d_y, act, (uint64_t)n * t->ne[1]);
}

static const float *f32d(const q4_store *st, const q4_tensor *t) {
    return t ? (const float *)q4_store_dev(st, t) : NULL;
}

static bool mm(const q4_store *st, const q4_tensor *t, const float *x, float *y) {
    if (!t || t->n_dims < 2) return false;
    uint64_t ncols = t->ne[0], nrows = t->ne[1];
    const uint8_t *d = q4_store_dev(st, t);
    if (d && q4_hip_ok())
        return q4_hip_gemv_dev(t->ggml_type, d, nrows, ncols, x, y, 1.f);
    const uint8_t *h = q4_store_get(st, t);
    if (!h) return false;
    if (q4_hip_ok())
        return q4_hip_gemv(t->ggml_type, h, nrows, ncols, x, y, 1.f);
    memset(y, 0, (size_t)nrows * sizeof(float));
    return q4_gemv(t->ggml_type, h, nrows, ncols, x, y, 1.f);
}

static const float *f32w(const q4_store *st, const q4_tensor *t) {
    return t ? (const float *)q4_store_get(st, t) : NULL;
}

/* ple_conv1d may be any gguf type (F32 in IQ3E, F16 in GSQ) — f32w-style
 * raw access is F32-only, so dequantize once into s->ple_kw. */
static const float *ple_conv_w(q4_sess *s, int layer) {
    const q4_gguf *g = s->g;
    if (s->ple_kw) return s->ple_kw;
    const q4_tensor *ct = T(g, "blk.%d.ple_conv1d.weight", layer);
    if (!ct || ct->n_dims < 2) return NULL;
    const uint64_t nr = ct->ne[1], nc = ct->ne[0];
    const uint64_t rb = q4_row_bytes(ct->ggml_type, nc);
    const uint8_t *raw = q4_store_get(s->st, ct);
    if (!raw || !rb) return NULL;
    float *kw = (float *)malloc(nr * nc * sizeof(float));
    if (!kw) return NULL;
    for (uint64_t r = 0; r < nr; r++) {
        if (!q4_dequant_row(ct->ggml_type, raw + r * rb, nc, kw + r * nc)) {
            free(kw);
            return NULL;
        }
    }
    s->ple_kw = kw;
    return kw;
}


static void grouped_rms(const float *x, const float *w, float *y, uint32_t n_embd,
                        uint32_t hc, float eps) {
    for (uint32_t c = 0; c < hc; c++)
        q4_rms_norm(x + c * n_embd, w ? w + c * n_embd : NULL, y + c * n_embd,
                    n_embd, eps);
}

static bool hc_mix(q4_sess *s, const float *res, float *mixed, float *inject,
                   int layer, bool attn) {
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    const q4_tensor *wn = T(g, attn ? "blk.%d.hc_attn_norm.weight" : "blk.%d.hc_ffn_norm.weight", layer);
    const q4_tensor *wd = T(g, attn ? "blk.%d.hc_attn_down.weight" : "blk.%d.hc_ffn_down.weight", layer);
    const q4_tensor *wu = T(g, attn ? "blk.%d.hc_attn_up.weight" : "blk.%d.hc_ffn_up.weight", layer);
    const q4_tensor *wi = T(g, attn ? "blk.%d.hc_attn_inject.weight" : "blk.%d.hc_ffn_inject.weight", layer);
    float *xn = s->ws;
    grouped_rms(res, f32w(s->st, wn), xn, n_embd, hc, g->rms_eps ? g->rms_eps : 1e-6f);
    float *lo = xn + hc_dim;
    if (!mm(s->st, wd, xn, lo)) return false;
    const float inv_hc = 1.0f / (float)hc;
    for (uint32_t i = 0; i < g->hc_rank; i++) lo[i] *= inv_hc;
    q4_silu(lo, g->hc_rank);
    float *gate = lo + g->hc_rank;
    if (!mm(s->st, wu, lo, gate)) return false;
    q4_sigmoid(gate, hc_dim);
    float *gated = gate + hc_dim;
    for (uint32_t i = 0; i < hc_dim; i++) gated[i] = xn[i] * gate[i];
    memset(mixed, 0, n_embd * sizeof(float));
    for (uint32_t c = 0; c < hc; c++)
        for (uint32_t i = 0; i < n_embd; i++) mixed[i] += gated[c * n_embd + i];
    for (uint32_t i = 0; i < n_embd; i++) mixed[i] *= inv_hc;
    if (inject && wi) {
        if (!mm(s->st, wi, xn, inject)) return false;
    }
    return true;
}

static void hc_combine(float *res, const float *block, const float *inject,
                       uint32_t n_embd, uint32_t hc) {
    const float inv = 1.0f / (float)hc;
    for (uint32_t c = 0; c < hc; c++) {
        float w = 2.0f / (1.0f + expf(-inject[c] * inv));
        for (uint32_t i = 0; i < n_embd; i++)
            res[c * n_embd + i] += block[i] * w;
    }
}

static void rope_head(float *x, uint32_t n_rot, int32_t pos, float base,
                      const int32_t *sec) {
    int pair = 0;
    int32_t ps[4] = {pos, pos, pos, 0};
    uint32_t n_pairs = n_rot / 2;
    for (int s = 0; s < 4; s++) {
        int n = sec[s];
        for (int k = 0; k < n && (uint32_t)pair < n_pairs; k++, pair++) {
            float freq = powf(base, -(float)pair / (float)n_pairs);
            float ang = (float)ps[s] * freq;
            float c = cosf(ang), si = sinf(ang);
            /* NEOX pairs (p, p + n_rot/2), as the GPU kernels */
            float x0 = x[pair], x1 = x[pair + n_pairs];
            x[pair] = x0 * c - x1 * si;
            x[pair + n_pairs] = x0 * si + x1 * c;
        }
    }
}

static bool gdn_one(q4_sess *s, int layer, const float *x, float *y) {
    const q4_gguf *g = s->g;
    const uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    const uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    const uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    const uint32_t d_v = (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h;
    const uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    const uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    const uint32_t gi = s->gdn_ix[layer];
    float *qkv = s->ws;
    float *z = qkv + conv_ch;
    if (!mm(s->st, T(g, "blk.%d.attn_qkv.weight", layer), x, qkv)) return false;
    if (!mm(s->st, T(g, "blk.%d.attn_gate.weight", layer), x, z)) return false;
    float *beta = z + n_v_h * d_v;
    float *alpha = beta + n_v_h;
    if (!mm(s->st, T(g, "blk.%d.ssm_beta.weight", layer), x, beta)) return false;
    if (!mm(s->st, T(g, "blk.%d.ssm_alpha.weight", layer), x, alpha)) return false;
    q4_sigmoid(beta, n_v_h);

    const float *cw = f32w(s->st, T(g, "blk.%d.ssm_conv1d.weight", layer));
    float *cstate = s->gdn_conv + (size_t)gi * (ksz - 1) * conv_ch;
    float *conv_in = alpha + n_v_h;
    memcpy(conv_in, cstate, (ksz - 1) * conv_ch * sizeof(float));
    memcpy(conv_in + (ksz - 1) * conv_ch, qkv, conv_ch * sizeof(float));
    float *conv_out = conv_in + ksz * conv_ch;
    for (uint32_t c = 0; c < conv_ch; c++) {
        float acc = 0;
        for (uint32_t k = 0; k < ksz; k++)
            acc += cw[k + c * ksz] * conv_in[k * conv_ch + c];
        conv_out[c] = acc / (1.0f + expf(-acc)); /* silu */
    }
    memmove(cstate, conv_in + conv_ch, (ksz - 1) * conv_ch * sizeof(float));

    float *q = conv_out;
    float *k = q + d_k * n_k_h;
    float *v = k + d_k * n_k_h;
    for (uint32_t h = 0; h < n_k_h; h++) {
        q4_l2_norm(q + h * d_k, d_k, g->rms_eps ? g->rms_eps : 1e-6f);
        q4_l2_norm(k + h * d_k, d_k, g->rms_eps ? g->rms_eps : 1e-6f);
    }
    /* This GGUF tiles V heads for grouped broadcast:
     * HF grouped [G0_v0..v{r-1}, G1_v0..] → tiled [G0_v0, G1_v0, ..., G0_v1, ...].
     * Pair V[vh] with K[vh % n_k_h], not vh / gqa. */
    const float qscale = 1.0f / sqrtf((float)d_k);
    const float *A = f32w(s->st, T(g, "blk.%d.ssm_a", layer));
    const float *dt = f32w(s->st, T(g, "blk.%d.ssm_dt.bias", layer));
    float *Sbase = s->gdn_S + (size_t)gi * n_v_h * d_v * d_v;
    float *core = v + d_v * n_v_h;
    for (uint32_t vh = 0; vh < n_v_h; vh++) {
        uint32_t kh = n_k_h ? (vh % n_k_h) : 0;
        const float *qh = q + kh * d_k;
        const float *khv = k + kh * d_k;
        const float *vhv = v + vh * d_v;
        float g_log = A[vh] * q4_softplus(alpha[vh] + dt[vh]);
        float gv = expf(g_log);
        float b = beta[vh];
        float *S = Sbase + (size_t)vh * d_v * d_v; /* S[i][col] row-major i*d + col */
        float *yhv = core + vh * d_v;
        for (uint32_t col = 0; col < d_v; col++) {
            float kv = 0;
            for (uint32_t i = 0; i < d_k; i++) kv += S[i * d_v + col] * khv[i];
            float delta = (vhv[col] - gv * kv) * b;
            float att = 0;
            for (uint32_t i = 0; i < d_k; i++) {
                S[i * d_v + col] = gv * S[i * d_v + col] + khv[i] * delta;
                att += S[i * d_v + col] * qh[i];
            }
            yhv[col] = att * qscale;
        }
    }
    const float *nw = f32w(s->st, T(g, "blk.%d.ssm_norm.weight", layer));
    for (uint32_t vh = 0; vh < n_v_h; vh++) {
        float *h = core + vh * d_v;
        q4_rms_norm(h, nw, h, d_v, g->rms_eps ? g->rms_eps : 1e-6f);
        float *zh = z + vh * d_v;
        for (uint32_t i = 0; i < d_v; i++)
            h[i] *= 1.0f / (1.0f + expf(-zh[i]));
    }
    return mm(s->st, T(g, "blk.%d.ssm_out.weight", layer), core, y);
}

static bool qsa_one(q4_sess *s, int layer, const float *x, float *y) {
    const q4_gguf *g = s->g;
    const uint32_t n_head = g->n_head ? g->n_head : 24;
    const uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    const uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    const uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    float *qg = s->ws;
    if (!mm(s->st, T(g, "blk.%d.attn_q.weight", layer), x, qg)) return false;
    float *kcur = qg + n_head * hd * 2;
    float *vcur = kcur + n_kvh * hd;
    if (!mm(s->st, T(g, "blk.%d.attn_k.weight", layer), x, kcur)) return false;
    if (!mm(s->st, T(g, "blk.%d.attn_v.weight", layer), x, vcur)) return false;
    const float *qn = f32w(s->st, T(g, "blk.%d.attn_q_norm.weight", layer));
    const float *kn = f32w(s->st, T(g, "blk.%d.attn_k_norm.weight", layer));
    float *Q = vcur + n_kvh * hd;
    float *G = Q + n_head * hd;
    for (uint32_t h = 0; h < n_head; h++) {
        memcpy(Q + h * hd, qg + h * hd * 2, hd * sizeof(float));
        memcpy(G + h * hd, qg + h * hd * 2 + hd, hd * sizeof(float));
        q4_rms_norm(Q + h * hd, qn, Q + h * hd, hd, g->rms_eps ? g->rms_eps : 1e-6f);
        rope_head(Q + h * hd, n_rot, s->pos, g->rope_freq_base ? g->rope_freq_base : 1e7f,
                  g->rope_sections);
    }
    for (uint32_t h = 0; h < n_kvh; h++) {
        q4_rms_norm(kcur + h * hd, kn, kcur + h * hd, hd, g->rms_eps ? g->rms_eps : 1e-6f);
        rope_head(kcur + h * hd, n_rot, s->pos, g->rope_freq_base ? g->rope_freq_base : 1e7f,
                  g->rope_sections);
    }
    uint32_t qi = s->qsa_ix[layer];
    uint32_t t = s->n_kv;
    if (t >= s->max_kv) return false;
    memcpy(s->qsa_k + ((size_t)qi * s->max_kv + t) * n_kvh * hd, kcur,
           n_kvh * hd * sizeof(float));
    memcpy(s->qsa_v + ((size_t)qi * s->max_kv + t) * n_kvh * hd, vcur,
           n_kvh * hd * sizeof(float));
    const uint32_t n_all = t + 1;
    const float scale = 1.0f / sqrtf((float)hd);
    float *attn = G + n_head * hd;
    const uint32_t gqa = n_head / n_kvh;
    for (uint32_t h = 0; h < n_head; h++) {
        uint32_t kh = h / gqa;
        float *sc = attn + h * n_all;
        float m = -INFINITY;
        for (uint32_t p = 0; p < n_all; p++) {
            const float *kk = s->qsa_k + ((size_t)qi * s->max_kv + p) * n_kvh * hd +
                              kh * hd;
            float dot = 0;
            for (uint32_t i = 0; i < hd; i++) dot += Q[h * hd + i] * kk[i];
            sc[p] = dot * scale;
            if (sc[p] > m) m = sc[p];
        }
        float sum = 0;
        for (uint32_t p = 0; p < n_all; p++) {
            sc[p] = expf(sc[p] - m);
            sum += sc[p];
        }
        float inv = sum > 0 ? 1.f / sum : 0;
        float *oh = attn + n_head * n_all + h * hd;
        memset(oh, 0, hd * sizeof(float));
        for (uint32_t p = 0; p < n_all; p++) {
            float w = sc[p] * inv;
            const float *vv = s->qsa_v + ((size_t)qi * s->max_kv + p) * n_kvh * hd +
                              kh * hd;
            for (uint32_t i = 0; i < hd; i++) oh[i] += w * vv[i];
        }
        for (uint32_t i = 0; i < hd; i++)
            oh[i] *= 1.0f / (1.0f + expf(-G[h * hd + i]));
    }
    float *flat = attn + n_head * n_all;
    return mm(s->st, T(g, "blk.%d.attn_output.weight", layer), flat, y);
}

static bool ple_one(q4_sess *s, int layer, float *res, int32_t token) {
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    int32_t rows[64];
    if (!q4_ple_hash(s->ple, &token, 1, s->ple_hist, rows)) return false;
    float *emb = s->ws;
    if (!q4_ple_gather(s->ple, rows, g->ple_n_heads, emb)) return false;
    float *key = emb + g->ple_n_heads * g->n_embd_ple;
    float *val = key + hc_dim;
    if (!mm(s->st, T(g, "blk.%d.ple_key.weight", layer), emb, key)) return false;
    if (!mm(s->st, T(g, "blk.%d.ple_value.weight", layer), emb, val)) return false;
    float *kn = s->ws + 8 * hc_dim;
    grouped_rms(key, f32w(s->st, T(g, "blk.%d.ple_norm_key.weight", layer)), kn,
                n_embd, hc, g->rms_eps);
    float *qn = kn + hc_dim;
    grouped_rms(res, f32w(s->st, T(g, "blk.%d.ple_norm_query.weight", layer)), qn,
                n_embd, hc, g->rms_eps);
    float gate[8];
    const float isqrt = 1.0f / sqrtf((float)n_embd);
    for (uint32_t c = 0; c < hc; c++) {
        float dot = 0;
        for (uint32_t i = 0; i < n_embd; i++)
            dot += kn[c * n_embd + i] * qn[c * n_embd + i];
        dot *= isqrt;
        float mag = sqrtf(fabsf(dot) < 1e-6f ? 1e-6f : fabsf(dot));
        gate[c] = 1.0f / (1.0f + expf(-(copysignf(mag, dot))));
    }
    float *gated = qn + hc_dim;
    for (uint32_t c = 0; c < hc; c++)
        for (uint32_t i = 0; i < n_embd; i++)
            gated[c * n_embd + i] = val[i] * gate[c];
    float *cn = gated + hc_dim;
    grouped_rms(gated, f32w(s->st, T(g, "blk.%d.ple_norm_conv.weight", layer)), cn,
                n_embd, hc, g->rms_eps);
    const uint32_t kern = g->ple_conv_kernel ? g->ple_conv_kernel : 4;
    const uint32_t dil = g->ple_ngram ? g->ple_ngram : 3;
    const uint32_t hist = (kern - 1) * dil;
    const float *kw = ple_conv_w(s, layer);
    if (!kw) return false;
    memmove(s->ple_conv, s->ple_conv + hc_dim, hist * hc_dim * sizeof(float));
    memcpy(s->ple_conv + hist * hc_dim, cn, hc_dim * sizeof(float));
    float *co = cn + hc_dim;
    memset(co, 0, hc_dim * sizeof(float));
    for (uint32_t k = 0; k < kern; k++) {
        uint32_t start = hist - (kern - 1 - k) * dil;
        const float *src = s->ple_conv + start * hc_dim;
        for (uint32_t c = 0; c < hc_dim; c++)
            co[c] += kw[k + c * kern] * src[c];
    }
    q4_silu(co, hc_dim);
    for (uint32_t i = 0; i < hc_dim; i++) res[i] += gated[i] + co[i];
    return true;
}

static bool hc_mix_d(q4_sess *s, const float *d_res, float *d_mixed, float *d_inj,
                     int layer, bool attn, bool normed) {
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    const q4_tensor *wn = T(g, attn ? "blk.%d.hc_attn_norm.weight" : "blk.%d.hc_ffn_norm.weight", layer);
    const q4_tensor *wd = T(g, attn ? "blk.%d.hc_attn_down.weight" : "blk.%d.hc_ffn_down.weight", layer);
    const q4_tensor *wu = T(g, attn ? "blk.%d.hc_attn_up.weight" : "blk.%d.hc_ffn_up.weight", layer);
    const q4_tensor *wi = T(g, attn ? "blk.%d.hc_attn_inject.weight" : "blk.%d.hc_ffn_inject.weight", layer);
    float *xn = s->d_ws;
    float *lo = xn + hc_dim;
    float *gate = lo + g->hc_rank;
    float *gated = gate + hc_dim;
    if (!normed &&
        !q4_hip_grouped_rms(d_res, f32d(s->st, wn), xn, n_embd, hc,
                            g->rms_eps ? g->rms_eps : 1e-6f)) {
        fprintf(stderr, "q4: hcmix%d rms wn=%p\n", attn, (void *)f32d(s->st, wn));
        return false;
    }
    /* down and inject both read xn: one multi-seg launch, the down rows
     * carrying the scale+silu epilogue. */
    const uint8_t *wdd = wd ? q4_store_dev(s->st, wd) : NULL;
    const uint8_t *wid = (d_inj && wi) ? q4_store_dev(s->st, wi) : NULL;
    bool hcf = false;
    if (wdd && wid && wd->ne[0] == hc_dim && wi->ne[0] == hc_dim) {
        q4_seg_t hc_segs[2] = {
            { wdd, lo,    wd->ne[1], wd->ggml_type, 1.f / (float)hc },
            { wid, d_inj, wi->ne[1], wi->ggml_type, 0.f },
        };
        hcf = q4_hip_gemv_m(hc_segs, 2, xn, hc_dim);
    }
    if (!hcf && !mm_d_silu(s->st, wd, xn, lo, 1.f / (float)hc)) {
        fprintf(stderr, "q4: hcmix%d silu wd=%p ty=%u ne=%llu,%llu\n", attn,
                wd ? (void *)q4_store_dev(s->st, wd) : NULL,
                wd ? wd->ggml_type : 0,
                wd ? (unsigned long long)wd->ne[0] : 0,
                wd ? (unsigned long long)wd->ne[1] : 0);
        return false;
    }
    if (!mm_d(s->st, wu, lo, gate)) {
        fprintf(stderr, "q4: hcmix%d up wu=%p ty=%u\n", attn,
                wu ? (void *)q4_store_dev(s->st, wu) : NULL,
                wu ? wu->ggml_type : 0);
        return false;
    }
    if (!q4_hip_mulsig_mean(xn, gate, d_mixed, n_embd, hc)) {
        if (!q4_hip_mul_sig(gated, xn, gate, hc_dim)) {
            fprintf(stderr, "q4: hcmix%d mul_sig\n", attn);
            return false;
        }
        if (!q4_hip_mean_hc(gated, d_mixed, n_embd, hc)) {
            fprintf(stderr, "q4: hcmix%d mean\n", attn);
            return false;
        }
    }
    if (!hcf && d_inj && wi && !mm_d(s->st, wi, xn, d_inj)) {
        fprintf(stderr, "q4: hcmix%d inj wi=%p ty=%u\n", attn,
                wi ? (void *)q4_store_dev(s->st, wi) : NULL,
                wi ? wi->ggml_type : 0);
        return false;
    }
    return true;
}

static bool hc_mix_dn_x(q4_sess *s, const float *d_res, float *d_mixed,
                        float *d_inj, int layer, bool attn, uint32_t n,
                        bool xn_ready);
static bool hc_mix_dn(q4_sess *s, const float *d_res, float *d_mixed, float *d_inj,
                      int layer, bool attn, uint32_t n) {
    return hc_mix_dn_x(s, d_res, d_mixed, d_inj, layer, attn, n, false);
}
/* xn_ready: the grouped rms of d_res already sits in s->d_pws (written by
 * a fused hc combine + rms). */
static bool hc_mix_dn_x(q4_sess *s, const float *d_res, float *d_mixed,
                        float *d_inj, int layer, bool attn, uint32_t n,
                        bool xn_ready) {
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    const q4_tensor *wn = T(g, attn ? "blk.%d.hc_attn_norm.weight" : "blk.%d.hc_ffn_norm.weight", layer);
    const q4_tensor *wd = T(g, attn ? "blk.%d.hc_attn_down.weight" : "blk.%d.hc_ffn_down.weight", layer);
    const q4_tensor *wu = T(g, attn ? "blk.%d.hc_attn_up.weight" : "blk.%d.hc_ffn_up.weight", layer);
    const q4_tensor *wi = T(g, attn ? "blk.%d.hc_attn_inject.weight" : "blk.%d.hc_ffn_inject.weight", layer);
    float *xn = s->d_pws;
    float *lo = xn + (size_t)n * hc_dim;
    float *gate = lo + (size_t)n * g->hc_rank;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    if (!xn_ready &&
        !q4_hip_grouped_rms_n(d_res, f32d(s->st, wn), xn, n_embd, hc, eps, n))
        return false;
    /* hc down and inject read the same xn rows: one launch when both
     * are bf16 (q4_hip_gemv_bf16_n2), each row as the single calls */
    const bool inj2 = d_inj && wi && wd && wd->ggml_type == Q4_T_BF16 &&
        wi->ggml_type == Q4_T_BF16 && wd->ne[0] == hc_dim &&
        wi->ne[0] == hc_dim &&
        q4_hip_gemv_bf16_n2(q4_store_dev(s->st, wd), lo, (uint32_t)wd->ne[1],
                            1.f / (float)hc, q4_store_dev(s->st, wi), d_inj,
                            (uint32_t)wi->ne[1], 0.f, xn, hc_dim, n);
    if (!inj2 && !mm_d_n_silu(s->st, wd, xn, lo, 1.f / (float)hc, n))
        return false;
    const uint8_t *dwu = wu ? q4_store_dev(s->st, wu) : NULL;
    if (!(dwu && wu->ne[1] == hc_dim &&
          q4_hip_hcup_mix_n(wu->ggml_type, dwu, lo, xn, d_mixed,
                            (uint32_t)wu->ne[0], n_embd, hc, n))) {
        if (!mm_d_n(s->st, wu, lo, gate, n)) return false;
        if (!q4_hip_mulsig_mean_n(xn, gate, d_mixed, n_embd, hc, n))
            return false;
    }
    if (!inj2 && d_inj && wi && !mm_d_n(s->st, wi, xn, d_inj, n))
        return false;
    return true;
}

static bool gdn_one_d(q4_sess *s, int layer, const float *d_x, float *d_y) {
    const q4_gguf *g = s->g;
    const uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    const uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    const uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    const uint32_t d_v = (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h;
    const uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    const uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    const uint32_t gi = s->gdn_ix[layer];
    float *qkv = s->d_ws;
    float *z = qkv + conv_ch;
    float *beta = z + n_v_h * d_v;
    float *alpha = beta + n_v_h;
    float *conv_out = alpha + n_v_h;
    float *core = conv_out + conv_ch;
    /* One launch for all four input projections (mixed q8_0/f32 segs). */
    const q4_tensor *tqkv = T(g, "blk.%d.attn_qkv.weight", layer);
    const q4_tensor *tgat = T(g, "blk.%d.attn_gate.weight", layer);
    const q4_tensor *tbet = T(g, "blk.%d.ssm_beta.weight", layer);
    const q4_tensor *talp = T(g, "blk.%d.ssm_alpha.weight", layer);
    q4_seg_t in_segs[4] = {
        { q4_store_dev(s->st, tqkv), qkv,   tqkv ? tqkv->ne[1] : 0, tqkv ? tqkv->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, tgat), z,     tgat ? tgat->ne[1] : 0, tgat ? tgat->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, tbet), beta,  tbet ? tbet->ne[1] : 0, tbet ? tbet->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, talp), alpha, talp ? talp->ne[1] : 0, talp ? talp->ggml_type : 0 , 0.f },
    };
    if (!q4_hip_gemv_m(in_segs, 4, d_x, g->n_embd)) {
        if (!mm_d(s->st, tqkv, d_x, qkv)) return false;
        if (!mm_d(s->st, tgat, d_x, z)) return false;
        if (!mm_d(s->st, tbet, d_x, beta)) return false;
        if (!mm_d(s->st, talp, d_x, alpha)) return false;
    }
    /* beta sigmoid is folded into gdn_step/gdn_tail. */
    float *cstate = s->d_gdn_conv + (size_t)gi * (ksz - 1) * conv_ch;
    const float *cw = f32d(s->st, T(g, "blk.%d.ssm_conv1d.weight", layer));
    if (!cw) return false;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    const float *A = f32d(s->st, T(g, "blk.%d.ssm_a", layer));
    const float *dt = f32d(s->st, T(g, "blk.%d.ssm_dt.bias", layer));
    const float *nw = f32d(s->st, T(g, "blk.%d.ssm_norm.weight", layer));
    float *Sbase = s->d_gdn_S + (size_t)gi * n_v_h * d_v * d_v;
    if (d_k == d_v &&
        q4_hip_gdn_tail(cstate, qkv, cw, conv_ch, ksz, Sbase, beta, alpha,
                        A, dt, nw, z, core, n_v_h, n_k_h, d_v, eps,
                        1.f / sqrtf((float)d_k))) {
        /* one launch did conv+shift+l2+step+rms-gate */
    } else {
        if (!q4_hip_gdn_convshift(cstate, qkv, cw, conv_out, conv_ch, ksz))
            return false;
        float *q = conv_out;
        float *k = q + d_k * n_k_h;
        float *v = k + d_k * n_k_h;
        if (!q4_hip_l2_heads(conv_out, 2 * n_k_h, d_k, eps)) return false;
        if (!q4_hip_gdn_step(Sbase, q, k, v, beta, alpha, A, dt, core, n_v_h,
                             n_k_h, d_v, 1.f / sqrtf((float)d_k)))
            return false;
        if (!q4_hip_rms_gate_heads(core, nw, z, n_v_h, d_v, eps))
            return false;
    }
    return mm_d(s->st, T(g, "blk.%d.ssm_out.weight", layer), core, d_y);
}

/* Sequential GDN recurrence for one token; qkv/z/beta/alpha/core already on GPU. */
static bool gdn_apply_d(q4_sess *s, int layer, float *qkv, float *z, float *beta,
                        float *alpha, float *core) {
    const q4_gguf *g = s->g;
    const uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    const uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    const uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    const uint32_t d_v = (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h;
    const uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    const uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    const uint32_t gi = s->gdn_ix[layer];
    float *cstate = s->d_gdn_conv + (size_t)gi * (ksz - 1) * conv_ch;
    const float *cw = f32d(s->st, T(g, "blk.%d.ssm_conv1d.weight", layer));
    if (!cw) return false;
    float *conv_out = s->d_ws;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    const float *A = f32d(s->st, T(g, "blk.%d.ssm_a", layer));
    const float *dt = f32d(s->st, T(g, "blk.%d.ssm_dt.bias", layer));
    const float *nw = f32d(s->st, T(g, "blk.%d.ssm_norm.weight", layer));
    float *Sbase = s->d_gdn_S + (size_t)gi * n_v_h * d_v * d_v;
    if (d_k == d_v &&
        q4_hip_gdn_tail(cstate, qkv, cw, conv_ch, ksz, Sbase, beta, alpha,
                        A, dt, nw, z, core, n_v_h, n_k_h, d_v, eps,
                        1.f / sqrtf((float)d_k)))
        return true;
    if (!q4_hip_gdn_convshift(cstate, qkv, cw, conv_out, conv_ch, ksz))
        return false;
    float *q = conv_out;
    float *k = q + d_k * n_k_h;
    float *v = k + d_k * n_k_h;
    if (!q4_hip_l2_heads(conv_out, 2 * n_k_h, d_k, eps)) return false;
    if (!q4_hip_gdn_step(Sbase, q, k, v, beta, alpha, A, dt, core, n_v_h,
                         n_k_h, d_v, 1.f / sqrtf((float)d_k)))
        return false;
    return q4_hip_rms_gate_heads(core, nw, z, n_v_h, d_v, eps);
}

static int kv_kind_of(void) {
    const char *e = getenv("Q4_KV");
    if (e && (!strcmp(e, "f32") || !strcmp(e, "fp32"))) return 0;
    if (e && (!strcmp(e, "f16") || !strcmp(e, "fp16"))) return 1;
    return 2;
}

static uint32_t kv_token_bytes(const q4_sess *s) {
    const q4_gguf *g = s->g;
    uint32_t kvw = (g->n_head_kv ? g->n_head_kv : 2) *
                   (g->n_embd_head_k ? g->n_embd_head_k : 256);
    if (s->kv_kind == 2) return (kvw / 32u) * 34u;
    return kvw * (s->kv_kind == 1 ? 2u : 4u);
}

static uint8_t *kv_at(const q4_sess *s, int is_v, uint32_t qi, uint32_t tok) {
    uint8_t *base = (uint8_t *)(is_v ? s->d_qsa_v : s->d_qsa_k);
    return base + ((size_t)qi * s->max_kv + tok) * s->kv_bpt;
}

static bool kv_write(const q4_sess *s, void *dst, const float *src, uint64_t n) {
    if (s->kv_kind == 2) return q4_hip_f32_to_q8(dst, src, n);
    if (s->kv_kind == 1) return q4_hip_f32_to_f16(dst, src, n);
    return q4_hip_copy(dst, src, n);
}

static uint32_t kv_tag(const q4_sess *s) {
    return s->kv_kind == 2 ? 8u : s->kv_kind == 1 ? 2u : 4u;
}

static void *idx_raw_at(const q4_sess *s, uint32_t qi) {
    return (uint8_t *)s->d_idx_raw +
           ((size_t)qi * s->max_kv * s->idx_dim) * sizeof(uint16_t);
}

static float *idx_pool_at(const q4_sess *s, uint32_t qi) {
    return s->d_idx_pool + (size_t)qi * s->max_blocks * s->idx_dim;
}

/* Store this chunk's raw indexer keys and refresh any block they complete. */
static bool idx_note_k(q4_sess *s, int layer, const float *d_x, uint32_t n_tok,
                       uint32_t pos0) {
    if (!s->d_idx_raw || !s->d_idx_k || n_tok == 0) return true;
    const q4_gguf *g = s->g;
    const q4_tensor *wk = T(g, "blk.%d.indexer.k_proj.weight", layer);
    const q4_tensor *wn = T(g, "blk.%d.indexer.k_norm.weight", layer);
    if (!wk || !wn) return true;
    if (!mm_d_n(s->st, wk, d_x, s->d_idx_k, n_tok)) return false;
    uint32_t qi = s->qsa_ix[layer];
    void *raw = idx_raw_at(s, qi);
    uint8_t *dst = (uint8_t *)raw + (size_t)pos0 * s->idx_dim * sizeof(uint16_t);
    if (!q4_hip_f32_to_f16(dst, s->d_idx_k, (uint64_t)n_tok * s->idx_dim))
        return false;
    uint32_t n_all = pos0 + n_tok;
    uint32_t b0 = pos0 / s->idx_ratio;
    uint32_t b1 = n_all / s->idx_ratio;
    if (b1 <= b0) return true;
    float base = g->rope_freq_base ? g->rope_freq_base : 1e7f;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    return q4_hip_qsa_pool(raw, idx_pool_at(s, qi), f32d(s->st, wn), s->idx_dim,
                           s->idx_ratio, b0, b1, n_rot, base, eps);
}

static bool idx_rebuild(q4_sess *s) {
    if (!s->d_idx_raw || s->n_kv < s->idx_ratio) return true;
    const q4_gguf *g = s->g;
    float base = g->rope_freq_base ? g->rope_freq_base : 1e7f;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    uint32_t b1 = s->n_kv / s->idx_ratio;
    for (uint32_t il = 0; il < g->n_layer; il++) {
        if (!q4_layer_is_qsa(g, (int)il)) continue;
        const q4_tensor *wn = T(g, "blk.%d.indexer.k_norm.weight", (int)il);
        if (!wn) continue;
        uint32_t qi = s->qsa_ix[il];
        if (!q4_hip_qsa_pool(idx_raw_at(s, qi), idx_pool_at(s, qi), f32d(s->st, wn),
                             s->idx_dim, s->idx_ratio, 0, b1, n_rot, base, eps))
            return false;
    }
    return true;
}

/* Q, K, V of the main attention are already in the cache. Run dense GQA, or
 * the indexer plus a fixed-width gather, then write the gated attention. */
static bool qsa_attend(q4_sess *s, int layer, const float *d_x, const float *d_q,
                       const float *d_g, float *d_o, uint32_t n_tok, uint32_t pos0) {
    const q4_gguf *g = s->g;
    const uint32_t n_head = g->n_head ? g->n_head : 24;
    const uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    const uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    const float scale = 1.f / sqrtf((float)hd);
    uint32_t qi = s->qsa_ix[layer];
    uint8_t *kbase = kv_at(s, 0, qi, 0);
    uint8_t *vbase = kv_at(s, 1, qi, 0);
    uint32_t n_all = pos0 + n_tok;
    if (!idx_note_k(s, layer, d_x, n_tok, pos0)) return false;
    int sparse = s->d_idx_raw && s->d_idx_score && s->idx_width &&
                 n_all > s->idx_width &&
                 T(g, "blk.%d.indexer.q_proj.weight", layer);
    const int32_t *sel = NULL;
    if (!sparse) {
        if (!q4_hip_qsa_decode_n(d_q, kbase, vbase, d_g, d_o, n_head, n_kvh,
                                 hd, n_tok, pos0, s->max_kv, scale, s->kv_kind))
            return false;
    } else {
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "q4: QSA sparse  width %u  (top-%u blocks of %u + tail)\n",
                s->idx_width, s->idx_kblk, s->idx_ratio);
    }
    const q4_tensor *wq = T(g, "blk.%d.indexer.q_proj.weight", layer);
    const q4_tensor *nqw = T(g, "blk.%d.indexer.q_norm.weight", layer);
    if (!wq || !nqw || !s->d_idx_q) return false;
    if (!mm_d_n(s->st, wq, d_x, s->d_idx_q, n_tok)) return false;
    float base = g->rope_freq_base ? g->rope_freq_base : 1e7f;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    uint32_t ih = g->indexer_n_head ? g->indexer_n_head : 4;
    uint32_t idim = s->idx_dim;
    if (!q4_hip_qsa_k_prep_n(s->d_idx_q, f32d(s->st, nqw), ih, idim, n_rot,
                             (int32_t)pos0, base, eps, n_tok))
        return false;
    uint32_t n_blocks = n_all / s->idx_ratio;
    if (!q4_hip_qsa_score(s->d_idx_q, idx_pool_at(s, qi), s->d_idx_score, n_tok,
                          n_blocks, s->max_blocks, ih, idim, pos0, s->idx_ratio))
        return false;
    if (!q4_hip_qsa_topk(s->d_idx_score, s->d_idx_sel, n_tok, n_blocks, s->max_blocks,
                         s->idx_kblk, s->idx_ratio, pos0, s->idx_width))
        return false;
    sel = s->d_idx_sel;
    if (!q4_hip_qsa_decode_idx(d_q, kbase, vbase, d_g, d_o, sel,
                                 s->idx_width, n_head, n_kvh, hd, n_tok, s->max_kv,
                                 scale, s->kv_kind))
        return false;
    }
    return true;
}

static bool qsa_one_d(q4_sess *s, int layer, const float *d_x, float *d_y) {
    const q4_gguf *g = s->g;
    const uint32_t n_head = g->n_head ? g->n_head : 24;
    const uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    const uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    float *dq = s->d_ws;
    float *dk = dq + n_head * hd * 2;
    float *dv = dk + n_kvh * hd;
    float *d_q = dv + n_kvh * hd;
    float *d_g = d_q + n_head * hd;
    float *d_o = d_g + n_head * hd;
    const q4_tensor *t_q = T(g, "blk.%d.attn_q.weight", layer);
    const q4_tensor *t_k = T(g, "blk.%d.attn_k.weight", layer);
    const q4_tensor *t_v = T(g, "blk.%d.attn_v.weight", layer);
    q4_seg_t qkv_segs[3] = {
        { q4_store_dev(s->st, t_q), dq, t_q ? t_q->ne[1] : 0, t_q ? t_q->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, t_k), dk, t_k ? t_k->ne[1] : 0, t_k ? t_k->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, t_v), dv, t_v ? t_v->ne[1] : 0, t_v ? t_v->ggml_type : 0 , 0.f },
    };
    if (!q4_hip_gemv_m(qkv_segs, 3, d_x, g->n_embd)) {
        if (!mm_d(s->st, t_q, d_x, dq)) return false;
        if (!mm_d(s->st, t_k, d_x, dk)) return false;
        if (!mm_d(s->st, t_v, d_x, dv)) return false;
    }
    const uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    float base = g->rope_freq_base ? g->rope_freq_base : 1e7f;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    const float *qn = f32d(s->st, T(g, "blk.%d.attn_q_norm.weight", layer));
    const float *kn = f32d(s->st, T(g, "blk.%d.attn_k_norm.weight", layer));
    if (!q4_hip_qsa_q_prep(dq, qn, d_q, d_g, n_head, hd, n_rot, s->pos, base, eps))
        return false;
    if (!q4_hip_qsa_k_prep(dk, kn, n_kvh, hd, n_rot, s->pos, base, eps))
        return false;
    uint32_t qi = s->qsa_ix[layer];
    uint32_t t = s->n_kv;
    if (t >= s->max_kv) {
        fprintf(stderr, "q4: context full (%u)\n", s->max_kv);
        return false;
    }
    if (n_kvh == 0) return false;
    size_t kvw = (size_t)n_kvh * hd;
    const uint32_t n_all = t + 1;
    const uint32_t gqa = n_head / n_kvh;
    if (s->d_qsa_k && s->d_qsa_v) {
        uint8_t *kbase = kv_at(s, 0, qi, 0);
        uint8_t *vbase = kv_at(s, 1, qi, 0);
        if (!kv_write(s, kbase + (size_t)t * s->kv_bpt, dk, kvw) ||
            !kv_write(s, vbase + (size_t)t * s->kv_bpt, dv, kvw))
            return false;
        if (qsa_attend(s, layer, d_x, d_q, d_g, d_o, 1, t))
            return mm_d(s->st, T(g, "blk.%d.attn_output.weight", layer), d_o, d_y);
    }
    (void)gqa;
    fprintf(stderr, "q4: qsa gpu decode failed layer %d n_kv=%u\n", layer, n_all);
    return false;
}

/* Graph-capture variant of qsa_one_d (single token, sparse indexer path):
 * every position-dependent value is read on-device from the counter pushed
 * once per token by q4_hip_pos_push, so the replayed graph stays correct as
 * pos advances. Requires the sparse path to be active at capture time. */
static bool qsa_one_gd(q4_sess *s, int layer, const float *d_x, float *d_y) {
    const q4_gguf *g = s->g;
    const uint32_t n_head = g->n_head ? g->n_head : 24;
    const uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    const uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    float *dq = s->d_ws;
    float *dk = dq + n_head * hd * 2;
    float *dv = dk + n_kvh * hd;
    float *d_q = dv + n_kvh * hd;
    float *d_g = d_q + n_head * hd;
    float *d_o = d_g + n_head * hd;
    const q4_tensor *t_q = T(g, "blk.%d.attn_q.weight", layer);
    const q4_tensor *t_k = T(g, "blk.%d.attn_k.weight", layer);
    const q4_tensor *t_v = T(g, "blk.%d.attn_v.weight", layer);
    q4_seg_t qkv_segs[5] = {
        { q4_store_dev(s->st, t_q), dq, t_q ? t_q->ne[1] : 0, t_q ? t_q->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, t_k), dk, t_k ? t_k->ne[1] : 0, t_k ? t_k->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, t_v), dv, t_v ? t_v->ne[1] : 0, t_v ? t_v->ggml_type : 0 , 0.f },
    };
    /* The two indexer projections read the same d_x: fold them into the
     * qkv launch; gemv_m validates every segment type itself. */
    const q4_tensor *wk0 = T(g, "blk.%d.indexer.k_proj.weight", layer);
    const q4_tensor *wq0 = T(g, "blk.%d.indexer.q_proj.weight", layer);
    bool idx_fused = false;
    if (wk0 && wq0 && s->d_idx_k && s->d_idx_q) {
        qkv_segs[3] = (q4_seg_t){ q4_store_dev(s->st, wk0), s->d_idx_k,
                                  wk0->ne[1], wk0->ggml_type, 0.f };
        qkv_segs[4] = (q4_seg_t){ q4_store_dev(s->st, wq0), s->d_idx_q,
                                  wq0->ne[1], wq0->ggml_type, 0.f };
        idx_fused = q4_hip_gemv_m(qkv_segs, 5, d_x, g->n_embd);
    }
    if (!idx_fused && !q4_hip_gemv_m(qkv_segs, 3, d_x, g->n_embd)) {
        if (!mm_d(s->st, t_q, d_x, dq)) return false;
        if (!mm_d(s->st, t_k, d_x, dk)) return false;
        if (!mm_d(s->st, t_v, d_x, dv)) return false;
    }
    const uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    float base = g->rope_freq_base ? g->rope_freq_base : 1e7f;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    const float *qn = f32d(s->st, T(g, "blk.%d.attn_q_norm.weight", layer));
    const float *kn = f32d(s->st, T(g, "blk.%d.attn_k_norm.weight", layer));
    /* indexer tensors are looked up early: when its projections rode the
     * fused qkv launch, idx_q's norm+rope can join the prep3 launch too. */
    const q4_tensor *wk = T(g, "blk.%d.indexer.k_proj.weight", layer);
    const q4_tensor *wn = T(g, "blk.%d.indexer.k_norm.weight", layer);
    const q4_tensor *wq = T(g, "blk.%d.indexer.q_proj.weight", layer);
    const q4_tensor *nqw = T(g, "blk.%d.indexer.q_norm.weight", layer);
    uint32_t ih = g->indexer_n_head ? g->indexer_n_head : 4;
    if (!wk || !wn || !wq || !nqw || !s->d_idx_raw || !s->d_idx_k ||
        !s->d_idx_q || !s->d_idx_score || !s->d_idx_sel || !s->idx_width)
        return false;
    if (!q4_hip_qsa_prep3_d(dq, qn, d_q, d_g, n_head, dk, kn, n_kvh,
                            idx_fused ? s->d_idx_q : NULL,
                            idx_fused ? f32d(s->st, nqw) : NULL,
                            idx_fused ? ih : 0,
                            hd, s->idx_dim, n_rot, base, eps))
        return false;
    uint32_t qi = s->qsa_ix[layer];
    size_t kvw = (size_t)n_kvh * hd;
    const float scale = 1.f / sqrtf((float)hd);
    uint8_t *kbase = kv_at(s, 0, qi, 0);
    uint8_t *vbase = kv_at(s, 1, qi, 0);
    if (!s->d_qsa_k || !s->d_qsa_v) return false;
    if (!q4_hip_kv_append_d(kbase, vbase, dk, dv, (uint32_t)kvw, s->kv_bpt,
                            s->kv_kind))
        return false;
    /* indexer: note this token's key, refresh the completed pool block */
    if (!idx_fused && !mm_d(s->st, wk, d_x, s->d_idx_k)) return false;
    void *raw = idx_raw_at(s, qi);
    if (!q4_hip_f32_to_f16_d(raw, s->idx_dim, s->d_idx_k, s->idx_dim))
        return false;
    if (!q4_hip_qsa_pool_d(raw, idx_pool_at(s, qi), f32d(s->st, wn),
                           s->idx_dim, s->idx_ratio, n_rot, base, eps))
        return false;
    if (!idx_fused && !mm_d(s->st, wq, d_x, s->d_idx_q)) return false;
    if (!idx_fused &&
        !q4_hip_qsa_k_prep_d(s->d_idx_q, f32d(s->st, nqw), ih, s->idx_dim,
                             n_rot, base, eps))
        return false;
    q4_hip_mark_pt(12);          /* idx q ready; score+topk start */
    if (!q4_hip_qsa_score_d(s->d_idx_q, idx_pool_at(s, qi), s->d_idx_score,
                            s->max_blocks, ih, s->idx_dim, s->idx_ratio))
        return false;
    if (!q4_hip_qsa_topk_d(s->d_idx_score, s->d_idx_sel, s->max_blocks,
                           s->idx_kblk, s->idx_ratio, s->idx_width))
        return false;
    q4_hip_mark_pt(13);          /* selection done; gather+attn start */
    if (!q4_hip_qsa_decode_idx(d_q, kbase, vbase, d_g, d_o, s->d_idx_sel,
                               s->idx_width, n_head, n_kvh, hd, 1, s->max_kv,
                               scale, s->kv_kind))
        return false;
    q4_hip_mark_pt(14);          /* split attention done */
    return mm_d(s->st, T(g, "blk.%d.attn_output.weight", layer), d_o, d_y);
}

/* Graph-capture variant for the dense (pre-sparse) QSA path: same as
 * qsa_one_d but every position-dependent value is read on-device from the
 * pushed pos counter. The indexer key is still tracked (raw append +
 * self-gating pool refresh) so the graph remains valid right up to the
 * sparse switch — the caller recaptures as sparse once that activates. */
static bool qsa_one_gdd(q4_sess *s, int layer, const float *d_x, float *d_y) {
    const q4_gguf *g = s->g;
    const uint32_t n_head = g->n_head ? g->n_head : 24;
    const uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    const uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    float *dq = s->d_ws;
    float *dk = dq + n_head * hd * 2;
    float *dv = dk + n_kvh * hd;
    float *d_q = dv + n_kvh * hd;
    float *d_g = d_q + n_head * hd;
    float *d_o = d_g + n_head * hd;
    const q4_tensor *t_q = T(g, "blk.%d.attn_q.weight", layer);
    const q4_tensor *t_k = T(g, "blk.%d.attn_k.weight", layer);
    const q4_tensor *t_v = T(g, "blk.%d.attn_v.weight", layer);
    const q4_tensor *wk = T(g, "blk.%d.indexer.k_proj.weight", layer);
    const q4_tensor *wn = T(g, "blk.%d.indexer.k_norm.weight", layer);
    /* Indexer K tracking only exists while the raw/pool buffers do; it reads
     * the same d_x as qkv so it rides the same gemv_m launch. */
    const int track = wk && wn && s->d_idx_raw && s->d_idx_k;
    q4_seg_t qkv_segs[4] = {
        { q4_store_dev(s->st, t_q), dq, t_q ? t_q->ne[1] : 0, t_q ? t_q->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, t_k), dk, t_k ? t_k->ne[1] : 0, t_k ? t_k->ggml_type : 0 , 0.f },
        { q4_store_dev(s->st, t_v), dv, t_v ? t_v->ne[1] : 0, t_v ? t_v->ggml_type : 0 , 0.f },
    };
    uint32_t nseg = 3;
    bool idx_fused = false;
    if (track) {
        qkv_segs[3] = (q4_seg_t){ q4_store_dev(s->st, wk), s->d_idx_k,
                                  wk->ne[1], wk->ggml_type, 0.f };
        idx_fused = q4_hip_gemv_m(qkv_segs, 4, d_x, g->n_embd);
    }
    if (!idx_fused && !q4_hip_gemv_m(qkv_segs, nseg, d_x, g->n_embd)) {
        if (!mm_d(s->st, t_q, d_x, dq)) return false;
        if (!mm_d(s->st, t_k, d_x, dk)) return false;
        if (!mm_d(s->st, t_v, d_x, dv)) return false;
    }
    const uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    float base = g->rope_freq_base ? g->rope_freq_base : 1e7f;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    const float *qn = f32d(s->st, T(g, "blk.%d.attn_q_norm.weight", layer));
    const float *kn = f32d(s->st, T(g, "blk.%d.attn_k_norm.weight", layer));
    if (!q4_hip_qsa_prep3_d(dq, qn, d_q, d_g, n_head, dk, kn, n_kvh,
                            NULL, NULL, 0, hd, s->idx_dim, n_rot, base, eps))
        return false;
    uint32_t qi = s->qsa_ix[layer];
    size_t kvw = (size_t)n_kvh * hd;
    const float scale = 1.f / sqrtf((float)hd);
    uint8_t *kbase = kv_at(s, 0, qi, 0);
    uint8_t *vbase = kv_at(s, 1, qi, 0);
    if (!s->d_qsa_k || !s->d_qsa_v) return false;
    if (!q4_hip_kv_append_d(kbase, vbase, dk, dv, (uint32_t)kvw, s->kv_bpt,
                            s->kv_kind))
        return false;
    if (track) {
        if (!idx_fused && !mm_d(s->st, wk, d_x, s->d_idx_k)) return false;
        void *raw = idx_raw_at(s, qi);
        if (!q4_hip_f32_to_f16_d(raw, s->idx_dim, s->d_idx_k, s->idx_dim))
            return false;
        if (!q4_hip_qsa_pool_d(raw, idx_pool_at(s, qi), f32d(s->st, wn),
                               s->idx_dim, s->idx_ratio, n_rot, base, eps))
            return false;
    }
    if (!q4_hip_qsa_decode_d(d_q, kbase, vbase, d_g, d_o, n_head, n_kvh, hd,
                             s->max_kv, scale, s->kv_kind))
        return false;
    return mm_d(s->st, T(g, "blk.%d.attn_output.weight", layer), d_o, d_y);
}

static bool ple_one_d(q4_sess *s, int layer, float *d_res, int32_t token) {
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    int32_t rows[64];
    if (!q4_ple_hash(s->ple, &token, 1, s->ple_hist, rows)) return false;
    float emb_h[16 * 160];
    if (g->ple_n_heads * g->n_embd_ple > 16 * 160) return false;
    if (!q4_ple_gather(s->ple, rows, g->ple_n_heads, emb_h)) return false;
    if (!q4_hip_h2d(s->d_emb, emb_h, g->ple_n_heads * g->n_embd_ple * sizeof(float)))
        return false;
    float *key = s->d_ws;
    float *val = key + hc_dim;
    float *kn = val + n_embd;
    float *qn = kn + hc_dim;
    float *gated = qn + hc_dim;
    float *cn = gated + hc_dim;
    float *co = cn + hc_dim;
    if (!mm_d(s->st, T(g, "blk.%d.ple_key.weight", layer), s->d_emb, key))
        return false;
    if (!mm_d(s->st, T(g, "blk.%d.ple_value.weight", layer), s->d_emb, val))
        return false;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    if (!q4_hip_grouped_rms(key, f32d(s->st, T(g, "blk.%d.ple_norm_key.weight", layer)),
                            kn, n_embd, hc, eps))
        return false;
    if (!q4_hip_grouped_rms(d_res, f32d(s->st, T(g, "blk.%d.ple_norm_query.weight", layer)),
                            qn, n_embd, hc, eps))
        return false;
    /* per-stream gate is 4 dots — do on host */
    float kn_h[4 * 2560], qn_h[4 * 2560], val_h[2560];
    if (!q4_hip_d2h(kn_h, kn, hc_dim * sizeof(float))) return false;
    if (!q4_hip_d2h(qn_h, qn, hc_dim * sizeof(float))) return false;
    if (!q4_hip_d2h(val_h, val, n_embd * sizeof(float))) return false;
    float gate[8];
    const float isqrt = 1.0f / sqrtf((float)n_embd);
    for (uint32_t c = 0; c < hc; c++) {
        float dot = 0;
        for (uint32_t i = 0; i < n_embd; i++)
            dot += kn_h[c * n_embd + i] * qn_h[c * n_embd + i];
        dot *= isqrt;
        float mag = sqrtf(fabsf(dot) < 1e-6f ? 1e-6f : fabsf(dot));
        gate[c] = 1.0f / (1.0f + expf(-(copysignf(mag, dot))));
    }
    float gated_h[4 * 2560];
    for (uint32_t c = 0; c < hc; c++)
        for (uint32_t i = 0; i < n_embd; i++)
            gated_h[c * n_embd + i] = val_h[i] * gate[c];
    if (!q4_hip_h2d(gated, gated_h, hc_dim * sizeof(float))) return false;
    if (!q4_hip_grouped_rms(gated, f32d(s->st, T(g, "blk.%d.ple_norm_conv.weight", layer)),
                            cn, n_embd, hc, eps))
        return false;
    const uint32_t kern = g->ple_conv_kernel ? g->ple_conv_kernel : 4;
    const uint32_t dil = g->ple_ngram ? g->ple_ngram : 3;
    const uint32_t hist = (kern - 1) * dil;
    /* conv on host (tiny 10*10240) */
    float hist_h[10 * 10240];
    float cn_h[10240];
    if (hc_dim > 10240 || hist + 1 > 10) return false;
    if (!q4_hip_d2h(hist_h, s->d_ple_conv, (hist + 1) * hc_dim * sizeof(float)))
        return false;
    if (!q4_hip_d2h(cn_h, cn, hc_dim * sizeof(float))) return false;
    memmove(hist_h, hist_h + hc_dim, hist * hc_dim * sizeof(float));
    memcpy(hist_h + hist * hc_dim, cn_h, hc_dim * sizeof(float));
    /* ple_conv1d may be any gguf type — dequantize via ple_conv_w. */
    const float *kw = ple_conv_w(s, layer);
    if (!kw) return false;
    float co_h[10240];
    memset(co_h, 0, hc_dim * sizeof(float));
    for (uint32_t k = 0; k < kern; k++) {
        uint32_t start = hist - (kern - 1 - k) * dil;
        const float *src = hist_h + start * hc_dim;
        for (uint32_t c = 0; c < hc_dim; c++)
            co_h[c] += kw[k + c * kern] * src[c];
    }
    q4_silu(co_h, hc_dim);
    if (!q4_hip_h2d(s->d_ple_conv, hist_h, (hist + 1) * hc_dim * sizeof(float)))
        return false;
    if (!q4_hip_h2d(co, co_h, hc_dim * sizeof(float))) return false;
    if (!q4_hip_add(co, gated, co, hc_dim)) return false;
    if (!q4_hip_add(d_res, d_res, co, hc_dim)) return false;
    return true;
}

/* The original per-token PLE loop: kept for the mtp_stash snapshots and as
 * the Q4_PLE_BATCH=0 fallback. */
static bool ple_seq_d(q4_sess *s, int layer, const int32_t *toks, uint32_t n) {
    const q4_gguf *g = s->g;
    const uint32_t hc_dim = (g->hc_mult ? g->hc_mult : 4) * g->n_embd;
    for (uint32_t t = 0; t < n; t++) {
        if (!q4_hip_copy(s->d_res, s->d_pref + (size_t)t * hc_dim, hc_dim))
            return false;
        if (!ple_one_d(s, layer, s->d_res, toks[t])) return false;
        if (g->ple_ngram >= 2) {
            memmove(s->ple_hist, s->ple_hist + 1,
                    (g->ple_ngram - 2) * sizeof(int32_t));
            s->ple_hist[g->ple_ngram - 2] = toks[t];
        }
        if (s->mtp_stash && s->d_ple_snap && t < (uint32_t)s->mtp_max) {
            q4_hip_copy(s->d_ple_snap[t], s->d_ple_conv, 10ull * hc_dim);
            memcpy(s->ple_hist_snap + (size_t)t * 8, s->ple_hist,
                   sizeof(s->ple_hist));
        }
        if (!q4_hip_copy(s->d_pref + (size_t)t * hc_dim, s->d_res, hc_dim))
            return false;
    }
    return true;
}

struct ple_gather_a {
    const q4_ple *p;
    const int32_t *rows;
    int64_t n;
    float *out;
    int ok;
};
static void *ple_gather_th(void *a_) {
    struct ple_gather_a *a = a_;
    a->ok = q4_ple_gather(a->p, a->rows, a->n, a->out) ? 1 : 0;
    return NULL;
}

static int ple_gather_threads(void) {
    static int nth = -1;
    if (nth < 0) {
        const char *e = getenv("Q4_PLE_THREADS");
        nth = e && e[0] ? atoi(e) : 8;
        if (nth < 1) nth = 1;
        if (nth > 32) nth = 32;
    }
    return nth;
}

/* Slice a rows-gather over ple_gather_threads() pthreads. */
static int ple_gather_slices(const q4_ple *p, const int32_t *rows,
                             int64_t n_rows, uint32_t w, float *out) {
    struct ple_gather_a ga[32];
    pthread_t th[32];
    uint8_t started[32] = {0};
    int ns = 0;
    int64_t slice = (n_rows + ple_gather_threads() - 1) / ple_gather_threads();
    for (int i = 0; i < ple_gather_threads(); i++) {
        int64_t o = (int64_t)i * slice;
        if (o >= n_rows) break;
        ga[i] = (struct ple_gather_a){p, rows + o,
                                      n_rows - o < slice ? n_rows - o : slice,
                                      out + (size_t)o * w, 0};
        ns = i + 1;
        /* a slice whose thread could not start runs here: a skipped slice
         * used to leave its rows unwritten (and the joins mismatched) */
        if (pthread_create(&th[i], NULL, ple_gather_th, &ga[i]) == 0)
            started[i] = 1;
        else
            (void)ple_gather_th(&ga[i]);
    }
    int ok = ns > 0;
    for (int i = 0; i < ns; i++) {
        if (started[i]) pthread_join(th[i], NULL);
        if (!ga[i].ok) ok = 0;
    }
    return ok;
}

/* Background job: hash + gather one chunk's PLE rows into job-owned buffers
 * (bit-identical to the synchronous path — same rows, same order, same
 * dequant). */
static void *ple_job_th(void *p_) {
    struct ple_job *j = p_;
    q4_sess *s = j->s;
    const q4_gguf *g = s->g;
    const uint32_t nh = g->ple_n_heads;
    const uint32_t w = g->n_embd_ple ? g->n_embd_ple : 160;
    j->ok = 0;
    if (q4_ple_hash(s->ple, j->toks_copy, (int64_t)j->n, j->hist, j->rows))
        j->ok = ple_gather_slices(s->ple, j->rows, (int64_t)j->n * nh, w,
                                  j->emb);
    return NULL;
}

static bool ple_ahead_on(void) {
    static int on = -1;
    if (on < 0)
        on = !(getenv("Q4_PLE_AHEAD") && getenv("Q4_PLE_AHEAD")[0] == '0');
    return on;
}

static bool ple_job_start(q4_sess *s, const int32_t *toks, int n,
                          const int32_t *hist) {
    if (!s->ple || n <= 0) return false;
    const q4_gguf *g = s->g;
    const uint32_t nh = g->ple_n_heads;
    const uint32_t w = g->n_embd_ple ? g->n_embd_ple : 160;
    static uint64_t ple_seq;
    struct ple_job *j = NULL;
    for (int i = 0; i < 2; i++) {
        if (!s->ple_jobs[i].live) { j = &s->ple_jobs[i]; break; }
        /* already prefetching this very chunk → nothing to do */
        if (s->ple_jobs[i].toks == toks && s->ple_jobs[i].n == n &&
            !memcmp(s->ple_jobs[i].hist, hist, sizeof(s->ple_jobs[i].hist)))
            return true;
    }
    if (!j) {
        /* Both slots busy with stale (unconsumed) jobs: evict the older
         * one — its chunk can only ever take the synchronous path now. */
        struct ple_job *o =
            s->ple_jobs[0].seq < s->ple_jobs[1].seq ? &s->ple_jobs[0]
                                                  : &s->ple_jobs[1];
        pthread_join(o->th, NULL);
        o->live = 0;
        j = o;
    }
    if ((size_t)n > j->cap) {
        if (j->emb && j->emb_pinned) q4_hip_host_unregister(j->emb);
        free(j->rows);
        free(j->emb);
        free(j->toks_copy);
        j->rows = malloc((size_t)n * nh * sizeof(int32_t));
        j->emb = malloc((size_t)n * nh * w * sizeof(float));
        j->toks_copy = malloc((size_t)n * sizeof(int32_t));
        j->emb_pinned = 0;
        if (!j->rows || !j->emb || !j->toks_copy) {
            free(j->rows);
            free(j->emb);
            free(j->toks_copy);
            j->rows = NULL;
            j->emb = NULL;
            j->toks_copy = NULL;
            j->cap = 0;
            return false;
        }
        j->cap = (size_t)n;
        if (!j->emb_pinned)
            j->emb_pinned = q4_hip_host_register(
                j->emb, (size_t)n * nh * w * sizeof(float))
                                ? 1
                                : 0;
    }
    j->s = s;
    j->toks = toks;
    j->n = n;
    j->seq = ++ple_seq;
    memcpy(j->toks_copy, toks, (size_t)n * sizeof(int32_t));
    memcpy(j->hist, hist, sizeof(j->hist));
    j->ok = 0;
    j->live = 1;
    if (pthread_create(&j->th, NULL, ple_job_th, j) != 0) {
        j->live = 0;
        return false;
    }
    return true;
}

/* Join + take a job iff it exactly matches this ple_chunk_d call; a live job
 * for a different chunk is left running for its own consume. */
static struct ple_job *ple_job_take(q4_sess *s, const int32_t *toks, int n) {
    for (int i = 0; i < 2; i++) {
        struct ple_job *j = &s->ple_jobs[i];
        if (!j->live || j->n != n) continue;
        if (memcmp(j->hist, s->ple_hist, sizeof(j->hist))) continue;
        if (memcmp(j->toks_copy, toks, (size_t)n * sizeof(int32_t)))
            continue;
        pthread_join(j->th, NULL);
        return j;
    }
    return NULL;
}

static void ple_jobs_close(q4_sess *s) {
    for (int i = 0; i < 2; i++) {
        struct ple_job *j = &s->ple_jobs[i];
        if (j->live) pthread_join(j->th, NULL);
        j->live = 0;
        if (j->emb && j->emb_pinned) q4_hip_host_unregister(j->emb);
        free(j->rows);
        free(j->emb);
        free(j->toks_copy);
        j->rows = NULL;
        j->emb = NULL;
        j->toks_copy = NULL;
        j->cap = 0;
    }
}

/* Start the NEXT chunk's gather early so it overlaps the current chunk.
 * The hist it assumes is the tail of the current chunk — i.e. the ngram-1
 * ids immediately before `toks` in the caller's (contiguous) id array; the
 * consume check in ple_chunk_d verifies it. */
bool q4_sess_ple_ahead(q4_sess *s, const int32_t *toks, int n) {
    if (!s || !toks || n <= 0 || !s->ple || !ple_ahead_on()) return false;
    int32_t hist[8];
    memcpy(hist, s->ple_hist, sizeof(hist));
    const int nh2 = (int)(s->g->ple_ngram ? s->g->ple_ngram : 3) - 1;
    for (int k = 0; k < nh2 && k < 8; k++) hist[k] = toks[k - nh2];
    return ple_job_start(s, toks, n, hist);
}

/* Whole-chunk PLE: one hash pass, parallel row gather, one H2D, then all-GPU
 * projection/gate/conv. d_pws layout per token: emb nh*w | key->kn->gated->cn
 * hc_dim | qn->co hc_dim | val n_embd (fits under the hc need). */
static bool ple_chunk_d(q4_sess *s, int layer, const int32_t *toks, uint32_t n) {
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    const uint32_t kern = g->ple_conv_kernel ? g->ple_conv_kernel : 4;
    const uint32_t dil = g->ple_ngram ? g->ple_ngram : 3;
    const uint32_t hist = (kern - 1) * dil;
    const uint32_t nh = g->ple_n_heads;
    const uint32_t w = g->n_embd_ple ? g->n_embd_ple : 160;
    const uint32_t emb_w = nh * w;
    int32_t *rows = NULL;
    float *emb_h = NULL;
    struct ple_job *job = ple_ahead_on() ? ple_job_take(s, toks, (int)n) : NULL;
    static int plep = -1;
    if (plep < 0) plep = getenv("Q4_PLE_PROF") && getenv("Q4_PLE_PROF")[0] == '1';
    double tp = plep ? q4_now() : 0;
    int ok = 0;
    if (job) {
        rows = job->rows;
        emb_h = job->emb;
        ok = job->ok;
        job->live = 0;
        if (plep)
            fprintf(stderr, "ple prof: prefetched (%llu rows)\n",
                    (unsigned long long)((int64_t)n * nh));
    } else {
        rows = malloc((size_t)n * nh * sizeof(int32_t));
        emb_h = malloc((size_t)n * emb_w * sizeof(float));
        if (!rows || !emb_h || !s->d_ple_conv2) {
            free(rows);
            free(emb_h);
            return false;
        }
        const double t_hash = tp;
        if (q4_ple_hash(s->ple, toks, (int64_t)n, s->ple_hist, rows)) {
            tp = plep ? q4_now() : 0;
            ok = ple_gather_slices(s->ple, rows, (int64_t)n * nh, w, emb_h);
            if (plep)
                fprintf(stderr,
                        "ple prof: hash %.3f gather %.3f (%llu rows x %u B)\n",
                        tp - t_hash, q4_now() - tp,
                        (unsigned long long)((int64_t)n * nh),
                        q4_ple_row_bytes(s->ple));
            tp = q4_now();
        }
    }
    if (!s->d_ple_conv2) {
        if (!job) { free(rows); free(emb_h); }
        return false;
    }
    float *emb = s->d_pws;
    float *key = emb + (size_t)n * emb_w; /* -> kn -> gated */
    float *qn = key + (size_t)n * hc_dim; /* -> cn */
    float *co = qn + (size_t)n * hc_dim;
    float *val = co + (size_t)n * hc_dim;
    float *cn = qn;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    const double t_h2d = tp;
    if (ok)
        ok = q4_hip_h2d(emb, emb_h, (size_t)n * emb_w * sizeof(float));
    if (plep) { tp = q4_now(); }
    if (ok) {
        const q4_tensor *wk = T(g, "blk.%d.ple_norm_key.weight", layer);
        const q4_tensor *wq = T(g, "blk.%d.ple_norm_query.weight", layer);
        const q4_tensor *wc = T(g, "blk.%d.ple_norm_conv.weight", layer);
        /* ple_conv1d may be non-F32 — upload a dequantized device copy once. */
        float *kw = s->d_ple_kw;
        if (!kw) {
            const float *hkw = ple_conv_w(s, layer);
            const q4_tensor *ct =
                T(g, "blk.%d.ple_conv1d.weight", layer);
            if (!hkw || !ct) return false;
            uint64_t nw = ct->ne[0] * ct->ne[1];
            kw = q4_hip_malloc(nw * sizeof(float));
            if (!kw || !q4_hip_h2d(kw, hkw, nw * sizeof(float)))
                return false;
            s->d_ple_kw = kw;
        }
        ok = mm_d_n(s->st, T(g, "blk.%d.ple_key.weight", layer), emb, key, n) &&
             mm_d_n(s->st, T(g, "blk.%d.ple_value.weight", layer), emb, val, n) &&
             q4_hip_grouped_rms_n(key, f32d(s->st, wk), key, n_embd, hc, eps, n) &&
             q4_hip_grouped_rms_n(s->d_pref, f32d(s->st, wq), qn, n_embd, hc, eps, n) &&
             q4_hip_ple_gate(key, qn, val, n_embd, hc, n) &&
             q4_hip_grouped_rms_n(key, f32d(s->st, wc), cn, n_embd, hc, eps, n) &&
             q4_hip_ple_conv(co, cn, s->d_ple_conv, kw, hc_dim, kern, dil,
                             hist, n) &&
             q4_hip_ple_state(s->d_ple_conv2, s->d_ple_conv, cn, n, hc_dim,
                              hist + 1) &&
             q4_hip_add(co, co, key, (uint64_t)n * hc_dim) &&
             q4_hip_add(s->d_pref, s->d_pref, co, (uint64_t)n * hc_dim);
        if (plep) {
            q4_hip_sync();
            fprintf(stderr, "ple prof: h2d %.3f gpu %.3f\n",
                    tp - t_h2d, q4_now() - tp);
        }
    }
    /* MTP per-token snapshots without the sequential path: the conv state
     * after token t is just the window of its last hist+1 conv inputs —
     * ple_state(T=t+1) selects them from (old state tail ++ cn). The host
     * n-gram ring is replayed the same way ple_seq_d does. */
    int nsn = 0;
    if (ok && s->mtp_stash && s->d_ple_snap) {
        nsn = n < (uint32_t)s->mtp_max ? (int)n : s->mtp_max;
        int32_t hh[sizeof(s->ple_hist) / sizeof(int32_t)];
        memcpy(hh, s->ple_hist, sizeof(hh));
        for (int t = 0; t < nsn && ok; t++) {
            ok = q4_hip_ple_state(s->d_ple_snap[t], s->d_ple_conv, cn,
                                  (uint32_t)t + 1, hc_dim, hist + 1);
            if (g->ple_ngram >= 2) {
                memmove(hh, hh + 1, (g->ple_ngram - 2) * sizeof(int32_t));
                hh[g->ple_ngram - 2] = toks[t];
            }
            memcpy(s->ple_hist_snap + (size_t)t * 8, hh,
                   sizeof(s->ple_hist));
        }
    }
    /* Same final ple_hist as the per-token loop. */
    if (g->ple_ngram >= 2) {
        for (uint32_t t = 0; t < n; t++) {
            memmove(s->ple_hist, s->ple_hist + 1,
                    (g->ple_ngram - 2) * sizeof(int32_t));
            s->ple_hist[g->ple_ngram - 2] = toks[t];
        }
    }
    if (ok) {
        float *tp = s->d_ple_conv;
        s->d_ple_conv = s->d_ple_conv2;
        s->d_ple_conv2 = tp;
    }
    if (!job) {
        free(rows);
        free(emb_h);
    }
    return ok != 0;
}

static bool embed_hip(q4_sess *s, int32_t token, uint32_t cell, float *d_out) {
    const q4_gguf *g = s->g;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    float emb[2560];
    if (n_embd > 2560) return false;
    const float *row = cell < s->emb_rows_n ? s->emb_rows[cell] : NULL;
    if (row) {
        if (!q4_hip_h2d(s->d_emb, row, n_embd * sizeof(float))) return false;
        return q4_hip_repeat_hc(s->d_emb, d_out, n_embd, hc);
    }
    if (!embed_tok(s, token, emb)) {
        fprintf(stderr, "q4: embed failed tok=%d vocab=%u\n", token, g->n_vocab);
        return false;
    }
    if (!q4_hip_h2d(s->d_emb, emb, n_embd * sizeof(float))) return false;
    return q4_hip_repeat_hc(s->d_emb, d_out, n_embd, hc);
}

static int qsa_sparse_ready(q4_sess *s, int il);
static bool attn_hip(q4_sess *s, int il, int32_t token) {
    const q4_gguf *g = s->g;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    float *d_mixed = s->d_ws + 65536;
    float *d_block = s->d_ws + 81920;
    float *d_inj = s->d_ws + 98304;
    const int prof = q4_prof();
    double tt = prof ? q4_now() : 0;
    if (g->ple_layer == (uint32_t)il && s->ple) {
        if (!ple_one_d(s, il, s->d_res, token)) {
            fprintf(stderr, "q4: ple failed layer %d tok=%d\n", il, token);
            return false;
        }
        if (g->ple_ngram >= 2) {
            memmove(s->ple_hist, s->ple_hist + 1,
                    (g->ple_ngram - 2) * sizeof(int32_t));
            s->ple_hist[g->ple_ngram - 2] = token;
        }
    }
    if (prof) { q4_hip_sync(); q4_prof_dec[5] += q4_now() - tt; tt = q4_now(); }
    q4_hip_mark_pt(6);
    if (!hc_mix_d(s, s->d_res, d_mixed, d_inj, il, true, false)) {
        fprintf(stderr, "q4: hc-attn mix failed layer %d\n", il);
        return false;
    }
    if (prof) { q4_hip_sync(); q4_prof_dec[6] += q4_now() - tt; tt = q4_now(); }
    q4_hip_mark_pt(7);
    if (q4_layer_is_qsa(g, il)) {
        bool ok;
        if (q4_hip_capturing() && q4_hip_pos_dev_ok())
            ok = qsa_sparse_ready(s, il) ? qsa_one_gd(s, il, d_mixed, d_block)
                                       : qsa_one_gdd(s, il, d_mixed, d_block);
        else
            ok = qsa_one_d(s, il, d_mixed, d_block);
        if (!ok) {
            fprintf(stderr, "q4: qsa failed layer %d n_kv=%u\n", il, s->n_kv);
            return false;
        }
    } else {
        if (!gdn_one_d(s, il, d_mixed, d_block)) {
            fprintf(stderr, "q4: gdn failed layer %d\n", il);
            return false;
        }
    }
    if (prof) { q4_hip_sync(); q4_prof_dec[7] += q4_now() - tt; tt = q4_now(); }
    q4_hip_mark_pt(8);
    /* Fold the post-attn combine with the FFN norm when possible: one
     * launch writes res += block*w and xn = rms(res)*wn_ffn. */
    const q4_tensor *wnf = T(g, "blk.%d.hc_ffn_norm.weight", il);
    bool fnormed = wnf &&
        q4_hip_hc_comb_rms(s->d_res, d_block, d_inj, f32d(s->st, wnf),
                           s->d_ws, n_embd, hc,
                           g->rms_eps ? g->rms_eps : 1e-6f);
    if (!fnormed &&
        !q4_hip_hc_combine(s->d_res, d_block, d_inj, n_embd, hc)) {
        fprintf(stderr, "q4: hc-attn combine failed layer %d\n", il);
        return false;
    }
    q4_hip_mark_pt(9);
    if (!hc_mix_d(s, s->d_res, d_mixed, d_inj, il, false, fnormed)) {
        fprintf(stderr, "q4: hc-ffn mix failed layer %d\n", il);
        return false;
    }
    q4_hip_mark_pt(10);
    if (prof) { q4_hip_sync(); q4_prof_dec[8] += q4_now() - tt; }
    return true;
}

double q4_prof_dec[16]; /* 0 embed 1 attn 2 moe 3 combine 4 head
                           5 ple 6 hc-attn 7 qsa/gdn 8 comb+hc-ffn */

static bool head_hip(q4_sess *s, int32_t *out_id, float *out_logit);

/* QSA layers are graphable once the sparse indexer path is permanently
 * active (n_all > idx_width monotonic): device-position kernels keep the
 * replayed graph correct as pos advances. Before that point they stay
 * eager and retry capture each token. */
static int qsa_sparse_ready(q4_sess *s, int il) {
    const q4_gguf *g = s->g;
    return q4_hip_pos_dev_ok() && s->d_idx_raw && s->d_idx_score &&
           s->idx_width && s->n_kv + 1 > s->idx_width &&
           T(g, "blk.%d.indexer.q_proj.weight", il) &&
           T(g, "blk.%d.indexer.k_proj.weight", il) &&
           T(g, "blk.%d.indexer.k_norm.weight", il);
}

/* Graphable decode layer: QSA works once the device pos counter is live —
 * dense until sparse locks in, then the layer is recaptured as sparse (the
 * gA_sp flag detects the switch). Not the PLE layer (host round-trips);
 * CPU-expert path must be active; profiling off (its per-stage syncs cannot
 * live inside a capture). */
static int graphable_layer(q4_sess *s, int il) {
    const q4_gguf *g = s->g;
    if (s->graph_off || q4_prof()) return 0;
    if (s->ple && (uint32_t)il == g->ple_layer) return 0;
    if (q4_layer_is_qsa(g, il) &&
        !q4_hip_qsa_dec_d_ok(g->n_head ? g->n_head : 24,
                             g->n_head_kv ? g->n_head_kv : 2,
                             g->n_embd_head_k ? g->n_embd_head_k : 256))
        return 0;
    return q4_moe_cpu_active(g);
}

/* Q4_STEP_PROF=1: host-side wall time per decode phase (cheap, ~4 clock
 * reads per layer). Distinguishes host serialization from GPU work. */
double q4_prof_step[8];
double q4_prof_gpu_ms;          /* g_str busy window per decode (sum) */
double q4_prof_gpu_ph[8];       /* per-phase GPU event time (slots 1..5) */
double q4_prof_gpu_sec[8];      /* gA sections: hc1 attn comb ffnmix moea
                                   + QSA subsections: prep score+topk dec */
static void gpu_mark(int slot) {
    double ms = q4_hip_mark_ms_slot(slot);
    if (ms > 0) q4_prof_gpu_ph[slot] += ms;
}
static int step_prof(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("Q4_STEP_PROF"); v = e && e[0] == '1'; }
    return v;
}

static bool layer_hip(q4_sess *s, int il, int32_t token) {
    const q4_gguf *g = s->g;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const int prof = q4_prof();
    const int sp = step_prof();
    double tt = (prof || sp) ? q4_now() : 0;
    float *d_mixed = s->d_ws + 65536;
    float *d_block = s->d_ws + 81920;
    float *d_inj = s->d_ws + 98304;
    float *d_moe = s->d_ws + 120000;
    if (graphable_layer(s, il)) {
        void *ga = s->gA[il];
        /* QSA dense->sparse switch: a graph captured dense would keep
         * running dense attention past idx_width; drop it so the layer is
         * recaptured in sparse mode. */
        if (ga && q4_layer_is_qsa(g, il) &&
            (int)s->gA_sp[il] != qsa_sparse_ready(s, il)) {
            q4_hip_graph_free(ga);
            s->gA[il] = ga = NULL;
        }
        if (s->devroute < 0)
            s->devroute = q4_moe_dev_route(g, s->ex);
        if (s->devroute) {
            /* Device-routed layer: attention + whole MoE in ONE graph —
             * no host seam at all. */
            if (!ga) {
                /* Non-F32 sgate converts lazily inside the layer body —
                 * malloc + hipMalloc + sync H2D would invalidate capture. */
                q4_moe_dev_warm(g, s->st, il);
                if (!q4_hip_graph_begin()) {
                    fprintf(stderr, "q4: graph begin failed; eager\n");
                    s->graph_off = 1;
                    goto eager;
                }
                bool ok = attn_hip(s, il, token) &&
                          q4_moe_gpu_dev(g, s->st, s->ex, il, s->d_res,
                                         d_inj, hc, d_mixed, d_block, d_moe);
                ga = q4_hip_graph_end();
                if (!ok || !ga) {
                    if (ga) { q4_hip_graph_free(ga); ga = NULL; }
                    s->graph_off = 1;
                    fprintf(stderr,
                            "q4: devroute capture failed layer %d; eager\n",
                            il);
                    goto eager;
                }
                s->gA[il] = ga;
                s->gA_sp[il] = (uint8_t)qsa_sparse_ready(s, il);
                if (getenv("Q4_GNODES"))
                    fprintf(stderr, "q4: gnodes dev L%d %s %u\n", il,
                            q4_layer_is_qsa(g, il) ? "qsa" : "gdn",
                            q4_hip_graph_nodes(ga));
            }
            if (sp) q4_hip_mark_begin_slot(1);
            if (!q4_hip_graph_launch(ga)) {
                fprintf(stderr, "q4: devroute launch failed layer %d\n", il);
                s->graph_off = 1;
                goto eager;
            }
            if (sp) gpu_mark(1);
            goto done;
        }
        if (!ga) {
            /* Capture records without executing; instantiate and launch so
             * this token's real work still happens. */
            q4_moe_dev_warm(g, s->st, il); /* lazy sgate f32 conv: capture-illegal */
            if (!q4_hip_graph_begin()) {
                fprintf(stderr, "q4: graph begin failed; eager\n");
                s->graph_off = 1;
                goto eager;
            }
            bool ok = attn_hip(s, il, token) &&
                      q4_moe_gpu_a(g, s->st, il, d_mixed, d_block, d_moe);
            ga = q4_hip_graph_end();
            if (!ok || !ga) {
                if (ga) { q4_hip_graph_free(ga); ga = NULL; }
                s->graph_off = 1;
                fprintf(stderr, "q4: graph capture failed layer %d; eager\n",
                        il);
                goto eager;
            }
            s->gA[il] = ga;
            s->gA_sp[il] = (uint8_t)qsa_sparse_ready(s, il);
            if (getenv("Q4_GNODES"))
                fprintf(stderr, "q4: gnodes gA L%d %s %u\n", il,
                        q4_layer_is_qsa(g, il) ? "qsa" : "gdn",
                        q4_hip_graph_nodes(ga));
        }
        if (sp) q4_hip_mark_begin_slot(1);
        if (!q4_hip_graph_launch(ga)) {
            fprintf(stderr, "q4: graph A launch failed layer %d\n", il);
            s->graph_off = 1;
            goto eager;
        }
        if (sp) gpu_mark(1);   /* drains gA; hostpart sync is then a no-op */
        {
            static int announced = 0;
            if (!announced) {
                announced = 1;
                fprintf(stderr, "q4: decode graphs live (layer %d first)\n",
                        il);
            }
        }
        if (prof) { q4_hip_sync(); q4_prof_dec[1] += q4_now() - tt;
                    tt = q4_now(); }
        if (sp) { q4_prof_step[0] += q4_now() - tt; tt = q4_now(); }
        int rc = q4_moe_hostpart(g, s->ex, il);
        if (sp) {
            q4_prof_step[1] += q4_now() - tt; tt = q4_now();
            /* gA drained by hostpart's stream sync; read section events */
            for (int b = 0; b < 5; b++) {
                double ms = q4_hip_mark_elapsed(6 + b, 7 + b);
                if (ms > 0) q4_prof_gpu_sec[b] += ms;
            }
            /* QSA-only subsections (slots 12-14 only record on QSA layers):
             * prep = qkv gemv+norm/rope+idx q, sel = score+topk,
             * dec = gather+split attention. */
            static const int qs[3][2] = {{7, 12}, {12, 13}, {13, 14}};
            static int qsdbg = 0;
            for (int b = 0; b < 3; b++) {
                double ms = q4_hip_mark_elapsed(qs[b][0], qs[b][1]);
                if (ms > 0) q4_prof_gpu_sec[5 + b] += ms;
                else if (qsdbg < 6) {
                    qsdbg++;
                    fprintf(stderr, "q4: qs elapsed %d->%d = %.3f\n",
                            qs[b][0], qs[b][1], ms);
                }
            }
        }
        if (rc < 0) return false;
        if (rc == 0) {
            /* CPU dispatch refused; run routed experts on GPU eagerly using
             * the ids already computed by hostpart, then the tail graph's
             * work happens eagerly too. */
            if (!q4_moe_gpu_tail(g, s->ex, il, d_mixed, d_block, d_moe))
                return false;
            if (!q4_hip_hc_combine(s->d_res, d_block, d_inj, n_embd, hc))
                return false;
        } else {
            void *gb = s->gB[il];
            if (!gb) {
                if (!q4_hip_graph_begin()) { s->graph_off = 1; goto b_eager; }
                bool ok = q4_moe_gpu_b_hc(g, s->ex, il, s->d_res, d_inj, hc);
                if (!ok) fprintf(stderr, "q4: gB: moe_gpu_b_hc failed\n");
                gb = q4_hip_graph_end();
                if (!ok || !gb) {
                    if (gb) q4_hip_graph_free(gb);
                    s->graph_off = 1;
                    fprintf(stderr,
                            "q4: graph B capture failed layer %d; eager\n",
                            il);
                    goto b_eager;
                }
                s->gB[il] = gb;
                if (getenv("Q4_GNODES"))
                    fprintf(stderr, "q4: gnodes gB L%d %u\n", il,
                            q4_hip_graph_nodes(gb));
            }
            if (sp) q4_hip_mark_begin_slot(2);
            if (!q4_hip_graph_launch(gb)) {
                fprintf(stderr, "q4: graph B launch failed layer %d\n", il);
                s->graph_off = 1;
                goto b_eager;
            }
            if (sp) gpu_mark(2);
            goto done;
        b_eager:
            if (!q4_moe_gpu_b_hc(g, s->ex, il, s->d_res, d_inj, hc)) return false;
        }
    done:
        if (prof) { q4_hip_sync(); q4_prof_dec[3] += q4_now() - tt; }
        if (sp) q4_prof_step[2] += q4_now() - tt;
        return true;
    }
eager:
    q4_hip_clear(); /* stale error from an aborted capture must not fail us */
    if (sp) q4_hip_mark_begin_slot(4);
    if (!attn_hip(s, il, token)) return false;
    if (sp) gpu_mark(4);
    if (prof) { q4_hip_sync(); q4_prof_dec[1] += q4_now() - tt; tt = q4_now(); }
    if (sp) { q4_prof_step[3] += q4_now() - tt; tt = q4_now(); }
    if (sp) q4_hip_mark_begin_slot(5);
    if (!q4_moe_layer_dev(g, s->st, s->ex, il, d_mixed, d_block, d_moe)) {
        fprintf(stderr, "q4: moe failed layer %d\n", il);
        return false;
    }
    if (sp) gpu_mark(5);
    if (prof) { q4_hip_sync(); q4_prof_dec[2] += q4_now() - tt; tt = q4_now(); }
    if (sp) {
        q4_prof_step[4] += q4_now() - tt; tt = q4_now();
        /* Eager layers record the same boundary points; stream is drained
         * by the marks above. */
        for (int b = 0; b < 5; b++) {
            double ms = q4_hip_mark_elapsed(6 + b, 7 + b);
            if (ms > 0) q4_prof_gpu_sec[b] += ms;
        }
    }
    if (!q4_hip_hc_combine(s->d_res, d_block, d_inj, n_embd, hc)) {
        fprintf(stderr, "q4: hc-ffn combine failed layer %d\n", il);
        return false;
    }
    if (prof) { q4_hip_sync(); q4_prof_dec[3] += q4_now() - tt; }
    if (sp) q4_prof_step[5] += q4_now() - tt;
    return true;
}

static bool decode_hip(q4_sess *s, int32_t token, int32_t *out_id, float *out_logit,
                       bool want_logits) {
    const q4_gguf *g = s->g;
    const int prof = q4_prof();
    const int sp = step_prof();
    q4_hip_clear();
    if (sp) q4_hip_mark_begin();
    double tt = (prof || sp) ? q4_now() : 0;
    /* Devroute token boundary: drain deferred expert refills (stream is
     * drained here — last token's head synced) and surface dispatcher
     * errors.  No-op when devroute is off. */
    if (s->devroute > 0 && q4_moe_dev_tick(s->ex) < 0) return false;
    if (!embed_hip(s, token, s->n_kv, s->d_res)) return false;
    if (prof) { q4_hip_sync(); q4_prof_dec[0] += q4_now() - tt; tt = q4_now(); }
    if (sp) { q4_prof_step[6] += q4_now() - tt; tt = q4_now(); }
    if (s->n_kv >= s->max_kv) {
        fprintf(stderr, "q4: context full (%u)\n", s->max_kv);
        return false;
    }
    /* Publish pos for graphs that read it from the device counter. */
    q4_hip_pos_push(s->pos);
    for (uint32_t il = 0; il < g->n_layer; il++)
        if (!layer_hip(s, (int)il, token)) return false;
    if (prof) { q4_hip_sync(); tt = q4_now(); }
    s->n_kv++;
    s->pos++;
    if (sp) tt = q4_now();
    if (want_logits) {
        bool r = head_hip(s, out_id, out_logit);
        if (prof) q4_prof_dec[4] += q4_now() - tt;
        if (sp) {
            q4_prof_step[7] += q4_now() - tt;
            double ms = q4_hip_mark_ms();
            if (ms > 0) q4_prof_gpu_ms += ms;
        }
        /* A cpx_wait that timed out this token let the stream merge a
         * possibly-incomplete CPU output — don't emit that token.  Runs
         * on every path that can merge one (devroute mailboxes AND the
         * legacy moe_cpx_h wait), not just devroute. */
        if (r && q4_moe_dev_check() < 0) return false;
        return r;
    }
    if (sp) {
        double ms = q4_hip_mark_ms();
        if (ms > 0) q4_prof_gpu_ms += ms;
    }
    /* Same guard on the logits-free path: only reached when the caller
     * skipped the head, but a wedged pipeline must not be hidden. */
    if (q4_moe_dev_check() < 0) return false;
    if (out_id) *out_id = 0;
    if (out_logit) *out_logit = 0;
    return true;
}

/* GPU half of the decode head (graphable): hc norm mix, output gemv, async
 * D2H of the logit row into the pinned host buffer. */
static bool head_gpu_hip(q4_sess *s) {
    const q4_gguf *g = s->g;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t hc_dim = hc * n_embd;
    float *d_mixed = s->d_ws + 65536;
    const q4_tensor *hn = q4_find_tensor(g, "output_hc_norm.weight");
    const q4_tensor *hd = q4_find_tensor(g, "output_hc_down.weight");
    const q4_tensor *hu = q4_find_tensor(g, "output_hc_up.weight");
    float *xn = s->d_ws;
    float *lo = xn + hc_dim;
    float *gate = lo + g->hc_rank;
    float *gated = gate + hc_dim;
    if (!q4_hip_grouped_rms(s->d_res, f32d(s->st, hn), xn, n_embd, hc, g->rms_eps))
        return false;
    if (!mm_d_silu(s->st, hd, xn, lo, 1.f / (float)hc)) return false;
    if (!mm_d(s->st, hu, lo, gate)) return false;
    if (!q4_hip_mulsig_mean(xn, gate, d_mixed, n_embd, hc)) {
        if (!q4_hip_mul_sig(gated, xn, gate, hc_dim)) return false;
        if (!q4_hip_mean_hc(gated, d_mixed, n_embd, hc)) return false;
    }
    const q4_tensor *out = q4_find_tensor(g, "output.weight");
    if (!out) return false;
    if (!mm_d(s->st, out, d_mixed, s->d_logits)) return false;
    float *dst = s->h_logits ? s->h_logits : s->ws;
    return q4_hip_d2h_async(dst, s->d_logits,
                          (uint64_t)g->n_vocab * sizeof(float));
}

static bool head_hip(q4_sess *s, int32_t *out_id, float *out_logit) {
    const q4_gguf *g = s->g;
    if (!s->graph_off && !q4_prof() && s->h_logits) {
        void *gh = s->g_head;
        if (!gh) {
            if (q4_hip_graph_begin()) {
                bool ok = head_gpu_hip(s);
                gh = q4_hip_graph_end();
                if (ok && gh) {
                    s->g_head = gh;
                    if (getenv("Q4_GNODES"))
                        fprintf(stderr, "q4: gnodes head %u\n",
                                q4_hip_graph_nodes(gh));
                }
                else {
                    if (gh) q4_hip_graph_free(gh);
                    s->graph_off = 1;
                }
            } else s->graph_off = 1;
        }
        if (s->g_head) {
            if (step_prof()) q4_hip_mark_begin_slot(3);
            if (!q4_hip_graph_launch(s->g_head) || !q4_hip_stream_sync()) {
                s->graph_off = 1;
            } else {
                if (step_prof()) gpu_mark(3);
                const float *dst = s->h_logits;
                int32_t best = 0;
                float bv = dst[0];
                for (uint32_t i = 1; i < g->n_vocab; i++) {
                    if (dst[i] > bv) { bv = dst[i]; best = (int32_t)i; }
                }
                if (out_id) *out_id = best;
                if (out_logit) *out_logit = bv;
                return true;
            }
        }
    }
    /* eager (or fallback): gpu chain + sync + host argmax */
    if (!head_gpu_hip(s)) return false;
    if (!q4_hip_stream_sync()) return false;
    const float *dst = s->h_logits ? s->h_logits : s->ws;
    int32_t best = 0;
    float bv = dst[0];
    for (uint32_t i = 1; i < g->n_vocab; i++) {
        if (dst[i] > bv) { bv = dst[i]; best = (int32_t)i; }
    }
    if (out_id) *out_id = best;
    if (out_logit) *out_logit = bv;
    return true;
}

bool q4_sess_head(q4_sess *s, int32_t *out_id, float *out_logit) {
    if (!s || !s->d_res || !q4_hip_ok()) return false;
    return head_hip(s, out_id, out_logit);
}

static bool sess_kv_reserve(q4_sess *s, uint32_t n_pos);

/* ---- L1 lending: the prompt scratch lives in the arena tail ----
 * Carve order and sizes at chunk T: d_pref, d_mixb, d_yb, d_injb, d_pws,
 * MMQ gu/d weight gather, GDN scan h/oc/gates, hgemm weight / activation /
 * workspace.  q4_sess_lend_bytes is what
 * the engine sizes the tail from, so the two cannot disagree. */
#define LEND_N 13
#define LEND_ALIGN 256ull
static void lend_plan(const q4_gguf *g, uint32_t T, uint64_t sz[LEND_N]) {
    const uint64_t hc = g->hc_mult ? g->hc_mult : 4, ne = g->n_embd;
    sz[0] = (uint64_t)T * hc * ne * sizeof(float);
    sz[1] = (uint64_t)T * ne * sizeof(float);
    sz[2] = (uint64_t)T * ne * sizeof(float);
    sz[3] = (uint64_t)T * hc * sizeof(float);
    sz[4] = q4_pws_floats(g, T) * sizeof(float);
    q4_mmq_wbuf_need(g, &sz[5], &sz[6]);
    sz[7] = sz[8] = sz[9] = 0;
    const char *gr = getenv("Q4_GDN_REC");
    if (!(gr && gr[0] == '0') && T >= 64) {
        const uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
        const uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
        const uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
        const uint32_t d_v =
            n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
        q4_hip_gdn_scan_need(T, d_k * n_k_h * 2 + d_v * n_v_h, n_v_h, d_v,
                             &sz[7]);
    }
    /* widest dense GEMM input: the hc down projection (hc * n_embd) */
    q4_hgemm_lend_need(T, hc * ne, &sz[10], &sz[11], &sz[12]);
}

uint64_t q4_sess_lend_bytes(const q4_gguf *g, uint32_t T) {
    if (!g || !T) return 0;
    uint64_t sz[LEND_N], tot = 0;
    lend_plan(g, T, sz);
    for (int i = 0; i < LEND_N; i++)
        tot += (sz[i] + LEND_ALIGN - 1) & ~(LEND_ALIGN - 1);
    return tot;
}

/* The permanent 8-row set verify batches use (MTP_MAX_DRAFT+1 rows). */
static bool sess_mini(q4_sess *s) {
    s->lend_mode = 1;
    if (!s->m_pref) {
        const q4_gguf *g = s->g;
        const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
        const uint32_t n_embd = g->n_embd;
        s->m_pws_floats = q4_pws_floats(g, 8);
        s->m_pref = q4_hip_malloc((size_t)8 * hc * n_embd * sizeof(float));
        s->m_mixb = q4_hip_malloc((size_t)8 * n_embd * sizeof(float));
        s->m_yb = q4_hip_malloc((size_t)8 * n_embd * sizeof(float));
        s->m_injb = q4_hip_malloc((size_t)8 * hc * sizeof(float));
        s->m_pws = q4_hip_malloc((size_t)s->m_pws_floats * sizeof(float));
    }
    if (!s->m_pref || !s->m_mixb || !s->m_yb || !s->m_injb || !s->m_pws)
        return false;
    if (!s->lent) {
        s->d_pref = s->m_pref;
        s->d_mixb = s->m_mixb;
        s->d_yb = s->m_yb;
        s->d_injb = s->m_injb;
        s->d_pws = s->m_pws;
        s->pref_chunk = 8;
        s->pws_floats = s->m_pws_floats;
    }
    return true;
}

static bool sess_lend_begin(q4_sess *s) {
    s->lend_mode = 1;
    uint64_t have = 0;
    uint8_t *b = (uint8_t *)q4_expert_lend_begin(s->ex, &have);
    const uint32_t T = q4_pref_chunk();
    uint64_t sz[LEND_N];
    lend_plan(s->g, T, sz);
    const uint64_t need = q4_sess_lend_bytes(s->g, T);
    if (!b || need > have) {
        fprintf(stderr, "q4: L1 lend: tail %.0f MiB < scratch %.0f MiB\n",
                (double)have / 1048576.0, (double)need / 1048576.0);
        return false;
    }
    uint8_t *p[LEND_N];
    for (int i = 0; i < LEND_N; i++) {
        p[i] = sz[i] ? b : NULL;
        b += (sz[i] + LEND_ALIGN - 1) & ~(LEND_ALIGN - 1);
    }
    s->d_pref = (float *)p[0];
    s->d_mixb = (float *)p[1];
    s->d_yb = (float *)p[2];
    s->d_injb = (float *)p[3];
    s->d_pws = (float *)p[4];
    s->pref_chunk = T;
    s->pws_floats = q4_pws_floats(s->g, T);
    if (p[5] && p[6]) {
        if (!q4_mmq_wbuf_lend(p[5], sz[5], p[6], sz[6])) return false;
    } else {
        (void)q4_mmq_wbuf_lend(NULL, 0, NULL, 0);
    }
    if (p[7]) {
        void *gp[3] = {p[7], p[8], p[9]};
        q4_hip_gdn_scan_lend(gp, &sz[7]);
    } else {
        q4_hip_gdn_scan_lend(NULL, NULL);
    }
    if (p[10] && p[11] && p[12])
        q4_hgemm_lend(p[10], p[11], sz[11], p[12]);
    else
        q4_hgemm_lend(NULL, NULL, 0, NULL);
    s->lent = 1;
    return true;
}

bool q4_sess_lend_end(q4_sess *s) {
    if (!s || !s->lent) return true;
    q4_hip_sync();
    (void)q4_mmq_wbuf_lend(NULL, 0, NULL, 0);
    q4_hip_gdn_scan_lend(NULL, NULL);
    q4_hgemm_lend(NULL, NULL, 0, NULL);
    s->lent = 0;
    s->d_pref = s->d_mixb = s->d_yb = s->d_injb = s->d_pws = NULL;
    double t0 = q4_now();
    int put = q4_expert_lend_end(s->ex);
    static int logged;
    if (!logged || getenv("Q4_PROFILE")) {
        logged = 1;
        fprintf(stderr, "q4: L1 lend: %d experts refilled in %.0f ms\n", put,
                (q4_now() - t0) * 1e3);
    }
    if (!sess_mini(s)) return false;
    /* Verify graphs bake the 8-row set's pointers: captured once, here,
     * outside any decode timer. */
    if (s->mtp_max > 0)
        q4_moe_verify_capture(s->g, s->st, s->ex, s->d_mixb, s->d_yb,
                              s->d_pws, s->pws_floats);
    return put >= 0;
}

bool q4_sess_prefill(q4_sess *s, const int32_t *toks, int n) {
    if (!s || !toks || n <= 0) return false;
    if (!sess_kv_reserve(s, s->n_kv + (uint32_t)n)) return false;
    if (!s->d_res || !q4_hip_ok()) {
        for (int i = 0; i < n; i++)
            if (!q4_sess_decode_ex(s, toks[i], NULL, NULL, false)) return false;
        return true;
    }
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    if (s->d_res && q4_hip_ok() && q4_expert_lend_on(s->ex)) {
        /* Never fall back to a hipMalloc'd scratch here: the arena already
         * holds that VRAM. */
        if (!s->lent && !sess_lend_begin(s)) return false;
        if ((uint32_t)n > s->pref_chunk) {
            fprintf(stderr, "q4: prefill batch %d > lent chunk %u\n", n,
                    s->pref_chunk);
            return false;
        }
    } else if (!s->d_pref || (uint32_t)n > s->pref_chunk) {
        /* The MTP verify path allocates an 8-row mini set; a real prefill
         * needs the full chunk — free the mini and grow. */
        if (s->d_pref) {
            q4_hip_free(s->d_pref);
            q4_hip_free(s->d_mixb);
            q4_hip_free(s->d_yb);
            q4_hip_free(s->d_injb);
            q4_hip_free(s->d_pws);
            s->d_pref = s->d_mixb = s->d_yb = s->d_injb = s->d_pws = NULL;
        }
        s->pref_chunk = q4_pref_chunk();
        if ((uint32_t)n > s->pref_chunk) s->pref_chunk = (uint32_t)n;
        s->pws_floats = q4_pws_floats(g, s->pref_chunk);
        s->d_pref = q4_hip_malloc((size_t)s->pref_chunk * hc_dim * sizeof(float));
        s->d_mixb = q4_hip_malloc((size_t)s->pref_chunk * n_embd * sizeof(float));
        s->d_yb = q4_hip_malloc((size_t)s->pref_chunk * n_embd * sizeof(float));
        s->d_injb = q4_hip_malloc((size_t)s->pref_chunk * hc * sizeof(float));
        s->d_pws = q4_hip_malloc((size_t)s->pws_floats * sizeof(float));
    }
    if (!s->d_pref || !s->d_mixb || !s->d_yb || !s->d_injb || !s->d_pws ||
        n > (int)s->pref_chunk) {
        for (int i = 0; i < n; i++)
            if (!q4_sess_decode_ex(s, toks[i], NULL, NULL, false)) return false;
        return true;
    }
    /* Layer-ahead readahead runs during the whole prefill. */
    q4_ahead ahead;
    q4_expert_ahead_begin(&ahead, s->ex, 0);
    bool ok = prefill_hip(s, toks, n, &ahead);
    q4_expert_ahead_end(&ahead);
    /* Never leave a prefetch thread owning the io pool into decode or an
     * aborted prefill. */
    q4_expert_pf_join(s->ex);
    /* Prompt prefill owns the final mix/y/scratch pointers. Verify graphs
     * are recorded here, outside the MTP decode timer. A later grow frees
     * these buffers; the capture call rebuilds when the pointers differ. */
    if (ok && !s->lent)
        q4_moe_verify_capture(s->g, s->st, s->ex, s->d_mixb, s->d_yb, s->d_pws,
                              s->pws_floats);
    return ok;
}

/* Set by the engine once it has seen NaN logits: every later pass checks
 * the rows after each layer stage and names the first non-finite one. */
int q4_nan_trap;
static int nan_trap_hit;
static void nan_trap(uint32_t il, const char *stage, const float *d, uint64_t n,
                     uint32_t nt) {
    if (!q4_nan_trap || nan_trap_hit || !d || !n) return;
    q4_hip_sync();
    float *h = malloc((size_t)n * sizeof(float));
    if (!h) return;
    if (q4_hip_d2h(h, d, (size_t)n * sizeof(float))) {
        uint64_t bad = 0, first = n;
        for (uint64_t i = 0; i < n; i++)
            if (!isfinite(h[i])) {
                if (first == n) first = i;
                bad++;
            }
        if (bad) {
            fprintf(stderr, "q4 nan trap: layer %u %s: %llu of %llu non-finite "
                    "(first at %llu, row %llu of %u)\n", il, stage,
                    (unsigned long long)bad, (unsigned long long)n,
                    (unsigned long long)first,
                    (unsigned long long)(first / (n / nt)), nt);
            nan_trap_hit = 1;
        }
    }
    free(h);
}

static bool prefill_hip(q4_sess *s, const int32_t *toks, int n, q4_ahead *ahead) {
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc_dim = hc * n_embd;
    q4_hip_clear();
    /* Lazy dispatcher bring-up: with MTP on, decode_hip/layer_hip may never
     * run (every pass is draft or verify), so the devroute dispatcher must
     * be started here — q4_moe_dev_route also powers it up when the
     * decode-side path itself stays disabled (Q4_DEVROUTE=0). */
    if (!q4_moe_dev_active())
        (void)q4_moe_dev_route(g, s->ex);
    /* Pass boundary under devroute-n: the previous pass's logits readback
     * drained the compute stream, so deferred devtab evictions (cleared
     * last pass) may safely refill now — one tick per pass. */
    double vtk0 = q4_now();
    if (q4_moe_dev_active() && q4_moe_dev_tick(s->ex) < 0) return false;
    s->vprof_tick = q4_now() - vtk0;
    uint32_t base = s->n_kv;
    uint32_t nt = (uint32_t)n;
    /* Stage layer 0's experts under the embedding loop + layer-0 attention.
     * Later layers self-launch inside q4_moe_layer_dev_n. */
    if (nt >= 256) (void)q4_expert_pf_begin(s->ex, 0);
    /* Start this chunk's PLE gather on a thread so it overlaps embed +
     * layer 0; ple_chunk_d joins it at the PLE layer (exact-match consume,
     * so a wrong guess costs nothing but the gather itself).  Skipped for
     * tiny windows (verify/ppl): the gather is a millisecond there. */
    if (s->ple && ple_ahead_on() && nt >= 64)
        (void)ple_job_start(s, toks, n, s->ple_hist);
    for (int t = 0; t < n; t++) {
        if (!embed_hip(s, toks[t], base + (uint32_t)t,
                       s->d_pref + (size_t)t * hc_dim))
            return false;
    }
    const uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    const uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    const uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    const uint32_t d_v = n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
    const uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    const uint32_t n_head = g->n_head ? g->n_head : 24;
    const uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    const uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    const uint32_t n_rot = g->n_rot ? g->n_rot : 64;
    float rope_base = g->rope_freq_base ? g->rope_freq_base : 1e7f;
    float eps = g->rms_eps ? g->rms_eps : 1e-6f;
    uint32_t conv_hist_ = (g->ssm_d_conv ? g->ssm_d_conv : 4) > 1
                              ? (g->ssm_d_conv ? g->ssm_d_conv : 4) - 1
                              : 1;
    static int vtl_env = -1;
    if (vtl_env < 0)
        vtl_env = getenv("Q4_VTL") && getenv("Q4_VTL")[0] == '1';
    const int vtl = vtl_env && nt >= 2 && nt <= 8;
    if (vtl) q4_moe_vtl_rows(nt);
    /* Q4_COMB_FUSE (default on): each hc combine also writes the grouped
     * rms the next hc mix starts with (bit-identical, one launch less). */
    static int comb_env = -1;
    if (comb_env < 0)
        comb_env = !(getenv("Q4_COMB_FUSE") && getenv("Q4_COMB_FUSE")[0] == '0');
    const bool comb_fuse = comb_env != 0;
    bool xn_ready = false;
    /* Q4_SNAP_SCAN (default on): each layer's scan writes its own incoming
     * S / conv state into the snapshot as it loads them, instead of one
     * 113 MB copy ahead of the layers. */
    static int snap_scan = -1;
    if (snap_scan < 0)
        snap_scan = !(getenv("Q4_SNAP_SCAN") && getenv("Q4_SNAP_SCAN")[0] == '0');
    const bool snap_in_scan = s->mtp_stash && s->d_gdn_S_snap && snap_scan &&
                              nt < 64 && d_k == d_v;
    /* Q4_SNAP_CHECK=1 (diagnostic): also copy the state the old way into a
     * side buffer and compare it with the in-scan snapshot after the pass. */
    static int snap_chk = -1;
    static float *snap_ref;
    if (snap_chk < 0)
        snap_chk = getenv("Q4_SNAP_CHECK") && getenv("Q4_SNAP_CHECK")[0] == '1';
    const size_t snap_n = (size_t)s->n_gdn * n_v_h * d_v * d_v;
    if (snap_chk && snap_in_scan) {
        if (!snap_ref) snap_ref = q4_hip_malloc(snap_n * sizeof(float));
        if (snap_ref) (void)q4_hip_copy_async(snap_ref, s->d_gdn_S, snap_n);
    }
    if (s->mtp_stash && s->d_gdn_S_snap) {
        /* Pre-batch snapshot: rollback point for a rejected verify prefix.
         * Queued ahead of the layers that mutate S. The closing d_res copy
         * drains the stream before any host read of the snap. */
        s->n_kv0 = base;
        s->pos0 = (int32_t)base;
        if (!snap_in_scan &&
            (!q4_hip_copy_async(s->d_gdn_S_snap, s->d_gdn_S,
                                (uint64_t)s->n_gdn * n_v_h * d_v * d_v) ||
             !q4_hip_copy_async(s->d_gdn_conv_snap, s->d_gdn_conv,
                                (uint64_t)s->n_gdn * conv_hist_ * conv_ch)))
            return false;
    }
    nan_trap_hit = 0;
    if (q4_nan_trap) {
        fprintf(stderr, "q4 nan trap: pass of %d tokens at %u:", n, base);
        for (int t = 0; t < n && t < 8; t++) fprintf(stderr, " %d", toks[t]);
        fputc('\n', stderr);
    }
    nan_trap(0, "embedding (d_pref before layer 0)", s->d_pref,
             (uint64_t)nt * hc_dim, nt);
    for (uint32_t il = 0; il < g->n_layer; il++) {
        SEC_T0;
        if (vtl) q4_moe_vtl_stamp((int)il, 0);
        if (ahead) ahead->cur = il;
        if (g->ple_layer == il && s->ple) {
            static int ple_batch = -1;
            if (ple_batch < 0)
                ple_batch = !(getenv("Q4_PLE_BATCH") &&
                              getenv("Q4_PLE_BATCH")[0] == '0');
            /* mtp_stash also takes the batched path: ple_chunk_d emits the
             * per-token conv-state snapshots itself. */
            if (ple_batch) {
                if (!ple_chunk_d(s, (int)il, toks, (uint32_t)n))
                    return false;
            } else if (!ple_seq_d(s, (int)il, toks, (uint32_t)n))
                return false;
            nan_trap(il, "after PLE (d_pref)", s->d_pref, (uint64_t)nt * hc_dim, nt);
        }
        SEC_ADD(0);
        if (!hc_mix_dn_x(s, s->d_pref, s->d_mixb, s->d_injb, (int)il, true, nt,
                         xn_ready))
            return false;
        xn_ready = false;
        SEC_ADD(1);
        if (q4_layer_is_qsa(g, (int)il)) {
            float *qg = s->d_pws;
            float *k = qg + (size_t)nt * n_head * hd * 2;
            float *v = k + (size_t)nt * n_kvh * hd;
            float *qbat = v + (size_t)nt * n_kvh * hd;
            float *gbat = qbat + (size_t)nt * n_head * hd;
            float *attn = qg; /* overwrite packed Q after prep */
            /* q / k / v share one activation quantization (int8 n-row) */
            q4_mmvq_hold(s->d_mixb);
            const bool qkv_ok =
                mm_d_n(s->st, T(g, "blk.%d.attn_q.weight", (int)il), s->d_mixb,
                       qg, nt) &&
                mm_d_n(s->st, T(g, "blk.%d.attn_k.weight", (int)il), s->d_mixb,
                       k, nt) &&
                mm_d_n(s->st, T(g, "blk.%d.attn_v.weight", (int)il), s->d_mixb,
                       v, nt);
            q4_mmvq_release();
            if (!qkv_ok) return false;
            const float *qn = f32d(s->st, T(g, "blk.%d.attn_q_norm.weight", (int)il));
            const float *kn = f32d(s->st, T(g, "blk.%d.attn_k_norm.weight", (int)il));
            uint32_t qi = s->qsa_ix[il];
            size_t kvw = (size_t)n_kvh * hd;
            if (!q4_hip_qsa_q_prep_n(qg, qn, qbat, gbat, n_head, hd, n_rot,
                                     (int32_t)base, rope_base, eps, nt))
                return false;
            if (!q4_hip_qsa_k_prep_n(k, kn, n_kvh, hd, n_rot, (int32_t)base,
                                     rope_base, eps, nt))
                return false;
            if (!s->d_qsa_k || !s->d_qsa_v) return false;
            uint8_t *kbase = kv_at(s, 0, qi, 0);
            uint8_t *vbase = kv_at(s, 1, qi, 0);
            if (!kv_write(s, kbase + (size_t)base * s->kv_bpt, k,
                          (uint64_t)nt * kvw) ||
                !kv_write(s, vbase + (size_t)base * s->kv_bpt, v,
                          (uint64_t)nt * kvw))
                return false;
            /* The indexer scratch (score/sel) is ctx-sized for 512 tokens:
             * attend in 512-token slices instead of growing it. */
            for (uint32_t t0 = 0; t0 < nt; t0 += 512) {
                uint32_t sn = nt - t0 < 512 ? nt - t0 : 512;
                if (!qsa_attend(s, (int)il, s->d_mixb + (size_t)t0 * n_embd,
                                qbat + (size_t)t0 * n_head * hd,
                                gbat + (size_t)t0 * n_head * hd,
                                attn + (size_t)t0 * n_head * hd, sn,
                                base + t0))
                    return false;
            }
            if (!mm_d_n(s->st, T(g, "blk.%d.attn_output.weight", (int)il), attn,
                        s->d_yb, nt))
                return false;
            SEC_ADD(2);
        } else {
            float *qkv = s->d_pws;
            float *z = qkv + (size_t)nt * conv_ch;
            float *core = z + (size_t)nt * n_v_h * d_v;
            float *beta = core + (size_t)nt * n_v_h * d_v;
            float *alpha = beta + (size_t)nt * n_v_h;
            double g0 = q4_now();
            /* the four input projections share one activation
             * quantization (int8 n-row) */
            q4_mmvq_hold(s->d_mixb);
            const q4_tensor *tbe = T(g, "blk.%d.ssm_beta.weight", (int)il);
            const q4_tensor *tal = T(g, "blk.%d.ssm_alpha.weight", (int)il);
            /* beta + alpha in one launch when both are bf16 */
            const bool ba2 = tbe && tal && tbe->ggml_type == Q4_T_BF16 &&
                tal->ggml_type == Q4_T_BF16 && tbe->ne[0] == n_embd &&
                tal->ne[0] == n_embd;
            const bool in_ok =
                mm_d_n(s->st, T(g, "blk.%d.attn_qkv.weight", (int)il),
                       s->d_mixb, qkv, nt) &&
                mm_d_n(s->st, T(g, "blk.%d.attn_gate.weight", (int)il),
                       s->d_mixb, z, nt) &&
                ((ba2 && q4_hip_gemv_bf16_n2(
                             q4_store_dev(s->st, tbe), beta,
                             (uint32_t)tbe->ne[1], 0.f,
                             q4_store_dev(s->st, tal), alpha,
                             (uint32_t)tal->ne[1], 0.f, s->d_mixb, n_embd,
                             nt)) ||
                 (mm_d_n(s->st, tbe, s->d_mixb, beta, nt) &&
                  mm_d_n(s->st, tal, s->d_mixb, alpha, nt)));
            q4_mmvq_release();
            if (!in_ok) return false;
            /* beta stays raw: gdn_step applies the sigmoid itself. */
            if (q4_prof()) q4_hip_sync();
            double g1 = q4_now();
            const uint32_t gi3 = s->gdn_ix[il];
            const uint32_t ksz3 = g->ssm_d_conv ? g->ssm_d_conv : 4;
            float *cstate3 =
                s->d_gdn_conv + (size_t)gi3 * (ksz3 - 1) * conv_ch;
            float *Sbase3 =
                s->d_gdn_S + (size_t)gi3 * n_v_h * d_v * d_v;
            float eps3 = g->rms_eps ? g->rms_eps : 1e-6f;
            int scan_ok = 0;
            static int gscan = -1;
            if (gscan < 0)
                gscan = !(getenv("Q4_GDN_SCAN") &&
                          getenv("Q4_GDN_SCAN")[0] == '0');
            if (snap_in_scan && !(d_k == d_v && gscan &&
                                  q4_hip_gdn_scan_snap(
                                      s->d_gdn_S_snap + (size_t)gi3 * n_v_h * d_v * d_v,
                                      s->d_gdn_conv_snap + (size_t)gi3 * (ksz3 - 1) * conv_ch,
                                      nt))) {
                /* no single-kernel scan for this layer: copy its state */
                if (!q4_hip_copy_async(s->d_gdn_S_snap + (size_t)gi3 * n_v_h * d_v * d_v,
                                       Sbase3, (uint64_t)n_v_h * d_v * d_v) ||
                    !q4_hip_copy_async(s->d_gdn_conv_snap + (size_t)gi3 * (ksz3 - 1) * conv_ch,
                                       cstate3, (uint64_t)(ksz3 - 1) * conv_ch))
                    return false;
            }
            /* the MTP stash of this layer's token inputs rides in the scan
             * (no separate copy kernel) when the single-kernel path runs */
            const bool want_stash =
                s->mtp_stash && s->d_gdn_stash && n <= s->mtp_max;
            float *const sb_in =
                want_stash ? s->d_gdn_stash + (size_t)s->gdn_ix[il] *
                                                  s->mtp_max * s->gdn_stash_row
                           : NULL;
            const bool stash_scan =
                want_stash && d_k == d_v && gscan &&
                q4_hip_gdn_scan_stash(sb_in, s->mtp_max, nt);
            if (d_k == d_v && gscan) {
                scan_ok = q4_hip_gdn_scan(
                    cstate3, qkv, f32d(s->st, T(g, "blk.%d.ssm_conv1d.weight",
                                                (int)il)),
                    conv_ch, ksz3, Sbase3, beta, alpha,
                    f32d(s->st, T(g, "blk.%d.ssm_a", (int)il)),
                    f32d(s->st, T(g, "blk.%d.ssm_dt.bias", (int)il)),
                    f32d(s->st, T(g, "blk.%d.ssm_norm.weight", (int)il)), z,
                    core, nt, n_v_h, n_k_h, d_v, eps3,
                    1.f / sqrtf((float)d_k));
                if (scan_ok) {
                    s->n_kv = base + nt - 1;
                    s->pos = (int32_t)s->n_kv;
                }
            }
            if (!scan_ok) {
                for (int t = 0; t < n; t++) {
                    s->n_kv = base + (uint32_t)t;
                    s->pos = (int32_t)s->n_kv;
                    if (!gdn_apply_d(s, (int)il, qkv + (size_t)t * conv_ch,
                                     z + (size_t)t * n_v_h * d_v,
                                     beta + (size_t)t * n_v_h,
                                     alpha + (size_t)t * n_v_h,
                                     core + (size_t)t * n_v_h * d_v))
                        return false;
                }
            }
            if (q4_prof()) q4_hip_sync();
            double g2 = q4_now();
            if (want_stash && stash_scan && scan_ok &&
                getenv("Q4_STASH_CHECK") && il == 2) {
                /* diagnostic: the scan's stash vs the copy it replaces */
                static float *hbuf[2];
                static size_t hcap;
                const size_t ne = (size_t)s->mtp_max * s->gdn_stash_row;
                if (hcap < ne) {
                    free(hbuf[0]); free(hbuf[1]);
                    hbuf[0] = malloc(ne * 4); hbuf[1] = malloc(ne * 4);
                    hcap = ne;
                }
                q4_hip_stream_sync();
                q4_hip_d2h(hbuf[0], sb_in, ne * 4);
                float *sb = sb_in;
                float *const dst[4] = {
                    sb, sb + (size_t)s->mtp_max * conv_ch,
                    sb + (size_t)s->mtp_max * (conv_ch + n_v_h * d_v),
                    sb + (size_t)s->mtp_max * (conv_ch + n_v_h * d_v + n_v_h)};
                const float *const src[4] = {qkv, z, beta, alpha};
                const uint64_t len[4] = {(uint64_t)n * conv_ch,
                                         (uint64_t)n * n_v_h * d_v,
                                         (uint64_t)n * n_v_h,
                                         (uint64_t)n * n_v_h};
                q4_hip_copy_multi(4, dst, src, len);
                q4_hip_stream_sync();
                q4_hip_d2h(hbuf[1], sb_in, ne * 4);
                size_t nd = 0, tot = 0;
                for (int k = 0; k < 4; k++) {
                    const size_t o = dst[k] - sb;
                    for (size_t i = 0; i < len[k]; i++, tot++)
                        nd += memcmp(&hbuf[0][o + i], &hbuf[1][o + i], 4) != 0;
                }
                static int shown;
                if (shown++ < 5)
                    fprintf(stderr, "q4 stash check: n %d  %zu of %zu differ\n",
                            n, nd, tot);
            }
            if (want_stash && !(stash_scan && scan_ok)) {
                /* Stash the per-token GDN inputs (D2D, stream-ordered so
                 * they complete before the next layer reuses d_pws) so a
                 * rejected verify prefix can be replayed onto the
                 * pre-batch state. gdn_apply_d is a pure function of
                 * state + these inputs. */
                float *sb = s->d_gdn_stash +
                            (size_t)s->gdn_ix[il] * s->mtp_max * s->gdn_stash_row;
                float *const dst[4] = {
                    sb, sb + (size_t)s->mtp_max * conv_ch,
                    sb + (size_t)s->mtp_max * (conv_ch + n_v_h * d_v),
                    sb + (size_t)s->mtp_max * (conv_ch + n_v_h * d_v + n_v_h)};
                const float *const src[4] = {qkv, z, beta, alpha};
                const uint64_t len[4] = {(uint64_t)n * conv_ch,
                                         (uint64_t)n * n_v_h * d_v,
                                         (uint64_t)n * n_v_h,
                                         (uint64_t)n * n_v_h};
                if (!q4_hip_copy_multi(4, dst, src, len)) return false;
            }
            if (!mm_d_n(s->st, T(g, "blk.%d.ssm_out.weight", (int)il), core,
                        s->d_yb, nt))
                return false;
            if (q4_prof()) q4_hip_sync();
            double g3 = q4_now();
            if (q4_prof() && (il == 2 || il == 46))
                fprintf(stderr, "  gdn L%d: proj %.1f  seq %.1f  out %.1f ms\n",
                        il, (g1 - g0) * 1e3, (g2 - g1) * 1e3, (g3 - g2) * 1e3);
            SEC_ADD(7);
        }
        if (comb_fuse) {
            /* attention combine + the FFN-side grouped rms in one launch */
            if (!q4_hip_hc_comb_rms_n(
                    s->d_pref, s->d_yb, NULL, s->d_injb,
                    f32d(s->st, T(g, "blk.%d.hc_ffn_norm.weight", (int)il)),
                    s->d_pws, n_embd, hc, eps, nt))
                return false;
        } else if (!q4_hip_hc_combine_n(s->d_pref, s->d_yb, s->d_injb, n_embd,
                                        hc, nt))
            return false;
        SEC_ADD(3);
        nan_trap(il, "after attention+combine (d_pref)", s->d_pref,
                 (uint64_t)nt * hc * n_embd, nt);
        if (!hc_mix_dn_x(s, s->d_pref, s->d_mixb, s->d_injb, (int)il, false, nt,
                         comb_fuse))
            return false;
        SEC_ADD(4);
        nan_trap(il, "ffn input (d_mixb)", s->d_mixb, (uint64_t)nt * n_embd, nt);
        if (vtl) q4_moe_vtl_stamp((int)il, 1);
        (void)q4_moe_take_hy();   /* nothing stale may be merged below */
        if (!q4_moe_layer_dev_n(g, s->st, s->ex, (int)il, s->d_mixb, s->d_yb,
                                s->d_pws, nt, s->pws_floats)) {
            fprintf(stderr, "q4: batched moe failed layer %u\n", il);
            return false;
        }
        SEC_ADD(5);
        {
            /* MoE combine (+ the deferred CPU-miss rows), fused with the next
             * layer's attention-side grouped rms unless PLE rewrites d_pref
             * in between. */
            const float *hy = q4_moe_take_hy();
            const uint32_t nx = il + 1;
            nan_trap(il, "moe GPU part (d_yb)", s->d_yb, (uint64_t)nt * n_embd, nt);
            nan_trap(il, "moe CPU-miss rows (hy)", hy, (uint64_t)nt * n_embd, nt);
            if (comb_fuse && nx < g->n_layer &&
                !(g->ple_layer == nx && s->ple)) {
                if (!q4_hip_hc_comb_rms_n(
                        s->d_pref, s->d_yb, hy, s->d_injb,
                        f32d(s->st,
                             T(g, "blk.%d.hc_attn_norm.weight", (int)nx)),
                        s->d_pws, n_embd, hc, eps, nt))
                    return false;
                xn_ready = true;
            } else if (hy) {
                if (!q4_hip_hc_comb_rms_n(s->d_pref, s->d_yb, hy, s->d_injb,
                                          NULL, NULL, n_embd, hc, eps, nt))
                    return false;
            } else if (!q4_hip_hc_combine_n(s->d_pref, s->d_yb, s->d_injb,
                                            n_embd, hc, nt))
                return false;
        }
        SEC_ADD(6);
        nan_trap(il, "after moe combine (d_pref)", s->d_pref,
                 (uint64_t)nt * hc * n_embd, nt);
        if (vtl) q4_moe_vtl_stamp((int)il, 7);
        if (s->pref_hook) s->pref_hook(il + 1, g->n_layer, s->pref_hook_u);
    }
    /* A cpx_wait timeout during the pass may have merged an incomplete
     * CPU-miss output into d_pref — fail instead of committing it. */
    if (q4_moe_dev_active() && q4_moe_dev_check() < 0) return false;
    s->n_kv = base + nt;
    s->pos = (int32_t)s->n_kv;
    if (q4_prof()) {
        fprintf(stderr,
                "q4 prof: ple %.3f hcA %.3f qsa %.3f c1 %.3f hcF %.3f moe %.3f "
                "c2 %.3f gdn %.3f\n",
                q4_prof_sec[0], q4_prof_sec[1], q4_prof_sec[2], q4_prof_sec[3],
                q4_prof_sec[4], q4_prof_sec[5], q4_prof_sec[6], q4_prof_sec[7]);
        memset(q4_prof_sec, 0, sizeof(q4_prof_sec));
    }
    if (snap_chk && snap_in_scan && snap_ref) {
        float *a = malloc(snap_n * 4), *b = malloc(snap_n * 4);
        if (a && b && q4_hip_d2h(a, snap_ref, snap_n * 4) &&
            q4_hip_d2h(b, s->d_gdn_S_snap, snap_n * 4)) {
            size_t bad = 0;
            for (size_t i = 0; i < snap_n; i++) bad += a[i] != b[i];
            fprintf(stderr, "q4 snap check: %zu / %zu differ\n", bad, snap_n);
        }
        free(a);
        free(b);
    }
    s->vprof_enq = q4_now();
    return q4_hip_copy(s->d_res, s->d_pref + (size_t)(n - 1) * hc_dim, hc_dim);
}

/* ---- KV paging ------------------------------------------------------- */
#define KV_PAGE_STEP 16384u
static int g_kv_paging;

void q4_sess_set_kv_paging(int on) { g_kv_paging = on; }

void q4_sess_set_kv_donor(q4_sess *s, bool (*fn)(void *, uint32_t),
                          void *ctx) {
    if (!s) return;
    s->kv_donor = fn;
    s->kv_donor_ctx = ctx;
}

/* Region r (0 K, 1 V, 2 raw keys, 3 pooled keys) of QSA layer qi: base,
 * byte offset of the layer and its bytes at `cap` positions. */
static void *kv_region(const q4_sess *s, int r, uint32_t qi, uint32_t cap,
                       uint64_t *off, uint64_t *len) {
    const uint64_t raw_b = (uint64_t)s->idx_dim * sizeof(uint16_t);
    const uint64_t pool_b = (uint64_t)s->idx_dim * sizeof(float);
    const uint32_t ratio = s->idx_ratio ? s->idx_ratio : 4;
    switch (r) {
    case 0:
    case 1:
        *off = (uint64_t)qi * s->max_kv * s->kv_bpt;
        *len = (uint64_t)cap * s->kv_bpt;
        return r ? (void *)s->d_qsa_v : (void *)s->d_qsa_k;
    case 2:
        *off = (uint64_t)qi * s->max_kv * raw_b;
        *len = (uint64_t)cap * raw_b;
        return s->d_idx_raw;
    default:
        *off = (uint64_t)qi * s->max_blocks * pool_b;
        *len = (uint64_t)((cap + ratio - 1) / ratio) * pool_b;
        return s->d_idx_pool;
    }
}

/* Map the paged caches up to n_pos positions (rounded up to a step),
 * asking the engine for chunks when the pool is short. */
static bool sess_kv_reserve(q4_sess *s, uint32_t n_pos) {
    if (!s->kv_paged || n_pos <= s->kv_cap) return true;
    uint32_t cap = (n_pos + KV_PAGE_STEP - 1) / KV_PAGE_STEP * KV_PAGE_STEP;
    if (cap > s->max_kv) cap = s->max_kv;
    uint32_t need = 0, nr = 0;
    void *vas[4 * Q4_MAX_LAYER];
    uint64_t offs[4 * Q4_MAX_LAYER], lens[4 * Q4_MAX_LAYER];
    for (int r = 0; r < 4; r++)
        for (uint32_t qi = 0; qi < s->n_qsa && qi < Q4_MAX_LAYER; qi++) {
            uint64_t off, len;
            void *va = kv_region(s, r, qi, cap, &off, &len);
            if (!va) continue;
            need += q4_hip_vreg_need(va, off, len);
            vas[nr] = va;
            offs[nr] = off;
            lens[nr++] = len;
        }
    /* the donor hands over slots until the pool holds every chunk these
     * ranges need (pinned: their own homes, see q4_hip_vreg_short) */
    for (uint32_t sh; (sh = q4_hip_vreg_short(vas, offs, lens, (int)nr)) > 0;)
        if (!s->kv_donor || !s->kv_donor(s->kv_donor_ctx, sh)) {
            fprintf(stderr, "q4: KV paging: %u chunks for %u positions, pool %u, "
                    "%u short\n", need, cap, q4_hip_vpool_n(), sh);
            return false;
        }
    /* one position step at a time: a step's chunks take their homes (pinned
     * VMM) together, the higher steps the lower arena chunks */
    for (uint32_t sc = s->kv_cap; sc < cap;) {
        sc = sc + KV_PAGE_STEP < cap ? sc + KV_PAGE_STEP : cap;
        for (int r = 0; r < 4; r++)
            for (uint32_t qi = 0; qi < s->n_qsa; qi++) {
                uint64_t off, len;
                void *va = kv_region(s, r, qi, sc, &off, &len);
                if (va && !q4_hip_vreg_map(va, off, len)) {
                    fprintf(stderr, "q4: KV paging: map failed at %u positions\n",
                            sc);
                    return false;
                }
            }
    }
    static int logged;
    if (logged++ < 64)
        fprintf(stderr, "q4: KV mapped %u -> %u positions (+%u chunks, pool %u)\n",
                s->kv_cap, cap, need, q4_hip_vpool_n());
    s->kv_cap = cap;
    if (getenv("Q4_VMM_AUDIT") && getenv("Q4_VMM_AUDIT")[0] == '1')
        q4_hip_vmm_audit("kv-map");
    return true;
}

bool q4_sess_kv_reserve(q4_sess *s, uint32_t n_pos) {
    return s && sess_kv_reserve(s, n_pos);
}

/* Give back the paged caches' chunks above n_pos (rounded up to a step);
 * the device must be idle.  Returns the chunks returned to the pool. */
uint32_t q4_sess_kv_trim(q4_sess *s, uint32_t n_pos) {
    if (!s || !s->kv_paged) return 0;
    if (n_pos < s->n_kv) n_pos = s->n_kv;
    uint32_t cap = (n_pos + KV_PAGE_STEP - 1) / KV_PAGE_STEP * KV_PAGE_STEP;
    if (cap < KV_PAGE_STEP) cap = KV_PAGE_STEP;
    if (cap >= s->kv_cap) return 0;
    q4_hip_sync();
    uint32_t put = 0;
    uint64_t offs[Q4_MAX_LAYER], lens[Q4_MAX_LAYER];
    for (int r = 0; r < 4; r++) {
        void *va = NULL;
        for (uint32_t qi = 0; qi < s->n_qsa && qi < Q4_MAX_LAYER; qi++)
            va = kv_region(s, r, qi, cap, &offs[qi], &lens[qi]);
        if (va) put += q4_hip_vreg_trim(va, offs, lens, (int)s->n_qsa);
    }
    fprintf(stderr, "q4: KV trimmed %u -> %u positions (%u chunks back, pool %u)\n",
            s->kv_cap, cap, put, q4_hip_vpool_n());
    s->kv_cap = cap;
    return put;
}

uint32_t q4_sess_kv_cap(const q4_sess *s) { return s ? s->kv_cap : 0; }

/* Bytes of the paged caches at full context (the engine's pool size). */
uint64_t q4_sess_kv_paged_bytes(const q4_sess *s) {
    if (!s || !s->kv_paged) return 0;
    uint64_t t = 0;
    for (int r = 0; r < 4; r++)
        for (uint32_t qi = 0; qi < s->n_qsa; qi++) {
            uint64_t off, len;
            if (kv_region(s, r, qi, s->max_kv, &off, &len)) t += len;
        }
    return t;
}

q4_sess *q4_sess_open(const q4_gguf *g, q4_store *st, q4_expert_cache *c,
                      q4_ple *ple, uint32_t max_kv) {
    if (!g || !st || !c) return NULL;
    q4_sess *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->g = g;
    s->st = st;
    s->ex = c;
    s->ple = ple;
    s->max_kv = max_kv ? max_kv : 256;
    s->kv_kind = kv_kind_of();
    s->kv_bpt = 0;
    uint32_t hc_dim = (g->hc_mult ? g->hc_mult : 4) * g->n_embd;
    s->res = calloc(hc_dim, sizeof(float));
    s->ws = calloc(1u << 20, sizeof(float)); /* 4 MiB scratch */
    uint32_t ng = 0, nq = 0;
    for (uint32_t i = 0; i < g->n_layer; i++) {
        if (q4_layer_is_qsa(g, (int32_t)i)) {
            s->qsa_ix[i] = nq++;
            s->gdn_ix[i] = UINT32_MAX;
        } else {
            s->gdn_ix[i] = ng++;
            s->qsa_ix[i] = UINT32_MAX;
        }
    }
    s->n_gdn = ng;
    s->n_qsa = nq;
    uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    uint32_t d_v = n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
    uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    uint32_t conv_hist = ksz > 1 ? ksz - 1 : 1;
    s->gdn_S = calloc((size_t)ng * n_v_h * d_v * d_v, sizeof(float));
    s->gdn_conv = calloc((size_t)ng * conv_hist * conv_ch, sizeof(float));
    uint32_t kvw = (g->n_head_kv ? g->n_head_kv : 2) *
                   (g->n_embd_head_k ? g->n_embd_head_k : 256);
    size_t qsa_n = (size_t)nq * s->max_kv * kvw;
    uint32_t hist = 9;
    s->ple_conv = calloc((hist + 1) * hc_dim, sizeof(float));
    s->h_logits = g->n_vocab ? calloc(g->n_vocab, sizeof(float)) : NULL;
    if (s->h_logits)
        q4_hip_host_register(s->h_logits, (size_t)g->n_vocab * sizeof(float));
    {
        const char *ge = getenv("Q4_GRAPH");
        s->graph_off = (ge && ge[0] == '0');
    }
    s->devroute = -1;
    for (int i = 0; i < 8; i++) s->ple_hist[i] = (int32_t)g->ple_eos;
    if (!s->res || !s->ws || !s->gdn_S || !s->gdn_conv || !s->ple_conv ||
        (g->n_vocab && !s->h_logits)) {
        q4_sess_close(s);
        return NULL;
    }
    if (q4_hip_ok()) {
        size_t f = sizeof(float);
        s->d_res = q4_hip_malloc(hc_dim * f);
        s->d_ws = q4_hip_malloc((size_t)(1u << 20) * f);
        s->d_gdn_S = q4_hip_malloc((size_t)ng * n_v_h * d_v * d_v * f);
        s->d_gdn_conv = q4_hip_malloc((size_t)ng * conv_hist * conv_ch * f);
        s->d_logits = q4_hip_malloc((size_t)g->n_vocab * f);
        s->d_ple_conv = q4_hip_malloc((size_t)(hist + 1) * hc_dim * f);
        s->d_ple_conv2 = q4_hip_malloc((size_t)(hist + 1) * hc_dim * f);
        s->d_emb = q4_hip_malloc((size_t)g->n_embd * f);
        if (qsa_n) {
            s->kv_bpt = kv_token_bytes(s);
            uint32_t kvw = (g->n_head_kv ? g->n_head_kv : 2) *
                           (g->n_embd_head_k ? g->n_embd_head_k : 256);
            uint64_t kv_bytes = kvw ? (uint64_t)qsa_n / kvw * s->kv_bpt : 0;
            s->kv_paged = g_kv_paging;
            s->d_qsa_k = s->kv_paged ? q4_hip_vreg_alloc(kv_bytes)
                                     : q4_hip_malloc(kv_bytes);
            s->d_qsa_v = s->kv_paged ? q4_hip_vreg_alloc(kv_bytes)
                                     : q4_hip_malloc(kv_bytes);
            /* QSA indexer: raw fp16 keys + pooled fp32 block keys. A miss
             * leaves the dense path in place. */
            {
                uint32_t ratio = 4, topk = g->indexer_top_k ? g->indexer_top_k : 2048;
                uint32_t dim = g->indexer_head_size ? g->indexer_head_size : 128;
                for (uint32_t i = 0; i < g->n_layer && i < Q4_MAX_LAYER; i++)
                    if (g->compress_ratio[i]) { ratio = g->compress_ratio[i]; break; }
                uint32_t kblk = ratio ? topk / ratio : 0;
                uint32_t width = kblk && ratio ? topk + ratio - 1 : 0;
                uint32_t blocks = ratio ? s->max_kv / ratio : 0;
                uint32_t chunk = 512;
                if (kblk && kblk <= 512 && dim == 128 && blocks && nq) {
                    size_t raw_n = (size_t)nq * s->max_kv * dim * sizeof(uint16_t);
                    size_t pool_n = (size_t)nq * blocks * dim * sizeof(float);
                    size_t score_n = (size_t)chunk * blocks * sizeof(float);
                    size_t sel_n = (size_t)chunk * width * sizeof(int32_t);
                    s->d_idx_raw = s->kv_paged ? q4_hip_vreg_alloc(raw_n)
                                               : q4_hip_malloc(raw_n);
                    s->d_idx_pool = s->kv_paged ? q4_hip_vreg_alloc(pool_n)
                                                : q4_hip_malloc(pool_n);
                    s->d_idx_score = q4_hip_malloc(score_n);
                    s->d_idx_sel = q4_hip_malloc(sel_n);
                    s->d_idx_q = q4_hip_malloc((size_t)chunk * 4u * dim * sizeof(float));
                    s->d_idx_k = q4_hip_malloc((size_t)chunk * dim * sizeof(float));
                    if (s->d_idx_raw && s->d_idx_pool && s->d_idx_score &&
                        s->d_idx_sel && s->d_idx_q && s->d_idx_k &&
                        q4_hip_qsa_topk_alloc()) {
                        s->idx_dim = dim;
                        s->idx_ratio = ratio;
                        s->idx_kblk = kblk;
                        s->idx_width = width;
                        s->max_blocks = blocks;
                        fprintf(stderr,
                                "q4: QSA indexer  c%u  budget %u tok  cache %.2f GiB\n",
                                ratio, width,
                                (double)(raw_n + pool_n + score_n + sel_n) /
                                    (1024.0 * 1024.0 * 1024.0));
                    } else {
                        q4_hip_free(s->d_idx_raw);
                        q4_hip_free(s->d_idx_pool);
                        q4_hip_free(s->d_idx_score);
                        q4_hip_free(s->d_idx_sel);
                        q4_hip_free(s->d_idx_q);
                        q4_hip_free(s->d_idx_k);
                        s->d_idx_raw = NULL;
                        s->d_idx_pool = NULL;
                        s->d_idx_score = NULL;
                        s->d_idx_sel = NULL;
                        s->d_idx_q = NULL;
                        s->d_idx_k = NULL;
                        fprintf(stderr, "q4: QSA indexer alloc failed, dense attention\n");
                    }
                }
            }
            if (!s->d_qsa_k || !s->d_qsa_v) {
                q4_hip_free(s->d_qsa_k);
                q4_hip_free(s->d_qsa_v);
                s->d_qsa_k = s->d_qsa_v = NULL;
                fprintf(stderr,
                        "q4: QSA KV %.2f GiB did not fit on the GPU at ctx %u\n",
                        (double)qsa_n * 2 * sizeof(float) / (1024.0 * 1024.0 * 1024.0),
                        s->max_kv);
            }
        }
        if (!s->d_res || !s->d_ws || !s->d_gdn_S || !s->d_gdn_conv ||
            !s->d_logits || !s->d_ple_conv || !s->d_emb) {
            q4_sess_close(s);
            return NULL;
        }
        q4_hip_fill(s->d_res, 0, hc_dim);
        q4_hip_fill(s->d_gdn_S, 0, (uint64_t)ng * n_v_h * d_v * d_v);
        q4_hip_fill(s->d_gdn_conv, 0, (uint64_t)ng * conv_hist * conv_ch);
        q4_hip_fill(s->d_ple_conv, 0, (uint64_t)(hist + 1) * hc_dim);
    }
    if (!s->d_qsa_k || !s->d_qsa_v) {
        s->qsa_k = calloc(qsa_n, sizeof(float));
        s->qsa_v = calloc(qsa_n, sizeof(float));
        if (qsa_n && (!s->qsa_k || !s->qsa_v)) {
            q4_sess_close(s);
            return NULL;
        }
    }
    return s;
}

void q4_sess_close(q4_sess *s) {
    if (!s) return;
    ple_jobs_close(s);
    free(s->res);
    free(s->ws);
    free(s->gdn_S);
    free(s->gdn_conv);
    free(s->qsa_k);
    free(s->qsa_v);
    free(s->ple_conv);
    free(s->ple_kw);
    q4_hip_free(s->d_ple_kw);
    q4_hip_free(s->d_res);
    q4_hip_free(s->d_ws);
    q4_hip_free(s->d_gdn_S);
    q4_hip_free(s->d_gdn_conv);
    q4_hip_free(s->d_logits);
    q4_hip_free(s->d_logits_n);
    q4_hip_free(s->d_ple_conv);
    q4_hip_free(s->d_emb);
    q4_hip_free(s->d_qsa_k);
    q4_hip_free(s->d_qsa_v);
    q4_hip_free(s->d_idx_raw);
    q4_hip_free(s->d_idx_pool);
    q4_hip_free(s->d_idx_score);
    q4_hip_free(s->d_idx_sel);
    q4_hip_free(s->d_idx_q);
    q4_hip_free(s->d_idx_k);
    if (s->lend_mode) {
        q4_hip_free(s->m_pref);
        q4_hip_free(s->m_mixb);
        q4_hip_free(s->m_yb);
        q4_hip_free(s->m_injb);
        q4_hip_free(s->m_pws);
    } else {
        q4_hip_free(s->d_pref);
        q4_hip_free(s->d_mixb);
        q4_hip_free(s->d_yb);
        q4_hip_free(s->d_injb);
        q4_hip_free(s->d_pws);
    }
    q4_hip_free(s->d_ple_conv2);
    q4_hip_free(s->d_gdn_S_snap);
    q4_hip_free(s->d_gdn_conv_snap);
    q4_hip_free(s->d_gdn_stash);
    q4_hip_free(s->d_gdn_replay);
    if (s->d_ple_snap) {
        for (int t = 0; t < s->mtp_max; t++) q4_hip_free(s->d_ple_snap[t]);
        free(s->d_ple_snap);
    }
    free(s->ple_hist_snap);
    for (int i = 0; i < Q4_MAX_LAYER; i++) {
        q4_hip_graph_free(s->gA[i]);
        q4_hip_graph_free(s->gB[i]);
    }
    q4_hip_graph_free(s->g_head);
    if (s->h_logits) q4_hip_host_unregister(s->h_logits);
    free(s->h_logits);
    free(s);
}

static int nan_report(const char *tag, int il, const float *v, uint32_t n) {
    if (!getenv("Q4_DEBUG")) return 0;
    for (uint32_t i = 0; i < n; i++) {
        if (isnan(v[i]) || isinf(v[i])) {
            fprintf(stderr, "q4: %s layer %d i=%u v=%g\n", tag, il, i,
                    (double)v[i]);
            return 1;
        }
    }
    return 0;
}

static bool embed_tok(q4_sess *s, int32_t tok, float *emb) {
    const q4_tensor *t = q4_find_tensor(s->g, "token_embd.weight");
    if (!t || tok < 0 || (uint32_t)tok >= s->g->n_vocab) return false;
    uint64_t rb = q4_row_bytes(t->ggml_type, t->ne[0]);
    const uint8_t *w = q4_store_get(s->st, t);
    if (!w || rb == 0) return false;
    return q4_dequant_row(t->ggml_type, w + (uint64_t)tok * rb, t->ne[0], emb);
}

bool q4_sess_decode_ex(q4_sess *s, int32_t token, int32_t *out_id, float *out_logit,
                       bool want_logits) {
    if (!s) return false;
    if (!sess_kv_reserve(s, s->n_kv + 1)) return false;
    if (s->lent && !q4_sess_lend_end(s)) return false;
    if (s->d_res && q4_hip_ok())
        return decode_hip(s, token, out_id, out_logit, want_logits);
    const q4_gguf *g = s->g;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t hc_dim = hc * n_embd;
    float *emb = s->ws + (1u << 18);
    if (!embed_tok(s, token, emb)) return false;
    for (uint32_t c = 0; c < hc; c++)
        memcpy(s->res + c * n_embd, emb, n_embd * sizeof(float));

    float *mixed = emb + n_embd;
    float *block = mixed + n_embd;
    float inject[8];

    for (uint32_t il = 0; il < g->n_layer; il++) {
        if (g->ple_layer == il && s->ple) {
            if (!ple_one(s, (int)il, s->res, token)) return false;
            if (nan_report("ple", (int)il, s->res, hc_dim)) return false;
        }
        if (!hc_mix(s, s->res, mixed, inject, (int)il, true)) return false;
        if (nan_report("hc_attn_mix", (int)il, mixed, n_embd)) return false;
        if (q4_layer_is_qsa(g, (int)il)) {
            if (!qsa_one(s, (int)il, mixed, block)) return false;
            if (nan_report("qsa", (int)il, block, n_embd)) return false;
        } else {
            if (!gdn_one(s, (int)il, mixed, block)) return false;
            if (nan_report("gdn", (int)il, block, n_embd)) return false;
        }
        hc_combine(s->res, block, inject, n_embd, hc);
        if (nan_report("hc_attn_comb", (int)il, s->res, hc_dim)) return false;
        if (!hc_mix(s, s->res, mixed, inject, (int)il, false)) return false;
        if (nan_report("hc_ffn_mix", (int)il, mixed, n_embd)) return false;
        if (!q4_moe_layer_store(g, s->st, s->ex, (int)il, mixed, block))
            return false;
        if (nan_report("moe", (int)il, block, n_embd)) return false;
        hc_combine(s->res, block, inject, n_embd, hc);
    }
    if (g->ple_ngram >= 2) {
        memmove(s->ple_hist, s->ple_hist + 1,
                (g->ple_ngram - 2) * sizeof(int32_t));
        s->ple_hist[g->ple_ngram - 2] = token;
    }
    s->n_kv++;
    s->pos++;

    if (!want_logits) {
        if (out_id) *out_id = 0;
        if (out_logit) *out_logit = 0;
        return true;
    }

    /* final mixer = output norm */
    const q4_tensor *hn = q4_find_tensor(g, "output_hc_norm.weight");
    const q4_tensor *hd = q4_find_tensor(g, "output_hc_down.weight");
    const q4_tensor *hu = q4_find_tensor(g, "output_hc_up.weight");
    float *xn = s->ws;
    grouped_rms(s->res, f32w(s->st, hn), xn, n_embd, hc, g->rms_eps);
    float *lo = xn + hc_dim;
    if (!mm(s->st, hd, xn, lo)) return false;
    for (uint32_t i = 0; i < g->hc_rank; i++) lo[i] *= 1.f / (float)hc;
    q4_silu(lo, g->hc_rank);
    float *gate = lo + g->hc_rank;
    if (!mm(s->st, hu, lo, gate)) return false;
    q4_sigmoid(gate, hc_dim);
    memset(mixed, 0, n_embd * sizeof(float));
    for (uint32_t c = 0; c < hc; c++)
        for (uint32_t i = 0; i < n_embd; i++)
            mixed[i] += xn[c * n_embd + i] * gate[c * n_embd + i];
    for (uint32_t i = 0; i < n_embd; i++) mixed[i] /= (float)hc;

    const q4_tensor *out = q4_find_tensor(g, "output.weight");
    if (!out) return false;
    float *logits = s->h_logits ? s->h_logits : s->ws + ((1u << 20) - g->n_vocab);
    if (!mm(s->st, out, mixed, logits)) return false;
    int32_t best = 0;
    float bv = logits[0];
    for (uint32_t i = 1; i < g->n_vocab; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = (int32_t)i;
        }
    }
    if (out_id) *out_id = best;
    if (out_logit) *out_logit = bv;
    return true;
}

bool q4_sess_decode(q4_sess *s, int32_t token, int32_t *out_id, float *out_logit) {
    return q4_sess_decode_ex(s, token, out_id, out_logit, true);
}

void q4_sess_set_embd_rows(q4_sess *s, const float *const *rows, uint32_t n) {
    if (!s) return;
    s->emb_rows = rows;
    s->emb_rows_n = rows ? n : 0;
}

void q4_sess_reset(q4_sess *s) {
    if (!s || !s->g) return;
    const q4_gguf *g = s->g;
    uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    uint32_t hc_dim = hc * g->n_embd;
    uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    uint32_t d_v = n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
    uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    uint32_t conv_hist = ksz > 1 ? ksz - 1 : 1;
    uint32_t kvw = (g->n_head_kv ? g->n_head_kv : 2) *
                   (g->n_embd_head_k ? g->n_embd_head_k : 256);
    s->n_kv = 0;
    s->pos = 0;
    s->mtp_n = 0;
    s->mtp_stash = 0;
    /* A mailbox overrun or wedged cpx_wait aborts the request it hit; the
     * next request starts clean — the dispatcher and pool outlive it. */
    q4_moe_dev_reset();
    for (int i = 0; i < 8; i++) s->ple_hist[i] = (int32_t)g->ple_eos;
    memset(s->res, 0, hc_dim * sizeof(float));
    memset(s->gdn_S, 0, (size_t)s->n_gdn * n_v_h * d_v * d_v * sizeof(float));
    memset(s->gdn_conv, 0, (size_t)s->n_gdn * conv_hist * conv_ch * sizeof(float));
    memset(s->ple_conv, 0, 10ull * hc_dim * sizeof(float));
    if (s->qsa_k)
        memset(s->qsa_k, 0, (size_t)s->n_qsa * s->max_kv * kvw * sizeof(float));
    if (s->qsa_v)
        memset(s->qsa_v, 0, (size_t)s->n_qsa * s->max_kv * kvw * sizeof(float));
    if (s->d_res && q4_hip_ok()) {
        q4_hip_fill(s->d_res, 0, hc_dim);
        q4_hip_fill(s->d_gdn_S, 0, (uint64_t)s->n_gdn * n_v_h * d_v * d_v);
        q4_hip_fill(s->d_gdn_conv, 0, (uint64_t)s->n_gdn * conv_hist * conv_ch);
        q4_hip_fill(s->d_ple_conv, 0, 10ull * hc_dim);
    }
}

uint32_t q4_sess_n_kv(const q4_sess *s) {
    return s ? s->n_kv : 0;
}

uint32_t q4_sess_max_kv(const q4_sess *s) {
    return s ? s->max_kv : 0;
}

const float *q4_sess_logits(const q4_sess *s) {
    return s ? s->h_logits : NULL;
}

/* ---- session-state snapshot (prefix cache for chat turns) ----
 * Saves every recurrent input the suffix prefill needs: QSA KV, GDN
 * state + conv history, PLE conv/hist, last residual, positions and the
 * token ids that produced them. Restore requires an EXACT prefix match
 * of the whole saved sequence (GDN state exists only at the sequence
 * end, so a longer snapshot cannot be rewound). A second file written
 * at a prompt/chunk boundary is just another snapshot of this form. */

#define Q4_SESS_MAGIC 0x53345134u /* "Q4SS" LE */
/* 5 appends indexer raw keys; pad is the KV width tag.  6: the KV and the
 * pooled indexer keys are rotated with NEOX pairs (a 5 file holds the old
 * interleaved rotation, which Q4_ROPE_NEOX=0 still writes and reads). */
static uint32_t sess_ver(void) {
    const char *rn = getenv("Q4_ROPE_NEOX");
    return rn && rn[0] == '0' ? 5u : 6u;
}
#define Q4_SESS_VER sess_ver()

typedef struct {
    uint32_t magic, version;
    uint64_t bytes_total; /* model guard */
    uint32_t n_layer, n_expert;
    uint32_t max_kv, n_kv, pos;
    uint32_t n_qsa, kvw;
    uint32_t n_gdn, n_v_h, d_v, conv_hist, conv_ch, hc_dim;
    uint32_t n_ids;
    uint32_t prompt_n; /* tokens before generation */
    uint32_t nmsg;     /* messages used to build the prompt */
    uint32_t pad;
    uint64_t user_hash; /* first user message; OpenCode mutates system */
} q4_sess_hdr;

static bool wr_(FILE *f, const void *p, size_t n) { return fwrite(p, 1, n, f) == n; }
static bool rd_(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static bool save_buf(FILE *f, const float *dev, uint64_t nfloat, float *stage,
                     uint64_t stage_n) {
    for (uint64_t o = 0; o < nfloat; o += stage_n) {
        uint64_t n = nfloat - o < stage_n ? nfloat - o : stage_n;
        if (!q4_hip_d2h(stage, dev + o, n * sizeof(float))) return false;
        if (!wr_(f, stage, (size_t)n * sizeof(float))) return false;
    }
    return true;
}

/* Q4_RESTORE_VERIFY=1: read every restored buffer back and count the
 * floats that differ from the file (copies into remapped KV VAs). */
static uint64_t g_restore_bad, g_restore_n;

static bool load_buf(FILE *f, float *dev, uint64_t nfloat, float *stage,
                     uint64_t stage_n) {
    static int verify = -1;
    if (verify < 0) {
        const char *e = getenv("Q4_RESTORE_VERIFY");
        verify = e && e[0] == '1';
    }
    float *chk = verify ? malloc((size_t)stage_n * sizeof(float)) : NULL;
    for (uint64_t o = 0; o < nfloat; o += stage_n) {
        uint64_t n = nfloat - o < stage_n ? nfloat - o : stage_n;
        if (!rd_(f, stage, (size_t)n * sizeof(float))) { free(chk); return false; }
        if (!q4_hip_h2d(dev + o, stage, n * sizeof(float))) { free(chk); return false; }
        if (chk && q4_hip_d2h(chk, dev + o, n * sizeof(float))) {
            g_restore_n += n;
            for (uint64_t i = 0; i < n; i++)
                g_restore_bad += memcmp(&chk[i], &stage[i], sizeof(float)) != 0;
        }
    }
    free(chk);
    return true;
}

static void sess_dims(const q4_sess *s, uint32_t *kvw, uint32_t *conv_ch,
                      uint32_t *conv_hist, uint32_t *n_v_h, uint32_t *d_v,
                      uint32_t *hc_dim) {
    const q4_gguf *g = s->g;
    uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    uint32_t nvh = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    uint32_t dk = g->ssm_d_state ? g->ssm_d_state : 128;
    uint32_t dv = nvh ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / nvh : 128;
    uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    *n_v_h = nvh;
    *d_v = dv;
    *kvw = (g->n_head_kv ? g->n_head_kv : 2) *
           (g->n_embd_head_k ? g->n_embd_head_k : 256);
    *conv_ch = dk * n_k_h * 2 + dv * nvh;
    *conv_hist = ksz > 1 ? ksz - 1 : 1;
    *hc_dim = (g->hc_mult ? g->hc_mult : 4) * g->n_embd;
}

bool q4_sess_save(const q4_sess *s, const char *path, const int32_t *ids,
                  int n_ids, int prompt_n, int nmsg, uint64_t user_hash) {
    if (!s || !path || !ids || n_ids <= 0 || !s->d_res) return false;
    if (prompt_n < 0) prompt_n = 0;
    if (prompt_n > n_ids) prompt_n = n_ids;
    if (nmsg < 0) nmsg = 0;
    const q4_gguf *g = s->g;
    q4_sess_hdr h;
    memset(&h, 0, sizeof(h));
    h.magic = Q4_SESS_MAGIC;
    h.version = Q4_SESS_VER;
    h.bytes_total = g->bytes_total;
    h.n_layer = g->n_layer;
    h.n_expert = g->n_expert;
    h.max_kv = s->max_kv;
    h.n_kv = s->n_kv;
    h.pos = (uint32_t)s->pos;
    h.n_qsa = s->n_qsa;
    h.n_gdn = s->n_gdn;
    h.n_ids = (uint32_t)n_ids;
    h.prompt_n = (uint32_t)prompt_n;
    h.nmsg = (uint32_t)nmsg;
    h.pad = kv_tag(s);
    h.user_hash = user_hash;
    sess_dims(s, &h.kvw, &h.conv_ch, &h.conv_hist, &h.n_v_h, &h.d_v, &h.hc_dim);
    char tmp[4096 + 16];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    bool ok = wr_(f, &h, sizeof(h)) &&
              wr_(f, ids, (size_t)n_ids * sizeof(int32_t)) &&
              wr_(f, s->ple_hist, sizeof(s->ple_hist));
    float *stage = ok ? malloc((4u << 20) * sizeof(float)) : NULL;
    if (!stage) ok = false;
    const uint64_t stage_n = 4u << 20;
    if (ok) { /* residual + PLE conv */
        ok = save_buf(f, s->d_res, h.hc_dim, stage, stage_n) &&
             save_buf(f, s->d_ple_conv, 10ull * h.hc_dim, stage, stage_n);
    }
    if (ok) { /* QSA KV, per layer */
        for (uint32_t qi = 0; qi < s->n_qsa && ok; qi++) {
            const float *kb = (const float *)kv_at(s, 0, qi, 0);
            const float *vb = (const float *)kv_at(s, 1, qi, 0);
            uint64_t nfloat = ((uint64_t)h.n_kv * s->kv_bpt) / sizeof(float);
            ok = save_buf(f, kb, nfloat, stage, stage_n) &&
                 save_buf(f, vb, nfloat, stage, stage_n);
        }
    }
    if (ok) { /* GDN state + conv history */
        ok = save_buf(f, s->d_gdn_S,
                      (uint64_t)h.n_gdn * h.n_v_h * h.d_v * h.d_v, stage,
                      stage_n) &&
             save_buf(f, s->d_gdn_conv, (uint64_t)h.n_gdn * h.conv_hist * h.conv_ch,
                      stage, stage_n);
    }
    if (ok) {
        uint32_t meta[4] = {s->idx_dim, s->n_qsa, s->n_kv, s->d_idx_raw ? 1u : 0u};
        ok = wr_(f, meta, sizeof(meta));
        if (ok && s->d_idx_raw && s->idx_dim) {
            uint64_t nfloat =
                ((uint64_t)s->n_kv * s->idx_dim * sizeof(uint16_t)) / sizeof(float);
            for (uint32_t qi = 0; qi < s->n_qsa && ok; qi++)
                ok = save_buf(f, (float *)idx_raw_at(s, qi), nfloat, stage, stage_n);
        }
    }
    free(stage);
    if (fclose(f) != 0) ok = false;
    if (!ok) {
        remove(tmp);
        return false;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        return false;
    }
    return true;
}

/* Returns the saved prefix length on exact match (state restored), else 0. */
int q4_sess_restore(q4_sess *s, const char *path, const int32_t *ids,
                    int n_ids) {
    if (!s || !path || !ids || n_ids <= 0 || !s->d_res) return 0;
    const q4_gguf *g = s->g;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    q4_sess_hdr h;
    uint32_t kvw, conv_ch, conv_hist, n_v_h, d_v, hc_dim;
    sess_dims(s, &kvw, &conv_ch, &conv_hist, &n_v_h, &d_v, &hc_dim);
    bool ok = rd_(f, &h, sizeof(h)) && h.magic == Q4_SESS_MAGIC &&
              h.version == Q4_SESS_VER && h.bytes_total == g->bytes_total &&
              h.n_layer == g->n_layer && h.n_expert == g->n_expert &&
              h.max_kv == s->max_kv && h.n_qsa == s->n_qsa &&
              h.n_gdn == s->n_gdn && h.kvw == kvw && h.conv_ch == conv_ch &&
              h.conv_hist == conv_hist && h.n_v_h == n_v_h && h.d_v == d_v &&
              h.hc_dim == hc_dim && h.pad == kv_tag(s) &&
              (h.n_kv == h.n_ids || h.n_kv + 1 == h.n_ids) &&
              h.n_ids > 0 && (int)h.n_ids <= n_ids && h.n_kv <= s->max_kv;
    int32_t *sids = ok ? malloc((size_t)h.n_ids * sizeof(int32_t)) : NULL;
    if (!sids) ok = false;
    if (ok) {
        ok = rd_(f, sids, (size_t)h.n_ids * sizeof(int32_t));
        if (ok) {
            uint32_t bad = 0;
            while (bad < h.n_ids && sids[bad] == ids[bad]) bad++;
            if (bad < h.n_ids) {
                fprintf(stderr,
                        "q4: session cache prefix differs at token %u/%u "
                        "(saved %d new %d); recomputing\n",
                        bad, h.n_ids, sids[bad],
                        bad < (uint32_t)n_ids ? ids[bad] : -1);
                ok = false;
            }
        }
    }
    int32_t ph[8];
    if (ok) ok = rd_(f, ph, sizeof(ph));
    if (ok) ok = sess_kv_reserve(s, h.n_kv);
    float *stage = ok ? malloc((4u << 20) * sizeof(float)) : NULL;
    if (!stage) ok = false;
    const uint64_t stage_n = 4u << 20;
    if (ok)
        ok = load_buf(f, s->d_res, hc_dim, stage, stage_n) &&
             load_buf(f, s->d_ple_conv, 10ull * hc_dim, stage, stage_n);
    if (ok) {
        for (uint32_t qi = 0; qi < s->n_qsa && ok; qi++) {
            float *kb = (float *)kv_at(s, 0, qi, 0);
            float *vb = (float *)kv_at(s, 1, qi, 0);
            uint64_t nfloat = ((uint64_t)h.n_kv * s->kv_bpt) / sizeof(float);
            ok = load_buf(f, kb, nfloat, stage, stage_n) &&
                 load_buf(f, vb, nfloat, stage, stage_n);
        }
    }
    if (ok)
        ok = load_buf(f, s->d_gdn_S, (uint64_t)h.n_gdn * n_v_h * d_v * d_v,
                      stage, stage_n) &&
             load_buf(f, s->d_gdn_conv, (uint64_t)h.n_gdn * conv_hist * conv_ch,
                      stage, stage_n);
    uint32_t idx_meta[4] = {0, 0, 0, 0};
    if (ok) ok = rd_(f, idx_meta, sizeof(idx_meta));
    if (ok && idx_meta[3]) {
        if (!s->d_idx_raw || idx_meta[0] != s->idx_dim || idx_meta[1] != s->n_qsa ||
            idx_meta[2] != h.n_kv) {
            ok = false;
        } else {
            uint64_t nfloat =
                ((uint64_t)h.n_kv * s->idx_dim * sizeof(uint16_t)) / sizeof(float);
            for (uint32_t qi = 0; qi < s->n_qsa && ok; qi++)
                ok = load_buf(f, (float *)idx_raw_at(s, qi), nfloat, stage, stage_n);
        }
    }
    free(stage);
    free(sids);
    fclose(f);
    if (!ok) {
        q4_sess_reset(s);
        return 0;
    }
    memcpy(s->ple_hist, ph, sizeof(ph));
    s->n_kv = h.n_kv;
    s->pos = (int32_t)h.pos;
    if (idx_meta[3] && !idx_rebuild(s)) {
        q4_sess_reset(s);
        return 0;
    }
    /* A turn that stopped on its token limit saved its last sampled id
     * without feeding it (n_kv = n_ids - 1): resume at n_kv, the prompt
     * pass feeds that id again.  Such files used to be rejected, so the
     * next turn recomputed the whole prompt. */
    if (g_restore_n) {
        fprintf(stderr, "q4 restore verify: %llu of %llu floats differ (n_kv %u)\n",
                (unsigned long long)g_restore_bad, (unsigned long long)g_restore_n,
                (unsigned)h.n_kv);
        g_restore_bad = g_restore_n = 0;
    }
    return (int)h.n_kv;
}

/* ---- MTP verify support ---- */

const float *q4_sess_pref_row(const q4_sess *s, int i) {
    if (!s || !s->d_pref || i < 0 || (uint32_t)i >= s->pref_chunk) return NULL;
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    return s->d_pref + (size_t)i * hc * g->n_embd;
}

/* Device residual (hc_dim) of the last processed token. The MTP head uses
 * it as the hidden-state input when no prefill ran this turn. */
const float *q4_sess_res_dev(const q4_sess *s) {
    return s && s->d_res ? s->d_res : NULL;
}

bool q4_sess_mtp_prep(q4_sess *s, int n_max) {
    if (!s || !s->d_res || !q4_hip_ok() || n_max <= 0 || n_max > 8) return false;
    if (s->d_gdn_S_snap) return true; /* already prepared */
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t hc_dim = hc * g->n_embd;
    const uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    const uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    const uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    const uint32_t d_v = n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
    const uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    const uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    const uint32_t conv_hist = ksz > 1 ? ksz - 1 : 1;
    s->mtp_max = n_max;
    s->gdn_stash_row = conv_ch + n_v_h * d_v + 2 * n_v_h;
    s->ple_hist_snap = malloc((size_t)n_max * 8 * sizeof(int32_t));
    s->d_ple_snap = calloc((size_t)n_max, sizeof(float *));
    s->d_gdn_S_snap = q4_hip_malloc((size_t)s->n_gdn * n_v_h * d_v * d_v *
                                    sizeof(float));
    s->d_gdn_conv_snap = q4_hip_malloc((size_t)s->n_gdn * conv_hist * conv_ch *
                                       sizeof(float));
    s->d_gdn_stash = q4_hip_malloc((size_t)s->n_gdn * n_max * s->gdn_stash_row *
                                   sizeof(float));
    s->d_gdn_replay = q4_hip_malloc((size_t)n_max * n_v_h * d_v * sizeof(float));
    if (s->d_ple_snap)
        for (int t = 0; t < n_max; t++)
            s->d_ple_snap[t] =
                q4_hip_malloc((size_t)10 * hc_dim * sizeof(float));
    bool ok = s->d_gdn_stash && s->ple_hist_snap && s->d_ple_snap &&
              s->d_gdn_S_snap && s->d_gdn_conv_snap && s->d_gdn_replay;
    if (s->d_ple_snap)
        for (int t = 0; t < n_max && ok; t++)
            if (!s->d_ple_snap[t]) ok = false;
    if (!ok) {
        fprintf(stderr, "q4: mtp prep failed (snapshot buffers)\n");
        return false;
    }
    return true;
}

bool q4_sess_prefill_tokens(q4_sess *s, const int32_t *toks, int n) {
    if (!s || !toks || n <= 0 || n > 8) return false;
    if (!sess_kv_reserve(s, s->n_kv + (uint32_t)n)) return false;
    if (!s->d_res || !q4_hip_ok()) return false;
    if (s->lent && !q4_sess_lend_end(s)) return false;
    if (q4_expert_lend_on(s->ex) && !s->d_pref && !sess_mini(s)) return false;
    if (!s->d_pref) {
        /* Verify batches never exceed 8 rows (MTP_MAX_DRAFT+1); a cache-hit
         * session may reach here with VRAM already committed, so keep the
         * mini set small (~1 MiB). A later real prefill grows it back. */
        const q4_gguf *g = s->g;
        const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
        const uint32_t n_embd = g->n_embd;
        s->pref_chunk = 8;
        s->pws_floats = q4_pws_floats(g, s->pref_chunk);
        s->d_pref = q4_hip_malloc((size_t)s->pref_chunk * hc * n_embd *
                                  sizeof(float));
        s->d_mixb = q4_hip_malloc((size_t)s->pref_chunk * n_embd * sizeof(float));
        s->d_yb = q4_hip_malloc((size_t)s->pref_chunk * n_embd * sizeof(float));
        s->d_injb = q4_hip_malloc((size_t)s->pref_chunk * hc * sizeof(float));
        s->d_pws = q4_hip_malloc((size_t)s->pws_floats * sizeof(float));
        if (!s->d_pref || !s->d_mixb || !s->d_yb || !s->d_injb || !s->d_pws)
            return false;
    }
    if (n > (int)s->pref_chunk) return false;
    s->mtp_stash = 1;
    s->mtp_n = n;
    static int vprof = -1;
    if (vprof < 0) vprof = getenv("Q4_VPROF") && getenv("Q4_VPROF")[0] == '1';
    double vt0 = vprof ? q4_now() : 0;
    q4_mmvq_verify(1);
    bool ok = prefill_hip(s, toks, n, NULL);
    q4_mmvq_verify(0);
    if (vprof) {
        static double acc_enq, acc_all, acc_tick;
        static int calls;
        acc_enq += s->vprof_enq - vt0;
        acc_all += q4_now() - vt0;
        acc_tick += s->vprof_tick;
        if (++calls % 20 == 0) {
            fprintf(stderr, "q4 vprof: verify tick %.2f  host enqueue %.2f ms  total %.2f ms (n=%d)\n",
                    acc_tick * 1e3 / 20, acc_enq * 1e3 / 20, acc_all * 1e3 / 20, n);
            acc_enq = acc_all = acc_tick = 0;
        }
    }
    s->mtp_stash = 0;
    q4_expert_pf_join(s->ex);
    return ok;
}

bool q4_sess_logits_at(q4_sess *s, int row) {
    if (!s || !s->d_res || !q4_hip_ok()) return false;
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t hc_dim = hc * g->n_embd;
    const float *src = q4_sess_pref_row(s, row);
    if (!src) return false;
    if (!q4_hip_copy(s->d_res, src, hc_dim)) return false;
    return head_hip(s, NULL, NULL);
}

/* MTP verify tail: the lm head over verify rows [0, n) in one batched
 * pass, then a device top-k (k = 1..128) per row. One output GEMV reads
 * the 497 MiB head once for all rows (q4_sess_logits_at runs it, a
 * 0.95 MiB D2H and a host scan per row). Syncs; read the rows with
 * q4_hip_topk_rows_read. Does not touch d_res or the host logits. */
bool q4_sess_verify_topk(q4_sess *s, int n, uint32_t k) {
    if (!s || !s->d_pref || n <= 0 || n > 8 || (uint32_t)n > s->pref_chunk ||
        !q4_hip_ok())
        return false;
    const q4_gguf *g = s->g;
    const uint32_t n_embd = g->n_embd;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t hc_dim = hc * n_embd;
    const uint32_t nt = (uint32_t)n;
    const q4_tensor *hn = q4_find_tensor(g, "output_hc_norm.weight");
    const q4_tensor *hd = q4_find_tensor(g, "output_hc_down.weight");
    const q4_tensor *hu = q4_find_tensor(g, "output_hc_up.weight");
    const q4_tensor *out = q4_find_tensor(g, "output.weight");
    if (!hn || !hd || !hu || !out) return false;
    float *xn = s->d_ws;                              /* n x hc_dim */
    float *lo = xn + (size_t)nt * hc_dim;             /* n x hc_rank */
    float *gate = lo + (size_t)nt * g->hc_rank;       /* n x hc_dim */
    float *mixed = gate + (size_t)nt * hc_dim;        /* n x n_embd */
    if ((size_t)(mixed - s->d_ws) + (size_t)nt * n_embd > (1u << 20))
        return false;
    if (!s->d_logits_n) {
        s->d_logits_n = q4_hip_malloc((size_t)8 * g->n_vocab * sizeof(float));
        if (!s->d_logits_n) return false;
    }
    if (!q4_hip_grouped_rms_n(s->d_pref, f32d(s->st, hn), xn, n_embd, hc,
                              g->rms_eps, nt) ||
        !mm_d_n(s->st, hd, xn, lo, nt) ||
        !q4_hip_scale_silu(lo, 1.f / (float)hc, (uint64_t)nt * g->hc_rank) ||
        !mm_d_n(s->st, hu, lo, gate, nt) ||
        !q4_hip_mulsig_mean_n(xn, gate, mixed, n_embd, hc, nt) ||
        !mm_d_n(s->st, out, mixed, s->d_logits_n, nt) ||
        !q4_hip_topk_rows_launch(s->d_logits_n, g->n_vocab, nt, k))
        return false;
    return q4_hip_stream_sync();
}

bool q4_sess_mtp_commit(q4_sess *s, int n_acc) {
    if (!s || !s->d_res || !q4_hip_ok()) return false;
    int n = s->mtp_n;
    if (n <= 0) return false;
    if (n_acc >= n) {
        s->mtp_n = 0;
        return true; /* full accept: state already consistent */
    }
    if (n_acc <= 0 || !s->d_gdn_S_snap || !s->d_gdn_stash || n_acc > s->mtp_max)
        return false;
    const q4_gguf *g = s->g;
    const uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    const uint32_t hc_dim = hc * g->n_embd;
    const uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    const uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    const uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    const uint32_t d_v = n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
    const uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    const uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    const uint32_t conv_hist = ksz > 1 ? ksz - 1 : 1;
    /* Roll the sequential state back to the pre-batch snapshot ... */
    if (!q4_hip_copy(s->d_gdn_S, s->d_gdn_S_snap,
                     (uint64_t)s->n_gdn * n_v_h * d_v * d_v))
        return false;
    if (!q4_hip_copy(s->d_gdn_conv, s->d_gdn_conv_snap,
                     (uint64_t)s->n_gdn * conv_hist * conv_ch))
        return false;
    if (s->d_ple_snap && s->d_ple_snap[n_acc - 1]) {
        if (!q4_hip_copy(s->d_ple_conv, s->d_ple_snap[n_acc - 1],
                         10ull * hc_dim))
            return false;
        memcpy(s->ple_hist, s->ple_hist_snap + (size_t)(n_acc - 1) * 8,
               8 * sizeof(int32_t));
    }
    /* ... then replay the accepted prefix through the GDN recurrence,
     * reading the stashed inputs straight from device memory. Layer order
     * is irrelevant (per-layer state); tokens are sequential. */
    for (uint32_t il = 0; il < g->n_layer; il++) {
        uint32_t gi = s->gdn_ix[il];
        if (gi == UINT32_MAX) continue;
        float *sb = s->d_gdn_stash +
                    (size_t)gi * s->mtp_max * s->gdn_stash_row;
        float *zb = sb + (size_t)s->mtp_max * conv_ch;
        float *bb = zb + (size_t)s->mtp_max * n_v_h * d_v;
        float *ab = bb + (size_t)s->mtp_max * n_v_h;
        /* Q4_COMMIT_SCAN (default on): the accepted prefix in one gdn_scan
         * per layer — the verify's own kernel on its own stashed inputs —
         * instead of n_acc x (conv + step + ...) launches per layer. */
        static int cscan = -1;
        if (cscan < 0)
            cscan = !(getenv("Q4_COMMIT_SCAN") && getenv("Q4_COMMIT_SCAN")[0] == '0');
        if (cscan && d_k == d_v) {
            const float eps = g->rms_eps ? g->rms_eps : 1e-6f;
            if (q4_hip_gdn_scan(
                    s->d_gdn_conv + (size_t)gi * conv_hist * conv_ch, sb,
                    f32d(s->st, T(g, "blk.%d.ssm_conv1d.weight", (int)il)),
                    conv_ch, ksz, s->d_gdn_S + (size_t)gi * n_v_h * d_v * d_v,
                    bb, ab, f32d(s->st, T(g, "blk.%d.ssm_a", (int)il)),
                    f32d(s->st, T(g, "blk.%d.ssm_dt.bias", (int)il)),
                    f32d(s->st, T(g, "blk.%d.ssm_norm.weight", (int)il)), zb,
                    s->d_gdn_replay, (uint32_t)n_acc, n_v_h, n_k_h, d_v, eps,
                    1.f / sqrtf((float)d_k))) {
                /* Q4_COMMIT_CHECK=1: redo this layer per token from the
                 * snapshot and log max |dS| (diagnostic). */
                static int cchk = -1;
                if (cchk < 0)
                    cchk = getenv("Q4_COMMIT_CHECK") && getenv("Q4_COMMIT_CHECK")[0] == '1';
                if (cchk) {
                    const size_t ns = (size_t)n_v_h * d_v * d_v;
                    const size_t nc = (size_t)conv_hist * conv_ch;
                    float *a = malloc(ns * 4), *b = malloc(ns * 4);
                    float *S0 = s->d_gdn_S + (size_t)gi * ns;
                    float *C0 = s->d_gdn_conv + (size_t)gi * nc;
                    if (a && b && q4_hip_d2h(a, S0, ns * 4) &&
                        q4_hip_copy(S0, s->d_gdn_S_snap + (size_t)gi * ns, ns) &&
                        q4_hip_copy(C0, s->d_gdn_conv_snap + (size_t)gi * nc, nc)) {
                        for (int t = 0; t < n_acc; t++)
                            (void)gdn_apply_d(s, (int)il, sb + (size_t)t * conv_ch,
                                              zb + (size_t)t * n_v_h * d_v,
                                              bb + (size_t)t * n_v_h,
                                              ab + (size_t)t * n_v_h,
                                              s->d_gdn_replay);
                        if (q4_hip_d2h(b, S0, ns * 4)) {
                            double md = 0, mx = 0;
                            for (size_t i = 0; i < ns; i++) {
                                double d = fabs((double)a[i] - b[i]);
                                if (d > md) md = d;
                                if (fabs(b[i]) > mx) mx = fabs(b[i]);
                            }
                            fprintf(stderr, "q4 commit check L%u n_acc %d: max|dS| %.3g (max %.3g)\n",
                                    il, n_acc, md, mx);
                        }
                    }
                    free(a);
                    free(b);
                }
                continue;
            }
        }
        for (int t = 0; t < n_acc; t++) {
            if (!gdn_apply_d(s, (int)il, sb + (size_t)t * conv_ch,
                             zb + (size_t)t * n_v_h * d_v,
                             bb + (size_t)t * n_v_h, ab + (size_t)t * n_v_h,
                             s->d_gdn_replay))
                return false;
        }
    }
    if (!q4_hip_copy(s->d_res, q4_sess_pref_row(s, n_acc - 1), hc_dim))
        return false;
    s->n_kv = s->n_kv0 + (uint32_t)n_acc;
    s->pos = s->pos0 + n_acc;
    s->mtp_n = 0;
    return true;
}

int q4_sess_peek_ids(const q4_sess *s, const char *path, int32_t *ids, int max_ids,
                     int *prompt_n, int *nmsg, uint64_t *user_hash) {
    if (prompt_n) *prompt_n = 0;
    if (nmsg) *nmsg = 0;
    if (user_hash) *user_hash = 0;
    if (!s || !path || !ids || max_ids <= 0 || !s->g) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    q4_sess_hdr h;
    bool ok = rd_(f, &h, sizeof(h)) && h.magic == Q4_SESS_MAGIC &&
              h.version == Q4_SESS_VER && h.bytes_total == s->g->bytes_total &&
              h.n_layer == s->g->n_layer && h.n_expert == s->g->n_expert &&
              h.n_ids > 0 && (int)h.n_ids <= max_ids &&
              h.prompt_n <= h.n_ids;
    if (ok) ok = rd_(f, ids, (size_t)h.n_ids * sizeof(int32_t));
    fclose(f);
    if (!ok) return -1;
    if (prompt_n) *prompt_n = (int)h.prompt_n;
    if (nmsg) *nmsg = (int)h.nmsg;
    if (user_hash) *user_hash = h.user_hash;
    return (int)h.n_ids;
}

