/* MTP (multi-token prediction) draft head for Qwen3.8-Flash-Next.
 *
 * Loads an Unsloth `mtp-...-shared-Q8_0.gguf` nextn head (arch qwen4exp,
 * blk.<n_layer-1> = one full MoE layer + eh_proj glue) and runs speculative
 * decoding on top of the q4 session: the head drafts up to N tokens, the
 * main model verifies them in one batched pass. Verification is exact, so
 * output matches non-MTP decoding; only the speed changes.
 *
 * Head forward:
 *   e = rms_norm(emb(tok)) * enorm            (n_embd)
 *   h = grouped_rms(h_state) * hnorm          (hc x n_embd, per stream)
 *   res[c] = eh_proj([e | h[c]])              (concat order: e first)
 *   then one standard layer: hc_attn mix -> dense GQA attention -> MoE
 *   -> hc_ffn mix; res' = next hidden estimate,
 *   logits = output( hc_head_mixer(res') ).
 * When Q4_MTP_RT names a runtime directory, the 512 draft experts are a
 * resident Q2_0 pack and one graph replays the whole step (dense, MoE,
 * sample). Otherwise a draft step replays two graphs (dense prefix, then
 * logits and the device sample) with the MoE layer eager between them.
 * Rope position, the KV slot and the token embedding are pushed outside
 * the replay. Q4_MTP_GRAPH=0 keeps the eager step. The shared head borrows
 * token_embd + output from the main model.
 *
 * Head KV: one QSA-style linear buffer. Entries carry their own rope
 * position; draft-chain entries (estimates) are overwritten by the repair
 * pass with entries computed from the real residuals, so the invariant is:
 * slots [0, kv_fill) always describe the accepted sequence.
 */

#include "q4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MTP_PWS_N (32u << 20)    /* batched-MoE scratch floats (128 MiB) */
#define MTP_ROWS_CAP 64          /* hard cap on a resident head batch */
#define MTP_ROWS_LEGACY 8        /* non-resident / Q4_MTP_ROWS=8 rollback */
#define MTP_MAX_DRAFT 3

/* d_ws regions are computed in mtp_layout() from rows_alloc. XN holds the
 * hc mix (and eh_proj scratch); QG the attention pack; then mixed, block
 * and inj. d_resb is its own buffer so it survives every helper call.
 * The batched MoE scratch is m->d_pws, not d_ws. */

struct q4_mtp {
    q4_gguf g;            /* head gguf (nextn layer + glue) */
    int n_stop;           /* turn-ending ids: a draft chain ends at one */
    int32_t stop[2];
    q4_store *st;         /* head dense + shared expert store (VRAM) */
    q4_expert_cache *ex;  /* head routed experts */
    const q4_gguf *gm;    /* main model (token_embd, output, hparams) */
    q4_store *stm;
    int32_t layer;        /* head layer index */
    uint32_t n_embd, hc, hc_dim, hc_rank;
    uint32_t n_head, n_kvh, hd, n_rot;
    uint32_t n_vocab;
    float eps, rope_base;
    /* head KV */
    uint32_t kv_cap, kv_fill;
    /* Position of cell kv_fill-1 when cells [0, kv_fill) are real (the
     * main model's residuals, consecutive positions); -1 while a cycle's
     * chain estimates sit there or after a fallback.  Gates the head-KV
     * save beside a session snapshot. */
    int64_t kv_last;
    float *d_kv_k, *d_kv_v;
    /* scratch */
    float *d_ws, *d_pws, *d_resb;
    float *d_res;         /* current chain estimate (hc_dim) */
    float *d_emb, *d_cat, *d_logits;
    float *h_logits, *h_emb;
    /* d_ws layout (float offsets) and the batch cap. max_rows may drop
     * from rows_alloc to MTP_ROWS_LEGACY once the resident image is known
     * to be incomplete. rows_resolved latches that decision. */
    uint32_t max_rows, rows_alloc, rows_resolved;
    uint32_t off_xn, off_qg, off_mixed, off_block, off_inj, ws_n;
    /* stats */
    uint64_t proposed, accepted, cycles, gated;
    double tree_pa2; uint64_t tree_rej[8]; double tree_pa2d[8]; /* Q4_MTP_TREESTAT */
    double dt_ov[16]; uint64_t dt_n;   /* Q4_MTP_DTSTAT */
    /* Replayed draft step. With q2 set, g_step is the whole forward
     * including the resident experts. Otherwise g_pre is the dense prefix
     * (through ffn mix) and g_suf is the combine, logits and device sample,
     * with MoE eager between them. graph_off sticks after a capture
     * failure. graph_mode is 0 for greedy and 1 for top-k; graph_k is the
     * baked k. */
    void *g_pre, *g_suf, *g_step;
    int graph_off, graph_mode;
    uint32_t graph_k;
    /* Resident Q2_0 experts: q2 + id * Q4_Q2_PACK_BLOB. d_eids/d_ewts are
     * the device top-k (q2_k entries, allocated to 64). */
    uint8_t *q2;
    int32_t *d_eids;
    float *d_ewts;
    /* token row of every (expert, row) pair, token-major: pair i of row
     * i/k. Constant for every batch, filled once at open. */
    int32_t *d_tok;
    uint32_t q2_k;
    /* 1 = the last head projection used the slab top-k (no full-vocab
     * logits buffer); 0 = the full GEMV path. The draft step's sample is
     * read back accordingly. */
    int slab;
    /* Draft head over the rt/draft_vocab.bin token subset (Strata's
     * draft head): rows of the main output.weight gathered at open,
     * d_dvocab maps a subset row back to its token id inside the top-k.
     * NULL: the full 248320-row head. */
    uint8_t *d_dhead;
    int32_t *d_dvocab;
    uint32_t n_dvocab;
};

static int mtp_dbg(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_MTP_DEBUG") && getenv("Q4_MTP_DEBUG")[0] != '0';
    return v;
}

static const q4_tensor *mt(const q4_gguf *g, const char *fmt, int layer) {
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), fmt, layer);
    return q4_find_tensor(g, n);
}

static bool mmd(const q4_store *st, const q4_tensor *t, const float *d_x,
                float *d_y) {
    if (!t || t->n_dims < 2 || !d_x || !d_y) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    return q4_hip_gemv_dd(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y, 1.f);
}

static bool mmd_n(const q4_store *st, const q4_tensor *t, const float *d_x,
                  float *d_y, uint32_t n) {
    if (!t || t->n_dims < 2 || !d_x || !d_y || n == 0) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    /* 9..32 rows (eh_proj over n x hc streams in the KV repair: 3 tokens =
     * 12 rows) fall off the 2..8-row kernels onto a batch GEMV that took
     * ~0.5 ms for the 14 MB eh_proj; chunks of 8 reuse the n-row path. */
    if (n > 8 && n <= 32) {
        for (uint32_t r0 = 0; r0 < n; r0 += 8) {
            const uint32_t k = n - r0 < 8 ? n - r0 : 8;
            if (!q4_hip_gemv_dd_n(t->ggml_type, d, t->ne[1], t->ne[0],
                                  d_x + (size_t)r0 * t->ne[0],
                                  d_y + (size_t)r0 * t->ne[1], 1.f, k))
                return false;
        }
        return true;
    }
    return q4_hip_gemv_dd_n(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y, 1.f,
                            n);
}

static bool mmd_n_silu(const q4_store *st, const q4_tensor *t, const float *d_x,
                       float *d_y, float act, uint32_t n) {
    if (!t || t->n_dims < 2 || !d_x || !d_y || n == 0) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    if (q4_hip_gemv_dd_n_act(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y,
                             1.f, act, n))
        return true;
    return mmd_n(st, t, d_x, d_y, n) &&
           q4_hip_scale_silu(d_y, act, (uint64_t)n * t->ne[1]);
}

static const float *fd(const q4_store *st, const q4_tensor *t) {
    return t ? (const float *)q4_store_dev(st, t) : NULL;
}

static bool hc_mix_mn(q4_mtp *m, const float *d_res, float *d_mixed,
                      float *d_inj, bool attn, uint32_t n) {
    const q4_gguf *g = &m->g;
    const uint32_t hc_dim = m->hc_dim;
    const uint32_t n_embd = m->n_embd;
    const q4_tensor *wn = mt(g, attn ? "blk.%d.hc_attn_norm.weight"
                                     : "blk.%d.hc_ffn_norm.weight", m->layer);
    const q4_tensor *wd = mt(g, attn ? "blk.%d.hc_attn_down.weight"
                                     : "blk.%d.hc_ffn_down.weight", m->layer);
    const q4_tensor *wu = mt(g, attn ? "blk.%d.hc_attn_up.weight"
                                     : "blk.%d.hc_ffn_up.weight", m->layer);
    const q4_tensor *wi = mt(g, attn ? "blk.%d.hc_attn_inject.weight"
                                     : "blk.%d.hc_ffn_inject.weight", m->layer);
    float *xn = m->d_ws + m->off_xn;
    float *lo = xn + (size_t)n * hc_dim;
    float *gate = lo + (size_t)n * m->hc_rank;
    float *gated = gate + (size_t)n * hc_dim;
    if ((size_t)(gated - xn) + (size_t)n * hc_dim > m->off_qg - m->off_xn)
        return false;
    if (!q4_hip_grouped_rms_n(d_res, fd(m->st, wn), xn, n_embd, m->hc, m->eps,
                              n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc mix rms failed\n");
        return false;
    }
    if (!mmd_n_silu(m->st, wd, xn, lo, 1.f / (float)m->hc, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc down gemv failed\n");
        return false;
    }
    if (!mmd_n(m->st, wu, lo, gate, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc up gemv failed\n");
        return false;
    }
    if (!q4_hip_mulsig_mean_n(xn, gate, d_mixed, n_embd, m->hc, n)) return false;
    if (d_inj && wi && !mmd_n(m->st, wi, xn, d_inj, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc inject gemv failed\n");
        return false;
    }
    return true;
}

/* The head's K/V for n rows: k/v projections of the attention mix, k norm +
 * rope at positions base..base+n-1, appended at kv_fill. k and v are n-row
 * scratch inside the QG region. Shared by the full forward (attn_mn) and the
 * prompt pass, which needs nothing but these cells. */
static bool kv_append_mn(q4_mtp *m, int32_t base, const float *d_mix, float *k,
                         float *v, uint32_t n) {
    const q4_gguf *g = &m->g;
    const uint32_t hd = m->hd;
    if (m->kv_fill + n > m->kv_cap) return false;
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_k.weight", m->layer), d_mix, k, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn k gemv failed\n");
        return false;
    }
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_v.weight", m->layer), d_mix, v, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn v gemv failed\n");
        return false;
    }
    size_t kvw = (size_t)m->n_kvh * hd;
    if (!q4_hip_qsa_k_prep_n(k,
                             fd(m->st, mt(g, "blk.%d.attn_k_norm.weight", m->layer)),
                             m->n_kvh, hd, m->n_rot, base, m->rope_base, m->eps,
                             n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: k prep failed\n");
        return false;
    }
    if (!q4_hip_copy_async(m->d_kv_k + (size_t)m->kv_fill * kvw, k,
                           (size_t)n * kvw)) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: attn kv copy k failed fill=%u n=%u\n",
                    m->kv_fill, n);
        return false;
    }
    if (!q4_hip_copy_async(m->d_kv_v + (size_t)m->kv_fill * kvw, v,
                           (size_t)n * kvw)) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: attn kv copy v failed fill=%u n=%u\n",
                    m->kv_fill, n);
        return false;
    }
    m->kv_fill += n;
    return true;
}

/* Head attention window: the most recent cells kept when the KV fills up.
 * Q4_MTP_WINDOW, default 32768 (Strata's mtp_window), at most half the
 * capacity so a compaction always frees room for a whole prefill chunk. */
static uint32_t mtp_window(const q4_mtp *m) {
    static long v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MTP_WINDOW");
        v = e && e[0] ? atol(e) : 32768;
        if (v < 256) v = 256;
    }
    uint32_t w = (uint32_t)v;
    return w > m->kv_cap / 2 ? m->kv_cap / 2 : w;
}

/* Make room for `need` more cells. When they do not fit, the newest
 * window's cells move to the front (each cell's K already carries its rope
 * position, so a moved cell attends exactly as before) and the older ones
 * drop out of the head's view. Called only where no cell index is held
 * across the call: before a prompt chunk and at the start of a cycle. */
static bool kv_make_room(q4_mtp *m, uint32_t need) {
    if (m->kv_fill + need <= m->kv_cap) return true;
    uint32_t keep = mtp_window(m);
    if (keep > m->kv_fill) keep = m->kv_fill;
    if (keep + need > m->kv_cap) return false;
    const uint32_t from = m->kv_fill - keep;
    if (!from) return false; /* unreachable: kv_fill <= keep fits above */
    const size_t kvw = (size_t)m->n_kvh * m->hd;
    /* dst < src: forward copies of at most `from` cells never overlap */
    for (uint32_t c = 0; c < keep; c += from) {
        uint32_t nc = keep - c < from ? keep - c : from;
        if (!q4_hip_copy_async(m->d_kv_k + c * kvw, m->d_kv_k + (from + c) * kvw,
                               nc * kvw) ||
            !q4_hip_copy_async(m->d_kv_v + c * kvw, m->d_kv_v + (from + c) * kvw,
                               nc * kvw))
            return false;
    }
    static int told;
    if (!told) {
        told = 1;
        fprintf(stderr, "q4 mtp: head KV full at %u cells; keeping the last %u\n",
                m->kv_fill, keep);
    }
    m->kv_fill = keep;
    return true;
}

/* Dense GQA attention of the head layer over its own KV.
 * d_mix (n x n_embd) -> d_y (n x n_embd); positions base..base+n-1. */
static bool attn_mn(q4_mtp *m, int32_t base, const float *d_mix, float *d_y,
                    uint32_t n) {
    const q4_gguf *g = &m->g;
    const uint32_t hd = m->hd;
    if (m->kv_fill + n > m->kv_cap) return false;
    float *qg = m->d_ws + m->off_qg;
    float *k = qg + (size_t)n * m->n_head * hd * 2;
    float *v = k + (size_t)n * m->n_kvh * hd;
    float *qbat = v + (size_t)n * m->n_kvh * hd;
    float *gbat = qbat + (size_t)n * m->n_head * hd;
    float *attn = qg; /* overwrite packed Q after prep */
    if ((size_t)(gbat - qg) + (size_t)n * m->n_head * hd >
        m->off_mixed - m->off_qg)
        return false;
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_q.weight", m->layer), d_mix, qg, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn q gemv failed\n");
        return false;
    }
    if (!q4_hip_qsa_q_prep_n(qg,
                             fd(m->st, mt(g, "blk.%d.attn_q_norm.weight", m->layer)),
                             qbat, gbat, m->n_head, hd, m->n_rot, base,
                             m->rope_base, m->eps, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: q prep failed\n");
        return false;
    }
    if (!kv_append_mn(m, base, d_mix, k, v, n)) return false;
    float scale = 1.f / sqrtf((float)hd);
    /* The attention window is bounded by the number of rows actually
     * written (kv_fill - n before this batch), not by `base`: on a
     * session-cache hit the head KV starts empty while `base` jumps to
     * n_kv-1, and indexing by position would read unwritten rows. */
    if (!q4_hip_qsa_decode_n(qbat, m->d_kv_k, m->d_kv_v, gbat, attn, m->n_head,
                             m->n_kvh, hd, n, m->kv_fill - n, m->kv_cap,
                             scale, 0)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: qsa decode failed\n");
        return false;
    }
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_output.weight", m->layer), attn, d_y,
               n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn out gemv failed\n");
        return false;
    }
    return true;
}

/* Host-decode an embedding row from the main model store. */
static bool emb_row(q4_mtp *m, int32_t tok, float *emb) {
    const q4_tensor *temb = q4_find_tensor(m->gm, "token_embd.weight");
    if (!temb || tok < 0 || (uint32_t)tok >= m->gm->n_vocab) return false;
    uint64_t rb = q4_row_bytes(temb->ggml_type, temb->ne[0]);
    const uint8_t *w = q4_store_get(m->stm, temb);
    if (!w || rb == 0) return false;
    if (m->n_embd > 2560) return false;
    return q4_dequant_row(temb->ggml_type, w + (uint64_t)tok * rb, temb->ne[0],
                          emb);
}

/* (token, hidden row) -> head residual row (eh_proj glue). */
static bool eh_proj_m(q4_mtp *m, int32_t tok, const float *d_h, float *d_out) {
    const q4_tensor *enorm = mt(&m->g, "blk.%d.nextn.enorm.weight", m->layer);
    const q4_tensor *hnorm = mt(&m->g, "blk.%d.nextn.hnorm.weight", m->layer);
    const q4_tensor *proj = mt(&m->g, "blk.%d.nextn.eh_proj.weight", m->layer);
    if (!enorm || !hnorm || !proj) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj tensors missing\n");
        return false;
    }
    float emb[2560];
    if (!emb_row(m, tok, emb)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: emb_row failed tok=%d\n", tok);
        return false;
    }
    if (!q4_hip_h2d(m->d_emb, emb, m->n_embd * sizeof(float))) return false;
    float *e4 = m->d_ws + m->off_xn;              /* hc x n_embd */
    float *en = e4 + m->hc_dim;                   /* normed emb (n_embd) */
    float *h4 = en + m->n_embd;                   /* hc x n_embd */
    if ((size_t)(h4 - m->d_ws) + m->hc_dim > m->off_qg) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj ws overflow\n");
        return false;
    }
    if (!q4_hip_grouped_rms(m->d_emb, fd(m->st, enorm), en, m->n_embd, 1,
                            m->eps)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: enorm rms failed\n");
        return false;
    }
    if (!q4_hip_repeat_hc(en, e4, m->n_embd, m->hc)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: repeat_hc failed\n");
        return false;
    }
    /* A joint RMS over the whole hc*n_embd row was tried;
     * measured worse here (26.6% vs 35.5% acceptance) — q4's residual
     * streams carry different per-stream scales, which joint RMS is not
     * invariant to. Keep the per-stream form. */
    if (!q4_hip_grouped_rms(d_h, fd(m->st, hnorm), h4, m->n_embd, m->hc,
                            m->eps)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hnorm rms failed\n");
        return false;
    }
    /* One kernel, same layout as the per-stream copies: row c = [e_c | h_c].
     * The embedding upload above stays synchronous; this buffer is a stack
     * row and the n<=8 caller reuses it on the next token. */
    if (!q4_hip_eh_cat(m->d_cat, e4, h4, m->n_embd, m->hc)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj cat failed\n");
        return false;
    }
    if (!mmd_n(m->st, proj, m->d_cat, d_out, m->hc)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj gemv failed\n");
        return false;
    }
    return true;
}

/* n>1 eh_proj. Same per-stream RMS and [e|h] concat as eh_proj_m, one
 * H2D and one GEMV for the whole batch. n==1 stays on eh_proj_m (one
 * embedding upload from a stack row). */
static bool eh_proj_n(q4_mtp *m, const int32_t *toks, const float *d_h,
                      float *d_out, uint32_t n) {
    const q4_tensor *enorm = mt(&m->g, "blk.%d.nextn.enorm.weight", m->layer);
    const q4_tensor *hnorm = mt(&m->g, "blk.%d.nextn.hnorm.weight", m->layer);
    const q4_tensor *proj = mt(&m->g, "blk.%d.nextn.eh_proj.weight", m->layer);
    if (!enorm || !hnorm || !proj || !m->h_emb || n > m->rows_alloc)
        return false;
    if (m->n_embd > 2560) return false;
    for (uint32_t t = 0; t < n; t++) {
        if (!emb_row(m, toks[t], m->h_emb + (size_t)t * m->n_embd)) {
            if (mtp_dbg())
                fprintf(stderr, "q4 mtp: emb_row failed tok=%d\n", toks[t]);
            return false;
        }
    }
    if (!q4_hip_h2d(m->d_emb, m->h_emb,
                    (size_t)n * m->n_embd * sizeof(float)))
        return false;
    /* XN is larger than n*(2*hc_dim + n_embd), so the three temps fit. */
    float *en = m->d_ws + m->off_xn;                  /* n x n_embd */
    float *e4 = en + (size_t)n * m->n_embd;           /* n x hc x n_embd */
    float *h4 = e4 + (size_t)n * m->hc_dim;           /* n x hc x n_embd */
    if ((size_t)(h4 - (m->d_ws + m->off_xn)) + (size_t)n * m->hc_dim >
        m->off_qg - m->off_xn) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj_n ws overflow\n");
        return false;
    }
    if (!q4_hip_grouped_rms_n(m->d_emb, fd(m->st, enorm), en, m->n_embd, 1,
                              m->eps, n))
        return false;
    if (!q4_hip_repeat_hc_n(en, e4, m->n_embd, m->hc, n)) return false;
    if (!q4_hip_grouped_rms_n(d_h, fd(m->st, hnorm), h4, m->n_embd, m->hc,
                              m->eps, n))
        return false;
    if (!q4_hip_eh_cat(m->d_cat, e4, h4, m->n_embd, n * m->hc)) return false;
    if (!mmd_n(m->st, proj, m->d_cat, d_out, n * m->hc)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj_n gemv failed\n");
        return false;
    }
    return true;
}

/* lm head: hc_head mixer (carries the output norm) + shared output GEMV.
 * Input: residual row (d_res). Fills m->h_logits. */
/* The head's own hyper-connection mix into d_mixed (the lm-head input).
 * Split out of logits_m so the head projection can be run either as a
 * full-vocab GEMV (host sampling fallback) or as a slab top-k. */
static bool logits_mix_m(q4_mtp *m, float *d_mixed) {
    const q4_gguf *g = &m->g;
    const uint32_t hc_dim = m->hc_dim;
    const uint32_t n_embd = m->n_embd;
    float *xn = m->d_ws + m->off_xn;
    float *lo = xn + hc_dim;
    float *gate = lo + m->hc_rank;
    if ((size_t)(d_mixed - m->d_ws) + n_embd > m->off_qg) return false;
    const q4_tensor *hn = mt(g, "blk.%d.nextn.hc_head_norm.weight", m->layer);
    const q4_tensor *hd_ = mt(g, "blk.%d.nextn.hc_head_down.weight", m->layer);
    const q4_tensor *hu = mt(g, "blk.%d.nextn.hc_head_up.weight", m->layer);
    if (!q4_hip_grouped_rms(m->d_res, fd(m->st, hn), xn, n_embd, m->hc, m->eps))
        return false;
    if (!mmd(m->st, hd_, xn, lo)) return false;
    if (!q4_hip_scale_silu(lo, 1.f / (float)m->hc, m->hc_rank)) return false;
    if (!mmd(m->st, hu, lo, gate)) return false;
    return q4_hip_mulsig_mean(xn, gate, d_mixed, n_embd, m->hc);
}

/* d_mixed must come from logits_mix_m. Full-vocab logits into d_logits
 * (0.95 MiB device + host), allocated on demand: the head top-k path never
 * needs it. */
static bool logits_bufs_ensure(q4_mtp *m) {
    if (m->d_logits && m->h_logits) return true;
    size_t n = (size_t)m->n_vocab * sizeof(float);
    if (!m->d_logits)
        m->d_logits = q4_hip_malloc((size_t)m->n_vocab * sizeof(float));
    if (!m->h_logits) m->h_logits = malloc(n);
    if (!m->d_logits || !m->h_logits) {
        q4_hip_free(m->d_logits);
        free(m->h_logits);
        m->d_logits = NULL;
        m->h_logits = NULL;
        return false;
    }
    return true;
}

static bool logits_full_m(q4_mtp *m, const float *d_mixed) {
    if (!logits_bufs_ensure(m)) return false;
    const q4_tensor *out = q4_find_tensor(m->gm, "output.weight");
    if (!out) return false;
    return mmd(m->stm, out, d_mixed, m->d_logits);
}

/* Top-k over the draft head's vocab on the device (q4_hip_head_topk: one
 * GEMV into its own row buffer + the wide selection; the subset head maps
 * rows back to token ids). Writes the pairs into the shared top-k pinned
 * buffers; read them back with q4_hip_topk_read after a sync. */
static bool logits_topk_m(q4_mtp *m, const float *d_mixed, uint32_t k) {
    const q4_tensor *out = q4_find_tensor(m->gm, "output.weight");
    if (!out) return false;
    if (m->d_dhead) {
        if (!q4_hip_head_topk(out->ggml_type, m->d_dhead, out->ne[0],
                              m->n_dvocab, d_mixed, k, m->d_dvocab))
            return false;
        m->slab = 1;
        return true;
    }
    const uint8_t *d = q4_store_dev(m->stm, out);
    if (!d) return false;
    if (!q4_hip_head_topk(out->ggml_type, d, out->ne[0], out->ne[1], d_mixed, k,
                          NULL))
        return false;
    m->slab = 1;
    return true;
}

static bool logits_m(q4_mtp *m) {
    const uint32_t n_embd = m->n_embd;
    float *xn = m->d_ws + m->off_xn;
    float *d_mixed = xn + 3u * m->hc_dim + m->hc_rank;
    if ((size_t)(d_mixed - m->d_ws) + n_embd > m->off_qg) return false;
    if (!logits_mix_m(m, d_mixed)) return false;
    m->slab = 0;
    return logits_full_m(m, d_mixed);
}

/* d_ws regions for `rows` tokens. Sizes match the live checks in hc_mix_mn
 * and attn_mn (relative to each region, not an absolute constant). */
static void mtp_layout(q4_mtp *m, uint32_t rows) {
    uint32_t xn = rows * (3u * m->hc_dim + m->hc_rank);
    uint32_t qg = rows * m->hd * (4u * m->n_head + 2u * m->n_kvh);
    uint32_t mixed = rows * m->n_embd;
    uint32_t block = rows * m->n_embd;
    uint32_t inj = rows * m->hc;
    m->off_xn = 0;
    m->off_qg = xn;
    m->off_mixed = xn + qg;
    m->off_block = m->off_mixed + mixed;
    m->off_inj = m->off_block + block;
    m->ws_n = m->off_inj + inj;
}

/* Batch cap. Above 8 only when every head expert is in the DMA-registered
 * image: a wider group does not fit in the 128-slot L1, and staging it
 * stickies the set until moe fail #14 disables MTP. Q4_MTP_ROWS=8 keeps
 * today's per-token path even if the image is resident. */
static uint32_t mtp_rows(q4_mtp *m) {
    if (m->rows_resolved) return m->max_rows;
    uint32_t want = m->max_rows;
    if (want > MTP_ROWS_LEGACY) {
        int full = m->q2 != NULL;
        if (!full) {
            full = q4_expert_res_devbase(m->ex, m->layer) != NULL;
            uint32_t nexp = m->g.n_expert ? m->g.n_expert : 512;
            for (uint32_t e = 0; full && e < nexp; e++)
                if (q4_expert_res_slot(m->ex, m->layer, (int32_t)e) < 0)
                    full = 0;
        }
        if (!full) {
            fprintf(stderr,
                    "q4 mtp: head image is not fully resident; batch stays %d\n",
                    MTP_ROWS_LEGACY);
            want = MTP_ROWS_LEGACY;
        } else if (!m->q2) {
            fprintf(stderr, "q4 mtp: resident head, batch %u\n", want);
        }
    }
    m->max_rows = want;
    m->rows_resolved = 1;
    return want;
}

static bool mtp_moe_q2(q4_mtp *m, float *d_x, float *d_y, uint32_t n);

/* Byte offset of one gate/up slice inside a packed expert (rows = n_ff,
 * columns = in_dim, Q2_0 at 18 bytes per 64-column block). */
static uint64_t q2_gu_bytes(uint32_t n_ff, uint32_t in_dim) {
    return (uint64_t)n_ff * ((uint64_t)in_dim >> 6) * 18u;
}

/* Batched routed MoE over the resident pack, for a wide prompt group. One
 * router GEMV for every row, one device softmax+top-k, then two launches
 * over the (expert, row) pairs — the per-token loop is ~90 launches per
 * token, this is ~10 per whole group. Only the prompt path calls this:
 * draft/repair steps stay at n <= 3 on the per-token loop, whose scratch
 * overlaps the captured draft graph. */
static bool mtp_moe_q2_win(q4_mtp *m, float *d_x, float *d_y, uint32_t n) {
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff = m->g.n_ff_exp ? m->g.n_ff_exp : Q4_Q2_PACK_FF;
    const uint32_t n_exp = m->g.n_expert;
    const uint32_t k = m->q2_k;
    if (!m->q2 || !m->d_eids || !m->d_ewts || !m->d_tok || !n || !k ||
        k > 64 || !n_exp || n_exp > 512 || n > m->rows_alloc)
        return false;
    const uint32_t nr = n * k;
    const uint64_t gu = q2_gu_bytes(n_ff, n_embd);
    if (getenv("Q4_MTP_DEBUG")) {
        static unsigned seen = 0;
        if (seen++ < 8)
            fprintf(stderr, "q4 mtp: moe_q2_win n=%u k=%u nr=%u n_ff=%u "
                            "n_embd=%u n_exp=%u\n",
                    n, k, nr, n_ff, n_embd, n_exp);
    }
    const q4_tensor *rt = mt(&m->g, "blk.%d.ffn_gate_inp.weight", m->layer);
    const q4_tensor *gs = mt(&m->g, "blk.%d.ffn_gate_shexp.weight", m->layer);
    const q4_tensor *us = mt(&m->g, "blk.%d.ffn_up_shexp.weight", m->layer);
    const q4_tensor *ds = mt(&m->g, "blk.%d.ffn_down_shexp.weight", m->layer);
    if (!rt || !gs || !us || !ds) return false;
    /* Shared-expert input gate: one row of n_embd, so ne[0] == n_embd and
     * ne[1] == 1. Present on this head, and batched below. */
    const q4_tensor *sg = mt(&m->g, "blk.%d.ffn_gate_inp_shexp.weight",
                             m->layer);
    int gated = sg && sg->n_dims >= 2 && sg->ne[0] == n_embd && sg->ne[1] == 1;
    /* d_pws layout: rlog n*n_exp | hh nr*n_ff | hh2 nr*n_ff | dn n*n_embd |
     * dot n (one shared-gate logit per row) */
    float *rlog = m->d_pws;
    float *hh = rlog + (size_t)n * n_exp;
    float *hh2 = hh + (size_t)nr * n_ff;
    float *dn = hh2 + (size_t)nr * n_ff;
    float *dot = dn + (size_t)n * n_embd;
    if ((size_t)(dot + n - m->d_pws) > MTP_PWS_N) return false;
    if (!q4_hip_fill(d_y, 0.f, (size_t)n * n_embd)) return false;
    if (!mmd_n(m->st, rt, d_x, rlog, n)) return false;
    if (!q4_hip_route_pf(rlog, m->d_eids, m->d_ewts, n, n_exp, k)) return false;
    if (!q4_hip_moe_q2_win(m->q2, Q4_Q2_PACK_BLOB, m->d_eids, m->d_tok,
                           m->d_ewts, nr, 0, gu, 2u * gu, d_x, hh, d_y, n_ff,
                           n_embd, n_embd))
        return false;
    /* Shared expert: batched gate/up, one silu*mul over the whole group,
     * batched down, then add. */
    if (!mmd_n(m->st, gs, d_x, hh, n) || !mmd_n(m->st, us, d_x, hh2, n) ||
        !q4_hip_silu_mul(hh, hh2, (size_t)n * n_ff) ||
        !mmd_n(m->st, ds, hh, dn, n))
        return false;
    if (gated) {
        if (!mmd_n(m->st, sg, d_x, dot, n) || !q4_hip_sigmoid(dot, n) ||
            !q4_hip_axpy_rows(d_y, dn, dot, n_embd, n))
            return false;
    } else if (!q4_hip_add(d_y, d_y, dn, (size_t)n * n_embd)) {
        return false;
    }
    return true;
}

/* The prompt pass: head K/V for rows at positions base..base+n-1 and
 * nothing else. A cell depends only on (R_j, t_{j+1}) through eh_proj, the
 * attention-side hc mix and the k/v projections; the attention, the MoE and
 * the head logits of a prompt row are never read (every cycle reseeds the
 * chain from the verifier's residual), so they are not computed. The MoE is
 * also why the full pass is capped at 8 rows on a non-resident head; this
 * one takes the whole d_ws allocation. Q4_MTP_KVONLY=0 runs the full pass. */
static bool fwd_kv(q4_mtp *m, int32_t base, const int32_t *toks,
                   const float *d_h_rows, uint32_t n) {
    if (!n || n > m->rows_alloc) return false;
    float *d_mixed = m->d_ws + m->off_mixed;
    float *qg = m->d_ws + m->off_qg;
    float *k = qg;
    float *v = k + (size_t)n * m->n_kvh * m->hd;
    if ((size_t)(v - qg) + (size_t)n * m->n_kvh * m->hd > m->off_mixed - m->off_qg)
        return false;
    if (n == 1) {
        if (!eh_proj_m(m, toks[0], d_h_rows, m->d_resb)) return false;
    } else if (!eh_proj_n(m, toks, d_h_rows, m->d_resb, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: kv eh_proj_n failed\n");
        return false;
    }
    if (!hc_mix_mn(m, m->d_resb, d_mixed, NULL, true, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: kv hc_attn failed\n");
        return false;
    }
    return kv_append_mn(m, base, d_mixed, k, v, n);
}

static int mtp_kvonly(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MTP_KVONLY");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* One batched head pass over rows at positions base..base+n-1.
 * d_h_rows: n x hc_dim device rows (main-model residuals or the chain
 * estimate). Leaves the new estimate in m->d_res (last row).
 * logits: 0 = none, 1 = device d_logits only, 2 = also d2h m->h_logits. */
static bool fwd_m(q4_mtp *m, int32_t base, const int32_t *toks,
                  const float *d_h_rows, uint32_t n, uint32_t logits) {
    const q4_gguf *g = &m->g;
    const uint32_t n_embd = m->n_embd;
    const uint32_t hc_dim = m->hc_dim;
    if (!n || n > mtp_rows(m)) return false;
    if (m->kv_fill + n > m->kv_cap) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: head KV full fill=%u n=%u cap=%u\n",
                    m->kv_fill, n, m->kv_cap);
        return false;
    }
    float *d_mixed = m->d_ws + m->off_mixed;
    float *d_block = m->d_ws + m->off_block;
    float *d_inj = m->d_ws + m->off_inj;
    /* n<=8 keeps the per-token glue, including every draft and repair
     * step. Wider prefill batches are the only caller of eh_proj_n. */
    if (n <= MTP_ROWS_LEGACY) {
        for (uint32_t t = 0; t < n; t++)
            if (!eh_proj_m(m, toks[t], d_h_rows + (size_t)t * hc_dim,
                           m->d_resb + (size_t)t * hc_dim)) {
                if (mtp_dbg())
                    fprintf(stderr, "q4 mtp: fwd eh_proj failed t=%u tok=%d\n",
                            t, toks[t]);
                return false;
            }
    } else if (!eh_proj_n(m, toks, d_h_rows, m->d_resb, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd eh_proj_n failed\n");
        return false;
    }
    if (!hc_mix_mn(m, m->d_resb, d_mixed, d_inj, true, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd hc_attn failed\n");
        return false;
    }
    if (!attn_mn(m, base, d_mixed, d_block, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd attn failed\n");
        return false;
    }
    if (!q4_hip_hc_combine_n(m->d_resb, d_block, d_inj, n_embd, m->hc, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd comb1 failed\n");
        return false;
    }
    if (!hc_mix_mn(m, m->d_resb, d_mixed, d_inj, false, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd hc_ffn failed\n");
        return false;
    }
    if (m->q2) {
        if (!mtp_moe_q2(m, d_mixed, d_block, n)) {
            if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd moe failed\n");
            return false;
        }
    } else if (!q4_moe_layer_dev_n(g, m->st, m->ex, m->layer, d_mixed, d_block,
                                   m->d_pws, n, MTP_PWS_N)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd moe failed\n");
        return false;
    }
    if (!q4_hip_hc_combine_n(m->d_resb, d_block, d_inj, n_embd, m->hc, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd comb2 failed\n");
        return false;
    }
    /* Same stream as logits_m. The host does not read d_res. */
    if (!q4_hip_copy_async(m->d_res, m->d_resb + (size_t)(n - 1) * hc_dim,
                           hc_dim)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd res copy failed\n");
        return false;
    }
    if (logits && !logits_m(m)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd logits failed\n");
        return false;
    }
    if (logits >= 2 &&
        !q4_hip_d2h(m->h_logits, m->d_logits, m->n_vocab * sizeof(float))) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd logits d2h failed\n");
        return false;
    }
    return true;
}

/* ---- open / close ---- */

/* Router + resident Q2 experts + shared expert, one token at a time.
 * d_eids/d_ewts are reused across tokens: the previous token's kernels
 * read them on this stream before the next top-k writes them. Scratch is
 * carved from d_pws and reused the same way. */
static bool mtp_moe_q2(q4_mtp *m, float *d_x, float *d_y, uint32_t n) {
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff = m->g.n_ff_exp ? m->g.n_ff_exp : Q4_Q2_PACK_FF;
    const uint32_t n_exp = m->g.n_expert;
    const uint32_t k = m->q2_k;
    if (!m->q2 || !m->d_eids || !m->d_ewts || !n || !k || k > 64 || !n_exp ||
        n_exp > 512)
        return false;
    /* A wide prompt group is batched (mtp_moe_q2_win): ~10 launches for the
     * whole group instead of ~90 per token, and the shared-expert input gate
     * is applied per row there. n == 1 (every draft and repair step) stays
     * on the per-token loop, whose scratch the capture replays. */
    if (n > 1) return mtp_moe_q2_win(m, d_x, d_y, n);
    const uint64_t gu = q2_gu_bytes(n_ff, n_embd);
    float *rlog = m->d_pws;
    float *hh = rlog + 528;
    float *hh2 = hh + (size_t)k * n_ff;
    float *dn = hh2 + (size_t)k * n_ff;
    float *dot = dn + (size_t)k * n_embd;
    if ((size_t)(dot + 4 - m->d_pws) > MTP_PWS_N) return false;
    const q4_tensor *rt = mt(&m->g, "blk.%d.ffn_gate_inp.weight", m->layer);
    const q4_tensor *gs = mt(&m->g, "blk.%d.ffn_gate_shexp.weight", m->layer);
    const q4_tensor *us = mt(&m->g, "blk.%d.ffn_up_shexp.weight", m->layer);
    const q4_tensor *ds = mt(&m->g, "blk.%d.ffn_down_shexp.weight", m->layer);
    const q4_tensor *sg = mt(&m->g, "blk.%d.ffn_gate_inp_shexp.weight", m->layer);
    if (!rt || !gs || !us || !ds) return false;
    for (uint32_t t = 0; t < n; t++) {
        const float *x = d_x + (size_t)t * n_embd;
        float *y = d_y + (size_t)t * n_embd;
        if (!mmd(m->st, rt, x, rlog) || !q4_hip_fill(y, 0.f, n_embd) ||
            !q4_hip_softmax_topk_launch(rlog, n_exp, k, m->d_eids, m->d_ewts))
            return false;
        if (!q4_hip_moe_q2(m->q2, Q4_Q2_PACK_BLOB, m->d_eids, m->d_ewts, k, 0,
                           gu, 2u * gu, x, hh, hh2, dn, y, n_ff, n_embd,
                           n_embd))
            return false;
        /* Routed down has consumed hh/hh2/dn. Shared reuses them. */
        if (!mmd(m->st, gs, x, hh) || !mmd(m->st, us, x, hh2) ||
            !q4_hip_silu_mul(hh, hh2, n_ff) || !mmd(m->st, ds, hh, dn))
            return false;
        if (sg && sg->n_dims >= 2 && sg->ne[0] == n_embd && sg->ne[1] == 1) {
            if (!mmd(m->st, sg, x, dot) || !q4_hip_sigmoid(dot, 1) ||
                !q4_hip_axpy_dalpha(y, dn, dot, n_embd))
                return false;
        } else if (!q4_hip_axpy(y, dn, 1.f, n_embd)) {
            return false;
        }
    }
    return true;
}

static q4_tensor *mt_mut(q4_gguf *g, const char *fmt, int layer) {
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof n, fmt, layer);
    if (!g || !g->tensors) return NULL;
    for (uint32_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->tensors[i].name, n)) return &g->tensors[i];
    return NULL;
}

typedef struct {
    char name[160];
    uint32_t ty, rows, cols;
    uint64_t off, nbytes;
} mtp_rt_ent;

static int mtp_rt_kind(const char *s) {
    if (!strcmp(s, "q8_0")) return Q4_T_Q8_0;
    if (!strcmp(s, "bf16")) return Q4_T_BF16;
    if (!strcmp(s, "f32")) return Q4_T_F32;
    return -1;
}

static int mtp_rt_load_txt(const char *path, mtp_rt_ent *e, int cap) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int n = 0;
    while (n < cap && fgets(line, sizeof line, f)) {
        char name[160], kind[16];
        unsigned rows = 0, cols = 0;
        unsigned long long off = 0, nb = 0;
        if (sscanf(line, "%159s %15s %u %u %llu %llu", name, kind, &rows, &cols,
                   &off, &nb) != 6)
            continue;
        int ty = mtp_rt_kind(kind);
        if (ty < 0 || !rows || !cols || !nb) continue;
        snprintf(e[n].name, sizeof e[n].name, "%s", name);
        e[n].ty = (uint32_t)ty;
        e[n].rows = rows;
        e[n].cols = cols;
        e[n].off = off;
        e[n].nbytes = nb;
        n++;
    }
    fclose(f);
    return n;
}

static const mtp_rt_ent *mtp_rt_get(const mtp_rt_ent *e, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (!strcmp(e[i].name, name)) return &e[i];
    return NULL;
}

static uint8_t *mtp_rt_read(FILE *bin, uint64_t off, uint64_t n) {
    if (!n) return NULL;
    uint8_t *p = malloc((size_t)n);
    if (!p) return NULL;
    if (fseek(bin, (long)off, SEEK_SET) != 0 ||
        fread(p, 1, (size_t)n, bin) != (size_t)n) {
        free(p);
        return NULL;
    }
    return p;
}

static int mtp_install(q4_mtp *m, q4_tensor *t, uint32_t ty, uint32_t rows,
                       uint32_t cols, const uint8_t *host, uint64_t nbytes) {
    if (!t || !host || !nbytes) return 0;
    void *d = q4_hip_malloc((size_t)nbytes);
    if (!d || !q4_hip_h2d(d, host, (size_t)nbytes) ||
        !q4_store_replace_dev(m->st, t, d)) {
        q4_hip_free(d);
        return 0;
    }
    t->ggml_type = ty;
    t->n_dims = 2;
    t->ne[0] = cols;
    t->ne[1] = rows;
    t->nbytes = nbytes;
    return 1;
}

/* 24 direct renames plus eh_proj, which is the per-row Q8 concat of the
 * two input projections. A miss keeps that GGUF tensor. */
static const struct {
    const char *q4;
    const char *rt;
} mtp_rt_map[] = {
    {"blk.%d.nextn.enorm.weight", "pre_fc_norm_embedding.weight"},
    {"blk.%d.nextn.hnorm.weight", "pre_fc_norm_hidden.weight"},
    {"blk.%d.nextn.hc_head_norm.weight", "hyper_connection_mixer.hc_norm.weight"},
    {"blk.%d.nextn.hc_head_down.weight",
     "hyper_connection_mixer.input_mix_weight_down.weight"},
    {"blk.%d.nextn.hc_head_up.weight",
     "hyper_connection_mixer.input_mix_weight_up.weight"},
    {"blk.%d.hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight"},
    {"blk.%d.hc_attn_down.weight",
     "attn_hyper_connection.input_mix_weight_down.weight"},
    {"blk.%d.hc_attn_up.weight",
     "attn_hyper_connection.input_mix_weight_up.weight"},
    {"blk.%d.hc_attn_inject.weight",
     "attn_hyper_connection.block_inject_weight.weight"},
    {"blk.%d.hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight"},
    {"blk.%d.hc_ffn_down.weight",
     "mlp_hyper_connection.input_mix_weight_down.weight"},
    {"blk.%d.hc_ffn_up.weight",
     "mlp_hyper_connection.input_mix_weight_up.weight"},
    {"blk.%d.hc_ffn_inject.weight",
     "mlp_hyper_connection.block_inject_weight.weight"},
    {"blk.%d.attn_q_norm.weight", "self_attn.q_norm.weight"},
    {"blk.%d.attn_k_norm.weight", "self_attn.k_norm.weight"},
    {"blk.%d.attn_q.weight", "self_attn.q_proj.weight"},
    {"blk.%d.attn_k.weight", "self_attn.k_proj.weight"},
    {"blk.%d.attn_v.weight", "self_attn.v_proj.weight"},
    {"blk.%d.attn_output.weight", "self_attn.o_proj.weight"},
    {"blk.%d.ffn_gate_inp.weight", "mlp.gate.weight"},
    {"blk.%d.ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight"},
    {"blk.%d.ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"},
    {"blk.%d.ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight"},
    {"blk.%d.ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"},
};

static void mtp_dense_overlay(q4_mtp *m, const char *dir) {
    char tp[4096], dp[4096];
    const int nmap = (int)(sizeof mtp_rt_map / sizeof mtp_rt_map[0]);
    const int expect = nmap + 1; /* + eh_proj */
    if (snprintf(tp, sizeof tp, "%s/dense.txt", dir) >= (int)sizeof tp ||
        snprintf(dp, sizeof dp, "%s/dense.bin", dir) >= (int)sizeof dp) {
        fprintf(stderr, "q4 mtp: dense runtime missing\n");
        fprintf(stderr, "q4 mtp: dense runtime overlay 0/%d\n", expect);
        return;
    }
    mtp_rt_ent ent[40];
    int ne = mtp_rt_load_txt(tp, ent, 40);
    FILE *bin = fopen(dp, "rb");
    if (ne <= 0 || !bin) {
        if (bin) fclose(bin);
        fprintf(stderr, "q4 mtp: dense runtime missing\n");
        fprintf(stderr, "q4 mtp: dense runtime overlay 0/%d\n", expect);
        return;
    }
    int got = 0;
    int logged = 0;
    for (int i = 0; i < nmap; i++) {
        const mtp_rt_ent *e = mtp_rt_get(ent, ne, mtp_rt_map[i].rt);
        q4_tensor *t = mt_mut(&m->g, mtp_rt_map[i].q4, m->layer);
        uint64_t rb = e ? q4_row_bytes(e->ty, e->cols) : 0;
        uint8_t *host = NULL;
        int ok = e && t && rb && rb * e->rows == e->nbytes;
        if (ok) host = mtp_rt_read(bin, e->off, e->nbytes);
        if (host && mtp_install(m, t, e->ty, e->rows, e->cols, host, e->nbytes))
            got++;
        else if (!logged) {
            logged = 1;
            fprintf(stderr, "q4 mtp: dense runtime kept gguf tensor %s\n",
                    mtp_rt_map[i].q4);
        }
        free(host);
    }
    /* eh_proj = per-row concat of the two Q8 projections (2560 → 5120). */
    {
        const mtp_rt_ent *we = mtp_rt_get(ent, ne, "fc_embedding.weight");
        const mtp_rt_ent *wh = mtp_rt_get(ent, ne, "fc_hidden.weight");
        q4_tensor *t = mt_mut(&m->g, "blk.%d.nextn.eh_proj.weight", m->layer);
        const uint32_t row_b = 2720;
        const uint32_t rows = 2560;
        int shape = we && wh && t && we->ty == Q4_T_Q8_0 && wh->ty == Q4_T_Q8_0 &&
                    we->rows == rows && wh->rows == rows && we->cols == rows &&
                    wh->cols == rows && we->nbytes == (uint64_t)rows * row_b &&
                    wh->nbytes == (uint64_t)rows * row_b;
        uint8_t *emb = shape ? mtp_rt_read(bin, we->off, we->nbytes) : NULL;
        uint8_t *hid = shape ? mtp_rt_read(bin, wh->off, wh->nbytes) : NULL;
        uint8_t *cat = (emb && hid) ? malloc((size_t)rows * row_b * 2) : NULL;
        if (cat) {
            for (uint32_t r = 0; r < rows; r++) {
                memcpy(cat + (size_t)r * row_b * 2, emb + (size_t)r * row_b,
                       row_b);
                memcpy(cat + (size_t)r * row_b * 2 + row_b,
                       hid + (size_t)r * row_b, row_b);
            }
            if (mtp_install(m, t, Q4_T_Q8_0, rows, rows * 2, cat,
                            (uint64_t)rows * row_b * 2))
                got++;
            else if (!logged) {
                logged = 1;
                fprintf(stderr,
                        "q4 mtp: dense runtime kept gguf tensor eh_proj\n");
            }
        } else if (!logged) {
            logged = 1;
            fprintf(stderr, "q4 mtp: dense runtime kept gguf tensor eh_proj\n");
        }
        free(cat);
        free(emb);
        free(hid);
    }
    fclose(bin);
    fprintf(stderr, "q4 mtp: dense runtime overlay %d/%d\n", got, expect);
}

/* Q4_MTP_RT is a directory of experts.bin + dense.bin + dense.txt. A shape
 * or size mismatch refuses the pack and leaves the GGUF head in place. */
static int mtp_pack_load(q4_mtp *m) {
    const char *dir = getenv("Q4_MTP_RT");
    if (!dir || !dir[0]) return 0;
    if (m->g.n_expert != Q4_Q2_PACK_NE || m->n_embd != Q4_Q2_PACK_H ||
        m->g.n_ff_exp != Q4_Q2_PACK_FF || m->g.n_expert_used == 0 ||
        m->g.n_expert_used > 64) {
        fprintf(stderr, "q4 mtp: draft expert pack refused (shape)\n");
        return 0;
    }
    char ep[4096];
    if (snprintf(ep, sizeof ep, "%s/experts.bin", dir) >= (int)sizeof ep) {
        fprintf(stderr, "q4 mtp: draft expert pack refused (shape)\n");
        return 0;
    }
    FILE *f = fopen(ep, "rb");
    if (!f) {
        fprintf(stderr, "q4 mtp: draft expert pack missing\n");
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        fprintf(stderr, "q4 mtp: draft expert pack refused (size)\n");
        return 0;
    }
    long sz = ftell(f);
    const long want = (long)Q4_Q2_PACK_NE * (long)Q4_Q2_PACK_BLOB;
    if (sz != want || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "q4 mtp: draft expert pack refused (size)\n");
        return 0;
    }
    uint8_t *dev = q4_hip_malloc((size_t)Q4_Q2_PACK_NE * Q4_Q2_PACK_BLOB);
    uint8_t *plane = malloc(Q4_Q2_PACK_BLOB);
    uint8_t *slot = malloc(Q4_Q2_PACK_BLOB);
    /* A batched routed-MoE group needs rows*k entries, not one token's k:
 * MTP_ROWS_CAP rows x topk. */
    int32_t *eids =
        dev ? q4_hip_malloc((size_t)MTP_ROWS_CAP * 64u * sizeof(int32_t))
             : NULL;
    float *ewts =
        dev ? q4_hip_malloc((size_t)MTP_ROWS_CAP * 64u * sizeof(float))
             : NULL;
    if (!dev || !plane || !slot || !eids || !ewts) {
        free(plane);
        free(slot);
        q4_hip_free(dev);
        q4_hip_free(eids);
        q4_hip_free(ewts);
        fclose(f);
        fprintf(stderr, "q4 mtp: draft expert pack alloc failed\n");
        return 0;
    }
    int ok = 1;
    for (uint32_t i = 0; ok && i < Q4_Q2_PACK_NE; i++) {
        if (fread(plane, 1, Q4_Q2_PACK_BLOB, f) != Q4_Q2_PACK_BLOB ||
            !q4_q2_planar_to_ggml(plane, slot) ||
            !q4_hip_h2d(dev + (size_t)i * Q4_Q2_PACK_BLOB, slot,
                        Q4_Q2_PACK_BLOB))
            ok = 0;
    }
    fclose(f);
    free(plane);
    free(slot);
    if (!ok) {
        q4_hip_free(dev);
        q4_hip_free(eids);
        q4_hip_free(ewts);
        fprintf(stderr, "q4 mtp: draft expert pack load failed\n");
        return 0;
    }
    m->q2 = dev;
    m->d_eids = eids;
    m->d_ewts = ewts;
    m->q2_k = m->g.n_expert_used;
    fprintf(stderr, "q4 mtp: resident draft experts, %u x Q2_0 (675 MiB)\n",
            Q4_Q2_PACK_NE);
    mtp_dense_overlay(m, dir);
    return 1;
}

/* Q4_MTP_SUBSET=1: rt/draft_vocab.bin (int32 token ids, Strata's
 * tools/draft_vocab.py) -> a draft head of just those rows of the main
 * output.weight (106299 of 248320, 213 MiB). Off by default: measured
 * 2026-10-11 (greedy, L1 3300) it saves ~1 ms of draft per cycle but a
 * token outside the subset is never drafted — code tok/cycle 3.08 -> 2.75,
 * decode 42.7 -> 39.3 t/s; ja/agent within noise. Needs the wide top-k
 * (Q4_TOPK_WIDE) for the id map. */
static void dhead_load(q4_mtp *m) {
    const char *dir = getenv("Q4_MTP_RT");
    const char *on = getenv("Q4_MTP_SUBSET");
    const char *tw = getenv("Q4_TOPK_WIDE");
    if (!dir || !dir[0] || !(on && on[0] == '1') || (tw && tw[0] == '0'))
        return;
    const q4_tensor *out = q4_find_tensor(m->gm, "output.weight");
    const uint8_t *src = out ? q4_store_dev(m->stm, out) : NULL;
    if (!src) return;
    char path[4096];
    if (snprintf(path, sizeof path, "%s/draft_vocab.bin", dir) >=
        (int)sizeof path)
        return;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t *ids = NULL;
    uint32_t n = sz > 0 && sz % 4 == 0 ? (uint32_t)(sz / 4) : 0;
    if (n) ids = malloc((size_t)sz);
    if (!ids || fread(ids, 4, n, f) != n) {
        fclose(f);
        free(ids);
        fprintf(stderr, "q4 mtp: %s unreadable; full draft head\n", path);
        return;
    }
    fclose(f);
    for (uint32_t i = 0; i < n; i++)
        if (ids[i] < 0 || (uint32_t)ids[i] >= m->n_vocab ||
            (uint64_t)out->ne[1] <= (uint64_t)ids[i]) {
            free(ids);
            fprintf(stderr, "q4 mtp: %s has id out of range; full draft head\n",
                    path);
            return;
        }
    uint64_t rb = q4_row_bytes(out->ggml_type, out->ne[0]);
    m->d_dvocab = q4_hip_malloc((size_t)n * sizeof(int32_t));
    m->d_dhead = q4_hip_malloc((size_t)n * rb);
    if (!m->d_dvocab || !m->d_dhead ||
        !q4_hip_h2d(m->d_dvocab, ids, (size_t)n * sizeof(int32_t)) ||
        !q4_hip_gather_rows(m->d_dhead, src, rb, m->d_dvocab, n) ||
        !q4_hip_stream_sync()) {
        q4_hip_free(m->d_dvocab);
        q4_hip_free(m->d_dhead);
        m->d_dvocab = NULL;
        m->d_dhead = NULL;
        q4_hip_clear();
        free(ids);
        fprintf(stderr, "q4 mtp: draft head subset failed; full draft head\n");
        return;
    }
    free(ids);
    m->n_dvocab = n;
    fprintf(stderr, "q4 mtp: draft head over %u tokens (%.1f MiB)\n", n,
            (double)n * (double)rb / 1048576.0);
}

q4_mtp *q4_mtp_open(const char *head_path, const q4_gguf *gmain,
                    q4_store *stmain, uint32_t head_kv, uint32_t n_l1,
                    uint64_t l2_bytes) {
    if (!head_path || !head_path[0] || !gmain || !stmain || !q4_hip_ok())
        return NULL;
    q4_mtp *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->gm = gmain;
    m->stm = stmain;
    /* keep_mtp_layer=1: this file IS the draft block; peeling it would
     * drop blk.N from the head's own forward. */
    if (!q4_gguf_open_ex(&m->g, head_path, 1)) {
        fprintf(stderr, "q4 mtp: failed to open %s\n", head_path);
        free(m);
        return NULL;
    }
    /* Pairing check: the head must be a nextn export of THIS model.
     * Mismatched pairs are rejected rather than producing bad drafts. */
    if (!m->g.arch || strcmp(m->g.arch, "qwen4exp") != 0 ||
        m->g.n_embd != gmain->n_embd ||
        (m->g.hc_mult ? m->g.hc_mult : 4) !=
            (gmain->hc_mult ? gmain->hc_mult : 4) ||
        m->g.hc_rank != gmain->hc_rank ||
        m->g.n_expert != gmain->n_expert ||
        m->g.n_expert_used != gmain->n_expert_used ||
        m->g.n_head != gmain->n_head ||
        m->g.n_head_kv != gmain->n_head_kv ||
        (m->g.n_vocab && gmain->n_vocab && m->g.n_vocab != gmain->n_vocab)) {
        fprintf(stderr, "q4 mtp: head does not match this model\n");
        q4_mtp_close(m);
        return NULL;
    }
    if (m->g.n_layer < 1 || m->g.n_layer >= Q4_MAX_LAYER) {
        q4_mtp_close(m);
        return NULL;
    }
    m->layer = (int32_t)m->g.n_layer - 1;
    if (!mt(&m->g, "blk.%d.nextn.eh_proj.weight", m->layer) ||
        !mt(&m->g, "blk.%d.nextn.hnorm.weight", m->layer) ||
        !mt(&m->g, "blk.%d.ffn_gate_exps.weight", m->layer)) {
        fprintf(stderr, "q4 mtp: %s is not a qwen4exp MTP head\n", head_path);
        q4_mtp_close(m);
        return NULL;
    }
    m->n_embd = gmain->n_embd;
    m->hc = gmain->hc_mult ? gmain->hc_mult : 4;
    m->hc_dim = m->hc * m->n_embd;
    m->hc_rank = gmain->hc_rank ? gmain->hc_rank : 320;
    m->n_head = gmain->n_head ? gmain->n_head : 24;
    m->n_kvh = gmain->n_head_kv ? gmain->n_head_kv : 2;
    m->hd = gmain->n_embd_head_k ? gmain->n_embd_head_k : 256;
    m->n_rot = m->g.n_rot ? m->g.n_rot : (gmain->n_rot ? gmain->n_rot : 64);
    m->n_vocab = gmain->n_vocab;
    m->eps = gmain->rms_eps ? gmain->rms_eps : 1e-6f;
    m->rope_base = gmain->rope_freq_base ? gmain->rope_freq_base : 1e7f;
    m->kv_cap = head_kv ? head_kv : 65536;
    m->kv_fill = 0;
    m->kv_last = -1;

    m->st = q4_store_open(&m->g);
    if (!m->st) {
        fprintf(stderr, "q4 mtp: head dense store failed\n");
        q4_mtp_close(m);
        return NULL;
    }
    if (!q4_store_upload_hip(m->st)) {
        fprintf(stderr, "q4 mtp: head VRAM upload failed\n");
        q4_mtp_close(m);
        return NULL;
    }
    q4_store_free_host_except(m->st, &m->g);

    /* Head experts default to page cache, not resident DRAM: pinning the
     * head's 2.5 GiB on top of the 45 GiB main image is what pushed the box
     * into the 2026-09-29 global OOM (pinned pages resist kernel reaping).
     * Q4_MTP_RESIDENT=1 restores the old behaviour when RAM allows.
     * A loaded Q2 pack replaces that L1, so the cache keeps one unused
     * slot and no image. */
    int hres = 0;
    {
        const char *mr = getenv("Q4_MTP_RESIDENT");
        if (mr && mr[0] && mr[0] != '0') hres = 1;
    }
    int packed = mtp_pack_load(m);
    uint32_t slots = n_l1 ? n_l1 : 64;
    if (packed) slots = 1;
    m->ex = q4_expert_cache_open_arena_r(&m->g, slots, l2_bytes, 1, hres);
    if (!m->ex) {
        fprintf(stderr, "q4 mtp: head expert cache failed\n");
        q4_mtp_close(m);
        return NULL;
    }

    /* Q4_MTP_ROWS=8 (or unset, and the head is not pinned) is the current
     * 8-row path. A pin request sizes the scratch for MTP_ROWS_CAP; the
     * first forward drops back to 8 if the image did not actually fill.
     * A loaded Q2 pack puts all 512 draft experts in VRAM, which is the
     * same "fully resident" premise mtp_rows() checks — the prompt-length
     * head pass then runs 64 rows at a time instead of 8. */
    {
        uint32_t want = MTP_ROWS_LEGACY;
        const char *re = getenv("Q4_MTP_ROWS");
        if (re && re[0]) {
            int v = atoi(re);
            if (v < 1) v = 1;
            if (v > MTP_ROWS_CAP) v = MTP_ROWS_CAP;
            want = (uint32_t)v;
        } else if (hres || packed) {
            want = MTP_ROWS_CAP;
        }
        m->max_rows = want;
        m->rows_alloc = want;
        mtp_layout(m, want);
    }
    if (packed) {
        /* d_tok[i] = i / q2_k: the token row each (expert, row) pair of a
         * routed-MoE batch belongs to. Same table for every batch. */
        const size_t np = (size_t)m->rows_alloc * m->q2_k;
        int32_t *h = malloc(np * sizeof(int32_t));
        if (!h) {
            fprintf(stderr, "q4 mtp: pair table alloc failed\n");
            q4_mtp_close(m);
            return NULL;
        }
        for (uint32_t t = 0; t < m->rows_alloc; t++)
            for (uint32_t j = 0; j < m->q2_k; j++)
                h[(size_t)t * m->q2_k + j] = (int32_t)t;
        m->d_tok = q4_hip_malloc(np * sizeof(int32_t));
        if (!m->d_tok || !q4_hip_h2d(m->d_tok, h, np * sizeof(int32_t))) {
            free(h);
            fprintf(stderr, "q4 mtp: pair table upload failed\n");
            q4_mtp_close(m);
            return NULL;
        }
        free(h);
    }
    size_t f = sizeof(float);
    m->d_ws = q4_hip_malloc((size_t)m->ws_n * f);
    m->d_pws = q4_hip_malloc(MTP_PWS_N * f);
    m->d_resb = q4_hip_malloc((size_t)m->rows_alloc * m->hc_dim * f);
    m->d_res = q4_hip_malloc(m->hc_dim * f);
    m->d_emb = q4_hip_malloc((size_t)m->rows_alloc * m->n_embd * f);
    m->d_cat = q4_hip_malloc((size_t)m->rows_alloc * m->hc * 2 * m->n_embd * f);
    /* d_logits / h_logits (0.95 MiB each at a 248320 vocab) are allocated
     * lazily by logits_bufs_ensure: the head top-k path never needs them. */
    m->h_emb = malloc((size_t)m->rows_alloc * m->n_embd * f);
    m->d_kv_k = q4_hip_malloc((size_t)m->kv_cap * m->n_kvh * m->hd * f);
    m->d_kv_v = q4_hip_malloc((size_t)m->kv_cap * m->n_kvh * m->hd * f);
    if (!m->d_ws || !m->d_pws || !m->d_resb || !m->d_res || !m->d_emb ||
        !m->d_cat || !m->h_emb || !m->d_kv_k || !m->d_kv_v) {
        fprintf(stderr, "q4 mtp: device buffers failed\n");
        q4_mtp_close(m);
        return NULL;
    }
    q4_hip_fill(m->d_res, 0, m->hc_dim);
    dhead_load(m);
    return m;
}

void q4_mtp_close(q4_mtp *m) {
    if (!m) return;
    if (m->g_pre) q4_hip_graph_free(m->g_pre);
    if (m->g_suf) q4_hip_graph_free(m->g_suf);
    if (m->g_step) q4_hip_graph_free(m->g_step);
    m->g_pre = m->g_suf = m->g_step = NULL;
    q4_hip_free(m->q2);
    q4_hip_free(m->d_eids);
    q4_hip_free(m->d_ewts);
    q4_hip_free(m->d_tok);
    q4_hip_free(m->d_ws);
    q4_hip_free(m->d_pws);
    q4_hip_free(m->d_resb);
    q4_hip_free(m->d_res);
    q4_hip_free(m->d_emb);
    q4_hip_free(m->d_cat);
    q4_hip_free(m->d_logits);
    q4_hip_free(m->d_dhead);
    q4_hip_free(m->d_dvocab);
    q4_hip_free(m->d_kv_k);
    q4_hip_free(m->d_kv_v);
    free(m->h_logits);
    free(m->h_emb);
    q4_expert_cache_close(m->ex);
    q4_store_close(m->st);
    q4_gguf_close(&m->g);
    free(m);
}

/* VRAM beyond the head's own L1 slots. Includes the session-side verify
 * shadows (GDN snapshot ~118 MiB), which q4_sess_mtp_prep allocates. */
uint64_t q4_mtp_fixed_vram(const q4_mtp *m) {
    if (!m) return 0;
    uint64_t b = m->g.bytes_dense + m->g.bytes_shared;
    b += 2ull * m->kv_cap * m->n_kvh * m->hd * sizeof(float);
    b += (uint64_t)m->ws_n * sizeof(float);
    b += (uint64_t)MTP_PWS_N * sizeof(float);
    b += (uint64_t)m->rows_alloc * m->hc_dim * sizeof(float);
    b += (uint64_t)m->rows_alloc * m->n_embd * sizeof(float);
    b += (uint64_t)m->rows_alloc * m->hc * 2ull * m->n_embd * sizeof(float);
    b += (uint64_t)(m->hc_dim + m->n_vocab) * sizeof(float);
    if (m->d_dhead) {
        const q4_tensor *out = q4_find_tensor(m->gm, "output.weight");
        if (out)
            b += (uint64_t)m->n_dvocab *
                 (q4_row_bytes(out->ggml_type, out->ne[0]) + sizeof(int32_t));
    }
    /* session shadows: GDN S + conv + PLE snaps + replay staging */
    b += q4_gdn_state_bytes(m->gm) + 12ull * Q4_GIB / 1024;
    return b;
}

uint64_t q4_mtp_slot_bytes(const q4_mtp *m) {
    return m && m->ex ? q4_expert_cache_slot_bytes(m->ex) : 0;
}

q4_expert_cache *q4_mtp_cache(q4_mtp *m) {
    return m ? m->ex : NULL;
}

int q4_mtp_experts_resident(const q4_mtp *m) {
    return m && m->q2 != NULL;
}

void q4_mtp_stats(const q4_mtp *m, uint64_t *proposed, uint64_t *accepted,
                  uint64_t *cycles, uint64_t *gated) {
    if (proposed) *proposed = m ? m->proposed : 0;
    if (accepted) *accepted = m ? m->accepted : 0;
    if (cycles) *cycles = m ? m->cycles : 0;
    if (gated) *gated = m ? m->gated : 0;
}

/* ---- prefill + cycle ---- */

/* Build head KV for a chunk that q4_sess_prefill just processed. Rows come
 * from the session's prefill buffer; positions are the chunk's real ones.
 * Head cell j pairs the residual at position j with the token at j+1
 * (prefill(R[i], nxt=cur[i+1], cell=i)), so each row consumes the
 * FOLLOWING token; next_tok is the token at position base+n (the next
 * chunk's first token), or -1 when this chunk ends the prompt — the final
 * cell is then deferred to the first draft step of the first cycle. */
/* Head cells for prompt rows 0..wr-1 of a chunk at positions base..: row i
 * pairs rows[i] with toks[i+1], and row n-1 (when wr == n) with next_tok.
 * kvonly picks the K/V-only pass, else the full head forward. */
static bool prefill_rows(q4_mtp *m, const float *rows, uint32_t base,
                         const int32_t *toks, int n, int wr, int32_t next_tok,
                         int kvonly) {
    const uint32_t hc_dim = m->hc_dim;
    const int step = (int)(kvonly ? m->rows_alloc : mtp_rows(m));
    for (int i = 0; i < wr; i += step) {
        int c = wr - i < step ? wr - i : step;
        int tail = i + c > n - 1; /* this sub-batch holds row n-1 */
        int body = tail ? c - 1 : c;
        if (body > 0 &&
            !(kvonly ? fwd_kv(m, (int32_t)(base + (uint32_t)i), toks + i + 1,
                              rows + (size_t)i * hc_dim, (uint32_t)body)
                     : fwd_m(m, (int32_t)(base + (uint32_t)i), toks + i + 1,
                             rows + (size_t)i * hc_dim, (uint32_t)body, false)))
            return false;
        if (tail &&
            !(kvonly ? fwd_kv(m, (int32_t)(base + (uint32_t)n - 1), &next_tok,
                              rows + (size_t)(n - 1) * hc_dim, 1)
                     : fwd_m(m, (int32_t)(base + (uint32_t)n - 1), &next_tok,
                             rows + (size_t)(n - 1) * hc_dim, 1, false)))
            return false;
    }
    return true;
}

/* Q4_MTP_KVCHECK=1: write a chunk's cells with the K/V-only pass, then again
 * with the full forward over the same rows, and report how far they differ.
 * In-process, because the main model's prefill is not bit-reproducible run
 * to run (expert atomics). Leaves the full pass's cells in place. */
static bool prefill_kvcheck(q4_mtp *m, const float *rows, uint32_t base,
                            const int32_t *toks, int n, int wr,
                            int32_t next_tok) {
    const uint32_t f0 = m->kv_fill;
    const size_t kvw = (size_t)m->n_kvh * m->hd, nf = (size_t)wr * kvw;
    float *a = malloc(nf * 4 * sizeof(float));
    if (!a) return false;
    float *b = a + 2 * nf;
    bool ok = true;
    for (int pass = 0; ok && pass < 2; pass++) {
        float *dst = pass ? b : a;
        m->kv_fill = f0;
        ok = prefill_rows(m, rows, base, toks, n, wr, next_tok, !pass) &&
             q4_hip_d2h(dst, m->d_kv_k + (size_t)f0 * kvw, nf * sizeof(float)) &&
             q4_hip_d2h(dst + nf, m->d_kv_v + (size_t)f0 * kvw, nf * sizeof(float));
    }
    if (ok) {
        double md[2] = { 0, 0 }, mx[2] = { 0, 0 };
        for (size_t i = 0; i < 2 * nf; i++) {
            int w = i >= nf;
            double d = fabs((double)a[i] - (double)b[i]);
            if (d > md[w]) md[w] = d;
            if (fabs((double)b[i]) > mx[w]) mx[w] = fabs((double)b[i]);
        }
        fprintf(stderr,
                "q4 mtp kvcheck: cells %u..%u  k max|d| %.3g (max|k| %.3g)  "
                "v max|d| %.3g (max|v| %.3g)\n",
                f0, f0 + (uint32_t)wr - 1, md[0], mx[0], md[1], mx[1]);
    }
    free(a);
    return ok;
}

/* Build head KV for a chunk that q4_sess_prefill just processed. Rows come
 * from the session's prefill buffer; positions are the chunk's real ones.
 * Head cell j pairs the residual at position j with the token at j+1
 * (prefill(R[i], nxt=cur[i+1], cell=i)), so each row consumes the
 * FOLLOWING token; next_tok is the token at position base+n (the next
 * chunk's first token), or -1 when this chunk ends the prompt — the final
 * cell is then deferred to the first draft step of the first cycle. */
bool q4_mtp_prefill_chunk(q4_mtp *m, q4_sess *s, const int32_t *toks, int n,
                          int32_t next_tok) {
    if (!m || !s || !toks || n <= 0) return false;
    uint32_t n_kv = q4_sess_n_kv(s);
    if (n_kv < (uint32_t)n) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: prefill chunk n_kv=%u < n=%d\n", n_kv, n);
        return false;
    }
    const float *rows = q4_sess_pref_row(s, 0);
    if (!rows) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: no pref rows (d_pref NULL?)\n");
        return false; /* sequential-decode fallback: skip head KV */
    }
    /* Cells written here: positions base..base+wr-1; wr = n when next_tok
     * exists (row n-1 pairs with it), else n-1 (the last cell is written
     * by the first draft step once the first generated token is known). */
    int wr = next_tok >= 0 ? n : n - 1;
    if (wr > 0 && !kv_make_room(m, (uint32_t)wr)) return false;
    uint32_t base = n_kv - (uint32_t)n; /* first chunk position */
    static int kvcheck = -1;
    if (kvcheck < 0) {
        const char *e = getenv("Q4_MTP_KVCHECK");
        kvcheck = e && e[0] == '1';
    }
    const bool ok =
        kvcheck && wr > 0
            ? prefill_kvcheck(m, rows, base, toks, n, wr, next_tok)
            : prefill_rows(m, rows, base, toks, n, wr, next_tok, mtp_kvonly());
    /* Contiguous with the cells before only when those end at base-1
     * (a fresh head, or the previous chunk's / a restore bridge's). */
    if (ok && wr > 0)
        m->kv_last = (m->kv_fill == (uint32_t)wr ||
                      m->kv_last == (int64_t)base - 1)
                         ? (int64_t)base + wr - 1
                         : -1;
    else if (!ok)
        m->kv_last = -1;
    return ok;
}

/* Seed the chain when the prompt came entirely from the session cache (no
 * prefill ran): prime d_res with the restored residual at position n_kv-1.
 * No head-KV cell is written here — cell n_kv-1 pairs that residual with
 * the first GENERATED token, which only exists inside the first cycle;
 * draft step 0 writes it. The head KV simply starts empty (attention is
 * bounded by kv_fill, not position). */
bool q4_mtp_seed(q4_mtp *m, q4_sess *s, int32_t tok) {
    (void)tok;
    if (!m || !s) return false;
    if (q4_sess_n_kv(s) == 0) return false;
    const float *res = q4_sess_res_dev(s);
    if (!res) return false;
    return q4_hip_copy(m->d_res, res, m->hc_dim);
}

/* Head KV beside a session snapshot (the engine's prefix cache).  A
 * restored session used to start the head from an empty KV, so after a
 * cache hit the drafts attended only to the tokens prefilled in this
 * request — every agent turn but the first.  The file keeps cells
 * [0, kv_fill) (rope already applied; positions kv_last - kv_fill + 1 ..
 * kv_last), tied to the snapshot by its token count and an id hash. */
#define MTP_KV_MAGIC 0x484b4d51u /* "QMKH" */
typedef struct {
    uint32_t magic, version, kvw, n_kv;
    uint32_t kv_fill, pad;
    int64_t kv_last;
    uint64_t ids_hash;
} mtp_kv_hdr;

static bool mtp_kv_io(FILE *f, float *d, uint64_t nfloat, int load) {
    const uint64_t stage_n = 4u << 20;
    float *stage = malloc((size_t)(nfloat < stage_n ? nfloat : stage_n) * 4);
    bool ok = stage != NULL;
    for (uint64_t o = 0; ok && o < nfloat; o += stage_n) {
        const uint64_t c = nfloat - o < stage_n ? nfloat - o : stage_n;
        ok = load ? fread(stage, 4, (size_t)c, f) == c &&
                        q4_hip_h2d(d + o, stage, (size_t)c * 4)
                  : q4_hip_d2h(stage, d + o, (size_t)c * 4) &&
                        fwrite(stage, 4, (size_t)c, f) == c;
    }
    free(stage);
    return ok;
}

bool q4_mtp_kv_save(q4_mtp *m, const char *path, uint32_t n_kv,
                    uint64_t ids_hash) {
    if (!path || !path[0]) return false;
    /* Only real cells that end where the snapshot's next token pairs:
     * position n_kv - 2 (n_kv - 1 needs that token) or n_kv - 1. */
    if (!m || !m->kv_fill || m->kv_last < 0 ||
        (m->kv_last + 2 != (int64_t)n_kv && m->kv_last + 1 != (int64_t)n_kv)) {
        (void)remove(path);
        return false;
    }
    const uint64_t kvw = (uint64_t)m->n_kvh * m->hd;
    mtp_kv_hdr h = {MTP_KV_MAGIC, 1, (uint32_t)kvw, n_kv, m->kv_fill, 0,
                    m->kv_last, ids_hash};
    char tmp[4096 + 16];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1 &&
              mtp_kv_io(f, m->d_kv_k, kvw * m->kv_fill, 0) &&
              mtp_kv_io(f, m->d_kv_v, kvw * m->kv_fill, 0);
    ok = (fclose(f) == 0) && ok;
    if (ok) ok = rename(tmp, path) == 0;
    if (!ok) (void)remove(tmp);
    return ok;
}

int q4_mtp_kv_load(q4_mtp *m, const char *path, uint32_t n_kv,
                   uint64_t ids_hash) {
    if (!m || !path || !path[0]) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    const uint64_t kvw = (uint64_t)m->n_kvh * m->hd;
    mtp_kv_hdr h;
    bool ok = fread(&h, sizeof(h), 1, f) == 1 && h.magic == MTP_KV_MAGIC &&
              h.version == 1 && h.kvw == kvw && h.n_kv == n_kv &&
              h.ids_hash == ids_hash && h.kv_fill > 0 &&
              h.kv_fill <= m->kv_cap &&
              (h.kv_last + 2 == (int64_t)n_kv || h.kv_last + 1 == (int64_t)n_kv);
    if (ok)
        ok = mtp_kv_io(f, m->d_kv_k, kvw * h.kv_fill, 1) &&
             mtp_kv_io(f, m->d_kv_v, kvw * h.kv_fill, 1);
    fclose(f);
    if (!ok) {
        m->kv_fill = 0;
        m->kv_last = -1;
        return 0;
    }
    m->kv_fill = h.kv_fill;
    m->kv_last = h.kv_last;
    return 1;
}

/* After a restore, before the rest of the prompt is prefilled: the cell
 * at position n_kv - 1 pairs the restored residual (the session's d_res,
 * overwritten by the next prefill) with the first new token.  The chunk
 * pass writes from position n_kv on. */
bool q4_mtp_kv_bridge(q4_mtp *m, q4_sess *s, int32_t next_tok) {
    if (!m || !s || next_tok < 0) return false;
    const uint32_t n_kv = q4_sess_n_kv(s);
    if (!n_kv || m->kv_last != (int64_t)n_kv - 2) return true;
    const float *res = q4_sess_res_dev(s);
    if (!res || !kv_make_room(m, 1) ||
        !fwd_kv(m, (int32_t)n_kv - 1, &next_tok, res, 1)) {
        m->kv_last = -1;
        return false;
    }
    m->kv_last = (int64_t)n_kv - 1;
    return true;
}

static int32_t argmax(const float *l, uint32_t n) {
    int32_t b = 0;
    float v = l[0];
    for (uint32_t i = 1; i < n; i++)
        if (l[i] > v) {
            v = l[i];
            b = (int32_t)i;
        }
    return b;
}

/* Host fallback for the probability of the max under softmax. */
static float softpmax(const float *l, uint32_t n) {
    float mx = l[0];
    for (uint32_t i = 1; i < n; i++)
        if (l[i] > mx) mx = l[i];
    double s = 0;
    for (uint32_t i = 0; i < n; i++) s += exp((double)(l[i] - mx));
    return (float)(1.0 / s);
}

static float mtp_pmin(void) {
    static int init;
    static float v;
    if (!init) {
        const char *e = getenv("Q4_MTP_PMIN");
        v = e ? (float)atof(e) : 0.5f;
        init = 1;
    }
    return v;
}

/* The greedy draft gate compares the softmax p of the draft with pmin,
 * like the top-k path and Strata's --spec-min-p. Before 2026-10-11 the
 * greedy slab path compared the raw logit, so it never gated (measured:
 * code 42.7 -> 43.1 t/s, ja/agent within noise). Q4_MTP_GATE_P=0 restores
 * the raw-logit compare; the old slab top-k (Q4_TOPK_WIDE=0) has no p. */
static int mtp_gate_p(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MTP_GATE_P");
        const char *w = getenv("Q4_TOPK_WIDE");
        v = !(e && e[0] == '0') && !(w && w[0] == '0');
    }
    return v;
}

static double mtp_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int mtp_prof(void) {
    static int v = -1;
    if (v < 0) {
        const char *p = getenv("Q4_PROFILE");
        const char *m = getenv("Q4_MTP_PROF");
        v = (p && p[0] && p[0] != '0') || (m && m[0] && m[0] != '0');
    }
    return v;
}

static double t_draft, t_verify, t_logits, t_commit, t_repair;
static double t_cpx_us;              /* cpx_wait us inside the verify pass */
static unsigned long long t_cpx_n;
static uint64_t t_cycles;
#define MTP_PH_NONE 0
#define MTP_PH_DRAFT 1
#define MTP_PH_VERIFY 2
#define MTP_PH_LOGITS 3
#define MTP_PH_COMMIT 4
#define MTP_PH_REPAIR 5
#define MTP_PT(acc, ph)                                                    \
    do {                                                                   \
        if (prof) {                                                        \
            q4_hip_sync();                                                 \
            acc += mtp_now() - tt;                                         \
            if ((ph) == MTP_PH_VERIFY) {                                   \
                unsigned long long cpn = 0;                                \
                double wa = q4_hip_cpx_wait_prof(&cpn);                    \
                if (cpn) {                                                 \
                    t_cpx_us += wa * (double)cpn;                          \
                    t_cpx_n += cpn;                                        \
                }                                                          \
            }                                                              \
        }                                                                  \
        if (prof && (ph) == MTP_PH_LOGITS)                                 \
            q4_moe_vtl_collect((int)m->gm->n_layer);                       \
        q4_hip_gemv_prof_phase(ph);                                        \
        tt = mtp_now();                                                    \
    } while (0)

/* Q4_MTP_DTSTAT: print and clear (request end and session reset). */
static void mtp_dtstat_print(q4_mtp *m) {
    if (!m->dt_n) return;
    fprintf(stderr, "q4 mtp dtstat: %llu drafts  E[accept] top_p=req:"
                    " dt0.7 %.3f dt0.85 %.3f dt1 %.3f dt1.15 %.3f | "
                    "top_p=1: %.3f %.3f %.3f %.3f | argmax p(mode) %.3f\n",
            (unsigned long long)m->dt_n, m->dt_ov[0] / m->dt_n,
            m->dt_ov[1] / m->dt_n, m->dt_ov[2] / m->dt_n,
            m->dt_ov[3] / m->dt_n, m->dt_ov[4] / m->dt_n,
            m->dt_ov[5] / m->dt_n, m->dt_ov[6] / m->dt_n,
            m->dt_ov[7] / m->dt_n, m->dt_ov[8] / m->dt_n);
    memset(m->dt_ov, 0, sizeof m->dt_ov);
    m->dt_n = 0;
}

void q4_mtp_prof_print(q4_mtp *m, FILE *f) {
    if (m) mtp_dtstat_print(m);
    (void)m;
    if (!f || !t_cycles) return;
    double c = (double)t_cycles;
    fprintf(f, "q4 mtp prof: cycles %llu\n"
               "  draft   %.3f s  %.2f ms/cyc\n"
               "  verify  %.3f s  %.2f ms/cyc\n"
               "  logits  %.3f s  %.2f ms/cyc\n"
               "  commit  %.3f s  %.2f ms/cyc\n"
               "  repair  %.3f s  %.2f ms/cyc\n",
            (unsigned long long)t_cycles,
            t_draft, t_draft * 1e3 / c,
            t_verify, t_verify * 1e3 / c,
            t_logits, t_logits * 1e3 / c,
            t_commit, t_commit * 1e3 / c,
            t_repair, t_repair * 1e3 / c);
    if (t_cpx_n)
        fprintf(f, "  cpxwait %.1f ms/cyc  (%.0f us x%llu)\n",
                t_cpx_us / 1e3 / c, t_cpx_us / (double)t_cpx_n,
                (unsigned long long)t_cpx_n);
    q4_moe_vtl_print(f);
    t_draft = t_verify = t_logits = t_commit = t_repair = 0;
    t_cpx_us = 0;
    t_cpx_n = 0;
    t_cycles = 0;
}

/* n=1 attention that reads rope position and the KV slot from the device
 * counters. Does not advance kv_fill: the caller counts a successful
 * launch once. Capture records this; it does not run it. */
static bool draft_attn_d(q4_mtp *m, const float *d_mix, float *d_y) {
    const q4_gguf *g = &m->g;
    const uint32_t hd = m->hd;
    const uint32_t n = 1;
    float *qg = m->d_ws + m->off_qg;
    float *k = qg + (size_t)n * m->n_head * hd * 2;
    float *v = k + (size_t)n * m->n_kvh * hd;
    float *qbat = v + (size_t)n * m->n_kvh * hd;
    float *gbat = qbat + (size_t)n * m->n_head * hd;
    float *attn = qg;
    if ((size_t)(gbat - qg) + (size_t)n * m->n_head * hd >
        m->off_mixed - m->off_qg)
        return false;
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_q.weight", m->layer), d_mix, qg, n) ||
        !mmd_n(m->st, mt(g, "blk.%d.attn_k.weight", m->layer), d_mix, k, n) ||
        !mmd_n(m->st, mt(g, "blk.%d.attn_v.weight", m->layer), d_mix, v, n))
        return false;
    if (!q4_hip_qsa_q_prep_d(
            qg, fd(m->st, mt(g, "blk.%d.attn_q_norm.weight", m->layer)), qbat,
            gbat, m->n_head, hd, m->n_rot, m->rope_base, m->eps) ||
        !q4_hip_qsa_k_prep_d(
            k, fd(m->st, mt(g, "blk.%d.attn_k_norm.weight", m->layer)),
            m->n_kvh, hd, m->n_rot, m->rope_base, m->eps))
        return false;
    uint32_t kvw = m->n_kvh * hd;
    if (!q4_hip_kv_append_kv(m->d_kv_k, m->d_kv_v, k, v, kvw, kvw * 4u, 0))
        return false;
    float scale = 1.f / sqrtf((float)hd);
    if (!q4_hip_qsa_decode_kv(qbat, m->d_kv_k, m->d_kv_v, gbat, attn, m->n_head,
                              m->n_kvh, hd, m->kv_cap, scale, 0))
        return false;
    return mmd_n(m->st, mt(g, "blk.%d.attn_output.weight", m->layer), attn, d_y,
                 n);
}

/* Dense prefix of one draft step, through the ffn mix. The token row is
 * already in d_emb. MoE and the sample are not in this graph. */
static bool draft_prefix(q4_mtp *m) {
    const q4_tensor *enorm = mt(&m->g, "blk.%d.nextn.enorm.weight", m->layer);
    const q4_tensor *hnorm = mt(&m->g, "blk.%d.nextn.hnorm.weight", m->layer);
    const q4_tensor *proj = mt(&m->g, "blk.%d.nextn.eh_proj.weight", m->layer);
    if (!enorm || !hnorm || !proj) return false;
    float *e4 = m->d_ws + m->off_xn;
    float *en = e4 + m->hc_dim;
    float *h4 = en + m->n_embd;
    if ((size_t)(h4 - m->d_ws) + m->hc_dim > m->off_qg) return false;
    if (!q4_hip_grouped_rms(m->d_emb, fd(m->st, enorm), en, m->n_embd, 1,
                            m->eps) ||
        !q4_hip_repeat_hc(en, e4, m->n_embd, m->hc) ||
        !q4_hip_grouped_rms(m->d_res, fd(m->st, hnorm), h4, m->n_embd, m->hc,
                            m->eps) ||
        !q4_hip_eh_cat(m->d_cat, e4, h4, m->n_embd, m->hc) ||
        !mmd_n(m->st, proj, m->d_cat, m->d_resb, m->hc))
        return false;
    float *d_mixed = m->d_ws + m->off_mixed;
    float *d_block = m->d_ws + m->off_block;
    float *d_inj = m->d_ws + m->off_inj;
    if (!hc_mix_mn(m, m->d_resb, d_mixed, d_inj, true, 1) ||
        !draft_attn_d(m, d_mixed, d_block) ||
        !q4_hip_hc_combine_n(m->d_resb, d_block, d_inj, m->n_embd, m->hc, 1) ||
        !hc_mix_mn(m, m->d_resb, d_mixed, d_inj, false, 1))
        return false;
    return true;
}

/* Combine the MoE output, publish the next residual, then the device sample.
 * The sample buffer addresses are stable, so the launch stays in the graph. */
static bool draft_suffix(q4_mtp *m, int greedy, uint32_t k) {
    float *d_block = m->d_ws + m->off_block;
    float *d_inj = m->d_ws + m->off_inj;
    float *xn = m->d_ws + m->off_xn;
    float *d_mixed = xn + 3u * m->hc_dim + m->hc_rank;
    if (!q4_hip_hc_combine_n(m->d_resb, d_block, d_inj, m->n_embd, m->hc, 1) ||
        !q4_hip_copy_async(m->d_res, m->d_resb, m->hc_dim) ||
        !logits_mix_m(m, d_mixed))
        return false;
    /* Slab top-k: the full-vocab logits vector is never written. Inside a
     * capture no allocation is legal, so the full-GEMV fallback only runs
     * when its destination was warmed before the capture; otherwise the
     * capture fails and the caller drops to eager (which can allocate). */
    if (m->slab) return logits_topk_m(m, d_mixed, greedy ? 1u : k);
    if (!m->d_logits) return false;
    if (!logits_m(m)) return false;
    return greedy ? q4_hip_argmax_launch(m->d_logits, m->n_vocab)
                  : q4_hip_topk_launch(m->d_logits, m->n_vocab, k);
}

static int mtp_graph_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MTP_GRAPH");
        v = !(e && e[0] == '0');
    }
    return v;
}

static void draft_graphs_drop(q4_mtp *m) {
    if (m->g_pre) q4_hip_graph_free(m->g_pre);
    if (m->g_suf) q4_hip_graph_free(m->g_suf);
    if (m->g_step) q4_hip_graph_free(m->g_step);
    m->g_pre = m->g_suf = m->g_step = NULL;
}

static int draft_graph_fail(q4_mtp *m, void *a, void *b) {
    if (a) q4_hip_graph_free(a);
    if (b) q4_hip_graph_free(b);
    m->g_pre = m->g_suf = NULL;
    m->graph_off = 1;
    q4_hip_clear();
    fprintf(stderr, "q4 mtp: draft graph capture failed; eager\n");
    return 0;
}

/* Capture both graphs before any real step. A later change of greedy vs
 * top-k, or of k, recaptures: those arguments are baked into the sample. */
static int draft_graphs_ensure(q4_mtp *m, int greedy, uint32_t k) {
    if (m->graph_off || !mtp_graph_on()) return 0;
    if (!q4_hip_kv_dev_ok() ||
        !q4_hip_qsa_dec_d_ok(m->n_head, m->n_kvh, m->hd))
        return 0;
    (void)mtp_rows(m);
    int mode = greedy ? 0 : 1;
    if (m->q2) {
        if (m->g_step && m->graph_mode == mode && (greedy || m->graph_k == k))
            return 1;
    } else if (m->g_pre && m->g_suf && m->graph_mode == mode &&
               (greedy || m->graph_k == k)) {
        return 1;
    }
    draft_graphs_drop(m);
    /* First pass allocates the mapped sample buffers and the slab top-k
     * buffers. hipMalloc inside a capture is illegal, so both paths are
     * warmed here. The slab path is preferred; the full-vocab GEMV is the
     * fallback (and then needs its lazy logits buffers). */
    q4_hip_head_topk_warm();
    int warm = 0;
    {
        float *xn = m->d_ws + m->off_xn;
        float *dm = xn + 3u * m->hc_dim + m->hc_rank;
        if ((size_t)(dm - m->d_ws) + m->n_embd <= m->off_qg)
            warm = logits_topk_m(m, dm, greedy ? 1u : k);
    }
    if (!warm) {
        if (!logits_bufs_ensure(m)) return draft_graph_fail(m, NULL, NULL);
        if (!(greedy ? q4_hip_argmax_launch(m->d_logits, m->n_vocab)
                     : q4_hip_topk_launch(m->d_logits, m->n_vocab, k)))
            return draft_graph_fail(m, NULL, NULL);
    }
    if (m->q2) {
        int32_t zid[64];
        float zw[64];
        memset(zid, 0, sizeof zid);
        memset(zw, 0, sizeof zw);
        const uint32_t n_ff = m->g.n_ff_exp ? m->g.n_ff_exp : Q4_Q2_PACK_FF;
        float *hh = m->d_pws;
        float *hh2 = hh + (size_t)m->q2_k * n_ff;
        float *dn = hh2 + (size_t)m->q2_k * n_ff;
        float *y = m->d_ws + m->off_block;
        if (!warm || !q4_hip_h2d(m->d_eids, zid, sizeof zid) ||
            !q4_hip_h2d(m->d_ewts, zw, sizeof zw) ||
            !q4_hip_moe_q2(m->q2, Q4_Q2_PACK_BLOB, m->d_eids, m->d_ewts,
                           m->q2_k, 0, 460800, 921600,
                           m->d_ws + m->off_mixed, hh, hh2, dn, y, n_ff,
                           m->n_embd, m->n_embd) ||
            !q4_hip_stream_sync())
            return draft_graph_fail(m, NULL, NULL);
        q4_hip_clear();
        if (!q4_hip_graph_begin()) return draft_graph_fail(m, NULL, NULL);
        int ok = draft_prefix(m) &&
                 mtp_moe_q2(m, m->d_ws + m->off_mixed, m->d_ws + m->off_block,
                            1) &&
                 draft_suffix(m, greedy, k);
        void *step = q4_hip_graph_end();
        if (!ok || !step) return draft_graph_fail(m, step, NULL);
        m->g_step = step;
        m->graph_mode = mode;
        m->graph_k = k;
        static int announced = 0;
        if (!announced) {
            announced = 1;
            fprintf(stderr, "q4 mtp: draft graph live\n");
        }
        return 1;
    }
    if (!warm || !q4_hip_stream_sync()) return draft_graph_fail(m, NULL, NULL);
    q4_hip_clear();
    if (!q4_hip_graph_begin()) return draft_graph_fail(m, NULL, NULL);
    int ok = draft_prefix(m);
    void *pre = q4_hip_graph_end();
    if (!ok || !pre) return draft_graph_fail(m, pre, NULL);
    if (!q4_hip_graph_begin()) return draft_graph_fail(m, pre, NULL);
    ok = draft_suffix(m, greedy, k);
    void *suf = q4_hip_graph_end();
    if (!ok || !suf) return draft_graph_fail(m, pre, suf);
    m->g_pre = pre;
    m->g_suf = suf;
    m->graph_mode = mode;
    m->graph_k = k;
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "q4 mtp: draft graph live\n");
    }
    return 1;
}

/* Q4_MTP_VTAIL=0: the verify tail runs q4_sess_logits_at per row (head
 * graph, 0.95 MiB D2H, host scan) instead of one batched head pass. */
static int mtp_vtail(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MTP_VTAIL");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* Target distribution of verify row `row`: argmax into *best (greedy) or
 * the top-k distribution into tid, tprob, *tk. From the batched tail when
 * vtail, else (or when the row's top value is not finite — the per-row
 * path fills the host logits the engine's NaN check reads) per row. */
/* The sampled-draft gate must not look at the drawn draft.  Stopping the
 * chain when q(draft) < pmin proposes a draft from q restricted to
 * q >= pmin while the accept test divides by the unrestricted q, and the
 * cut position is then sampled from p: the emitted law is
 * sum_{d pass} q(d) [spec outcome] + P(fail) p, not p.  With pmin 0.5
 * only the mode can pass, so every draft position i > 0 where the head is
 * confident (q_max >= 0.5) emitted the mode with p_m (2 - q_m) instead of
 * p_m (p = q = 0.6 / 0.4: 0.84 / 0.16).  The gate now compares q_max — a
 * function of the head's distribution, not of the draw — so a proposed
 * draft is a plain sample from q and each position emits p exactly.
 * Q4_MTP_GATE_DRAWN=1 restores the biased gate (A/B only). */
static float mtp_gate_stat(const int32_t *ids, const float *probs, int k,
                           int32_t draft) {
    static int drawn = -1;
    if (drawn < 0) {
        const char *e = getenv("Q4_MTP_GATE_DRAWN");
        drawn = e && e[0] == '1';
    }
    if (drawn) return q4_dist_prob(ids, probs, k, draft);
    float mx = 0.f;
    for (int a = 0; a < k; a++)
        if (probs[a] > mx) mx = probs[a];
    return mx;
}

static int32_t g_row_top1; /* the last verify_row's raw argmax (Q4_TOP1_DUMP) */
static bool verify_row(q4_sess *s, int vtail, int row, uint32_t vk,
                       float temp, int top_k, float top_p, uint32_t n_vocab,
                       int32_t *best, int32_t *tid, float *tprob, int *tk) {
    if (vtail) {
        int32_t ids[Q4_DIST_MAX];
        float vals[Q4_DIST_MAX];
        q4_hip_topk_rows_read((uint32_t)row, ids, vals, vk);
        g_row_top1 = ids[0];
        if (ids[0] >= 0 && isfinite(vals[0])) {
            if (temp <= 1e-5f)
                *best = ids[0];
            else
                *tk = q4_dist_from_topk_k(ids, vals, (int)vk, top_k, temp,
                                          top_p, tid, tprob);
            return true;
        }
    }
    if (!q4_sess_logits_at(s, row)) return false;
    g_row_top1 = argmax(q4_sess_logits(s), n_vocab);
    if (temp <= 1e-5f)
        *best = argmax(q4_sess_logits(s), n_vocab);
    else
        *tk = q4_dist_build(q4_sess_logits(s), n_vocab, temp, top_k, top_p,
                            tid, tprob);
    return true;
}

int q4_mtp_cycle(q4_mtp *m, q4_sess *s, int32_t c0, int n_draft, float temp,
                 int top_k, float top_p, unsigned *rng, int32_t *out_drafts,
                 int *n_out, int32_t *next_c0, const float **next_logits) {
    *n_out = 0;
    if (next_logits) *next_logits = NULL;
    if (!m || !s || n_draft <= 0 || !out_drafts || !next_c0) return -1;
    if (q4_sess_n_kv(s) + (uint32_t)n_draft + 2 >= q4_sess_max_kv(s))
        return -1;
    uint32_t p = q4_sess_n_kv(s); /* c0 sits at position p */
    int nd = n_draft > MTP_MAX_DRAFT ? MTP_MAX_DRAFT : n_draft;
    /* The draft writes nd cells and the repair rewrites them in place:
     * fill0 below must stay valid for the whole cycle. */
    if (!kv_make_room(m, (uint32_t)nd + 2)) return -1;

    /* 1. Draft chain. Each step consumes (token, previous estimate) and
     * writes an estimate entry into the head KV (overwritten by repair).
     * Head cell p-1+i pairs the residual BEFORE the token with the token
     * (residual position convention). Step 0
     * must consume the REAL residual at p-1 — the row whose logits
     * produced c0 — not the head's accumulated estimate: reseed d_res
     * from the session residual every cycle. */
    const float *seed_res = q4_sess_res_dev(s);
    if (!seed_res || !q4_hip_copy(m->d_res, seed_res, m->hc_dim))
        return -1;
    int32_t draft[MTP_MAX_DRAFT];
    int32_t did[MTP_MAX_DRAFT][Q4_DIST_MAX];
    float dprob[MTP_MAX_DRAFT][Q4_DIST_MAX];
    int dk_n[MTP_MAX_DRAFT];
    uint32_t fill0 = m->kv_fill;
    /* The chain writes estimates past fill0 until the repair: the cells
     * are real again only on a clean return (kv_last set there). */
    const bool kv_contig = fill0 == 0 || m->kv_last == (int64_t)p - 2;
    m->kv_last = -1;
    int32_t h = c0;
    int prof = mtp_prof();
    const float pmin = mtp_pmin();
    /* Q4_MTP_PCHAIN: also stop once the product of the drafts' p (the
     * chance the whole chain is accepted) drops under it — each verify row
     * costs ~6.5 ms (~16 token-expert misses), so a row is worth adding
     * only while that product beats ~row cost x tokens/ms. */
    static float pchain = -1.f;
    if (pchain < 0.f) {
        const char *e = getenv("Q4_MTP_PCHAIN");
        pchain = e ? (float)atof(e) : 0.f;
    }
    float chain = 1.f;
    float glog_p[MTP_MAX_DRAFT] = {0};
    /* Device sample for greedy and for top-k in 1..128. top_k <= 0 is the
     * full-vocab host distribution. Capture is outside the timed loop. */
    int greedy = temp <= 1e-5f;
    /* Q4_MTP_DTSTAT=1 (diagnostic): per verified draft, the acceptance
     * probability sum_x min(p, q') the draft would have had with q' built
     * from the same head logits at other temperatures / top-p. */
    static int dtstat = -1;
    if (dtstat < 0) {
        const char *e = getenv("Q4_MTP_DTSTAT");
        dtstat = e && e[0] == '1';
    }
    int32_t dr_ids[MTP_MAX_DRAFT][Q4_DIST_MAX];
    float dr_vals[MTP_MAX_DRAFT][Q4_DIST_MAX];
    int dr_n[MTP_MAX_DRAFT] = {0};
    uint32_t sk = 0;
    int devsamp = greedy;
    if (!greedy && top_k > 0 && top_k <= Q4_DIST_MAX) {
        devsamp = 1;
        sk = (uint32_t)top_k;
    }
    int graphed = devsamp && draft_graphs_ensure(m, greedy, greedy ? 1u : sk);
    q4_hip_gemv_prof_phase(MTP_PH_DRAFT);
    double tt = prof ? mtp_now() : 0;
    for (int i = 0; i < nd; i++) {
        int32_t base = (int32_t)p + i - 1;
        float pc = 0.f;
        int host_row = 0;
        if (graphed) {
            /* One barrier per step: the confidence gate needs this token
             * before the next step can start. A failed launch is not
             * retried — the prefix may already have written a KV slot. */
            if (m->kv_fill + 1 > m->kv_cap) return -1;
            if (!emb_row(m, h, m->h_emb)) return -1;
            if (!q4_hip_h2d_async(m->d_emb, m->h_emb,
                                  (size_t)m->n_embd * sizeof(float)))
                return -1;
            q4_hip_pos_push(base);
            q4_hip_kv_push((int32_t)m->kv_fill);
            if (m->q2 && m->g_step) {
                if (!q4_hip_graph_launch(m->g_step)) {
                    fprintf(stderr, "q4 mtp: draft step launch failed\n");
                    return -1;
                }
                m->kv_fill += 1;
                if (!q4_hip_stream_sync()) return -1;
            } else {
                if (!q4_hip_graph_launch(m->g_pre)) {
                    fprintf(stderr, "q4 mtp: draft prefix launch failed\n");
                    return -1;
                }
                m->kv_fill += 1;
                if (!q4_moe_layer_dev_n(&m->g, m->st, m->ex, m->layer,
                                        m->d_ws + m->off_mixed,
                                        m->d_ws + m->off_block, m->d_pws, 1,
                                        MTP_PWS_N))
                    return -1;
                if (!q4_hip_graph_launch(m->g_suf)) {
                    fprintf(stderr, "q4 mtp: draft suffix launch failed\n");
                    return -1;
                }
                if (!q4_hip_stream_sync()) return -1;
            }
        } else if (devsamp) {
            /* Eager device sample: forward without logits, then the slab
             * top-k over the head's vocab (no full-vocab buffer). */
            if (!fwd_m(m, base, &h, m->d_res, 1, 0u)) return -1;
            float *xn = m->d_ws + m->off_xn;
            float *dm = xn + 3u * m->hc_dim + m->hc_rank;
            int launched = 0;
            if ((size_t)(dm - m->d_ws) + m->n_embd <= m->off_qg &&
                logits_mix_m(m, dm) && logits_topk_m(m, dm, greedy ? 1u : sk))
                launched = q4_hip_stream_sync();
            if (!launched) {
                /* Full-vocab fallback: needs the lazy logits buffers. */
                q4_hip_clear();
                if (!logits_bufs_ensure(m)) return -1;
                if (!logits_full_m(m, dm)) return -1;
                int ok = greedy
                             ? q4_hip_argmax_launch(m->d_logits, m->n_vocab)
                             : q4_hip_topk_launch(m->d_logits, m->n_vocab, sk);
                if (!ok || !q4_hip_stream_sync()) return -1;
                if (!q4_hip_d2h(m->h_logits, m->d_logits,
                                m->n_vocab * sizeof(float)))
                    return -1;
                host_row = 1;
            }
        } else {
            if (!fwd_m(m, base, &h, m->d_res, 1, 2u)) return -1;
            host_row = 1;
        }
        int32_t tids[Q4_DIST_MAX];
        float tvs[Q4_DIST_MAX];
        uint32_t rk = greedy ? 1u : sk;
        if (!host_row) {
            /* m->slab means the last head projection used the head top-k,
             * whose pairs land in the pinned sample buffers directly. A
             * greedy read of tvs[0] is a raw logit; mtp_gate_p swaps in
             * the softmax p below. */
            if (m->slab)
                q4_hip_topk_read(tids, tvs, rk);
            else if (greedy)
                q4_hip_argmax_read(&draft[i], &pc);
            else
                q4_hip_topk_read(tids, tvs, sk);
            if (greedy) {
                if (m->slab) {
                    draft[i] = tids[0];
                    pc = tvs[0];
                    /* The wide head top-k also leaves the softmax p of
                     * the max; the slab path only had the raw logit,
                     * which almost never fell under pmin. */
                    if (mtp_gate_p()) q4_hip_argmax_read(NULL, &pc);
                }
            } else {
                if (dtstat) {
                    memcpy(dr_ids[i], tids, sk * sizeof(int32_t));
                    memcpy(dr_vals[i], tvs, sk * sizeof(float));
                    dr_n[i] = (int)sk;
                }
                dk_n[i] = q4_dist_from_topk(tids, tvs, (int)sk, temp, top_p,
                                            did[i], dprob[i]);
                draft[i] = q4_dist_sample(did[i], dprob[i], dk_n[i], rng);
                pc = mtp_gate_stat(did[i], dprob[i], dk_n[i], draft[i]);
            }
        } else if (greedy) {
            draft[i] = argmax(m->h_logits, m->n_vocab);
            pc = softpmax(m->h_logits, m->n_vocab);
        } else {
            dk_n[i] = q4_dist_build(m->h_logits, m->n_vocab, temp, top_k,
                                    top_p, did[i], dprob[i]);
            draft[i] = q4_dist_sample(did[i], dprob[i], dk_n[i], rng);
            pc = mtp_gate_stat(did[i], dprob[i], dk_n[i], draft[i]);
        }
        /* Confidence gate: past the first draft the head must
         * be >= pmin confident to keep extending the chain; below it the
         * chain stops here (this draft is dropped, not proposed). */
        chain *= pc;
        glog_p[i] = pc;
        if (i > 0 && ((pmin > 0.f && pc < pmin) ||
                      (pchain > 0.f && chain < pchain))) {
            m->gated++;
            nd = i;
            break;
        }
        m->proposed++;
        h = draft[i];
    }
    MTP_PT(t_draft, MTP_PH_VERIFY);

    /* 2. Verify [c0, d0 .. d_{nd-1}] in one batched main-model pass. */
    int32_t batch[MTP_MAX_DRAFT + 1];
    batch[0] = c0;
    for (int i = 0; i < nd; i++) batch[1 + i] = draft[i];
    int nb = nd + 1;
    if (!q4_sess_prefill_tokens(s, batch, nb)) return -1; /* nothing kept */
    MTP_PT(t_verify, MTP_PH_LOGITS);

    /* Return protocol: 0 = ok (drafts in out_drafts, next token in
     * *next_c0). 1 = recovered, only c0 was kept (*n_out == 0), the caller
     * must continue in plain mode from q4_sess_logits(). 2 = recovered,
     * the accepted drafts in out_drafts WERE kept, continue plain after
     * emitting them. -1 = nothing was processed, the caller must run
     * q4_sess_decode_ex(c0) itself. */
    int32_t tid[Q4_DIST_MAX];
    float tprob[Q4_DIST_MAX];
    int n_acc = 1; /* c0 is a target sample: always verified */
    int rc = 0;
    /* Batched tail: one head pass + a device top-k for all nb rows. The
     * device top-k is q4_topk's order, so q4_dist_from_topk over it is
     * q4_dist_build over the row (top_k is capped at 128 in both). */
    uint32_t vk = greedy ? 1u
                         : (top_k > 0 && top_k <= Q4_DIST_MAX ? (uint32_t)top_k
                                                               : 0u);
    /* presence penalty: fetch the extra candidates a penalized token's
     * place may go to; every row's history is the drafts ahead of it */
    if (vk > 1u && q4_pen_extra()) {
        vk += (uint32_t)q4_pen_extra();
        if (vk > Q4_DIST_MAX) vk = Q4_DIST_MAX;
    }
    const int pen_mark = q4_pen_mark();
    int32_t row_top1[MTP_MAX_DRAFT + 1] = {-1, -1, -1, -1};
    int vtail = vk && mtp_vtail() && q4_sess_verify_topk(s, nb, vk);
    if (!vtail) q4_hip_clear();
    for (int i = 0; i < nd; i++) {
        /* Target distribution at position p+i predicts position p+i+1. */
        int32_t best = 0;
        int tk = 0;
        if (i > 0) q4_pen_push(draft[i - 1]);
        if (!verify_row(s, vtail, i, vk, temp, top_k, top_p, m->n_vocab,
                        &best, tid, tprob, &tk)) {
            rc = 1;
            break;
        }
        row_top1[i] = g_row_top1;
        int acc;
        if (temp <= 1e-5f) {
            acc = (best == draft[i]);
            if (!acc) *next_c0 = best;
        } else {
            int32_t rs = 0;
            if (mtp_dbg() && i == 0) {
                fprintf(stderr, "q4 mtp: row0 target");
                for (int a = 0; a < tk && a < 6; a++)
                    fprintf(stderr, " %d:%.3f", tid[a], tprob[a]);
                fprintf(stderr, " | draft");
                for (int a = 0; a < dk_n[i] && a < 6; a++)
                    fprintf(stderr, " %d:%.3f", did[i][a], dprob[i][a]);
                /* the same row through the one-row float head */
                int32_t fi[Q4_DIST_MAX];
                float fp[Q4_DIST_MAX];
                if (q4_sess_logits_at(s, 0)) {
                    const int fk = q4_dist_build(q4_sess_logits(s), m->n_vocab,
                                                 temp, top_k, top_p, fi, fp);
                    fprintf(stderr, " | float");
                    for (int a = 0; a < fk && a < 6; a++)
                        fprintf(stderr, " %d:%.3f", fi[a], fp[a]);
                }
                fprintf(stderr, "\n");
            }
            if (dtstat && dr_n[i]) {
                static const float dts[8] = {0.7f, 0.85f, 1.f, 1.15f,
                                             0.7f, 0.85f, 1.f, 1.15f};
                for (int v = 0; v < 8; v++) {
                    int32_t qi[Q4_DIST_MAX];
                    float qp[Q4_DIST_MAX];
                    const int qn = q4_dist_from_topk(
                        dr_ids[i], dr_vals[i], dr_n[i], temp * dts[v],
                        v < 4 ? top_p : 1.f, qi, qp);
                    double ov = 0.0;
                    for (int a = 0; a < tk; a++) {
                        const float qa = q4_dist_prob(qi, qp, qn, tid[a]);
                        ov += tprob[a] < qa ? tprob[a] : qa;
                    }
                    m->dt_ov[v] += ov;
                }
                /* Strata's exact scheme: draft the head's mode, accept when
                 * the target's own sample equals it — p(mode_q). */
                int32_t md = dr_ids[i][0];
                float mv = dr_vals[i][0];
                for (int a = 1; a < dr_n[i]; a++)
                    if (dr_vals[i][a] > mv) { mv = dr_vals[i][a]; md = dr_ids[i][a]; }
                m->dt_ov[8] += q4_dist_prob(tid, tprob, tk, md);
                m->dt_n++;
            }
            acc = q4_spec_accept(tid, tprob, tk, did[i], dprob[i], dk_n[i],
                                 draft[i], rng, &rs);
            if (!acc) *next_c0 = rs;
            if (!acc && getenv("Q4_MTP_TREESTAT")) {
                /* A second candidate at this position (the draft
                 * distribution without draft[i], renormalized) would be
                 * accepted against the residual norm(max(0, p - q)) with
                 * probability sum_x min(r(x), q2(x)). */
                float r[Q4_DIST_MAX], rs_ = 0.f;
                for (int a = 0; a < tk; a++) {
                    float d = tprob[a] - q4_dist_prob(did[i], dprob[i], dk_n[i], tid[a]);
                    r[a] = d > 0.f ? d : 0.f;
                    rs_ += r[a];
                }
                float q1 = q4_dist_prob(did[i], dprob[i], dk_n[i], draft[i]);
                double pa2 = 0.0;
                if (rs_ > 0.f && q1 < 1.f)
                    for (int a = 0; a < tk; a++) {
                        if (tid[a] == draft[i]) continue;
                        float q2 = q4_dist_prob(did[i], dprob[i], dk_n[i], tid[a]) / (1.f - q1);
                        float rr = r[a] / rs_;
                        pa2 += rr < q2 ? rr : q2;
                    }
                m->tree_pa2 += pa2;
                if (i < 8) { m->tree_rej[i]++; m->tree_pa2d[i] += pa2; }
            }
        }
        if (acc) {
            out_drafts[i] = draft[i];
            m->accepted++;
            n_acc = i + 2;
        } else {
            break;
        }
        {
            /* An accepted turn-ending draft ends the cycle there: the
             * drafts after it would be committed but never emitted, and a
             * state holding tokens past the turn's end cannot be restored
             * by the next turn (its prompt continues after the stop). */
            int st = 0;
            for (int k = 0; k < m->n_stop; k++) st |= draft[i] == m->stop[k];
            if (st) {
                *next_c0 = draft[i];
                break;
            }
        }
        if (i == nd - 1) {
            /* All drafts accepted: sample the bonus token from the last row
             * (position p+nd, predicting p+nd+1). */
            int32_t bb = 0;
            int btk = 0;
            q4_pen_push(draft[nd - 1]);
            const int vr_ok = verify_row(s, vtail, nd, vk, temp, top_k, top_p,
                                         m->n_vocab, &bb, tid, tprob, &btk);
            row_top1[nd] = g_row_top1;
            if (!vr_ok) {
                rc = 1;
                n_acc = nd; /* the bonus sample is void */
                *next_c0 = 0;
            } else if (temp <= 1e-5f) {
                *next_c0 = bb;
            } else {
                *next_c0 = q4_dist_sample(tid, tprob, btk, rng);
            }
        }
    }
    m->cycles++;
    q4_pen_rewind(pen_mark); /* the engine pushes what it emits */
    {
        /* Q4_TOP1_DUMP=<file>: "pos top1" for every row whose prediction
         * was used (rows 0..n_acc-1: position p+i+1), to compare the
         * decode path with a teacher-forced prompt pass over the same ids
         * (q4 ppl with Q4_PPL_IDS, Q4_PPL_DUMP's argmax column). */
        static FILE *tf;
        static int tfi;
        if (!tfi) {
            tfi = 1;
            const char *e = getenv("Q4_TOP1_DUMP");
            if (e && e[0]) tf = fopen(e, "a");
        }
        if (tf && rc == 0) {
            for (int i = 0; i < n_acc && i <= nd; i++)
                fprintf(tf, "%u %d\n", p + (uint32_t)i + 1, row_top1[i]);
            fflush(tf);
        }
    }

    if (mtp_dbg()) {
        fprintf(stderr, "q4 mtp: cycle p=%u drafts=[", p);
        for (int i = 0; i < nd; i++) fprintf(stderr, " %d", draft[i]);
        fprintf(stderr, "] n_acc=%d next_c0=%d top1_row0=%d\n", n_acc, *next_c0,
                argmax(q4_sess_logits(s), m->n_vocab));
    }
    if (prof) t_cycles++;
    MTP_PT(t_logits, MTP_PH_COMMIT);

    /* 3. Keep the accepted prefix (rollback + replay when the tail was
     * rejected). */
    /* Q4_MTP_GLOG=<file>: one line per cycle, "nd accepted p0 p1 .." —
     * offline calibration of the draft gate (run with Q4_MTP_PMIN=0). */
    {
        static FILE *gl;
        static int gli;
        if (!gli) {
            gli = 1;
            const char *e = getenv("Q4_MTP_GLOG");
            if (e && e[0]) gl = fopen(e, "a");
        }
        if (gl) {
            fprintf(gl, "%d %d", nd, n_acc - 1);
            for (int i = 0; i < nd; i++) fprintf(gl, " %.4f", glog_p[i]);
            fputc('\n', gl);
            fflush(gl);
        }
    }
    if (rc != 0) {
        /* A failed row read keeps only c0, but the verify pass advanced
         * the session (KV, GDN state) over every batch row: roll back. */
        if (!q4_sess_mtp_commit(s, 1)) {
            MTP_PT(t_commit, MTP_PH_REPAIR);
            return -1;
        }
    } else if (!q4_sess_mtp_commit(s, n_acc)) {
        /* Redo from the intact pre-batch snapshot, keeping only c0. */
        if (q4_sess_mtp_commit(s, 1)) {
            rc = 1;
        } else {
            MTP_PT(t_commit, MTP_PH_REPAIR);
            return -1; /* unrecoverable: caller re-decodes c0 */
        }
    }
    MTP_PT(t_commit, MTP_PH_REPAIR);
    if (rc != 0) {
        *n_out = 0;
        q4_sess_logits_at(s, 0);
        if (next_logits) *next_logits = q4_sess_logits(s);
        return rc;
    }
    *n_out = n_acc - 1;

    /* 4. Repair the head KV cells of the committed tokens with the real
     * residuals. Cell p+t-1 pairs the residual at p+t-1 (pref_row t-1)
     * with batch[t] — the token at position p+t. Cell p-1 (c0's pair) was
     * written by draft step 0 with the real seed residual, so only the
     * accepted drafts' cells (t >= 1) still hold chain estimates. The
     * cells/rows are contiguous, so one batched pass replaces n_acc-1
     * single-token head forwards.  Only the cells are needed (the next
     * draft reseeds from the verifier's residual), so it is the prompt's
     * K/V-only pass: the full forward attended over the whole head KV and
     * ran the head MoE — 11 ms/cycle at 10K context, 36 at 34K. */
    m->kv_fill = fill0 + 1;
    int repaired = 0;
    static int rcheck = -1;
    if (rcheck < 0) {
        const char *e = getenv("Q4_MTP_KVCHECK");
        rcheck = e && e[0] == '1';
    }
    if (rcheck && n_acc > 1 && q4_sess_pref_row(s, 0)) {
        /* Q4_MTP_KVCHECK=1: the repair's cells by both passes. */
        const size_t kvw = (size_t)m->n_kvh * m->hd;
        const size_t nf = (size_t)(n_acc - 1) * kvw;
        float *a = malloc(nf * 4 * sizeof(float));
        const float *r0 = q4_sess_pref_row(s, 0);
        bool ok = a != NULL;
        for (int pass = 0; ok && pass < 2; pass++) {
            float *dst = a + (size_t)pass * 2 * nf;
            m->kv_fill = fill0 + 1;
            ok = (pass ? fwd_m(m, (int32_t)p, batch + 1, r0,
                               (uint32_t)(n_acc - 1), false)
                       : fwd_kv(m, (int32_t)p, batch + 1, r0,
                                (uint32_t)(n_acc - 1))) &&
                 q4_hip_d2h(dst, m->d_kv_k + (size_t)(fill0 + 1) * kvw,
                            nf * sizeof(float)) &&
                 q4_hip_d2h(dst + nf, m->d_kv_v + (size_t)(fill0 + 1) * kvw,
                            nf * sizeof(float));
        }
        if (ok) {
            double md = 0, mx = 0;
            for (size_t i = 0; i < 2 * nf; i++) {
                double d = fabs((double)a[i] - (double)a[2 * nf + i]);
                if (d > md) md = d;
                if (fabs((double)a[2 * nf + i]) > mx) mx = fabs((double)a[2 * nf + i]);
            }
            fprintf(stderr, "q4 mtp kvcheck: repair %d cells  max|d| %.3g "
                            "(max %.3g)\n", n_acc - 1, md, mx);
        }
        free(a);
        m->kv_fill = fill0 + 1;
    }
    if (n_acc > 1) {
        const float *r0 = q4_sess_pref_row(s, 0);
        if (r0 && (mtp_kvonly()
                       ? fwd_kv(m, (int32_t)p, batch + 1, r0,
                                (uint32_t)(n_acc - 1))
                       : fwd_m(m, (int32_t)p, batch + 1, r0,
                               (uint32_t)(n_acc - 1), false))) {
            repaired = n_acc - 1;
        } else {
            /* Batched repair failed: degrade to per-token so the cells
             * that still fit are repaired. */
            m->kv_fill = fill0 + 1;
            for (int t = 1; t < n_acc; t++) {
                const float *hrow = q4_sess_pref_row(s, t - 1);
                if (!hrow) break;
                if (!fwd_m(m, (int32_t)p + t - 1, &batch[t], hrow, 1,
                           false))
                    break;
                repaired = t;
            }
        }
    }
    MTP_PT(t_repair, MTP_PH_NONE);
    if (repaired < n_acc - 1) {
        /* Head KV partially repaired: drafts degrade but stay exact. The
         * caller keeps the accepted tokens and switches to plain mode. */
        q4_sess_logits_at(s, n_acc - 1);
        if (next_logits) *next_logits = q4_sess_logits(s);
        return 2;
    }
    /* Cells [.., fill0 + n_acc) now hold positions .. p + n_acc - 2. */
    if (kv_contig) m->kv_last = (int64_t)p + n_acc - 2;
    /* The batched tail leaves the host logits untouched (stale): the
     * caller keeps its previous pointer. */
    if (next_logits && !vtail) *next_logits = q4_sess_logits(s);
    return 0;
}

void q4_mtp_reset(q4_mtp *m) {
    /* The session restarted (q4_sess_reset): head positions restart at 0,
     * so all carried-over entries would be stale duplicates. Stats reset
     * too — the per-request acceptance/mean-len line reads them against
     * this request's token count. */
    if (m) {
        m->kv_fill = 0;
        m->kv_last = -1;
        if (getenv("Q4_MTP_TREESTAT") && m->cycles) {
            uint64_t nr = 0;
            for (int i = 0; i < 8; i++) nr += m->tree_rej[i];
            fprintf(stderr, "q4 mtp treestat: cycles %llu rejections %llu  mean pa2 %.3f  by depth:",
                    (unsigned long long)m->cycles, (unsigned long long)nr,
                    nr ? m->tree_pa2 / (double)nr : 0.0);
            for (int i = 0; i < 4; i++)
                fprintf(stderr, " d%d %llu/%.3f", i, (unsigned long long)m->tree_rej[i],
                        m->tree_rej[i] ? m->tree_pa2d[i] / (double)m->tree_rej[i] : 0.0);
            fprintf(stderr, "\n");
        }
        mtp_dtstat_print(m);
        m->tree_pa2 = 0;
        memset(m->tree_rej, 0, sizeof m->tree_rej);
        memset(m->tree_pa2d, 0, sizeof m->tree_pa2d);
        m->proposed = m->accepted = m->cycles = m->gated = 0;
    }
}

void q4_mtp_set_stops(q4_mtp *m, int32_t a, int32_t b) {
    if (!m) return;
    m->n_stop = 0;
    if (a >= 0) m->stop[m->n_stop++] = a;
    if (b >= 0 && b != a) m->stop[m->n_stop++] = b;
}
