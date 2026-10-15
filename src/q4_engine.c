#include "q4.h"

#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

struct q4_engine {
    q4_gguf g;
    q4_store *st;
    q4_expert_cache *ex;
    q4_ple *ple;
    q4_tok *tok;
    q4_sess *sess;
    q4_mtp *mtp;
    int mtp_fail;
    /* Images of the next generate (q4_engine_set_images; rows borrowed
     * from the caller) and the arrays images_bind builds from them. */
    q4_image *img;
    int n_img;
    int32_t *img_rids;
    const float **img_rows;
    int32_t *img_tab;
    int img_orphan_end;   /* kept cells of pictures the request dropped end here */
    int mrope_on;
    uint32_t ctx;
    uint32_t slots;
    unsigned rng;
    float temp;
    int top_k;
    float top_p;
    float presence;            /* presence penalty, 0 = off */
    int render_only;           /* next turn: build the prompt, dump, stop */
    float def_temp, def_top_p; /* startup sampling, a request's fallback */
    int def_top_k;
    float def_presence;
    /* thinking off: the model card's instruct set */
    float nt_temp, nt_top_p, nt_presence;
    int nt_top_k;
    int err_code;              /* last turn: Q4_ERR_* */
    char err[256];
    int prompt_tokens;
    int completion_tokens;
    int show_thinking;
    int hit_limit;
    char *reason;
    int reason_n;
    int reason_cap;
    char name[128];
    char model_path[4096];
    pthread_mutex_t mu;
    void (*progress)(int prefill_i, int prefill_n, int gen_i, void *u);
    void *progress_u;
    /* Set by the HTTP client when the socket dies. Same thread as generate. */
    volatile int *cancel;
    int pref_base;   /* tokens completed before the current chunk */
    int pref_chunk;  /* tokens in the current chunk */
    int pref_total;  /* prompt tokens */
};

/* Next prefill chunk of `rem` tokens: the same number of chunks as
 * ceil(rem / chunk), but equal sizes (rounded up to 64).  Each chunk
 * streams every non-resident expert once over PCIe (~1.7 s); a short
 * tail chunk (8192 + 1700 for a 10K prompt) cannot hide that behind its
 * own compute, two ~4.9K chunks can.  Q4_PREF_BALANCE=0: full chunks. */
static int pref_split(int rem, int chunk) {
    static int bal = -1;
    if (bal < 0)
        bal = !(getenv("Q4_PREF_BALANCE") && getenv("Q4_PREF_BALANCE")[0] == '0');
    if (rem <= chunk) return rem;
    if (!bal) return chunk;
    const int nch = (rem + chunk - 1) / chunk;
    int c = (rem + nch - 1) / nch;
    c = (c + 63) / 64 * 64;
    return c > chunk ? chunk : c;
}

static void pref_layer_hook(uint32_t il, uint32_t nl, void *u) {
    q4_engine *e = u;
    if (!e || !e->progress || !nl) return;
    int done = e->pref_base + (int)((uint64_t)il * (uint64_t)e->pref_chunk / nl);
    e->progress(done, e->pref_total, 0, e->progress_u);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[128];
    unsigned long avail = 0, freeb = 0, v;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %lu", &v) == 1) avail = v;
        else if (sscanf(line, "MemFree: %lu", &v) == 1) freeb = v;
    }
    fclose(f);
    return (avail ? avail : freeb) * 1024ull;
}

static uint64_t default_l2(void) {
    uint64_t want = 24ull * Q4_GIB;
    const char *e = getenv("Q4_L2_GIB");
    if (e && e[0]) {
        int v = atoi(e);
        want = v <= 0 ? 0 : (uint64_t)v * Q4_GIB;
    }
    /* Same floor the expert cache enforces. 16 GiB plus a 45 GiB resident
     * image does not fit in 60 GiB. */
    uint64_t reserve = 4ull * Q4_GIB;
    const char *r = getenv("Q4_DRAM_RESERVE_GIB");
    if (r && r[0]) {
        int v = atoi(r);
        reserve = v < 4 ? 4 : (uint64_t)v * Q4_GIB;
    }
    uint64_t avail = mem_available();
    if (avail && want + reserve > avail) {
        uint64_t cap = avail > reserve ? avail - reserve : 0;
        if (cap < want) {
            fprintf(stderr,
                    "q4: L2 %.0f GiB capped to %.0f GiB (avail %.0f, reserve %.0f)\n",
                    (double)want / Q4_GIB, (double)cap / Q4_GIB,
                    (double)avail / Q4_GIB, (double)reserve / Q4_GIB);
            want = cap;
        }
    }
    return want;
}

static uint32_t warm_pin_cap(const q4_engine *e);

static uint32_t default_ctx(uint32_t ctx, const q4_gguf *g) {
    if (!ctx) {
        const char *e = getenv("Q4_CTX");
        ctx = e && e[0] ? (uint32_t)atoi(e) : 262144;
    }
    if (!ctx) ctx = 262144;
    if (g && g->n_ctx_train && ctx > g->n_ctx_train) ctx = g->n_ctx_train;
    return ctx;
}

static int tok_is_special(const q4_tok *t, int32_t id) {
    if (!t) return 1;
    if (id == q4_tok_eos(t) || id == q4_tok_im_end(t)) return 1;
    char buf[64];
    int n = q4_tok_decode(t, &id, 1, buf, (int)sizeof(buf));
    if (n <= 0) return 1;
    return buf[0] == '<' && buf[1] == '|';
}

/* KV paging donor: hand at least `chunks` more chunks' worth of expert
 * slots to the KV (q4_sess_kv_* calls it while the pool is short; with
 * pinned chunks a call may not cover a specific home, so it is called
 * again until the pool holds every chunk the map needs). */
static bool engine_kv_donor(void *ctx, uint32_t chunks) {
    q4_engine *e = (q4_engine *)ctx;
    const uint64_t sb = q4_expert_cache_slot_bytes(e->ex);
    const uint64_t ch = q4_hip_vchunk();
    double t0 = now_s();
    uint32_t n = (uint32_t)(((uint64_t)chunks * ch + sb - 1) / sb) + 1;
    uint32_t given = q4_expert_kv_donate(e->ex, n);
    if (!given) return false;
    fprintf(stderr, "q4: KV took %u expert slots (%u handed over)  %.0f ms\n",
            given, q4_expert_kv_given(e->ex), (now_s() - t0) * 1e3);
    if (getenv("Q4_VMM_AUDIT") && getenv("Q4_VMM_AUDIT")[0] == '1')
        q4_hip_vmm_audit("donor");
    return true;
}

/* At a request start (session reset): unmap the KV above this prompt and
 * take as many slots back as the pool covers. */
static void engine_kv_fit(q4_engine *e, uint32_t n_prompt) {
    if (!e->sess || !q4_hip_arena_is_vmm()) return;
    const int audit = getenv("Q4_VMM_AUDIT") && getenv("Q4_VMM_AUDIT")[0] == '1';
    if (audit) q4_hip_vmm_audit("fit-start");
    if (getenv("Q4_SLOT_CHECK") && getenv("Q4_SLOT_CHECK")[0] == '1') {
        q4_expert_slot_check(e->ex, "fit-start");
        q4_expert_devtab_check(e->ex, "fit-start");
    }
    /* Trim + reclaim only with pinned VMM chunks (q4_hip_vmm_pinned): a VA
     * remapped to a different chunk is not safe to copy into - the first H2D
     * uploads into reclaimed slots landed in the chunk the VA had before
     * (2026-10-15: slot 4082's bytes in slot 4081's chunk at the same offset,
     * slot 4082 stale) - the persistent "NaN after a long generation" bug.
     * Q4_KV_RECLAIM=0 keeps the KV's chunks until restart (grow-only),
     * =1 forces the reclaim (with Q4_VMM_PIN=0: the old, unsafe path). */
    static int reclaim = -1;
    if (reclaim < 0) {
        const char *r = getenv("Q4_KV_RECLAIM");
        reclaim = r && r[0] ? r[0] == '1' : q4_hip_vmm_pinned();
    }
    if (reclaim && q4_expert_kv_given(e->ex)) {
        (void)q4_sess_kv_trim(e->sess, n_prompt + 1);
        uint32_t back =
            q4_expert_kv_reclaim(e->ex, q4_expert_kv_given(e->ex));
        if (back)
            fprintf(stderr,
                    "q4: KV returned %u expert slots (%u still handed over)\n",
                    back, q4_expert_kv_given(e->ex));
    }
    /* Map the whole prompt's KV now, with the device idle: a chunk mapped
     * while kernels run stalls them (the driver preempts the queues to
     * update the page tables) — 0.4-0.6 s at each 16K step mid-prompt. */
    (void)q4_sess_kv_reserve(e->sess, n_prompt + 1);
    if (audit) q4_hip_vmm_audit("fit-end");
    if (getenv("Q4_SLOT_CHECK") && getenv("Q4_SLOT_CHECK")[0] == '1')
        q4_expert_slot_check(e->ex, "fit-end");
}

q4_engine *q4_engine_open_full(const char *model, uint32_t ctx, uint64_t l2_bytes,
                               const char *mtp_path) {
    if (!model || !model[0]) return NULL;
    q4_engine *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->temp = 0.7f;
    e->top_k = 20;
    e->top_p = 0.8f;
    e->rng = (unsigned)time(NULL) ^ 0x9e3779b9u;
    e->show_thinking =
        getenv("Q4_SHOW_THINKING") && getenv("Q4_SHOW_THINKING")[0] == '1';
    pthread_mutex_init(&e->mu, NULL);
    if (!q4_gguf_open(&e->g, model)) {
        fprintf(stderr, "q4: failed to open %s\n", model);
        pthread_mutex_destroy(&e->mu);
        free(e);
        return NULL;
    }
    /* What a request that sets no sampling gets: the model's own
     * general.sampling (Qwen3.8: temp 1, top_k 20, top_p 0.95) — OpenCode
     * sends neither temperature nor top_p for an openai-compatible model
     * unless its config says it may, so this IS the agent's sampling.
     * Q4_TEMP / Q4_TOP_K / Q4_TOP_P override it; Q4_SEED pins the sampler
     * for A/B probes. */
    if (e->g.samp_has_temp) e->temp = e->g.samp_temp;
    if (e->g.samp_top_k > 0) e->top_k = (int)e->g.samp_top_k;
    if (e->g.samp_top_p > 0.f) e->top_p = e->g.samp_top_p;
    if (getenv("Q4_SEED")) e->rng = (unsigned)strtoul(getenv("Q4_SEED"), NULL, 0);
    if (getenv("Q4_TEMP")) e->temp = (float)atof(getenv("Q4_TEMP"));
    if (getenv("Q4_TOP_K")) e->top_k = atoi(getenv("Q4_TOP_K"));
    if (getenv("Q4_TOP_P")) e->top_p = (float)atof(getenv("Q4_TOP_P"));
    e->def_temp = e->temp;
    e->def_top_k = e->top_k;
    e->def_top_p = e->top_p;
    e->def_presence = 0.f;
    /* Thinking off: the Qwen3.8 card's instruct set (temp 0.7, top_p 0.8,
     * top_k 20, presence_penalty 1.5; general.sampling holds only the
     * thinking set).  An explicit Q4_TEMP / Q4_TOP_K / Q4_TOP_P /
     * Q4_PRESENCE applies to both modes. */
    e->nt_temp = 0.7f;
    e->nt_top_k = 20;
    e->nt_top_p = 0.8f;
    e->nt_presence = 1.5f;
    if (getenv("Q4_TEMP")) e->nt_temp = e->temp;
    if (getenv("Q4_TOP_K")) e->nt_top_k = e->top_k;
    if (getenv("Q4_TOP_P")) e->nt_top_p = e->top_p;
    if (getenv("Q4_PRESENCE"))
        e->def_presence = e->nt_presence = (float)atof(getenv("Q4_PRESENCE"));
    fprintf(stderr, "q4: default sampling temp %.2f top_k %d top_p %.2f "
            "presence %.2f%s; thinking off %.2f / %d / %.2f / %.2f\n",
            e->temp, e->top_k, e->top_p, e->def_presence,
            e->g.samp_has_temp ? " (general.sampling)" : "", e->nt_temp,
            e->nt_top_k, e->nt_top_p, e->nt_presence);
    snprintf(e->name, sizeof(e->name), "qwen3.8-flash-next");
    snprintf(e->model_path, sizeof(e->model_path), "%s", model);
    e->ctx = default_ctx(ctx, &e->g);
    if (!l2_bytes) l2_bytes = default_l2();

    bool hip = q4_hip_init();
    fprintf(stderr, "q4: hip %s  ctx %u  model %s\n", hip ? "ok gfx1100" : "cpu",
            e->ctx, e->name);

    double t0 = now_s();
    e->st = q4_store_open(&e->g);
    if (!e->st) {
        fprintf(stderr, "q4: dense store load failed\n");
        q4_engine_close(e);
        return NULL;
    }
    fprintf(stderr, "q4: dense host  %.2f s\n", now_s() - t0);
    bool dense_on_gpu = false;
    if (hip) {
        t0 = now_s();
        if (!q4_store_upload_hip(e->st))
            fprintf(stderr, "q4: VRAM upload failed, GEMV will H2D\n");
        else {
            fprintf(stderr, "q4: dense VRAM %.2f s  (%.2f GiB)\n", now_s() - t0,
                    (double)(e->g.bytes_dense + e->g.bytes_shared) / Q4_GIB);
            dense_on_gpu = true;
        }
    }
    /* Free host dense copies (keep token_embd + PLE conv): leaves DRAM for
     * the expert L2. Q4_FREE_HOST_DENSE=0 keeps the old behavior. */
    {
        const char *fd = getenv("Q4_FREE_HOST_DENSE");
        if (dense_on_gpu && (!fd || fd[0] != '0')) {
            q4_store_free_host_except(e->st, &e->g);
            /* glibc keeps the freed dense image in the heap, so MemAvailable
             * still looks too small for the 45 GiB resident alloc. */
            malloc_trim(0);
            fprintf(stderr, "q4: host dense freed (token_embd + PLE conv kept)\n");
        }
    }

    q4_machine m = q4_machine_this_pc(e->ctx);
    /* Budget L1 off live free VRAM: the dense image is already resident, so
     * free-now + dense reconstructs the planner's "VRAM available" input
     * with the desktop's real share already priced in (the static 23 GiB
     * default under-fills the 25.75 GiB card by ~10 GiB). */
    if (dense_on_gpu) {
        uint64_t fr = q4_hip_vram_free();
        if (fr) m.vram_bytes = fr + e->g.bytes_dense + e->g.bytes_shared;
    }
    q4_place p = q4_place_plan(&e->g, &m);
    q4_place_print(&p, &e->g, stderr);
    fprintf(stderr, "q4: QSA KV %.2f GiB  GDN state %.2f GiB  (%u QSA layers)\n",
            (double)q4_qsa_kv_bytes(&e->g, e->ctx) / Q4_GIB,
            (double)q4_gdn_state_bytes(&e->g) / Q4_GIB, q4_n_qsa_layers(&e->g));

    uint32_t slots = p.l1_experts;
    /* Leave VRAM for ROCm scratch. Filling 24 GiB made OpenCode decode hang.
     * Q4_L1_SLOTS can only cap below the planner value, never push past it:
     * a bigger prefill chunk shrinks the budget the planner computed. */
    uint32_t cap = 8192;
    const char *ls = getenv("Q4_L1_SLOTS");
    if (ls && ls[0]) {
        int v = atoi(ls);
        if (v > 0) cap = (uint32_t)v;
        if (cap > p.l1_experts)
            fprintf(stderr,
                    "q4: Q4_L1_SLOTS %u exceeds planner %u; clamped\n",
                    cap, p.l1_experts);
    }
    if (slots > cap) slots = cap;
    if (slots < 64) slots = 64;
    e->slots = slots;

    /* Reserve VRAM for the MTP head before the main GPU L1 is sized. The
     * fixed part (dense + KV + scratch + verify shadows) plus a small L1. */
    bool want_mtp = hip && mtp_path && mtp_path[0];
    uint32_t mtp_slots = 128; /* a prefill group of 8 tokens routes up to
                               * 80 distinct head experts: 64 would evict
                               * mid-batch (moe _n fail #14) */
    {
        const char *ms0 = getenv("Q4_MTP_SLOTS");
        if (ms0 && ms0[0]) {
            int v = atoi(ms0);
            if (v > 0) mtp_slots = (uint32_t)v;
        }
    }
    uint64_t head_est = 0;
    if (want_mtp) {
        /* The head's real device footprint, in three pieces (same
         * inventory as q4_mtp_fixed_vram):
         *  - device upload: the head gguf's dense+shared bytes
         *    (~433 MiB for the shared-Q8_0 head, real value read below)
         *  - fixed buffers ~528 MiB: KV 2x64Kx2kvh x 256hd fp32 = 256,
         *    d_pws batched-MoE scratch = 128, d_ws/small ~7, devtab ~0.1,
         *    session verify shadows ~131 (q4_gdn_state_bytes + ~13)
         *  - runtime extras ~400 MiB: first-request kernel uploads,
         *    graph instantiation and the head L1 stager — moved here
         *    from the 1.5 GiB slack (back to 1.0 below).
         * plus the head L1 arena at its real per-slot bytes (~5 MiB
         * Q8_0, not the old flat 6).
         * No MMQ wbuf term any more: the head never runs MMQ (arena != 0),
         * so the shared wbuf stays at main strides inside p.scratch_vram.
         * The old 1792+6/slot+dn form reserved ~3.72 GiB and cost ~1500
         * main slots for a real footprint of ~2 GiB.
         * Q4_MTP_RESIDENT=1 (or Q4_MTP_ROWS>8) sizes the head scratch for
         * a 64-row batch (~19 MiB over the 8-row buffers). That delta is
         * the +32 MiB below; it is not taken out of the 928 slack, and
         * it does not grow the head L1. */
        uint64_t hup = 448ull << 20, hslot = 5ull << 20;
        {
            q4_gguf hg;
            if (q4_gguf_open_ex(&hg, mtp_path, 1)) {
                if (hg.bytes_dense + hg.bytes_shared)
                    hup = hg.bytes_dense + hg.bytes_shared;
                uint64_t pb = q4_expert_packed_bytes(&hg);
                if (pb) hslot = pb;
                q4_gguf_close(&hg);
            }
        }
        uint64_t slack = 928ull << 20;
        {
            int wide = 0;
            const char *mr = getenv("Q4_MTP_RESIDENT");
            const char *rows = getenv("Q4_MTP_ROWS");
            if (mr && mr[0] && mr[0] != '0') wide = 1;
            if (rows && rows[0] && atoi(rows) > 8) wide = 1;
            if (wide) slack += 32ull << 20;
        }
        head_est = hup + (uint64_t)mtp_slots * hslot + slack;
        /* Reserve head_est *bytes* of main-arena capacity: divide by the
         * real per-slot bytes, not the head's (3x bigger) — the old
         * head_est/msb form under-covered by ~2/3, and the shortfall came
         * out of the measured reserve (pf8k drove vfree under the floor). */
        uint64_t msb = q4_expert_packed_bytes(&e->g);
        if (!msb) msb = e->g.per_expert_bytes;
        if (!msb) msb = 3ull << 20;
        uint32_t reserve = (uint32_t)((head_est + msb - 1) / msb);
        if (reserve < slots) {
            slots -= reserve;
            fprintf(stderr,
                    "q4: L1 slots -> %u (%u reserved for MTP head, "
                    "~%.2f GiB VRAM)\n",
                    slots, reserve, (double)head_est / Q4_GIB);
        }
    }

    /* L1 lending: the prompt's big device scratch (counted in
     * p.scratch_vram, so not in `slots`) becomes n_tail extra arena slots
     * that are borrowed only while a prompt runs.  Same VRAM, more expert
     * slots during decode.  Q4_L1_LEND=0 restores separate allocations. */
    uint32_t n_tail = 0;
    uint64_t lend_b = 0;
    if (hip) {
        const char *le = getenv("Q4_L1_LEND");
        uint64_t msb = q4_expert_packed_bytes(&e->g);
        if (!(le && le[0] == '0') && msb) {
            lend_b = q4_sess_lend_bytes(&e->g, q4_pref_chunk());
            if (lend_b)
                n_tail = (uint32_t)((lend_b + 65536 + msb - 1) / msb);
        }
    }
    /* KV paging (Q4_KV_PAGE, default on with lending): the QSA KV and the
     * indexer key caches (sized for the full ctx) reserve VA only; n_pool
     * arena slots below the lend tail hand their 2 MiB chunks over as the
     * context grows (q4_sess_kv_*, q4_expert_kv_*) and take them back at
     * the next shorter request.  At full context the experts get what
     * they had with the fixed buffers; below it the rest are LRU slots. */
    uint32_t n_pool = 0;
    if (hip && n_tail) {
        const char *kp = getenv("Q4_KV_PAGE");
        uint64_t msb = q4_expert_packed_bytes(&e->g);
        if (!(kp && kp[0] == '0') && msb) {
            const uint32_t nq = q4_n_qsa_layers(&e->g);
            const uint64_t dim = e->g.indexer_head_size
                                     ? e->g.indexer_head_size
                                     : 128;
            uint64_t kvb = q4_qsa_kv_bytes(&e->g, e->ctx) +
                           (uint64_t)nq * e->ctx * dim * 2 +
                           (uint64_t)nq * (e->ctx / 4 + 1) * dim * 4;
            /* + rounding: the full map is each region's whole chunks (a
             * shorter map never needs more), and each donation strands at
             * most a partial chunk at its boundary */
            kvb += 16ull * q4_hip_vchunk();
            /* Regions off chunk boundaries (any ctx that is not a large
             * power of two) need more than their bytes: ctx 4096 needed 72
             * chunks from a 50-chunk pool and every prompt failed. */
            const uint64_t kvc =
                (q4_kv_page_chunks(&e->g, e->ctx, q4_hip_vchunk()) + 16ull) *
                q4_hip_vchunk();
            if (kvc > kvb) kvb = kvc;
            n_pool = (uint32_t)((kvb + msb - 1) / msb);
            q4_sess_set_kv_paging(1);
            q4_hip_arena_vmm(1);
        }
    }

    e->ex = NULL;
    uint64_t l2b = l2_bytes;
    while (!e->ex) {
        e->ex = q4_expert_cache_open_ex(&e->g, slots + n_tail + n_pool, l2b);
        if (e->ex) break;
        if (l2b > 8ull * Q4_GIB) {
            l2b /= 2;
            fprintf(stderr, "q4: L2 alloc failed, retry %.0f GiB\n",
                    (double)l2b / Q4_GIB);
            continue;
        }
        fprintf(stderr, "q4: expert cache failed\n");
        q4_engine_close(e);
        return NULL;
    }
    /* The resident image is deferred until the GPU arena size is final
     * (below): the static quota and the image's exclusion set both key off
     * the allocated slot count, not the plan. */

    /* Session first so QSA KV claims VRAM before L1 experts. */
    e->ple = q4_ple_open(&e->g);
    uint32_t try_ctx = e->ctx;
    while (try_ctx >= 4096) {
        e->sess = q4_sess_open(&e->g, e->st, e->ex, e->ple, try_ctx);
        if (e->sess && n_pool)
            q4_sess_set_kv_donor(e->sess, engine_kv_donor, e);
        if (e->sess) {
            e->ctx = try_ctx;
            break;
        }
        fprintf(stderr, "q4: session ctx %u failed, retry %u\n", try_ctx,
                try_ctx / 2);
        try_ctx /= 2;
    }
    if (!e->sess) {
        fprintf(stderr, "q4: session open failed\n");
        q4_engine_close(e);
        return NULL;
    }
    fprintf(stderr, "q4: session ctx %u  QSA %s\n", q4_sess_max_kv(e->sess),
            "gpu/host");

    uint32_t got = 0;
    if (hip) {
        uint64_t sb = q4_expert_cache_slot_bytes(e->ex);
        uint32_t s = slots;
        /* Size the arena from MEASURED free VRAM, not just the plan: ROCm
         * heap/page-table/cmd-buffer overhead (~2.4 GiB at this alloc count)
         * and the pending lazy set (prefill workspaces, MMQ gathers,
         * hipBLASLt scratch — all in p.scratch_vram) are invisible to the
         * planner's arithmetic. The clamp keeps reserve + pending free,
         * so steady-state vfree ends up at the reserve, not below it.
         * The MTP head is NOT re-subtracted here: `slots` was already
         * reduced by head_est above, which is its reservation — the
         * head's ~0.35 GiB of runtime extras is inside head_est now,
         * so the slack only covers the main model's first-request
         * growth that never goes through q4_hip_malloc: hipBLASLt's
         * internal workspaces, graph exec instantiation, kernel
         * uploads and per-BO page tables — measured ~1.2 GiB of
         * device-side growth at REQ0 vs ~1.9 GiB of tracked mallocs. */
        /* head_est is in `keep`: when this clamp binds (a large
         * Q4_L1_SLOTS), the head's reservation out of `slots` above no
         * longer applies, and the head (allocated later) used to come out
         * of the reserve — a chunk-1024 run dipped to vfree 872 MiB. */
        uint64_t fr0 = q4_hip_vram_free();
        s += n_tail + n_pool;
        if (fr0) {
            uint64_t scr = p.scratch_vram > lend_b && n_tail
                               ? p.scratch_vram - lend_b
                               : (n_tail ? 0 : p.scratch_vram);
            uint64_t keep = p.reserve_vram + scr + head_est +
                            (1024ull << 20);
            uint32_t fits =
                fr0 > keep ? (uint32_t)((fr0 - keep) / sb) : 0;
            fprintf(stderr,
                    "q4: VRAM free %.2f GiB  keep %.2f (reserve %.2f + "
                    "scratch %.2f + head %.2f + 1)  fits %u  want %u "
                    "(lend tail %u)\n",
                    (double)fr0 / Q4_GIB, (double)keep / Q4_GIB,
                    (double)p.reserve_vram / Q4_GIB, (double)scr / Q4_GIB,
                    (double)head_est / Q4_GIB, fits, s, n_tail);
            if (fits < s) {
                fprintf(stderr,
                        "q4: L1 slots %u -> %u (measured free %.2f GiB)\n",
                        s, fits, (double)fr0 / Q4_GIB);
                s = fits;
            }
        }
        while (s >= 64) {
            if (q4_hip_experts_alloc(s, sb)) {
                got = s;
                break;
            }
            fprintf(stderr, "q4: GPU L1 %u slots failed, retry %u\n", s, s / 2);
            s /= 2;
        }
        if (!got) {
            fprintf(stderr, "q4: GPU L1 alloc failed\n");
            q4_engine_close(e);
            return NULL;
        }
        if (got < q4_expert_cache_n_slots(e->ex)) {
            fprintf(stderr, "q4: cap host L1 %u -> %u to match GPU\n",
                    q4_expert_cache_n_slots(e->ex), got);
            q4_expert_cache_cap_slots(e->ex, got);
        }
        /* Never turn lending off here: the arena was sized with the
         * scratch inside it, so a separate scratch allocation on top would
         * overcommit VRAM by ~3.3 GiB (2026-10-12: vfree 427 MiB, watchdog
         * kill).  Too small an arena for the tail is a startup error. */
        if (n_tail && got < n_tail + n_pool + 2u * e->g.n_expert + 128u) {
            fprintf(stderr,
                    "q4: GPU L1 %u slots cannot hold the %u-slot prompt "
                    "scratch tail — lower Q4_PREF_CHUNK or free VRAM\n",
                    got, n_tail);
            q4_engine_close(e);
            return NULL;
        }
        uint32_t base = got - n_tail - n_pool;
        q4_expert_res_plan(e->ex, base);
        if (n_tail && !q4_expert_lend_init(e->ex, n_tail)) {
            fprintf(stderr, "q4: L1 lend init failed\n");
            q4_engine_close(e);
            return NULL;
        }
        if (n_pool && !q4_expert_kv_init(e->ex, n_pool)) {
            fprintf(stderr, "q4: KV paging init failed\n");
            q4_engine_close(e);
            return NULL;
        }
        e->slots = got;
        fprintf(stderr, "q4: GPU L1 %u slots  %.2f GiB", got,
                (double)got * sb / Q4_GIB);
        if (n_tail)
            fprintf(stderr, "  (last %u = %.2f GiB lent to the prompt "
                    "scratch)", n_tail, (double)n_tail * sb / Q4_GIB);
        if (n_pool)
            fprintf(stderr, "  (%u below = %.2f GiB pageable to the KV)",
                    n_pool, (double)n_pool * sb / Q4_GIB);
        fprintf(stderr, "\n");
        /* Static set: warm-file top-q experts per layer sized
         * from the ALLOCATED slot count, staged into fixed L1 slots, and
         * only the loaded set is excluded from the DRAM image. */
        q4_expert_static_load(e->ex, base);
        if (n_pool) q4_expert_kv_open(e->ex);
    }
    /* res_plan must run BEFORE static_load (it picks the set) — keep it
     * inside the hip block; the non-hip call with n_gpu=0 only parses the
     * warm file for the image quotas. */
    if (!hip) q4_expert_res_plan(e->ex, 0);
    if (q4_expert_res_build(e->ex) < 0) {
        fprintf(stderr, "q4: expert image build failed\n");
        q4_engine_close(e);
        return NULL;
    }
    fprintf(stderr, "q4: DRAM L2 %.2f GiB\n",
            (double)q4_expert_l2_bytes(e->ex) / Q4_GIB);
    if (q4_expert_l2_bytes(e->ex)) {
        t0 = now_s();
        fprintf(stderr, "q4: prefetching routed experts into DRAM...\n");
        q4_expert_prefetch(e->ex, stderr);
        fprintf(stderr, "q4: DRAM L2 filled %.2f s  %u experts\n", now_s() - t0,
                q4_expert_l2_filled(e->ex));
        const char *ml = getenv("Q4_MLOCK_L2");
        if (!ml || ml[0] != '0') {
            if (q4_expert_cache_mlock_l2(e->ex))
                fprintf(stderr, "q4: L2 mlocked (kswapd cannot steal experts)\n");
        }
    }
    if (q4_expert_res_audit(e->ex, stderr) < 0) {
        fprintf(stderr, "q4: residency audit failed — refusing to run\n");
        q4_engine_close(e);
        return NULL;
    }
    if (q4_expert_lend_on(e->ex)) {
        t0 = now_s();
        int put = q4_expert_lend_end(e->ex);
        fprintf(stderr, "q4: L1 lend tail filled: %d warm experts  %.2f s\n",
                put, now_s() - t0);
    }

    if (hip) {
        /* Opt-in for the partial-L2 staging arena only. Resident experts
         * have no host mirror; hipHostRegister of that image is not used. */
        if (getenv("Q4_PIN_ARENA") && getenv("Q4_PIN_ARENA")[0] == '1') {
            if (q4_expert_is_resident(e->ex)) {
                fprintf(stderr,
                        "q4: arena pin skipped (H2D reads the resident image)\n");
            } else {
                double tp = now_s();
                if (q4_expert_cache_pin_arena(e->ex))
                    fprintf(stderr, "q4: L1 host arena pinned (DMA H2D)  %.2f s\n",
                            now_s() - tp);
                else
                    fprintf(stderr, "q4: L1 host arena pin failed, pageable H2D\n");
            }
        }
    }

    /* MTP draft head: dense store + expert cache (arena 1) + verify shadows.
     * Any failure here disables MTP but keeps the engine usable. */
    if (want_mtp) {
        uint32_t head_kv = 65536;
        const char *kv = getenv("Q4_MTP_KV");
        if (kv && kv[0]) head_kv = (uint32_t)atoi(kv);
        const char *ms = getenv("Q4_MTP_SLOTS");
        if (ms && ms[0]) {
            int v = atoi(ms);
            if (v > 0) mtp_slots = (uint32_t)v;
        }
        /* Head experts read from the head GGUF's page cache by default.
         * A DRAM L2 (old default 2 GiB) is opt-in: Q4_MTP_L2_GIB=<n>. */
        uint64_t head_l2 = 0;
        const char *ml2 = getenv("Q4_MTP_L2_GIB");
        if (ml2 && ml2[0]) {
            int v = atoi(ml2);
            head_l2 = v <= 0 ? 0 : (uint64_t)v * Q4_GIB;
        }
        e->mtp = q4_mtp_open(mtp_path, &e->g, e->st, head_kv, mtp_slots, head_l2);
        if (!e->mtp) {
            fprintf(stderr, "q4: MTP disabled (head open failed)\n");
        } else if (q4_mtp_experts_resident(e->mtp)) {
            /* The Q2 pack is the head's expert store. Skip the Q8 L1
             * arena and the resident image; a failed alloc must not
             * disable MTP. */
            if (!q4_sess_mtp_prep(e->sess, 5)) {
                fprintf(stderr, "q4: MTP disabled (session prep failed)\n");
                q4_mtp_close(e->mtp);
                e->mtp = NULL;
            } else {
                fprintf(stderr, "q4: MTP ready  draft head '%s'\n", mtp_path);
            }
        } else {
            uint64_t msb = q4_mtp_slot_bytes(e->mtp);
            uint32_t hs = q4_expert_cache_n_slots(q4_mtp_cache(e->mtp));
            uint32_t got = 0;
            while (hs >= 16) {
                if (q4_hip_experts_alloc_a(1, hs, msb)) {
                    got = hs;
                    break;
                }
                fprintf(stderr, "q4: MTP L1 %u slots failed, retry %u\n", hs,
                        hs / 2);
                hs /= 2;
            }
            if (!got) {
                fprintf(stderr, "q4: MTP disabled (GPU L1 alloc failed)\n");
                q4_mtp_close(e->mtp);
                e->mtp = NULL;
            } else {
                if (got < q4_expert_cache_n_slots(q4_mtp_cache(e->mtp)))
                    q4_expert_cache_cap_slots(q4_mtp_cache(e->mtp), got);
                /* Q4_MTP_RESIDENT=1 defers the head image just like the
                 * main cache — build it now that the head arena exists. */
                q4_expert_res_plan(q4_mtp_cache(e->mtp), got);
                if (q4_expert_res_build(q4_mtp_cache(e->mtp)) < 0) {
                    fprintf(stderr, "q4: MTP disabled (head image failed)\n");
                    q4_mtp_close(e->mtp);
                    e->mtp = NULL;
                }
                if (!e->mtp) {
                    /* build failed */
                } else if (!q4_sess_mtp_prep(e->sess, 5)) {
                    fprintf(stderr, "q4: MTP disabled (session prep failed)\n");
                    q4_mtp_close(e->mtp);
                    e->mtp = NULL;
                } else {
                    if (q4_expert_l2_bytes(q4_mtp_cache(e->mtp))) {
                        double tp = now_s();
                        q4_expert_prefetch(q4_mtp_cache(e->mtp), stderr);
                        fprintf(stderr, "q4: MTP L2 filled %.2f s\n",
                                now_s() - tp);
                        const char *ml = getenv("Q4_MLOCK_L2");
                        if (!ml || ml[0] != '0')
                            q4_expert_cache_mlock_l2(q4_mtp_cache(e->mtp));
                    }
                    fprintf(stderr, "q4: MTP ready  draft head '%s'\n", mtp_path);
                }
            }
        }
    }

    e->tok = q4_tok_open(&e->g);
    if (!e->tok) {
        fprintf(stderr, "q4: tokenizer failed (missing ggml.tokens?)\n");
        q4_engine_close(e);
        return NULL;
    }
    return e;
}

q4_engine *q4_engine_open(const char *model, uint32_t ctx, uint64_t l2_bytes) {
    return q4_engine_open_full(model, ctx, l2_bytes, NULL);
}

void q4_engine_close(q4_engine *e) {
    if (!e) return;
    /* Dispatcher thread reads e->ex and e->g — join it before teardown. */
    q4_moe_dev_stop();
    free(e->img);
    free(e->img_rids);
    free(e->img_rows);
    free(e->img_tab);
    q4_mtp_close(e->mtp);
    q4_sess_close(e->sess);
    q4_tok_close(e->tok);
    q4_ple_close(e->ple);
    q4_expert_cache_close(e->ex);
    q4_hip_experts_free();
    q4_store_close(e->st);
    q4_hip_shutdown();
    q4_gguf_close(&e->g);
    pthread_mutex_destroy(&e->mu);
    free(e->reason);
    free(e);
}

void q4_engine_set_sample(q4_engine *e, float temp, int top_k, float top_p) {
    if (!e) return;
    e->temp = temp;
    e->top_k = top_k;
    e->top_p = top_p;
}

void q4_engine_set_presence(q4_engine *e, float presence) {
    if (e) e->presence = presence > 0.f ? presence : 0.f;
}

void q4_engine_set_render_only(q4_engine *e, int on) {
    if (e) e->render_only = on;
}

void q4_engine_set_progress(q4_engine *e,
                            void (*cb)(int prefill_i, int prefill_n, int gen_i,
                                       void *u),
                            void *u) {
    if (!e) return;
    e->progress = cb;
    e->progress_u = u;
}

void q4_engine_set_cancel(q4_engine *e, volatile int *flag) {
    if (!e) return;
    e->cancel = flag;
}

uint32_t q4_engine_ctx(const q4_engine *e) {
    return e ? e->ctx : 0;
}

const char *q4_engine_name(const q4_engine *e) {
    return e ? e->name : "";
}

const char *q4_engine_model_path(const q4_engine *e) {
    return e ? e->model_path : "";
}

int q4_engine_last_prompt_tokens(const q4_engine *e) {
    return e ? e->prompt_tokens : 0;
}

int q4_engine_last_completion_tokens(const q4_engine *e) {
    return e ? e->completion_tokens : 0;
}

const char *q4_engine_last_reasoning(const q4_engine *e) {
    return e && e->reason ? e->reason : "";
}

int q4_engine_last_hit_limit(const q4_engine *e) {
    return e ? e->hit_limit : 0;
}

int q4_engine_last_error(const q4_engine *e, const char **msg) {
    if (msg) *msg = e ? e->err : "";
    return e ? e->err_code : 0;
}

void q4_engine_default_sample(const q4_engine *e, float *temp, int *top_k,
                              float *top_p) {
    if (!e) return;
    if (temp) *temp = e->def_temp;
    if (top_k) *top_k = e->def_top_k;
    if (top_p) *top_p = e->def_top_p;
}

void q4_engine_default_sample_mode(const q4_engine *e, int think, float *temp,
                                   int *top_k, float *top_p, float *presence) {
    if (!e) return;
    if (temp) *temp = think ? e->def_temp : e->nt_temp;
    if (top_k) *top_k = think ? e->def_top_k : e->nt_top_k;
    if (top_p) *top_p = think ? e->def_top_p : e->nt_top_p;
    if (presence) *presence = think ? e->def_presence : e->nt_presence;
}

static uint64_t hash_str(const char *s) {
    uint64_t h = 14695981039346656037ull;
    if (!s) return 0;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ull;
    }
    return h ? h : 1;
}

static uint64_t first_user_hash(const char *const *roles, const char *const *contents,
                                int nmsg) {
    for (int i = 0; i < nmsg; i++)
        if (roles && roles[i] && strcmp(roles[i], "user") == 0)
            return hash_str(contents ? contents[i] : "");
    return 0;
}

static int reason_append(q4_engine *e, const char *p, int n) {
    if (!e || n <= 0) return 0;
    if (e->reason_n + n + 1 > e->reason_cap) {
        int cap = e->reason_cap ? e->reason_cap : 8192;
        while (cap < e->reason_n + n + 1) cap *= 2;
        if (cap > 8 << 20) cap = 8 << 20;
        if (e->reason_n + n + 1 > cap) return -1;
        char *q = realloc(e->reason, (size_t)cap);
        if (!q) return -1;
        e->reason = q;
        e->reason_cap = cap;
    }
    memcpy(e->reason + e->reason_n, p, (size_t)n);
    e->reason_n += n;
    e->reason[e->reason_n] = 0;
    return 0;
}

/* Decode one BPE piece into complete UTF-8, holding a trailing partial sequence. */
static int flush_utf8_piece(unsigned char *hold, int *nh, const char *piece, int pn,
                            char *emit, int max_emit, int *n_emit) {
    unsigned char tmp[264];
    if (*nh + pn > (int)sizeof(tmp)) *nh = 0;
    memcpy(tmp, hold, (size_t)*nh);
    memcpy(tmp + *nh, piece, (size_t)pn);
    int tot = *nh + pn, i = 0, eo = 0;
    while (i < tot) {
        unsigned char c = tmp[i];
        int L = 1;
        if (c < 0x80) L = 1;
        else if ((c & 0xe0) == 0xc0) L = 2;
        else if ((c & 0xf0) == 0xe0) L = 3;
        else if ((c & 0xf8) == 0xf0) L = 4;
        else {
            i++;
            continue;
        }
        if (i + L > tot) break;
        int ok = 1;
        for (int k = 1; k < L; k++)
            if ((tmp[i + k] & 0xc0) != 0x80) ok = 0;
        if (!ok) {
            i++;
            continue;
        }
        if (eo + L > max_emit) break;
        memcpy(emit + eo, tmp + i, (size_t)L);
        eo += L;
        i += L;
    }
    *nh = tot - i;
    if (*nh > 0) memcpy(hold, tmp + i, (size_t)*nh);
    *n_emit = eo;
    return 0;
}

/* Emit one generated token: think-state tracking, UTF-8 flush, streaming.
 * Returns 0 = continue, 1 = eos/im_end (stop), 2 = output full (stop). */
static int emit_piece(q4_engine *e, int32_t id, char *out, int *o, int max_out,
                      int *in_think, int *think_toks, unsigned char *hold,
                      int *nh, void (*stream_cb)(const char *, int, void *),
                      void *u, int t, int32_t im_end, int32_t eos, int32_t th,
                      int32_t te) {
    if (id == eos || (im_end >= 0 && id == im_end)) return 1;
    if (th >= 0 && id == th) {
        *in_think = 1;
        *nh = 0;
    } else if (te >= 0 && id == te) {
        if (*in_think)
            fprintf(stderr, "\nq4: </think> after %d tok\n", *think_toks);
        *in_think = 0;
        *nh = 0;
    } else if (!tok_is_special(e->tok, id) && (!*in_think || e->show_thinking)) {
        char piece[256];
        int pn = q4_tok_decode(e->tok, &id, 1, piece, (int)sizeof(piece));
        if (pn > 0) {
            char emitbuf[264];
            int eo = 0;
            flush_utf8_piece(hold, nh, piece, pn, emitbuf, (int)sizeof(emitbuf),
                             &eo);
            if (eo > 0) {
                emitbuf[eo] = 0;
                int kind = *in_think ? Q4_PIECE_REASONING : Q4_PIECE_CONTENT;
                if (kind == Q4_PIECE_REASONING) {
                    if (reason_append(e, emitbuf, eo) < 0) {
                        fprintf(stderr, "\nq4: reasoning buffer full at gen %d\n",
                                t);
                        return 2;
                    }
                } else {
                    if (*o + eo >= max_out) return 2;
                    memcpy(out + *o, emitbuf, (size_t)eo);
                    *o += eo;
                    out[*o] = 0;
                }
                if (stream_cb) stream_cb(emitbuf, kind, u);
                if (t == 0)
                    fprintf(stderr, "q4: first piece kind=%s [%s]\n",
                            kind == Q4_PIECE_REASONING ? "think" : "content",
                            emitbuf);
            }
        }
    }
    if (*in_think) (*think_toks)++;
    return 0;
}

static int client_gone(const q4_engine *e) {
    return e && e->cancel && *e->cancel;
}

static int prompt_cache_path(const char *path, char *out, size_t cap) {
    if (!path || !path[0] || !out || cap < 8) return -1;
    int n = snprintf(out, cap, "%s.prompt", path);
    if (n < 0 || (size_t)n >= cap) {
        if (cap) out[0] = 0;
        return -1;
    }
    return 0;
}

/* Multi-conversation session cache (Q4_SESS_SLOTS, default 4, 1 = the old
 * single file): slot 0 is Q4_SESS_CACHE itself, slot k > 0 <path>.<k>, each
 * with its main snapshot, .prompt checkpoint and .mtp sidecars.  A request
 * restores from whichever slot holds the longest exact prefix, splices the
 * newest slot of its conversation (first user message), and saves to its
 * conversation's slot - anything else (an OpenCode title or subagent
 * request, another session) to an empty or the least recently used slot, so
 * it no longer overwrites the main conversation's snapshot. */
#define SESS_SLOT_MAX 16
static int sess_slots(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_SESS_SLOTS");
        v = e && e[0] ? atoi(e) : 4;
        if (v < 1) v = 1;
        if (v > SESS_SLOT_MAX) v = SESS_SLOT_MAX;
    }
    return v;
}

static void slot_main(const char *base, int k, char *out, size_t cap) {
    if (k) snprintf(out, cap, "%s.%d", base, k);
    else snprintf(out, cap, "%s", base);
}

/* Newest mtime of a slot's main / .prompt file, 0 when it has neither. */
static double slot_mtime(const char *main_path) {
    char pp[4200];
    struct stat st;
    double t = 0;
    if (stat(main_path, &st) == 0) t = (double)st.st_mtim.tv_sec + st.st_mtim.tv_nsec * 1e-9;
    snprintf(pp, sizeof(pp), "%s.prompt", main_path);
    if (stat(pp, &st) == 0) {
        const double u = (double)st.st_mtim.tv_sec + st.st_mtim.tv_nsec * 1e-9;
        if (u > t) t = u;
    }
    return t;
}

static void slot_touch(const char *main_path) {
    char pp[4200];
    (void)utimensat(AT_FDCWD, main_path, NULL, 0);
    snprintf(pp, sizeof(pp), "%s.prompt", main_path);
    (void)utimensat(AT_FDCWD, pp, NULL, 0);
}

static void slot_drop(const char *main_path) {
    char pp[4200];
    (void)unlink(main_path);
    snprintf(pp, sizeof(pp), "%s.mtp", main_path);
    (void)unlink(pp);
    snprintf(pp, sizeof(pp), "%s.prompt", main_path);
    (void)unlink(pp);
    snprintf(pp, sizeof(pp), "%s.prompt.mtp", main_path);
    (void)unlink(pp);
}

/* Degenerate-loop stop: the last LOOP_WIN generated tokens repeat with a
 * period of at most LOOP_WIN / 4.  The check this replaces (the last 8
 * tokens equal to the 8 before them) cut real output and reported it as a
 * normal stop: a 9-column markdown table's |---|---| row, four equal
 * lines in a diff. */
#define LOOP_RING 512
#define LOOP_WIN 256
static int loop_period(const int32_t *r, int t) {
    if (t < LOOP_WIN) return 0;
    for (int P = 1; P <= LOOP_WIN / 4; P++) {
        int k = 0;
        while (k < LOOP_WIN - P &&
               r[(t - 1 - k) & (LOOP_RING - 1)] ==
                   r[(t - 1 - k - P) & (LOOP_RING - 1)])
            k++;
        if (k == LOOP_WIN - P) return P;
    }
    return 0;
}

/* FNV-1a over a snapshot's ids: ties the MTP head-KV file to it. */
static uint64_t ids_hash(const int32_t *ids, uint32_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t *b = (const uint8_t *)ids;
    for (size_t i = 0; i < (size_t)n * sizeof(int32_t); i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

/* The head KV next to a session snapshot: <snapshot>.mtp.
 * Q4_MTP_SNAP=0: neither written nor read (the head restarts empty). */
static int mtp_snap_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MTP_SNAP");
        v = !(e && e[0] == '0');
    }
    return v;
}
static void mtp_kv_save_beside(q4_engine *e, const char *snap,
                               const int32_t *ids, int n_ids) {
    char hp[4096 + 8];
    snprintf(hp, sizeof(hp), "%s.mtp", snap);
    const uint32_t nkv = q4_sess_n_kv(e->sess);
    /* An MTP turn cut by its token limit may have committed drafts it
     * never emitted (n_kv > n_ids): that snapshot cannot be restored. */
    if (!e->mtp || e->mtp_fail || !mtp_snap_on() || nkv > (uint32_t)n_ids ||
        !q4_mtp_kv_save(e->mtp, hp, nkv, ids_hash(ids, nkv)))
        (void)remove(hp);
}

/* Longest snapshot whose ids are an exact prefix of ids[0..n).
 * A mismatch inside the snapshot cannot be restored: GDN state is
 * only valid at the snapshot's end. */
static int cache_prefix_len(q4_sess *sess, const char *path, const int32_t *ids,
                            int n, int32_t *scratch, int scratch_n) {
    if (!sess || !path || !path[0] || !ids || n <= 0 || !scratch || scratch_n <= 0)
        return 0;
    int sn = q4_sess_peek_ids(sess, path, scratch, scratch_n, NULL, NULL, NULL);
    if (sn <= 0 || sn > n) return 0;
    if (memcmp(scratch, ids, (size_t)sn * sizeof(int32_t)) != 0) return 0;
    return sn;
}

/* Q4_NEXT_PROBE diagnostics: softmax top-k of the current logits plus the
 * probability of one watched token, decoded to text. */
static void probe_dump(q4_engine *e, const char *tag, int32_t watch) {
    const float *lg = q4_sess_logits(e->sess);
    const uint32_t nv = e->g.n_vocab;
    if (!lg || !nv) return;
    float mx = lg[0];
    for (uint32_t i = 1; i < nv; i++) if (lg[i] > mx) mx = lg[i];
    double z = 0.0;
    for (uint32_t i = 0; i < nv; i++) z += exp((double)lg[i] - mx);
    int32_t ti[8];
    float tv[8];
    int nt = 0;
    for (uint32_t i = 0; i < nv; i++) {
        const float v = lg[i];
        if (nt == 8 && v <= tv[7]) continue;
        int k = nt < 8 ? nt++ : 7;
        while (k > 0 && tv[k - 1] < v) {
            ti[k] = ti[k - 1];
            tv[k] = tv[k - 1];
            k--;
        }
        ti[k] = (int32_t)i;
        tv[k] = v;
    }
    fprintf(stderr, "q4 probe %s:", tag);
    for (int k = 0; k < nt; k++) {
        char buf[64];
        int m = q4_tok_decode(e->tok, &ti[k], 1, buf, (int)sizeof(buf) - 1);
        buf[m > 0 ? m : 0] = 0;
        for (char *c = buf; *c; c++) if (*c == '\n') *c = '|';
        fprintf(stderr, " [%s %.4f]", buf, exp((double)tv[k] - mx) / z);
    }
    if (watch >= 0 && (uint32_t)watch < nv)
        fprintf(stderr, "  P(watch %d)=%.4f", watch,
                exp((double)lg[watch] - mx) / z);
    fprintf(stderr, "\n");
}

/* Prefill ids[0..n) from an empty session the way a request does (no
 * cache, no head KV), then the lm head: q4_sess_logits holds the last row. */
static int probe_prefill(q4_engine *e, const int32_t *ids, int n) {
    q4_sess_reset(e->sess);
    engine_kv_fit(e, (uint32_t)n);
    const int chunk = (int)q4_pref_chunk();
    for (int i = 0; i < n;) {
        const int c = pref_split(n - i, chunk);
        if (!q4_sess_prefill(e->sess, ids + i, c)) return 0;
        i += c;
    }
    return q4_sess_lend_end(e->sess) && q4_sess_head(e->sess, NULL, NULL);
}

/* An image cell's cache key: a vocabulary-free id from the picture's hash
 * and the cell, so two pictures of one size never read as one prefix in
 * the session files (the model itself sees <|image_pad|> there). */
#define IMG_KEY_BASE 0x40000000
static int32_t img_key(uint64_t h, int32_t j) {
    uint64_t z = h + (uint64_t)(j + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (int32_t)(IMG_KEY_BASE | (uint32_t)(z >> 34));
}

/* Pictures seen by this server (and earlier ones: <Q4_SESS_CACHE>.imgs),
 * keyed by their first cell's key.  OpenCode replaces older pictures in a
 * long conversation with "[This image was removed ...]" text; a splice
 * then keeps cells of a picture the request no longer carries.  Those
 * cells are already in the restored KV and need no rows - only the
 * picture's extent, so the M-RoPE positions after it stay right. */
typedef struct { int32_t key0; int32_t n, nx, ny; uint64_t hash; } img_reg_ent;
#define IMG_REG_CAP 8192
static img_reg_ent g_img_reg[IMG_REG_CAP];
static int g_img_reg_loaded;

static img_reg_ent *img_reg_find(int32_t key0) {
    uint32_t h = (uint32_t)key0 * 2654435761u % IMG_REG_CAP;
    for (int i = 0; i < IMG_REG_CAP; i++, h = (h + 1) % IMG_REG_CAP) {
        if (g_img_reg[h].key0 == key0) return &g_img_reg[h];
        if (g_img_reg[h].key0 == 0) return NULL;
    }
    return NULL;
}

static int img_reg_put(int32_t key0, uint64_t hash, int32_t n, int32_t nx, int32_t ny) {
    uint32_t h = (uint32_t)key0 * 2654435761u % IMG_REG_CAP;
    for (int i = 0; i < IMG_REG_CAP; i++, h = (h + 1) % IMG_REG_CAP) {
        if (g_img_reg[h].key0 == key0) return 0;
        if (g_img_reg[h].key0 == 0) {
            g_img_reg[h] = (img_reg_ent){key0, n, nx, ny, hash};
            return 1;
        }
    }
    return 0; /* full: the picture just cannot be kept later */
}

static void img_reg_path(char *out, size_t cap) {
    const char *c = getenv("Q4_SESS_CACHE");
    if (c && c[0]) snprintf(out, cap, "%s.imgs", c);
    else out[0] = 0;
}

static void img_reg_load(void) {
    if (g_img_reg_loaded) return;
    g_img_reg_loaded = 1;
    char path[4200];
    img_reg_path(path, sizeof(path));
    FILE *f = path[0] ? fopen(path, "r") : NULL;
    if (!f) return;
    long long k0;
    unsigned long long hh;
    int n, nx, ny;
    while (fscanf(f, "%lld %llu %d %d %d", &k0, &hh, &n, &nx, &ny) == 5)
        if (n > 0 && nx > 0 && ny > 0 && n == nx * ny)
            (void)img_reg_put((int32_t)k0, (uint64_t)hh, n, nx, ny);
    fclose(f);
}

static void img_reg_note(const q4_image *im) {
    img_reg_load();
    const int32_t k0 = img_key(im->hash, 0);
    if (!img_reg_put(k0, im->hash, im->n, im->nx, im->ny)) return;
    char path[4200];
    img_reg_path(path, sizeof(path));
    FILE *f = path[0] ? fopen(path, "a") : NULL;
    if (!f) return;
    fprintf(f, "%lld %llu %d %d %d\n", (long long)k0, (unsigned long long)im->hash,
            im->n, im->nx, im->ny);
    fclose(f);
}

static int has_img_key(const int32_t *ids, int n) {
    for (int i = 0; i < n; i++)
        if (ids[i] >= IMG_KEY_BASE) return 1;
    return 0;
}

/* Binds the request's images to the prompt, in order.  The template
 * leaves one <|image_pad|> per picture: it expands to the picture's n
 * cells.  A run of image keys (cells a spliced snapshot kept) must be
 * that picture's keys.  ids becomes the cache-key array, e->img_rids the
 * ids the model reads (the pad id at image cells: the PLE hash takes it),
 * e->img_rows each cell's embedding row, e->img_tab the M-RoPE (t, h, w)
 * per cell (mtmd's: t = p, h = p + y, w = p + x; the text after an image
 * continues at p + max(nx, ny)); *shift is cell - position past the
 * prompt.  Returns the new length, -1 on a mismatch (msg), -2 when the
 * expansion does not fit cap. */
static int images_bind(q4_engine *e, int32_t *ids, int n, int cap,
                       int32_t *shift, char *msg, size_t msgn) {
    const int32_t pad = q4_tok_id(e->tok, "<|image_pad|>");
    const uint32_t n_embd = e->g.n_embd;
    if (pad < 0 && e->n_img > 0) {
        snprintf(msg, msgn, "this model has no <|image_pad|> token");
        return -1;
    }
    free(e->img_rids);
    free(e->img_rows);
    free(e->img_tab);
    e->img_rids = malloc((size_t)cap * sizeof(int32_t));
    e->img_rows = calloc((size_t)cap, sizeof(*e->img_rows));
    e->img_tab = malloc((size_t)cap * 3 * sizeof(int32_t));
    int32_t *src = malloc((size_t)n * sizeof(int32_t));
    if (!e->img_rids || !e->img_rows || !e->img_tab || !src) {
        free(src);
        snprintf(msg, msgn, "out of memory binding images");
        return -1;
    }
    memcpy(src, ids, (size_t)n * sizeof(int32_t));
    int o = 0, k = 0, rc = 0;
    int64_t p = 0;
    e->img_orphan_end = 0;
    for (int i = 0; i < e->n_img; i++) img_reg_note(&e->img[i]);
    for (int i = 0; i < n && rc == 0;) {
        const int32_t t = src[i];
        const int key = t >= IMG_KEY_BASE;
        if (t != pad && !key) {
            if (o >= cap) { rc = -2; break; }
            ids[o] = e->img_rids[o] = t;
            e->img_tab[o * 3] = e->img_tab[o * 3 + 1] = e->img_tab[o * 3 + 2] =
                (int32_t)p;
            p++, o++, i++;
            continue;
        }
        /* kept cells of a picture this request no longer carries (or
         * carries elsewhere): known extent, already in the snapshot */
        if (key && (k >= e->n_img || t != img_key(e->img[k].hash, 0))) {
            const img_reg_ent *r = img_reg_find(t);
            int ok = r != NULL && i + r->n <= n;
            for (int j = 0; ok && j < r->n; j++)
                ok = src[i + j] == img_key(r->hash, j);
            if (ok && o + r->n <= cap) {
                for (int j = 0; j < r->n; j++, o++) {
                    ids[o] = src[i + j];
                    e->img_rids[o] = pad;
                    e->img_rows[o] = NULL;
                    e->img_tab[o * 3] = (int32_t)p;
                    e->img_tab[o * 3 + 1] = (int32_t)(p + j / r->nx);
                    e->img_tab[o * 3 + 2] = (int32_t)(p + j % r->nx);
                }
                e->img_orphan_end = o;
                p += r->nx > r->ny ? r->nx : r->ny;
                i += r->n;
                continue;
            }
        }
        if (k >= e->n_img) {
            snprintf(msg, msgn, "the prompt has more image placeholders (%s) "
                     "than images (%d)", key ? "cached" : "<|image_pad|>",
                     e->n_img);
            rc = -1;
            break;
        }
        const q4_image *im = &e->img[k++];
        if (key)
            for (int j = 0; j < im->n; j++)
                if (i + j >= n || src[i + j] != img_key(im->hash, j)) {
                    snprintf(msg, msgn, "image %d does not match the cached "
                             "conversation", k);
                    rc = -1;
                    break;
                }
        if (rc) break;
        if (o + im->n > cap) { rc = -2; break; }
        for (int j = 0; j < im->n; j++, o++) {
            ids[o] = img_key(im->hash, j);
            e->img_rids[o] = pad;
            e->img_rows[o] = im->rows + (size_t)j * n_embd;
            e->img_tab[o * 3] = (int32_t)p;
            e->img_tab[o * 3 + 1] = (int32_t)(p + j / im->nx);
            e->img_tab[o * 3 + 2] = (int32_t)(p + j % im->nx);
        }
        p += im->nx > im->ny ? im->nx : im->ny;
        i += key ? im->n : 1;
    }
    free(src);
    if (rc == 0 && k != e->n_img) {
        snprintf(msg, msgn, "%d image(s) but %d placeholder(s) in the prompt",
                 e->n_img, k);
        rc = -1;
    }
    if (rc) return rc;
    *shift = (int32_t)(o - p);
    return o;
}

void q4_engine_set_images(q4_engine *e, const q4_image *imgs, int n) {
    if (!e) return;
    free(e->img);
    e->img = NULL;
    e->n_img = 0;
    if (!imgs || n <= 0) return;
    e->img = malloc((size_t)n * sizeof(*imgs));
    if (!e->img) return;
    memcpy(e->img, imgs, (size_t)n * sizeof(*imgs));
    e->n_img = n;
}

/* Q4_DUMP_OUT=<dir>: each request's messages and its generated tokens
 * decoded verbatim (think block, answer, raw <tool_call> XML) to
 * <dir>/req-<unix time>-<seq>.txt — what the model emitted, before the
 * server turns tool calls into JSON. */
static void dump_out(q4_engine *e, const char *const *roles,
                     const char *const *contents, int nmsg,
                     const int32_t *prompt, int n_prompt, const int32_t *gen,
                     int n_gen) {
    const char *dir = getenv("Q4_DUMP_OUT");
    if (!dir || !dir[0]) return;
    static unsigned seq;
    char path[4096];
    snprintf(path, sizeof(path), "%s/req-%ld-%04u.txt", dir, (long)time(NULL),
             seq++);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "### prompt %d tok, generated %d tok, temp %.2f top_k %d top_p %.2f\n",
            n_prompt, n_gen, e->temp, e->top_k, e->top_p);
    for (int i = 0; i < nmsg; i++)
        fprintf(f, "### message %d %s\n%s\n", i, roles && roles[i] ? roles[i] : "?",
                contents && contents[i] ? contents[i] : "");
    const int cap = n_gen * 64 + 16;
    char *txt = malloc((size_t)cap);
    if (txt) {
        q4_tok_decode(e->tok, gen, n_gen, txt, cap);
        fprintf(f, "### output\n%s\n", txt);
        free(txt);
    }
    /* the prompt ids the model read (a spliced turn keeps the saved ids;
     * image cells are their img_key), for diffs against a template render */
    if (gen && n_gen > 0) {
        fprintf(f, "### output ids\n");
        for (int i = 0; i < n_gen; i++)
            fprintf(f, "%d%c", gen[i], i + 1 < n_gen ? ' ' : '\n');
    }
    if (prompt) {
        fprintf(f, "### prompt ids\n");
        for (int i = 0; i < n_prompt; i++)
            fprintf(f, "%d%c", prompt[i], i + 1 < n_prompt ? ' ' : '\n');
    }
    fclose(f);
}

static int generate_locked(q4_engine *e, const char *const *roles,
                           const char *const *contents, int nmsg, char *out,
                           int max_out, int max_new,
                           void (*stream_cb)(const char *piece, int kind, void *u),
                           void *u) {
    if (!e || !e->sess || !e->tok || !out || max_out <= 1) return -1;
    /* A failed turn must not report the previous turn's usage. */
    e->prompt_tokens = e->completion_tokens = 0;
    e->hit_limit = 0;
    e->err_code = Q4_ERR_NONE;
    e->err[0] = 0;
    if (client_gone(e)) return -1;
    if (max_new <= 0) max_new = 256;
    int32_t *ids = malloc((size_t)e->ctx * sizeof(int32_t));
    if (!ids) return -1;
    const char *cache_base = getenv("Q4_SESS_CACHE");
    const int nslot = cache_base && cache_base[0] && !e->render_only ? sess_slots() : 0;
    /* cache_path / prompt_path: the slot this request saves to (decided
     * after the restore); the scan below fills the per-slot state */
    char slot_path[SESS_SLOT_MAX][4096];
    int slot_n[SESS_SLOT_MAX], slot_nmsg[SESS_SLOT_MAX];
    uint64_t slot_uh[SESS_SLOT_MAX];
    double slot_t[SESS_SLOT_MAX];
    char cache_buf[4096];
    const char *cache_path = NULL;
    char prompt_path[4096];
    prompt_path[0] = 0;
    cache_buf[0] = 0;
    int n = 0;
    int spliced = 0, splice_slot = -1;
    uint64_t uh = 0;
    int cache_seen = 0, cache_nmsg = 0, cache_user_ok = 0;
    int slot_same[SESS_SLOT_MAX] = {0};
    if (nslot > 0) {
        int32_t *saved = malloc((size_t)e->ctx * sizeof(int32_t));
        uh = first_user_hash(roles, contents, nmsg);
        /* "the same conversation": the first user message matches AND the
         * snapshot shares at least half of this request's render up to that
         * message.  OpenCode's title request repeats the session's first
         * user message under its own short system prompt; keyed by the
         * message alone, the next agent turn spliced onto the title
         * snapshot (system prompt and tools gone).  OpenCode rewrites the
         * tail of its system text every turn, so the test is a long common
         * prefix, not equality. */
        int fu = -1;
        for (int i = 0; i < nmsg && fu < 0; i++)
            if (roles && roles[i] && strcmp(roles[i], "user") == 0) fu = i;
        int32_t *conv = fu >= 0 ? malloc((size_t)e->ctx * sizeof(int32_t)) : NULL;
        int conv_n = conv ? q4_tok_apply_messages(e->tok, roles, contents, fu + 1,
                                                  conv, (int)e->ctx)
                          : 0;
        if (conv_n < 0) conv_n = 0;
        int best = -1;
        for (int k = 0; k < nslot && saved; k++) {
            slot_main(cache_base, k, slot_path[k], sizeof(slot_path[k]));
            slot_t[k] = slot_mtime(slot_path[k]);
            int sp = 0;
            slot_n[k] = q4_sess_peek_ids(e->sess, slot_path[k], saved, (int)e->ctx,
                                         &sp, &slot_nmsg[k], &slot_uh[k]);
            int peek_n = slot_n[k];
            if (slot_n[k] <= 0) {
                /* a slot with only its .prompt: identify it by that */
                char pp[4200];
                snprintf(pp, sizeof(pp), "%s.prompt", slot_path[k]);
                peek_n = q4_sess_peek_ids(e->sess, pp, saved, (int)e->ctx, &sp,
                                          &slot_nmsg[k], &slot_uh[k]);
                slot_n[k] = 0; /* nothing to splice */
                if (peek_n <= 0) slot_uh[k] = 0, slot_nmsg[k] = 0;
            }
            if (peek_n > 0 && uh && slot_uh[k] == uh) {
                int l = 0;
                const int m = peek_n < conv_n ? peek_n : conv_n;
                while (l < m && saved[l] == conv[l]) l++;
                slot_same[k] = conv_n <= 16 || 2 * l >= conv_n;
            }
            if (slot_n[k] > 0) {
                cache_seen = 1;
                if (uh && slot_uh[k] == uh) cache_user_ok = 1;
                cache_nmsg = slot_nmsg[k];
            }
            if (slot_n[k] > 0 && slot_nmsg[k] > 0 && nmsg > slot_nmsg[k] &&
                slot_n[k] < (int)e->ctx - 64 && slot_same[k] &&
                (best < 0 || slot_t[k] > slot_t[best]))
                best = k;
        }
        free(conv);
        int saved_n = -1, saved_nmsg = 0;
        uint64_t saved_uh = 0;
        if (best >= 0) {
            int sp = 0;
            saved_n = q4_sess_peek_ids(e->sess, slot_path[best], saved, (int)e->ctx,
                                       &sp, &saved_nmsg, &saved_uh);
            splice_slot = best;
        }
        /* OpenCode rewrites the system prompt every turn (files, clock,
         * tools). GDN cannot skip a changed prefix, but we still splice
         * the saved ids — the conversation is identified by the first
         * user message, not the system tokens. A resend with the same
         * message count is not a splice; the prompt snapshot covers it. */
        if (saved_n > 0 && saved_nmsg > 0 && nmsg > saved_nmsg &&
            saved_nmsg <= nmsg && saved_n < (int)e->ctx - 64 && uh &&
            uh == saved_uh) {
            memcpy(ids, saved, (size_t)saved_n * sizeof(int32_t));
            n = saved_n;
            int32_t im_end = q4_tok_im_end(e->tok);
            if (im_end >= 0 && ids[n - 1] != im_end && n + 8 < (int)e->ctx)
                ids[n++] = im_end;
            n += q4_tok_encode(e->tok, "\n", ids + n, (int)e->ctx - n);
            int from = saved_nmsg;
            if (from < nmsg && roles[from] &&
                strcmp(roles[from], "assistant") == 0)
                from++;
            n = q4_tok_apply_messages_from(e->tok, roles, contents, nmsg, from,
                                           ids, (int)e->ctx, n);
            spliced = 1;
            fprintf(stderr,
                    "q4: session splice kept %d tok + %d new  (msg %d..%d, slot %d)\n",
                    saved_n, n - saved_n, from, nmsg, splice_slot);
        }
        free(saved);
    }
render_again:
    if (!spliced)
        n = q4_tok_apply_messages(e->tok, roles, contents, nmsg, ids, (int)e->ctx);
    /* Images: expand and key the pad cells; the model reads rids. */
    const int32_t *rids = ids;
    int32_t mshift = 0;
    int with_img = 0;
    e->img_orphan_end = 0;
    if (n > 0 && (e->n_img > 0 || has_img_key(ids, n))) {
        char bm[256];
        int nb = images_bind(e, ids, n, (int)e->ctx, &mshift, bm, sizeof(bm));
        if (nb == -1 && spliced) {
            /* the kept ids hold other pictures: render the whole thing */
            fprintf(stderr, "q4: session splice dropped (%s)\n", bm);
            spliced = 0;
            n = q4_tok_apply_messages(e->tok, roles, contents, nmsg, ids,
                                      (int)e->ctx);
            nb = n <= 0 ? -1
                 : e->n_img > 0 ? images_bind(e, ids, n, (int)e->ctx, &mshift,
                                              bm, sizeof(bm))
                                : n;
        }
        if (nb == -1) {
            fprintf(stderr, "q4: images: %s\n", bm);
            e->err_code = Q4_ERR_REQUEST;
            snprintf(e->err, sizeof(e->err), "%s", bm);
            free(ids);
            return -1;
        }
        n = nb == -2 ? (int)e->ctx : nb;   /* -2: reported as too long below */
        if (e->img_orphan_end > 0)
            fprintf(stderr, "q4: session splice keeps the cells of pictures the "
                    "request no longer carries (up to cell %d)\n", e->img_orphan_end);
        /* kept cells of dropped pictures still move the M-RoPE positions */
        with_img = nb > 0 && (e->n_img > 0 || e->img_orphan_end > 0);
        if (with_img) rids = e->img_rids;
    }
    if (e->render_only) {
        /* the request's own render (no session splice), ids to
         * Q4_DUMP_OUT, no model work: diffs against a reference template */
        e->render_only = 0;
        e->prompt_tokens = n;
        const char *dir = getenv("Q4_DUMP_OUT");
        static unsigned rseq;
        if (dir && dir[0] && n > 0) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/render-%04u.ids", dir, rseq++);
            FILE *f = fopen(path, "w");
            if (f) {
                for (int i = 0; i < n; i++)
                    fprintf(f, "%d%c", ids[i], i + 1 < n ? ' ' : '\n');
                fclose(f);
            }
        }
        fprintf(stderr, "q4: render only, %d prompt tokens\n", n);
        free(ids);
        out[0] = 0;
        return 0;
    }
    if (n <= 0 || n + 8 >= (int)q4_sess_max_kv(e->sess)) {
        fprintf(stderr, "q4: prompt tokens %d do not fit ctx %u\n", n,
                q4_sess_max_kv(e->sess));
        /* the wording clients match to compact (OpenAI's) */
        e->err_code = Q4_ERR_CONTEXT;
        if (n > 0)
            snprintf(e->err, sizeof(e->err),
                     "This model's maximum context length is %u tokens. "
                     "However, your messages resulted in %d tokens.",
                     q4_sess_max_kv(e->sess) - 8, n);
        else
            snprintf(e->err, sizeof(e->err),
                     "This model's maximum context length is %u tokens. "
                     "However, your messages resulted in more tokens.",
                     q4_sess_max_kv(e->sess) - 8);
        free(ids);
        return -1;
    }
    {
        int room = (int)q4_sess_max_kv(e->sess) - n - 8;
        if (room < 1) room = 1;
        if (max_new > room) max_new = room;
    }
    q4_sess_reset(e->sess);
    if (with_img) {
        if (!q4_hip_mrope_set(e->img_tab, (uint32_t)n, mshift)) {
            e->err_code = Q4_ERR_INTERNAL;
            snprintf(e->err, sizeof(e->err), "internal error: image positions");
            free(ids);
            return -1;
        }
        e->mrope_on = 1;
        q4_sess_set_embd_rows(e->sess, e->img_rows, (uint32_t)n);
    } else if (e->mrope_on) {
        (void)q4_hip_mrope_set(NULL, 0, 0);
        e->mrope_on = 0;
    }
    engine_kv_fit(e, (uint32_t)n);
    q4_mtp_reset(e->mtp);
    /* A cycle failure drops one request to plain decode, not the process:
     * the head restarts from an empty KV with the session. */
    if (e->mtp && e->mtp_fail) {
        fprintf(stderr, "q4: MTP back on after the previous request's fallback\n");
        e->mtp_fail = 0;
    }
    q4_expert_clear_sticky(e->ex);
    e->prompt_tokens = n;
    e->completion_tokens = 0;
    e->hit_limit = 0;
    e->reason_n = 0;
    if (e->reason) e->reason[0] = 0;
    /* Prefix cache. The main file is the last finished turn (prompt plus
     * generated tokens). The .prompt file is the prompt boundary, written
     * when prefill finishes or the client disconnects, so a retry of the
     * same body resumes instead of recomputing. The longer exact prefix
     * wins. A longer snapshot cannot be rewound. */
    int start = 0;
    int hit_side = 0;
    int restore_slot = -1;
    const char *restore_src = NULL;
    if (nslot > 0) {
        int32_t *scratch = malloc((size_t)e->ctx * sizeof(int32_t));
        /* candidates: every slot's main and .prompt, longest prefix first
         * (a full-prompt .prompt beats an equal main) */
        int cl[2 * SESS_SLOT_MAX], ck[2 * SESS_SLOT_MAX], cs[2 * SESS_SLOT_MAX];
        int nc = 0;
        static char cpath[2 * SESS_SLOT_MAX][4200];
        for (int k = 0; k < nslot && scratch; k++)
            for (int side = 0; side < 2; side++) {
                if (side)
                    snprintf(cpath[nc], sizeof(cpath[nc]), "%s.prompt", slot_path[k]);
                else
                    snprintf(cpath[nc], sizeof(cpath[nc]), "%s", slot_path[k]);
                const int len = cache_prefix_len(e->sess, cpath[nc], ids, n, scratch,
                                                 (int)e->ctx);
                if (len <= 0) continue;
                cl[nc] = len;
                ck[nc] = k;
                cs[nc] = side;
                nc++;
            }
        free(scratch);
        int used[2 * SESS_SLOT_MAX] = {0};
        double tc = now_s();
        for (int tries = 0; tries < nc && start <= 0; tries++) {
            int b = -1;
            for (int i = 0; i < nc; i++)
                if (!used[i] &&
                    (b < 0 || cl[i] > cl[b] ||
                     (cl[i] == cl[b] && cs[i] > cs[b]) ||
                     (cl[i] == cl[b] && cs[i] == cs[b] && slot_t[ck[i]] > slot_t[ck[b]])))
                    b = i;
            used[b] = 1;
            start = q4_sess_restore(e->sess, cpath[b], ids, n);
            if (start > 0) {
                restore_slot = ck[b];
                restore_src = cpath[b];
                hit_side = cs[b];
            }
        }
        if (start > 0) {
            fprintf(stderr, "q4: session cache hit%s %d/%d tok (slot %d, %.2f s)\n",
                    hit_side ? " prompt" : "", start, n, restore_slot, now_s() - tc);
            slot_touch(slot_path[restore_slot]);
            char hp[4200 + 8];
            snprintf(hp, sizeof(hp), "%s.mtp", restore_src);
            if (e->mtp && mtp_snap_on() &&
                q4_mtp_kv_load(e->mtp, hp, (uint32_t)start,
                               ids_hash(ids, (uint32_t)start))) {
                const bool br =
                    start >= n || q4_mtp_kv_bridge(e->mtp, e->sess, rids[start]);
                fprintf(stderr, "q4: mtp head KV restored%s\n",
                        br ? "" : " (bridge failed)");
            }
        }
        /* a splice that kept cells of pictures the request dropped has no
         * rows for them: only a restore past them is valid - else the
         * whole render, as before */
        if (spliced && e->img_orphan_end > start) {
            fprintf(stderr, "q4: session splice dropped (restored %d of %d kept "
                    "cells of removed pictures)\n", start, e->img_orphan_end);
            spliced = 0;
            splice_slot = -1;
            start = 0;
            hit_side = 0;
            q4_sess_reset(e->sess);
            goto render_again;
        }
        /* save slot: the conversation's own (the spliced one, the one we
         * restored if it is this conversation, or its newest slot that is
         * not ahead of this request); else an empty slot, else the least
         * recently used that is neither the restore source nor a longer
         * turn of this conversation (an OpenCode title request shares the
         * first user message with the session it names) */
        int sv = -1;
        if (spliced) sv = splice_slot;
        if (sv < 0 && restore_slot >= 0 && slot_same[restore_slot] &&
            slot_nmsg[restore_slot] <= nmsg)
            sv = restore_slot;
        if (sv < 0) {
            int nw = -1;
            for (int k = 0; k < nslot; k++)
                if (slot_same[k] && slot_nmsg[k] <= nmsg &&
                    (nw < 0 || slot_t[k] > slot_t[nw]))
                    nw = k;
            sv = nw;
        }
        for (int k = 0; k < nslot && sv < 0; k++)
            if (slot_t[k] == 0) sv = k;
        for (int pass = 0; pass < 2 && sv < 0; pass++)
            for (int k = 0; k < nslot; k++) {
                if (pass == 0 && (k == restore_slot || slot_same[k] ||
                                  (uh && slot_uh[k] == uh)))
                    continue;
                if (sv < 0 || slot_t[k] < slot_t[sv]) sv = k;
            }
        if (sv < 0) sv = 0;
        snprintf(cache_buf, sizeof(cache_buf), "%s", slot_path[sv]);
        cache_path = cache_buf;
        prompt_cache_path(cache_path, prompt_path, sizeof(prompt_path));
        const int same_conv = slot_same[sv];
        if (slot_t[sv] > 0 && !same_conv && sv != restore_slot) {
            /* evicting another conversation: no stale main / sidecars */
            slot_drop(cache_path);
            fprintf(stderr, "q4: session slot %d reused for this conversation\n", sv);
        }
        /* the .prompt file this request may write is the restore source */
        if (restore_src && strcmp(restore_src, prompt_path) != 0) hit_side = 0;
        if (start <= 0 && cache_seen)
            fprintf(stderr,
                    "q4: session cache miss (nmsg %d/%d user %s)\n", cache_nmsg,
                    nmsg, cache_user_ok ? "match" : "mismatch");
    }
    {
        FILE *st = fopen("/proc/self/status", "r");
        if (st) {
            char line[128];
            while (fgets(line, sizeof(line), st)) {
                unsigned long sw = 0;
                if (sscanf(line, "VmSwap: %lu", &sw) == 1 && sw > 256ull * 1024ull) {
                    fprintf(stderr,
                            "q4: WARNING self VmSwap %.1f GiB — L2 is in swap, "
                            "decode will collapse. Close browsers / do not delete "
                            "large files while q4 runs.\n",
                            (double)sw / (1024.0 * 1024.0));
                    break;
                }
            }
            fclose(st);
        }
    }
    fprintf(stderr, "q4: generate prompt=%d max_new=%d (layer-first prefill)\n", n,
            max_new);
    fflush(stderr);
    double t0 = now_s();
    const int chunk = (int)q4_pref_chunk();
    e->pref_base = 0;
    e->pref_chunk = chunk;
    e->pref_total = n;
    q4_sess_set_pref_hook(e->sess, pref_layer_hook, e);
    int filled = start;
    int ckpt_at = 0;
    int ckpt_fail = 0;
    int nan_stop = 0;
    double ckpt_dt = 0;
    for (int i = start; i < n && !q4_stop;) {
        if (client_gone(e)) break;
        int c = pref_split(n - i, chunk);
        e->pref_base = i;
        e->pref_chunk = c;
        if (i + c < n) {
            int cn = pref_split(n - i - c, chunk);
            (void)q4_sess_ple_ahead(e->sess, rids + i + c, cn);
        }
        if (!q4_sess_prefill(e->sess, rids + i, c)) {
            q4_sess_set_pref_hook(e->sess, NULL, NULL);
            (void)q4_sess_lend_end(e->sess);
            fprintf(stderr, "q4: prefill failed at %d / %d  id=%d\n", i, n,
                    ids[i]);
            e->err_code = Q4_ERR_INTERNAL;
            snprintf(e->err, sizeof(e->err), "internal error: prefill failed");
            free(ids);
            return -1;
        }
        if (e->mtp && !e->mtp_fail &&
            !q4_mtp_prefill_chunk(e->mtp, e->sess, rids + i, c,
                                  i + c < n ? rids[i + c] : -1)) {
            fprintf(stderr, "q4: MTP head prefill failed, disabling MTP\n");
            q4_mtp_close(e->mtp);
            e->mtp = NULL;
        }
        i += c;
        filled = i;
        fprintf(stderr, "\rprefill %d/%d", i, n);
        if (getenv("Q4_PROFILE"))
            fprintf(stderr, "  (chunk %.2f s  io %.2f)", now_s() - t0,
                    q4_prof_moe_io);
        fflush(stderr);
        if (e->progress) e->progress(i, n, 0, e->progress_u);
        /* One snapshot of tokens [0, i). That is enough to resume: GDN
         * state is valid at this boundary. Skip the intermediate chunks
         * while the client is still connected. The main file still holds
         * the previous finished turn until decode ends. */
        if (prompt_path[0] && (i == n || client_gone(e)) &&
            (uint32_t)i == q4_sess_n_kv(e->sess)) {
            double tc = now_s();
            if (q4_sess_save(e->sess, prompt_path, ids, i, i, nmsg, uh)) {
                mtp_kv_save_beside(e, prompt_path, ids, i);
                ckpt_at = i;
                ckpt_dt = now_s() - tc;
            } else if (!ckpt_fail) {
                fprintf(stderr, "\nq4: prompt checkpoint failed at %d\n", i);
                ckpt_fail = 1;
            }
        }
        if (client_gone(e)) break;
    }
    q4_sess_set_pref_hook(e->sess, NULL, NULL);
    /* Prompt done (head KV included): hand the arena tail back to the
     * experts.  Inside the prefill timer — the refill is part of TTFT. */
    if (!q4_sess_lend_end(e->sess)) {
        fprintf(stderr, "\nq4: L1 lend return failed\n");
        e->err_code = Q4_ERR_INTERNAL;
        snprintf(e->err, sizeof(e->err), "internal error: L1 lend return failed");
        free(ids);
        return -1;
    }
    if (client_gone(e)) {
        fprintf(stderr, "\nq4: client closed at prefill %d/%d\n", filled, n);
        if (ckpt_at > 0)
            fprintf(stderr, "q4: prompt checkpoint %d tok (%.2f s)\n", ckpt_at,
                    ckpt_dt);
        fflush(stderr);
        free(ids);
        return -1;
    }
    /* A full-cache hit never entered the loop, so .prompt would stay
     * older than the prompt. Write it before decode appends tokens to
     * the main file and makes a same-body retry miss. */
    if (prompt_path[0] && start == n && !hit_side &&
        (uint32_t)n == q4_sess_n_kv(e->sess)) {
        double tc = now_s();
        if (q4_sess_save(e->sess, prompt_path, ids, n, n, nmsg, uh)) {
            mtp_kv_save_beside(e, prompt_path, ids, n);
            ckpt_at = n;
            ckpt_dt = now_s() - tc;
        } else if (!ckpt_fail) {
            fprintf(stderr, "q4: prompt checkpoint failed at %d\n", n);
            ckpt_fail = 1;
        }
    }
    if (ckpt_at == n && ckpt_at > 0)
        fprintf(stderr, "q4: prompt checkpoint %d tok (%.2f s)\n", ckpt_at,
                ckpt_dt);
    if (!q4_sess_head(e->sess, NULL, NULL)) {
        fprintf(stderr, "\nq4: lm-head after prefill failed\n");
        e->err_code = Q4_ERR_INTERNAL;
        snprintf(e->err, sizeof(e->err), "internal error: lm-head after prefill failed");
        free(ids);
        return -1;
    }
    double dt = now_s() - t0;
    {
        int did = n - start;
        fprintf(stderr, "\rprefill %d+%d/%d  %.2f t/s                \n", start, did,
                n, dt > 0 && did > 0 ? did / dt : 0);
    }
    if (getenv("Q4_PROFILE")) {
        extern double q4_prof_sec[8];
        extern double q4_prof_moe_cw;
        fprintf(stderr,
                "q4 prof prefill: moe io %.2f (copy_wait %.2f)  d2h %.2f  "
                "gemm %.2f s\n"
                "q4 prof sections: ple %.2f hc_attn %.2f qsa %.2f comb %.2f "
                "hc_ffn %.2f moe %.2f comb2 %.2f gdn %.2f s\n",
                q4_prof_moe_io, q4_prof_moe_cw, q4_prof_moe_d2h, q4_prof_moe_gemm,
                q4_prof_sec[0], q4_prof_sec[1], q4_prof_sec[2], q4_prof_sec[3],
                q4_prof_sec[4], q4_prof_sec[5], q4_prof_sec[6], q4_prof_sec[7]);
    }
    fflush(stderr);
    if (e->progress) e->progress(n, n, 0, e->progress_u);

    if (getenv("Q4_NEXT_PROBE")) {
        /* Diagnostic (the request then ends): the distribution after a
         * forced first token (Q4_NEXT_PROBE_TOK, default "提示"; watched
         * token Q4_NEXT_PROBE_WATCH, default "词"), reached by (b) prefill
         * + decode, (a) one prefill of prompt + token, (c) 4- and 2-row
         * verify batches (tail top-k vs the float one-row head), and with
         * Q4_NEXT_PROBE_MTP=K (d) K fresh MTP cycles seeded with the token
         * (Q4_NEXT_PROBE_NODRAFT=1: head prefill, then a verify without a
         * draft).  Q4_NEXT_PROBE_IDS=<file> appends the prompt ids.  Found
         * the JSON / trim prompt bugs and the n-dependent int8 GEMVs. */
        int32_t ft[8], wt[8];
        const char *fs = getenv("Q4_NEXT_PROBE_TOK");
        const char *ws = getenv("Q4_NEXT_PROBE_WATCH");
        const int nf = q4_tok_encode(e->tok, fs && fs[0] ? fs : "提示", ft, 8);
        const int nw = q4_tok_encode(e->tok, ws && ws[0] ? ws : "词", wt, 8);
        const int32_t f0 = nf > 0 ? ft[0] : -1, w0 = nw > 0 ? wt[0] : -1;
        fprintf(stderr, "q4 probe: prompt %d tok, forced %d, watch %d\n", n,
                f0, w0);
        const char *idp = getenv("Q4_NEXT_PROBE_IDS");
        if (idp && idp[0]) {
            FILE *fi = fopen(idp, "a");
            if (fi) {
                for (int k = 0; k < n; k++) fprintf(fi, "%d%c", ids[k], k + 1 < n ? ' ' : '\n');
                fclose(fi);
            }
        }
        probe_dump(e, "first", f0);
        int32_t nx = 0;
        float nl = 0.f;
        if (f0 >= 0 && q4_sess_decode(e->sess, f0, &nx, &nl))
            probe_dump(e, "b:decode", w0);
        ids[n] = f0;
        if (f0 >= 0 && probe_prefill(e, ids, n + 1))
            probe_dump(e, "a:prefill", w0);
        /* (c) the verify batch [f, x, x, x]: row 0 must not depend on x */
        int32_t at[8];
        const int na = q4_tok_encode(e->tok, "された", at, 8);
        const int32_t xs[3] = {w0, na > 0 ? at[0] : -1, na > 0 ? at[0] : -1};
        for (int v = 0; v < 3 && f0 >= 0 && xs[v] >= 0; v++) {
            const int32_t batch[4] = {f0, xs[v], xs[v], xs[v]};
            const int nbt = v == 2 ? 2 : 4;   /* v 2: the 2-row verify */
            char tag[32];
            snprintf(tag, sizeof(tag), "c:verify%d+%d row0", nbt, xs[v]);
            if (probe_prefill(e, ids, n) &&
                q4_sess_prefill_tokens(e->sess, batch, nbt)) {
                /* the verify tail's batched head + device top-k (what the
                 * MTP acceptance reads) vs the one-row float head */
                if (q4_sess_verify_topk(e->sess, nbt, 20)) {
                    int32_t ti[20], di[20];
                    float tv[20], dp[20];
                    q4_hip_topk_rows_read(0, ti, tv, 20);
                    const int k = q4_dist_from_topk(ti, tv, 20, 1.f, 1.f, di, dp);
                    fprintf(stderr, "q4 probe %s vtail: top %d logit %.4f",
                            tag, ti[0], tv[0]);
                    for (int j = 0; j < k; j++)
                        if (di[j] == w0) fprintf(stderr, "  P(watch)=%.4f", dp[j]);
                    fprintf(stderr, "\n");
                }
                if (q4_sess_logits_at(e->sess, 0)) {
                    const float *lg = q4_sess_logits(e->sess);
                    float mx = lg[0];
                    int32_t am = 0;
                    for (uint32_t j = 1; j < e->g.n_vocab; j++)
                        if (lg[j] > mx) { mx = lg[j]; am = (int32_t)j; }
                    fprintf(stderr, "q4 probe %s float: top %d logit %.4f\n",
                            tag, am, mx);
                    probe_dump(e, tag, w0);
                }
            }
        }
        /* (d) Q4_NEXT_PROBE_MTP=K: K fresh MTP cycles seeded with the
         * forced token on the same prompt; the second output token's
         * frequency of the watched token vs the target P above. */
        const char *km = getenv("Q4_NEXT_PROBE_MTP");
        if (km && e->mtp && f0 >= 0) {
            const int K = atoi(km);
            int nd_env = 3;
            if (getenv("Q4_MTP_N")) nd_env = atoi(getenv("Q4_MTP_N"));
            int hit = 0, tot = 0, prop = 0;
            for (int k = 0; k < K; k++) {
                q4_sess_reset(e->sess);
                engine_kv_fit(e, (uint32_t)n);
                q4_mtp_reset(e->mtp);
                const int chunk = (int)q4_pref_chunk();
                int ok = 1;
                for (int i = 0; i < n && ok;) {
                    const int c = pref_split(n - i, chunk);
                    ok = q4_sess_prefill(e->sess, ids + i, c) &&
                         q4_mtp_prefill_chunk(e->mtp, e->sess, ids + i, c,
                                              i + c < n ? ids[i + c] : -1);
                    i += c;
                }
                if (!ok || !q4_sess_lend_end(e->sess) ||
                    !q4_sess_head(e->sess, NULL, NULL))
                    break;
                if (getenv("Q4_NEXT_PROBE_NODRAFT")) {
                    /* head prefill ran, no draft: the verify row 0 alone */
                    const int32_t vb[4] = {f0, xs[1], xs[1], xs[1]};
                    if (q4_sess_prefill_tokens(e->sess, vb, 4) &&
                        q4_sess_logits_at(e->sess, 0))
                        probe_dump(e, "e:headpf+verify row0", w0);
                    continue;
                }
                int32_t dr[8];
                int ndo = 0;
                int32_t nxt = 0;
                const float *nl = NULL;
                const int crc = q4_mtp_cycle(e->mtp, e->sess, f0, nd_env,
                                             e->temp, e->top_k, e->top_p,
                                             &e->rng, dr, &ndo, &nxt, &nl);
                if (crc != 0) continue;
                const int32_t second = ndo > 0 ? dr[0] : nxt;
                tot++;
                hit += second == w0;
                prop += ndo > 0;
            }
            fprintf(stderr, "q4 probe d:mtp  second==watch %d / %d  (draft[0] "
                    "accepted %d)\n", hit, tot, prop);
        }
        free(ids);
        return -1;
    }
    const float *logits = q4_sess_logits(e->sess);
    if (!logits) {
        free(ids);
        return -1;
    }
    {
        /* Q4_DUMP_FIRST_LOGITS=<path>: the raw logits of each request's
         * last prompt position to <path>.<n> (n_vocab floats) — Strata's
         * STRATA_DUMP_FIRST_LOGITS, so the engines can be compared by KL
         * on the same prompts. */
        static unsigned fl_n;
        const char *fl = getenv("Q4_DUMP_FIRST_LOGITS");
        if (fl && fl[0]) {
            char path[4096];
            snprintf(path, sizeof(path), "%s.%u", fl, fl_n++);
            FILE *f = fopen(path, "wb");
            if (!f || fwrite(logits, sizeof(float), e->g.n_vocab, f) !=
                          e->g.n_vocab)
                fprintf(stderr, "q4: Q4_DUMP_FIRST_LOGITS: cannot write %s\n",
                        path);
            if (f) fclose(f);
        }
    }
    int o = 0;
    out[0] = 0;
    int32_t im_end = q4_tok_im_end(e->tok);
    int32_t eos = q4_tok_eos(e->tok);
    int32_t th = q4_tok_id(e->tok, "<think>");
    int32_t te = q4_tok_id(e->tok, "</think>");
    /* Prompt already opened <think> when thinking is on; generation starts
     * inside reasoning. Tags themselves must not go into the answer body. */
    int in_think = q4_tok_think_enabled();
    int think_toks = 0;
    int32_t recent[LOOP_RING];
    unsigned char hold[8];
    int nh = 0;
    int32_t *gen_ids = cache_path && cache_path[0]
                           ? malloc((size_t)max_new * sizeof(int32_t))
                           : NULL;
    if (e->mtp) q4_mtp_set_stops(e->mtp, im_end, eos);
    int n_gen = 0;
    q4_pen_begin(e->temp > 1e-5f ? e->presence : 0.f, e->g.n_vocab);
    t0 = now_s();
    int t = 0;
    int n_draft = 2;
    {
        const char *dn = getenv("Q4_MTP_N");
        if (dn && dn[0]) {
            int v = atoi(dn);
            if (v >= 1) n_draft = v;
        }
    }
    int mtp_mode = e->mtp && !e->mtp_fail;
    int32_t c0 = 0;
    if (mtp_mode) {
        /* Seed the chain when the whole prompt came from the session cache
         * and no prefill chunk ran this turn. */
        if (start >= n) {
            if (n <= 0 || !q4_mtp_seed(e->mtp, e->sess, rids[n - 1])) {
                fprintf(stderr, "q4: MTP seed failed, plain decode\n");
                mtp_mode = 0;
                e->mtp_fail = 1;
            }
        }
    }
    if (mtp_mode)
        c0 = q4_sample(logits, e->g.n_vocab, e->temp, e->top_k, e->top_p,
                       &e->rng);
    int stop = 0;
    int closed = 0;
    uint32_t repin_every = 64;
    {
        const char *re = getenv("Q4_REPIN_EVERY");
        if (re && re[0]) {
            long v = strtol(re, NULL, 10);
            repin_every = v < 0 ? 0 : (uint32_t)v;
        }
    }
    while (t < max_new && !q4_stop && !stop && !client_gone(e)) {
        q4_expert_reclaim_check(e->ex, "decode", t);
        if (isnan(logits[0]) || isinf(logits[0]) ||
            isnan(logits[e->g.n_vocab / 2]) || isinf(logits[e->g.n_vocab / 2]) ||
            isnan(logits[e->g.n_vocab - 1]) || isinf(logits[e->g.n_vocab - 1])) {
            fprintf(stderr, "\nq4: NaN logits at gen %d, stop\n", t);
            /* what is broken: the L1 slots' bytes and the device table
             * the verify path reads (both sync; this request is lost) */
            q4_expert_reclaim_check(e->ex, "nan", -t - 1);
            /* slots whose bytes differ from the DRAM image are re-copied,
             * so later requests do not inherit them */
            q4_expert_slot_repair(e->ex, "nan");
            q4_expert_devtab_check(e->ex, "nan");
            q4_nan_trap = 1;
            nan_stop = 1;
            break;
        }
        int32_t id;
        if (mtp_mode)
            id = c0;
        else
            id = q4_sample(logits, e->g.n_vocab, e->temp, e->top_k, e->top_p,
                           &e->rng);
        int rc = emit_piece(e, id, out, &o, max_out, &in_think, &think_toks,
                            hold, &nh, stream_cb, u, t, im_end, eos, th, te);
        if (rc == 1) break;
        if (rc == 2) {
            e->hit_limit = 1;
            break;
        }
        t++;
        e->completion_tokens++;
        /* Adaptive re-pin: promote recurrent hot experts so the LRU tail
         * stops thrashing on working sets larger than L1. O(n_slots).
         * Skipped in static mode — S already holds the hot experts, and
         * accumulated pins would strangle the smaller dynamic pool. */
        if (repin_every && e->ex && !q4_expert_static_on(e->ex) &&
            (t % repin_every) == 0)
            (void)q4_expert_pin_thresh(e->ex, warm_pin_cap(e), 8);
        if (gen_ids) gen_ids[n_gen++] = id;
        q4_pen_push(id);
        recent[(t - 1) & (LOOP_RING - 1)] = id;
        if (t % 4 == 0 || t == 1) {
            fprintf(stderr, "\rdecode %d/%d", t, max_new);
            fflush(stderr);
            if (e->progress) e->progress(n, n, t, e->progress_u);
        }
        if (t >= max_new) {
            e->hit_limit = 1;
            break;
        }
        if (mtp_mode) {
            if (q4_sess_n_kv(e->sess) + (uint32_t)n_draft + 2 >=
                q4_sess_max_kv(e->sess)) {
                e->hit_limit = 1;
                break;
            }
            int32_t drafts[4];
            int nd_out = 0;
            int32_t nxt = 0;
            const float *nlog = NULL;
            int crc = q4_mtp_cycle(e->mtp, e->sess, id, n_draft, e->temp,
                                   e->top_k, e->top_p, &e->rng, drafts, &nd_out,
                                   &nxt, &nlog);
            if (crc == 1) {
                /* Recovered with only c0 kept: continue in plain mode. */
                fprintf(stderr, "\nq4: MTP cycle recovered at gen %d\n", t);
                e->mtp_fail = 1;
                mtp_mode = 0;
                logits = q4_sess_logits(e->sess);
                if (!logits) break;
                continue;
            }
            if (crc < 0) {
                fprintf(stderr,
                        "\nq4: MTP cycle failed at gen %d, plain decode\n", t);
                e->mtp_fail = 1;
                mtp_mode = 0;
                /* c0 was emitted but not yet processed by the model. */
                if (!q4_sess_decode_ex(e->sess, id, NULL, NULL, true)) break;
                logits = q4_sess_logits(e->sess);
                if (!logits) break;
                continue;
            }
            if (crc == 2) {
                /* Partial repair: keep the accepted tokens, then plain. */
                fprintf(stderr, "\nq4: MTP partial repair at gen %d\n", t);
                e->mtp_fail = 1;
                mtp_mode = 0;
            }
            for (int i = 0; i < nd_out; i++) {
                rc = emit_piece(e, drafts[i], out, &o, max_out, &in_think,
                                &think_toks, hold, &nh, stream_cb, u, t, im_end,
                                eos, th, te);
                if (rc == 1) {
                    /* the stop token was a draft, so the model read it: the
                     * saved ids end with it (n_kv == n_ids), and the next
                     * turn's prompt restores from this state */
                    if (gen_ids && n_gen < max_new &&
                        (drafts[i] == im_end || drafts[i] == eos))
                        gen_ids[n_gen++] = drafts[i];
                    stop = 1;
                    break;
                }
                if (rc == 2) {
                    e->hit_limit = 1;
                    stop = 1;
                    break;
                }
                t++;
                e->completion_tokens++;
                if (gen_ids) gen_ids[n_gen++] = drafts[i];
                q4_pen_push(drafts[i]);
                recent[(t - 1) & (LOOP_RING - 1)] = drafts[i];
                if (t % 4 == 0 || t == 1) {
                    fprintf(stderr, "\rdecode %d/%d", t, max_new);
                    fflush(stderr);
                    if (e->progress) e->progress(n, n, t, e->progress_u);
                }
                if (t >= max_new) {
                    e->hit_limit = 1;
                    stop = 1;
                    break;
                }
            }
            if (stop) break;
            if (crc == 2) {
                /* Partial repair: sample the next token ourselves. */
                logits = q4_sess_logits(e->sess);
                if (!logits) break;
                continue;
            }
            c0 = nxt;
            logits = nlog ? nlog : logits;
        } else {
            if (q4_sess_n_kv(e->sess) + 1 >= q4_sess_max_kv(e->sess)) {
                e->hit_limit = 1;
                break;
            }
            if (!q4_sess_decode_ex(e->sess, id, NULL, NULL, true)) {
                fprintf(stderr, "\nq4: decode failed at gen %d\n", t);
                break;
            }
            logits = q4_sess_logits(e->sess);
            if (!logits) break;
        }
        const int lp = loop_period(recent, t);
        if (lp) {
            fprintf(stderr, "\nq4: repeat loop (period %d) at gen %d, stop\n",
                    lp, t);
            e->hit_limit = 1; /* cut, not finished: finish_reason length */
            break;
        }
    }
    if (t >= max_new) e->hit_limit = 1;
    q4_expert_reclaim_check(e->ex, "end", -t - 1);
    if (in_think && think_toks)
        fprintf(stderr, "\nq4: still thinking at stop (%d think tok, limit=%d)\n",
                think_toks, e->hit_limit);
    dt = now_s() - t0;
    if (e->completion_tokens)
        fprintf(stderr, "\rdecode %d tok  %.2f t/s                \n",
                e->completion_tokens,
                dt > 0 ? e->completion_tokens / dt : 0);
    /* Same phase split the probes print (Q4_PROFILE fills it in q4_fwd).
     * Per token in ms, not the raw probe sums. */
    if (getenv("Q4_PROFILE") && e->completion_tokens) {
        double inv = 1e3 / e->completion_tokens;
        fprintf(stderr,
                "  dec/tok ms    embed %.2f  attn %.2f  moe %.2f  comb %.2f  "
                "head %.2f\n",
                q4_prof_dec[0] * inv, q4_prof_dec[1] * inv, q4_prof_dec[2] * inv,
                q4_prof_dec[3] * inv, q4_prof_dec[4] * inv);
        fprintf(stderr,
                "  attn-detail   ple %.2f  hc-attn %.2f  qsa/gdn %.2f  "
                "comb+ffnmix %.2f\n",
                q4_prof_dec[5] * inv, q4_prof_dec[6] * inv, q4_prof_dec[7] * inv,
                q4_prof_dec[8] * inv);
        memset(q4_prof_dec, 0, sizeof q4_prof_dec);
    }
    if (getenv("Q4_STEP_PROF") && e->completion_tokens) {
        double inv = 1.0 / e->completion_tokens;
        fprintf(stderr,
                "  step/tok ms   gA+sync %.2f  hostpart %.2f  gB %.2f  "
                "e-attn %.2f  e-moe %.2f  e-comb %.2f  embed %.2f  head %.2f\n",
                q4_prof_step[0] * inv * 1e3, q4_prof_step[1] * inv * 1e3,
                q4_prof_step[2] * inv * 1e3, q4_prof_step[3] * inv * 1e3,
                q4_prof_step[4] * inv * 1e3, q4_prof_step[5] * inv * 1e3,
                q4_prof_step[6] * inv * 1e3, q4_prof_step[7] * inv * 1e3);
        memset(q4_prof_step, 0, sizeof(q4_prof_step));
        if (q4_prof_gpu_sec[0] || q4_prof_gpu_sec[1] || q4_prof_gpu_sec[2] ||
            q4_prof_gpu_sec[3] || q4_prof_gpu_sec[4])
            fprintf(stderr,
                    "  gpu/tok ms    hc1 %.2f  attn %.2f  comb %.2f  "
                    "ffnmix %.2f  moea %.2f\n",
                    q4_prof_gpu_sec[0] * inv, q4_prof_gpu_sec[1] * inv,
                    q4_prof_gpu_sec[2] * inv, q4_prof_gpu_sec[3] * inv,
                    q4_prof_gpu_sec[4] * inv);
        if (q4_prof_gpu_sec[5] || q4_prof_gpu_sec[6] || q4_prof_gpu_sec[7])
            fprintf(stderr,
                    "  qsa/tok ms    prep %.2f  sel %.2f  dec %.2f\n",
                    q4_prof_gpu_sec[5] * inv, q4_prof_gpu_sec[6] * inv,
                    q4_prof_gpu_sec[7] * inv);
        memset(q4_prof_gpu_sec, 0, sizeof(q4_prof_gpu_sec));
    }
    if (getenv("Q4_STEP_PROF") && e->completion_tokens) {
        static unsigned long long pj, pm, pl;
        unsigned long long dj, dm, dl;
        q4_moe_dev_stats(&dj, &dm, &dl);
        if (dj - pj) {
            double sub[4] = { 0, 0, 0, 0 };
            double rt = q4_moe_route_prof(sub);
            if (rt >= 0.0)
                fprintf(stderr,
                        "  devroute      jobs %llu  misses %llu  "
                        "%.0f us/job  route %.0f us "
                        "(scan %.0f topk %.0f pub %.0f)\n",
                        dj - pj, dm - pm,
                        (double)(dl - pl) / (double)(dj - pj), rt,
                        sub[0], sub[1], sub[2]);
            else
                fprintf(stderr,
                        "  devroute      jobs %llu  misses %llu  "
                        "%.0f us/job\n",
                        dj - pj, dm - pm,
                        (double)(dl - pl) / (double)(dj - pj));
        }
        {
            static uint64_t pv[11];
            uint64_t v[11];
            q4_expert_stage_stats(e->ex, v);
            if (v[6] + v[8] + v[9] + v[10] - pv[6] - pv[8] - pv[9] - pv[10])
                fprintf(stderr, "  prefetch      %llu uploads  %llu hit  "
                        "(%llu resident/pending, %llu no victim, %llu busy)\n",
                        (unsigned long long)(v[6] - pv[6]),
                        (unsigned long long)(v[7] - pv[7]),
                        (unsigned long long)(v[8] - pv[8]),
                        (unsigned long long)(v[9] - pv[9]),
                        (unsigned long long)(v[10] - pv[10]));
            if (v[0] + v[1] - pv[0] - pv[1])
                fprintf(stderr, "  stage admit   %llu admitted (%llu deferred, "
                        "%llu dropped, %llu tick uploads)  %llu rejected  "
                        "%llu dup\n",
                        (unsigned long long)(v[0] - pv[0]),
                        (unsigned long long)(v[2] - pv[2]),
                        (unsigned long long)(v[3] - pv[3]),
                        (unsigned long long)(v[4] - pv[4]),
                        (unsigned long long)(v[1] - pv[1]),
                        (unsigned long long)(v[5] - pv[5]));
            memcpy(pv, v, sizeof(pv));
            static unsigned long long ps0, pn0;
            unsigned long long ts, tn_;
            q4_moe_tail_skip_stats(&ts, &tn_);
            if (tn_ > pn0)
                fprintf(stderr, "  tail skip     %llu of %llu missed pairs "
                        "skipped (%.1f%%)\n", ts - ps0, tn_ - pn0,
                        100.0 * (double)(ts - ps0) / (double)(tn_ - pn0));
            ps0 = ts;
            pn0 = tn_;
        }
        {
            unsigned long long wc = 0;
            double wa = q4_hip_cpx_wait_prof(&wc);
            if (wc && wa >= 0)
                fprintf(stderr, "  cpxwait       %.0f us x%llu = %.1f ms/tok\n",
                        wa, wc, wa * (double)wc / 1e3 /
                                (double)e->completion_tokens);
            extern double q4_prof_gpu_ms;
            fprintf(stderr, "  gpu-busy/tok  %.2f ms\n",
                    q4_prof_gpu_ms / (double)e->completion_tokens);
            q4_prof_gpu_ms = 0;
        }
        pj = dj; pm = dm; pl = dl;
        extern double q4_prof_moe_cw;
        static double pi, pcw, pg, pt, pj2, png, pnc, pnl;
        if (q4_prof_moe_ncall > pnl)
            fprintf(stderr,
                    "  moe/layer     topk %.0f us  cw %.0f us  io %.0f us  "
                    "gemm %.0f us  join %.0f us  gpu-hit %.1f  cpu %.1f\n",
                    (q4_prof_moe_topk - pt) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_cw - pcw) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_io - pi) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_gemm - pg) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_join - pj2) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_ng - png) / (q4_prof_moe_ncall - pnl),
                    (q4_prof_moe_nc - pnc) / (q4_prof_moe_ncall - pnl));
        pi = q4_prof_moe_io; pcw = q4_prof_moe_cw; pg = q4_prof_moe_gemm;
        pt = q4_prof_moe_topk; pj2 = q4_prof_moe_join;
        png = q4_prof_moe_ng; pnc = q4_prof_moe_nc; pnl = q4_prof_moe_ncall;
    }
    if (e->mtp) {
        uint64_t prop = 0, acc = 0, cyc = 0, gat = 0;
        q4_mtp_stats(e->mtp, &prop, &acc, &cyc, &gat);
        if (cyc)
            fprintf(stderr,
                    "q4: mtp draft acceptance = %.5f (%llu accepted / %llu "
                    "drafted, %llu gated), mean len = %.2f (%llu cycles)\n",
                    prop ? (double)acc / (double)prop : 0.0,
                    (unsigned long long)acc, (unsigned long long)prop,
                    (unsigned long long)gat,
                    (double)e->completion_tokens / (double)cyc,
                    (unsigned long long)cyc);
        {
            const char *mp = getenv("Q4_MTP_PROF");
            if (getenv("Q4_PROFILE") || (mp && mp[0] && mp[0] != '0')) {
                q4_mtp_prof_print(e->mtp, stderr);
                /* Weight traffic charged to each MTP phase, per type. */
                if (getenv("Q4_GEMV_PROF"))
                    q4_hip_gemv_prof_report();
            }
        }
        if (getenv("Q4_PROFILE"))
            q4_moe_sb_prof_print(stderr);
    }
    /* Weight bytes and GPU time per GEMV type.  The MTP arms live inside
     * the block above (phase-tagged); plain decode has no phases, so print
     * the same table here or Q4_GEMV_PROF is a no-op without a head. */
    if (getenv("Q4_GEMV_PROF"))
        q4_hip_gemv_prof_report();
    if (getenv("Q4_PROFILE"))
        fprintf(stderr, "q4 prof decode: moe io %.2f  d2h %.2f  gemm %.2f s\n",
                q4_prof_moe_io, q4_prof_moe_d2h, q4_prof_moe_gemm);
    fflush(stderr);
    if (client_gone(e)) closed = 1;
    if (gen_ids && n_gen > 0) dump_out(e, roles, contents, nmsg, ids, n, gen_ids, n_gen);
    if (gen_ids) {
        if (nan_stop) {
            /* the state is NaN from somewhere in this request: no snapshot,
             * and the prompt checkpoint it wrote goes too */
            if (ckpt_at > 0 && prompt_path[0]) {
                char hp[4200];
                snprintf(hp, sizeof(hp), "%s.mtp", prompt_path);
                (void)unlink(prompt_path);
                (void)unlink(hp);
            }
            fprintf(stderr, "q4: session cache not saved (NaN)%s\n",
                    ckpt_at > 0 ? ", prompt checkpoint dropped" : "");
        } else if (n_gen <= 0 || q4_stop || closed) {
            fprintf(stderr,
                    "q4: session cache not saved (gen %d stop %d closed %d)\n",
                    n_gen, (int)q4_stop, closed);
        } else {
            int32_t *all = malloc((size_t)(n + n_gen) * sizeof(int32_t));
            if (all) {
                memcpy(all, ids, (size_t)n * sizeof(int32_t));
                memcpy(all + n, gen_ids, (size_t)n_gen * sizeof(int32_t));
                double tc = now_s();
                uh = first_user_hash(roles, contents, nmsg);
                const int saved =
                    q4_sess_save(e->sess, cache_path, all, n + n_gen, n, nmsg, uh);
                if (saved) mtp_kv_save_beside(e, cache_path, all, n + n_gen);
                if (saved)
                    fprintf(stderr,
                            "q4: session cache saved %d tok (prompt %d msg %d, %.2f s)\n",
                            n + n_gen, n, nmsg, now_s() - tc);
                free(all);
            }
        }
        free(gen_ids);
    }
    free(ids);
    return o;
}

int q4_engine_generate_msgs(q4_engine *e, const char *const *roles,
                            const char *const *contents, int nmsg, char *out,
                            int max_out, int max_new,
                            void (*stream_cb)(const char *piece, int kind, void *u),
                            void *u) {
    if (!e) return -1;
    pthread_mutex_lock(&e->mu);
    int n = generate_locked(e, roles, contents, nmsg, out, max_out, max_new,
                            stream_cb, u);
    /* the images (and their rows) are this request's only */
    q4_sess_set_embd_rows(e->sess, NULL, 0);
    q4_engine_set_images(e, NULL, 0);
    free(e->img_rids);
    free(e->img_rows);
    free(e->img_tab);
    e->img_rids = NULL;
    e->img_rows = NULL;
    e->img_tab = NULL;
    pthread_mutex_unlock(&e->mu);
    return n;
}

int q4_engine_generate(q4_engine *e, const char *system, const char *user,
                       char *out, int max_out, int max_new,
                       void (*stream_cb)(const char *piece, int kind, void *u),
                       void *u) {
    const char *roles[2];
    const char *contents[2];
    int n = 0;
    if (system && system[0]) {
        roles[n] = "system";
        contents[n] = system;
        n++;
    }
    roles[n] = "user";
    contents[n] = user ? user : "";
    n++;
    return q4_engine_generate_msgs(e, roles, contents, n, out, max_out, max_new,
                                   stream_cb, u);
}

#define Q4_WARM_MAGIC 0x57345134u /* "Q4WS" LE */
#define Q4_WARM_VER 1u

/* One decode step touches n_layer * top-k distinct experts (480 here).
 * Pin ~1.5× that, and never more than 1/4 of L1, so a long prefill can
 * still evict. Q4_PIN_CAP is an exact top-N override. */
static uint32_t warm_pin_cap(const q4_engine *e) {
    uint32_t nslots = e && e->ex ? q4_expert_cache_n_slots(e->ex) : 0;
    return q4_expert_warm_cap(nslots, e->g.n_layer, e->g.n_expert_used);
}

static int pin_cap_exact(void) {
    const char *env = getenv("Q4_PIN_CAP");
    return env && env[0] && strtol(env, NULL, 10) > 0;
}

static void preview_line(FILE *fp, const char *s) {
    if (!fp || !s || !s[0]) return;
    char b[161];
    int n = 0;
    for (int i = 0; s[i] && n < 160; i++) {
        unsigned char c = (unsigned char)s[i];
        b[n++] = c < 32 ? ' ' : (char)c;
    }
    b[n] = 0;
    fprintf(fp, "  sample: %s\n", b);
}

int q4_engine_warm_save(q4_engine *e, const char *path) {
    if (!e || !path) return -1;
    uint32_t n = q4_expert_pinned_count(e->ex);
    if (!n) return -1;
    int32_t *layers = malloc((size_t)n * sizeof(int32_t));
    int32_t *eids = malloc((size_t)n * sizeof(int32_t));
    uint32_t *hits = malloc((size_t)n * sizeof(uint32_t));
    if (!layers || !eids || !hits) {
        free(layers); free(eids); free(hits);
        return -1;
    }
    q4_expert_pinned_list(e->ex, layers, eids, hits, n);
    struct {
        uint32_t magic, version;
        uint64_t bytes_total;
        uint32_t n_layer, n_expert, count;
    } h = {Q4_WARM_MAGIC, Q4_WARM_VER, e->g.bytes_total, e->g.n_layer,
           e->g.n_expert, n};
    char tmp[4096 + 16];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    bool ok = f && fwrite(&h, 1, sizeof(h), f) == sizeof(h) &&
              fwrite(layers, 1, (size_t)n * 4, f) == (size_t)n * 4 &&
              fwrite(eids, 1, (size_t)n * 4, f) == (size_t)n * 4 &&
              fwrite(hits, 1, (size_t)n * 4, f) == (size_t)n * 4;
    if (f) fclose(f);
    free(layers); free(eids); free(hits);
    if (!ok) {
        if (f) remove(tmp);
        return -1;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return (int)n;
}

int q4_engine_warm_load(q4_engine *e, const char *path, FILE *fp) {
    if (!e || !path) return -1;
    /* Static L1 experts already hold the warm set; pinning it again is
     * redundant (S replaces the warm pin). */
    if (q4_expert_static_on(e->ex)) return 0;
    return q4_expert_warm_load(e->ex, &e->g, path, warm_pin_cap(e), fp);
}

int q4_engine_route_save(q4_engine *e, const char *path) {
    if (!e || !path) return -1;
    return q4_expert_route_save(e->ex, &e->g, path, stderr);
}

int q4_engine_pin_ja(q4_engine *e, FILE *fp) {
    if (!e) return -1;
    if (q4_expert_static_on(e->ex)) return 0;
    static const char *prompts[] = {
        "日本語で自己紹介してください。日常会話、技術、法律、文学のどれが得意か、"
        "短く答えてください。",
        "次の文章を丁寧なビジネス日本語に直してください。「明日の会議、資料まだだ"
        "から後で送る。よろしく。」",
        "Rustで簡単なHTTPサーバーを書く手順を、日本語で箇条書きにしてください。"
        "エラー処理と並行性にも触れて。",
        "二次方程式 ax^2+bx+c=0 の解の公式を導出し、日本語で途中式を説明して"
        "ください。",
        "日本国憲法第9条の趣旨を、高校生向けの日本語で200字以内にまとめて"
        "ください。",
        "夏目漱石『こころ』の先生とKの関係を、Spoilers込みで短く解説して。",
        "光合成の明反応と暗反応の違いを、日本語の科学用語を使って説明して。",
        "次を自然な日本語に訳して: Caching routed experts in VRAM avoids SSD "
        "round-trips on every decode step.",
        "このC言語のバグを日本語で指摘して。int n=strlen(s); char b[n]; "
        "strcpy(b,s); return b;",
        "京都の夏の過ごし方、屋内で涼しく働く人向けに、具体的なアドバイスを"
        "日本語で。",
    };
    const int np = (int)(sizeof(prompts) / sizeof(prompts[0]));
    if (fp)
        fprintf(fp, "q4: Japanese expert warmup (%d prompts)...\n", np);
    float save_temp = e->temp;
    int save_k = e->top_k;
    float save_p = e->top_p;
    e->temp = 0.f;
    e->top_k = 1;
    e->top_p = 1.f;
    uint32_t nslots = q4_expert_cache_n_slots(e->ex);
    /* Pin after every prompt. Pinning occupied slots 0..cap on the first
     * generate froze the first-touched experts and starved later prompts. */
    uint32_t cap = warm_pin_cap(e);
    uint32_t step = e->g.n_layer * (e->g.n_expert_used ? e->g.n_expert_used : 10u);
    if (fp) {
        fprintf(fp,
                "q4: warmup %d prompts, then pin %s %u / %u L1 (decode step %u)\n",
                np, pin_cap_exact() ? "top" : "stable", cap, nslots, step);
        fflush(fp);
    }
    char buf[4096];
    for (int i = 0; i < np && !q4_stop; i++) {
        if (fp) fprintf(fp, "  ja[%d/%d] ", i + 1, np);
        int n = q4_engine_generate(e, "日本語で簡潔に答えてください。", prompts[i],
                                   buf, (int)sizeof(buf), 16, NULL, NULL);
        if (fp)
            fprintf(fp, "gen %d B\n", n < 0 ? 0 : n);
        if (fp && i == 0) {
            const char *s = (n > 0) ? buf : q4_engine_last_reasoning(e);
            preview_line(fp, s);
        }
        fflush(fp ? fp : stderr);
    }
    uint32_t add = pin_cap_exact() ? q4_expert_pin_hottest(e->ex, cap)
                                   : q4_expert_pin_stable(e->ex, cap, 8);
    if (fp)
        fprintf(fp, "q4: pinned %u L1 slots (cap %u)\n", add, cap);
    e->temp = save_temp;
    e->top_k = save_k;
    e->top_p = save_p;
    if (fp) {
        uint64_t l1h = 0, l2h = 0, ssd = 0, bytes = 0;
        q4_expert_cache_stats_ex(e->ex, &l1h, &l2h, &ssd, &bytes);
        fprintf(fp,
                "q4: pinned %u / %u L1 slots  cache L1 %llu  L2 %llu  SSD miss %llu\n",
                q4_expert_pinned_count(e->ex), nslots,
                (unsigned long long)l1h, (unsigned long long)l2h,
                (unsigned long long)ssd);
    }
    q4_expert_clear_sticky(e->ex);
    q4_sess_reset(e->sess);
    return (int)q4_expert_pinned_count(e->ex);
}

const q4_gguf *q4_engine_gguf(const q4_engine *e) {
    return e ? &e->g : NULL;
}

static int32_t argmax_logits(const float *logits, uint32_t n) {
    int32_t id = 0;
    float best = logits ? logits[0] : 0.f;
    for (uint32_t i = 1; i < n; i++) {
        if (logits[i] > best) {
            best = logits[i];
            id = (int32_t)i;
        }
    }
    return id;
}

/* Teacher-forced quality probe: feed ids in 8-token verify windows and score
 * every next token from the target logits of its row. Window w covers
 * ids[o..o+7]; row r predicts ids[o+r+1]. Reports mean NLL, perplexity and
 * top-1 agreement; dump (if non-NULL) gets "pos target nll argmax" lines. */
int q4_engine_ppl(q4_engine *e, const int32_t *ids, int n, FILE *fp,
                  FILE *dump) {
    if (!e || !ids || n < 2 || !fp) return -1;
    uint32_t max_kv = q4_sess_max_kv(e->sess);
    if (n + 16 > (int)max_kv) n = (int)max_kv - 16;
    if (n < 2) return -1;
    const uint32_t nv = e->g.n_vocab;
    q4_sess_reset(e->sess);
    q4_expert_clear_sticky(e->ex);
    double nll_sum = 0;
    uint64_t scored = 0, top1 = 0;
    double t0 = now_s();
    /* Q4_PPL_PREFILL=N: score the first N tokens through the prompt path
     * (q4_sess_prefill chunks, every row's logits), then hand the L1 tail
     * back and score the rest in verify windows — covers the lent prefill
     * scratch, which the window path never touches. */
    int o0 = 0;
    {
        const char *pe = getenv("Q4_PPL_PREFILL");
        int np = pe && pe[0] ? atoi(pe) : 0;
        if (np > n - 1) np = n - 1;
        const int chunk = (int)q4_pref_chunk();
        while (o0 < np && !q4_stop) {
            int w = np - o0 < chunk ? np - o0 : chunk;
            if (!q4_sess_prefill(e->sess, ids + o0, w)) {
                fprintf(fp, "PPL_ABORT prompt prefill at %d\n", o0);
                return -1;
            }
            for (int r = 0; r < w; r++) {
                if (!q4_sess_logits_at(e->sess, r)) {
                    fprintf(fp, "PPL_ABORT prompt logits at %d\n", o0 + r);
                    return -1;
                }
                const float *lg = q4_sess_logits(e->sess);
                const int32_t tgt = ids[o0 + r + 1];
                int32_t am = argmax_logits(lg, nv);
                float mx = lg[am];
                double se = 0;
                for (uint32_t i = 0; i < nv; i++)
                    se += exp((double)(lg[i] - mx));
                double nll = -((double)(lg[tgt] - mx) - log(se));
                nll_sum += nll;
                scored++;
                if (am == tgt) top1++;
                if (dump)
                    fprintf(dump, "%d %d %.6f %d\n", o0 + r + 1, tgt, nll, am);
            }
            o0 += w;
            fprintf(fp, "  ppl prompt %d  nll %.4f  top1 %.2f%%\n", o0,
                    nll_sum / (double)scored,
                    100.0 * (double)top1 / (double)scored);
            fflush(fp);
        }
        if (!q4_sess_lend_end(e->sess)) {
            fprintf(fp, "PPL_ABORT lend\n");
            return -1;
        }
    }
    /* Q4_PPL_WIN (1..8, default 8): rows per scored verify window — 4
     * matches an MTP cycle's verify (c0 + 3 drafts). */
    int pw = 8;
    if (getenv("Q4_PPL_WIN")) pw = atoi(getenv("Q4_PPL_WIN"));
    if (pw < 1 || pw > 8) pw = 8;
    for (int o = o0; o + 1 < n && !q4_stop; o += pw) {
        int w = n - o < pw ? n - o : pw;
        if (!q4_sess_prefill_tokens(e->sess, ids + o, w)) {
            fprintf(fp, "PPL_ABORT prefill at %d\n", o);
            return -1;
        }
        for (int r = 0; r < w && o + r + 1 < n; r++) {
            if (!q4_sess_logits_at(e->sess, r)) {
                fprintf(fp, "PPL_ABORT logits at %d\n", o + r);
                return -1;
            }
            const float *lg = q4_sess_logits(e->sess);
            const int32_t tgt = ids[o + r + 1];
            if (!lg || tgt < 0 || (uint32_t)tgt >= nv) {
                fprintf(fp, "PPL_ABORT row %d\n", o + r);
                return -1;
            }
            int32_t am = argmax_logits(lg, nv);
            float mx = lg[am];
            double se = 0;
            for (uint32_t i = 0; i < nv; i++) se += exp((double)(lg[i] - mx));
            double nll = -((double)(lg[tgt] - mx) - log(se));
            nll_sum += nll;
            scored++;
            if (am == tgt) top1++;
            if (dump)
                fprintf(dump, "%d %d %.6f %d\n", o + r + 1, tgt, nll, am);
        }
        if (w == pw && (o / pw) % 64 == 63) {
            fprintf(fp, "  ppl at %d  nll %.4f  top1 %.2f%%\n", o + w,
                    nll_sum / (double)scored, 100.0 * (double)top1 / (double)scored);
            fflush(fp);
        }
    }
    double dt = now_s() - t0;
    if (!scored) return -1;
    double mn = nll_sum / (double)scored;
    fprintf(fp, "PPL_OK tokens %llu  nll %.5f  ppl %.4f  top1 %.3f%%  %.1f s\n",
            (unsigned long long)scored, mn, exp(mn),
            100.0 * (double)top1 / (double)scored, dt);
    fflush(fp);
    return 0;
}

int q4_engine_bench(q4_engine *e, const int32_t *ids, int n, int n_dec, FILE *fp) {
    if (!e || !ids || n <= 0 || !fp) return -1;
    if (n_dec < 0) n_dec = 0;
    uint32_t max_kv = q4_sess_max_kv(e->sess);
    if (n_dec + 8 >= (int)max_kv) n_dec = (int)max_kv > 16 ? (int)max_kv - 16 : 0;
    if (n + n_dec + 4 > (int)max_kv) n = (int)max_kv - n_dec - 4;
    if (n < 1) return -1;
    q4_sess_reset(e->sess);
    q4_expert_clear_sticky(e->ex);
    uint64_t l1a = 0, l2a = 0, ssda = 0, ba = 0;
    q4_expert_cache_stats_ex(e->ex, &l1a, &l2a, &ssda, &ba);
    fprintf(fp,
            "bench start  tokens=%d decode=%d ctx=%u pinned=%u L1=%u\n",
            n, n_dec, q4_sess_max_kv(e->sess), q4_expert_pinned_count(e->ex),
            q4_expert_cache_n_slots(e->ex));
    fflush(fp);
    double t_all = now_s();
    int off = 0, chunk_i = 0;
    while (off < n && !q4_stop) {
        int csz = n - off;
        if (csz > 512) csz = 512;
        uint64_t l1b = 0, l2b = 0, ssdb = 0, bb = 0;
        q4_expert_cache_stats_ex(e->ex, &l1b, &l2b, &ssdb, &bb);
        uint64_t avail = mem_available();
        if (avail && avail < 8ull * Q4_GIB) {
            fprintf(fp, "BENCH_ABORT mem %.2f GiB\n", (double)avail / Q4_GIB);
            fflush(fp);
            return -1;
        }
        if (off + csz < n)
            (void)q4_sess_ple_ahead(e->sess, ids + off + csz,
                                    n - off - csz > 512 ? 512 : n - off - csz);
        double t0 = now_s();
        if (!q4_sess_prefill(e->sess, ids + off, csz)) {
            fprintf(fp, "BENCH_ABORT prefill at %d\n", off);
            fflush(fp);
            return -1;
        }
        double dt = now_s() - t0;
        off += csz;
        chunk_i++;
        uint64_t l1c = 0, l2c = 0, ssdc = 0, bc = 0;
        q4_expert_cache_stats_ex(e->ex, &l1c, &l2c, &ssdc, &bc);
        uint64_t d1 = l1c - l1b, d2 = l2c - l2b, ds = ssdc - ssdb;
        uint64_t den = d1 + d2 + ds;
        fprintf(fp,
                "  chunk %d  +%d  at %d  %.2f s  %.1f tok/s  L1 %llu  L2 %llu  "
                "SSD %llu  hit %.0f%%  ram %.1f GiB\n",
                chunk_i, csz, off, dt, dt > 0 ? (double)csz / dt : 0,
                (unsigned long long)d1, (unsigned long long)d2,
                (unsigned long long)ds, den ? 100.0 * (double)d1 / (double)den : 0,
                (double)mem_available() / Q4_GIB);
        fflush(fp);
    }
    if (!q4_sess_lend_end(e->sess)) {
        fprintf(fp, "BENCH_ABORT lend\n");
        return -1;
    }
    double tall = now_s() - t_all;
    if (!q4_sess_head(e->sess, NULL, NULL)) {
        fprintf(fp, "BENCH_ABORT lm-head\n");
        return -1;
    }
    const float *logits = q4_sess_logits(e->sess);
    if (!logits) {
        fprintf(fp, "BENCH_ABORT logits\n");
        return -1;
    }
    int32_t id = argmax_logits(logits, e->g.n_vocab);
    int32_t *gen = n_dec ? malloc((size_t)n_dec * sizeof(int32_t)) : NULL;
    int got = 0;
    double tdec = 0, tdec_rest = 0;
    uint64_t l1d0 = 0, l2d0 = 0, s0 = 0, b0 = 0;
    q4_expert_cache_stats_ex(e->ex, &l1d0, &l2d0, &s0, &b0);
    /* Weight-read counters from here on cover the decode loop only. */
    if (getenv("Q4_GEMV_PROF")) q4_hip_gemv_prof_reset();
    for (int i = 0; i < n_dec && !q4_stop; i++) {
        if (mem_available() && mem_available() < 8ull * Q4_GIB) {
            fprintf(fp, "BENCH_ABORT mem during decode\n");
            free(gen);
            return -1;
        }
        int32_t nxt = 0;
        float lg = 0.f;
        double t0 = now_s();
        if (!q4_sess_decode_ex(e->sess, id, &nxt, &lg, true)) {
            fprintf(fp, "BENCH_ABORT decode at %d\n", i);
            free(gen);
            return -1;
        }
        double dt = now_s() - t0;
        fprintf(fp, "  dec %d  id %d -> %d  %.3f s  %.2f tok/s\n", i, id, nxt, dt,
                dt > 0 ? 1.0 / dt : 0);
        fflush(fp);
        if (gen) gen[got] = nxt;
        id = nxt;
        got++;
        tdec += dt;
        if (i > 0) tdec_rest += dt;
    }
    uint64_t l1e = 0, l2e = 0, se = 0, be = 0;
    q4_expert_cache_stats_ex(e->ex, &l1e, &l2e, &se, &be);
    fprintf(fp, "prefill total  %d tok  %.2f s  %.1f tok/s\n", n, tall,
            tall > 0 ? (double)n / tall : 0);
    if (got > 0) {
        double w = tdec / got;
        fprintf(fp, "decode mean    %.3f s  %.2f tok/s  (%d tok)\n", w,
                w > 0 ? 1.0 / w : 0, got);
    }
    if (got > 1) {
        double w = tdec_rest / (got - 1);
        fprintf(fp, "decode rest    %.3f s  %.2f tok/s  (%d tok)\n", w,
                w > 0 ? 1.0 / w : 0, got - 1);
    }
    /* Weight bytes read per type while decoding. */
    if (getenv("Q4_GEMV_PROF")) q4_hip_gemv_prof_report();
    if (gen && got > 0 && e->tok) {
        char text[512];
        int nb = q4_tok_decode(e->tok, gen, got, text, (int)sizeof(text) - 1);
        if (nb < 0) nb = 0;
        text[nb] = 0;
        preview_line(fp, text);
    }
    fprintf(fp,
            "expert cache   L1 +%llu  L2 +%llu  SSD %llu  SSD %.2f GiB  pinned %u\n",
            (unsigned long long)(l1e - l1a), (unsigned long long)(l2e - l2a),
            (unsigned long long)(se - ssda), (double)(be - ba) / Q4_GIB,
            q4_expert_pinned_count(e->ex));
    uint64_t dd1 = l1e - l1d0, dd2 = l2e - l2d0;
    uint64_t dden = dd1 + dd2 + (se - s0);
    fprintf(fp, "decode hits    L1 %llu  L2 %llu  hit %.0f%%\n",
            (unsigned long long)dd1, (unsigned long long)dd2,
            dden ? 100.0 * (double)dd1 / (double)dden : 0);
    fprintf(fp, "BENCH_OK\n");
    fflush(fp);
    free(gen);
    return 0;
}
