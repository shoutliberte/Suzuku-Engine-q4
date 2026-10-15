#define _GNU_SOURCE
#include "q4.h"

#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

typedef struct {
    int32_t  layer;
    int32_t  eid;
    uint32_t lru;
    uint32_t hits;
    uint8_t  pinned;
    uint8_t  sticky;
    uint8_t  filling; /* reserved by an in-flight reader thread */
    uint64_t put_seq; /* last async H2D seq; <= upload_mark means landed */
    uint8_t  fixed;  /* static-resident expert: never evicted/rebound */
    uint8_t  dev_vis; /* occupancy is published in the device residency table */
    uint8_t  spec;   /* decode prefetch bound it; cleared by its first hit */
    int64_t  dev_quiet; /* token idx when the occupant was unmapped from devtab;
                         * -1 = never published.  A slot may only be overwritten
                         * for a GPU upload once its last published mapping has
                         * been invisible for a whole decode token. */
} q4_slot;

/* Reader pool: parallel SSD pread / L2 memcpy for expert staging. */
typedef struct {
    pthread_t       th[8];
    int             n_th;
    int             quit;
    int             started;
    /* One job at a time across ALL submitters (decode stager, devroute
     * dispatcher, prefetch thread, sync callers): the job fields below are
     * a single shared set, so a second submission mid-job would interleave
     * ids/next/done and deadlock the waiters. */
    pthread_mutex_t job_mu;
    pthread_mutex_t mu;
    pthread_cond_t  cv_work;
    pthread_cond_t  cv_done;
    uint32_t        seq;       /* job generation */
    int32_t         layer;
    const int32_t  *ids;
    uint32_t        n;
    uint32_t        next;      /* next unclaimed index */
    uint32_t        done;
    int             failed;
    int             gpu_upload;
    int             may_defer; /* devroute: unsafe slots may defer to tick */
} q4_io_pool;

typedef struct {
    const q4_tensor *gate, *up, *down;
    uint64_t off_gate, off_up, off_down, total;
} q4_layer_ex;

struct q4_expert_cache {
    const q4_gguf *g;
    uint32_t       n_slots;
    q4_slot       *slots;
    uint8_t       *arena;
    uint64_t       slot_bytes;
    uint32_t       clock;
    uint64_t       hits;
    uint64_t       misses;
    uint64_t       l2_hits;
    uint64_t       bytes_read;
    q4_layer_ex    layers[Q4_MAX_LAYER];
    uint32_t       n_l2;
    q4_slot       *l2_slots;
    uint8_t       *l2_arena;
    size_t         l2_arena_bytes;
    int            resident; /* every routed expert is in l2_arena, tight-packed */
    int            res_registered; /* l2_arena hipHostRegister'ed for DMA */
    uint64_t       res_off[Q4_MAX_LAYER];
    /* Byte budget for the resident fill. When the full image exceeds the
     * RAM budget each layer fills only its first res_quota[L] experts —
     * the unfilled remainder keeps its pages untouched (no RSS cost, not
     * registered) and stages from the GGUF through the io pool like
     * non-resident experts. */
    uint64_t       res_fill_budget;
    /* Arbitrary eid->image-offset map: the warm set (Q4_WARM_FILE) is
     * placed first so partial residency keeps the hottest experts in
     * DRAM rather than the lowest-numbered ones. res_ids[res_idoff[L]+k]
     * is the inverse map used by the fill loop. */
    int64_t       *res_pos;  /* layer*map_stride+eid -> byte offset, -1 none */
    int32_t       *res_ids;  /* packed [sum(res_quota[L])] -> eid */
    uint64_t       res_idoff[Q4_MAX_LAYER]; /* res_ids base per layer */
    int            l2_locked;
    uint32_t       l2_fill;
    int32_t       *l1_map; /* n_layer * n_expert, -1 empty */
    int32_t       *l2_map;
    uint32_t       map_stride;
    uint64_t      *res_ok; /* resident fill bitmap (layer*n_exp + eid) */
    /* Per-layer resident quota: with a hit profile each layer keeps the
     * fewest top-hit experts covering the same fraction of routed traffic
     * (coverage-equalized); without a profile it's the uniform count. */
    uint32_t      *res_quota;
    /* Routing profile: route_note bumps [layer*map_stride+eid] for every
     * selected expert (any path — dispatcher, sb, eager, prefill, serial).
     * route_save writes the nonzero entries in warm-file format so a later
     * run's res_warm_parse can build a hot-set resident image. */
    uint32_t      *route_hits;
    q4_io_pool     pool;
    int            arena_registered;
    int            gpu_arena; /* HIP expert arena id (0 main / 1 MTP head) */
    /* Hybrid CPU+GPU decode: monotonically increasing H2D enqueue counter and
     * the value observed at the last copy-stream drain. A slot whose put_seq
     * is <= upload_mark has definitely landed in the GPU L1 arena. */
    _Atomic uint64_t put_ctr;
    uint64_t       upload_mark;
    /* Next-layer whole-expert prefetch: one helper thread stages every
     * expert of pf_layer through the io pool while the current layer's
     * GEMMs run on the compute stream. The pool accepts a single job at a
     * time, so pf_active serializes stage_batch callers (see the join
     * guard there). */
    pthread_t      pf_th;
    _Atomic int    pf_active;
    int            pf_rc;
    int32_t        pf_layer;
    int32_t       *pf_ids; /* 0..n_exp-1, allocated on first use */
    /* Device residency table for graph-resident routing: one i32 per
     * (layer,eid) holding the L1 slot index or -1.  Writes are enqueued on
     * the copy stream so they stay ordered with payload uploads. */
    int32_t       *devtab;      /* device array, map_stride entries */
    int64_t        dev_tok;     /* decode token counter (bumped per tick) */
    /* Static resident-image lookup for route_dev_k: one i32 per
     * (layer,eid) holding res_slot+1 for a filled resident expert, else 0.
     * res_pos/res_ok never change after the initial fill, so this table is
     * uploaded once and never rewritten. */
    int32_t       *restab;      /* device array, map_stride entries */
    /* Static L1 set: stat_q warm experts per layer live in
     * fixed L1 slots and are excluded from the DRAM image. 0 = off. */
    int            stat_on;
    uint32_t       stat_q;
    int32_t       *stat_ids;   /* packed [layer*stat_q + i] */
    uint64_t      *stat_map_b; /* bitmap layer*map_stride+eid */
    /* Deferred resident build: the image is sized against the ALLOCATED
     * GPU arena (the measured-free clamp can shrink it below the plan),
     * so the resident block runs in three steps — res_plan (warm parse +
     * static set from the real slot count), static_load, res_build
     * (image).  res_pending: open chose resident mode but the image is
     * not built yet.  res_w* stash the warm-file parse between them. */
    int            want_res;
    int            res_pending;
    int32_t       *res_wl, *res_we;
    uint32_t      *res_wh;
    uint32_t       res_wn;
    uint32_t       res_wst[Q4_MAX_LAYER], res_wcn[Q4_MAX_LAYER];
    uint64_t       res_wht[Q4_MAX_LAYER];
    /* Slots whose payload was deferred until their devtab eviction quiesced
     * for a full token; drained by q4_expert_dev_tick(). */
    struct { uint32_t s; int32_t layer, eid; } defer[512];
    int            defer_n;
    /* Decayed routing frequency per (layer, expert), fed by the verify
     * publishes (q4_expert_freq_note); freq_on gates the staging
     * admission test in io_do_one. */
    float         *freq;
    int            freq_on;
    int64_t        freq_tok;    /* dev_tok of the last freq update: a stale
                                 * table (plain decode after an MTP fallback)
                                 * must not gate staging */
    /* Keys sitting in defer[] (refill queued for the next tick): a second
     * miss before the tick must not defer the same expert into another
     * slot — the tick bound both, l1_map/devtab kept only the last, and
     * the other copy sat in L1 unreachable until LRU aged it out. */
    uint8_t       *defer_key;
    uint64_t       stage_adm, stage_rej, stage_def, stage_drop, stage_tick, stage_dup;
    /* Decode prefetch (q4_expert_dprefetch): uploads, first hits on them,
     * candidates already resident/pending, no safe victim, busy pool. */
    uint64_t       dpf_up, dpf_used, dpf_skip, dpf_novic, dpf_busy;
    int            dpf_inpass;  /* an in-pass victim was cleared this pass */
    /* Pending devtab writes (dt_push/dt_flush): applied in order by one
     * scatter kernel on the copy stream per flush. */
    pthread_mutex_t dt_mu;
    uint32_t       dt_idx[64];
    int32_t        dt_val[64];
    uint32_t       dt_n;
    /* L1 lending (q4_expert_lend_*): slots [lend_lo, n_slots) double as the
     * prompt's device scratch.  Between prompts they hold a second warm
     * tier (ranks stat_q.. of each layer's warm list, all in the DRAM
     * image) as plain LRU slots, or fixed with Q4_L1_LEND_FILL=static.  While
     * lent they are blocked (fixed, empty) and absent from devtab. */
    uint32_t       lend_lo;     /* 0 = no lendable tail */
    int            lend_out;    /* tail currently borrowed */
    int            lend_fixed;  /* refill as fixed slots */
    uint32_t       lend_q;      /* tier-2 candidates per layer */
    int32_t       *lend_ids;    /* [layer*lend_q + i], -1 none */
    uint32_t      *pf_hist;     /* [layer*512 + e] prompt token picks */
    int            pf_hist_n;   /* chunks counted since the lend began */
    /* KV paging (q4_expert_kv_*): slots [kv_lo, kv_hi) below the lend tail
     * are ordinary LRU slots that hand their VRAM to the context-sized
     * buffers as the context grows; [kv_f, kv_hi) is handed out (blocked,
     * its whole 2 MiB chunks unmapped into the pool). */
    uint32_t       kv_lo, kv_hi, kv_f;
    /* Q4_RECLAIM_CHECK: the last reclaimed range, checks left, seen-good */
    uint32_t       rc_lo, rc_hi;
    int            rc_left;
    uint8_t       *rc_ok;
};

static double expert_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Weak default; the server's strong symbol (set by its signal handler) wins. */
__attribute__((weak)) volatile sig_atomic_t q4_stop;

static int64_t pread_full(int fd, void *buf, uint64_t n, uint64_t off) {
    uint8_t *p = buf;
    uint64_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, p + got, (size_t)(n - got), (off_t)(off + got));
        if (r == 0) break;
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        got += (uint64_t)r;
    }
    return (int64_t)got;
}

q4_expert_cache *q4_expert_cache_open(const q4_gguf *g, uint32_t n_slots) {
    return q4_expert_cache_open_ex(g, n_slots, 0);
}

/* Minimal parse of a q4_engine_warm_save file for the resident-fill
 * planner: returns count and fills *lp/*ep/*hp with (layer, eid, hits)
 * arrays sorted by (layer asc, hits desc). 0/-1 = no usable warm set. */
static uint32_t res_warm_parse(const q4_gguf *g, int32_t **lp,
                               int32_t **ep, uint32_t **hp) {
    *lp = *ep = NULL; *hp = NULL;
    const char *wf = getenv("Q4_WARM_FILE");
    if (!wf || !wf[0]) return 0;
    FILE *f = fopen(wf, "rb");
    if (!f) return 0;
    struct {
        uint32_t magic, version;
        uint64_t bytes_total;
        uint32_t n_layer, n_expert, count;
    } h;
    bool ok = fread(&h, 1, sizeof(h), f) == sizeof(h) &&
              h.magic == 0x57345134u && h.version == 1u &&
              h.bytes_total == g->bytes_total && h.n_layer == g->n_layer &&
              h.n_expert == g->n_expert && h.count > 0 && h.count <= 65536;
    int32_t *ls = ok ? malloc((size_t)h.count * 4) : NULL;
    int32_t *es = ok ? malloc((size_t)h.count * 4) : NULL;
    uint32_t *hs = ok ? malloc((size_t)h.count * 4) : NULL;
    if (ls && es && hs)
        ok = fread(ls, 1, (size_t)h.count * 4, f) == (size_t)h.count * 4 &&
             fread(es, 1, (size_t)h.count * 4, f) == (size_t)h.count * 4 &&
             fread(hs, 1, (size_t)h.count * 4, f) == (size_t)h.count * 4;
    fclose(f);
    if (!ok) { free(ls); free(es); free(hs); return 0; }
    /* Sort by (layer asc, hits desc) — insertion sort is fine at ≤64K. */
    for (uint32_t i = 1; i < h.count; i++) {
        int32_t l = ls[i], e = es[i];
        uint32_t hh = hs[i];
        uint32_t j = i;
        while (j > 0 && (ls[j - 1] > l ||
                         (ls[j - 1] == l && hs[j - 1] < hh))) {
            ls[j] = ls[j - 1]; es[j] = es[j - 1]; hs[j] = hs[j - 1];
            j--;
        }
        ls[j] = l; es[j] = e; hs[j] = hh;
    }
    *lp = ls; *ep = es; *hp = hs;
    return h.count;
}

/* In the static L1 set?  Those experts live in fixed GPU slots and have
 * no image slot (res_pos stays -1, res_has false). */
static int stat_is(const q4_expert_cache *c, int32_t layer, int32_t eid) {
    if (!c || !c->stat_map_b || layer < 0 || eid < 0 ||
        (uint32_t)layer >= Q4_MAX_LAYER || (uint32_t)eid >= c->map_stride)
        return 0;
    uint64_t b = (uint64_t)layer * c->map_stride + (uint32_t)eid;
    return (int)((c->stat_map_b[b >> 6] >> (b & 63)) & 1);
}

static q4_expert_cache *open_impl(const q4_gguf *g, uint32_t n_l1,
                                  uint64_t l2_bytes, int res_mode);

q4_expert_cache *q4_expert_cache_open_arena(const q4_gguf *g, uint32_t n_l1,
                                            uint64_t l2_bytes, int arena) {
    q4_expert_cache *c = q4_expert_cache_open_ex(g, n_l1, l2_bytes);
    if (c) c->gpu_arena = arena;
    return c;
}

/* res_mode: <0 use Q4_EXPERT_RESIDENT env, 0 force off, 1 force on. */
q4_expert_cache *q4_expert_cache_open_arena_r(const q4_gguf *g, uint32_t n_l1,
                                              uint64_t l2_bytes, int arena,
                                              int res_mode) {
    q4_expert_cache *c = open_impl(g, n_l1, l2_bytes, res_mode);
    if (c) c->gpu_arena = arena;
    return c;
}

int q4_expert_cache_arena(const q4_expert_cache *c) {
    return c ? c->gpu_arena : 0;
}

static void *io_worker(void *arg);

static int io_threads_default(void) {
    const char *e = getenv("Q4_IO_THREADS");
    int v = e && e[0] ? atoi(e) : 6;
    if (v < 1) v = 1;
    if (v > 8) v = 8;
    return v;
}

static void io_pool_start(q4_expert_cache *c) {
    q4_io_pool *p = &c->pool;
    memset(p, 0, sizeof(*p));
    pthread_mutex_init(&p->job_mu, NULL);
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    p->started = 1;
    /* >=1 worker always: the resident-mode bounce path lives in io_do_one;
     * with zero workers the serial fallback cannot stage unfilled experts
     * and callers then eval experts that were never bound (Q4_IO_THREADS=1
     * measured as garbage output, nll ~8.7 — not a tuning knob failure). */
    int want = io_threads_default() - 1; /* main thread also drains */
    if (want < 1) want = 1;
    for (int i = 0; i < want; i++) {
        if (pthread_create(&p->th[p->n_th], NULL, io_worker, c) != 0)
            break;
        p->n_th++;
    }
}

static void io_pool_stop(q4_expert_cache *c) {
    q4_io_pool *p = &c->pool;
    if (!p->started) return;
    pthread_mutex_lock(&p->mu);
    p->quit = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->n_th; i++) pthread_join(p->th[i], NULL);
    p->n_th = 0;
    pthread_mutex_destroy(&p->job_mu);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    p->started = 0;
}

q4_expert_cache *q4_expert_cache_open_ex(const q4_gguf *g, uint32_t n_l1,
                                         uint64_t l2_bytes) {
    return open_impl(g, n_l1, l2_bytes, -1);
}

static q4_expert_cache *open_impl(const q4_gguf *g, uint32_t n_l1,
                                  uint64_t l2_bytes, int res_mode) {
    if (!g || n_l1 == 0 || g->per_expert_bytes == 0) return NULL;
    q4_expert_cache *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    pthread_mutex_init(&c->dt_mu, NULL);
    c->g = g;
    c->n_slots = n_l1;
    uint64_t max_ex = g->per_expert_bytes;
    for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
        q4_layer_ex *e = &c->layers[L];
        if (!q4_expert_parts(g, (int32_t)L, &e->gate, &e->up, &e->down,
                             &e->off_gate, &e->off_up, &e->off_down, &e->total))
            continue;
        if (e->total > max_ex) max_ex = e->total;
    }
    c->slot_bytes = max_ex;
    c->map_stride = g->n_expert ? g->n_expert : 512;
    /* Leave RAM for the rest of the host. The old host mirror
     * of the GPU slots, on top of the 45 GiB image, OOM-killed the box. */
    /* 4 GiB, not 10. A 10 GiB floor refused the 45 GiB image whenever
     * MemAvailable was under 55 GiB, and the old fallback then filled the
     * leftover with an L2 that swapped. 4 GiB still leaves the desktop. */
    uint64_t ram_keep = 4ull * Q4_GIB;
    {
        const char *rg = getenv("Q4_DRAM_RESERVE_GIB");
        if (rg && rg[0]) {
            int v = atoi(rg);
            if (v > 0) ram_keep = (uint64_t)v * Q4_GIB;
        }
        if (ram_keep < 4ull * Q4_GIB) ram_keep = 4ull * Q4_GIB;
    }
    uint64_t ram_avail = 0;
    malloc_trim(0);
    {
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) {
            char line[128];
            unsigned long kb = 0;
            while (fgets(line, sizeof line, mf)) {
                if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) break;
            }
            fclose(mf);
            ram_avail = (uint64_t)kb * 1024ull;
        }
    }
    uint64_t ram_budget = ram_avail > ram_keep ? ram_avail - ram_keep : 0;
    /* Host mirror of GPU slots is not allocated once experts are resident.
     * Count it only for the partial-L2 fallback. */
    uint64_t host_l1 = 0;
    int want_res = res_mode;
    if (want_res < 0) {
        const char *er0 = getenv("Q4_EXPERT_RESIDENT");
        want_res = er0 && er0[0] && er0[0] != '0';
    }
    if (!want_res) host_l1 = (uint64_t)n_l1 * max_ex;
    if (!want_res && host_l1 + (1ull << 30) > ram_budget && max_ex) {
        uint32_t fit = (uint32_t)((ram_budget > (1ull << 30) ? ram_budget - (1ull << 30) : 0) / max_ex);
        if (fit < 64) fit = 64;
        if (n_l1 > fit) {
            fprintf(stderr,
                    "q4: cap L1 %u -> %u to keep %.0f GiB RAM free\n",
                    n_l1, fit, (double)ram_keep / Q4_GIB);
            n_l1 = fit;
            c->n_slots = n_l1;
            host_l1 = (uint64_t)n_l1 * max_ex;
        }
    }
    /* Resident image + static set: deferred until the GPU L1 arena is
     * allocated — its size can shrink below the plan (measured-free
     * clamp), and both the static quota q and the image's exclusion set
     * must match the REAL slot count.  q4_expert_res_plan / static_load /
     * res_build run from the engine after the arena exists. */
    if (want_res) {
        c->want_res = 1;
        c->res_pending = 1;
    }
    uint32_t map_n = g->n_layer * c->map_stride;
    c->l1_map = malloc((size_t)map_n * sizeof(int32_t));
    c->l2_map = malloc((size_t)map_n * sizeof(int32_t));
    {
        const char *rp = getenv("Q4_ROUTE_PROF");
        if (rp && rp[0])
            c->route_hits = calloc((size_t)map_n, sizeof(uint32_t));
    }
    c->freq = calloc((size_t)map_n, sizeof(float));
    c->defer_key = calloc((size_t)map_n, 1);
    if (c->l1_map)
        for (uint32_t i = 0; i < map_n; i++) c->l1_map[i] = -1;
    if (c->l2_map)
        for (uint32_t i = 0; i < map_n; i++) c->l2_map[i] = -1;
    c->slots = calloc(n_l1, sizeof(q4_slot));
    /* Host mirror only for non-resident caches; when the resident build is
     * still pending, res_build allocates it on the fallback path. */
    if (!c->resident && !c->res_pending) {
        size_t arena = (size_t)n_l1 * (size_t)c->slot_bytes;
        arena = (arena + 4095u) & ~(size_t)4095u;
        if (posix_memalign((void **)&c->arena, 4096, arena) != 0)
            c->arena = NULL;
        if (c->arena) madvise(c->arena, arena, MADV_DONTDUMP);
    }
    if (!c->resident && !c->res_pending && l2_bytes >= c->slot_bytes) {
        c->n_l2 = (uint32_t)(l2_bytes / c->slot_bytes);
        c->l2_slots = calloc(c->n_l2, sizeof(q4_slot));
        size_t a2 = (size_t)c->n_l2 * (size_t)c->slot_bytes;
        a2 = (a2 + 4095u) & ~(size_t)4095u;
        if (posix_memalign((void **)&c->l2_arena, 4096, a2) != 0)
            c->l2_arena = NULL;
        if (c->l2_arena) {
            c->l2_arena_bytes = a2;
            /* THP + zram turned a 36 GiB L2 into incompressible 2 MiB pages
             * that leaked in zram after exit. Keep 4K pages. */
            madvise(c->l2_arena, a2, MADV_NOHUGEPAGE);
            madvise(c->l2_arena, a2, MADV_DONTFORK);
            madvise(c->l2_arena, a2, MADV_DONTDUMP);
        }
        if (c->l2_slots && c->l2_arena) {
            for (uint32_t i = 0; i < c->n_l2; i++) {
                c->l2_slots[i].layer = -1;
                c->l2_slots[i].eid = -1;
            }
        } else {
            free(c->l2_slots);
            free(c->l2_arena);
            c->l2_slots = NULL;
            c->l2_arena = NULL;
            c->n_l2 = 0;
        }
    }
    if (!c->slots || (!c->resident && !c->res_pending && !c->arena) ||
        !c->l1_map || !c->l2_map) {
        q4_expert_cache_close(c);
        return NULL;
    }
    for (uint32_t i = 0; i < n_l1; i++) {
        c->slots[i].layer = -1;
        c->slots[i].eid = -1;
        c->slots[i].dev_quiet = -1;
    }
    io_pool_start(c);
    return c;
}

static int32_t map_key(const q4_expert_cache *c, int32_t layer, int32_t eid) {
    if (layer < 0 || eid < 0 || (uint32_t)eid >= c->map_stride) return -1;
    return (int32_t)((uint32_t)layer * c->map_stride + (uint32_t)eid);
}

static int lru_keep_old(void) {   /* Q4_TICK_LRU=0: the old stamp */
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_TICK_LRU");
        v = (e && e[0] == '0') ? 1 : 0;
    }
    return v;
}

static void defer_unkey(q4_expert_cache *c, int i) {
    int32_t k = map_key(c, c->defer[i].layer, c->defer[i].eid);
    if (k >= 0 && c->defer_key) c->defer_key[k] = 0;
}

/* ---- device residency table (graph-resident MoE routing) ----------------
 * devtab[layer*stride+eid] mirrors l1_map for kernels: route_dev_k reads it
 * to split top-k picks into GPU hits and CPU misses without any host sync.
 * Writes ride the copy stream, ordered with the payload uploads:
 *   - clear old occupant BEFORE its slot's payload is overwritten,
 *   - publish new occupant AFTER its payload landed.
 * A slot whose last published mapping was cleared during the current token
 * is NOT safe to overwrite — a route kernel may have looked it up earlier
 * this token and could still be reading the payload.  Such slots are
 * "deferred": cleared now, refilled at the next token tick once the clear
 * has been invisible for a whole token. */
static uint8_t *res_ptr(const q4_expert_cache *c, int32_t layer, int32_t eid);
static int res_has(const q4_expert_cache *c, int32_t layer, int32_t eid);

int q4_expert_dev_enable(q4_expert_cache *c) {
    if (!c) return 0;
    if (c->devtab) return 1;
    if (!q4_hip_ok()) return 0;
    uint32_t map_n = c->g->n_layer * c->map_stride;
    int32_t *dt = (int32_t *)q4_hip_malloc((size_t)map_n * sizeof(int32_t));
    if (!dt) return 0;
    int32_t *h = malloc((size_t)map_n * sizeof(int32_t));
    if (!h) { q4_hip_free(dt); return 0; }
    for (uint32_t i = 0; i < map_n; i++) h[i] = -1;
    for (uint32_t i = 0; i < map_n; i++) {
        int32_t s = c->l1_map[i];
        if (s >= 0) {
            h[i] = s;
            c->slots[s].dev_vis = 1;
        }
    }
    if (!q4_hip_copy_memset(dt, 0xff, (size_t)map_n * sizeof(int32_t)) ||
        !q4_hip_copy_h2d(dt, h, (size_t)map_n * sizeof(int32_t))) {
        free(h);
        q4_hip_free(dt);
        return 0;
    }
    q4_hip_copy_wait();   /* table + pending payloads settled before use */
    free(h);
    c->devtab = dt;
    c->dev_tok = 0;
    while (c->defer_n > 0) defer_unkey(c, --c->defer_n);
    /* Static resident table: lets route_dev_k classify L1 misses into
     * GPU-read resident experts vs true CPU mailbox misses.  Value is the
     * resident slot index +1 so base+slot*stride addressing works the same
     * as the L1-hit path; res state is fixed by the initial fill. */
    if (c->resident && c->res_registered && c->res_pos && c->res_ok) {
        uint32_t n = c->g->n_layer * c->map_stride;
        int32_t *rt = (int32_t *)q4_hip_malloc((size_t)n * sizeof(int32_t));
        int32_t *rh = rt ? calloc(n, sizeof(int32_t)) : NULL;
        if (rt && rh) {
            for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER;
                 L++) {
                const q4_layer_ex *le = &c->layers[L];
                if (!le->total) continue;
                for (uint32_t e = 0; e < c->map_stride; e++) {
                    uint64_t b = (uint64_t)L * c->map_stride + e;
                    int64_t p = c->res_pos[b];
                    if (p < 0) continue;
                    if (!((c->res_ok[b >> 6] >> (b & 63)) & 1)) continue;
                    rh[b] = (int32_t)((uint64_t)(p - (int64_t)c->res_off[L])
                                      / le->total) + 1;
                }
            }
            if (q4_hip_copy_h2d(rt, rh, (size_t)n * sizeof(int32_t))) {
                q4_hip_copy_wait();
                c->restab = rt;
            } else {
                q4_hip_free(rt);
            }
        } else if (rt) {
            q4_hip_free(rt);
        }
        free(rh);
    }
    return 1;
}

int32_t *q4_expert_dev_row(q4_expert_cache *c, int32_t layer) {
    if (!c || !c->devtab || layer < 0 ||
        layer >= (int32_t)c->g->n_layer)
        return NULL;
    return c->devtab + (size_t)layer * c->map_stride;
}

/* Row of the static resident table (restab) for a layer — same indexing as
 * dev_row.  NULL when no resident image is live. */
int32_t *q4_expert_res_devrow(q4_expert_cache *c, int32_t layer) {
    if (!c || !c->restab || layer < 0 ||
        layer >= (int32_t)c->g->n_layer)
        return NULL;
    return c->restab + (size_t)layer * c->map_stride;
}

/* Slot s may be overwritten by a GPU upload this token iff its last
 * published devtab mapping became invisible before this token began. */
static int dev_overwritable(const q4_expert_cache *c, uint32_t s) {
    return !c->slots[s].dev_vis && c->slots[s].dev_quiet < c->dev_tok;
}

/* Publish/clear helpers — stream-ordered writes on the copy stream.  The
 * writes are batched: a write is enqueued at the next dt_flush (one
 * scatter kernel per flush, entries applied in push order).  A clear must
 * be flushed before its slot's payload is overwritten (dev_evict's
 * flush_now); a publish is pushed after its payload's put, so any later
 * flush keeps publish-after-payload.  Every staging entry point flushes
 * before it returns. */
static void dt_flush_locked(q4_expert_cache *c) {
    if (c->dt_n && c->devtab)
        (void)q4_hip_copy_scatter_i32(c->devtab, c->dt_idx, c->dt_val,
                                      c->dt_n);
    c->dt_n = 0;
}

static void dt_flush(q4_expert_cache *c) {
    pthread_mutex_lock(&c->dt_mu);
    dt_flush_locked(c);
    pthread_mutex_unlock(&c->dt_mu);
}

/* Flush when a clear is pending: a put must never precede a clear that a
 * racing stage batch pushed for the same slot (the dispatcher's staging can
 * overlap the main thread's tick). Pending publishes alone may wait. */
static void dt_flush_clears(q4_expert_cache *c) {
    pthread_mutex_lock(&c->dt_mu);
    for (uint32_t i = 0; i < c->dt_n; i++)
        if (c->dt_val[i] < 0) {
            dt_flush_locked(c);
            break;
        }
    pthread_mutex_unlock(&c->dt_mu);
}

static void dt_push(q4_expert_cache *c, int32_t k, int32_t v) {
    pthread_mutex_lock(&c->dt_mu);
    if (c->dt_n == 64) dt_flush_locked(c);
    c->dt_idx[c->dt_n] = (uint32_t)k;
    c->dt_val[c->dt_n] = v;
    c->dt_n++;
    pthread_mutex_unlock(&c->dt_mu);
}

static void devtab_clear(q4_expert_cache *c, int32_t layer, int32_t eid) {
    int32_t k = map_key(c, layer, eid);
    if (k >= 0) dt_push(c, k, -1);
}

static void devtab_publish(q4_expert_cache *c, uint32_t s, int32_t layer,
                           int32_t eid) {
    int32_t k = map_key(c, layer, eid);
    if (k >= 0) {
        dt_push(c, k, (int32_t)s);
        c->slots[s].dev_vis = 1;
    }
}

/* Hide the slot's CURRENT occupant from device-side lookups.  flush_now:
 * the slot's payload is about to be rewritten, so the clear must be on the
 * copy stream first; a deferred slot (refilled at a later tick) may wait
 * for the batch. */
static void dev_evict(q4_expert_cache *c, uint32_t s, int flush_now) {
    if (c->slots[s].dev_vis && c->slots[s].layer >= 0) {
        devtab_clear(c, c->slots[s].layer, c->slots[s].eid);
        c->slots[s].dev_vis = 0;
        c->slots[s].dev_quiet = c->dev_tok;
    }
    if (flush_now) dt_flush(c);
}

/* Enqueue the slot's payload and publish it afterwards.  Returns false if
 * the upload could not be queued (caller treats like a staging failure). */
static int dev_upload(q4_expert_cache *c, uint32_t s, int32_t layer,
                      int32_t eid, const uint8_t *src, uint64_t n) {
    if (s >= q4_hip_experts_n_a(c->gpu_arena)) return 0;
    dt_flush_clears(c);
    if (!q4_hip_experts_put_async_a(c->gpu_arena, s, src, n)) return 0;
    __atomic_store_n(&c->slots[s].put_seq,
                     __atomic_add_fetch(&c->put_ctr, 1, __ATOMIC_RELAXED),
                     __ATOMIC_RELEASE);
    devtab_publish(c, s, layer, eid);
    return 1;
}

/* One decode token boundary on the host (called with the compute stream
 * fully drained): quiesced evictions become safe to overwrite. */
void q4_expert_dev_tick(q4_expert_cache *c) {
    if (!c || !c->devtab) return;
    /* Clears pushed last pass go out before the drain below, so they have
     * landed before any refill overwrites a slot. */
    dt_flush(c);
    /* Decode prefetch overwrote slots whose devtab entries it cleared in
     * the same pass (layers the GPU had already passed): the next pass's
     * routes must read those clears, so the compute stream waits for the
     * copy stream here (device-side; normally already complete). */
    if (c->dpf_inpass) {
        c->dpf_inpass = 0;
        if (q4_hip_copy_fence()) (void)q4_hip_stream_wait_copy();
    }
    c->dev_tok++;
    if (c->defer_n <= 0) return;
    /* Deferred payloads may only be enqueued once no kernel can still be
     * holding a hit on a quiesced slot — the compute stream must be
     * drained.  At a real token boundary (post-logits D2H) it already is. */
    q4_hip_stream_sync();
    q4_hip_copy_wait();   /* clears enqueued last token have landed */
    int n = c->defer_n;
    c->defer_n = 0;
    int32_t last_layer = -1;
    for (int i = 0; i < n; i++) {
        uint32_t s = c->defer[i].s;
        int32_t layer = c->defer[i].layer, eid = c->defer[i].eid;
        const uint8_t *src = NULL;
        uint64_t nb = 0;
        int32_t k = map_key(c, layer, eid);
        if (k >= 0 && c->defer_key) c->defer_key[k] = 0;
        if (k >= 0 && c->l1_map[k] >= 0 && c->l1_map[k] != (int32_t)s) {
            /* bound elsewhere meanwhile (a sync stage): leave s empty */
            c->slots[s].filling = 0;
            c->slots[s].layer = -1;
            c->slots[s].eid = -1;
            continue;
        }
        if (c->resident && (uint32_t)layer < Q4_MAX_LAYER &&
            c->layers[layer].total) {
            src = res_has(c, layer, eid) ? res_ptr(c, layer, eid) : NULL;
            nb = c->layers[layer].total;
        } else if (k >= 0 && c->l2_map && c->l2_map[k] >= 0 && c->l2_arena) {
            src = c->l2_arena +
                  (size_t)c->l2_map[k] * (size_t)c->slot_bytes;
            nb = c->layers[layer].total;
        }
        /* Without a resident/L2 source the deferred entry is dropped — the
         * miss is simply re-dispatched to the CPU when routed again. */
        if (!src || !nb || !dev_overwritable(c, s)) {
            c->slots[s].filling = 0;
            continue;
        }
        int32_t okk = map_key(c, layer, eid);
        c->slots[s].layer = layer;
        c->slots[s].eid = eid;
        c->slots[s].hits = 1;
        /* Stamp the bind: the slot still carried its evicted occupant's
         * stamp — the LRU minimum that made it the victim — so every
         * deferred refill was the next victim again and only a few hundred
         * slots churned while the rest of L1 kept the prompt's experts. */
        if (!lru_keep_old()) c->slots[s].lru = ++c->clock;
        if (okk >= 0) c->l1_map[okk] = (int32_t)s;
        c->slots[s].filling = 0;
        /* One scatter per layer group: each layer's refills turn visible
         * as soon as its payloads land, not after a whole 64-entry batch. */
        if (layer != last_layer) dt_flush(c);
        last_layer = layer;
        if (dev_upload(c, s, layer, eid, src, nb)) c->stage_tick++;
    }
    dt_flush(c);
}

const uint8_t *q4_expert_cache_data(q4_expert_cache *c, int32_t layer, int32_t eid) {
    if (!c || layer < 0 || eid < 0) return NULL;
    int32_t k = map_key(c, layer, eid);
    if (k < 0) return NULL;
    int32_t s = c->l1_map[k];
    if (s < 0) return NULL;
    if (c->resident)
        return res_has(c, layer, eid) ? res_ptr(c, layer, eid) : NULL;
    if (!c->arena) return NULL;
    return c->arena + (size_t)s * (size_t)c->slot_bytes;
}

bool q4_expert_cache_pin_arena(q4_expert_cache *c) {
    if (!c || !c->arena || c->arena_registered) return false;
    size_t arena = (size_t)c->n_slots * (size_t)c->slot_bytes;
    arena = (arena + 4095u) & ~(size_t)4095u;
    if (!q4_hip_host_register(c->arena, arena)) return false;
    c->arena_registered = 1;
    return true;
}

bool q4_expert_cache_mlock_l2(q4_expert_cache *c) {
    if (!c || !c->l2_arena || !c->l2_arena_bytes || c->l2_locked) return false;
    /* 45 GiB resident + RLIMIT_MEMLOCK 8192 KB: mlock either fails or, if
     * the limit were raised, pins the whole machine. */
    if (c->resident) {
        fprintf(stderr,
                "q4: skip mlock of resident experts (%.2f GiB)\n",
                (double)c->l2_arena_bytes / (1024.0 * 1024.0 * 1024.0));
        return false;
    }
    struct rlimit r = { RLIM_INFINITY, RLIM_INFINITY };
    (void)setrlimit(RLIMIT_MEMLOCK, &r);
    if (mlock(c->l2_arena, c->l2_arena_bytes) != 0) {
        fprintf(stderr, "q4: mlock L2 failed (%s) — browsing can swap experts\n",
                strerror(errno));
        return false;
    }
    c->l2_locked = 1;
    return true;
}

void q4_expert_cache_close(q4_expert_cache *c) {
    if (!c) return;
    /* Devroute teardown first: joins the dispatcher (which reads this
     * cache) then frees its device mailbox/ring and pinned host buffers. */
    q4_moe_dev_free();
    /* An in-flight decode-time staging job on the global submitter thread
     * still reads this cache — wait it out before freeing (teardown segv
     * after an aborted decode). */
    q4_expert_stage_join(c);
    if (c->pf_active) {
        pthread_join(c->pf_th, NULL);
        c->pf_active = 0;
    }
    io_pool_stop(c);
    const char *rp = getenv("Q4_ROUTE_PROF");
    if (rp && rp[0] && c->route_hits && c->g)
        q4_expert_route_save(c, c->g, rp, stderr);
    /* Device residency tables: sync first so any in-flight kernel reading
     * them finishes before the free. */
    if (c->devtab || c->restab) {
        q4_hip_sync();
        if (c->devtab) { q4_hip_free(c->devtab); c->devtab = NULL; }
        if (c->restab) { q4_hip_free(c->restab); c->restab = NULL; }
    }
    if (c->arena_registered) q4_hip_host_unregister(c->arena);
    if (c->res_registered && c->l2_arena)
        q4_hip_host_unregister(c->l2_arena);
    if (c->l2_locked && c->l2_arena)
        munlock(c->l2_arena, c->l2_arena_bytes);
    free(c->slots);
    free(c->rc_ok);
    free(c->arena);
    free(c->l2_slots);
    free(c->l2_arena);
    free(c->l1_map);
    free(c->l2_map);
    free(c->res_ok);
    free(c->res_pos);
    free(c->res_ids);
    free(c->stat_ids);
    free(c->stat_map_b);
    free(c->lend_ids);
    free(c->pf_hist);
    free(c->res_wl);
    free(c->res_we);
    free(c->res_wh);
    free(c->res_quota);
    free(c->route_hits);
    free(c->freq);
    free(c->defer_key);
    free(c->pf_ids);
    free(c);
}

/* Q4_EVICT_LFU=1: with the routing-frequency table live, the staging
 * victim is the bound slot whose (layer, expert) was routed least lately
 * (decayed per pass), ties broken by LRU, instead of the LRU slot.
 * Replayed verify traces: ~10% fewer token-expert misses than LRU victims
 * with the same TinyLFU admission. */
static int evict_lfu_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_EVICT_LFU");
        v = e && e[0] == '1';
    }
    return v;
}

static uint32_t find_slot(q4_expert_cache *c, int32_t layer, int32_t eid) {
    int32_t k = map_key(c, layer, eid);
    if (k >= 0 && c->l1_map[k] >= 0) return (uint32_t)c->l1_map[k];
    if (evict_lfu_on() && c->freq_on && c->freq &&
        c->dev_tok - c->freq_tok <= 1) {
        uint32_t empty = UINT32_MAX, v = UINT32_MAX;
        float vf = 0.f;
        uint32_t vl = 0;
        for (uint32_t i = 0; i < c->n_slots; i++) {
            const q4_slot *sl = &c->slots[i];
            if (sl->filling || sl->fixed || sl->pinned || sl->sticky) continue;
            if (sl->layer < 0) {
                if (empty == UINT32_MAX) empty = i;
                continue;
            }
            const int32_t sk = map_key(c, sl->layer, sl->eid);
            const float f = sk >= 0 ? c->freq[sk] : 0.f;
            if (v == UINT32_MAX || f < vf || (f == vf && sl->lru < vl)) {
                v = i;
                vf = f;
                vl = sl->lru;
            }
        }
        if (empty != UINT32_MAX) return empty;
        if (v != UINT32_MAX) return v;
    }
    uint32_t empty = UINT32_MAX, victim = 0, victim_lru = UINT32_MAX;
    uint32_t free_v = UINT32_MAX, free_lru = UINT32_MAX;
    for (uint32_t i = 0; i < c->n_slots; i++) {
        if (c->slots[i].filling || c->slots[i].fixed) continue;
        if (c->slots[i].layer < 0 && empty == UINT32_MAX) empty = i;
        if (c->slots[i].lru < victim_lru) {
            victim_lru = c->slots[i].lru;
            victim = i;
        }
        if (!c->slots[i].pinned && !c->slots[i].sticky &&
            c->slots[i].lru < free_lru) {
            free_lru = c->slots[i].lru;
            free_v = i;
        }
    }
    if (empty != UINT32_MAX) return empty;
    if (free_v != UINT32_MAX) return free_v;
    return victim;
}

static void l1_bind(q4_expert_cache *c, uint32_t s, int32_t layer, int32_t eid) {
    if (c->slots[s].layer >= 0) {
        int32_t ok = map_key(c, c->slots[s].layer, c->slots[s].eid);
        if (ok >= 0) c->l1_map[ok] = -1;
    }
    c->slots[s].layer = layer;
    c->slots[s].eid = eid;
    c->slots[s].pinned = 0;
    c->slots[s].sticky = 0;
    c->slots[s].hits = 0;
    __atomic_store_n(&c->slots[s].put_seq, 0, __ATOMIC_RELEASE);
    int32_t k = map_key(c, layer, eid);
    if (k >= 0) c->l1_map[k] = (int32_t)s;
}

static const q4_tensor *find_layer_suffix(const q4_gguf *g, int32_t layer,
                                          const char *suffix) {
    char want[Q4_MAX_NAME];
    snprintf(want, sizeof(want), "blk.%d.%s", layer, suffix);
    return q4_find_tensor(g, want);
}

bool q4_expert_parts(const q4_gguf *g, int32_t layer, const q4_tensor **gate,
                     const q4_tensor **up, const q4_tensor **down,
                     uint64_t *off_gate, uint64_t *off_up, uint64_t *off_down,
                     uint64_t *total) {
    if (!g || layer < 0) return false;
    const q4_tensor *tg = find_layer_suffix(g, layer, "ffn_gate_exps.weight");
    const q4_tensor *tu = find_layer_suffix(g, layer, "ffn_up_exps.weight");
    const q4_tensor *td = find_layer_suffix(g, layer, "ffn_down_exps.weight");
    if (!tg || !tu || !td || tg->n_experts <= 0) return false;
    uint64_t sg = tg->nbytes / (uint64_t)tg->n_experts;
    uint64_t su = tu->nbytes / (uint64_t)tu->n_experts;
    uint64_t sd = td->nbytes / (uint64_t)td->n_experts;
    if (gate) *gate = tg;
    if (up) *up = tu;
    if (down) *down = td;
    if (off_gate) *off_gate = 0;
    if (off_up) *off_up = sg;
    if (off_down) *off_down = sg + su;
    if (total) *total = sg + su + sd;
    return true;
}

static int64_t load_one(const q4_gguf *g, const q4_tensor *t, int32_t eid,
                        uint8_t *dst) {
    if (!t || t->n_experts <= 0 || eid < 0 || eid >= t->n_experts) return -1;
    uint64_t slice = t->nbytes / (uint64_t)t->n_experts;
    const q4_file *f = &g->files[t->shard];
    uint64_t off = f->data_off + t->offset + slice * (uint64_t)eid;
    int64_t n = pread_full(f->fd, dst, slice, off);
    if (n < 0 || (uint64_t)n != slice) return -1;
    return (int64_t)slice;
}

/* Canonical pack: gate, up, down. Positions come from res_pos (the warm
 * set is placed first); whether a positioned expert was actually populated
 * is decided by res_ok (res_has) — a failed fill leaves its pages
 * uninitialized. */
static uint8_t *res_ptr(const q4_expert_cache *c, int32_t layer, int32_t eid) {
    if (!c || layer < 0 || (uint32_t)layer >= Q4_MAX_LAYER || eid < 0 ||
        (uint32_t)eid >= c->map_stride)
        return NULL;
    if (!c->layers[layer].total || !c->res_pos) return NULL;
    int64_t p = c->res_pos[(uint64_t)layer * c->map_stride + eid];
    if (p < 0 ||
        (uint64_t)p + c->layers[layer].total > c->l2_arena_bytes)
        return NULL;
    return c->l2_arena + p;
}

/* res_ptr bounds alone do not prove the expert was actually read in (a
 * failed prefill leaves in-bounds pages uninitialized). Callers that READ
 * resident bytes for a GPU upload must check the fill bitmap. */
static int res_has(const q4_expert_cache *c, int32_t layer, int32_t eid) {
    if (!res_ptr(c, layer, eid) || !c->res_ok) return 0;
    if (!c->g || eid >= (int32_t)c->g->n_expert) return 0;
    uint64_t b = (uint64_t)layer * c->map_stride + (uint64_t)eid;
    return (int)((c->res_ok[b >> 6] >> (b & 63)) & 1);
}

/* Zero-copy GPU view of the resident image: the arena is
 * hipHostRegisterMapped, so kernels can read expert bytes over PCIe
 * without a staging copy.  base = devptr + res_off[layer]; an expert's
 * "slot" is its index inside that layer's tight-packed region
 * (res_pos[b] = res_off[L] + k*total).  Both return <0/NULL for
 * unregistered arenas, missing layers and unfilled slots. */
const uint8_t *q4_expert_res_devbase(q4_expert_cache *c, int32_t layer) {
    if (!c || !c->res_registered || !c->l2_arena || layer < 0 ||
        (uint32_t)layer >= Q4_MAX_LAYER || !c->layers[layer].total)
        return NULL;
    uint8_t *d = (uint8_t *)q4_hip_host_devptr(c->l2_arena);
    return d ? d + c->res_off[layer] : NULL;
}

uint64_t q4_expert_res_stride(const q4_expert_cache *c, int32_t layer) {
    if (!c || layer < 0 || (uint32_t)layer >= Q4_MAX_LAYER) return 0;
    return c->layers[layer].total;
}

int32_t q4_expert_res_slot(const q4_expert_cache *c, int32_t layer,
                           int32_t eid) {
    if (!res_has(c, layer, eid)) return -1;
    int64_t off =
        c->res_pos[(uint64_t)layer * c->map_stride + eid] - c->res_off[layer];
    int64_t tot = (int64_t)c->layers[layer].total;
    if (off < 0 || tot <= 0 || (off % tot)) return -1;
    int64_t k = off / tot;
    return k <= INT32_MAX ? (int32_t)k : -1;
}

static void drop_expert_cache(const q4_gguf *g, const q4_tensor *t, int32_t eid) {
    if (!g || !t || t->n_experts <= 0) return;
    uint64_t slice = t->nbytes / (uint64_t)t->n_experts;
    const q4_file *f = &g->files[t->shard];
    uint64_t off = f->data_off + t->offset + slice * (uint64_t)eid;
    (void)posix_fadvise(f->fd, (off_t)off, (off_t)slice, POSIX_FADV_DONTNEED);
}

static int64_t load_expert_to(q4_expert_cache *c, uint8_t *dst, int32_t layer,
                              int32_t eid) {
    if ((uint32_t)layer >= Q4_MAX_LAYER || !dst) return -1;
    const q4_layer_ex *L = &c->layers[layer];
    if (!L->gate) return -1;
    if (L->total > c->slot_bytes) return -1;
    if (load_one(c->g, L->gate, eid, dst + L->off_gate) < 0) return -1;
    if (load_one(c->g, L->up, eid, dst + L->off_up) < 0) return -1;
    if (load_one(c->g, L->down, eid, dst + L->off_down) < 0) return -1;
    return (int64_t)L->total;
}

static int64_t load_expert(q4_expert_cache *c, uint32_t slot, int32_t layer,
                           int32_t eid) {
    return load_expert_to(c, c->arena + (size_t)slot * (size_t)c->slot_bytes,
                          layer, eid);
}

int64_t q4_expert_cache_touch(q4_expert_cache *c, int32_t layer, int32_t eid) {
    if (!c || layer < 0 || eid < 0) return -1;
    c->clock++;
    int32_t k = map_key(c, layer, eid);
    if (k >= 0 && c->l1_map[k] >= 0) {
        uint32_t s = (uint32_t)c->l1_map[k];
        c->slots[s].lru = c->clock;
        c->slots[s].hits++;
        c->hits++;
        return 0;
    }
    uint32_t s = find_slot(c, layer, eid);
    if (c->resident && (uint32_t)layer < Q4_MAX_LAYER &&
        c->layers[layer].total) {
        if (!res_has(c, layer, eid))
            return -1; /* unfilled: no host arena to stage through —
                          the io pool's bounce path owns these */
        /* GPU upload reads the resident image directly. No host mirror. */
        l1_bind(c, s, layer, eid);
        c->slots[s].lru = c->clock;
        c->slots[s].hits = 1;
        c->l2_hits++;
        return (int64_t)c->layers[layer].total;
    }
    /* DRAM L2 hit: memcpy into L1 staging (not SSD). */
    if (k >= 0 && c->l2_map && c->l2_map[k] >= 0 && c->l2_arena) {
        uint32_t ls = (uint32_t)c->l2_map[k];
        memcpy(c->arena + (size_t)s * (size_t)c->slot_bytes,
               c->l2_arena + (size_t)ls * (size_t)c->slot_bytes,
               (size_t)c->layers[layer].total);
        c->l2_slots[ls].lru = c->clock;
        l1_bind(c, s, layer, eid);
        c->slots[s].lru = c->clock;
        c->slots[s].hits = 1;
        c->l2_hits++;
        return (int64_t)c->layers[layer].total;
    }
    int64_t n = load_expert(c, s, layer, eid);
    if (n < 0) return -1;
    l1_bind(c, s, layer, eid);
    c->slots[s].lru = c->clock;
    c->slots[s].hits = 1;
    c->misses++;
    c->bytes_read += (uint64_t)n;
    /* keep a copy in L2 if there is a free slot */
    if (c->l2_arena && c->l2_fill < c->n_l2 && k >= 0 && c->l2_map[k] < 0) {
        uint32_t ls = c->l2_fill++;
        memcpy(c->l2_arena + (size_t)ls * (size_t)c->slot_bytes,
               c->arena + (size_t)s * (size_t)c->slot_bytes, (size_t)n);
        c->l2_slots[ls].layer = layer;
        c->l2_slots[ls].eid = eid;
        c->l2_slots[ls].lru = c->clock;
        c->l2_map[k] = (int32_t)ls;
    }
    return n;
}

/* Process one claimed batch item (a guaranteed miss). Lock held on entry;
 * released around I/O. */
static void io_do_one(q4_expert_cache *c, q4_io_pool *p, int32_t layer,
                      int32_t eid) {
    c->clock++;
    int32_t k = map_key(c, layer, eid);
    if (k >= 0 && c->defer_key && c->defer_key[k] && p->gpu_upload &&
        p->may_defer && c->devtab) {
        c->stage_dup++;
        return;   /* already queued for the next tick */
    }
    uint32_t s = find_slot(c, layer, eid);
    /* Decode staging admission (TinyLFU): a miss only displaces the LRU
     * victim when it has been routed more often lately.  The staging DMA
     * reads DRAM beside the CPU-miss jobs, and the H2D link carries ~400
     * experts per verify pass, so one-off misses must not flush experts
     * that keep coming back (replayed verify traces: 40% fewer DRAM expert
     * reads per pass, 7% fewer misses than plain LRU). */
    if (c->freq_on && c->freq && p->gpu_upload && p->may_defer &&
        c->dev_tok - c->freq_tok <= 1 && c->slots[s].layer >= 0 && k >= 0) {
        const int32_t vk = map_key(c, c->slots[s].layer, c->slots[s].eid);
        /* Q4_ADMIT_MARGIN: the candidate must lead by this much routing
         * (decayed) — with LFU victims (whose frequency is the minimum)
         * the plain test let twice the uploads through */
        static float margin = -1.f;
        if (margin < 0.f) {
            const char *e = getenv("Q4_ADMIT_MARGIN");
            margin = e ? (float)atof(e) : 0.f;
            if (margin < 0.f) margin = 0.f;
        }
        if (vk >= 0 && c->freq[k] <= c->freq[vk] + margin) {
            c->stage_rej++;
            return;
        }
    }
    if (p->gpu_upload && p->may_defer) c->stage_adm++;
    c->slots[s].filling = 1;
    /* Devroute: a slot whose devtab mapping went stale only this token is
     * unsafe to overwrite — a kernel may have looked it up already and its
     * exec could still read the payload.  Quiesce the slot (inval write on
     * the copy stream) and defer the refill to the next token tick. */
    if (p->gpu_upload && p->may_defer && c->devtab &&
        !dev_overwritable(c, s)) {
        dev_evict(c, s, 0);
        if (c->defer_n < (int)(sizeof(c->defer) / sizeof(c->defer[0]))) {
            c->defer[c->defer_n].s = s;
            c->defer[c->defer_n].layer = layer;
            c->defer[c->defer_n].eid = eid;
            c->defer_n++;
            c->stage_def++;
            if (k >= 0 && c->defer_key) c->defer_key[k] = 1;
        } else {
            c->slots[s].filling = 0;
            c->stage_drop++;
        }
        /* l1_map: hide the old binding regardless (host side is free to
         * unmap immediately — it never reads payloads). */
        if (c->slots[s].layer >= 0) {
            int32_t ok = map_key(c, c->slots[s].layer, c->slots[s].eid);
            if (ok >= 0) c->l1_map[ok] = -1;
        }
        if (c->slots[s].layer >= 0) c->slots[s].layer = -1;
        return;
    }
    /* Unmap the victim's old binding NOW: until the fill completes, a stale
     * mapping must not let another reader believe that expert is resident.
     * With devtab live, also inval the device-side entry — the clear write
     * lands on the copy stream before the payload overwrite below. */
    if (c->slots[s].layer >= 0) {
        if (c->devtab) dev_evict(c, s, 1);
        int32_t ok = map_key(c, c->slots[s].layer, c->slots[s].eid);
        if (ok >= 0) c->l1_map[ok] = -1;
    }
    uint32_t l2s = UINT32_MAX;
    if (k >= 0 && c->l2_map && c->l2_map[k] >= 0 && c->l2_arena)
        l2s = (uint32_t)c->l2_map[k];
    int gpu = p->gpu_upload;
    pthread_mutex_unlock(&p->mu);

    int from_res = c->resident && (uint32_t)layer < Q4_MAX_LAYER &&
                   res_has(c, layer, eid);
    uint8_t *tmp = NULL;
    uint8_t *dst = from_res ? res_ptr(c, layer, eid)
                     : c->arena ? c->arena + (size_t)s * (size_t)c->slot_bytes
                     : gpu ? (tmp = malloc(c->slot_bytes))  /* partial resident:
                            no host arena — bounce through a scratch buffer */
                     : NULL;
    int64_t nread;
    if (!dst) {
        free(tmp);
        pthread_mutex_lock(&p->mu);
        c->slots[s].filling = 0;
        /* !gpu under partial residency means "no host staging exists" —
         * the unfilled tail stays a CPU miss, not a batch error. A NULL
         * dst WITH gpu requested is a real malloc failure. */
        if (gpu) p->failed = 1;
        return;
    }
    if (from_res) {
        nread = (int64_t)c->layers[layer].total;
    } else if (l2s != UINT32_MAX) {
        memcpy(dst, c->l2_arena + (size_t)l2s * (size_t)c->slot_bytes,
               (size_t)c->layers[layer].total);
        nread = (int64_t)c->layers[layer].total;
    } else {
        /* unfilled resident tail (or non-resident): file read into the
         * slot buffer / bounce arena */
        nread = c->arena ? load_expert(c, s, layer, eid)
                         : load_expert_to(c, dst, layer, eid);
    }

    pthread_mutex_lock(&p->mu);
    if (nread < 0) {
        c->slots[s].filling = 0;
        p->failed = 1;
        free(tmp);
        return;
    }
    /* bind (old occupant already unmapped at claim time) */
    c->slots[s].layer = layer;
    c->slots[s].eid = eid;
    c->slots[s].pinned = 0;
    c->slots[s].sticky = 0;
    c->slots[s].hits = 1;
    c->slots[s].lru = c->clock;
    __atomic_store_n(&c->slots[s].put_seq, 0, __ATOMIC_RELEASE);
    if (k >= 0) c->l1_map[k] = (int32_t)s;
    if (from_res || l2s != UINT32_MAX) {
        if (!from_res && c->l2_slots) c->l2_slots[l2s].lru = c->clock;
        c->l2_hits++;
    } else {
        c->misses++;
        c->bytes_read += (uint64_t)nread;
        /* Rare at serve time (L2 pre-filled): keep the L2 copy. Slot s is
         * still marked filling, so no other worker can rebind it meanwhile. */
        if (c->l2_arena && c->l2_fill < c->n_l2 && k >= 0 && c->l2_map[k] < 0) {
            uint32_t ls = c->l2_fill++;
            memcpy(c->l2_arena + (size_t)ls * (size_t)c->slot_bytes, dst,
                   (size_t)nread);
            c->l2_slots[ls].layer = layer;
            c->l2_slots[ls].eid = eid;
            c->l2_slots[ls].lru = c->clock;
            c->l2_map[k] = (int32_t)ls;
        }
    }
    c->slots[s].filling = 0;
    int up_fail = 0;
    if (gpu) {
        if (s >= q4_hip_experts_n_a(c->gpu_arena)) {
            up_fail = 1;
        } else {
            pthread_mutex_unlock(&p->mu);
            bool ok;
            if (c->devtab) {
                /* Ordered devtab inval-before / publish-after. */
                ok = dev_upload(c, s, layer, eid, dst, (uint64_t)nread) != 0;
            } else {
                ok = q4_hip_experts_put_async_a(c->gpu_arena, s, dst,
                                                (uint64_t)nread);
                if (ok)
                    __atomic_store_n(&c->slots[s].put_seq,
                                     __atomic_add_fetch(&c->put_ctr, 1,
                                                        __ATOMIC_RELAXED),
                                     __ATOMIC_RELEASE);
            }
            pthread_mutex_lock(&p->mu);
            if (!ok) up_fail = 1;
        }
        if (up_fail) p->failed = 1;
    }
    if (tmp) {
        /* The bounce buffer must outlive the async H2D it fed. This path is
         * the rare unfilled-tail fill; a copy-stream drain here is fine. */
        q4_hip_copy_wait();
        free(tmp);
    }
}

static void *io_worker(void *arg) {
    q4_expert_cache *c = arg;
    q4_io_pool *p = &c->pool;
    uint32_t seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->quit && seen == p->seq)
            pthread_cond_wait(&p->cv_work, &p->mu);
        if (p->quit) break;
        seen = p->seq;
        while (p->next < p->n) {
            uint32_t i = p->next++;
            int32_t eid = p->ids[i];
            int32_t layer = p->layer;
            io_do_one(c, p, layer, eid);
            /* lock held again */
            p->done++;
            if (p->done == p->n) pthread_cond_signal(&p->cv_done);
        }
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

/* Body of stage_batch_run, called under p->job_mu. The pool accepts ONE job
 * at a time (shared ids/next/done fields); concurrent submitters — decode
 * stager, devroute dispatcher, prefetch thread, sync callers — would
 * interleave those fields and strand the waiters on cv_done. */
static int stage_batch_body(q4_expert_cache *c, int32_t layer,
                              const int32_t *ids, uint32_t n, int gpu_upload,
                              int may_defer) {
    q4_io_pool *p = &c->pool;
    if (!p->started || p->n_th <= 0) {
        /* Serial fallback: identical to the old staging loop. */
        for (uint32_t i = 0; i < n; i++) {
            int64_t nread = q4_expert_cache_touch(c, layer, ids[i]);
            if (nread < 0) {
                /* An expert that cannot be staged must never reach the GPU
                 * eval — the slot stays unbound and kernels would read
                 * stale bytes.  Fail loudly instead of silently skipping:
                 * callers above retry through the CPU dispatch or abort. */
                if (gpu_upload) return -1;
                continue; /* unfilled tail / read err → CPU miss */
            }
            if (gpu_upload && nread > 0) {
                int32_t s = q4_expert_cache_slot(c, layer, ids[i]);
                const uint8_t *pack = q4_expert_cache_data(c, layer, ids[i]);
                if (s < 0 || !pack ||
                    (uint32_t)s >= q4_hip_experts_n_a(c->gpu_arena))
                    return -1;
                if (c->devtab) {
                    /* Quiesce-then-defer applies on decode-time callers; an
                     * unsafe slot is simply left as a CPU miss until the
                     * next tick.  Sync callers (prefill, fallbacks) always
                     * take the slot — the compute stream is drained there. */
                    if (may_defer && !dev_overwritable(c, (uint32_t)s)) {
                        /* undo the fresh bind — nothing was uploaded */
                        int32_t kk = map_key(c, layer, ids[i]);
                        if (kk >= 0) c->l1_map[kk] = -1;
                        c->slots[s].layer = -1;
                        c->slots[s].eid = -1;
                        continue;
                    }
                    dev_evict(c, (uint32_t)s, 1);
                    if (!q4_hip_experts_put_async_a(c->gpu_arena,
                                                    (uint32_t)s, pack,
                                                    (uint64_t)nread))
                        return -1;
                    __atomic_store_n(&c->slots[s].put_seq,
                                     __atomic_add_fetch(&c->put_ctr, 1,
                                                        __ATOMIC_RELAXED),
                                     __ATOMIC_RELEASE);
                    devtab_publish(c, (uint32_t)s, layer, ids[i]);
                } else {
                    if (!q4_hip_experts_put_async_a(c->gpu_arena, (uint32_t)s,
                                                    pack, (uint64_t)nread))
                        return -1;
                    __atomic_store_n(&c->slots[s].put_seq,
                                     __atomic_add_fetch(&c->put_ctr, 1,
                                                        __ATOMIC_RELAXED),
                                     __ATOMIC_RELEASE);
                }
            }
        }
        return 0;
    }
    pthread_mutex_lock(&p->mu);
    /* Pass 1: resolve hits in the caller and PROTECT their slots (filling=1)
     * so miss workers can never evict an expert that this batch relies on. */
    int32_t miss_ids[1024];
    uint32_t n_miss = 0;
    for (uint32_t i = 0; i < n; i++) {
        int32_t k = map_key(c, layer, ids[i]);
        if (k >= 0 && c->l1_map[k] >= 0) {
            uint32_t s = (uint32_t)c->l1_map[k];
            c->slots[s].lru = ++c->clock;
            c->slots[s].hits++;
            c->hits++;
            c->slots[s].filling = 1;
        } else if (n_miss < 1024) {
            miss_ids[n_miss++] = ids[i];
        }
    }
    p->layer = layer;
    p->ids = miss_ids;
    p->n = n_miss;
    p->gpu_upload = gpu_upload;
    p->may_defer = may_defer;
    p->next = 0;
    p->done = 0;
    p->failed = 0;
    p->seq++;
    pthread_cond_broadcast(&p->cv_work);
    /* Main thread drains items too (it is one of the readers). */
    while (p->next < p->n) {
        uint32_t i = p->next++;
        int32_t eid = p->ids[i];
        io_do_one(c, p, layer, eid);
        p->done++;
        if (p->done == p->n) pthread_cond_signal(&p->cv_done);
    }
    while (p->done < p->n)
        pthread_cond_wait(&p->cv_done, &p->mu);
    int rc = p->failed ? -1 : 0;
    /* Unprotect the hit slots. */
    for (uint32_t i = 0; i < n; i++) {
        int32_t k = map_key(c, layer, ids[i]);
        if (k >= 0 && c->l1_map[k] >= 0) c->slots[c->l1_map[k]].filling = 0;
    }
    pthread_mutex_unlock(&p->mu);
    return rc;
}

static int stage_batch_locked(q4_expert_cache *c, int32_t layer,
                              const int32_t *ids, uint32_t n, int gpu_upload,
                              int may_defer) {
    int rc = stage_batch_body(c, layer, ids, n, gpu_upload, may_defer);
    if (c->devtab) dt_flush(c);   /* nothing stays pending past a batch */
    return rc;
}

static int stage_batch_run(q4_expert_cache *c, int32_t layer,
                           const int32_t *ids, uint32_t n, int gpu_upload,
                           int may_defer) {
    q4_io_pool *p = &c->pool;
    pthread_mutex_lock(&p->job_mu);
    int rc = stage_batch_locked(c, layer, ids, n, gpu_upload, may_defer);
    pthread_mutex_unlock(&p->job_mu);
    return rc;
}

/* Non-blocking variant for the devroute dispatcher's deferred staging: the
 * drain must never stall the completion pass (the GPU's cpx_wait spins on
 * done while the dispatcher waits), so a busy pool defers to the next pass.
 * Returns -2 when the job slot is busy, otherwise stage_batch_run's rc. */
static int stage_batch_try(q4_expert_cache *c, int32_t layer,
                           const int32_t *ids, uint32_t n, int gpu_upload,
                           int may_defer) {
    q4_io_pool *p = &c->pool;
    if (pthread_mutex_trylock(&p->job_mu)) return -2;
    int rc = stage_batch_locked(c, layer, ids, n, gpu_upload, may_defer);
    pthread_mutex_unlock(&p->job_mu);
    return rc;
}

int q4_expert_stage_batch_try(q4_expert_cache *c, int32_t layer,
                              const int32_t *ids, uint32_t n, int gpu_upload) {
    if (!c || !ids || n == 0) return 0;
    if (n > 1024) return -1;
    if (c->pf_active)
        return -2; /* prefetch holds the pool for its whole run */
    return stage_batch_try(c, layer, ids, n, gpu_upload, 1);
}

int q4_expert_stage_batch(q4_expert_cache *c, int32_t layer, const int32_t *ids,
                          uint32_t n, int gpu_upload) {
    if (!c || !ids || n == 0) return 0;
    if (n > 1024) return -1;
    /* The io pool runs one job at a time: a caller that races the prefetch
     * thread would interleave job fields, so it must wait the job out. The
     * prefetch thread itself calls stage_batch_run directly. */
    if (c->pf_active)
        q4_expert_pf_join(c);
    return stage_batch_run(c, layer, ids, n, gpu_upload, 1);
}

/* Same, but every miss is bound synchronously — required by callers that
 * consume the slots immediately (prefill meta upload, GPU fallback exec).
 * Safe because the compute stream is drained at those points. */
int q4_expert_stage_batch_sync(q4_expert_cache *c, int32_t layer,
                               const int32_t *ids, uint32_t n,
                               int gpu_upload) {
    if (!c || !ids || n == 0) return 0;
    if (n > 1024) return -1;
    if (c->pf_active)
        q4_expert_pf_join(c);
    return stage_batch_run(c, layer, ids, n, gpu_upload, 0);
}

/* Decode-time miss staging on a dedicated submitter thread: binding misses
 * into GPU L1 only matters for FUTURE tokens, so the decode hostpath posts
 * (miss ids + hit ids) and returns immediately instead of draining the io
 * pool itself (~10us per miss of critical-path time).  The submitter holds
 * the hit-slot protection across its synchronous stage_batch, so a miss
 * bind can never evict a slot the in-flight gB graph still needs; the
 * protection is released when the batch finishes.  The cache map is only
 * read lock-free from the decode thread, so every caller must join the
 * submitter before the next map lookup (stage_join below). */
typedef struct {
    q4_expert_cache *c;
    pthread_t        th;
    pthread_mutex_t  mu;
    pthread_cond_t   cv, cv_done;
    int              live, quit;
    int32_t          layer;
    int32_t          miss[64];
    uint32_t         n_miss;
    int32_t          hits[64];
    uint32_t         n_hits;
} q4_stager;

static q4_stager g_st;
static pthread_once_t g_st_once = PTHREAD_ONCE_INIT;

static void *st_worker(void *arg) {
    q4_stager *s = arg;
    pthread_mutex_lock(&s->mu);
    for (;;) {
        while (!s->quit && !s->live)
            pthread_cond_wait(&s->cv, &s->mu);
        if (s->quit) break;
        /* Run the bind batch off the decode critical path. */
        (void)stage_batch_run(s->c, s->layer, s->miss, s->n_miss, 1, 1);
        q4_expert_protect(s->c, s->layer, s->hits, s->n_hits, 0);
        s->live = 0;
        pthread_cond_signal(&s->cv_done);
    }
    pthread_mutex_unlock(&s->mu);
    return NULL;
}

static void st_init(void) {
    pthread_mutex_init(&g_st.mu, NULL);
    pthread_cond_init(&g_st.cv, NULL);
    pthread_cond_init(&g_st.cv_done, NULL);
    if (pthread_create(&g_st.th, NULL, st_worker, &g_st) != 0)
        g_st.th = 0;
}

/* Post a miss-staging batch to the submitter (joins any previous job).
 * `hits` are the experts running on the GPU this token — they stay
 * protected until the worker finishes binding the misses. */
int q4_expert_stage_async(q4_expert_cache *c, int32_t layer,
                          const int32_t *miss_ids, uint32_t n_miss,
                          const int32_t *hit_ids, uint32_t n_hits) {
    q4_stager *s = &g_st;
    pthread_once(&g_st_once, st_init);
    if (!s->th) return 0; /* no submitter: caller falls back to sync staging */
    pthread_mutex_lock(&s->mu);
    while (s->live)
        pthread_cond_wait(&s->cv_done, &s->mu);
    s->c = c;
    s->layer = layer;
    s->n_miss = n_miss > 64 ? 64 : n_miss;
    s->n_hits = n_hits > 64 ? 64 : n_hits;
    memcpy(s->miss, miss_ids, s->n_miss * sizeof(int32_t));
    memcpy(s->hits, hit_ids, s->n_hits * sizeof(int32_t));
    s->live = 1;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return 1;
}

/* Nonzero when a posted batch is still being bound: callers that hand off
 * hit-protection to the submitter must keep protecting until this clears. */
int q4_expert_stage_pending(void) {
    q4_stager *s = &g_st;
    if (!s->th) return 0;
    int live;
    pthread_mutex_lock(&s->mu);
    live = s->live;
    pthread_mutex_unlock(&s->mu);
    return live;
}

/* Wait out the submitter.  Required before any lock-free cache-map read on
 * the decode thread (the split loop's q4_expert_gpu_slot). */
void q4_expert_stage_join(q4_expert_cache *c) {
    (void)c;
    q4_stager *s = &g_st;
    pthread_mutex_lock(&s->mu);
    while (s->live)
        pthread_cond_wait(&s->cv_done, &s->mu);
    pthread_mutex_unlock(&s->mu);
}

static void *pf_worker(void *arg) {
    q4_expert_cache *c = arg;
    c->pf_rc = stage_batch_run(c, c->pf_layer, c->pf_ids, c->g->n_expert, 1,
                               0);
    return NULL;
}

/* Launch a background stage of ALL n_exp experts of `layer` (H2D via the
 * copy stream) while the current layer's compute runs. Callers must mark
 * the current layer's in-flight experts sticky first: find_slot only takes
 * them when no free victim remains, and the headroom check below keeps a
 * full sticky layer plus a whole prefetch's misses evictable at once.
 * Returns 1 when the thread is running. */
int q4_expert_pf_begin(q4_expert_cache *c, int32_t layer) {
    static int off = -1;
    if (off < 0) {
        const char *e = getenv("Q4_PF"); /* diagnostic: Q4_PF=0 kills prefetch */
        off = e && e[0] == '0';
    }
    if (off || !c || c->pf_active || layer < 0 ||
        layer >= (int32_t)c->g->n_layer)
        return 0;
    if (!c->layers[layer].gate) return 0; /* not a MoE layer */
    uint32_t n_exp = c->g->n_expert;
    if (!n_exp || n_exp > 1024) return 0;
    /* sticky(cur layer) <= n_exp and up to n_exp prefetch misses must both
     * fit beside the pins, else find_slot's last-resort victim could take
     * a sticky slot whose GEMMs are still queued. */
    uint32_t pinned = q4_expert_pinned_count(c);
    if (c->n_slots < pinned + 2 * n_exp + 64) return 0;
    if (!c->pf_ids) {
        c->pf_ids = malloc((size_t)n_exp * sizeof(int32_t));
        if (!c->pf_ids) return 0;
        for (uint32_t e = 0; e < n_exp; e++) c->pf_ids[e] = (int32_t)e;
    }
    c->pf_layer = layer;
    c->pf_rc = 0;
    /* Set before create so a racing stage_batch caller always sees the job
     * and joins instead of interleaving pool fields. */
    c->pf_active = 1;
    if (pthread_create(&c->pf_th, NULL, pf_worker, c) != 0) {
        c->pf_active = 0;
        return 0;
    }
    return 1;
}

/* Wait out a running prefetch. Returns its stage_batch status (0 ok).
 * Only the enqueue side is done at join; copies still in flight on g_copy
 * land at the caller's next q4_hip_copy_wait(). */
int q4_expert_pf_join(q4_expert_cache *c) {
    if (!c || !c->pf_active) return 0;
    pthread_join(c->pf_th, NULL);
    c->pf_active = 0;
    return c->pf_rc;
}

int32_t q4_expert_cache_slot(const q4_expert_cache *c, int32_t layer, int32_t eid) {
    if (!c) return -1;
    int32_t k = map_key(c, layer, eid);
    if (k < 0) return -1;
    return c->l1_map[k];
}

/* Record the async-H2D enqueue counter at a copy-stream drain point. Call
 * after q4_hip_copy_wait(); slots whose put_seq <= upload_mark have landed. */
void q4_expert_uploads_mark(q4_expert_cache *c) {
    if (!c) return;
    c->upload_mark = __atomic_load_n(&c->put_ctr, __ATOMIC_ACQUIRE);
}

/* L1 slot for (layer,eid) whose GPU copy has definitely landed, else -1. */
int32_t q4_expert_gpu_slot(const q4_expert_cache *c, int32_t layer,
                           int32_t eid) {
    if (!c) return -1;
    int32_t k = map_key(c, layer, eid);
    if (k < 0) return -1;
    int32_t s = c->l1_map[k];
    if (s < 0) return -1;
    uint64_t ps = __atomic_load_n(&c->slots[s].put_seq, __ATOMIC_ACQUIRE);
    if (ps == 0 || ps > c->upload_mark) return -1;
    return s;
}

/* Mark/unmark the slots bound to ids as in-flight (filling) so concurrent
 * staging cannot evict them. */
void q4_expert_protect(q4_expert_cache *c, int32_t layer, const int32_t *ids,
                       uint32_t n, int on) {
    if (!c || !ids) return;
    q4_io_pool *p = &c->pool;
    if (p->started) pthread_mutex_lock(&p->mu);
    for (uint32_t i = 0; i < n; i++) {
        int32_t k = map_key(c, layer, ids[i]);
        if (k >= 0 && c->l1_map[k] >= 0)
            c->slots[c->l1_map[k]].filling = on ? 1 : 0;
    }
    if (p->started) pthread_mutex_unlock(&p->mu);
}

/* Host pointer to the packed (gate|up|down) bytes of one expert, without
 * touching/evicting anything. Resident image first, then L1/L2 slots.
 * NULL means the caller must read rows itself (q4_expert_pread). */
const uint8_t *q4_expert_host_ptr(const q4_expert_cache *c, int32_t layer,
                                  int32_t eid) {
    if (!c || layer < 0 || eid < 0 || (uint32_t)layer >= Q4_MAX_LAYER)
        return NULL;
    if (c->resident && c->l2_arena && c->layers[layer].total)
        return res_has(c, layer, eid) ? res_ptr(c, layer, eid) : NULL;
    int32_t k = map_key(c, layer, eid);
    if (k < 0) return NULL;
    int32_t s = c->l1_map[k];
    if (s >= 0 && c->arena) return c->arena + (size_t)s * c->slot_bytes;
    if (c->l2_map && c->l2_map[k] >= 0 && c->l2_arena)
        return c->l2_arena + (size_t)c->l2_map[k] * c->slot_bytes;
    return NULL;
}

int64_t q4_expert_pread(q4_expert_cache *c, int32_t layer, int32_t eid,
                        uint8_t *dst) {
    if (!c || !dst) return -1;
    return load_expert_to(c, dst, layer, eid);
}

uint64_t q4_expert_cache_slot_bytes(const q4_expert_cache *c) {
    return c ? c->slot_bytes : 0;
}

uint32_t q4_expert_cache_n_slots(const q4_expert_cache *c) {
    return c ? c->n_slots : 0;
}

uint64_t q4_expert_packed_bytes(const q4_gguf *g) {
    if (!g) return 0;
    uint64_t max_ex = g->per_expert_bytes;
    for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
        uint64_t tot = 0;
        if (!q4_expert_parts(g, (int32_t)L, NULL, NULL, NULL, NULL, NULL, NULL,
                             &tot))
            continue;
        if (tot > max_ex) max_ex = tot;
    }
    return max_ex;
}

void q4_expert_cache_cap_slots(q4_expert_cache *c, uint32_t n) {
    if (!c || n == 0 || n >= c->n_slots) return;
    for (uint32_t i = n; i < c->n_slots; i++) {
        if (c->slots[i].layer >= 0) {
            int32_t k = map_key(c, c->slots[i].layer, c->slots[i].eid);
            if (k >= 0) c->l1_map[k] = -1;
            c->slots[i].layer = -1;
            c->slots[i].eid = -1;
            c->slots[i].pinned = 0;
        }
    }
    c->n_slots = n;
}

void q4_expert_cache_stats(const q4_expert_cache *c, uint64_t *hits,
                           uint64_t *misses, uint64_t *bytes) {
    if (hits) *hits = c ? c->hits : 0;
    if (misses) *misses = c ? (c->misses + c->l2_hits) : 0;
    if (bytes) *bytes = c ? c->bytes_read : 0;
}

void q4_expert_cache_stats_ex(const q4_expert_cache *c, uint64_t *l1_hits,
                              uint64_t *l2_hits, uint64_t *ssd_misses,
                              uint64_t *bytes) {
    if (l1_hits) *l1_hits = c ? c->hits : 0;
    if (l2_hits) *l2_hits = c ? c->l2_hits : 0;
    if (ssd_misses) *ssd_misses = c ? c->misses : 0;
    if (bytes) *bytes = c ? c->bytes_read : 0;
}

uint64_t q4_expert_l2_bytes(const q4_expert_cache *c) {
    if (!c) return 0;
    if (c->resident) return (uint64_t)c->l2_arena_bytes;
    return (uint64_t)c->n_l2 * c->slot_bytes;
}

uint32_t q4_expert_l2_filled(const q4_expert_cache *c) {
    return c ? c->l2_fill : 0;
}

/* Warm-set pin cap: ~1.5x one decode step (n_layer * top-k), never more
 * than 1/4 of L1. Q4_PIN_CAP is an exact top-N override. */
int q4_expert_static_on(const q4_expert_cache *c) {
    return c ? c->stat_on : 0;
}

/* ---- deferred resident image (res_pending caches) ----------------------
 * Order matters: the GPU L1 arena may come up smaller than the plan
 * (measured-free clamp), so the static quota and the image's exclusion
 * set are decided only once the real slot count is known:
 *   res_plan(n_gpu)  — parse the warm file, pick the static set
 *   static_load()    — stage the set into fixed L1 slots
 *   res_build()      — size/alloc the DRAM image around what loaded */

int q4_expert_res_plan(q4_expert_cache *c, uint32_t n_gpu) {
    if (!c || !c->res_pending) return 0;
    const q4_gguf *g = c->g;
    const uint32_t nexp = g->n_expert ? g->n_expert : 512;
    /* The warm/profile file is parsed up front: with hit counts the
     * per-layer quota becomes a coverage-equalized pick instead of a
     * uniform expert count, and Q4_L1_STATIC picks the static L1 set from
     * the same list. */
    c->res_wn = res_warm_parse(g, &c->res_wl, &c->res_we, &c->res_wh);
    memset(c->res_wst, 0xff, sizeof(c->res_wst));
    memset(c->res_wcn, 0, sizeof(c->res_wcn));
    memset(c->res_wht, 0, sizeof(c->res_wht));
    for (uint32_t i = 0; i < c->res_wn; i++) {
        int32_t L = c->res_wl[i];
        if (L < 0 || L >= (int32_t)Q4_MAX_LAYER) continue;
        if (c->res_wst[L] == UINT32_MAX) c->res_wst[L] = i;
        c->res_wcn[L]++;
        c->res_wht[L] += c->res_wh[i];
    }
    /* Static L1 experts: the top-stat_q warm experts per
     * layer stay on the GPU for the session and are left out of the DRAM
     * image entirely.  D = N - n_layer*q dynamic slots stays
     * >= 2*(n_exp-q)+64: prefill stages layer L and prefetches L+1
     * concurrently, so two layers of dynamic slots are needed.
     * n_gpu is the ALLOCATED arena size, not the plan. */
    uint32_t sq = 0;
    {
        const char *es = getenv("Q4_L1_STATIC");
        int want_static = !(es && es[0] == '0');
        uint32_t nlex = 0;
        for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++)
            if (c->layers[L].total) nlex++;
        if (want_static && !c->res_wn)
            fprintf(stderr,
                    "q4: L1 static off — no usable Q4_WARM_FILE\n");
        if (want_static && c->res_wn && nlex > 2 &&
            n_gpu > 2u * nexp + 64) {
            uint32_t q = (n_gpu - 2u * nexp - 64) / (nlex - 2);
            if (q >= 8) sq = q < nexp ? q : nexp;
            else
                fprintf(stderr,
                        "q4: L1 static off — q=%u < 8 "
                        "(N=%u, n_exp=%u, layers=%u)\n",
                        q, n_gpu, nexp, nlex);
        } else if (want_static && c->res_wn)
            fprintf(stderr,
                    "q4: L1 static off — N=%u GPU slots too small for the "
                    "dynamic set\n", n_gpu);
    }
    if (!sq) return 0;
    uint64_t nb = (uint64_t)g->n_layer * nexp;
    c->stat_map_b = calloc((size_t)(nb + 63) / 64, sizeof(uint64_t));
    c->stat_ids = malloc((size_t)Q4_MAX_LAYER * sq * sizeof(int32_t));
    if (!c->stat_map_b || !c->stat_ids) {
        free(c->stat_map_b); c->stat_map_b = NULL;
        free(c->stat_ids); c->stat_ids = NULL;
        return 0;
    }
    memset(c->stat_ids, 0xff, (size_t)Q4_MAX_LAYER * sq * sizeof(int32_t));
    for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
        if (!c->layers[L].total) continue;
        uint32_t m = c->res_wcn[L] < sq ? c->res_wcn[L] : sq;
        uint32_t per = 0;
        for (uint32_t i = 0; i < m; i++) {
            int32_t e = c->res_we[c->res_wst[L] + i];
            if (e < 0 || e >= (int32_t)nexp) continue;
            uint64_t b = (uint64_t)L * nexp + (uint32_t)e;
            c->stat_map_b[b >> 6] |= 1ull << (b & 63);
            c->stat_ids[(size_t)L * sq + per] = e;
            per++;
        }
    }
    c->stat_on = 1;
    c->stat_q = sq;
    /* stat_ids is per-layer sq packed; thin-profile layers have a shorter
     * tail — res_build recounts set bits after static_load drops any that
     * did not get a slot. */
    return 0;
}

int q4_expert_res_build(q4_expert_cache *c) {
    if (!c || !c->res_pending) return 0;
    const q4_gguf *g = c->g;
    c->res_pending = 0;
    int32_t *wl = c->res_wl, *we = c->res_we;
    uint32_t *wh = c->res_wh;
    uint32_t wn = c->res_wn;
    const uint32_t *wstart = c->res_wst, *wcnt = c->res_wcn;
    const uint64_t *whit = c->res_wht;
    uint32_t nexp = g->n_expert ? g->n_expert : 512;
    malloc_trim(0);
    uint64_t ram_avail = 0;
    {
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) {
            char line[128];
            unsigned long kb = 0;
            while (fgets(line, sizeof line, mf)) {
                if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) break;
            }
            fclose(mf);
            ram_avail = (uint64_t)kb * 1024ull;
        }
    }
    uint64_t ram_keep = 4ull * Q4_GIB;
    {
        const char *rg = getenv("Q4_DRAM_RESERVE_GIB");
        if (rg && rg[0]) {
            int v = atoi(rg);
            if (v > 0) ram_keep = (uint64_t)v * Q4_GIB;
        }
        if (ram_keep < 4ull * Q4_GIB) ram_keep = 4ull * Q4_GIB;
    }
    /* The resident image is pinned for DMA and VRAM overflow spills into
     * GTT; both are charged to the kernel, invisible to the memory cgroup
     * and to MemAvailable's "safe" margin once the box fills. The 2026-09-29
     * OOM dump had ~8.5 GiB unaccounted this way. Keep an extra allowance
     * on top of ram_keep for the resident gate. */
    uint64_t kern_keep = 4ull * Q4_GIB;
    {
        const char *kk = getenv("Q4_KERNEL_RESERVE_GIB");
        if (kk && kk[0]) {
            int v = atoi(kk);
            if (v >= 0) kern_keep = (uint64_t)v * Q4_GIB;
        }
    }
    /* The budget must also cover q4's own non-image RSS: dense host store,
     * KV cache, workspaces and io bounce buffers add ~6.5 GiB on top of the
     * resident image (measured: GSQ image 46.84 GiB -> RSS 53.4 GiB). */
    uint64_t host_keep = 6ull * Q4_GIB;
    {
        const char *hk = getenv("Q4_HOST_RESERVE_GIB");
        if (hk && hk[0]) {
            int v = atoi(hk);
            if (v >= 0) host_keep = (uint64_t)v * Q4_GIB;
        }
    }
    uint64_t res_budget = ram_avail > ram_keep + kern_keep + host_keep
                              ? ram_avail - ram_keep - kern_keep - host_keep
                              : 0;
    /* Q4_RES_BUDGET_GIB=<n>: force the resident-fill budget regardless of
     * MemAvailable — experiments only; the cgroup cap stays the real bound. */
    {
        const char *rb = getenv("Q4_RES_BUDGET_GIB");
        if (rb && rb[0]) {
            int v = atoi(rb);
            if (v > 0) res_budget = (uint64_t)v * Q4_GIB;
        }
    }
    /* sper = statics that actually occupy a GPU slot — static_load clears
     * the bits of any it could not place, so the image excludes only the
     * loaded set. */
    uint32_t sper[Q4_MAX_LAYER];
    memset(sper, 0, sizeof(sper));
    if (c->stat_map_b)
        for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++)
            if (c->layers[L].total)
                for (uint32_t e = 0; e < nexp; e++)
                    sper[L] += stat_is(c, (int32_t)L, (int32_t)e);
    uint64_t off = 0;
    for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
        if (c->layers[L].total)
            off += c->layers[L].total * (uint64_t)(nexp - sper[L]);
    }
    /* The MTP head is a second image (~2.5 GiB), built after the main one.
     * Q4_RES_BUDGET_GIB is the main image's cap (serve.sh sets 47) and must
     * not authorize this pin once that image has already been taken: a
     * partial head image cannot feed the wide batch, and pinning past
     * MemAvailable is the 2026-09-29 OOM. The main image already paid
     * host_keep and kern_keep, so this one only keeps the desktop floor
     * plus 1 GiB. Anything short of the whole image stays on page cache. */
    if (off && off < 4ull * Q4_GIB) {
        uint64_t head_keep = ram_keep + Q4_GIB;
        uint64_t head_budget =
            ram_avail > head_keep ? ram_avail - head_keep : 0;
        if (head_budget < off)
            res_budget = 0;
        else if (res_budget > head_budget)
            res_budget = head_budget;
    }
    /* Resident experts are uploaded straight from this image.  There is
     * no second host copy of the GPU slots.
     *
     * When the image would cross the RAM reserve, shrink it instead of
     * refusing: every layer keeps its first res_quota[L] experts, packed
     * contiguously. Misses then spread uniformly across layers (a
     * layer-prefix fill would make the tail layers miss on EVERY token)
     * and stage from the GGUF via the io pool's bounce path or the
     * CPU-expert pread fallback. */
    uint64_t fill = off <= res_budget ? off : res_budget;
    /* Page-trim only a budget that is actually short. Masking a full
     * budget makes fill < off by under 4 KiB, and the warm-file path
     * then treats that as "does not fit" and drops every cold expert
     * (mtp-res64: 40.49 of 43.64 GiB at 100% of profiled hits). */
    if (fill < off)
        fill &= ~4095ull;
    uint32_t *rq = calloc(Q4_MAX_LAYER, sizeof(uint32_t));
    uint64_t img = 0;
    if (off && rq) {
        if (wn && fill < off) {
            /* Coverage-equalized quotas: binary-search the hit fraction f
             * that makes the packed image fit. Each layer keeps the fewest
             * top-hit experts covering f of its routed traffic;
             * profile-blind layers get the uniform fallback so they don't
             * starve. */
            uint32_t uniform = nexp;
            {
                uint64_t t = off ? (fill * (uint64_t)nexp) / off : 0;
                uniform = t > nexp ? nexp : (uint32_t)t;
            }
            /* whd_* = the non-static tail of each layer's warm range: the
             * first sper[L] entries are the static set. */
            double flo = 0.0, fhi = 1.0;
            for (int it = 0; it < 24; it++) {
                double f = (flo + fhi) * 0.5;
                uint64_t need = 0;
                for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
                    const q4_layer_ex *le = &c->layers[L];
                    if (!le->total) continue;
                    uint32_t k;
                    if (wcnt[L] > sper[L]) {
                        uint32_t ws = wstart[L] + sper[L];
                        uint32_t wd = wcnt[L] - sper[L];
                        uint64_t whd = whit[L];
                        for (uint32_t i = 0; i < sper[L]; i++)
                            whd -= wh[wstart[L] + i];
                        uint64_t tgt = (uint64_t)(f * (double)whd);
                        uint64_t acc = 0;
                        k = 0;
                        while (k < wd && acc < tgt)
                            acc += wh[ws + k++];
                    } else {
                        k = uniform;
                        if (k > nexp - sper[L]) k = nexp - sper[L];
                    }
                    need += (uint64_t)k * le->total;
                }
                if (need <= fill) flo = f; else fhi = f;
            }
            uint64_t acc_h = 0, tot_h = 0;
            for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
                const q4_layer_ex *le = &c->layers[L];
                if (!le->total) continue;
                uint32_t k;
                if (wcnt[L] > sper[L]) {
                    uint32_t ws = wstart[L] + sper[L];
                    uint32_t wd = wcnt[L] - sper[L];
                    uint64_t whd = whit[L];
                    for (uint32_t i = 0; i < sper[L]; i++)
                        whd -= wh[wstart[L] + i];
                    uint64_t tgt = (uint64_t)(flo * (double)whd);
                    uint64_t acc = 0;
                    k = 0;
                    while (k < wd && acc < tgt)
                        acc += wh[ws + k++];
                    acc_h += acc; tot_h += whd;
                } else {
                    k = uniform;
                    if (k > nexp - sper[L]) k = nexp - sper[L];
                }
                if (k > nexp - sper[L]) k = nexp - sper[L];
                rq[L] = k;
            }
            fprintf(stderr,
                    "q4: resident quotas coverage-equalized "
                    "%.1f%% of routed hits (profile %u)\n",
                    tot_h ? 100.0 * (double)acc_h / (double)tot_h : 0.0, wn);
        } else {
            uint32_t q = nexp;
            if (fill < off) {
                uint64_t t = off ? (fill * (uint64_t)nexp) / off : 0;
                q = t > nexp ? nexp : (uint32_t)t;
            }
            for (uint32_t L = 0; L < Q4_MAX_LAYER; L++) {
                uint32_t d = nexp - sper[L];
                rq[L] = q > d ? d : q;
            }
        }
        uint64_t sb = 0;
        for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
            c->res_off[L] = img;
            c->res_idoff[L] = sb;
            if (c->layers[L].total) {
                img += c->layers[L].total * rq[L];
                sb += rq[L];
            }
        }
    }
    if (off && img == 0) {
        free(rq);
        free(wl); free(we); free(wh);
        c->res_wl = c->res_we = NULL; c->res_wh = NULL; c->res_wn = 0;
        fprintf(stderr,
                "q4: resident image %.2f GiB, fill budget ~0 "
                "(MemAvailable %.1f minus %.0f+%.0f+%.0f GiB "
                "reserve). Experts stay on the GGUF.\n",
                (double)off / Q4_GIB, (double)ram_avail / Q4_GIB,
                (double)ram_keep / Q4_GIB,
                (double)kern_keep / Q4_GIB,
                (double)host_keep / Q4_GIB);
        goto nonres;
    }
    if (off) {
        uint64_t nslots = 0;
        for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++)
            if (c->layers[L].total) nslots += rq[L];
        void *p = NULL;
        if (posix_memalign(&p, 4096, (size_t)img) != 0 || !p) {
            fprintf(stderr,
                    "q4: resident alloc %.2f GiB failed, partial L2\n",
                    (double)img / Q4_GIB);
            free(rq);
            free(wl); free(we); free(wh);
            c->res_wl = c->res_we = NULL; c->res_wh = NULL; c->res_wn = 0;
            goto nonres;
        }
        c->l2_arena = p;
        c->l2_arena_bytes = (size_t)img;
        c->res_quota = rq;
        c->resident = 1;
        uint64_t nb = (uint64_t)Q4_MAX_LAYER * nexp;
        c->res_ok = calloc((size_t)(nb + 63) / 64, sizeof(uint64_t));
        c->res_pos = malloc((size_t)nb * sizeof(int64_t));
        if (nslots)
            c->res_ids = malloc((size_t)nslots * sizeof(int32_t));
        if (!c->res_pos || (nslots && !c->res_ids) || !c->res_ok) {
            free(c->res_ok); c->res_ok = NULL;
            free(c->res_pos); free(c->res_ids);
            c->res_pos = NULL; c->res_ids = NULL;
            c->resident = 0;
            free(rq); c->res_quota = NULL;
            free(c->l2_arena); c->l2_arena = NULL; c->l2_arena_bytes = 0;
            free(wl); free(we); free(wh);
            c->res_wl = c->res_we = NULL; c->res_wh = NULL; c->res_wn = 0;
            goto nonres;
        }
        memset(c->res_pos, 0xff, (size_t)nb * sizeof(int64_t));
        /* Warm experts get resident slots first: the saved pin set
         * (Q4_WARM_FILE) marks which experts actually fire, so a partial
         * image holds them instead of low-numbered ones. */
        uint32_t wi = 0, wused = 0;
        for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
            const q4_layer_ex *le = &c->layers[L];
            if (!le->total) continue;
            uint64_t sbase = c->res_idoff[L];
            uint32_t k = 0;
            while (wi < wn && wl[wi] < (int32_t)L) wi++;
            for (; wi < wn && wl[wi] == (int32_t)L; wi++) {
                if (k >= rq[L]) break;
                int32_t e = we[wi];
                if (e < 0 || e >= (int32_t)nexp) continue;
                if (stat_is(c, (int32_t)L, e)) continue;
                uint64_t b = (uint64_t)L * c->map_stride + e;
                if (c->res_pos[b] >= 0) continue;
                c->res_pos[b] = (int64_t)(c->res_off[L] +
                                          (uint64_t)k * le->total);
                c->res_ids[sbase + k] = e;
                k++; wused++;
            }
            for (int32_t e = 0; e < (int32_t)nexp && k < rq[L]; e++) {
                if (stat_is(c, (int32_t)L, e)) continue;
                uint64_t b = (uint64_t)L * c->map_stride + e;
                if (c->res_pos[b] >= 0) continue;
                c->res_pos[b] = (int64_t)(c->res_off[L] +
                                          (uint64_t)k * le->total);
                c->res_ids[sbase + k] = e;
                k++;
            }
        }
        if (wn)
            fprintf(stderr,
                    "q4: resident fill prioritized %u warm experts "
                    "(of %u pinned)\n", wused, wn);
        free(wl); free(we); free(wh);
        c->res_wl = c->res_we = NULL; c->res_wh = NULL; c->res_wn = 0;
        /* 4K pages across 45 GiB cost ~4700 page walks per decode layer of
         * expert reads. THP on this buffer only (global THP=madvise would
         * be required to make this safe) is opt-in: Q4_RESIDENT_THP=1.
         * Default keeps 4K pages — 2 MiB incompressible pages leak into
         * zram under pressure. */
        const char *thp = getenv("Q4_RESIDENT_THP");
        if (thp && thp[0] == '1')
            madvise(c->l2_arena, c->l2_arena_bytes, MADV_HUGEPAGE);
        else
            madvise(c->l2_arena, c->l2_arena_bytes, MADV_NOHUGEPAGE);
        madvise(c->l2_arena, c->l2_arena_bytes, MADV_DONTFORK);
        /* Keep the ~45 GiB image out of core dumps: without this
         * systemd-coredump drops q4 cores (too big), and the image is just
         * GGUF bytes anyway. */
        madvise(c->l2_arena, c->l2_arena_bytes, MADV_DONTDUMP);
        if (img < off) {
            uint32_t qmin = nexp, qmax = 0;
            for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++) {
                if (!c->layers[L].total) continue;
                if (rq[L] < qmin) qmin = rq[L];
                if (rq[L] > qmax) qmax = rq[L];
            }
            fprintf(stderr,
                    "q4: partial resident — %.2f of %.2f GiB, "
                    "%u..%u of %u experts per layer "
                    "(MemAvailable %.1f minus %.0f+%.0f+%.0f GiB "
                    "reserve); the rest stage from the GGUF. "
                    "No fallback L2 (it thrashed before).\n",
                    (double)img / Q4_GIB, (double)off / Q4_GIB,
                    qmin, qmax, nexp, (double)ram_avail / Q4_GIB,
                    (double)ram_keep / Q4_GIB,
                    (double)kern_keep / Q4_GIB,
                    (double)host_keep / Q4_GIB);
        } else {
            if (c->stat_on) {
                uint32_t st = 0;
                for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER; L++)
                    st += sper[L];
                fprintf(stderr,
                        "q4: expert image %.2f GiB in DRAM — %u static "
                        "experts on GPU (q=%u/layer, %u dynamic slots), "
                        "%.1f GiB RAM left\n",
                        (double)img / Q4_GIB, st, c->stat_q,
                        c->n_slots > st ? c->n_slots - st : 0,
                        (double)(ram_avail - img) / Q4_GIB);
            } else
                fprintf(stderr,
                        "q4: routed experts resident %.2f GiB "
                        "(%.1f GiB RAM left)\n",
                        (double)img / Q4_GIB,
                        (double)(ram_avail - img) / Q4_GIB);
        }
    }
    return 0;
nonres:
    /* Resident path failed after the GPU arena existed: fall back to the
     * host mirror + L2 world the non-resident branch would have built. */
    free(c->stat_ids); c->stat_ids = NULL;
    free(c->stat_map_b); c->stat_map_b = NULL;
    c->stat_on = 0; c->stat_q = 0;
    {
        size_t arena = (size_t)c->n_slots * (size_t)c->slot_bytes;
        arena = (arena + 4095u) & ~(size_t)4095u;
        if (posix_memalign((void **)&c->arena, 4096, arena) != 0)
            c->arena = NULL;
        if (c->arena) madvise(c->arena, arena, MADV_DONTDUMP);
    }
    if (!c->arena) {
        fprintf(stderr, "q4: expert cache failed (host mirror alloc)\n");
        return -1;
    }
    return 0;
}

/* Startup coverage audit: every routed expert must be in exactly one of
 * {fixed GPU static slot, resident image slot, unfilled/nonresident tail}.
 * A stat bit without a bound fixed slot means the expert is in NEITHER
 * the image nor the GPU — that combination has to be impossible. */
int q4_expert_res_audit(const q4_expert_cache *c, FILE *fp) {
    if (!c || !c->g) return 0;
    const uint32_t nexp = c->g->n_expert ? c->g->n_expert : 512;
    uint32_t n_static = 0, n_img = 0, n_tail = 0, n_bad = 0;
    for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER; L++) {
        if (!c->layers[L].total) continue;
        for (uint32_t e = 0; e < nexp; e++) {
            int32_t k = map_key(c, (int32_t)L, (int32_t)e);
            int on_gpu = k >= 0 && c->l1_map[k] >= 0 &&
                         c->slots[c->l1_map[k]].fixed;
            int st = stat_is(c, (int32_t)L, (int32_t)e);
            int in_img = 0;
            if (c->res_pos) {
                int64_t p = c->res_pos[(uint64_t)L * c->map_stride + e];
                in_img = p >= 0 && c->res_ok &&
                         ((c->res_ok[((uint64_t)L * nexp + e) >> 6] >>
                           (((uint64_t)L * nexp + e) & 63)) & 1);
            }
            if (st) {
                n_static++;
                if (!on_gpu || in_img) n_bad++;   /* both or neither */
            } else if (in_img) {
                n_img++;
            } else {
                n_tail++;
            }
        }
    }
    if (fp)
        fprintf(fp,
                "q4: residency audit — %u static, %u image, %u staged, "
                "%u broken\n", n_static, n_img, n_tail, n_bad);
    return n_bad ? -1 : 0;
}

/* Load the static set into GPU L1 and freeze its slots.  Called after the
 * expert arena exists.  Experts are not in the DRAM image, so each stages
 * through the same GGUF bounce path partial-resident misses use.  If the
 * arena came up smaller than planned, only as many statics load as still
 * leave >= 2*(n_exp-q)+64 dynamic slots; the rest behave as ordinary
 * misses (slower but correct).  Returns the count actually loaded. */
uint32_t q4_expert_static_load(q4_expert_cache *c, uint32_t n_gpu) {
    if (!c || !c->stat_on || !c->stat_ids || !c->stat_q) return 0;
    const uint32_t nexp = c->g->n_expert ? c->g->n_expert : 512;
    const uint32_t need_dyn = 2u * (nexp - c->stat_q) + 64;
    uint32_t budget = n_gpu > need_dyn ? n_gpu - need_dyn : 0;
    if (!budget) {
        fprintf(stderr,
                "q4: L1 static skipped — %u GPU slots < %u needed for the "
                "dynamic set\n", n_gpu, need_dyn);
        return 0;
    }
    double t0 = expert_now();
    uint32_t loaded = 0, want = 0;
    for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER; L++) {
        if (!c->layers[L].total) continue;
        int32_t ids[512];
        uint32_t m = 0;
        for (uint32_t i = 0; i < c->stat_q && m < 512; i++) {
            int32_t e = c->stat_ids[(size_t)L * c->stat_q + i];
            if (e >= 0) ids[m++] = e;
        }
        want += m;
        if (!m || !budget) continue;
        if (m > budget) m = budget;
        if (q4_expert_stage_batch_sync(c, (int32_t)L, ids, m, 1) != 0) {
            fprintf(stderr, "q4: L1 static load failed at layer %u\n", L);
            continue;
        }
        for (uint32_t i = 0; i < m; i++) {
            int32_t k = map_key(c, (int32_t)L, ids[i]);
            if (k >= 0 && c->l1_map[k] >= 0)
                c->slots[c->l1_map[k]].fixed = 1;
        }
        loaded += m;
        budget -= m;
    }
    /* Any chosen expert that did not land a slot goes back into the DRAM
     * image: clear its static bit so res_build counts it resident again.
     * stat_set must equal the loaded set or the expert is nowhere. */
    for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER; L++) {
        if (!c->layers[L].total) continue;
        for (uint32_t i = 0; i < c->stat_q; i++) {
            int32_t e = c->stat_ids[(size_t)L * c->stat_q + i];
            if (e < 0 || e >= (int32_t)nexp) continue;
            int32_t k = map_key(c, (int32_t)L, e);
            int on = k >= 0 && c->l1_map[k] >= 0 &&
                     c->slots[c->l1_map[k]].fixed;
            if (!on) {
                uint64_t b = (uint64_t)L * nexp + (uint32_t)e;
                c->stat_map_b[b >> 6] &= ~(1ull << (b & 63));
                c->stat_ids[(size_t)L * c->stat_q + i] = -1;
            }
        }
    }
    /* The staged uploads ride the copy stream; devtab gets snapshotted
     * from l1_map later, so nothing downstream can wait on per-slot puts.
     * Drain now — decode must never read a slot whose H2D is in flight. */
    q4_hip_copy_wait();
    if (loaded < want)
        fprintf(stderr,
                "q4: L1 static %u/%u loaded — %u slots left for dynamics "
                "(rest stage on demand)\n", loaded, want, budget + 0u);
    else
        fprintf(stderr,
                "q4: L1 static %u experts fixed in %.2f s\n", loaded,
                expert_now() - t0);
    return loaded;
}

/* ---- L1 lending ----------------------------------------------------------
 * The prompt's big device scratch (prefill buffers, MMQ weight gather, GDN
 * scan) lives in the arena's last n_tail slots instead of separate
 * allocations: borrowed for the prompt, refilled with warm experts for
 * decode.  Total VRAM is the same as allocating the scratch; decode gets
 * n_tail more expert slots. */

/* Rebuild the device residency table from the published-slot state
 * (dev_vis): one H2D instead of a per-entry write for each tail slot.
 * Slots whose payload is deferred are not dev_vis, so they stay hidden. */
static int lend_devtab_sync(q4_expert_cache *c) {
    if (!c->devtab) return 1;
    uint32_t map_n = c->g->n_layer * c->map_stride;
    int32_t *h = malloc((size_t)map_n * sizeof(int32_t));
    if (!h) return 0;
    for (uint32_t i = 0; i < map_n; i++) h[i] = -1;
    for (uint32_t i = 0; i < c->n_slots; i++) {
        if (!c->slots[i].dev_vis || c->slots[i].layer < 0) continue;
        int32_t k = map_key(c, c->slots[i].layer, c->slots[i].eid);
        if (k >= 0) h[k] = (int32_t)i;
    }
    int ok = q4_hip_copy_h2d(c->devtab, h, (size_t)map_n * sizeof(int32_t));
    q4_hip_copy_wait();
    free(h);
    return ok;
}

/* Host-side quiesce for the tail: no stager post, prefetch run or pool
 * job (devroute deferred staging included) may be choosing or filling a
 * slot while the tail changes hands.  Stream drains alone do not cover
 * these — a verify's miss staging binds its slots in the io pool after
 * its H2D is queued.  Returns with job_mu held (when the pool runs). */
static void lend_quiesce(q4_expert_cache *c) {
    q4_expert_stage_join(c);
    q4_expert_pf_join(c);
    if (c->pool.started) pthread_mutex_lock(&c->pool.job_mu);
}

static void lend_unquiesce(q4_expert_cache *c) {
    if (c->pool.started) pthread_mutex_unlock(&c->pool.job_mu);
}

/* Call between res_plan and static_load with the arena's full slot count
 * already in c->n_slots: blocks the last n_tail slots (static_load and
 * staging never pick them) and picks the tier-2 candidates. */
int q4_expert_lend_init(q4_expert_cache *c, uint32_t n_tail) {
    if (!c || !n_tail || n_tail + 64 >= c->n_slots) return 0;
    const uint32_t nexp = c->g->n_expert ? c->g->n_expert : 512;
    uint32_t nlex = 0;
    for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER; L++)
        if (c->layers[L].total) nlex++;
    if (!nlex) return 0;
    c->lend_lo = c->n_slots - n_tail;
    for (uint32_t i = c->lend_lo; i < c->n_slots; i++) {
        c->slots[i].fixed = 1;
        c->slots[i].layer = -1;
        c->slots[i].eid = -1;
    }
    c->lend_out = 1; /* empty until the first q4_expert_lend_end */
    /* Default: the refilled tail is plain LRU (warm start, then adapts
     * to the request).  Fixed tier-2 measured worse: ctx-32k chat, misses
     * per verify layer 21.4 (no tail) -> 19.5 fixed / 15.4 LRU. */
    const char *f = getenv("Q4_L1_LEND_FILL");
    c->lend_fixed = f && strcmp(f, "static") == 0;
    /* Candidates: the warm ranks right after the static set, a few extra
     * per layer for experts the image left out. */
    c->lend_q = n_tail / nlex + 8;
    if (c->lend_q > nexp) c->lend_q = nexp;
    c->lend_ids = malloc((size_t)Q4_MAX_LAYER * c->lend_q * sizeof(int32_t));
    if (!c->lend_ids) return 0;
    memset(c->lend_ids, 0xff,
           (size_t)Q4_MAX_LAYER * c->lend_q * sizeof(int32_t));
    if (c->res_wl && c->res_wn) {
        for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER; L++) {
            if (!c->layers[L].total || c->res_wst[L] == UINT32_MAX) continue;
            uint32_t m = 0;
            for (uint32_t i = 0; i < c->res_wcn[L] && m < c->lend_q; i++) {
                int32_t e = c->res_we[c->res_wst[L] + i];
                if (e < 0 || e >= (int32_t)nexp) continue;
                if (i < c->stat_q && stat_is(c, (int32_t)L, e)) continue;
                c->lend_ids[(size_t)L * c->lend_q + m++] = e;
            }
        }
    }
    return 1;
}

/* ---- KV paging ----------------------------------------------------------
 * Call after lend_init and before static_load: blocks the n_pool slots
 * below the lend tail so no static (image-less) expert lands there;
 * kv_open after static_load turns them into plain LRU slots. */
int q4_expert_kv_init(q4_expert_cache *c, uint32_t n_pool) {
    const uint32_t hi = c && c->lend_lo ? c->lend_lo : (c ? c->n_slots : 0);
    if (!c || !n_pool || n_pool + 64 >= hi) return 0;
    c->kv_hi = c->kv_f = hi;
    c->kv_lo = hi - n_pool;
    for (uint32_t i = c->kv_lo; i < c->kv_hi; i++) {
        c->slots[i].fixed = 1;
        c->slots[i].layer = -1;
        c->slots[i].eid = -1;
    }
    return 1;
}

void q4_expert_kv_open(q4_expert_cache *c) {
    if (!c || !c->kv_hi) return;
    for (uint32_t i = c->kv_lo; i < c->kv_f; i++) c->slots[i].fixed = 0;
}

uint32_t q4_expert_kv_given(const q4_expert_cache *c) {
    return c ? c->kv_hi - c->kv_f : 0;
}

/* Hand n more slots (downward from the frontier) to the KV: drain, drop
 * them from l1_map/devtab, block them, unmap their whole chunks into the
 * pool.  Returns the slots handed over. */
uint32_t q4_expert_kv_donate(q4_expert_cache *c, uint32_t n) {
    if (!c || !c->kv_hi || !n) return 0;
    if (n > c->kv_f - c->kv_lo) n = c->kv_f - c->kv_lo;
    if (!n) return 0;
    const uint64_t stride = q4_hip_experts_stride_a(c->gpu_arena);
    lend_quiesce(c);
    dt_flush(c);
    q4_hip_sync();
    q4_hip_copy_wait();
    const uint32_t lo = c->kv_f - n;
    int w = 0;
    for (int i = 0; i < c->defer_n; i++)
        if (c->defer[i].s < lo || c->defer[i].s >= c->kv_f)
            c->defer[w++] = c->defer[i];
        else
            defer_unkey(c, i);
    c->defer_n = w;
    for (uint32_t i = lo; i < c->kv_f; i++) {
        q4_slot *sl = &c->slots[i];
        sl->filling = 0;
        if (sl->layer >= 0) {
            int32_t k = map_key(c, sl->layer, sl->eid);
            if (k >= 0 && c->l1_map[k] == (int32_t)i) c->l1_map[k] = -1;
        }
        sl->layer = -1;
        sl->eid = -1;
        sl->fixed = 1;
        sl->pinned = sl->sticky = 0;
        sl->dev_vis = 0;
        sl->hits = 0;
    }
    int ok = lend_devtab_sync(c);
    c->kv_f = lo;
    if (ok) ok = q4_hip_arena_donate((uint64_t)c->kv_f * stride,
                                     (uint64_t)c->kv_hi * stride);
    lend_unquiesce(c);
    return ok ? n : 0;
}

/* Take up to n slots back from the KV (the pool must hold the chunks to
 * remap): they come back empty, as LRU slots.  Returns the count. */
uint32_t q4_expert_kv_reclaim(q4_expert_cache *c, uint32_t n) {
    if (!c || !c->kv_hi || !n) return 0;
    if (n > c->kv_hi - c->kv_f) n = c->kv_hi - c->kv_f;
    const uint64_t stride = q4_hip_experts_stride_a(c->gpu_arena);
    const uint64_t hi = (uint64_t)c->kv_hi * stride;
    const uint32_t pool = q4_hip_vpool_n();
    /* largest n whose remap the pool covers (monotonic in n) */
    uint32_t lo_n = 0, hi_n = n;
    while (lo_n < hi_n) {
        uint32_t m = (lo_n + hi_n + 1) / 2;
        if (q4_hip_arena_remap_need((uint64_t)(c->kv_f + m) * stride, hi) <=
            pool)
            lo_n = m;
        else
            hi_n = m - 1;
    }
    n = lo_n;
    if (!n) return 0;
    q4_hip_sync();
    if (!q4_hip_arena_donate((uint64_t)(c->kv_f + n) * stride, hi)) {
        (void)q4_hip_arena_donate((uint64_t)c->kv_f * stride, hi);
        return 0;
    }
    {
        /* experiments: wait after the remap before any upload */
        const char *sy = getenv("Q4_RECLAIM_SYNC");
        if (sy && sy[0] == '1') q4_hip_sync();
        const char *sl = getenv("Q4_RECLAIM_SLEEP_MS");
        if (sl && atoi(sl) > 0) usleep((useconds_t)atoi(sl) * 1000);
        if (sy && sy[0] == '1') q4_hip_sync();
    }
    for (uint32_t i = c->kv_f; i < c->kv_f + n; i++) {
        c->slots[i].fixed = 0;
        c->slots[i].layer = -1;
        c->slots[i].eid = -1;
    }
    {
        const char *rc = getenv("Q4_RECLAIM_CHECK");
        if (rc && rc[0] && rc[0] != '0') {
            c->rc_lo = c->kv_f;
            c->rc_hi = c->kv_f + n;
            /* "end": only the engine's end-of-request / NaN calls check
             * (no per-step syncs); a number: that many per-step checks */
            c->rc_left = strcmp(rc, "end") == 0 ? -1 : atoi(rc) > 1 ? atoi(rc) : 48;
            free(c->rc_ok);
            c->rc_ok = calloc(n, 1);
            q4_hip_experts_watch(c->rc_lo, c->rc_hi);
        }
    }
    c->kv_f += n;
    return n;
}

/* Q4_RECLAIM_CHECK=<checks>: after a reclaim, compare the reclaimed slots
 * that hold an expert with its DRAM bytes at each call (the engine calls it
 * per decode step); a mismatch logs where in the slot / chunk it starts,
 * whether the slot was seen intact before, and whether a fresh synchronous
 * copy repairs it. */
void q4_expert_reclaim_check(q4_expert_cache *c, const char *tag, int step) {
    if (!c || !c->rc_ok) return;
    if (c->rc_left < 0) {
        if (step >= 0) return; /* per-step calls pass step >= 0 */
        step = -step - 1;
    } else if (c->rc_left == 0) {
        return;
    } else {
        c->rc_left--;
    }
    uint8_t *base = (uint8_t *)q4_hip_experts_base_a(c->gpu_arena);
    const uint64_t stride = q4_hip_experts_stride_a(c->gpu_arena);
    const uint64_t ch = q4_hip_vchunk();
    q4_hip_sync();
    q4_hip_copy_wait();
    uint8_t *buf = malloc((size_t)c->slot_bytes);
    if (!buf) return;
    uint32_t filled = 0, bad = 0;
    for (uint32_t s = c->rc_lo; s < c->rc_hi; s++) {
        const q4_slot *sl = &c->slots[s];
        if (sl->layer < 0 || sl->eid < 0 || sl->filling ||
            !res_has(c, sl->layer, sl->eid)) continue;
        const uint64_t n = c->layers[sl->layer].total;
        if (!n || n > c->slot_bytes) continue;
        if (!q4_hip_d2h(buf, base + (uint64_t)s * stride, (size_t)n)) break;
        filled++;
        const uint8_t *ref = res_ptr(c, sl->layer, sl->eid);
        if (!memcmp(buf, ref, (size_t)n)) {
            c->rc_ok[s - c->rc_lo] = 1;
            continue;
        }
        uint64_t nd = 0, f = n, l = 0;
        for (uint64_t i = 0; i < n; i++)
            if (buf[i] != ref[i]) {
                nd++;
                if (f == n) f = i;
                l = i;
            }
        const uint64_t a = (uint64_t)s * stride;
        fprintf(stderr, "q4 reclaim check %s step %d: slot %u (L%d e%d) %llu of %llu "
                "bytes differ at [%llu, %llu]; slot at chunk %llu+%llu, diff "
                "chunks %llu..%llu; %s\n", tag, step, s, sl->layer, sl->eid,
                (unsigned long long)nd, (unsigned long long)n,
                (unsigned long long)f, (unsigned long long)l,
                (unsigned long long)(a / ch), (unsigned long long)(a % ch),
                (unsigned long long)((a + f) / ch), (unsigned long long)((a + l) / ch),
                c->rc_ok[s - c->rc_lo] ? "was intact before" : "never seen intact");
        fprintf(stderr, "  got ");
        for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", buf[f + i < n ? f + i : 0]);
        fprintf(stderr, "  want ");
        for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", ref[f + i < n ? f + i : 0]);
        fputc('\n', stderr);
        if (q4_hip_h2d(base + a, ref, (size_t)n) &&
            q4_hip_d2h(buf, base + a, (size_t)n))
            fprintf(stderr, "  sync re-copy: %s\n",
                    memcmp(buf, ref, (size_t)n) ? "STILL BAD" : "repaired");
        bad++;
    }
    free(buf);
    if (bad || c->rc_left <= 0)
        fprintf(stderr, "q4 reclaim check %s step %d: %u filled of [%u, %u), %u bad, "
                "%d checks left\n", tag, step, filled, c->rc_lo, c->rc_hi, bad,
                c->rc_left);
}

/* Q4_SLOT_CHECK: every L1 slot that claims an expert with a DRAM copy
 * holds exactly that expert's bytes.  Returns the mismatches. */
static uint32_t slot_check(q4_expert_cache *c, const char *tag, int repair);

uint32_t q4_expert_slot_check(q4_expert_cache *c, const char *tag) {
    return slot_check(c, tag, 0);
}

/* The check, and every dynamic slot that differs is copied again (static
 * slots have no DRAM copy and are only reported). */
uint32_t q4_expert_slot_repair(q4_expert_cache *c, const char *tag) {
    return slot_check(c, tag, 1);
}

static uint32_t slot_check(q4_expert_cache *c, const char *tag, int repair) {
    if (!c || !c->slots) return 0;
    const uint8_t *base = q4_hip_experts_base_a(c->gpu_arena);
    const uint64_t stride = q4_hip_experts_stride_a(c->gpu_arena);
    if (!base || !stride) return 0;
    q4_hip_sync();
    q4_hip_copy_wait();
    uint8_t *buf = malloc((size_t)c->slot_bytes);
    if (!buf) return 0;
    uint32_t checked = 0, bad = 0, sbad = 0, schk = 0;
    /* slots without a DRAM copy (the static set): a checksum taken at the
     * first call, compared later */
    static uint64_t *sum0;
    static int32_t *key0;
    const int first = !sum0;
    if (first) {
        sum0 = calloc(c->n_slots, sizeof(uint64_t));
        key0 = malloc(c->n_slots * sizeof(int32_t));
        if (key0)
            for (uint32_t s = 0; s < c->n_slots; s++) key0[s] = -1;
    }
    for (uint32_t s = 0; s < c->n_slots; s++) {
        const q4_slot *sl = &c->slots[s];
        if (sl->layer < 0 || sl->eid < 0 || sl->filling) continue;
        if (!res_has(c, sl->layer, sl->eid)) {
            const uint64_t n = c->layers[sl->layer].total;
            if (!sum0 || !key0 || !n || n > c->slot_bytes) continue;
            if (!q4_hip_d2h(buf, base + (uint64_t)s * stride, (size_t)n)) break;
            uint64_t h = 1469598103934665603ull;
            for (uint64_t i = 0; i + 8 <= n; i += 8) {
                uint64_t w;
                memcpy(&w, buf + i, 8);
                h = (h ^ w) * 1099511628211ull;
            }
            const int32_t key = sl->layer * 4096 + sl->eid;
            if (key0[s] == key) {
                schk++;
                if (sum0[s] != h) {
                    if (sbad < 12)
                        fprintf(stderr, "q4 slot check %s: STATIC slot %u (L%d e%d) "
                                "changed\n", tag, s, sl->layer, sl->eid);
                    sbad++;
                }
            } else {
                key0[s] = key;
                sum0[s] = h;
            }
            continue;
        }
        const uint64_t n = c->layers[sl->layer].total;
        if (!n || n > c->slot_bytes) continue;
        if (!q4_hip_d2h(buf, base + (uint64_t)s * stride, (size_t)n)) break;
        checked++;
        if (memcmp(buf, res_ptr(c, sl->layer, sl->eid), (size_t)n)) {
            int fixed_now = 0;
            if (repair &&
                q4_hip_h2d((uint8_t *)base + (uint64_t)s * stride,
                           res_ptr(c, sl->layer, sl->eid), (size_t)n) &&
                q4_hip_d2h(buf, base + (uint64_t)s * stride, (size_t)n))
                fixed_now = !memcmp(buf, res_ptr(c, sl->layer, sl->eid), (size_t)n);
            if (bad < 12)
                fprintf(stderr, "q4 slot check %s: slot %u (L%d e%d) differs%s "
                        "[kv %u..%u frontier %u, lend %u]%s\n", tag, s, sl->layer,
                        sl->eid, sl->fixed ? " fixed" : "", c->kv_lo, c->kv_hi,
                        c->kv_f, c->lend_lo,
                        repair ? (fixed_now ? " - re-copied" : " - re-copy FAILED") : "");
            bad++;
        }
    }
    free(buf);
    fprintf(stderr, "q4 slot check %s: %u slots checked, %u differ; %u static "
            "compared, %u changed\n", tag, checked, bad, schk, sbad);
    (void)first;
    return bad + sbad;
}

/* devtab (device) against the host slots: every entry must name a slot
 * whose host state is that (layer, expert) and visible.  Returns the
 * mismatches. */
uint32_t q4_expert_devtab_check(q4_expert_cache *c, const char *tag) {
    if (!c || !c->devtab) return 0;
    q4_hip_sync();
    q4_hip_copy_wait();
    const uint32_t map_n = c->g->n_layer * c->map_stride;
    int32_t *h = malloc((size_t)map_n * sizeof(int32_t));
    if (!h || !q4_hip_d2h(h, c->devtab, (size_t)map_n * sizeof(int32_t))) {
        free(h);
        return 0;
    }
    uint32_t live = 0, bad = 0;
    for (uint32_t k = 0; k < map_n; k++) {
        const int32_t s = h[k];
        if (s < 0) continue;
        live++;
        const int32_t L = (int32_t)(k / c->map_stride), e = (int32_t)(k % c->map_stride);
        const q4_slot *sl = (uint32_t)s < c->n_slots ? &c->slots[s] : NULL;
        if (!sl || sl->layer != L || sl->eid != e || sl->filling) {
            if (bad < 12)
                fprintf(stderr, "q4 devtab check %s: L%d e%d -> slot %d, host has "
                        "L%d e%d%s%s [kv %u..%u frontier %u, lend %u]\n", tag, L, e,
                        s, sl ? sl->layer : -9, sl ? sl->eid : -9,
                        sl && sl->filling ? " filling" : "",
                        sl && sl->fixed ? " fixed" : "", c->kv_lo, c->kv_hi,
                        c->kv_f, c->lend_lo);
            bad++;
        }
    }
    free(h);
    fprintf(stderr, "q4 devtab check %s: %u entries, %u inconsistent\n", tag,
            live, bad);
    return bad;
}

int q4_expert_lend_on(const q4_expert_cache *c) {
    return c && c->lend_lo;
}

/* Borrow the tail: drain all device work, drop the tier-2 occupants from
 * l1_map/devtab and block the slots.  Returns the 64 KiB-aligned region
 * (NULL when lending is off). */
void *q4_expert_lend_begin(q4_expert_cache *c, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!c || !c->lend_lo) return NULL;
    const uint8_t *base = q4_hip_experts_base_a(c->gpu_arena);
    const uint64_t stride = q4_hip_experts_stride_a(c->gpu_arena);
    if (!base || !stride) return NULL;
    if (!c->lend_out) {
        if (c->pf_hist) {
            memset(c->pf_hist, 0, (size_t)Q4_MAX_LAYER * 512 * sizeof(uint32_t));
            c->pf_hist_n = 0;
        }
        lend_quiesce(c);
        dt_flush(c);
        q4_hip_sync();
        q4_hip_copy_wait();
        /* A deferred refill aimed at a tail slot would land in the
         * scratch: drop it (the expert is simply not resident).  Its slot
         * keeps filling=1 until the next tick — that, and a hit shield
         * (q4_expert_protect) left over from the last token, are the only
         * filling flags left once the jobs are joined and the device is
         * drained, so they are cleared, not waited on. */
        int w = 0;
        for (int i = 0; i < c->defer_n; i++)
            if (c->defer[i].s < c->lend_lo) c->defer[w++] = c->defer[i];
            else defer_unkey(c, i);
        c->defer_n = w;
        for (uint32_t i = c->lend_lo; i < c->n_slots; i++) {
            q4_slot *sl = &c->slots[i];
            sl->filling = 0;
            if (sl->layer >= 0) {
                int32_t k = map_key(c, sl->layer, sl->eid);
                if (k >= 0 && c->l1_map[k] == (int32_t)i) c->l1_map[k] = -1;
            }
            sl->layer = -1;
            sl->eid = -1;
            sl->fixed = 1;
            sl->pinned = sl->sticky = 0;
            sl->dev_vis = 0;
            sl->hits = 0;
        }
        c->lend_out = 1;
        int ok = lend_devtab_sync(c);
        lend_unquiesce(c);
        if (!ok) return NULL;
    }
    uintptr_t lo = (uintptr_t)(base + (uint64_t)c->lend_lo * stride);
    uintptr_t hi = (uintptr_t)(base + (uint64_t)c->n_slots * stride);
    uintptr_t a = (lo + 65535u) & ~(uintptr_t)65535u;
    if (a >= hi) return NULL;
    if (bytes) *bytes = hi - a;
    return (void *)a;
}

/* Return the tail: refill it from the DRAM image (tier-2 warm experts) and
 * republish.  Callers must have finished every kernel that touches the
 * scratch (this drains the device first anyway). */
/* Prompt routing histogram (Q4_LEND_PROMPT, default on): the prefill
 * adds each layer's per-expert token counts; the lend refill then loads
 * the experts this prompt used most instead of the global warm ranks —
 * the decode continues the prompt's text, so its first cycles miss less. */
static int lend_prompt_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_LEND_PROMPT");
        v = !(e && e[0] == '0');
    }
    return v;
}
void q4_expert_pf_hist_add(q4_expert_cache *c, int32_t layer,
                           const uint32_t *ek, uint32_t n_g, uint32_t n_exp) {
    if (!c || !c->lend_lo || !c->lend_out || !lend_prompt_on() || !ek ||
        layer < 0 || layer >= Q4_MAX_LAYER || n_exp > 512)
        return;
    if (!c->pf_hist) {
        c->pf_hist = calloc((size_t)Q4_MAX_LAYER * 512, sizeof(uint32_t));
        if (!c->pf_hist) return;
    }
    uint32_t *h = c->pf_hist + (size_t)layer * 512;
    for (uint32_t gno = 0; gno < n_g; gno++)
        for (uint32_t e = 0; e < n_exp; e++) h[e] += ek[gno * 512 + e];
    c->pf_hist_n++;
}

int q4_expert_lend_end(q4_expert_cache *c) {
    if (!c || !c->lend_lo || !c->lend_out) return 0;
    lend_quiesce(c);
    dt_flush(c);
    q4_hip_sync();
    q4_hip_copy_wait();
    const uint32_t nexp = c->g->n_expert ? c->g->n_expert : 512;
    uint32_t s = c->lend_lo, put = 0;
    const uint32_t n_tail = c->n_slots - c->lend_lo;
    uint32_t nlex = 0;
    for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER; L++)
        if (c->layers[L].total) nlex++;
    const uint32_t per = nlex ? n_tail / nlex : 0;
    const int use_hist = c->pf_hist && c->pf_hist_n && lend_prompt_on();
    {
        static int said;
        if (said++ < 4)
            fprintf(stderr, "q4: lend refill %s (%d layer-chunks counted)\n",
                    use_hist ? "from the prompt's routing" : "from warm ranks",
                    c->pf_hist_n);
    }
    for (uint32_t L = 0; L < c->g->n_layer && L < Q4_MAX_LAYER && c->lend_ids;
         L++) {
        if (!c->layers[L].total) continue;
        uint32_t m = 0;
        if (use_hist) {
            /* this prompt's most-picked experts first (ties: warm rank
             * order is not kept — counts only), then the warm ranks */
            const uint32_t *h = c->pf_hist + (size_t)L * 512;
            uint32_t ord[512], no = 0;
            for (uint32_t e = 0; e < nexp && e < 512; e++)
                if (h[e]) ord[no++] = e;
            for (uint32_t a = 1; a < no; a++) {
                uint32_t v = ord[a], b = a;
                while (b > 0 && h[ord[b - 1]] < h[v]) { ord[b] = ord[b - 1]; b--; }
                ord[b] = v;
            }
            for (uint32_t i = 0; i < no && m < per && s < c->n_slots; i++) {
                int32_t e = (int32_t)ord[i];
                int32_t k = map_key(c, (int32_t)L, e);
                if (k < 0 || c->l1_map[k] >= 0) continue;
                if (!res_has(c, (int32_t)L, e)) continue;
                if (!q4_hip_experts_put_async_a(c->gpu_arena, s,
                                                res_ptr(c, (int32_t)L, e),
                                                c->layers[L].total))
                    break;
                l1_bind(c, s, (int32_t)L, e);
                c->slots[s].fixed = (uint8_t)c->lend_fixed;
                c->slots[s].dev_vis = c->devtab ? 1 : 0;
                c->slots[s].dev_quiet = -1;
                s++;
                m++;
                put++;
            }
        }
        for (uint32_t i = 0; i < c->lend_q && m < per && s < c->n_slots;
             i++) {
            int32_t e = c->lend_ids[(size_t)L * c->lend_q + i];
            if (e < 0 || e >= (int32_t)nexp) continue;
            int32_t k = map_key(c, (int32_t)L, e);
            if (k < 0 || c->l1_map[k] >= 0) continue; /* already in L1 */
            if (!res_has(c, (int32_t)L, e)) continue;
            if (!q4_hip_experts_put_async_a(c->gpu_arena, s,
                                            res_ptr(c, (int32_t)L, e),
                                            c->layers[L].total))
                break;
            l1_bind(c, s, (int32_t)L, e);
            c->slots[s].fixed = (uint8_t)c->lend_fixed;
            c->slots[s].dev_vis = c->devtab ? 1 : 0;
            c->slots[s].dev_quiet = -1;
            s++;
            m++;
            put++;
        }
    }
    for (uint32_t i = s; i < c->n_slots; i++) c->slots[i].fixed = 0;
    q4_hip_copy_wait();
    c->lend_out = 0;
    int ok = lend_devtab_sync(c);
    lend_unquiesce(c);
    return ok ? (int)put : -1;
}

uint32_t q4_expert_warm_cap(uint32_t nslots, uint32_t n_layer,
                            uint32_t topk) {
    if (!nslots) return 0;
    const char *env = getenv("Q4_PIN_CAP");
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) {
            uint32_t c = (uint32_t)v;
            return c > nslots ? nslots : c;
        }
    }
    uint32_t step = n_layer * (topk ? topk : 10u);
    uint32_t cap = step + step / 2u;
    uint32_t max_pin = nslots / 4u;
    if (nslots > step && max_pin < step) max_pin = step;
    if (cap > max_pin) cap = max_pin;
    if (cap > nslots) cap = nslots;
    return cap;
}

/* Restore a saved pin set (q4_engine_warm_save format) into the cache:
 * keep the hottest `cap` entries, stage per layer, then pin. */
int q4_expert_warm_load(q4_expert_cache *c, const q4_gguf *g,
                        const char *path, uint32_t cap, FILE *fp) {
    if (!c || !g || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    struct {
        uint32_t magic, version;
        uint64_t bytes_total;
        uint32_t n_layer, n_expert, count;
    } h;
    bool ok = fread(&h, 1, sizeof(h), f) == sizeof(h) &&
              h.magic == 0x57345134u && h.version == 1u &&
              h.bytes_total == g->bytes_total && h.n_layer == g->n_layer &&
              h.n_expert == g->n_expert && h.count > 0 && h.count <= 65536;
    int32_t *layers = ok ? malloc((size_t)h.count * 4) : NULL;
    int32_t *eids = ok ? malloc((size_t)h.count * 4) : NULL;
    uint32_t *hits = ok ? malloc((size_t)h.count * 4) : NULL;
    if (!layers || !eids || !hits) ok = false;
    if (ok)
        ok = fread(layers, 1, (size_t)h.count * 4, f) == (size_t)h.count * 4 &&
             fread(eids, 1, (size_t)h.count * 4, f) == (size_t)h.count * 4 &&
             fread(hits, 1, (size_t)h.count * 4, f) == (size_t)h.count * 4;
    fclose(f);
    if (!ok) {
        free(layers); free(eids); free(hits);
        return -1;
    }
    uint32_t *ord = malloc((size_t)h.count * sizeof(uint32_t));
    if (!ord) {
        free(layers); free(eids); free(hits);
        return -1;
    }
    for (uint32_t i = 0; i < h.count; i++) ord[i] = i;
    for (uint32_t i = 1; i < h.count; i++) {
        uint32_t v = ord[i];
        uint32_t j = i;
        while (j > 0 && hits[ord[j - 1]] < hits[v]) {
            ord[j] = ord[j - 1];
            j--;
        }
        ord[j] = v;
    }
    uint32_t keep = h.count < cap ? h.count : cap;
    for (uint32_t i = 1; i < keep; i++) {
        uint32_t v = ord[i];
        uint32_t j = i;
        while (j > 0 && (layers[ord[j - 1]] > layers[v] ||
                         (layers[ord[j - 1]] == layers[v] &&
                          eids[ord[j - 1]] > eids[v]))) {
            ord[j] = ord[j - 1];
            j--;
        }
        ord[j] = v;
    }
    double t0 = expert_now();
    uint32_t pinned = 0;
    uint32_t i = 0;
    while (i < keep) {
        int32_t layer = layers[ord[i]];
        int32_t ids[512];
        uint32_t m = 0;
        while (i < keep && layers[ord[i]] == layer && m < 512)
            ids[m++] = eids[ord[i++]];
        if (q4_expert_stage_batch_sync(c, layer, ids, m, 1) != 0) {
            fprintf(stderr, "q4: warm load failed at layer %d\n", layer);
            free(ord); free(layers); free(eids); free(hits);
            return -1;
        }
        pinned += q4_expert_pin_list(c, layer, ids, m);
    }
    q4_hip_copy_wait();
    free(ord); free(layers); free(eids); free(hits);
    if (fp)
        fprintf(fp,
                "q4: warm set %u experts pinned in %.2f s (cap %u of %u)\n",
                pinned, expert_now() - t0, cap, h.count);
    return pinned ? 0 : -1;
}

int q4_expert_route_prof_on(const q4_expert_cache *c) {
    return c && c->route_hits != NULL;
}

/* GPU-side L1 hits (verify batches): refresh the slots' recency so the
 * staging victim is the least recently USED dynamic slot, not merely the
 * oldest bound one.  Lock-free on purpose: lru/clock are a heuristic and
 * a racing rebind only refreshes the slot's new occupant. */
void q4_expert_slots_touch(q4_expert_cache *c, const int32_t *s, uint32_t n) {
    if (!c || !s) return;
    for (uint32_t i = 0; i < n; i++) {
        const int32_t x = s[i];
        if (x < 0 || (uint32_t)x >= c->n_slots) continue;
        q4_slot *sl = &c->slots[x];
        if (sl->fixed) continue;
        sl->lru = ++c->clock;
        sl->hits++;
        if (sl->spec) {
            sl->spec = 0;
            c->dpf_used++;
        }
    }
}

static int dpf_age(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_DPF_AGE");
        v = e ? atoi(e) : 1500;
        if (v < 0) v = 0;
    }
    return v;
}

/* Decode prefetch: upload the experts layer T's router is predicted to
 * pick (ids, best first) while the GPU is at layer lcur = T - 1.  A
 * victim must be safe to overwrite mid-pass: quiesced (dev_overwritable),
 * or bound to a layer the GPU has already passed this pass (< lcur: the
 * publish of lcur proves every kernel of the earlier layers finished);
 * its devtab clear goes on the copy stream ahead of the payload, and the
 * next tick makes the compute stream wait for that clear.  The new
 * binding enters dpf_age() clock ticks old, so a wrong guess ages out
 * about a pass later unless a hit refreshes it.  Returns the number of
 * uploads queued, -2 if the staging pool is busy. */
int q4_expert_dprefetch(q4_expert_cache *c, int32_t lcur, int32_t T,
                        const int32_t *ids, uint32_t n) {
    if (!c || !c->devtab || !c->resident || !ids || T <= lcur || T < 0 ||
        T >= (int32_t)c->g->n_layer || !c->layers[T].total)
        return 0;
    q4_io_pool *p = &c->pool;
    if (c->pf_active || pthread_mutex_trylock(&p->job_mu)) {
        c->dpf_busy++;
        return -2;
    }
    if (n > 16) n = 16;
    uint32_t vs[16];
    int32_t ve[16];
    uint32_t nv = 0;
    pthread_mutex_lock(&p->mu);
    /* Bind every victim first and push all their devtab clears, so one
     * scatter carries them ahead of the payloads (each copy-queue kernel
     * stretches the compute queue's gaps). */
    for (uint32_t i = 0; i < n; i++) {
        const int32_t eid = ids[i];
        const int32_t k = map_key(c, T, eid);
        if (k < 0 || c->l1_map[k] >= 0 ||
            (c->defer_key && c->defer_key[k]) || !res_has(c, T, eid)) {
            c->dpf_skip++;
            continue;
        }
        uint32_t s = UINT32_MAX, vl = UINT32_MAX;
        int inpass = 0;
        for (uint32_t j = 0; j < c->n_slots; j++) {
            const q4_slot *sl = &c->slots[j];
            if (sl->filling || sl->fixed || sl->pinned || sl->sticky) continue;
            if (sl->lru >= vl) continue;
            const int ow = dev_overwritable(c, j);
            if (!ow && !(sl->layer >= 0 && sl->layer < lcur)) continue;
            s = j;
            vl = sl->lru;
            inpass = !ow;
        }
        if (s == UINT32_MAX) {
            c->dpf_novic++;
            break;
        }
        q4_slot *sl = &c->slots[s];
        if (sl->layer >= 0) {
            dev_evict(c, s, 0);
            const int32_t ok = map_key(c, sl->layer, sl->eid);
            if (ok >= 0) c->l1_map[ok] = -1;
        }
        if (inpass) c->dpf_inpass = 1;
        sl->layer = T;
        sl->eid = eid;
        sl->pinned = 0;
        sl->sticky = 0;
        sl->hits = 0;
        sl->spec = 1;
        const uint32_t age = (uint32_t)dpf_age();
        sl->lru = c->clock > age ? c->clock - age : 0;
        sl->filling = 1;
        c->l1_map[k] = (int32_t)s;
        vs[nv] = s;
        ve[nv] = eid;
        nv++;
    }
    pthread_mutex_unlock(&p->mu);
    dt_flush(c);   /* the clears, ahead of every payload below */
    int up = 0;
    for (uint32_t i = 0; i < nv; i++) {
        const uint32_t s = vs[i];
        const int ok = q4_hip_experts_put_async_a(
            c->gpu_arena, s, res_ptr(c, T, ve[i]), c->layers[T].total);
        pthread_mutex_lock(&p->mu);
        c->slots[s].filling = 0;
        if (ok) {
            __atomic_store_n(&c->slots[s].put_seq,
                             __atomic_add_fetch(&c->put_ctr, 1,
                                                __ATOMIC_RELAXED),
                             __ATOMIC_RELEASE);
            devtab_publish(c, s, T, ve[i]);
            c->dpf_up++;
            up++;
        } else {
            /* nothing queued: leave the slot empty and unmapped */
            const int32_t k = map_key(c, T, ve[i]);
            if (k >= 0) c->l1_map[k] = -1;
            c->slots[s].layer = -1;
            c->slots[s].eid = -1;
            c->slots[s].spec = 0;
        }
        pthread_mutex_unlock(&p->mu);
    }
    dt_flush(c);   /* one publish scatter after the payloads */
    pthread_mutex_unlock(&p->job_mu);
    return up;
}

/* Verify routing frequency: +1 per (layer, expert) per publish, decayed by
 * d per pass (the caller decays at each layer-0 publish).  Turns the
 * staging admission on. */
void q4_expert_freq_note(q4_expert_cache *c, int32_t layer,
                         const int32_t *eids, const float *w, uint32_t n,
                         float d) {
    if (!c || !c->freq || layer < 0 || layer >= (int32_t)c->g->n_layer)
        return;
    if (d > 0.f && layer == 0) {
        const uint32_t m = c->g->n_layer * c->map_stride;
        for (uint32_t i = 0; i < m; i++) c->freq[i] *= d;
    }
    for (uint32_t i = 0; i < n; i++) {
        const int32_t e = eids[i];
        if (e < 0 || (uint32_t)e >= c->map_stride) continue;
        c->freq[(uint32_t)layer * c->map_stride + (uint32_t)e] += w ? w[i] : 1.f;
    }
    c->freq_on = 1;
    c->freq_tok = c->dev_tok;
}

void q4_expert_stage_stats(const q4_expert_cache *c, uint64_t *v) {
    v[0] = c ? c->stage_adm : 0;
    v[1] = c ? c->stage_rej : 0;
    v[2] = c ? c->stage_def : 0;
    v[3] = c ? c->stage_drop : 0;
    v[4] = c ? c->stage_tick : 0;
    v[5] = c ? c->stage_dup : 0;
    v[6] = c ? c->dpf_up : 0;
    v[7] = c ? c->dpf_used : 0;
    v[8] = c ? c->dpf_skip : 0;
    v[9] = c ? c->dpf_novic : 0;
    v[10] = c ? c->dpf_busy : 0;
}

/* Route-trace helpers (Q4_ROUTE_TRACE): occupant of slot s, and a dump of
 * every slot as (layer, eid, fixed) int16 triples. */
int32_t q4_expert_slot_eid(const q4_expert_cache *c, int32_t s,
                           int32_t *layer) {
    if (!c || s < 0 || (uint32_t)s >= c->n_slots) return -1;
    if (layer) *layer = c->slots[s].layer;
    return c->slots[s].eid;
}

uint32_t q4_expert_slot_dump(const q4_expert_cache *c, int16_t *out,
                             uint32_t cap) {
    if (!c || !out) return 0;
    uint32_t n = c->n_slots < cap ? c->n_slots : cap;
    for (uint32_t i = 0; i < n; i++) {
        out[3 * i] = (int16_t)c->slots[i].layer;
        out[3 * i + 1] = (int16_t)c->slots[i].eid;
        out[3 * i + 2] = (int16_t)c->slots[i].fixed;
    }
    return n;
}

void q4_expert_route_note(q4_expert_cache *c, int32_t layer,
                          const int32_t *ids, int n) {
    if (!c || !c->route_hits || !ids || layer < 0 ||
        (uint32_t)layer >= Q4_MAX_LAYER)
        return;
    for (int i = 0; i < n; i++) {
        int32_t e = ids[i];
        if (e < 0 || (uint32_t)e >= c->map_stride) continue;
        uint64_t b = (uint64_t)layer * c->map_stride + (uint32_t)e;
        __atomic_add_fetch(&c->route_hits[b], 1u, __ATOMIC_RELAXED);
    }
}

int q4_expert_route_save(const q4_expert_cache *c, const q4_gguf *g,
                         const char *path, FILE *fp) {
    if (!c || !c->route_hits || !g || !path) return -1;
    uint32_t nl = g->n_layer, nx = c->map_stride;
    uint32_t count = 0;
    for (uint32_t L = 0; L < nl && L < Q4_MAX_LAYER; L++)
        for (uint32_t e = 0; e < nx; e++)
            if (c->route_hits[(uint64_t)L * nx + e]) count++;
    if (!count || count > 65536) {
        if (fp) fprintf(fp, "q4: route profile: %u nonzero (skip save)\n",
                        count);
        return -1;
    }
    int32_t *layers = malloc((size_t)count * 4);
    int32_t *eids = malloc((size_t)count * 4);
    uint32_t *hits = malloc((size_t)count * 4);
    if (!layers || !eids || !hits) {
        free(layers); free(eids); free(hits);
        return -1;
    }
    uint32_t w = 0;
    for (uint32_t L = 0; L < nl && L < Q4_MAX_LAYER; L++)
        for (uint32_t e = 0; e < nx; e++) {
            uint32_t h = c->route_hits[(uint64_t)L * nx + e];
            if (h) { layers[w] = (int32_t)L; eids[w] = (int32_t)e;
                     hits[w++] = h; }
        }
    struct {
        uint32_t magic, version;
        uint64_t bytes_total;
        uint32_t n_layer, n_expert, count;
    } hd = {0x57345134u, 1u, g->bytes_total, nl, g->n_expert, count};
    char tmp[4096 + 16];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    bool ok = f && fwrite(&hd, 1, sizeof(hd), f) == sizeof(hd) &&
              fwrite(layers, 1, (size_t)count * 4, f) == (size_t)count * 4 &&
              fwrite(eids, 1, (size_t)count * 4, f) == (size_t)count * 4 &&
              fwrite(hits, 1, (size_t)count * 4, f) == (size_t)count * 4;
    if (f) fclose(f);
    int rc = -1;
    if (ok && rename(tmp, path) == 0) rc = 0;
    else if (f) remove(tmp);
    if (fp)
        fprintf(fp, "q4: route profile %s -> %s (%u experts)\n",
                rc == 0 ? "saved" : "FAILED", path, count);
    free(layers); free(eids); free(hits);
    return rc;
}

uint32_t q4_expert_pinned_count(const q4_expert_cache *c) {
    if (!c) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < c->n_slots; i++)
        if (c->slots[i].layer >= 0 && c->slots[i].pinned) n++;
    return n;
}

uint32_t q4_expert_pin_occupied_cap(q4_expert_cache *c, uint32_t max_pinned) {
    return q4_expert_pin_hottest(c, max_pinned);
}

void q4_expert_clear_sticky(q4_expert_cache *c) {
    if (!c) return;
    for (uint32_t i = 0; i < c->n_slots; i++) c->slots[i].sticky = 0;
}

void q4_expert_sticky(q4_expert_cache *c, int32_t layer, const int32_t *ids,
                      uint32_t n) {
    if (!c || !ids || n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        int32_t k = map_key(c, layer, ids[i]);
        if (k < 0) continue;
        int32_t s = c->l1_map[k];
        if (s >= 0) c->slots[s].sticky = 1;
    }
}

/* Clear every pin, then rank occupied slots by hits descending. */
static uint32_t rank_occupied(q4_expert_cache *c, uint32_t **idx_out) {
    *idx_out = NULL;
    if (!c) return 0;
    uint32_t *idx = malloc((size_t)c->n_slots * sizeof(uint32_t));
    if (!idx) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < c->n_slots; i++) {
        c->slots[i].pinned = 0;
        if (c->slots[i].layer >= 0 && !c->slots[i].fixed) idx[n++] = i;
    }
    for (uint32_t i = 1; i < n; i++) {
        uint32_t v = idx[i];
        uint32_t j = i;
        while (j > 0 && c->slots[idx[j - 1]].hits < c->slots[v].hits) {
            idx[j] = idx[j - 1];
            j--;
        }
        idx[j] = v;
    }
    *idx_out = idx;
    return n;
}

uint32_t q4_expert_pin_hottest(q4_expert_cache *c, uint32_t max_pinned) {
    if (!c || max_pinned == 0) return 0;
    uint32_t *idx = NULL;
    uint32_t n = rank_occupied(c, &idx);
    if (!idx) return 0;
    uint32_t pin = n < max_pinned ? n : max_pinned;
    for (uint32_t i = 0; i < pin; i++) c->slots[idx[i]].pinned = 1;
    free(idx);
    return pin;
}

uint32_t q4_expert_pin_stable(q4_expert_cache *c, uint32_t max_pinned,
                              uint32_t hottest_div) {
    if (!c || max_pinned == 0) return 0;
    if (hottest_div == 0) hottest_div = 8;
    uint32_t *idx = NULL;
    uint32_t n = rank_occupied(c, &idx);
    if (!idx) return 0;
    uint32_t hottest = n ? c->slots[idx[0]].hits : 0;
    uint32_t min_hits = hottest / hottest_div;
    if (min_hits < 2) min_hits = 2;
    uint32_t pin = 0;
    for (uint32_t i = 0; i < n && pin < max_pinned; i++) {
        if (c->slots[idx[i]].hits < min_hits) break;
        c->slots[idx[i]].pinned = 1;
        pin++;
    }
    free(idx);
    return pin;
}

/* O(n_slots) re-pin for the decode loop: promote every slot whose hit
 * count is within hottest/div of the leader, without re-ranking. Unlike
 * pin_stable this never clears existing pins — it only promotes, so it is
 * safe to run mid-decode at token boundaries. */
uint32_t q4_expert_pin_thresh(q4_expert_cache *c, uint32_t max_pinned,
                              uint32_t hottest_div) {
    if (!c || max_pinned == 0) return 0;
    uint32_t hottest = 0, pin = 0;
    for (uint32_t i = 0; i < c->n_slots; i++)
        if (c->slots[i].layer >= 0 && !c->slots[i].fixed &&
            c->slots[i].hits > hottest)
            hottest = c->slots[i].hits;
    uint32_t min_hits = hottest / (hottest_div ? hottest_div : 8);
    if (min_hits < 2) min_hits = 2;
    for (uint32_t i = 0; i < c->n_slots && pin < max_pinned; i++) {
        if (c->slots[i].fixed) continue;
        if (c->slots[i].layer >= 0 && c->slots[i].hits >= min_hits &&
            !c->slots[i].pinned) {
            c->slots[i].pinned = 1;
            pin++;
        } else if (c->slots[i].pinned)
            pin++;
    }
    return pin;
}

int q4_expert_is_resident(const q4_expert_cache *c) {
    return c && c->resident;
}

uint32_t q4_expert_pin_occupied(q4_expert_cache *c) {
    return q4_expert_pin_occupied_cap(c, UINT32_MAX);
}

uint32_t q4_expert_pin_list(q4_expert_cache *c, int32_t layer,
                            const int32_t *ids, uint32_t n) {
    if (!c || !ids) return 0;
    uint32_t done = 0;
    for (uint32_t i = 0; i < n; i++) {
        int32_t k = map_key(c, layer, ids[i]);
        if (k < 0) continue;
        int32_t s = c->l1_map[k];
        if (s >= 0 && !c->slots[s].pinned) {
            c->slots[s].pinned = 1;
            done++;
        }
    }
    return done;
}

uint32_t q4_expert_pinned_list(const q4_expert_cache *c, int32_t *layers,
                               int32_t *eids, uint32_t *hits, uint32_t max) {
    if (!c) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < c->n_slots && n < max; i++) {
        if (c->slots[i].layer >= 0 && c->slots[i].pinned) {
            if (layers) layers[n] = c->slots[i].layer;
            if (eids) eids[n] = c->slots[i].eid;
            if (hits) hits[n] = c->slots[i].hits;
            n++;
        }
    }
    return n;
}

/* Layer-ahead whole-tensor readahead for prefill. Routed experts of layer
 * L+1.. are streamed into the Linux page cache while layer L computes, so the
 * staging pool reads warm pages instead of waiting on random SSD reads. */
static void *ahead_worker(void *arg) {
    q4_ahead *a = arg;
    q4_expert_cache *c = a->c;
    const q4_gguf *g = c->g;
    uint8_t *buf = malloc(4u << 20);
    if (!buf) return NULL;
    for (uint32_t L = 0; L < g->n_layer && L < Q4_MAX_LAYER && !a->stop; L++) {
        while (!a->stop && L > a->cur + 4) {
            struct timespec ts = {0, 2 * 1000 * 1000};
            nanosleep(&ts, NULL);
        }
        const q4_layer_ex *le = &c->layers[L];
        const q4_tensor *ts_[3] = {le->gate, le->up, le->down};
        for (int j = 0; j < 3 && !a->stop; j++) {
            const q4_tensor *t = ts_[j];
            if (!t) continue;
            const q4_file *f = &g->files[t->shard];
            uint64_t off = f->data_off + t->offset;
            for (uint64_t o = 0; o < t->nbytes && !a->stop; o += (4u << 20)) {
                uint64_t want = t->nbytes - o;
                if (want > (4u << 20)) want = (4u << 20);
                if (pread_full(f->fd, buf, want, off + o) < 0) break;
            }
        }
    }
    free(buf);
    return NULL;
}

void q4_expert_ahead_begin(q4_ahead *a, q4_expert_cache *c, uint32_t cur_layer) {
    if (!a || !c) return;
    memset(a, 0, sizeof(*a));
    a->c = c;
    a->cur = cur_layer;
    a->stop = 0;
    /* Default off: demand reads already saturate the SSD; readahead only
     * contends. Enable with Q4_AHEAD=1 when the pool leaves the SSD idle. */
    const char *on = getenv("Q4_AHEAD");
    if (!on || on[0] != '1') return;
    if (pthread_create(&a->th, NULL, ahead_worker, a) == 0) a->started = 1;
}

void q4_expert_ahead_end(q4_ahead *a) {
    if (!a || !a->started) return;
    a->stop = 1;
    pthread_join(a->th, NULL);
    a->started = 0;
}

uint32_t q4_expert_prefetch(q4_expert_cache *c, FILE *fp) {
    if (!c || !c->l2_arena) return 0;
    if (c->resident) {
        uint32_t n_layer = c->g->n_layer;
        uint32_t got = 0;
        uint64_t filled = 0;
        /* Layer-major so each layer's resident slice is a contiguous file
         * read; res_ids gives the slot->eid map (warm experts first). */
        for (uint32_t L = 0; L < n_layer && L < Q4_MAX_LAYER; L++) {
            const q4_layer_ex *le = &c->layers[L];
            if (!le->gate || !le->total || !c->res_ids || !c->res_quota)
                continue;
            uint64_t sbase = c->res_idoff[L];
            for (uint32_t k = 0; k < c->res_quota[L]; k++) {
                if (q4_stop) break;
                int32_t e = c->res_ids[sbase + k];
                uint8_t *dst = res_ptr(c, (int32_t)L, e);
                if (!dst) continue;
                int64_t n = load_expert_to(c, dst, (int32_t)L, e);
                if (n < 0) continue;
                drop_expert_cache(c->g, le->gate, e);
                drop_expert_cache(c->g, le->up, e);
                drop_expert_cache(c->g, le->down, e);
                filled += (uint64_t)n;
                c->l2_fill++;
                got++;
                if (c->res_ok) {
                    uint64_t b = (uint64_t)L * c->map_stride + (uint64_t)e;
                    c->res_ok[b >> 6] |= 1ull << (b & 63);
                }
                if (fp && (got % 512u) == 0u) {
                    fprintf(fp, "  resident %u  %.2f GiB\r", got,
                            (double)filled / Q4_GIB);
                    fflush(fp);
                }
            }
        }
        /* Inference stats should not count the one-time fill. */
        c->bytes_read = 0;
        c->misses = 0;
        if (fp)
            fprintf(fp, "  resident %u experts  %.2f GiB\n", got,
                    (double)filled / Q4_GIB);
        const char *pin = getenv("Q4_RESIDENT_PIN");
        if (!pin || pin[0] != '0') {
            double t0 = expert_now();
            /* The image is packed, so every page below l2_arena_bytes was
             * written by the fill — registering the whole range is right. */
            if (q4_hip_host_register(c->l2_arena, c->l2_arena_bytes)) {
                c->res_registered = 1;
                if (fp)
                    fprintf(fp, "  resident image pinned for DMA (%.1f s)\n",
                            expert_now() - t0);
            } else if (fp)
                fprintf(fp, "  resident image pin failed; puts stay staged\n");
        }
        return got;
    }
    if (c->n_l2 == 0) return 0;
    uint32_t n_layer = c->g->n_layer;
    uint32_t n_exp = c->g->n_expert;
    uint32_t got = 0;
    for (uint32_t e = 0; e < n_exp && c->l2_fill < c->n_l2; e++) {
        if (q4_stop) break;
        for (uint32_t L = 0; L < n_layer && c->l2_fill < c->n_l2; L++) {
            if (!c->layers[L].gate) continue;
            int32_t k = map_key(c, (int32_t)L, (int32_t)e);
            if (k < 0 || c->l2_map[k] >= 0) continue;
            uint32_t ls = c->l2_fill;
            uint8_t *dst = c->l2_arena + (size_t)ls * (size_t)c->slot_bytes;
            int64_t n = load_expert_to(c, dst, (int32_t)L, (int32_t)e);
            if (n < 0) continue;
            c->l2_slots[ls].layer = (int32_t)L;
            c->l2_slots[ls].eid = (int32_t)e;
            c->l2_slots[ls].lru = 0;
            c->l2_map[k] = (int32_t)ls;
            c->l2_fill++;
            c->bytes_read += (uint64_t)n;
            got++;
            if (fp && (got % 512u) == 0u) {
                fprintf(fp, "  L2 prefetch %u / %u  (%.2f GiB)\r", c->l2_fill,
                        c->n_l2,
                        (double)c->l2_fill * (double)c->slot_bytes / Q4_GIB);
                fflush(fp);
            }
        }
    }
    if (fp) {
        fprintf(fp, "  L2 prefetch %u / %u  (%.2f GiB)\n", c->l2_fill, c->n_l2,
                (double)c->l2_fill * (double)c->slot_bytes / Q4_GIB);
    }
    return got;
}
