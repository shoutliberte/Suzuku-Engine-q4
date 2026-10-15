#include "q4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* RX 7900 XTX 24 GiB + ~60 GiB DRAM. Same numbers ds4 settled on after
 * the 8 GiB model-span cache failed to hold 8.20 GiB of non-routed weights. */
/* Card is 24 GiB. The Hyprland desktop on this dGPU (2× wallpaperengine)
 * uses 1.27–1.71 GiB idle.  Since 21660f5 the reserve is REAL — the L1
 * arena is sized off measured free VRAM so steady-state vfree lands at
 * the reserve, not below it — so 2 GiB is again a usable margin (the
 * 2026-10-06 arena-eviction OOM happened when the reserve was only on
 * paper).  Q4_VRAM_RESERVE_GIB overrides; floor 1. */
#define Q4_VRAM_DEFAULT      (23ull * Q4_GIB)
#define Q4_RESERVE_DEFAULT   (2ull * Q4_GIB)
#define Q4_DRAM_DEFAULT      (60ull * Q4_GIB)

/* Prefill chunk knob: tokens per q4_sess_prefill call. Bigger chunks hide
 * expert staging better but grow the workspace (q4_pws_floats) and shrink
 * the L1 expert budget the planner can afford. */
uint32_t q4_pref_chunk(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_PREF_CHUNK");
        /* 6144 with the MMQ group at the same size (q4_mmq_group): one
         * expert GEMM per chunk (~120 rows per expert) instead of two
         * 4096-row groups of an 8192 chunk — pf8k 1070 -> 1186, pf32k
         * 1077 -> 1129 t/s — and a smaller lent scratch (static q 27 ->
         * 28).  8192 / 8192 is as fast but lends 0.5 GiB more (q 23,
         * decode lost ~5-10%). */
        v = e && e[0] ? atoi(e) : 6144;
        if (v < 512) v = 512;
        if (v > 8192) v = 8192;
        v = (v + 255) / 256 * 256;
        if (v > 8192) v = 8192;
    }
    return (uint32_t)v;
}

/* Floats of prefill d_pws for a chunk of T tokens: the max of the hc mixer,
 * QSA prep, GDN prep and batched-MoE layouts in prefill_hip /
 * q4_moe_layer_dev_n. Keep these formulas in sync with those sites. */
uint64_t q4_pws_floats(const q4_gguf *g, uint32_t T) {
    if (!g || !T) return 0;
    uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    uint64_t hc_dim = (uint64_t)hc * g->n_embd;
    uint64_t need_hc = (uint64_t)T * (3 * hc_dim + g->hc_rank);
    uint32_t n_head = g->n_head ? g->n_head : 24;
    uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    uint64_t need_qsa = (uint64_t)T * (4u * n_head * hd + 2u * n_kvh * hd);
    uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    uint32_t d_v = n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
    uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    uint64_t need_gdn =
        (uint64_t)T * (conv_ch + 2u * n_v_h * d_v + 2u * n_v_h);
    /* MoE: <=1024-row groups (<=Q4_MMQ_GROUP while MMQ is on — MMQ layers
     * carve bigger groups; over-reserving for layers it cannot run keeps
     * the formula model-agnostic). Mirrors the d_ws layout in
     * q4_moe_layer_dev_n. */
    uint32_t n_exp = g->n_expert, topk = g->n_expert_used;
    uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    uint32_t gcap = q4_mmq_on() ? q4_mmq_group() : 1024;
    uint32_t gsz = T < gcap ? T : gcap;
    uint32_t n_g = (T + gsz - 1) / gsz;
    uint64_t Gmax = (uint64_t)gsz * topk;
    /* Tail after d_meta: contiguous ek|stg|stgn (n_g*1025) + ids|wts
     * (2*T*topk) for the device-route path in q4_moe_layer_dev_n. */
    uint64_t need_moe = (uint64_t)T * n_exp + 2 * Gmax * n_ff +
                        (uint64_t)T * g->n_embd + Gmax * g->n_embd +
                        3 * (uint64_t)T * topk + T + 3 * 512 * n_g +
                        (uint64_t)n_g * 1025u + 2 * (uint64_t)T * topk;
    /* MMQ tail: bounds n_g*(n_exp+1) | eid|slot 2*n_exp | xq|hq bytes. */
    if (q4_mmq_on())
        need_moe += (uint64_t)n_g * (n_exp + 1) + 2 * n_exp +
                    (q4_mmq_q8_bytes(Gmax, g->n_embd) + 15) / 16 * 4 +
                    (q4_mmq_q8_bytes(Gmax, n_ff) + 15) / 16 * 4;
    /* Batched devroute (verify, T<=8 rows): rlog | he | he2 (64 rows/tok) |
     * dd | h | h2 | sh | sg — mirrors moe_dev_n_route's d_ws carve. */
    uint64_t devn_tok = T < 8 ? T : 8;
    uint64_t need_devn = devn_tok * (n_exp + 128ull * n_ff + 2ull * n_ff +
                                     2ull * g->n_embd + 1) +
                         64ull * g->n_embd;
    if (need_devn > need_moe) need_moe = need_devn;
    /* Batched PLE: emb + (kn->gated) + (qn->cn) + co + val per token. */
    uint64_t emb_w = (uint64_t)(g->ple_n_heads ? g->ple_n_heads : 16) *
                     (g->n_embd_ple ? g->n_embd_ple : 160);
    uint64_t need_ple =
        (uint64_t)T * (emb_w + 3 * hc_dim + g->n_embd);
    uint64_t m = need_hc;
    if (need_ple > m) m = need_ple;
    if (need_qsa > m) m = need_qsa;
    if (need_gdn > m) m = need_gdn;
    if (need_moe > m) m = need_moe;
    return m;
}

/* Prefill device buffers (d_pref, d_mixb, d_yb, d_injb, d_pws) for the
 * configured chunk plus hipMalloc slack for decode scratch and graphs. */
/* VRAM the MMQ weight buffers (GU/D, expert-id-indexed) claim:
 * n_exp * (max gu_bytes + max d_bytes) over MMQ-capable layers — exactly
 * what q4_mmq_wbuf allocates, via the same helper.  Counted in scratch so
 * the planner shrinks L1 slots instead of overcommitting. */
static uint64_t mmq_wbuf_bytes(const q4_gguf *g) {
    uint64_t gu = 0, d = 0;
    q4_mmq_wbuf_size(g, &gu, &d);
    /* +2*64K: q4_mmq_wbuf pads each buffer for the kernel's last-iteration
     * row-tail overread. */
    return (gu + d) * (uint64_t)g->n_expert + 131072;
}

/* VRAM the library-GEMM scratch claims (fp16 weight dequant + a chunk's
 * activation conversion + the hipBLASLt workspace; lent from the L1 tail
 * like the rest), 0 when the path is off. */
static uint64_t hgemm_scratch_bytes(const q4_gguf *g) {
    uint64_t wq = 0, xq = 0, ws = 0;
    uint64_t hc = g->hc_mult ? g->hc_mult : 4;
    q4_hgemm_lend_need(q4_pref_chunk(), hc * g->n_embd, &wq, &xq, &ws);
    return wq + xq + ws;
}

/* Pipelined GDN scan (Q4_GDN_REC) scratch: conv h [T][ch], pre-norm oc
 * [T][nvh*d], gates [2][T][nvh].  0 when off; the old kernel's q/k hist
 * staging is part of the 160 MiB misc reserve. */
static uint64_t gdnrec_scratch_bytes(const q4_gguf *g) {
    const char *e = getenv("Q4_GDN_REC");
    if (e && e[0] == '0') return 0;
    uint32_t T = q4_pref_chunk();
    uint32_t nkh = g->ssm_n_group ? g->ssm_n_group : 16;
    uint32_t nvh = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    uint32_t ds = g->ssm_d_state ? g->ssm_d_state : 128;
    uint32_t inner = g->ssm_d_inner ? g->ssm_d_inner : 6144;
    uint64_t ch = 2ull * ds * nkh + inner;
    return (uint64_t)T * (ch + (uint64_t)inner + 2ull * nvh) * 4;
}

static uint64_t scratch_bytes(const q4_gguf *g) {
    uint32_t T = q4_pref_chunk();
    uint32_t hc = g->hc_mult ? g->hc_mult : 4;
    uint64_t bufs =
        (uint64_t)T * ((uint64_t)(hc + 2) * g->n_embd + hc) * sizeof(float);
    return bufs + q4_pws_floats(g, T) * sizeof(float) +
           mmq_wbuf_bytes(g) + hgemm_scratch_bytes(g) +
           gdnrec_scratch_bytes(g) + (160ull << 20);
}

uint32_t q4_n_qsa_layers(const q4_gguf *g) {
    if (!g) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < g->n_layer; i++)
        if (q4_layer_is_qsa(g, (int32_t)i)) n++;
    return n;
}

uint64_t q4_qsa_kv_bytes(const q4_gguf *g, uint32_t ctx) {
    if (!g || !ctx) return 0;
    uint32_t nq = q4_n_qsa_layers(g);
    uint32_t n_kvh = g->n_head_kv ? g->n_head_kv : 2;
    uint32_t hd = g->n_embd_head_k ? g->n_embd_head_k : 256;
    /* K+V. Default q8_0 (34 B / 32 values). Q4_KV=f16 or f32 to widen.
     * MTP stays on its own fp32 cache. */
    const char *kv = getenv("Q4_KV");
    int f32 = kv && (strcmp(kv, "f32") == 0 || strcmp(kv, "fp32") == 0);
    int f16 = kv && (strcmp(kv, "f16") == 0 || strcmp(kv, "fp16") == 0);
    if (f32) return (uint64_t)nq * ctx * 2ull * n_kvh * hd * 4u;
    if (f16) return (uint64_t)nq * ctx * 2ull * n_kvh * hd * 2u;
    return (uint64_t)nq * ctx * 2ull * n_kvh * (hd / 32u) * 34u;
}

/* Chunks the paged KV maps at full ctx, counted the way the session's
 * reserve counts them (each region's whole chunk span, per QSA layer, for
 * K, V, raw keys and pooled keys; q4_sess_open's layout).  A region that
 * does not start or end on a chunk boundary costs a partial chunk at each
 * end: at ctx 4096 that is ~84 chunks for ~35 chunks of bytes. */
static uint64_t span_chunks(uint64_t off, uint64_t len, uint64_t chunk) {
    return len ? (off + len + chunk - 1) / chunk - off / chunk : 0;
}
uint64_t q4_kv_page_chunks(const q4_gguf *g, uint32_t ctx, uint64_t chunk) {
    if (!g || !ctx || !chunk) return 0;
    const uint32_t nq = q4_n_qsa_layers(g);
    if (!nq) return 0;
    const uint64_t bpt = q4_qsa_kv_bytes(g, ctx) / (2ull * nq * ctx);
    const uint64_t dim = g->indexer_head_size ? g->indexer_head_size : 128;
    uint32_t ratio = 4;
    for (uint32_t i = 0; i < g->n_layer && i < Q4_MAX_LAYER; i++)
        if (g->compress_ratio[i]) { ratio = g->compress_ratio[i]; break; }
    const uint64_t blocks = ctx / ratio;
    uint64_t n = 0;
    for (uint32_t qi = 0; qi < nq; qi++) {
        n += 2 * span_chunks(qi * ctx * bpt, ctx * bpt, chunk);
        n += span_chunks(qi * ctx * dim * 2, ctx * dim * 2, chunk);
        n += span_chunks(qi * blocks * dim * 4, blocks * dim * 4, chunk);
    }
    return n;
}

/* Matches the hipMallocs in q4_sess_open: raw fp16 keys, pooled fp32 block
 * keys, and the 512-token score/selection scratch. Zero when that path
 * would not allocate (indexer alloc then falls back to dense attention). */
uint64_t q4_qsa_indexer_bytes(const q4_gguf *g, uint32_t ctx) {
    if (!g || !ctx) return 0;
    uint32_t nq = q4_n_qsa_layers(g);
    uint32_t ratio = 4;
    uint32_t topk = g->indexer_top_k ? g->indexer_top_k : 2048;
    uint32_t dim = g->indexer_head_size ? g->indexer_head_size : 128;
    for (uint32_t i = 0; i < g->n_layer && i < Q4_MAX_LAYER; i++) {
        if (g->compress_ratio[i]) {
            ratio = g->compress_ratio[i];
            break;
        }
    }
    if (!nq || !ratio || dim != 128) return 0;
    uint32_t kblk = topk / ratio;
    if (!kblk || kblk > 512) return 0;
    uint32_t width = topk + ratio - 1;
    uint32_t blocks = ctx / ratio;
    uint32_t chunk = 512;
    if (!blocks) return 0;
    uint64_t raw = (uint64_t)nq * ctx * dim * sizeof(uint16_t);
    uint64_t pool = (uint64_t)nq * blocks * dim * sizeof(float);
    uint64_t score = (uint64_t)chunk * blocks * sizeof(float);
    uint64_t sel = (uint64_t)chunk * width * sizeof(int32_t);
    uint64_t q = (uint64_t)chunk * 4u * dim * sizeof(float);
    uint64_t k = (uint64_t)chunk * dim * sizeof(float);
    return raw + pool + score + sel + q + k;
}

uint64_t q4_gdn_state_bytes(const q4_gguf *g) {
    if (!g) return 0;
    uint32_t nq = q4_n_qsa_layers(g);
    uint32_t ng = g->n_layer > nq ? g->n_layer - nq : 0;
    uint32_t n_k_h = g->ssm_n_group ? g->ssm_n_group : 16;
    uint32_t n_v_h = g->ssm_dt_rank ? g->ssm_dt_rank : 48;
    uint32_t d_k = g->ssm_d_state ? g->ssm_d_state : 128;
    uint32_t d_v = n_v_h ? (g->ssm_d_inner ? g->ssm_d_inner : 6144) / n_v_h : 128;
    uint32_t conv_ch = d_k * n_k_h * 2 + d_v * n_v_h;
    uint32_t ksz = g->ssm_d_conv ? g->ssm_d_conv : 4;
    uint64_t S = (uint64_t)ng * n_v_h * d_v * d_v * sizeof(float);
    uint64_t C = (uint64_t)ng * (ksz > 1 ? ksz - 1 : 1) * conv_ch * sizeof(float);
    return S + C;
}

q4_machine q4_machine_this_pc(uint32_t ctx) {
    q4_machine m;
    memset(&m, 0, sizeof(m));
    m.vram_bytes = Q4_VRAM_DEFAULT;
    m.dram_bytes = Q4_DRAM_DEFAULT;
    m.vram_reserve_bytes = Q4_RESERVE_DEFAULT;
    {
        const char *rg = getenv("Q4_VRAM_RESERVE_GIB");
        if (rg && rg[0]) {
            int v = atoi(rg);
            if (v < 1) v = 1;
            m.vram_reserve_bytes = (uint64_t)v * Q4_GIB;
        }
    }
    m.ctx = ctx ? ctx : 262144;
    /* Filled in q4_place_plan once the GGUF is known. Placeholder until then. */
    m.kv_bytes = (uint64_t)m.ctx * 49152ull; /* 12 QSA × 2 × 2 × 256 × 4 */
    return m;
}

q4_place q4_place_plan(const q4_gguf *g, const q4_machine *m) {
    q4_place p;
    memset(&p, 0, sizeof(p));
    if (!g || !m) {
        p.note = "missing model or machine";
        return p;
    }

    /* PLE table stays on host (mmap + page cache). Mixers are in dense. */
    uint64_t dense_gpu = g->bytes_dense + g->bytes_shared;
    uint64_t kv = q4_qsa_kv_bytes(g, m->ctx) + q4_gdn_state_bytes(g);
    uint64_t idx = q4_qsa_indexer_bytes(g, m->ctx);
    uint64_t scratch = scratch_bytes(g);
    if (kv < 64ull * 1024ull * 1024ull) kv = 64ull * 1024ull * 1024ull;

    p.dense_vram = dense_gpu;
    p.kv_vram = kv;
    p.indexer_vram = idx;
    p.reserve_vram = m->vram_reserve_bytes;
    p.scratch_vram = scratch;
    p.dense_fits_vram =
        p.dense_vram + p.kv_vram + p.indexer_vram + scratch +
            p.reserve_vram < m->vram_bytes;

    uint64_t used = p.dense_vram + p.kv_vram + p.indexer_vram + scratch +
                    p.reserve_vram;
    if (used >= m->vram_bytes) {
        p.vram_used = used;
        p.vram_free = 0;
        p.l1_experts = 0;
        p.ok = false;
        p.note = "dense + KV + reserve exceed VRAM; shrink ctx or quant";
        return p;
    }
    p.l1_expert_vram = m->vram_bytes - used;
    uint64_t slot = q4_expert_packed_bytes(g);
    if (!slot) slot = g->per_expert_bytes;
    /* Packed experts are larger than the mean; ROCm also keeps a slack pool. */
    if (slot)
        p.l1_experts = (uint32_t)(p.l1_expert_vram / slot);
    p.vram_used = used + (uint64_t)p.l1_experts * (slot ? slot : 1);
    p.vram_free = m->vram_bytes - p.vram_used;

    /* PLE is SSD-backed demand I/O (~1.4 KB/token). Do not reserve 27 GiB DRAM
     * for it; Linux may cache hot 4K pages, and that fights expert L2. */
    p.ple_host = 0;
    /* DRAM leftover after a generous process working set becomes L2 page cache.
     * ds4: 90K F32 activations were 5.49 GiB; 256K ~15.6 GiB. */
    uint64_t host_ws = 8ull * Q4_GIB;
    if (m->ctx >= 90000) host_ws += 16ull * Q4_GIB;
    if (host_ws < m->dram_bytes)
        p.l2_page_cache = m->dram_bytes - host_ws;
    p.ok = p.dense_fits_vram && p.l1_experts >= (g->n_expert_used ? g->n_expert_used : 8);
    p.note = p.ok ? "dense resident, PLE on SSD, experts L1 VRAM + L2 page cache"
                  : "L1 too small for one token's routed set";
    return p;
}

void q4_place_print(const q4_place *p, const q4_gguf *g, FILE *fp) {
    fprintf(fp, "placement %s\n", p->ok ? "OK" : "TIGHT");
    fprintf(fp, "  dense+shared in VRAM  %.2f GiB%s\n",
            (double)p->dense_vram / Q4_GIB,
            p->dense_fits_vram ? "" : "  (DOES NOT FIT)");
    fprintf(fp, "  PLE (SSD demand)      %.2f GiB  (not reserved in DRAM/VRAM)\n",
            g ? (double)g->bytes_ple / Q4_GIB : 0);
    fprintf(fp, "  KV + GDN              %.2f GiB\n", (double)p->kv_vram / Q4_GIB);
    fprintf(fp, "  QSA indexer           %.2f GiB\n",
            (double)p->indexer_vram / Q4_GIB);
    fprintf(fp, "  device scratch        %.2f GiB  (prefill chunk %u)\n",
            (double)p->scratch_vram / Q4_GIB, q4_pref_chunk());
    fprintf(fp, "  VRAM reserve          %.2f GiB\n",
            (double)p->reserve_vram / Q4_GIB);
    fprintf(fp, "  L1 expert slots       %u  (%.2f GiB)\n", p->l1_experts,
            (double)p->l1_expert_vram / Q4_GIB);
    fprintf(fp, "  VRAM used/free        %.2f / %.2f GiB\n",
            (double)p->vram_used / Q4_GIB, (double)p->vram_free / Q4_GIB);
    fprintf(fp, "  L2 page-cache hint    %.2f GiB (Linux, reclaimable)\n",
            (double)p->l2_page_cache / Q4_GIB);
    if (g && g->n_expert_used && g->n_layer) {
        uint64_t per_tok = (uint64_t)g->n_layer * g->n_expert_used * g->per_expert_bytes;
        fprintf(fp, "  routed bytes / token  %.2f GiB (%u layers × %u experts)\n",
                (double)per_tok / Q4_GIB, g->n_layer, g->n_expert_used);
    }
    fprintf(fp, "  %s\n", p->note ? p->note : "");
}
