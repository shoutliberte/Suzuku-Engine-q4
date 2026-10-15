#define _GNU_SOURCE
#include "q4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "q4 — Flash-Next SSD/VRAM engine (this PC)\n"
            "usage:\n"
            "  q4 inspect      <model.gguf>\n"
            "  q4 plan         <model.gguf> [--ctx N]\n"
            "  q4 tensors      <model.gguf>\n"
            "  q4 ple-probe    <model.gguf> [--tokens N] [--io pread|mmap]\n"
            "  q4 io-bench     <model.gguf> [--tokens N] [--slots N] [--warm]\n"
            "  q4 moe-probe    <model.gguf> [--layer N] [--tokens N] [--slots N]\n"
            "  q4 decode-probe <model.gguf> [--tokens N] [--slots N] [--ctx N]\n"
            "  q4 prefill-probe <model.gguf> [--tokens N] [--slots N] [--ctx N]\n"
            "                  [--l2-gib N]\n"
            "  q4 chat         <model.gguf> [--ctx N] [--l2-gib N] [--no-pin-ja]\n"
            "  q4 serve        <model.gguf> [--host 127.0.0.1] [--port 8090]\n"
            "                  [--ctx N] [--l2-gib N] [--no-pin-ja]\n"
            "                  [--mtp head.gguf | --no-mtp] (env Q4_MTP)\n"
            "\n"
            "serve is OpenAI-compatible (OpenCode baseURL http://127.0.0.1:8090/v1).\n"
            "GET / is a minimal status page. Default port 8090.\n");
}

static const char *default_model(void) {
    const char *e = getenv("Q4_MODEL");
    return e && e[0] ? e : NULL;
}

static int cmd_inspect(const char *path) {
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    q4_gguf_print(&g, stdout);
    q4_gguf_close(&g);
    return 0;
}

static int cmd_plan(const char *path, uint32_t ctx) {
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    q4_gguf_print(&g, stdout);
    q4_machine m = q4_machine_this_pc(ctx);
    q4_place p = q4_place_plan(&g, &m);
    q4_place_print(&p, &g, stdout);
    q4_gguf_close(&g);
    return p.ok ? 0 : 2;
}

static int cmd_tensors(const char *path) {
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    q4_gguf_print(&g, stdout);
    fprintf(stdout, "\nlayer kinds (G=GDN Q=QSA); PLE injects at layer %u:\n",
            g.ple_layer);
    for (uint32_t i = 0; i < g.n_layer; i++) {
        fputc(q4_layer_is_qsa(&g, (int32_t)i) ? 'Q' : 'G', stdout);
        if ((i + 1) % 12 == 0) fputc('\n', stdout);
        else if ((i + 1) % 4 == 0) fputc(' ', stdout);
    }
    if (g.n_layer % 12) fputc('\n', stdout);
    q4_gguf_close(&g);
    return 0;
}

static int cmd_ple_probe(const char *path, uint32_t tokens, q4_ple_io io) {
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    q4_ple *ple = q4_ple_open_io(&g, io);
    if (!ple) {
        fprintf(stderr, "q4: no PLE table in this GGUF\n");
        q4_gguf_close(&g);
        return 1;
    }
    if (tokens == 0) tokens = 256;
    int32_t *toks = calloc(tokens, sizeof(int32_t));
    int32_t *rows = calloc((size_t)tokens * g.ple_n_heads, sizeof(int32_t));
    float *emb = malloc((size_t)g.ple_n_heads * g.n_embd_ple * sizeof(float));
    if (!toks || !rows || !emb) {
        free(toks); free(rows); free(emb);
        q4_ple_close(ple);
        q4_gguf_close(&g);
        return 1;
    }
    unsigned seed = 1;
    for (uint32_t i = 0; i < tokens; i++) {
        seed = seed * 1664525u + 1013904223u;
        toks[i] = (int32_t)(1000 + (seed % 50000));
    }
    double t0 = now_s();
    if (!q4_ple_hash(ple, toks, tokens, NULL, rows)) {
        fprintf(stderr, "q4: PLE hash failed\n");
        free(toks); free(rows); free(emb);
        q4_ple_close(ple);
        q4_gguf_close(&g);
        return 1;
    }
    /* sequential tokens: n-gram windows overlap; gathers are still random rows */
    double acc = 0;
    for (uint32_t i = 0; i < tokens; i++) {
        if (!q4_ple_gather(ple, rows + (size_t)i * g.ple_n_heads, g.ple_n_heads,
                           emb)) {
            fprintf(stderr, "q4: PLE gather failed at token %u\n", i);
            free(toks); free(rows); free(emb);
            q4_ple_close(ple);
            q4_gguf_close(&g);
            return 1;
        }
        acc += emb[0];
    }
    double dt = now_s() - t0;
    /* second pass: same rows, page cache should be warm */
    double t1 = now_s();
    for (uint32_t i = 0; i < tokens; i++) {
        if (!q4_ple_gather(ple, rows + (size_t)i * g.ple_n_heads, g.ple_n_heads,
                           emb))
            return 1;
        acc += emb[0];
    }
    double dt2 = now_s() - t1;
    {
        uint64_t nrow = g.ple_table >= 0 ? g.tensors[g.ple_table].ne[1] : 0;
        uint64_t rb = nrow ? g.bytes_ple / nrow : 0;
        fprintf(stdout,
                "PLE table     %.2f GiB  IQ4_NL  rows %llu  width %u  heads %u\n",
                (double)g.bytes_ple / Q4_GIB, (unsigned long long)nrow,
                g.n_embd_ple, g.ple_n_heads);
        fprintf(stdout, "tokens        %u  (~%llu bytes / token)  io=%s\n", tokens,
                (unsigned long long)g.ple_n_heads * rb,
                io == Q4_PLE_IO_MMAP ? "mmap" : "pread");
    }
    fprintf(stdout, "cold          %.3f s  %.1f tok/s\n", dt, tokens / dt);
    fprintf(stdout, "warm          %.3f s  %.1f tok/s\n", dt2, tokens / dt2);
    fprintf(stdout, "sink          %.3f\n", acc * 1e-30); /* keep acc live */
    free(toks); free(rows); free(emb);
    q4_ple_close(ple);
    q4_gguf_close(&g);
    return 0;
}

static int cmd_io_bench(const char *path, uint32_t tokens, uint32_t slots,
                        uint32_t ctx, bool warm) {
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    q4_machine m = q4_machine_this_pc(ctx);
    q4_place p = q4_place_plan(&g, &m);
    q4_gguf_print(&g, stdout);
    q4_place_print(&p, &g, stdout);
    if (slots == 0) {
        /* Host-side I/O bench; do not allocate the full VRAM L1 on DRAM. */
        slots = 256;
        if (p.l1_experts && p.l1_experts < slots) slots = p.l1_experts;
    }
    if (tokens == 0) tokens = 32;
    if (g.n_expert == 0 || g.n_layer == 0 || g.n_expert_used == 0) {
        fprintf(stderr, "q4: model is not a routed MoE GGUF\n");
        q4_gguf_close(&g);
        return 1;
    }
    q4_expert_cache *c = q4_expert_cache_open(&g, slots);
    if (!c) {
        fprintf(stderr, "q4: expert cache alloc failed (%u slots × %.2f MiB)\n",
                slots, (double)g.per_expert_bytes / (1024.0 * 1024.0));
        q4_gguf_close(&g);
        return 1;
    }
    fprintf(stdout, "io-bench tokens=%u slots=%u layers=%u topk=%u\n", tokens,
            slots, g.n_layer, g.n_expert_used);
    unsigned seed = 1;
    double t0 = now_s();
    for (uint32_t tok = 0; tok < tokens; tok++) {
        for (uint32_t L = 0; L < g.n_layer; L++) {
            for (uint32_t k = 0; k < g.n_expert_used; k++) {
                seed = seed * 1664525u + 1013904223u;
                int32_t eid = (int32_t)(seed % g.n_expert);
                if (q4_expert_cache_touch(c, (int32_t)L, eid) < 0) {
                    fprintf(stderr, "q4: read failed layer %u expert %d\n", L,
                            eid);
                    q4_expert_cache_close(c);
                    q4_gguf_close(&g);
                    return 1;
                }
            }
        }
    }
    double dt = now_s() - t0;
    uint64_t hits = 0, misses = 0, bytes = 0;
    q4_expert_cache_stats(c, &hits, &misses, &bytes);
    double tps = dt > 0 ? (double)tokens / dt : 0;
    fprintf(stdout, "time          %.3f s\n", dt);
    fprintf(stdout, "token-eq      %.2f /s  (expert I/O only, random router)\n",
            tps);
    fprintf(stdout, "hits/misses   %llu / %llu  (hit %.1f%%)\n",
            (unsigned long long)hits, (unsigned long long)misses,
            (hits + misses) ? 100.0 * (double)hits / (double)(hits + misses)
                            : 0);
    fprintf(stdout, "SSD bytes     %.2f GiB  (%.2f GB/s)\n",
            (double)bytes / Q4_GIB,
            dt > 0 ? (double)bytes / dt / 1e9 : 0);
    if (warm) {
        double t1 = now_s();
        seed = 1;
        for (uint32_t tok = 0; tok < tokens; tok++) {
            for (uint32_t L = 0; L < g.n_layer; L++) {
                for (uint32_t k = 0; k < g.n_expert_used; k++) {
                    seed = seed * 1664525u + 1013904223u;
                    int32_t eid = (int32_t)(seed % g.n_expert);
                    if (q4_expert_cache_touch(c, (int32_t)L, eid) < 0) {
                        q4_expert_cache_close(c);
                        q4_gguf_close(&g);
                        return 1;
                    }
                }
            }
        }
        double dt2 = now_s() - t1;
        uint64_t h2 = 0, m2 = 0, b2 = 0;
        q4_expert_cache_stats(c, &h2, &m2, &b2);
        fprintf(stdout, "warm repeat   %.3f s  %.2f tok-eq/s  extra SSD %.2f GiB\n",
                dt2, dt2 > 0 ? (double)tokens / dt2 : 0,
                (double)(b2 - bytes) / Q4_GIB);
        fprintf(stdout, "hits after    %llu / %llu  (hit %.1f%%)\n",
                (unsigned long long)h2, (unsigned long long)m2,
                (h2 + m2) ? 100.0 * (double)h2 / (double)(h2 + m2) : 0);
    }
    q4_expert_cache_close(c);
    q4_gguf_close(&g);
    return 0;
}

static int cmd_moe_probe(const char *path, uint32_t layer, uint32_t tokens,
                         uint32_t slots) {
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    if (tokens == 0) tokens = 4;
    if (slots == 0) slots = 64;
    if (layer >= g.n_layer) {
        fprintf(stderr, "q4: layer %u out of range (%u)\n", layer, g.n_layer);
        q4_gguf_close(&g);
        return 1;
    }
    bool hip = q4_hip_init();
    q4_expert_cache *c = q4_expert_cache_open(&g, slots);
    if (!c) {
        fprintf(stderr, "q4: expert cache failed\n");
        q4_gguf_close(&g);
        return 1;
    }
    if (hip)
        q4_hip_experts_alloc(slots, q4_expert_cache_slot_bytes(c));
    fprintf(stdout, "hip            %s\n", hip ? "ok" : "cpu");
    float *x = calloc(g.n_embd, sizeof(float));
    float *y = calloc(g.n_embd, sizeof(float));
    if (!x || !y) {
        free(x); free(y);
        q4_expert_cache_close(c);
        q4_gguf_close(&g);
        return 1;
    }
    unsigned seed = 1;
    for (uint32_t i = 0; i < g.n_embd; i++) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = ((int)(seed % 2000) - 1000) / 1000.0f;
    }
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    q4_expert_parts(&g, (int32_t)layer, &gate, &up, &down, &og, &ou, &od, &tot);
    fprintf(stdout, "moe-probe layer=%u  n_embd=%u  n_ff=%u  topk=%u  slots=%u\n",
            layer, g.n_embd, g.n_ff_exp, g.n_expert_used, slots);
    fprintf(stdout, "  gate %s  up %s  down %s  packed %.2f MiB/expert\n",
            gate ? q4_type_name(gate->ggml_type) : "?",
            up ? q4_type_name(up->ggml_type) : "?",
            down ? q4_type_name(down->ggml_type) : "?",
            (double)tot / (1024.0 * 1024.0));
    double t0 = now_s();
    if (!q4_moe_layer(&g, c, (int32_t)layer, x, y)) {
        fprintf(stderr, "q4: moe-probe failed (cold)\n");
        free(x); free(y);
        q4_expert_cache_close(c);
        q4_gguf_close(&g);
        return 1;
    }
    double dt_cold = now_s() - t0;
    double acc = 0;
    for (uint32_t i = 0; i < g.n_embd; i++) acc += y[i];
    t0 = now_s();
    for (uint32_t tok = 0; tok < tokens; tok++) {
        if (!q4_moe_layer(&g, c, (int32_t)layer, x, y)) {
            fprintf(stderr, "q4: moe-probe failed at token %u\n", tok);
            free(x); free(y);
            q4_expert_cache_close(c);
            q4_gguf_close(&g);
            return 1;
        }
        acc += y[0];
    }
    double dt = now_s() - t0;
    uint64_t hits = 0, misses = 0, bytes = 0;
    q4_expert_cache_stats(c, &hits, &misses, &bytes);
    fprintf(stdout, "cold 1 layer     %.3f s  (load router/shexp + miss I/O)\n",
            dt_cold);
    fprintf(stdout, "warm %u tokens    %.3f s  %.2f layer-tok/s  (this layer only)\n",
            tokens, dt, dt > 0 ? (double)tokens / dt : 0);
    fprintf(stdout, "hits/misses      %llu / %llu\n", (unsigned long long)hits,
            (unsigned long long)misses);
    fprintf(stdout, "SSD bytes        %.2f MiB\n", (double)bytes / (1024.0 * 1024.0));
    fprintf(stdout, "out[0]           %g  sink %g\n", (double)y[0], acc * 1e-30);
    fprintf(stdout, "note             48 layers would be ~1/%u of a decode step\n",
            g.n_layer ? g.n_layer : 48);
    free(x); free(y);
    q4_expert_cache_close(c);
    q4_gguf_close(&g);
    return 0;
}

static int probe_warm_load(q4_expert_cache *c, const q4_gguf *g);

static int cmd_decode_probe(const char *path, uint32_t tokens, uint32_t slots,
                            uint32_t ctx) {
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    if (tokens == 0) tokens = 4;
    if (ctx == 0) ctx = 256;
    bool hip = q4_hip_init();
    fprintf(stdout, "hip            %s\n", hip ? "ok gfx1100" : "cpu fallback");
    double t_load = now_s();
    q4_store *st = q4_store_open(&g);
    if (!st) {
        fprintf(stderr, "q4: dense store load failed\n");
        q4_gguf_close(&g);
        return 1;
    }
    fprintf(stdout, "dense host     %.2f s\n", now_s() - t_load);
    if (hip) {
        double t0 = now_s();
        if (!q4_store_upload_hip(st))
            fprintf(stderr, "q4: VRAM upload failed, using host GEMV via H2D\n");
        else
            fprintf(stdout, "dense VRAM     %.2f s\n", now_s() - t0);
    }
    q4_machine m = q4_machine_this_pc(ctx);
    /* Same live-VRAM budgeting as the engine path: the dense image is
     * already resident, so free-now + dense gives the planner the real
     * budget (desktop share included) instead of the static 23 GiB. */
    if (hip) {
        uint64_t fr = q4_hip_vram_free();
        if (fr) m.vram_bytes = fr + g.bytes_dense + g.bytes_shared;
    }
    q4_place p = q4_place_plan(&g, &m);
    if (slots == 0) {
        slots = hip ? p.l1_experts : 256;
        if (slots > 8192) slots = 8192;
        if (slots < 64) slots = 64;
    }
    uint64_t l2b = 32ull * Q4_GIB;
    {
        const char *e = getenv("Q4_L2_GIB");
        if (e && e[0]) {
            int v = atoi(e);
            l2b = v <= 0 ? 0 : (uint64_t)v * Q4_GIB;
        }
    }
    q4_expert_cache *c = NULL;
    while (!c) {
        c = q4_expert_cache_open_ex(&g, slots, l2b);
        if (c) break;
        if (l2b > 8ull * Q4_GIB) {
            l2b /= 2;
            fprintf(stderr, "q4: L2 alloc failed, retry %.0f GiB\n",
                    (double)l2b / Q4_GIB);
            continue;
        }
        fprintf(stderr, "q4: expert cache failed\n");
        q4_store_close(st);
        q4_gguf_close(&g);
        return 1;
    }
    fprintf(stdout, "DRAM L2        %.2f GiB  (%u slots reserved)\n",
            (double)q4_expert_l2_bytes(c) / Q4_GIB,
            (unsigned)(q4_expert_l2_bytes(c) / q4_expert_cache_slot_bytes(c)));
    if (q4_expert_l2_bytes(c)) {
        double t0 = now_s();
        fprintf(stdout, "prefetching routed experts into DRAM (watch RSS)...\n");
        q4_expert_prefetch(c, stdout);
        fprintf(stdout, "DRAM L2 filled %.2f s  %u experts  %.2f GiB\n",
                now_s() - t0, q4_expert_l2_filled(c),
                (double)q4_expert_l2_filled(c) * q4_expert_cache_slot_bytes(c) /
                    Q4_GIB);
    }
    if (hip) {
        uint64_t sb = q4_expert_cache_slot_bytes(c);
        uint32_t capped = q4_expert_cache_n_slots(c);
        if (capped && capped < slots) slots = capped;
        uint32_t s = slots, got = 0;
        while (s >= 64) {
            if (q4_hip_experts_alloc(s, sb)) {
                got = s;
                break;
            }
            fprintf(stderr, "q4: GPU L1 %u slots failed, retry %u\n", s, s / 2);
            s /= 2;
        }
        if (!got)
            fprintf(stderr, "q4: GPU L1 alloc failed (%u x %.2f MiB)\n", slots,
                    (double)sb / (1024.0 * 1024.0));
        else {
            if (got < q4_expert_cache_n_slots(c))
                q4_expert_cache_cap_slots(c, got);
            fprintf(stdout, "GPU L1         %u slots  %.2f GiB\n", got,
                    (double)got * sb / Q4_GIB);
            q4_expert_static_load(c, got);
        }
    }
    /* Optional L1 pin set (same format/cap as the engine warm load): pinned
     * experts can never be evicted, protecting the hot set from churn when
     * the per-layer working set exceeds the slot share. */
    if (getenv("Q4_WARM_FILE") && !q4_expert_static_on(c) &&
        probe_warm_load(c, &g))
        fprintf(stderr, "q4: Q4_WARM_FILE load failed — running unpinned\n");
    q4_ple *ple = q4_ple_open(&g);
    q4_sess *sess = q4_sess_open(&g, st, c, ple, ctx);
    if (!sess) {
        fprintf(stderr, "q4: session open failed\n");
        q4_ple_close(ple);
        q4_expert_cache_close(c);
        q4_store_close(st);
        q4_gguf_close(&g);
        return 1;
    }
    int32_t tok = (int32_t)(g.bos_id ? g.bos_id : 1);
    uint32_t warmup = 8;
    {
        const char *e = getenv("Q4_WARMUP");
        if (e && e[0]) warmup = (uint32_t)atoi(e);
    }
    if (warmup) {
        fprintf(stdout, "warmup         %u tokens (pin routed experts to L1/L2)\n",
                warmup);
        int32_t wtok = tok;
        for (uint32_t i = 0; i < warmup; i++) {
            int32_t nxt = 0;
            float lg = 0;
            if (!q4_sess_decode(sess, wtok, &nxt, &lg)) {
                fprintf(stderr, "q4: warmup failed\n");
                break;
            }
            wtok = nxt;
        }
        tok = wtok;
    }
    fprintf(stdout, "decode-probe tokens=%u start_id=%d layers=%u\n", tokens, tok,
            g.n_layer);
    double t_cold = 0, t_warm = 0;
    for (uint32_t i = 0; i < tokens; i++) {
        int32_t nxt = 0;
        float lg = 0;
        double t0 = now_s();
        if (!q4_sess_decode(sess, tok, &nxt, &lg)) {
            fprintf(stderr, "q4: decode failed at token %u (id %d)\n", i, tok);
            q4_sess_close(sess);
            q4_ple_close(ple);
            q4_expert_cache_close(c);
            q4_store_close(st);
            q4_gguf_close(&g);
            return 1;
        }
        double dt = now_s() - t0;
        if (i == 0) t_cold = dt;
        else t_warm += dt;
        fprintf(stdout, "  tok %u  id %d -> %d  logit %.3f  %.3f s  %.2f t/s\n", i,
                tok, nxt, lg, dt, dt > 0 ? 1.0 / dt : 0);
        tok = nxt;
    }
    uint64_t l1h = 0, l2h = 0, ssd = 0, bytes = 0;
    q4_expert_cache_stats_ex(c, &l1h, &l2h, &ssd, &bytes);
    fprintf(stdout, "cold            %.3f s  %.2f t/s\n", t_cold,
            t_cold > 0 ? 1.0 / t_cold : 0);
    if (tokens > 1) {
        double w = t_warm / (tokens - 1);
        fprintf(stdout, "warm mean       %.3f s  %.2f t/s  (%u tok)\n", w,
                w > 0 ? 1.0 / w : 0, tokens - 1);
    }
    fprintf(stdout, "moe prof        io %.3f s  d2h %.3f s  gemm %.3f s\n",
            q4_prof_moe_io, q4_prof_moe_d2h, q4_prof_moe_gemm);
    fprintf(stdout,
            "  topk %.3f  x-d2h %.3f  sgate %.3f  join %.3f  merge %.3f\n",
            q4_prof_moe_topk, q4_prof_moe_xd2h, q4_prof_moe_sg,
            q4_prof_moe_join, q4_prof_moe_merg);
    fprintf(stdout,
            "dec prof        embed %.3f  attn %.3f  moe %.3f  comb %.3f  head %.3f\n",
            q4_prof_dec[0], q4_prof_dec[1], q4_prof_dec[2], q4_prof_dec[3],
            q4_prof_dec[4]);
        fprintf(stderr,
            "  attn-detail   ple %.3f  hc-attn %.3f  qsa/gdn %.3f  comb+ffnmix %.3f\n",
            q4_prof_dec[5], q4_prof_dec[6], q4_prof_dec[7], q4_prof_dec[8]);
    fprintf(stderr,
            "  step          gA+sync %.3f  hostpart %.3f  gB %.3f  "
            "e-attn %.3f  e-moe %.3f  e-comb %.3f  embed %.3f  head %.3f\n",
            q4_prof_step[0], q4_prof_step[1], q4_prof_step[2],
            q4_prof_step[3], q4_prof_step[4], q4_prof_step[5],
            q4_prof_step[6], q4_prof_step[7]);
    fprintf(stderr,
            "  gpu-ms        win %.0f  gA %.0f  gB %.0f  head %.0f  "
            "e-attn %.0f  e-moe %.0f\n"
            "  gA-sections   hc1 %.0f  gdn %.0f  comb %.0f  ffnmix %.0f  moea %.0f\n",
            q4_prof_gpu_ms, q4_prof_gpu_ph[1], q4_prof_gpu_ph[2],
            q4_prof_gpu_ph[3], q4_prof_gpu_ph[4], q4_prof_gpu_ph[5],
            q4_prof_gpu_sec[0], q4_prof_gpu_sec[1], q4_prof_gpu_sec[2],
            q4_prof_gpu_sec[3], q4_prof_gpu_sec[4]);
    q4_hip_gemv_prof_report();
    fprintf(stdout,
            "expert cache    L1 hits %llu  L2 hits %llu  SSD misses %llu  "
            "SSD %.2f GiB\n",
            (unsigned long long)l1h, (unsigned long long)l2h,
            (unsigned long long)ssd, (double)bytes / Q4_GIB);
    {
        unsigned long long dj = 0, dm = 0, dl = 0;
        q4_moe_dev_stats(&dj, &dm, &dl);
        if (dj) {
            double sub[4] = { 0, 0, 0, 0 };
            double rt = q4_moe_route_prof(sub);
            if (rt >= 0.0)
                fprintf(stdout,
                        "devroute        jobs %llu  cpu-misses %llu  "
                        "lat %.0f us/job  route %.0f us "
                        "(scan %.0f topk %.0f pub %.0f)\n",
                        dj, dm, (double)dl / dj, rt, sub[0], sub[1], sub[2]);
            else
                fprintf(stdout,
                        "devroute        jobs %llu  cpu-misses %llu  "
                        "lat %.0f us/job\n",
                        dj, dm, (double)dl / dj);
        }
    }
    q4_sess_close(sess);
    q4_ple_close(ple);
    q4_expert_cache_close(c);
    q4_store_close(st);
    q4_hip_shutdown();
    q4_gguf_close(&g);
    return 0;
}

static void chat_stream(const char *piece, int kind, void *u) {
    (void)u;
    if (kind == Q4_PIECE_REASONING) fputs("\033[2m", stdout);
    fputs(piece, stdout);
    if (kind == Q4_PIECE_REASONING) fputs("\033[0m", stdout);
    fflush(stdout);
}

static q4_engine *open_engine(const char *model, uint32_t ctx, uint64_t l2_bytes,
                              int pin_ja, const char *mtp_path) {
    q4_engine *e = q4_engine_open_full(model, ctx, l2_bytes, mtp_path);
    if (!e) return NULL;
    /* Static L1 experts already cover the warm pin's job — the file's top
     * hits are fixed in the arena at startup, so both the warm-pin restore
     * and the Japanese warmup run no-op there. */
    if (pin_ja) {
        const char *wf = getenv("Q4_WARM_FILE");
        if (wf && wf[0] && q4_engine_warm_load(e, wf, stderr) == 0) {
            /* warm set restored: skip the Japanese warmup entirely */
        } else {
            if (q4_engine_pin_ja(e, stderr) < 0)
                fprintf(stderr, "q4: Japanese pin failed (continuing)\n");
            else if (wf && wf[0] && access(wf, F_OK) != 0 &&
                     q4_engine_warm_save(e, wf) > 0)
                fprintf(stderr, "q4: warm set saved to %s\n", wf);
        }
    }
    return e;
}

static int build_probe_ids(const q4_gguf *g, uint32_t want, int32_t **out) {
    q4_tok *tok = q4_tok_open(g);
    if (!tok) return -1;
    int32_t *ids = malloc((size_t)want * sizeof(int32_t));
    if (!ids) {
        q4_tok_close(tok);
        return -1;
    }
    /* Q4_PROBE_FILE: real text instead of the synthetic paragraph — the
     * repeated paragraph flatters expert routing and PLE row reuse. */
    const char *pf = getenv("Q4_PROBE_FILE");
    int ntok = -1;
    if (pf && pf[0]) {
        FILE *f = fopen(pf, "rb");
        if (!f) {
            fprintf(stderr, "q4: Q4_PROBE_FILE %s: cannot open\n", pf);
        } else {
            fseek(f, 0, SEEK_END);
            long fsz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *text = malloc((size_t)fsz + 1);
            if (text && fread(text, 1, (size_t)fsz, f) == (size_t)fsz) {
                text[fsz] = 0;
                ntok = q4_tok_encode(tok, text, ids, (int)want);
                if (ntok <= 0)
                    fprintf(stderr, "q4: Q4_PROBE_FILE %s: no tokens\n", pf);
            }
            free(text);
            fclose(f);
        }
        if (ntok <= 0) {
            free(ids);
            q4_tok_close(tok);
            return -1;
        }
    } else {
        size_t cap = (size_t)want * 24u + 8192u;
        char *text = malloc(cap);
        if (!text) {
            free(ids);
            q4_tok_close(tok);
            return -1;
        }
        size_t n = 0;
        const char *para =
            "Qwen3.8-Flash-Next のプレフィル速度とコンテキスト長を測る。"
            "各層はルーテッドエキスパートを DRAM から読み、QSA の KV を伸ばす。\n"
            "int add(int a, int b) { return a + b + 1; }\n";
        unsigned salt = 0;
        while (n + 640 < cap) {
            int w = snprintf(text + n, cap - n, "%s#%u\n", para, salt++);
            if (w <= 0) break;
            n += (size_t)w;
        }
        ntok = q4_tok_encode(tok, text, ids, (int)want);
        free(text);
    }
    q4_tok_close(tok);
    if (ntok <= 0) {
        free(ids);
        return -1;
    }
    *out = ids;
    return ntok;
}

/* Serve-shaped run: Japanese pin (or Q4_WARM_FILE), then a long prefill
 * and a greedy decode. One process. Q4_PROBE_DECODE sets the decode length. */
static int cmd_prefill_warm(const char *path, uint32_t tokens, uint32_t ctx,
                            uint64_t l2_bytes) {
    if (tokens == 0) tokens = 2048;
    if (ctx == 0) ctx = 65536;
    q4_engine *e = q4_engine_open_full(path, ctx, l2_bytes, NULL);
    if (!e) return 1;
    const char *wf = getenv("Q4_WARM_FILE");
    if (wf && wf[0] && q4_engine_warm_load(e, wf, stdout) == 0) {
        fprintf(stdout, "warm file      %s\n", wf);
    } else {
        if (q4_engine_pin_ja(e, stdout) < 0)
            fprintf(stdout, "q4: Japanese pin failed (continuing)\n");
        else if (wf && wf[0] && q4_engine_warm_save(e, wf) > 0)
            fprintf(stdout, "warm file saved %s\n", wf);
    }
    fflush(stdout);
    int n_dec = 64;
    {
        const char *d = getenv("Q4_PROBE_DECODE");
        if (d && d[0]) n_dec = atoi(d);
        if (n_dec < 0) n_dec = 0;
        if (n_dec > 256) n_dec = 256;
    }
    int32_t *ids = NULL;
    int ntok = build_probe_ids(q4_engine_gguf(e), tokens, &ids);
    if (ntok <= 0) {
        fprintf(stdout, "BENCH_ABORT tokenize\n");
        q4_engine_close(e);
        return 1;
    }
    if ((uint32_t)ntok > tokens) ntok = (int)tokens;
    int rc = q4_engine_bench(e, ids, ntok, n_dec, stdout);
    free(ids);
    q4_engine_close(e);
    return rc == 0 ? 0 : 1;
}

/* q4 ppl <model> --file F[,F2,...] [--tokens N] [--ctx N]: teacher-forced
 * quality probe over each UTF-8 text (raw, no chat template), one model
 * load for all files. Q4_PPL_DUMP=<dir> writes <dir>/<basename>.nll with
 * per-token "pos target nll argmax" lines for cross-model diffs. */
static char *read_text(const char *file) {
    FILE *tf = fopen(file, "rb");
    if (!tf) return NULL;
    fseek(tf, 0, SEEK_END);
    long sz = ftell(tf);
    fseek(tf, 0, SEEK_SET);
    char *text = sz > 0 ? malloc((size_t)sz + 1) : NULL;
    if (text && fread(text, 1, (size_t)sz, tf) != (size_t)sz) {
        free(text);
        text = NULL;
    }
    fclose(tf);
    if (text) text[sz] = 0;
    return text;
}

static int cmd_ppl(const char *path, const char *files, uint32_t tokens,
                   uint32_t ctx, uint64_t l2_bytes) {
    if (!files) {
        fprintf(stderr, "q4: ppl needs --file <utf8 text>[,more]\n");
        return 1;
    }
    if (ctx == 0) ctx = 16384;
    if (tokens == 0 || tokens > ctx - 16) tokens = ctx - 16;
    q4_engine *e = q4_engine_open_full(path, ctx, l2_bytes, NULL);
    if (!e) return 1;
    q4_tok *tok = q4_tok_open(q4_engine_gguf(e));
    int32_t *ids = tok ? malloc((size_t)tokens * sizeof(int32_t)) : NULL;
    const char *dp = getenv("Q4_PPL_DUMP");
    char *list = strdup(files);
    int rc = ids && list ? 0 : 1;
    for (char *sv = NULL, *f = list ? strtok_r(list, ",", &sv) : NULL;
         f && rc == 0; f = strtok_r(NULL, ",", &sv)) {
        char *text = read_text(f);
        if (!text) {
            fprintf(stderr, "q4: cannot read %s\n", f);
            rc = 1;
            break;
        }
        int ntok = 0;
        const char *idf = getenv("Q4_PPL_IDS");
        if (idf && idf[0]) {
            /* Q4_PPL_IDS=1: the file holds token ids (whitespace
             * separated), e.g. a dumped prompt + generation */
            for (char *q = text, *end; ntok < (int)tokens; q = end) {
                long v = strtol(q, &end, 10);
                if (end == q) break;
                ids[ntok++] = (int32_t)v;
            }
        } else {
            ntok = q4_tok_encode(tok, text, ids, (int)tokens);
        }
        free(text);
        fprintf(stdout, "ppl  file %s  tokens %d  ctx %u\n", f, ntok, ctx);
        FILE *dump = NULL;
        if (dp && dp[0]) {
            const char *b = strrchr(f, '/');
            char dpath[4096];
            snprintf(dpath, sizeof dpath, "%s/%s.nll", dp, b ? b + 1 : f);
            dump = fopen(dpath, "w");
        }
        if (ntok < 2 || q4_engine_ppl(e, ids, ntok, stdout, dump) != 0) rc = 1;
        if (dump) fclose(dump);
    }
    free(list);
    free(ids);
    q4_tok_close(tok);
    q4_engine_close(e);
    return rc;
}

/* Layer-first prefill, no full-expert prefetch. L2 fills from misses, which
 * is how a long prompt actually behaves on this PC. */
/* Restore the serve-time L1 pin set into the expert cache (same file
 * format/cap as q4_engine_warm_load). */
static int probe_warm_load(q4_expert_cache *c, const q4_gguf *g) {
    const char *wf = getenv("Q4_WARM_FILE");
    if (!wf || !wf[0]) return -1;
    uint32_t cap = q4_expert_warm_cap(q4_expert_cache_n_slots(c), g->n_layer,
                                    g->n_expert_used);
    return q4_expert_warm_load(c, g, wf, cap, stdout);
}

static int cmd_prefill_probe(const char *path, uint32_t tokens, uint32_t slots,
                             uint32_t ctx, uint64_t l2_bytes, uint32_t gen_n) {
    {
        const char *pw = getenv("Q4_PROBE_WARM");
        if (pw && pw[0] == '1')
            return cmd_prefill_warm(path, tokens, ctx, l2_bytes);
    }
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return 1;
    if (ctx == 0) ctx = 65536;
    if (tokens == 0) tokens = 2048;
    if (tokens > ctx) tokens = ctx;
    bool hip = q4_hip_init();
    fprintf(stdout, "hip            %s\n", hip ? "ok gfx1100" : "cpu fallback");
    q4_gguf_print(&g, stdout);
    q4_machine m = q4_machine_this_pc(ctx);
    /* Dense is not uploaded yet here: live free VRAM already includes the
     * space it will take, so it is the planner's vram_bytes as-is. */
    if (hip) {
        uint64_t fr = q4_hip_vram_free();
        if (fr) m.vram_bytes = fr;
    }
    q4_place p = q4_place_plan(&g, &m);
    q4_place_print(&p, &g, stdout);
    double t_load = now_s();
    q4_store *st = q4_store_open(&g);
    if (!st) {
        fprintf(stderr, "q4: dense store load failed\n");
        q4_gguf_close(&g);
        return 1;
    }
    fprintf(stdout, "dense host     %.2f s  %.2f GiB\n", now_s() - t_load,
            (double)(g.bytes_dense + g.bytes_shared) / Q4_GIB);
    if (hip && !q4_store_upload_hip(st)) {
        fprintf(stderr, "q4: VRAM upload failed\n");
        q4_store_close(st);
        q4_gguf_close(&g);
        return 1;
    }
    /* Drop the host dense copy before the expert arena. A full resident
     * set is ~45 GiB and will not fit beside a second 5 GiB copy. */
    if (hip) q4_store_free_host_except(st, &g);
    if (slots == 0) {
        slots = hip ? p.l1_experts : 256;
        if (slots > 8192) slots = 8192;
        if (slots < 64) slots = 64;
    }
    if (l2_bytes == 0) l2_bytes = 24ull * Q4_GIB;
    q4_expert_cache *c = NULL;
    while (!c) {
        c = q4_expert_cache_open_ex(&g, slots, l2_bytes);
        if (c) break;
        if (l2_bytes > 8ull * Q4_GIB) {
            l2_bytes /= 2;
            fprintf(stderr, "q4: L2 alloc failed, retry %.0f GiB\n",
                    (double)l2_bytes / Q4_GIB);
            continue;
        }
        fprintf(stderr, "q4: expert cache failed\n");
        q4_store_close(st);
        q4_gguf_close(&g);
        return 1;
    }
    fprintf(stdout, "DRAM L2        %.2f GiB  slot %.2f MiB\n",
            (double)q4_expert_l2_bytes(c) / Q4_GIB,
            (double)q4_expert_cache_slot_bytes(c) / (1024.0 * 1024.0));
    {
        const char *er = getenv("Q4_EXPERT_RESIDENT");
        if (er && er[0] && er[0] != '0' && q4_expert_l2_bytes(c) > 8ull * Q4_GIB) {
            double t0 = now_s();
            fprintf(stdout, "loading every routed expert into DRAM...\n");
            uint32_t n = q4_expert_prefetch(c, stdout);
            fprintf(stdout, "resident fill  %.2f s  %u experts\n", now_s() - t0, n);
        }
    }
    if (hip) {
        uint64_t sb = q4_expert_cache_slot_bytes(c);
        uint32_t capped = q4_expert_cache_n_slots(c);
        if (capped && capped < slots) slots = capped;
        uint32_t s = slots, got = 0;
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
            q4_expert_cache_close(c);
            q4_store_close(st);
            q4_gguf_close(&g);
            return 1;
        }
        if (got < q4_expert_cache_n_slots(c))
            q4_expert_cache_cap_slots(c, got);
        fprintf(stdout, "GPU L1         %u slots  %.2f GiB\n", got,
                (double)got * sb / Q4_GIB);
        q4_expert_static_load(c, got);
    }
    if (getenv("Q4_WARM_FILE") && !q4_expert_static_on(c) &&
        probe_warm_load(c, &g))
        fprintf(stderr, "q4: Q4_WARM_FILE load failed — running unpinned\n");
    q4_ple *ple = q4_ple_open(&g);
    q4_sess *sess = q4_sess_open(&g, st, c, ple, ctx);
    if (!sess) {
        fprintf(stderr, "q4: session open failed (ctx %u does not fit)\n", ctx);
        q4_ple_close(ple);
        q4_expert_cache_close(c);
        q4_store_close(st);
        q4_gguf_close(&g);
        return 1;
    }
    int32_t *ids = NULL;
    int ntok = build_probe_ids(&g, tokens, &ids);
    if (ntok <= 0) {
        fprintf(stderr, "q4: tokenize failed\n");
        q4_sess_close(sess);
        q4_ple_close(ple);
        q4_expert_cache_close(c);
        q4_store_close(st);
        q4_gguf_close(&g);
        return 1;
    }
    fprintf(stdout, "prefill-probe  tokens=%d ctx=%u\n", ntok, ctx);
    double t_all = now_s();
    int off = 0;
    int chunk_i = 0;
    int csz = 0;
    while (off < ntok) {
        csz = ntok - off;
        int pc = (int)q4_pref_chunk();
        if (csz > pc) csz = pc;
        if (off + csz < ntok) {
            int cn = ntok - off - csz;
            if (cn > pc) cn = pc;
            (void)q4_sess_ple_ahead(sess, ids + off + csz, cn);
        }
        double t0 = now_s();
        if (!q4_sess_prefill(sess, ids + off, csz)) {
            fprintf(stderr, "q4: prefill failed at %d / %d\n", off, ntok);
            free(ids);
            q4_sess_close(sess);
            q4_ple_close(ple);
            q4_expert_cache_close(c);
            q4_store_close(st);
            q4_gguf_close(&g);
            return 1;
        }
        double dt = now_s() - t0;
        off += csz;
        chunk_i++;
        fprintf(stdout, "  chunk %d  +%d  at %d  %.2f s  %.1f tok/s\n", chunk_i,
                csz, off, dt, dt > 0 ? (double)csz / dt : 0);
        fflush(stdout);
    }
    double tall = now_s() - t_all;
    uint64_t l1h = 0, l2h = 0, ssd = 0, bytes = 0;
    q4_expert_cache_stats_ex(c, &l1h, &l2h, &ssd, &bytes);
    fprintf(stdout, "prefill total  %d tok  %.2f s  %.1f tok/s\n", ntok, tall,
            tall > 0 ? (double)ntok / tall : 0);
    fprintf(stdout,
            "moe split      io %.2f s (copy_wait %.2f)  d2h %.2f  gemm %.2f  "
            "topk %.2f  rt %.2f  pf %.2f  sh %.2f\n",
            q4_prof_moe_io, q4_prof_moe_cw, q4_prof_moe_d2h,
            q4_prof_moe_gemm, q4_prof_moe_topk, q4_prof_moe_rt,
            q4_prof_moe_pf, q4_prof_moe_sh);
    q4_hip_gemv_prof_report();
    /* Diagnostic: top-3 logits of the last prefill token and of every
     * decode step, for comparing runs at fp-print precision
     * (Q4_LOGIT_DUMP=1). */
    int dump = getenv("Q4_LOGIT_DUMP") != NULL;
    if (dump && q4_sess_logits_at(sess, csz - 1)) {
        const float *lg = q4_sess_logits(sess);
        uint32_t nv = g.n_vocab;
        if (lg && nv) {
            int32_t ti[3] = {-1, -1, -1};
            float tv[3] = {0, 0, 0};
            for (uint32_t i = 0; i < nv; i++) {
                float v = lg[i];
                for (int k = 0; k < 3; k++) {
                    if (ti[k] < 0 || v > tv[k]) {
                        for (int j = 2; j > k; j--) {
                            ti[j] = ti[j - 1];
                            tv[j] = tv[j - 1];
                        }
                        ti[k] = (int32_t)i;
                        tv[k] = v;
                        break;
                    }
                }
            }
            fprintf(stdout, "  pref %d:%.9e %d:%.9e %d:%.9e\n",
                    ti[0], tv[0], ti[1], tv[1], ti[2], tv[2]);
        }
    }
    if (getenv("Q4_GEMV_PROF") || getenv("Q4_STEP_PROF") ||
        getenv("Q4_PROFILE")) {
        q4_hip_gemv_prof_reset();
        memset(q4_prof_dec, 0, sizeof(q4_prof_dec));
        memset(q4_prof_step, 0, sizeof(q4_prof_step));
        q4_prof_gpu_ms = 0;
        memset(q4_prof_gpu_ph, 0, sizeof(q4_prof_gpu_ph));
        memset(q4_prof_gpu_sec, 0, sizeof(q4_prof_gpu_sec));
        q4_prof_moe_io = q4_prof_moe_d2h = q4_prof_moe_gemm = 0;
        q4_prof_moe_cw = q4_prof_moe_xd2h = q4_prof_moe_topk = 0;
        q4_prof_moe_sg = q4_prof_moe_join = q4_prof_moe_merg = 0;
        q4_prof_moe_ng = q4_prof_moe_nc = q4_prof_moe_ncall = 0;
    }
    {
        int32_t tok = ids[ntok - 1];
        const int nd = (int)(gen_n ? gen_n : 16);
        int32_t *gen = malloc((size_t)nd * sizeof(int32_t));
        int got = 0;
        for (int i = 0; i < nd && gen; i++) {
            int32_t nxt = 0;
            float lg = 0.f;
            double t0 = now_s();
            if (!q4_sess_decode(sess, tok, &nxt, &lg)) {
                fprintf(stderr, "q4: decode failed at %d\n", i);
                break;
            }
            double dt = now_s() - t0;
            fprintf(stdout, "  dec %d  id %d -> %d  %.3f s  %.2f tok/s\n", i, tok,
                    nxt, dt, dt > 0 ? 1.0 / dt : 0);
            if (dump) {
                const float *lv = q4_sess_logits(sess);
                if (lv) {
                    int32_t b1 = -1; float v1 = 0.f;
                    for (uint32_t j = 0; j < g.n_vocab; j++)
                        if (j != (uint32_t)nxt && (b1 < 0 || lv[j] > v1)) {
                            b1 = (int32_t)j; v1 = lv[j];
                        }
                    fprintf(stdout,
                            "    top %d:%.9e runner %d:%.9e (margin %.6f)\n",
                            nxt, lg, b1, v1, lg - v1);
                }
            }

            gen[got++] = nxt;
            tok = nxt;
        }
        if (got > 0) {
            q4_tok *tk = q4_tok_open(&g);
            char text[512];
            int nb = tk ? q4_tok_decode(tk, gen, got, text, (int)sizeof(text) - 1) : -1;
            if (nb < 0) nb = 0;
            text[nb] = 0;
            for (int i = 0; i < nb; i++)
                if (text[i] < 32) text[i] = ' ';
            fprintf(stdout, "  sample: %s\n", text);
            q4_tok_close(tk);
        }
        free(gen);
    }
    /* Same decode profile tail as decode-probe (populated when
     * Q4_PROFILE=1 / Q4_STEP_PROF=1 during the gen loop). */
    {
        fprintf(stdout, "dec profile tail:\n");
        fprintf(stdout,
                "dec prof        embed %.3f  attn %.3f  moe %.3f  comb %.3f  head %.3f\n",
                q4_prof_dec[0], q4_prof_dec[1], q4_prof_dec[2], q4_prof_dec[3],
                q4_prof_dec[4]);
        fprintf(stdout,
                "  attn-detail   ple %.3f  hc-attn %.3f  qsa/gdn %.3f  comb+ffnmix %.3f\n",
                q4_prof_dec[5], q4_prof_dec[6], q4_prof_dec[7], q4_prof_dec[8]);
        fprintf(stdout,
                "  step          gA+sync %.3f  hostpart %.3f  gB %.3f  "
                "e-attn %.3f  e-moe %.3f  e-comb %.3f  embed %.3f  head %.3f\n",
                q4_prof_step[0], q4_prof_step[1], q4_prof_step[2],
                q4_prof_step[3], q4_prof_step[4], q4_prof_step[5],
                q4_prof_step[6], q4_prof_step[7]);
        fprintf(stdout,
                "  gpu-ms        win %.0f  gA %.0f  gB %.0f  head %.0f  "
                "e-attn %.0f  e-moe %.0f\n"
                "  gA-sections   hc1 %.0f  gdn %.0f  comb %.0f  ffnmix %.0f  moea %.0f\n",
                q4_prof_gpu_ms, q4_prof_gpu_ph[1], q4_prof_gpu_ph[2],
                q4_prof_gpu_ph[3], q4_prof_gpu_ph[4], q4_prof_gpu_ph[5],
                q4_prof_gpu_sec[0], q4_prof_gpu_sec[1], q4_prof_gpu_sec[2],
                q4_prof_gpu_sec[3], q4_prof_gpu_sec[4]);
        fprintf(stdout,
                "  moe prof      io %.3f  d2h %.3f  gemm %.3f  topk %.3f  "
                "xd2h %.3f  sg %.3f  join %.3f  merg %.3f\n",
                q4_prof_moe_io, q4_prof_moe_d2h, q4_prof_moe_gemm,
                q4_prof_moe_topk, q4_prof_moe_xd2h, q4_prof_moe_sg,
                q4_prof_moe_join, q4_prof_moe_merg);
        if (q4_prof_moe_ncall)
            fprintf(stdout,
                    "  moe split/tok gpu-experts %.1f  cpu-experts %.1f\n",
                    q4_prof_moe_ng / q4_prof_moe_ncall,
                    q4_prof_moe_nc / q4_prof_moe_ncall);
    }
    fprintf(stdout,
            "expert cache   L1 %llu  L2 %llu  SSD %llu  SSD %.2f GiB\n",
            (unsigned long long)l1h, (unsigned long long)l2h,
            (unsigned long long)ssd, (double)bytes / Q4_GIB);
    free(ids);
    q4_sess_close(sess);
    q4_ple_close(ple);
    q4_expert_cache_close(c);
    q4_store_close(st);
    q4_gguf_close(&g);
    return 0;
}

static int cmd_chat(const char *path, uint32_t ctx, uint64_t l2_bytes, int pin_ja,
                    int max_new, const char *system, const char *mtp_path) {
    q4_engine *e = open_engine(path, ctx, l2_bytes, pin_ja, mtp_path);
    if (!e) return 1;
    if (!system || !system[0]) system = "日本語で、簡潔かつ正確に答えてください。";
    fprintf(stdout,
            "q4 chat  ctx=%u  /reset  /exit  (Ctrl-D also exits)\n",
            q4_engine_ctx(e));
    fflush(stdout);
    char *roles[128];
    char *texts[128];
    int nmsg = 0;
    roles[nmsg] = strdup("system");
    texts[nmsg] = strdup(system);
    nmsg++;
    /* getline: a pasted prompt longer than a fixed buffer used to arrive
     * as several messages */
    char *line = NULL;
    size_t line_cap = 0;
    while (1) {
        fputs("you> ", stdout);
        fflush(stdout);
        if (getline(&line, &line_cap, stdin) < 0) break;
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!L) continue;
        if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
        if (!strcmp(line, "/reset")) {
            for (int i = 1; i < nmsg; i++) {
                free(roles[i]);
                free(texts[i]);
            }
            nmsg = 1;
            fprintf(stdout, "(reset)\n");
            continue;
        }
        if (nmsg + 2 >= 128) {
            fprintf(stderr, "q4: history full, /reset\n");
            continue;
        }
        roles[nmsg] = strdup("user");
        texts[nmsg] = strdup(line);
        nmsg++;
        char *out = malloc(1u << 18);
        if (!out) break;
        fputs("q4> ", stdout);
        fflush(stdout);
        int n = q4_engine_generate_msgs(e, (const char *const *)roles,
                                        (const char *const *)texts, nmsg, out,
                                        1 << 18, max_new, chat_stream, NULL);
        fputc('\n', stdout);
        if (n < 0) {
            fprintf(stderr, "q4: generate failed\n");
            free(out);
            nmsg--;
            free(roles[nmsg]);
            free(texts[nmsg]);
            continue;
        }
        roles[nmsg] = strdup("assistant");
        texts[nmsg] = strdup(out[0] ? out : "");
        nmsg++;
        free(out);
    }
    free(line);
    for (int i = 0; i < nmsg; i++) {
        free(roles[i]);
        free(texts[i]);
    }
    q4_engine_close(e);
    return 0;
}

static int cmd_serve(const char *path, uint32_t ctx, uint64_t l2_bytes, int pin_ja,
                     const char *host, int port, const char *mtp_path) {
    q4_engine *e = open_engine(path, ctx, l2_bytes, pin_ja, mtp_path);
    if (!e) return 1;
    int rc = q4_serve(e, host, port);
    q4_engine_close(e);
    return rc;
}

int main(int argc, char **argv) {
    q4_self_scope(argc, argv);
    q4_install_crashdump();
    if (argc < 2) {
        usage(stderr);
        return 1;
    }
    const char *cmd = argv[1];
    const char *model = default_model();
    uint32_t ctx = 65536, tokens = 32, slots = 0, layer = 0, gen = 16;
    int ctx_explicit = 0;
    uint64_t l2_bytes = 0;
    int port = 8090, max_new = 512, pin_ja = 1;
    const char *host = "127.0.0.1";
    const char *system = NULL;
    const char *mtp_path = getenv("Q4_MTP");
    bool warm = false;
    const char *ppl_file = NULL;
    q4_ple_io ple_io = Q4_PLE_IO_PREAD;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            ctx = (uint32_t)atoi(argv[++i]);
            ctx_explicit = 1;
        }
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            tokens = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--gen") == 0 && i + 1 < argc)
            gen = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc)
            slots = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc)
            layer = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc)
            host = argv[++i];
        else if (strcmp(argv[i], "--l2-gib") == 0 && i + 1 < argc)
            l2_bytes = (uint64_t)atoi(argv[++i]) * Q4_GIB;
        else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc)
            max_new = atoi(argv[++i]);
        else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc)
            system = argv[++i];
        else if (strcmp(argv[i], "--no-pin-ja") == 0)
            pin_ja = 0;
        else if (strcmp(argv[i], "--mtp") == 0 && i + 1 < argc)
            mtp_path = argv[++i];
        else if (strcmp(argv[i], "--no-mtp") == 0)
            mtp_path = NULL;
        else if (strcmp(argv[i], "--io") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "mmap") == 0) ple_io = Q4_PLE_IO_MMAP;
            else if (strcmp(v, "pread") == 0) ple_io = Q4_PLE_IO_PREAD;
            else {
                fprintf(stderr, "q4: --io wants pread or mmap\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--warm") == 0)
            warm = true;
        else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc)
            ppl_file = argv[++i];
        else if (argv[i][0] != '-')
            model = argv[i];
        else {
            fprintf(stderr, "q4: unknown arg %s\n", argv[i]);
            return 1;
        }
    }
    if (!model) {
        fprintf(stderr, "q4: pass a GGUF path or set Q4_MODEL\n");
        return 1;
    }
    if (strcmp(cmd, "inspect") == 0) return cmd_inspect(model);
    if (strcmp(cmd, "plan") == 0) return cmd_plan(model, ctx);
    if (strcmp(cmd, "tensors") == 0) return cmd_tensors(model);
    if (strcmp(cmd, "ple-probe") == 0) return cmd_ple_probe(model, tokens, ple_io);
    if (strcmp(cmd, "io-bench") == 0)
        return cmd_io_bench(model, tokens, slots, ctx, warm);
    if (strcmp(cmd, "moe-probe") == 0)
        return cmd_moe_probe(model, layer, tokens, slots);
    if (strcmp(cmd, "decode-probe") == 0)
        return cmd_decode_probe(model, tokens, slots, ctx);
    if (strcmp(cmd, "prefill-probe") == 0)
        return cmd_prefill_probe(model, tokens, slots, ctx, l2_bytes, gen);
    if (strcmp(cmd, "ppl") == 0)
        return cmd_ppl(model, ppl_file, tokens, ctx, l2_bytes);
    if (strcmp(cmd, "chat") == 0)
        return cmd_chat(model, ctx, l2_bytes, pin_ja, max_new, system, mtp_path);
    if (strcmp(cmd, "serve") == 0) {
        if (!ctx_explicit) ctx = 262144;
        return cmd_serve(model, ctx, l2_bytes, pin_ja, host, port, mtp_path);
    }
    usage(stderr);
    return 1;
}
