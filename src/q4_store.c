#define _GNU_SOURCE
#include "q4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct q4_store {
    const q4_gguf *g;
    uint8_t **h;
    uint8_t **d;
    uint32_t n;
    uint64_t host_bytes;
    uint64_t dev_bytes;
};

q4_store *q4_store_open(const q4_gguf *g) {
    if (!g) return NULL;
    q4_store *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->g = g;
    s->n = g->n_tensors;
    s->h = calloc(s->n, sizeof(uint8_t *));
    s->d = calloc(s->n, sizeof(uint8_t *));
    if (!s->h || !s->d) {
        q4_store_close(s);
        return NULL;
    }
    for (uint32_t i = 0; i < s->n; i++) {
        const q4_tensor *t = &g->tensors[i];
        if (t->kind == Q4_KIND_ROUTED || t->kind == Q4_KIND_PLE ||
            t->kind == Q4_KIND_MTP)
            continue;
        if (t->nbytes == 0) continue;
        uint8_t *p = NULL;
        if (posix_memalign((void **)&p, 64, (size_t)t->nbytes) != 0) p = malloc((size_t)t->nbytes);
        if (!p) {
            q4_store_close(s);
            return NULL;
        }
        if (!q4_tensor_read(g, t, p, t->nbytes)) {
            free(p);
            q4_store_close(s);
            return NULL;
        }
        s->h[i] = p;
        s->host_bytes += t->nbytes;
    }
    return s;
}

void q4_store_close(q4_store *s) {
    if (!s) return;
    if (s->h) {
        for (uint32_t i = 0; i < s->n; i++) free(s->h[i]);
        free(s->h);
    }
    if (s->d) {
        for (uint32_t i = 0; i < s->n; i++) {
            if (s->d[i]) q4_hip_free(s->d[i]);
        }
        free(s->d);
    }
    free(s);
}

const uint8_t *q4_store_get(const q4_store *s, const q4_tensor *t) {
    if (!s || !t || !s->g) return NULL;
    uint32_t i = (uint32_t)(t - s->g->tensors);
    if (i >= s->n) return NULL;
    return s->h[i];
}

const uint8_t *q4_store_get_name(const q4_store *s, const char *name) {
    return q4_store_get(s, q4_find_tensor(s->g, name));
}

const uint8_t *q4_store_dev(const q4_store *s, const q4_tensor *t) {
    if (!s || !t || !s->g) return NULL;
    uint32_t i = (uint32_t)(t - s->g->tensors);
    if (i >= s->n) return NULL;
    return s->d[i];
}

bool q4_store_replace_dev(q4_store *s, const q4_tensor *t, void *d) {
    if (!s || !t || !s->g || !d) return false;
    uint32_t i = (uint32_t)(t - s->g->tensors);
    if (i >= s->n) return false;
    if (s->d[i] == (uint8_t *)d) return true;
    if (s->d[i]) q4_hip_free(s->d[i]);
    s->d[i] = (uint8_t *)d;
    return true;
}

void q4_store_free_host_except(q4_store *s, const q4_gguf *g) {
    if (!s || !g) return;
    char ple_conv[Q4_MAX_NAME] = "";
    if (g->ple_layer != UINT32_MAX)
        snprintf(ple_conv, sizeof(ple_conv), "blk.%u.ple_conv1d.weight",
                 g->ple_layer);
    for (uint32_t i = 0; i < s->n; i++) {
        if (!s->h[i]) continue;
        const q4_tensor *t = &s->g->tensors[i];
        if (!strcmp(t->name, "token_embd.weight")) continue;
        if (ple_conv[0] && !strcmp(t->name, ple_conv)) continue;
        free(s->h[i]);
        s->h[i] = NULL;
    }
}

bool q4_store_upload_hip(q4_store *s) {
    if (!s || !q4_hip_ok()) return false;
    for (uint32_t i = 0; i < s->n; i++) {
        if (!s->h[i] || s->d[i]) continue;
        const q4_tensor *t = &s->g->tensors[i];
        void *d = q4_hip_malloc((size_t)t->nbytes);
        if (!d) {
            fprintf(stderr, "q4: hip malloc failed for %s (%.2f MiB)\n", t->name,
                    (double)t->nbytes / (1024.0 * 1024.0));
            return false;
        }
        if (!q4_hip_h2d(d, s->h[i], (size_t)t->nbytes)) {
            q4_hip_free(d);
            return false;
        }
        s->d[i] = d;
        s->dev_bytes += t->nbytes;
    }
    return true;
}
