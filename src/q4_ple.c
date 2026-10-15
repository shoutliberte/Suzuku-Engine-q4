#define _GNU_SOURCE
#include "q4.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define QK4_NL 32

/* ggml IQ4_NL lookup (ggml-quants). */
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

struct q4_ple {
    const q4_gguf *g;
    const q4_tensor *t;
    const uint8_t *map; /* mmap of the shard, or NULL → pread */
    size_t map_len;
    uint64_t row_bytes;
    uint64_t n_rows;
};

static float f16_to_f32(uint16_t h) {
    uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            e = 127 - 14;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3ffu;
            u = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void dequant_iq4_nl_row(const uint8_t *src, uint32_t n, float *dst) {
    const uint32_t n_blk = n / QK4_NL;
    for (uint32_t b = 0; b < n_blk; b++) {
        const uint8_t *blk = src + (size_t)b * 18u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        for (int j = 0; j < QK4_NL / 2; j++) {
            const uint8_t q = qs[j];
            dst[b * QK4_NL + j] = d * (float)kvalues_iq4nl[q & 0xf];
            dst[b * QK4_NL + j + QK4_NL / 2] = d * (float)kvalues_iq4nl[q >> 4];
        }
    }
}

/* Q4_0 PLE rows: same 18-byte/32-value blocks as IQ4_NL, linear map. */
static void dequant_q4_0_row(const uint8_t *src, uint32_t n, float *dst) {
    const uint32_t n_blk = n / QK4_NL;
    for (uint32_t b = 0; b < n_blk; b++) {
        const uint8_t *blk = src + (size_t)b * 18u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        for (int j = 0; j < QK4_NL / 2; j++) {
            const uint8_t q = qs[j];
            dst[b * QK4_NL + j] = d * (float)((int)(q & 0xf) - 8);
            dst[b * QK4_NL + j + QK4_NL / 2] =
                d * (float)((int)(q >> 4) - 8);
        }
    }
}

q4_ple *q4_ple_open(const q4_gguf *g) {
    return q4_ple_open_io(g, Q4_PLE_IO_PREAD);
}

q4_ple *q4_ple_open_io(const q4_gguf *g, q4_ple_io io) {
    if (!g || g->ple_table < 0) return NULL;
    const q4_tensor *t = &g->tensors[g->ple_table];
    if ((t->ggml_type != 20 /* IQ4_NL */ && t->ggml_type != 2 /* Q4_0 */) ||
        t->n_dims < 2)
        return NULL;
    q4_ple *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->g = g;
    p->t = t;
    p->n_rows = t->ne[1];
    p->row_bytes = t->nbytes / p->n_rows;
    const q4_file *f = &g->files[t->shard];
    /* Random 90-byte rows: keep DRAM for expert L2, not a 27 GiB n-gram map. */
    posix_fadvise(f->fd, 0, 0, POSIX_FADV_RANDOM);
    if (io == Q4_PLE_IO_MMAP) {
        p->map_len = (size_t)f->file_size;
        void *m = mmap(NULL, p->map_len, PROT_READ, MAP_PRIVATE, f->fd, 0);
        if (m == MAP_FAILED) {
            p->map = NULL;
            p->map_len = 0;
        } else {
            p->map = m;
            posix_madvise((void *)m, p->map_len, POSIX_MADV_RANDOM);
        }
    }
    return p;
}

void q4_ple_close(q4_ple *p) {
    if (!p) return;
    if (p->map && p->map_len)
        munmap((void *)p->map, p->map_len);
    free(p);
}

bool q4_ple_hash(const q4_ple *p, const int32_t *toks, int64_t n,
                 const int32_t *hist, int32_t *rows_out) {
    if (!p || !toks || !rows_out || n <= 0) return false;
    const q4_gguf *g = p->g;
    const int64_t n_gram = g->ple_ngram;
    const int64_t per = g->ple_heads_per;
    const int64_t n_heads = g->ple_n_heads;
    const int32_t eos = (int32_t)g->ple_eos;
    if (n_gram < 2 || n_heads == 0) return false;

    for (int64_t i = 0; i < n; i++) {
        int64_t ctx[Q4_MAX_PLE_NGRAM];
        ctx[0] = toks[i];
        bool cut = false;
        for (int64_t s = 1; s < n_gram; s++) {
            int64_t v = eos;
            const int64_t j = i - s;
            if (j >= 0) {
                v = toks[j];
            } else if (hist) {
                const int64_t k = (n_gram - 1) + j; /* hist[0] is oldest */
                if (k >= 0 && k < n_gram - 1) v = hist[k];
            }
            ctx[s] = cut ? eos : v;
            if (ctx[s] == eos) cut = true;
        }
        for (int64_t ng = 2; ng <= n_gram; ng++) {
            uint64_t mixed = (uint64_t)(uint32_t)ctx[0] * g->ple_mult[0];
            for (int64_t j = 1; j < ng; j++)
                mixed ^= (uint64_t)(uint32_t)ctx[j] * g->ple_mult[j];
            const int64_t base = (ng - 2) * per;
            for (int64_t h = 0; h < per; h++) {
                const int64_t hi = base + h;
                rows_out[i * n_heads + hi] = (int32_t)(
                    mixed % g->ple_vocab[hi] + g->ple_off[hi]);
            }
        }
    }
    return true;
}

bool q4_ple_gather(const q4_ple *p, const int32_t *rows, int64_t n_rows,
                   float *out) {
    if (!p || !rows || !out || n_rows <= 0) return false;
    const uint32_t width = p->g->n_embd_ple ? p->g->n_embd_ple : 160;
    const q4_file *f = &p->g->files[p->t->shard];
    const uint64_t base = f->data_off + p->t->offset;
    uint8_t tmp[256];
    if (p->row_bytes > sizeof(tmp)) return false;

    for (int64_t i = 0; i < n_rows; i++) {
        const int32_t r = rows[i];
        if (r < 0 || (uint64_t)r >= p->n_rows) return false;
        const uint64_t off = base + (uint64_t)r * p->row_bytes;
        const uint8_t *src;
        if (p->map) {
            src = p->map + off;
        } else {
            uint64_t got = 0;
            while (got < p->row_bytes) {
                ssize_t n = pread(f->fd, tmp + got, (size_t)(p->row_bytes - got),
                                  (off_t)(off + got));
                if (n <= 0) return false;
                got += (uint64_t)n;
            }
            src = tmp;
        }
        if (p->t->ggml_type == 2 /* Q4_0 */)
            dequant_q4_0_row(src, width, out + i * width);
        else
            dequant_iq4_nl_row(src, width, out + i * width);
    }
    return true;
}

uint32_t q4_ple_row_bytes(const q4_ple *p) { return p ? p->row_bytes : 0; }
