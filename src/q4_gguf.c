#include "q4.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" */

static uint32_t fnv1a(const char *s);

enum {
    GGUF_U8 = 0,
    GGUF_I8,
    GGUF_U16,
    GGUF_I16,
    GGUF_U32,
    GGUF_I32,
    GGUF_F32,
    GGUF_BOOL,
    GGUF_STRING,
    GGUF_ARRAY,
    GGUF_U64,
    GGUF_I64,
    GGUF_F64,
};

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "q4: out of memory (%zu)\n", n);
        abort();
    }
    return p;
}

static bool key_eq(const char *key, const char *arch, const char *field) {
    if (!key || !arch || !field) return false;
    size_t al = strlen(arch), fl = strlen(field);
    if (strlen(key) != al + 1 + fl) return false;
    return strncmp(key, arch, al) == 0 && key[al] == '.' &&
           memcmp(key + al + 1, field, fl) == 0;
}

static bool read_full(int fd, void *buf, size_t n) {
    uint8_t *p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r == 0) {
            errno = EPIPE;
            return false;
        }
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += (size_t)r;
        n -= (size_t)r;
    }
    return true;
}

static bool read_u32(int fd, uint32_t *v) {
    return read_full(fd, v, 4);
}
static bool read_u64(int fd, uint64_t *v) {
    return read_full(fd, v, 8);
}

static bool read_str(int fd, char **out) {
    uint64_t n = 0;
    if (!read_u64(fd, &n)) return false;
    if (n > 1ull << 20) return false;
    char *s = xmalloc((size_t)n + 1);
    if (n && !read_full(fd, s, (size_t)n)) {
        free(s);
        return false;
    }
    s[n] = 0;
    *out = s;
    return true;
}

static bool skip_bytes(int fd, uint64_t n) {
    if (lseek(fd, (off_t)n, SEEK_CUR) == (off_t)-1) return false;
    return true;
}

static uint64_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_U8:
    case GGUF_I8:
    case GGUF_BOOL:
        return 1;
    case GGUF_U16:
    case GGUF_I16:
        return 2;
    case GGUF_U32:
    case GGUF_I32:
    case GGUF_F32:
        return 4;
    case GGUF_U64:
    case GGUF_I64:
    case GGUF_F64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(int fd, uint32_t t);

static bool skip_array(int fd) {
    uint32_t et = 0;
    uint64_t n = 0;
    if (!read_u32(fd, &et) || !read_u64(fd, &n)) return false;
    if (et == GGUF_STRING) {
        for (uint64_t i = 0; i < n; i++) {
            char *s = NULL;
            if (!read_str(fd, &s)) return false;
            free(s);
        }
        return true;
    }
    if (et == GGUF_ARRAY) {
        for (uint64_t i = 0; i < n; i++) {
            if (!skip_array(fd)) return false;
        }
        return true;
    }
    uint64_t es = scalar_size(et);
    if (es == 0 || n > (UINT64_MAX / es)) return false;
    return skip_bytes(fd, es * n);
}

static bool skip_value(int fd, uint32_t t) {
    if (t == GGUF_STRING) {
        char *s = NULL;
        if (!read_str(fd, &s)) return false;
        free(s);
        return true;
    }
    if (t == GGUF_ARRAY) return skip_array(fd);
    uint64_t es = scalar_size(t);
    if (es == 0) return false;
    return skip_bytes(fd, es);
}

static bool kv_u32(int fd, uint32_t t, uint32_t *out) {
    if (t == GGUF_U32 || t == GGUF_I32) return read_u32(fd, out);
    if (t == GGUF_U64 || t == GGUF_I64) {
        uint64_t v = 0;
        if (!read_u64(fd, &v) || v > UINT32_MAX) return false;
        *out = (uint32_t)v;
        return true;
    }
    return skip_value(fd, t);
}

static bool kv_str(int fd, uint32_t t, char **out) {
    if (t == GGUF_STRING) return read_str(fd, out);
    skip_value(fd, t);
    return true;
}

static bool kv_arr_str(int fd, uint32_t t, char ***out, uint32_t *n_out) {
    if (t != GGUF_ARRAY) return skip_value(fd, t);
    uint32_t et = 0;
    uint64_t n = 0;
    if (!read_u32(fd, &et) || !read_u64(fd, &n)) return false;
    if (et != GGUF_STRING || n > 2000000ull) return false;
    char **a = calloc((size_t)n, sizeof(char *));
    if (!a) return false;
    for (uint64_t i = 0; i < n; i++) {
        if (!read_str(fd, &a[i])) {
            for (uint64_t j = 0; j < i; j++) free(a[j]);
            free(a);
            return false;
        }
    }
    *out = a;
    if (n_out) *n_out = (uint32_t)n;
    return true;
}

static bool kv_arr_u64(int fd, uint32_t t, uint64_t *dst, uint32_t cap,
                       uint32_t *n_out) {
    if (t != GGUF_ARRAY) return skip_value(fd, t);
    uint32_t et = 0;
    uint64_t n = 0;
    if (!read_u32(fd, &et) || !read_u64(fd, &n)) return false;
    if (n > cap) return false;
    for (uint64_t i = 0; i < n; i++) {
        if (et == GGUF_U64 || et == GGUF_I64) {
            if (!read_u64(fd, &dst[i])) return false;
        } else if (et == GGUF_U32 || et == GGUF_I32) {
            uint32_t v = 0;
            if (!read_u32(fd, &v)) return false;
            dst[i] = v;
        } else {
            return false;
        }
    }
    if (n_out) *n_out = (uint32_t)n;
    return true;
}

static bool kv_f32(int fd, uint32_t t, float *out) {
    if (t == GGUF_F32) return read_full(fd, out, 4);
    if (t == GGUF_F64) {
        double d = 0;
        if (!read_full(fd, &d, 8)) return false;
        *out = (float)d;
        return true;
    }
    return skip_value(fd, t);
}

static q4_kind classify(const char *name, int32_t *layer) {
    *layer = -1;
    const char *blk = strstr(name, "blk.");
    if (blk) {
        blk += 4;
        int32_t L = 0;
        if (isdigit((unsigned char)*blk)) {
            L = (int32_t)strtol(blk, NULL, 10);
            *layer = L;
        }
    }
    if (strstr(name, "per_layer_token_embd")) return Q4_KIND_PLE;
    if (strstr(name, "_exps")) return Q4_KIND_ROUTED;
    if (strstr(name, "_shexp") || strstr(name, "ffn_gate_inp")) return Q4_KIND_SHARED;
    return Q4_KIND_DENSE;
}

/* 00001-of-00003.gguf → fill siblings. */
static uint32_t discover_shards(const char *path, char out[][4096], uint32_t max) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    const char *of = strstr(base, "-of-");
    if (!of) {
        snprintf(out[0], 4096, "%s", path);
        return 1;
    }
    /* ...-00001-of-00003.gguf */
    const char *dash = of;
    while (dash > base && dash[-1] != '-') dash--;
    int idx = atoi(dash);
    int tot = atoi(of + 4);
    if (idx < 1 || tot < 1 || (uint32_t)tot > max) {
        snprintf(out[0], 4096, "%s", path);
        return 1;
    }
    char dir[4096];
    char prefix[4096];
    char suffix[256];
    size_t dirlen = (size_t)(base - path);
    memcpy(dir, path, dirlen);
    dir[dirlen] = 0;
    size_t prelen = (size_t)(dash - base);
    memcpy(prefix, base, prelen);
    prefix[prelen] = 0;
    const char *ext = strstr(of, ".gguf");
    snprintf(suffix, sizeof(suffix), "%s", ext ? ext : ".gguf");
    uint32_t n = 0;
    for (int i = 1; i <= tot; i++) {
        snprintf(out[n], 4096, "%s%s%05d-of-%05d%s", dir, prefix, i, tot, suffix);
        n++;
    }
    return n;
}

static bool open_file(q4_file *f, const char *path) {
    memset(f, 0, sizeof(*f));
    snprintf(f->path, sizeof(f->path), "%s", path);
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) {
        fprintf(stderr, "q4: open %s: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(f->fd, &st) != 0) {
        fprintf(stderr, "q4: fstat %s: %s\n", path, strerror(errno));
        close(f->fd);
        f->fd = -1;
        return false;
    }
    f->file_size = (uint64_t)st.st_size;
    return true;
}

static bool parse_header(q4_gguf *g, q4_file *f, bool take_kv) {
    if (lseek(f->fd, 0, SEEK_SET) == (off_t)-1) return false;
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!read_u32(f->fd, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "q4: %s is not GGUF\n", f->path);
        return false;
    }
    if (!read_u32(f->fd, &version) || !read_u64(f->fd, &n_tensors) ||
        !read_u64(f->fd, &n_kv))
        return false;
    if (version < 2 || version > 3) {
        fprintf(stderr, "q4: unsupported GGUF version %u in %s\n", version, f->path);
        return false;
    }
    if (n_tensors > 200000) return false;

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = NULL;
        uint32_t t = 0;
        if (!read_str(f->fd, &key) || !read_u32(f->fd, &t)) {
            free(key);
            return false;
        }
        bool keep = take_kv;
        const char *a = g->arch;
        if (keep && strcmp(key, "general.architecture") == 0) {
            free(g->arch);
            if (!kv_str(f->fd, t, &g->arch)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "general.sampling.temp") == 0) {
            if (!kv_f32(f->fd, t, &g->samp_temp)) {
                free(key);
                return false;
            }
            g->samp_has_temp = t == GGUF_F32 || t == GGUF_F64;
        } else if (keep && strcmp(key, "general.sampling.top_p") == 0) {
            if (!kv_f32(f->fd, t, &g->samp_top_p)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "general.sampling.top_k") == 0) {
            if (!kv_u32(f->fd, t, &g->samp_top_k)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "general.name") == 0) {
            free(g->name);
            if (!kv_str(f->fd, t, &g->name)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "block_count")) {
            if (!kv_u32(f->fd, t, &g->n_layer)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "embedding_length")) {
            if (!kv_u32(f->fd, t, &g->n_embd)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "feed_forward_length")) {
            if (!kv_u32(f->fd, t, &g->n_ff)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "expert_count")) {
            if (!kv_u32(f->fd, t, &g->n_expert)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "expert_used_count")) {
            if (!kv_u32(f->fd, t, &g->n_expert_used)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.head_count")) {
            if (!kv_u32(f->fd, t, &g->n_head)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "context_length")) {
            if (!kv_u32(f->fd, t, &g->n_ctx_train)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.head_count_kv")) {
            if (!kv_u32(f->fd, t, &g->n_head_kv)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "expert_feed_forward_length")) {
            if (!kv_u32(f->fd, t, &g->n_ff_exp)) {
                free(key);
                return false;
            }
        } else if (keep && a &&
                   key_eq(key, a, "expert_shared_feed_forward_length")) {
            if (!kv_u32(f->fd, t, &g->n_ff_shexp)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "full_attention_interval")) {
            if (!kv_u32(f->fd, t, &g->full_attn_interval)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "hyper_connection.count")) {
            if (!kv_u32(f->fd, t, &g->hc_mult)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "hyper_connection.low_rank")) {
            if (!kv_u32(f->fd, t, &g->hc_rank)) {
                free(key);
                return false;
            }
        } else if (keep && a &&
                   key_eq(key, a, "embedding_length_per_layer_input")) {
            if (!kv_u32(f->fd, t, &g->n_embd_ple)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.ngram_size")) {
            if (!kv_u32(f->fd, t, &g->ple_ngram)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.heads_per_ngram")) {
            if (!kv_u32(f->fd, t, &g->ple_heads_per)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.conv_kernel")) {
            if (!kv_u32(f->fd, t, &g->ple_conv_kernel)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.eos_token_id")) {
            if (!kv_u32(f->fd, t, &g->ple_eos)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.image_token_id")) {
            if (!kv_u32(f->fd, t, &g->ple_image)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.layers")) {
            uint64_t tmp[8];
            uint32_t n = 0;
            if (!kv_arr_u64(f->fd, t, tmp, 8, &n)) {
                free(key);
                return false;
            }
            if (n) g->ple_layer = (uint32_t)tmp[0];
        } else if (keep && a && key_eq(key, a, "ple.layer_multipliers")) {
            uint32_t n = 0;
            if (!kv_arr_u64(f->fd, t, g->ple_mult, Q4_MAX_PLE_NGRAM, &n)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.head_offsets")) {
            uint32_t n = 0;
            if (!kv_arr_u64(f->fd, t, g->ple_off, Q4_MAX_PLE_HEADS, &n)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ple.head_vocab_sizes")) {
            uint32_t n = 0;
            if (!kv_arr_u64(f->fd, t, g->ple_vocab, Q4_MAX_PLE_HEADS, &n)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.key_length")) {
            if (!kv_u32(f->fd, t, &g->n_embd_head_k)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.value_length")) {
            if (!kv_u32(f->fd, t, &g->n_embd_head_v)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "rope.dimension_count")) {
            if (!kv_u32(f->fd, t, &g->n_rot)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ssm.conv_kernel")) {
            if (!kv_u32(f->fd, t, &g->ssm_d_conv)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ssm.state_size")) {
            if (!kv_u32(f->fd, t, &g->ssm_d_state)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ssm.group_count")) {
            if (!kv_u32(f->fd, t, &g->ssm_n_group)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ssm.time_step_rank")) {
            if (!kv_u32(f->fd, t, &g->ssm_dt_rank)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "ssm.inner_size")) {
            if (!kv_u32(f->fd, t, &g->ssm_d_inner)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.indexer.head_count")) {
            if (!kv_u32(f->fd, t, &g->indexer_n_head)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.indexer.key_length")) {
            if (!kv_u32(f->fd, t, &g->indexer_head_size)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.indexer.top_k")) {
            if (!kv_u32(f->fd, t, &g->indexer_top_k)) {
                free(key);
                return false;
            }
        } else if (keep && a && key_eq(key, a, "attention.compress_ratios")) {
            uint64_t tmp[Q4_MAX_LAYER];
            uint32_t n = 0;
            if (!kv_arr_u64(f->fd, t, tmp, Q4_MAX_LAYER, &n)) {
                free(key);
                return false;
            }
            for (uint32_t k = 0; k < n && k < Q4_MAX_LAYER; k++)
                g->compress_ratio[k] = (uint32_t)tmp[k];
        } else if (keep && a && key_eq(key, a, "rope.dimension_sections")) {
            uint64_t tmp[4];
            uint32_t n = 0;
            if (!kv_arr_u64(f->fd, t, tmp, 4, &n)) {
                free(key);
                return false;
            }
            for (uint32_t k = 0; k < n && k < 4; k++)
                g->rope_sections[k] = (int32_t)tmp[k];
        } else if (keep && a && key_eq(key, a, "rope.freq_base")) {
            if (!kv_f32(f->fd, t, &g->rope_freq_base)) {
                free(key);
                return false;
            }
        } else if (keep && a &&
                   key_eq(key, a, "attention.layer_norm_rms_epsilon")) {
            if (!kv_f32(f->fd, t, &g->rms_eps)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            if (!kv_u32(f->fd, t, &g->eos_id)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            if (!kv_u32(f->fd, t, &g->bos_id)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "tokenizer.ggml.padding_token_id") == 0) {
            if (!kv_u32(f->fd, t, &g->pad_id)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "tokenizer.ggml.tokens") == 0) {
            if (!kv_arr_str(f->fd, t, &g->tok_vocab, &g->n_tok_vocab)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "tokenizer.ggml.merges") == 0) {
            if (!kv_arr_str(f->fd, t, &g->tok_merges, &g->n_tok_merges)) {
                free(key);
                return false;
            }
        } else if (keep && strcmp(key, "tokenizer.chat_template") == 0) {
            free(g->chat_template);
            if (!kv_str(f->fd, t, &g->chat_template)) {
                free(key);
                return false;
            }
        } else {
            if (!skip_value(f->fd, t)) {
                free(key);
                return false;
            }
        }
        free(key);
    }

    uint32_t start = g->n_tensors;
    q4_tensor *nt =
        realloc(g->tensors, (size_t)(start + n_tensors) * sizeof(q4_tensor));
    if (!nt) return false;
    g->tensors = nt;
    memset(g->tensors + start, 0, (size_t)n_tensors * sizeof(q4_tensor));

    for (uint64_t i = 0; i < n_tensors; i++) {
        q4_tensor *t = &g->tensors[start + i];
        char *name = NULL;
        uint32_t nd = 0;
        if (!read_str(f->fd, &name) || !read_u32(f->fd, &nd)) {
            free(name);
            return false;
        }
        if (nd == 0 || nd > Q4_MAX_DIMS) {
            free(name);
            return false;
        }
        t->n_dims = nd;
        for (uint32_t d = 0; d < nd; d++) {
            if (!read_u64(f->fd, &t->ne[d])) {
                free(name);
                return false;
            }
        }
        if (!read_u32(f->fd, &t->ggml_type) || !read_u64(f->fd, &t->offset)) {
            free(name);
            return false;
        }
        snprintf(t->name, sizeof(t->name), "%s", name);
        free(name);
        t->shard = (uint32_t)(f - g->files);
        t->kind = classify(t->name, &t->layer);
        if (t->kind == Q4_KIND_ROUTED && t->n_dims >= 1)
            t->n_experts = (int32_t)t->ne[t->n_dims - 1];
    }

    off_t pos = lseek(f->fd, 0, SEEK_CUR);
    if (pos < 0) return false;
    uint64_t align = 32;
    uint64_t data = ((uint64_t)pos + align - 1u) & ~(align - 1u);
    f->data_off = data;

    g->n_tensors = start + (uint32_t)n_tensors;
    return true;
}

static int cmp_tensor(const void *a, const void *b) {
    const q4_tensor *x = a, *y = b;
    if (x->shard != y->shard) return x->shard < y->shard ? -1 : 1;
    if (x->offset != y->offset) return x->offset < y->offset ? -1 : 1;
    return 0;
}

static void fill_nbytes(q4_gguf *g) {
    if (g->n_tensors == 0) return;
    q4_tensor **ord = xmalloc(g->n_tensors * sizeof(*ord));
    for (uint32_t i = 0; i < g->n_tensors; i++) ord[i] = &g->tensors[i];
    /* sort indices via qsort on a copy */
    q4_tensor *tmp = xmalloc(g->n_tensors * sizeof(q4_tensor));
    memcpy(tmp, g->tensors, g->n_tensors * sizeof(q4_tensor));
    qsort(tmp, g->n_tensors, sizeof(q4_tensor), cmp_tensor);
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        uint64_t next;
        if (i + 1 < g->n_tensors && tmp[i + 1].shard == tmp[i].shard)
            next = tmp[i + 1].offset;
        else {
            q4_file *f = &g->files[tmp[i].shard];
            next = f->file_size - f->data_off;
        }
        tmp[i].nbytes = next > tmp[i].offset ? next - tmp[i].offset : 0;
    }
    /* write nbytes back by name+shard+offset */
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        for (uint32_t j = 0; j < g->n_tensors; j++) {
            if (g->tensors[j].shard == tmp[i].shard &&
                g->tensors[j].offset == tmp[i].offset &&
                strcmp(g->tensors[j].name, tmp[i].name) == 0) {
                g->tensors[j].nbytes = tmp[i].nbytes;
                break;
            }
        }
    }
    free(tmp);
    free(ord);
}

/* The IQ3E merge appends the MTP head as blk.48 and sets block_count to 49.
 * The target model is still 48 layers. Keep the draft block out of the
 * forward, the dense VRAM upload, and the expert cache. */
static void peel_mtp(q4_gguf *g) {
    int32_t nextn = -1;
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        if (!strstr(g->tensors[i].name, ".nextn.") || g->tensors[i].layer < 0)
            continue;
        if (nextn < 0 || g->tensors[i].layer < nextn)
            nextn = g->tensors[i].layer;
    }
    if (nextn < 0) return;
    g->n_layer = (uint32_t)nextn;
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        if (g->tensors[i].layer >= nextn)
            g->tensors[i].kind = Q4_KIND_MTP;
    }
}

static void summarize(q4_gguf *g, int keep_mtp_layer) {
    uint64_t routed_experts = 0;
    g->ple_table = -1;
    if (!keep_mtp_layer) peel_mtp(g);
    if (g->ple_ngram >= 2 && g->ple_heads_per)
        g->ple_n_heads = (g->ple_ngram - 1) * g->ple_heads_per;
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        const q4_tensor *t = &g->tensors[i];
        g->bytes_total += t->nbytes;
        switch (t->kind) {
        case Q4_KIND_ROUTED:
            g->bytes_routed += t->nbytes;
            if (t->n_experts > 0) routed_experts += (uint64_t)t->n_experts;
            if (g->n_expert == 0 && t->n_experts > 0)
                g->n_expert = (uint32_t)t->n_experts;
            break;
        case Q4_KIND_SHARED:
            g->bytes_shared += t->nbytes;
            break;
        case Q4_KIND_PLE:
            g->bytes_ple += t->nbytes;
            if (g->ple_table < 0) g->ple_table = (int32_t)i;
            break;
        case Q4_KIND_MTP:
            g->bytes_mtp += t->nbytes;
            break;
        default:
            g->bytes_dense += t->nbytes;
            break;
        }
        if (t->kind != Q4_KIND_MTP && t->layer >= 0 &&
            (uint32_t)t->layer + 1u > g->n_layer)
            g->n_layer = (uint32_t)t->layer + 1u;
        if (strcmp(t->name, "token_embd.weight") == 0 && t->n_dims >= 2)
            g->n_vocab = (uint32_t)t->ne[1];
    }
    if (g->n_expert && g->bytes_routed)
        g->per_expert_bytes = g->bytes_routed / g->n_expert;
    /* per-expert across all layers: bytes_routed / (layers_with_moe * n_expert)
       Using n_expert as last-dim of one tensor undercounts layers. Prefer:
       bytes_routed / unique (layer,expert) — approx bytes_routed / (n_layer * n_expert)
       if every layer is MoE. */
    uint32_t moe_layers = 0;
    {
        bool seen[512];
        memset(seen, 0, sizeof(seen));
        for (uint32_t i = 0; i < g->n_tensors; i++) {
            if (g->tensors[i].kind == Q4_KIND_ROUTED && g->tensors[i].layer >= 0 &&
                (uint32_t)g->tensors[i].layer < 512)
                seen[g->tensors[i].layer] = true;
        }
        for (uint32_t i = 0; i < 512; i++)
            if (seen[i]) moe_layers++;
    }
    if (moe_layers && g->n_expert)
        g->per_expert_bytes = g->bytes_routed / ((uint64_t)moe_layers * g->n_expert);
    (void)routed_experts;
}

bool q4_layer_is_qsa(const q4_gguf *g, int32_t layer) {
    uint32_t iv = g && g->full_attn_interval ? g->full_attn_interval : 4;
    if (layer < 0) return false;
    return ((uint32_t)layer + 1u) % iv == 0;
}

bool q4_gguf_open(q4_gguf *g, const char *path) {
    return q4_gguf_open_ex(g, path, 0);
}

/* keep_mtp_layer=1 for the draft-head GGUF: blk.48 stays a normal layer
 * (ROUTED/SHARED/DENSE kinds) instead of being peeled as the main model's
 * nextn draft block. */
bool q4_gguf_open_ex(q4_gguf *g, const char *path, int keep_mtp_layer) {
    memset(g, 0, sizeof(*g));
    g->ple_layer = UINT32_MAX;
    g->ple_table = -1;
    char shards[Q4_MAX_FILES][4096];
    uint32_t n = discover_shards(path, shards, Q4_MAX_FILES);
    for (uint32_t i = 0; i < n; i++) {
        if (!open_file(&g->files[g->n_files], shards[i])) {
            q4_gguf_close(g);
            return false;
        }
        g->n_files++;
    }
    for (uint32_t i = 0; i < g->n_files; i++) {
        if (!parse_header(g, &g->files[i], i == 0)) {
            fprintf(stderr, "q4: failed to parse %s\n", g->files[i].path);
            q4_gguf_close(g);
            return false;
        }
    }
    fill_nbytes(g);
    summarize(g, keep_mtp_layer);
    /* name -> index hash for O(1) q4_find_tensor (hot in the decode loop) */
    uint32_t cap = 64;
    while (cap < g->n_tensors * 2) cap <<= 1;
    g->name_hash = xmalloc(cap * sizeof(int32_t));
    g->name_hash_cap = cap;
    for (uint32_t i = 0; i < cap; i++) g->name_hash[i] = -1;
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        uint32_t h = fnv1a(g->tensors[i].name) & (cap - 1);
        while (g->name_hash[h] >= 0) h = (h + 1) & (cap - 1);
        g->name_hash[h] = (int32_t)i;
    }
    return true;
}

void q4_gguf_close(q4_gguf *g) {
    if (!g) return;
    for (uint32_t i = 0; i < g->n_files; i++) {
        if (g->files[i].fd >= 0) close(g->files[i].fd);
        g->files[i].fd = -1;
    }
    free(g->tensors);
    free(g->arch);
    free(g->name);
    if (g->tok_vocab) {
        for (uint32_t i = 0; i < g->n_tok_vocab; i++) free(g->tok_vocab[i]);
        free(g->tok_vocab);
    }
    if (g->tok_merges) {
        for (uint32_t i = 0; i < g->n_tok_merges; i++) free(g->tok_merges[i]);
        free(g->tok_merges);
    }
    free(g->chat_template);
    free(g->name_hash);
    memset(g, 0, sizeof(*g));
}

void q4_gguf_print(const q4_gguf *g, FILE *fp) {
    fprintf(fp, "arch          %s\n", g->arch ? g->arch : "?");
    fprintf(fp, "name          %s\n", g->name ? g->name : "?");
    fprintf(fp, "files         %u\n", g->n_files);
    fprintf(fp, "tensors       %u\n", g->n_tensors);
    fprintf(fp, "layers        %u\n", g->n_layer);
    fprintf(fp, "n_embd        %u\n", g->n_embd);
    fprintf(fp, "n_ff          %u\n", g->n_ff);
    fprintf(fp, "n_expert      %u  used %u\n", g->n_expert, g->n_expert_used);
    fprintf(fp, "n_head        %u  kv %u\n", g->n_head, g->n_head_kv);
    fprintf(fp, "n_ff_exp      %u  shexp %u\n", g->n_ff_exp, g->n_ff_shexp);
    fprintf(fp, "n_ctx_train   %u\n", g->n_ctx_train);
    fprintf(fp, "n_vocab       %u\n", g->n_vocab);
    fprintf(fp, "head k/v      %u / %u  rot %u  rope_base %.0f  rms %g\n",
            g->n_embd_head_k, g->n_embd_head_v, g->n_rot, g->rope_freq_base,
            g->rms_eps);
    fprintf(fp, "GDN ssm       conv %u  state %u  groups %u  dt %u  inner %u\n",
            g->ssm_d_conv, g->ssm_d_state, g->ssm_n_group, g->ssm_dt_rank,
            g->ssm_d_inner);
    fprintf(fp, "QSA indexer   heads %u  dim %u  top_k %u  c4 on QSA layers\n",
            g->indexer_n_head, g->indexer_head_size, g->indexer_top_k);
    fprintf(fp, "HC            %u streams  rank %u  (wide %u)\n", g->hc_mult,
            g->hc_rank, g->hc_mult * g->n_embd);
    fprintf(fp, "QSA interval  every %u-th layer\n",
            g->full_attn_interval ? g->full_attn_interval : 4);
    fprintf(fp, "PLE           table %.2f GiB  layer %u  ngram %u  heads %u  row %u\n",
            (double)g->bytes_ple / Q4_GIB,
            g->ple_layer == UINT32_MAX ? 0 : g->ple_layer, g->ple_ngram,
            g->ple_n_heads, g->n_embd_ple);
    fprintf(fp, "MTP           %.2f GiB%s\n", (double)g->bytes_mtp / Q4_GIB,
            g->bytes_mtp == 0 ? "  (not in this GGUF)"
                              : "  (draft block, not in the main forward)");
    fprintf(fp, "total         %.2f GiB\n", (double)g->bytes_total / Q4_GIB);
    fprintf(fp, "  dense       %.2f GiB\n", (double)g->bytes_dense / Q4_GIB);
    fprintf(fp, "  shared      %.2f GiB\n", (double)g->bytes_shared / Q4_GIB);
    fprintf(fp, "  routed      %.2f GiB\n", (double)g->bytes_routed / Q4_GIB);
    fprintf(fp, "  PLE table   %.2f GiB (SSD demand I/O; DRAM is optional cache)\n",
            (double)g->bytes_ple / Q4_GIB);
    fprintf(fp, "per expert    %.2f MiB\n",
            (double)g->per_expert_bytes / (1024.0 * 1024.0));
}

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h;
}

const q4_tensor *q4_find_tensor(const q4_gguf *g, const char *name) {
    if (!g || !name) return NULL;
    if (g->name_hash) {
        uint32_t h = fnv1a(name) & (g->name_hash_cap - 1);
        while (g->name_hash[h] >= 0) {
            const q4_tensor *t = &g->tensors[g->name_hash[h]];
            if (strcmp(t->name, name) == 0) return t;
            h = (h + 1) & (g->name_hash_cap - 1);
        }
        return NULL;
    }
    for (uint32_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

bool q4_tensor_read(const q4_gguf *g, const q4_tensor *t, void *dst, uint64_t n) {
    if (!g || !t || !dst) return false;
    if (n == 0 || n > t->nbytes) n = t->nbytes;
    const q4_file *f = &g->files[t->shard];
    uint8_t *p = dst;
    uint64_t got = 0;
    uint64_t off = f->data_off + t->offset;
    while (got < n) {
        ssize_t r = pread(f->fd, p + got, (size_t)(n - got), (off_t)(off + got));
        if (r == 0) break;
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        got += (uint64_t)r;
    }
    return got == n;
}

bool q4_parse_gib(const char *s, uint64_t *bytes) {
    if (!s || !s[0] || !bytes) return false;
    *bytes = 0;
    size_t len = strlen(s);
    if (len > 2 && (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B'))
        len -= 2;
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++)
        if (!isdigit((unsigned char)s[i])) return false;
    errno = 0;
    unsigned long long v = strtoull(s, NULL, 10);
    if (errno || v == 0 || v > UINT64_MAX / Q4_GIB) return false;
    *bytes = (uint64_t)v * Q4_GIB;
    return true;
}
