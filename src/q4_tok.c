#include "q4.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOK_CAP (1u << 19)

typedef struct {
    uint64_t h;
    int32_t  id;
    uint32_t n;
} tok_ent;

typedef struct {
    uint64_t h;
    uint32_t n;
    uint32_t rank; /* 1 + merge index; 0 = empty */
} merge_ent;

struct q4_tok {
    const q4_gguf *g;
    tok_ent *tab;
    uint32_t cap;
    merge_ent *mtab;
    uint32_t mcap;
    int32_t *specials;
    uint32_t n_specials;
    int32_t  bos, eos, pad, im_start, im_end, think, think_end;
};

static uint32_t b2u[256];
static int u2b[512];
static int maps_ready;

static void maps_init(void) {
    if (maps_ready) return;
    int printable[256];
    memset(printable, 0, sizeof(printable));
    for (int i = 33; i <= 126; i++) printable[i] = 1;
    for (int i = 161; i <= 172; i++) printable[i] = 1;
    for (int i = 174; i <= 255; i++) printable[i] = 1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (printable[b]) b2u[b] = (uint32_t)b;
        else {
            b2u[b] = 256u + (uint32_t)n;
            n++;
        }
    }
    for (int i = 0; i < 512; i++) u2b[i] = -1;
    for (int b = 0; b < 256; b++)
        if (b2u[b] < 512) u2b[b2u[b]] = b;
    maps_ready = 1;
}

static int utf8_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static uint32_t utf8_cp(const char *s, int n) {
    unsigned char c = (unsigned char)s[0];
    if (n == 1) return c;
    if (n == 2) return ((c & 0x1f) << 6) | ((unsigned char)s[1] & 0x3f);
    if (n == 3)
        return ((c & 0x0f) << 12) | (((unsigned char)s[1] & 0x3f) << 6) |
               ((unsigned char)s[2] & 0x3f);
    if (n == 4)
        return ((c & 0x07) << 18) | (((unsigned char)s[1] & 0x3f) << 12) |
               (((unsigned char)s[2] & 0x3f) << 6) |
               ((unsigned char)s[3] & 0x3f);
    return 0xfffd;
}

static int utf8_put(char *d, uint32_t cp) {
    if (cp < 0x80) {
        d[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        d[0] = (char)(0xc0 | (cp >> 6));
        d[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    d[0] = (char)(0xe0 | (cp >> 12));
    d[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
    d[2] = (char)(0x80 | (cp & 0x3f));
    return 3;
}

static uint64_t hash_n(const char *s, uint32_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ull;
    }
    return h ? h : 1;
}

static void tok_put(q4_tok *t, const char *s, int32_t id) {
    uint32_t n = (uint32_t)strlen(s);
    uint64_t h = hash_n(s, n);
    uint32_t i = (uint32_t)(h & (t->cap - 1));
    for (;;) {
        if (t->tab[i].h == 0) {
            t->tab[i].h = h;
            t->tab[i].id = id;
            t->tab[i].n = n;
            return;
        }
        i = (i + 1) & (t->cap - 1);
    }
}

static int32_t tok_get(const q4_tok *t, const char *s, uint32_t n) {
    uint64_t h = hash_n(s, n);
    uint32_t i = (uint32_t)(h & (t->cap - 1));
    for (;;) {
        if (t->tab[i].h == 0) return -1;
        if (t->tab[i].h == h && t->tab[i].n == n &&
            memcmp(t->g->tok_vocab[t->tab[i].id], s, n) == 0)
            return t->tab[i].id;
        i = (i + 1) & (t->cap - 1);
    }
}

static void merge_put(q4_tok *t, const char *s, uint32_t rank) {
    uint32_t n = (uint32_t)strlen(s);
    uint64_t h = hash_n(s, n);
    uint32_t i = (uint32_t)(h & (t->mcap - 1));
    for (;;) {
        if (t->mtab[i].h == 0) {
            t->mtab[i].h = h;
            t->mtab[i].n = n;
            t->mtab[i].rank = rank;
            return;
        }
        i = (i + 1) & (t->mcap - 1);
    }
}

static uint32_t merge_get(const q4_tok *t, const char *s, uint32_t n) {
    uint64_t h = hash_n(s, n);
    uint32_t i = (uint32_t)(h & (t->mcap - 1));
    for (;;) {
        if (t->mtab[i].h == 0) return UINT32_MAX;
        if (t->mtab[i].h == h && t->mtab[i].n == n) {
            uint32_t r = t->mtab[i].rank - 1;
            if (r < t->g->n_tok_merges && t->g->tok_merges[r] &&
                memcmp(t->g->tok_merges[r], s, n) == 0)
                return r;
        }
        i = (i + 1) & (t->mcap - 1);
    }
}

q4_tok *q4_tok_open(const q4_gguf *g) {
    if (!g || !g->tok_vocab || g->n_tok_vocab == 0) return NULL;
    maps_init();
    q4_tok *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->g = g;
    t->cap = TOK_CAP;
    t->tab = calloc(t->cap, sizeof(tok_ent));
    t->mcap = TOK_CAP;
    t->mtab = calloc(t->mcap, sizeof(merge_ent));
    if (!t->tab || !t->mtab) {
        q4_tok_close(t);
        return NULL;
    }
    t->specials = calloc(g->n_tok_vocab, sizeof(int32_t));
    for (uint32_t i = 0; i < g->n_tok_vocab; i++) {
        const char *s = g->tok_vocab[i];
        if (!s) continue;
        tok_put(t, s, (int32_t)i);
        if ((s[0] == '<' && s[1] == '|') || !strcmp(s, "<think>") ||
            !strcmp(s, "</think>") || !strcmp(s, "<tool_call>") ||
            !strcmp(s, "</tool_call>") || !strcmp(s, "<tool_response>") ||
            !strcmp(s, "</tool_response>"))
            t->specials[t->n_specials++] = (int32_t)i;
    }
    for (uint32_t i = 0; i < g->n_tok_merges; i++)
        if (g->tok_merges[i]) merge_put(t, g->tok_merges[i], i + 1);
    t->bos = (int32_t)g->bos_id;
    t->eos = (int32_t)g->eos_id;
    t->pad = (int32_t)g->pad_id;
    t->im_start = tok_get(t, "<|im_start|>", 12);
    t->im_end = tok_get(t, "<|im_end|>", 10);
    t->think = tok_get(t, "<think>", 7);
    t->think_end = tok_get(t, "</think>", 8);
    return t;
}

void q4_tok_close(q4_tok *t) {
    if (!t) return;
    free(t->tab);
    free(t->mtab);
    free(t->specials);
    free(t);
}

int32_t q4_tok_id(const q4_tok *t, const char *s) {
    return t && s ? tok_get(t, s, (uint32_t)strlen(s)) : -1;
}

int32_t q4_tok_eos(const q4_tok *t) {
    return t ? t->eos : 0;
}

int32_t q4_tok_im_end(const q4_tok *t) {
    return t ? t->im_end : -1;
}

typedef struct {
    uint32_t off, len;
} bpe_span;

static int special_at(const q4_tok *t, const char *p, size_t rem) {
    int best_n = 0, best_id = -1;
    for (uint32_t i = 0; i < t->n_specials; i++) {
        int32_t id = t->specials[i];
        const char *s = t->g->tok_vocab[id];
        uint32_t n = (uint32_t)strlen(s);
        if (n > rem || n < 3) continue;
        if (memcmp(p, s, n) == 0 && (int)n > best_n) {
            best_n = (int)n;
            best_id = id;
        }
    }
    return best_id;
}

/* Merge candidates: (rank, left symbol's byte offset) orders them like a
 * rescan of every adjacent pair would (lowest rank, then leftmost); la / lb
 * are the two symbols' lengths when queued — a merge that touched either
 * since then makes the entry stale. */
typedef struct {
    uint32_t rank, off, la, lb;
    int32_t i;
} bpe_cand;

static int bpe_cand_lt(const bpe_cand *a, const bpe_cand *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->off < b->off;
}

static void bpe_push(bpe_cand *h, int *hn, bpe_cand c) {
    int i = (*hn)++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!bpe_cand_lt(&c, &h[p])) break;
        h[i] = h[p];
        i = p;
    }
    h[i] = c;
}

static bpe_cand bpe_pop(bpe_cand *h, int *hn) {
    bpe_cand top = h[0], last = h[--(*hn)];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= *hn) break;
        if (c + 1 < *hn && bpe_cand_lt(&h[c + 1], &h[c])) c++;
        if (!bpe_cand_lt(&h[c], &last)) break;
        h[i] = h[c];
        i = c;
    }
    if (*hn) h[i] = last;
    return top;
}

/* Queue the pair (a, b = nx[a]) when the merge table has it. */
static void bpe_queue(const q4_tok *t, const char *mapped, const bpe_span *sp,
                      int a, int b, bpe_cand *h, int *hn) {
    char key[512];
    const uint32_t la = sp[a].len, lb = sp[b].len;
    if (la + 1 + lb >= sizeof(key)) return;
    memcpy(key, mapped + sp[a].off, la);
    key[la] = ' ';
    memcpy(key + la + 1, mapped + sp[b].off, lb);
    const uint32_t r = merge_get(t, key, la + 1 + lb);
    if (r == UINT32_MAX) return;
    bpe_push(h, hn, (bpe_cand){r, sp[a].off, la, lb, a});
}

/* BPE over one pretokenized piece.  It used to rescan every adjacent pair
 * after each merge — O(n^2) lookups per piece: a 16K-character CJK run with
 * no punctuation took 12.6 s, 64K letters 20.6 s, a 20K '=' line 2.4 s.
 * Symbols are now a linked list and candidate pairs a heap. */
static int bpe_chunk(const q4_tok *t, const char *mapped, int mlen, int32_t *out,
                     int max_out) {
    if (mlen <= 0 || max_out <= 0) return 0;
    int nsp = 0;
    for (int i = 0; i < mlen;) {
        nsp++;
        i += utf8_len((unsigned char)mapped[i]);
    }
    bpe_span *sp = malloc((size_t)nsp * sizeof(*sp));
    int32_t *nx = malloc((size_t)nsp * sizeof(int32_t));
    int32_t *pv = malloc((size_t)nsp * sizeof(int32_t));
    /* n - 1 initial pairs, at most two new ones per merge */
    bpe_cand *h = malloc((size_t)(3 * nsp + 1) * sizeof(*h));
    if (!sp || !nx || !pv || !h) {
        free(sp); free(nx); free(pv); free(h);
        return 0;
    }
    int n = 0;
    for (int i = 0; i < mlen;) {
        int L = utf8_len((unsigned char)mapped[i]);
        sp[n].off = (uint32_t)i;
        sp[n].len = (uint32_t)L;
        pv[n] = n - 1;
        nx[n] = n + 1 < nsp ? n + 1 : -1;
        n++;
        i += L;
    }
    int hn = 0;
    for (int i = 0; i + 1 < n; i++) bpe_queue(t, mapped, sp, i, i + 1, h, &hn);
    while (hn) {
        const bpe_cand c = bpe_pop(h, &hn);
        const int a = c.i, b = nx[a];
        /* a absorbed by its left neighbour: len 0; a grew: len changed */
        if (b < 0 || sp[a].len != c.la || sp[b].len != c.lb) continue;
        sp[a].len += sp[b].len;
        sp[b].len = 0;
        nx[a] = nx[b];
        if (nx[b] >= 0) pv[nx[b]] = a;
        if (pv[a] >= 0) bpe_queue(t, mapped, sp, pv[a], a, h, &hn);
        if (nx[a] >= 0) bpe_queue(t, mapped, sp, a, nx[a], h, &hn);
    }
    int nout = 0;
    for (int i = 0; i >= 0 && nout < max_out; i = nx[i]) {
        int32_t id = tok_get(t, mapped + sp[i].off, sp[i].len);
        if (id < 0) continue;
        out[nout++] = id;
    }
    free(sp); free(nx); free(pv); free(h);
    return nout;
}

/* Qwen2 / GPT-2 pretokenize (tokenizer_config.pretokenize_regex).
 * BPE is run per piece. Skipping this lets merges cross word boundaries and
 * splits digits as one blob — the model then sees a different token stream
 * than the GGUF tokenizer this model ships with. */
#include "q4_ucat.inc"

static int ucat_in(const uint32_t (*r)[2], int n, uint32_t c) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (c < r[m][0]) hi = m - 1;
        else if (c > r[m][1]) lo = m + 1;
        else return 1;
    }
    return 0;
}
#define UCAT(t, c) ucat_in(t, (int)(sizeof(t) / sizeof(t[0])), c)

static int u_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    return UCAT(q4_ucat_L, c);
}

static int u_mark(uint32_t c) { return c >= 0x300 && UCAT(q4_ucat_M, c); }

static int u_number(uint32_t c) {
    if (c < 0x80) return c >= '0' && c <= '9';
    return UCAT(q4_ucat_N, c);
}

static int u_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B ||
           c == 0x0C || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static int u_nl(uint32_t c) {
    return c == '\n' || c == '\r';
}

static uint32_t peek_cp(const char *p, size_t rem, int *len) {
    if (rem == 0) {
        *len = 0;
        return 0;
    }
    int L = utf8_len((unsigned char)p[0]);
    if (L > (int)rem) L = 1;
    *len = L;
    return utf8_cp(p, L);
}

static int ci_eq(char a, char b) {
    if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
    if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
    return a == b;
}

static int qwen2_piece(const char *p, size_t rem) {
    if (rem == 0) return 0;
    /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (rem >= 2 && p[0] == '\'') {
        if (ci_eq(p[1], 's') || ci_eq(p[1], 't') || ci_eq(p[1], 'm') ||
            ci_eq(p[1], 'd'))
            return 2;
        if (rem >= 3 && ((ci_eq(p[1], 'r') && ci_eq(p[2], 'e')) ||
                         (ci_eq(p[1], 'v') && ci_eq(p[2], 'e')) ||
                         (ci_eq(p[1], 'l') && ci_eq(p[2], 'l'))))
            return 3;
    }
    int l0 = 0;
    uint32_t c0 = peek_cp(p, rem, &l0);
    /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
    {
        int off = 0;
        if (l0 && !u_nl(c0) && !u_letter(c0) && !u_number(c0)) off = l0;
        int body = 0;
        while (off + body < (int)rem) {
            int l = 0;
            uint32_t c = peek_cp(p + off + body, rem - (size_t)(off + body), &l);
            if (!l || (!u_letter(c) && !u_mark(c))) break;
            body += l;
        }
        if (body > 0) return off + body;
    }
    /* \p{N} */
    if (u_number(c0)) return l0;
    /*  ?[^\s\p{L}\p{N}\p{M}]+[\r\n]* */
    {
        int off = 0;
        if (c0 == ' ') off = 1;
        int body = 0;
        while (off + body < (int)rem) {
            int l = 0;
            uint32_t c = peek_cp(p + off + body, rem - (size_t)(off + body), &l);
            if (!l || u_space(c) || u_letter(c) || u_number(c) || u_mark(c))
                break;
            body += l;
        }
        if (body > 0) {
            int extra = 0;
            while (off + body + extra < (int)rem) {
                int l = 0;
                uint32_t c =
                    peek_cp(p + off + body + extra,
                            rem - (size_t)(off + body + extra), &l);
                if (!u_nl(c)) break;
                extra += l;
            }
            return off + body + extra;
        }
    }
    /* The remaining alternatives start in a whitespace run [0, n). */
    if (u_space(c0)) {
        int n = 0, last_nl = -1, last_cp = 0;
        while (n < (int)rem) {
            int l = 0;
            uint32_t c = peek_cp(p + n, rem - (size_t)n, &l);
            if (!l || !u_space(c)) break;
            if (u_nl(c)) last_nl = n + l;
            last_cp = n;
            n += l;
        }
        /* \s*[\r\n]+ : the greedy \s* backtracks to the run's last
         * newline, so the piece ends right after it. */
        if (last_nl > 0) return last_nl;
        /* \s+(?!\S) : a run followed by a non-space gives its last space
         * to the next piece (" word"); a one-space run falls to \s+. */
        if (n < (int)rem && last_cp > 0) return last_cp;
        return n > 0 ? n : l0;
    }
    return l0 > 0 ? l0 : 1;
}

int q4_tok_encode(const q4_tok *t, const char *utf8, int32_t *out, int max_out) {
    if (!t || !utf8 || !out || max_out <= 0) return 0;
    maps_init();
    int nout = 0;
    const char *p = utf8;
    size_t n = strlen(utf8);
    char *mapped = malloc(n * 3 + 8);
    if (!mapped) return 0;
    /* Special tokens split the text before pretokenization (as HF's added
     * tokens do): a piece never runs into one.  The punctuation rule took
     * "." + "<|" as one piece, so "...answer.<|vision_start|>" lost its
     * special token. */
    const char *seg_end = NULL;
    while (*p && nout < max_out) {
        size_t rem = n - (size_t)(p - utf8);
        int32_t sp = special_at(t, p, rem);
        if (sp >= 0) {
            out[nout++] = sp;
            p += strlen(t->g->tok_vocab[sp]);
            continue;
        }
        if (!seg_end || seg_end <= p) {
            seg_end = utf8 + n;
            for (const char *q = strchr(p + 1, '<'); q; q = strchr(q + 1, '<'))
                if (special_at(t, q, n - (size_t)(q - utf8)) >= 0) {
                    seg_end = q;
                    break;
                }
        }
        rem = (size_t)(seg_end - p);
        int plen = qwen2_piece(p, rem);
        if (plen <= 0) plen = 1;
        if ((size_t)plen > rem) plen = (int)rem;
        int mo = 0;
        for (int i = 0; i < plen; i++) {
            uint32_t cp = b2u[(unsigned char)p[i]];
            mo += utf8_put(mapped + mo, cp);
        }
        nout += bpe_chunk(t, mapped, mo, out + nout, max_out - nout);
        p += plen;
    }
    free(mapped);
    return nout;
}

int q4_utf8_safe_end(const char *s, int n) {
    if (!s || n <= 0) return 0;
    int i = n;
    int cont = 0;
    while (i > 0 && cont < 3 && ((unsigned char)s[i - 1] & 0xc0) == 0x80) {
        i--;
        cont++;
    }
    if (i == n) {
        unsigned char c = (unsigned char)s[n - 1];
        if (c < 0x80) return n;
        return n - 1;
    }
    if (i == 0) return 0;
    unsigned char c = (unsigned char)s[i - 1];
    int L = 1;
    if ((c & 0xe0) == 0xc0) L = 2;
    else if ((c & 0xf0) == 0xe0) L = 3;
    else if ((c & 0xf8) == 0xf0) L = 4;
    else return i - 1;
    if ((i - 1) + L > n) return i - 1;
    return n;
}

int q4_tok_decode(const q4_tok *t, const int32_t *ids, int n, char *out, int max_out) {
    if (!t || !out || max_out <= 0) return 0;
    maps_init();
    int o = 0;
    out[0] = 0;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || (uint32_t)id >= t->g->n_tok_vocab) continue;
        const char *s = t->g->tok_vocab[id];
        if (!s) continue;
        if (s[0] == '<' && s[1] == '|') continue; /* skip specials in text */
        int k = 0;
        while (s[k] && o + 4 < max_out) {
            int L = utf8_len((unsigned char)s[k]);
            uint32_t cp = utf8_cp(s + k, L);
            int b = (cp < 512) ? u2b[cp] : -1;
            if (b >= 0) out[o++] = (char)b;
            else {
                memcpy(out + o, s + k, (size_t)L);
                o += L;
            }
            k += L;
        }
    }
    out[o] = 0;
    return o;
}

static int emit_str(const q4_tok *t, const char *s, int32_t *out, int max_out, int n) {
    int k = q4_tok_encode(t, s, out + n, max_out - n);
    return n + k;
}

/* This request's chat_template kwargs (q4_tok_set_request): think -1 =
 * the server default (Q4_THINK), effort NULL = the template's default. */
static int g_think = -1;
static const char *g_effort;

void q4_tok_set_request(int think, const char *effort) {
    g_think = think;
    g_effort = effort;
}

int q4_tok_think_enabled(void) {
    if (g_think >= 0) return g_think;
    const char *e = getenv("Q4_THINK");
    return e && e[0] && e[0] != '0';
}

static int no_think_mode(void) {
    return !q4_tok_think_enabled();
}

static int text_has_think_block(const char *text) {
    return text && (strstr(text, "<think>") || strstr(text, "</think>"));
}

static int emit_history_assistant(const q4_tok *t, const char *text, int32_t *out,
                                  int max_out, int n) {
    if (n < max_out) out[n++] = t->im_start;
    n = emit_str(t, "assistant\n", out, max_out, n);
    if (text_has_think_block(text)) {
        n = emit_str(t, text, out, max_out, n);
    } else if (t->think >= 0 && t->think_end >= 0) {
        /* Close think so stored content is not treated as reasoning.
         * OpenCode (and Q4_SHOW_THINKING=0) replay only the answer. */
        if (n < max_out) out[n++] = t->think;
        n = emit_str(t, "\n\n", out, max_out, n);
        if (n < max_out) out[n++] = t->think_end;
        n = emit_str(t, "\n\n", out, max_out, n);
        n = emit_str(t, text ? text : "", out, max_out, n);
    } else {
        n = emit_str(t, text ? text : "", out, max_out, n);
    }
    if (n < max_out) out[n++] = t->im_end;
    return emit_str(t, "\n", out, max_out, n);
}

static int emit_assistant_open(const q4_tok *t, int32_t *out, int max_out, int n) {
    if (n < max_out) out[n++] = t->im_start;
    n = emit_str(t, "assistant\n", out, max_out, n);
    /* Qwen3.8 template: empty think block disables reasoning. */
    if (no_think_mode() && t->think >= 0 && t->think_end >= 0) {
        if (n < max_out) out[n++] = t->think;
        n = emit_str(t, "\n\n", out, max_out, n);
        if (n < max_out) out[n++] = t->think_end;
        n = emit_str(t, "\n\n", out, max_out, n);
    } else if (t->think >= 0) {
        if (n < max_out) out[n++] = t->think;
        n = emit_str(t, "\n", out, max_out, n);
    }
    return n;
}

int q4_tok_apply_chat(const q4_tok *t, const char *system, const char *user,
                      int32_t *out, int max_out) {
    const char *roles[2] = {"system", "user"};
    const char *texts[2] = {system, user ? user : ""};
    int sys = system && system[0];
    return q4_tok_apply_messages(t, roles + !sys, texts + !sys, 2 - !sys, out,
                                 max_out);
}

/* The template's reasoning instruction (chat_template.jinja): only while
 * thinking is on; reasoning_effort defaults to xhigh (Q4_EFFORT sets the
 * server default), high means xhigh, medium has no sentence. */
static const char *effort_text(void) {
    if (!q4_tok_think_enabled()) return NULL;
    const char *e = g_effort ? g_effort : getenv("Q4_EFFORT");
    if (!e || !e[0] || !strcmp(e, "xhigh") || !strcmp(e, "high"))
        return "Reasoning effort is set to xhigh. Please think carefully "
               "through the task, validate key assumptions, consider "
               "plausible alternatives, and prioritize correctness, "
               "consistency, and clarity in the final answer.";
    if (!strcmp(e, "low"))
        return "Reasoning effort is set to low. Keep your thinking brief and "
               "focused, moving directly to the conclusion without "
               "unnecessary elaboration.";
    return NULL;
}

/* Python str.strip() bounds of s (the template's |trim): ASCII and
 * Unicode whitespace (\x1c-\x1f, U+0085, U+00A0, U+1680, U+2000-200A,
 * U+2028/2029, U+202F, U+205F, U+3000) off both ends. */
static int py_space_at(const unsigned char *p, size_t n, size_t *len) {
    if (!n) return 0;
    const unsigned c = p[0];
    if (c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f)) {
        *len = 1;
        return 1;
    }
    if (c == 0xc2 && n >= 2 && (p[1] == 0x85 || p[1] == 0xa0)) {
        *len = 2;
        return 1;
    }
    if (n >= 3) {
        const unsigned cp = ((c & 0x0fu) << 12) | ((p[1] & 0x3fu) << 6) |
                            (p[2] & 0x3fu);
        if ((c & 0xf0) == 0xe0 &&
            (cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
             cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000)) {
            *len = 3;
            return 1;
        }
    }
    return 0;
}
static size_t py_space_before(const unsigned char *s, size_t n) {
    /* length of a whitespace char ending at s[n), 0 if none */
    for (size_t L = 1; L <= 3 && L <= n; L++) {
        size_t got = 0;
        if (py_space_at(s + n - L, L, &got) && got == L) return L;
    }
    return 0;
}
static char *py_strip_dup(const char *s) {
    const unsigned char *b = (const unsigned char *)(s ? s : "");
    size_t n = strlen((const char *)b), L = 0;
    while (n && py_space_at(b, n, &L)) b += L, n -= L;
    while (n && (L = py_space_before(b, n))) n -= L;
    char *o = malloc(n + 1);
    if (!o) return NULL;
    memcpy(o, b, n);
    o[n] = 0;
    return o;
}

static int is_system_role(const char *r) {
    return r && (!strcmp(r, "system") || !strcmp(r, "developer"));
}

/* The template's leading system turn: the effort sentence, then the
 * leading system/developer messages trimmed and joined by newlines.
 * Returns how many messages it consumed. */
static int emit_system_turn(const q4_tok *t, const char *const *roles,
                            const char *const *contents, int nmsg,
                            int32_t *out, int max_out, int *n) {
    const char *instr = effort_text();
    int ns = 0, any = 0;
    while (ns < nmsg && roles && is_system_role(roles[ns])) ns++;
    for (int i = 0; i < ns; i++) {
        const char *s = contents && contents[i] ? contents[i] : "";
        size_t L = strlen(s);
        while (L && isspace((unsigned char)*s)) s++, L--;
        while (L && isspace((unsigned char)s[L - 1])) L--;
        if (!L) continue;
        if (!any) {
            if (*n < max_out) out[(*n)++] = t->im_start;
            *n = emit_str(t, "system\n", out, max_out, *n);
            if (instr) *n = emit_str(t, instr, out, max_out, *n);
            if (instr) *n = emit_str(t, "\n\n", out, max_out, *n);
        } else {
            *n = emit_str(t, "\n", out, max_out, *n);
        }
        char *c = malloc(L + 1);
        if (c) {
            memcpy(c, s, L);
            c[L] = 0;
            *n = emit_str(t, c, out, max_out, *n);
            free(c);
        }
        any = 1;
    }
    if (!any && instr) {
        if (*n < max_out) out[(*n)++] = t->im_start;
        *n = emit_str(t, "system\n", out, max_out, *n);
        *n = emit_str(t, instr, out, max_out, *n);
        any = 1;
    }
    if (any) {
        if (*n < max_out) out[(*n)++] = t->im_end;
        *n = emit_str(t, "\n", out, max_out, *n);
    }
    return ns;
}

int q4_tok_apply_messages_from(const q4_tok *t, const char *const *roles,
                               const char *const *contents, int nmsg, int from,
                               int32_t *out, int max_out, int n) {
    if (!t || !out || max_out <= 0 || nmsg < 0) return n;
    if (from < 0) from = 0;
    if (from > nmsg) from = nmsg;
    if (t->im_start < 0 || t->im_end < 0) {
        for (int i = from; i < nmsg; i++)
            n = emit_str(t, contents && contents[i] ? contents[i] : "", out,
                         max_out, n);
        return n;
    }
    int last_assistant = 0;
    if (from > 0 && roles && roles[from - 1] &&
        strcmp(roles[from - 1], "assistant") == 0)
        last_assistant = 1;
    if (from == 0)
        from = emit_system_turn(t, roles, contents, nmsg, out, max_out, &n);
    for (int i = from; i < nmsg; i++) {
        const char *role = roles && roles[i] && roles[i][0] ? roles[i] : "user";
        /* the template renders every non-system message |trim */
        char *trimmed = py_strip_dup(contents && contents[i] ? contents[i] : "");
        const char *text = trimmed ? trimmed
                                   : (contents && contents[i] ? contents[i] : "");
        if (strcmp(role, "developer") == 0) role = "system";
        int last = (i == nmsg - 1);
        if (strcmp(role, "tool") == 0) {
            int first = !(i > 0 && roles[i - 1] &&
                          strcmp(roles[i - 1], "tool") == 0);
            int last_run = last || !roles[i + 1] ||
                           strcmp(roles[i + 1], "tool") != 0;
            if (first) {
                if (n < max_out) out[n++] = t->im_start;
                n = emit_str(t, "user", out, max_out, n);
            }
            n = emit_str(t, "\n<tool_response>\n", out, max_out, n);
            n = emit_str(t, text, out, max_out, n);
            n = emit_str(t, "\n</tool_response>", out, max_out, n);
            if (last_run) {
                if (n < max_out) out[n++] = t->im_end;
                n = emit_str(t, "\n", out, max_out, n);
            }
            last_assistant = 0;
            free(trimmed);
            continue;
        }
        if (last && strcmp(role, "assistant") == 0) {
            free(trimmed);
            return emit_assistant_open(t, out, max_out, n);
        }
        if (!last && strcmp(role, "assistant") == 0) {
            /* History must use a *closed* think block. Generation leaves
             * <think> open; replaying the answer inside that open block
             * made OpenCode show reasoning as the message body next turn. */
            n = emit_history_assistant(t, text, out, max_out, n);
            last_assistant = 1;
            free(trimmed);
            continue;
        }
        if (n < max_out) out[n++] = t->im_start;
        n = emit_str(t, role, out, max_out, n);
        n = emit_str(t, "\n", out, max_out, n);
        n = emit_str(t, text, out, max_out, n);
        if (n < max_out) out[n++] = t->im_end;
        n = emit_str(t, "\n", out, max_out, n);
        last_assistant = strcmp(role, "assistant") == 0;
        free(trimmed);
    }
    if (!last_assistant) n = emit_assistant_open(t, out, max_out, n);
    return n;
}

int q4_tok_apply_messages(const q4_tok *t, const char *const *roles,
                          const char *const *contents, int nmsg, int32_t *out,
                          int max_out) {
    return q4_tok_apply_messages_from(t, roles, contents, nmsg, 0, out, max_out,
                                      0);
}

static char *dup_n(const char *s, size_t n) {
    char *o = malloc(n + 1);
    if (!o) return NULL;
    memcpy(o, s, n);
    o[n] = 0;
    return o;
}

static char *trim_dup(const char *s, size_t n) {
    while (n && isspace((unsigned char)*s)) {
        s++;
        n--;
    }
    while (n && isspace((unsigned char)s[n - 1])) n--;
    return dup_n(s, n);
}

static char *json_escape_dup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *o = malloc(n * 6 + 3);
    if (!o) return NULL;
    size_t j = 0;
    o[j++] = '"';
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            o[j++] = '\\';
            o[j++] = (char)c;
        } else if (c == '\n') {
            o[j++] = '\\';
            o[j++] = 'n';
        } else if (c == '\r') {
            o[j++] = '\\';
            o[j++] = 'r';
        } else if (c == '\t') {
            o[j++] = '\\';
            o[j++] = 't';
        } else if (c < 0x20) {
            j += (size_t)sprintf(o + j, "\\u%04x", c);
        } else {
            o[j++] = (char)c;
        }
    }
    o[j++] = '"';
    o[j] = 0;
    return o;
}

static int looks_json_atom(const char *s) {
    if (!s || !s[0]) return 0;
    if (s[0] == '{' || s[0] == '[' || s[0] == '"') return 1;
    if (!strcmp(s, "true") || !strcmp(s, "false") || !strcmp(s, "null")) return 1;
    char *end = NULL;
    strtod(s, &end);
    return end && end != s && *end == 0;
}

static char *args_from_params(const char *body) {
    size_t cap = 256, n = 1;
    char *o = malloc(cap);
    if (!o) return NULL;
    o[0] = '{';
    o[1] = 0;
    int first = 1;
    const char *p = body;
    while ((p = strstr(p, "<parameter="))) {
        p += 11;
        const char *gt = strchr(p, '>');
        if (!gt) break;
        char *key = trim_dup(p, (size_t)(gt - p));
        p = gt + 1;
        if (*p == '\n') p++;
        /* Prefer a tag on its own line so file contents that happen to
         * mention </parameter> are less likely to truncate the value. */
        const char *end = strstr(p, "\n</parameter>");
        if (end) end++;
        else end = strstr(p, "</parameter>");
        if (!end) {
            free(key);
            break;
        }
        const char *vend = end;
        if (vend > p && vend[-1] == '\n') vend--;
        char *val = dup_n(p, (size_t)(vend - p));
        if (key && val && (!strcmp(key, "path") || !strcmp(key, "filePath") ||
                           !strcmp(key, "file_path") || !strcmp(key, "command"))) {
            char *t = trim_dup(val, strlen(val));
            free(val);
            val = t;
        }
        p = end + 12;
        char *kesc = json_escape_dup(key ? key : "");
        const char *vjson = NULL;
        char *vesc = NULL;
        if (looks_json_atom(val)) vjson = val;
        else {
            vesc = json_escape_dup(val ? val : "");
            vjson = vesc;
        }
        size_t add = strlen(kesc) + strlen(vjson) + 4;
        if (n + add + 2 >= cap) {
            cap = (n + add + 256) * 2;
            char *q = realloc(o, cap);
            if (!q) {
                free(key); free(val); free(kesc); free(vesc); free(o);
                return NULL;
            }
            o = q;
        }
        n += (size_t)sprintf(o + n, "%s%s:%s", first ? "" : ",", kesc, vjson);
        first = 0;
        free(key); free(val); free(kesc); free(vesc);
    }
    o[n++] = '}';
    o[n] = 0;
    return o;
}

static int parse_hermes_obj(const char *body, char **name, char **args) {
    const char *p = body;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '{') return 0;
    const char *nm = strstr(p, "\"name\"");
    if (!nm) return 0;
    nm = strchr(nm + 6, ':');
    if (!nm) return 0;
    nm++;
    while (*nm && isspace((unsigned char)*nm)) nm++;
    if (*nm != '"') return 0;
    nm++;
    const char *ne = nm;
    while (*ne && *ne != '"') {
        if (*ne == '\\' && ne[1]) ne += 2;
        else ne++;
    }
    *name = dup_n(nm, (size_t)(ne - nm));
    const char *ap = strstr(p, "\"arguments\"");
    if (!ap) {
        *args = strdup("{}");
        return 1;
    }
    ap = strchr(ap + 11, ':');
    if (!ap) {
        *args = strdup("{}");
        return 1;
    }
    ap++;
    while (*ap && isspace((unsigned char)*ap)) ap++;
    if (*ap == '"') {
        /* JSON string containing object */
        ap++;
        size_t cap = 64, n = 0;
        char *o = malloc(cap);
        while (*ap && *ap != '"') {
            char c = *ap++;
            if (c == '\\' && *ap) {
                char e = *ap++;
                if (e == 'n') c = '\n';
                else if (e == 't') c = '\t';
                else if (e == 'r') c = '\r';
                else c = e;
            }
            if (n + 2 >= cap) {
                cap *= 2;
                o = realloc(o, cap);
            }
            o[n++] = c;
        }
        o[n] = 0;
        *args = o;
        return 1;
    }
    if (*ap == '{' || *ap == '[') {
        const char *s = ap;
        int d = 0, ins = 0, esc = 0;
        do {
            if (ins) {
                if (esc) esc = 0;
                else if (*ap == '\\') esc = 1;
                else if (*ap == '"') ins = 0;
            } else if (*ap == '"') ins = 1;
            else if (*ap == '{' || *ap == '[') d++;
            else if (*ap == '}' || *ap == ']') d--;
            ap++;
        } while (*ap && d > 0);
        *args = dup_n(s, (size_t)(ap - s));
        return 1;
    }
    *args = strdup("{}");
    return 1;
}

static int parse_one_call(const char *block, char **name, char **args) {
    const char *p = block;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == '{') return parse_hermes_obj(p, name, args);
    const char *fn = strstr(p, "<function=");
    if (!fn) return 0;
    fn += 10;
    const char *gt = strchr(fn, '>');
    if (!gt) return 0;
    *name = dup_n(fn, (size_t)(gt - fn));
    const char *fn_end = strstr(gt, "</function>");
    size_t body_n = fn_end ? (size_t)(fn_end - (gt + 1)) : strlen(gt + 1);
    char *body = dup_n(gt + 1, body_n);
    *args = args_from_params(body);
    free(body);
    if (!*args) *args = strdup("{}");
    return *name != NULL;
}

int q4_qwen_parse_tool_calls(const char *text, char **prefix, char ***names,
                             char ***args_json, int *ncalls) {
    if (prefix) *prefix = NULL;
    if (names) *names = NULL;
    if (args_json) *args_json = NULL;
    if (ncalls) *ncalls = 0;
    if (!text) return 0;
    const char *first = strstr(text, "<tool_call>");
    if (!first) {
        if (prefix) *prefix = strdup(text);
        return 0;
    }
    if (prefix) *prefix = trim_dup(text, (size_t)(first - text));
    int cap = 4, n = 0;
    char **nm = calloc((size_t)cap, sizeof(char *));
    char **ag = calloc((size_t)cap, sizeof(char *));
    const char *p = first;
    while ((p = strstr(p, "<tool_call>"))) {
        p += 11;
        if (*p == '\n') p++;
        const char *end = strstr(p, "</tool_call>");
        char *block = end ? dup_n(p, (size_t)(end - p)) : strdup(p);
        char *name = NULL, *args = NULL;
        if (parse_one_call(block, &name, &args) && name) {
            if (n == cap) {
                cap *= 2;
                nm = realloc(nm, (size_t)cap * sizeof(char *));
                ag = realloc(ag, (size_t)cap * sizeof(char *));
            }
            nm[n] = name;
            ag[n] = args ? args : strdup("{}");
            n++;
        } else {
            free(name);
            free(args);
        }
        free(block);
        if (!end) break;
        p = end + 12;
    }
    if (names) *names = nm;
    else {
        for (int i = 0; i < n; i++) free(nm[i]);
        free(nm);
    }
    if (args_json) *args_json = ag;
    else {
        for (int i = 0; i < n; i++) free(ag[i]);
        free(ag);
    }
    if (ncalls) *ncalls = n;
    return n;
}

void q4_qwen_free_tool_calls(char **names, char **args_json, int n) {
    for (int i = 0; i < n; i++) {
        if (names) free(names[i]);
        if (args_json) free(args_json[i]);
    }
    free(names);
    free(args_json);
}

static const char *skip_json_item(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '{' && *p != '[') return p;
    int d = 0, ins = 0, esc = 0;
    do {
        if (ins) {
            if (esc) esc = 0;
            else if (*p == '\\') esc = 1;
            else if (*p == '"') ins = 0;
        } else if (*p == '"') ins = 1;
        else if (*p == '{' || *p == '[') d++;
        else if (*p == '}' || *p == ']') d--;
        p++;
    } while (*p && d > 0);
    return p;
}

static const char *skip_json_value(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p += 2;
            else p++;
        }
        if (*p == '"') p++;
        return p;
    }
    if (*p == '{' || *p == '[') return skip_json_item(p);
    while (*p && *p != ',' && *p != '}' && *p != ']' && !isspace((unsigned char)*p))
        p++;
    return p;
}

typedef struct {
    char *p;
    size_t n, cap;
} jbuf;

static int jb_add(jbuf *b, const char *s, size_t L) {
    if (b->n + L + 1 >= b->cap) {
        size_t nc = (b->n + L + 256) * 2;
        char *q = realloc(b->p, nc);
        if (!q) return -1;
        b->p = q;
        b->cap = nc;
    }
    memcpy(b->p + b->n, s, L);
    b->n += L;
    b->p[b->n] = 0;
    return 0;
}

static int jb_ch(jbuf *b, char c) {
    return jb_add(b, &c, 1);
}

static int drop_schema_key(const char *k, size_t n) {
    static const char *d[] = {
        "description", "title", "examples", "example", "default",
        "$schema", "$id", "$comment", "markdownDescription",
        "deprecated", "readOnly", "writeOnly",
        "minLength", "maxLength", "pattern", "format",
        "minItems", "maxItems", "uniqueItems",
        "minProperties", "maxProperties",
        "exclusiveMinimum", "exclusiveMaximum",
        "contentMediaType", "contentEncoding",
        "icon", "annotations", "meta",
    };
    for (size_t i = 0; i < sizeof(d) / sizeof(d[0]); i++)
        if (n == strlen(d[i]) && !memcmp(k, d[i], n)) return 1;
    return 0;
}

/* Compact a JSON value: minify and drop verbose schema keys.
 * OpenCode / MCP tool dumps are mostly description text. */
static int json_compact(const char **pp, jbuf *b) {
    const char *p = *pp;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == '"') {
        const char *s = p;
        p = skip_json_value(p);
        if (jb_add(b, s, (size_t)(p - s)) < 0) return -1;
        *pp = p;
        return 0;
    }
    if (*p != '{' && *p != '[') {
        const char *s = p;
        p = skip_json_value(p);
        if (p == s) return -1;
        if (jb_add(b, s, (size_t)(p - s)) < 0) return -1;
        *pp = p;
        return 0;
    }
    int is_arr = *p == '[';
    p++;
    if (jb_ch(b, is_arr ? '[' : '{') < 0) return -1;
    int first = 1;
    while (*p) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if ((is_arr && *p == ']') || (!is_arr && *p == '}')) {
            p++;
            break;
        }
        if (!is_arr) {
            if (*p != '"') return -1;
            const char *keytok = p;
            p++;
            const char *ks = p;
            while (*p && *p != '"') {
                if (*p == '\\' && p[1]) p += 2;
                else p++;
            }
            size_t kn = (size_t)(p - ks);
            if (*p == '"') p++;
            const char *key_end = p;
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p != ':') return -1;
            p++;
            while (*p && isspace((unsigned char)*p)) p++;
            int drop = drop_schema_key(ks, kn);
            if (!drop && kn == 20 && !memcmp(ks, "additionalProperties", 20) &&
                *p == '{')
                drop = 1;
            if (drop) {
                p = skip_json_value(p);
                continue;
            }
            if (!first && jb_ch(b, ',') < 0) return -1;
            first = 0;
            if (jb_add(b, keytok, (size_t)(key_end - keytok)) < 0) return -1;
            if (jb_ch(b, ':') < 0) return -1;
            if (json_compact(&p, b) < 0) return -1;
        } else {
            if (!first && jb_ch(b, ',') < 0) return -1;
            first = 0;
            if (json_compact(&p, b) < 0) return -1;
        }
    }
    if (jb_ch(b, is_arr ? ']' : '}') < 0) return -1;
    *pp = p;
    return 0;
}

static int cmp_tool_json(const void *a, const void *b) {
    const char *sa = *(char *const *)a;
    const char *sb = *(char *const *)b;
    const char *na = strstr(sa, "\"name\":\"");
    const char *nb = strstr(sb, "\"name\":\"");
    na = na ? na + 8 : "";
    nb = nb ? nb + 8 : "";
    return strcmp(na, nb);
}

char *q4_qwen_compact_tools_json(const char *tools_array) {
    if (!tools_array) return NULL;
    const char *p = tools_array;
    jbuf b = {0};
    b.cap = 512;
    b.p = malloc(b.cap);
    if (!b.p) return NULL;
    b.p[0] = 0;
    if (json_compact(&p, &b) < 0 || !b.p || b.p[0] != '[') {
        free(b.p);
        return NULL;
    }
    /* Stable order so a tool appearing later in OpenCode's array does not
     * rewrite the system prefix and kill the session cache. */
    char *src = b.p;
    const char *q = src;
    if (*q != '[') return src;
    q++;
    int n = 0, cap = 16;
    char **items = malloc((size_t)cap * sizeof(char *));
    if (!items) return src;
    while (*q && *q != ']') {
        while (*q && (*q == ',' || isspace((unsigned char)*q))) q++;
        if (*q != '{') break;
        const char *st = q;
        q = skip_json_item(q);
        size_t L = (size_t)(q - st);
        if (n == cap) {
            cap *= 2;
            items = realloc(items, (size_t)cap * sizeof(char *));
        }
        items[n] = dup_n(st, L);
        n++;
    }
    if (n > 1) qsort(items, (size_t)n, sizeof(char *), cmp_tool_json);
    jbuf o = {0};
    o.cap = b.n + 8;
    o.p = malloc(o.cap);
    if (!o.p) {
        for (int i = 0; i < n; i++) free(items[i]);
        free(items);
        return src;
    }
    o.p[0] = '[';
    o.n = 1;
    o.p[1] = 0;
    for (int i = 0; i < n; i++) {
        if (i && jb_ch(&o, ',') < 0) break;
        jb_add(&o, items[i], strlen(items[i]));
        free(items[i]);
    }
    jb_ch(&o, ']');
    free(items);
    free(src);
    return o.p;
}

char *q4_qwen_tools_preamble(const char *tools_array, int with_think) {
    (void)with_think; /* format trigger is the <tools> block, not the xhigh essay */
    char *compact = q4_qwen_compact_tools_json(tools_array);
    const char *body = compact ? compact : tools_array;
    if (!body) return NULL;
    const char *p = body;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '[') {
        free(compact);
        return NULL;
    }
    p++;
    const char *mid = "# Tools\n<tools>";
    const char *tail =
        "\n</tools>\n"
        "Call a function with this XML and nothing after it:\n"
        "<tool_call>\n<function=NAME>\n<parameter=KEY>\nvalue\n"
        "</parameter>\n</function>\n</tool_call>";
    size_t cap = strlen(mid) + strlen(tail) + strlen(body) + 8;
    char *o = malloc(cap);
    if (!o) {
        free(compact);
        return NULL;
    }
    size_t n = (size_t)sprintf(o, "%s", mid);
    int any = 0;
    while (*p && *p != ']') {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (*p == ']') break;
        if (*p != '{') break;
        const char *start = p;
        p = skip_json_item(p);
        size_t L = (size_t)(p - start);
        if (n + L + strlen(tail) + 2 >= cap) {
            cap = n + L + strlen(tail) + 256;
            char *q = realloc(o, cap);
            if (!q) {
                free(o);
                free(compact);
                return NULL;
            }
            o = q;
        }
        o[n++] = '\n';
        memcpy(o + n, start, L);
        n += L;
        o[n] = 0;
        any = 1;
    }
    if (!any) {
        free(o);
        free(compact);
        return NULL;
    }
    memcpy(o + n, tail, strlen(tail) + 1);
    free(compact);
    return o;
}

/* ---- tools as the chat template renders them ----
 * chat_template.jinja writes each tool as `tool | tojson`, which under
 * transformers' apply_chat_template is Python's json.dumps(tool,
 * ensure_ascii=False): ", " / ": " separators, keys in the order sent,
 * non-ASCII raw, strings re-escaped.  py_json re-serializes one JSON value
 * that way (string escapes decoded and re-encoded, floats as Python's
 * repr). */
static void jb_utf8(jbuf *b, uint32_t cp) {
    char u[4];
    int n;
    if (cp < 0x80) { u[0] = (char)cp; n = 1; }
    else if (cp < 0x800) { u[0] = (char)(0xc0 | (cp >> 6)); u[1] = (char)(0x80 | (cp & 0x3f)); n = 2; }
    else if (cp < 0x10000) { u[0] = (char)(0xe0 | (cp >> 12)); u[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
                             u[2] = (char)(0x80 | (cp & 0x3f)); n = 3; }
    else { u[0] = (char)(0xf0 | (cp >> 18)); u[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
           u[2] = (char)(0x80 | ((cp >> 6) & 0x3f)); u[3] = (char)(0x80 | (cp & 0x3f)); n = 4; }
    jb_add(b, u, (size_t)n);
}

static int hex4(const char *p, uint32_t *v) {
    uint32_t x = 0;
    for (int i = 0; i < 4; i++) {
        const char c = p[i];
        x <<= 4;
        if (c >= '0' && c <= '9') x |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') x |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') x |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    *v = x;
    return 1;
}

/* One escaped character in Python's ensure_ascii=False output. */
static void py_esc(jbuf *b, uint32_t cp) {
    char t[8];
    switch (cp) {
    case '"': jb_add(b, "\\\"", 2); return;
    case '\\': jb_add(b, "\\\\", 2); return;
    case '\n': jb_add(b, "\\n", 2); return;
    case '\r': jb_add(b, "\\r", 2); return;
    case '\t': jb_add(b, "\\t", 2); return;
    case '\b': jb_add(b, "\\b", 2); return;
    case '\f': jb_add(b, "\\f", 2); return;
    default: break;
    }
    if (cp < 0x20) {
        snprintf(t, sizeof t, "\\u%04x", cp);
        jb_add(b, t, 6);
        return;
    }
    jb_utf8(b, cp);
}

static const char *py_json_str(const char *p, jbuf *b) {
    /* p at the opening quote */
    p++;
    jb_ch(b, '"');
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            const char c = p[1];
            uint32_t cp = 0;
            p += 2;
            switch (c) {
            case 'n': cp = '\n'; break;
            case 'r': cp = '\r'; break;
            case 't': cp = '\t'; break;
            case 'b': cp = '\b'; break;
            case 'f': cp = '\f'; break;
            case 'u':
                if (!hex4(p, &cp)) { cp = 'u'; break; }
                p += 4;
                if (cp >= 0xd800 && cp < 0xdc00 && p[0] == '\\' && p[1] == 'u') {
                    uint32_t lo;
                    if (hex4(p + 2, &lo) && lo >= 0xdc00 && lo < 0xe000) {
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        p += 6;
                    }
                }
                break;
            default: cp = (unsigned char)c; break; /* \" \\ \/ */
            }
            py_esc(b, cp);
            continue;
        }
        const unsigned char u = (unsigned char)*p;
        if (u < 0x80) {
            py_esc(b, u);
            p++;
        } else {
            const char *s = p++;
            while ((*p & 0xc0) == 0x80) p++;
            jb_add(b, s, (size_t)(p - s));
        }
    }
    if (*p == '"') p++;
    jb_ch(b, '"');
    return p;
}

/* A JSON number as Python prints it: integers as written, anything with a
 * fraction or exponent as repr(float). */
static void py_num(const char *s, size_t n, jbuf *b) {
    int is_float = 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] == '.' || s[i] == 'e' || s[i] == 'E') is_float = 1;
    if (!is_float) {
        jb_add(b, s, n);
        return;
    }
    char tmp[64];
    if (n >= sizeof tmp) {
        jb_add(b, s, n);
        return;
    }
    memcpy(tmp, s, n);
    tmp[n] = 0;
    const double v = strtod(tmp, NULL);
    char r[40];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(r, sizeof r, "%.*e", prec - 1, v);
        if (strtod(r, NULL) == v) break;
    }
    /* r = d.ddde[+-]XX: Python repr uses positional for -4 <= exp < 16 */
    char *e = strchr(r, 'e');
    if (!e) {
        jb_add(b, r, strlen(r));
        return;
    }
    const int ex = atoi(e + 1);
    char dig[24];
    int nd = 0;
    const int neg = r[0] == '-';
    for (char *q = r + neg; q < e; q++)
        if (*q != '.') dig[nd++] = *q;
    while (nd > 1 && dig[nd - 1] == '0') nd--;
    char out[64];
    int o = 0;
    if (neg) out[o++] = '-';
    if (ex >= -4 && ex < 16) {
        if (ex < 0) {
            out[o++] = '0';
            out[o++] = '.';
            for (int i = 0; i < -ex - 1; i++) out[o++] = '0';
            for (int i = 0; i < nd; i++) out[o++] = dig[i];
        } else {
            for (int i = 0; i <= ex; i++) out[o++] = i < nd ? dig[i] : '0';
            out[o++] = '.';
            if (nd > ex + 1)
                for (int i = ex + 1; i < nd; i++) out[o++] = dig[i];
            else
                out[o++] = '0';
        }
        out[o] = 0;
    } else {
        out[o++] = dig[0];
        if (nd > 1) {
            out[o++] = '.';
            for (int i = 1; i < nd; i++) out[o++] = dig[i];
        }
        o += snprintf(out + o, sizeof out - (size_t)o, "e%c%02d", ex < 0 ? '-' : '+',
                      ex < 0 ? -ex : ex);
    }
    jb_add(b, out, strlen(out));
}

static const char *py_json(const char *p, jbuf *b) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == '"') return py_json_str(p, b);
    if (*p == '{' || *p == '[') {
        const int arr = *p == '[';
        const char close = arr ? ']' : '}';
        p++;
        jb_ch(b, arr ? '[' : '{');
        int first = 1;
        for (;;) {
            while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
            if (!*p) return p;
            if (*p == close) {
                p++;
                break;
            }
            if (!first) jb_add(b, ", ", 2);
            first = 0;
            if (!arr) {
                if (*p != '"') return p + strlen(p);
                p = py_json_str(p, b);
                while (*p && isspace((unsigned char)*p)) p++;
                if (*p == ':') p++;
                jb_add(b, ": ", 2);
            }
            p = py_json(p, b);
        }
        jb_ch(b, close);
        return p;
    }
    const char *s = p;
    p = skip_json_value(p);
    if (p > s && (*s == '-' || (*s >= '0' && *s <= '9')))
        py_num(s, (size_t)(p - s), b);
    else
        jb_add(b, s, (size_t)(p - s));
    return p;
}

static int cmp_tool_name(const void *a, const void *b) {
    const char *const *x = a, *const *y = b;
    return strcmp(x[1], y[1]);
}

/* The template's tools block for the system turn (it goes before the
 * system text): every tool as sent, re-serialized as tojson, in name order
 * (a stable prefix: OpenCode's tool order varies between requests). */
char *q4_qwen_tools_block(const char *tools_array) {
    if (!tools_array) return NULL;
    const char *p = tools_array;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '[') return NULL;
    p++;
    int n = 0, cap = 16;
    char **items = malloc((size_t)cap * 2 * sizeof(char *)); /* json, name */
    if (!items) return NULL;
    for (;;) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (*p != '{') break;
        jbuf t = {0};
        t.cap = 256;
        t.p = malloc(t.cap);
        if (!t.p) break;
        t.p[0] = 0;
        p = py_json(p, &t);
        if (n == cap) {
            cap *= 2;
            char **q = realloc(items, (size_t)cap * 2 * sizeof(char *));
            if (!q) {
                free(t.p);
                break;
            }
            items = q;
        }
        /* the name: the function object's, else the top level's */
        const char *nm = strstr(t.p, "\"function\": {");
        nm = strstr(nm ? nm : t.p, "\"name\": \"");
        const char *ns = nm ? nm + 9 : "";
        const char *ne = nm ? strchr(ns, '"') : ns;
        items[2 * n] = t.p;
        items[2 * n + 1] = dup_n(ns, (size_t)((ne ? ne : ns) - ns));
        n++;
    }
    if (n == 0) {
        free(items);
        return NULL;
    }
    qsort(items, (size_t)n, 2 * sizeof(char *), cmp_tool_name);
    static const char *head =
        "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    static const char *tail =
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
        "following format with NO suffix:\n\n<tool_call>\n"
        "<function=example_function_name>\n<parameter=example_parameter_1>\n"
        "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\nthat can span\n"
        "multiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
        "format: an inner <function=...></function> block must be nested "
        "within <tool_call></tool_call> XML tags\n- Required parameters MUST "
        "be specified\n- You may provide optional reasoning for your function "
        "call in natural language BEFORE the function call, but NOT after\n- "
        "If there is no function call available, answer the question like "
        "normal with your current knowledge and do not tell the user about "
        "function calls\n</IMPORTANT>";
    jbuf o = {0};
    o.cap = 1024;
    o.p = malloc(o.cap);
    if (o.p) {
        o.p[0] = 0;
        jb_add(&o, head, strlen(head));
        for (int i = 0; i < n; i++) {
            jb_ch(&o, '\n');
            jb_add(&o, items[2 * i], strlen(items[2 * i]));
        }
        jb_add(&o, tail, strlen(tail));
    }
    for (int i = 0; i < 2 * n; i++) free(items[i]);
    free(items);
    return o.p;
}

/* Presence penalty (OpenAI / vLLM semantics: every token this response
 * has emitted so far loses `pres` from its logit once, however often it
 * occurred).  The history is the engine's emitted tokens plus, inside an
 * MTP cycle, the drafts ahead of the row being judged (q4_pen_mark /
 * q4_pen_push / q4_pen_rewind), so every verify row sees exactly the
 * history its token would have in plain decode. */
static struct {
    float pres;
    int n, cap;
    int32_t *ids;   /* emitted ids, in order (duplicates kept for rewind) */
    uint8_t *seen;  /* per vocab id: occurrences in ids[], saturating */
    uint32_t n_vocab;
} g_pen;

void q4_pen_begin(float presence, uint32_t n_vocab) {
    if (g_pen.seen && g_pen.n_vocab != n_vocab) {
        free(g_pen.seen);
        g_pen.seen = NULL;
    }
    if (!g_pen.seen && presence > 0.f) {
        g_pen.seen = calloc(n_vocab, 1);
        g_pen.n_vocab = g_pen.seen ? n_vocab : 0;
    }
    if (g_pen.seen)
        for (int i = 0; i < g_pen.n; i++) g_pen.seen[g_pen.ids[i]] = 0;
    g_pen.n = 0;
    g_pen.pres = g_pen.seen ? presence : 0.f;
}

/* Extra candidates: the kept top-k is exact while no more than this many
 * of the candidates are penalized (top_k 20 -> 64 candidates); past that
 * a token ranked beyond them could have outranked a penalized one.  The
 * device selection costs a pass per candidate, so not the full 128. */
int q4_pen_extra(void) {
    return g_pen.pres > 0.f ? 44 : 0;
}

void q4_pen_push(int32_t id) {
    if (g_pen.pres <= 0.f || id < 0 || (uint32_t)id >= g_pen.n_vocab) return;
    if (g_pen.n == g_pen.cap) {
        const int c = g_pen.cap ? 2 * g_pen.cap : 4096;
        int32_t *p = realloc(g_pen.ids, (size_t)c * sizeof(int32_t));
        if (!p) return;
        g_pen.ids = p;
        g_pen.cap = c;
    }
    g_pen.ids[g_pen.n++] = id;
    if (g_pen.seen[id] < 255) g_pen.seen[id]++;
}

int q4_pen_mark(void) {
    return g_pen.n;
}

void q4_pen_rewind(int mark) {
    while (g_pen.n > mark && g_pen.n > 0) {
        const int32_t id = g_pen.ids[--g_pen.n];
        /* a saturated count is never taken back (255 pushes of one id
         * inside a rewind window cannot happen: windows are <= 8) */
        if (g_pen.seen[id] && g_pen.seen[id] < 255) g_pen.seen[id]--;
    }
}

/* Temp softmax and top-p over an already-selected top-k (rank order).
 * With a presence penalty the candidates may be more than `keep` (the
 * callers fetch extra so a penalized token's place is taken by the next
 * one): penalize, re-rank, keep the best `keep`. */
static int dist_finish_k(const int32_t *ids_in, const float *w_in, int top_k,
                         int keep, float temp, float top_p, int32_t *ids,
                         float *probs) {
    if (top_k <= 0) return 0;
    if (top_k > 128) top_k = 128;
    if (temp < 1e-5f) temp = 1e-5f;
    float w[128];
    int n = 0;
    for (int i = 0; i < top_k; i++) {
        if (ids_in[i] < 0) break;
        ids[n] = ids_in[i];
        w[n] = w_in[i];
        n++;
    }
    top_k = n;
    if (!top_k) return 0;
    if (g_pen.pres > 0.f) {
        int any = 0;
        for (int i = 0; i < top_k; i++)
            if (g_pen.seen[ids[i]]) {
                w[i] -= g_pen.pres;
                any = 1;
            }
        if (any) /* stable insertion sort, descending: ties keep rank order */
            for (int i = 1; i < top_k; i++) {
                const float wv = w[i];
                const int32_t iv = ids[i];
                int j = i - 1;
                while (j >= 0 && w[j] < wv) {
                    w[j + 1] = w[j];
                    ids[j + 1] = ids[j];
                    j--;
                }
                w[j + 1] = wv;
                ids[j + 1] = iv;
            }
    }
    if (keep > 0 && keep < top_k) top_k = keep;
    float m = w[0];
    for (int i = 1; i < top_k; i++)
        if (w[i] > m) m = w[i];
    for (int i = 0; i < top_k; i++) w[i] = (w[i] - m) / temp;
    float maxw = w[0];
    for (int i = 1; i < top_k; i++)
        if (w[i] > maxw) maxw = w[i];
    float s = 0;
    for (int i = 0; i < top_k; i++) {
        w[i] = expf(w[i] - maxw);
        s += w[i];
    }
    for (int i = 0; i < top_k; i++) w[i] /= s;
    if (top_p > 0 && top_p < 1) {
        float acc = 0;
        int cut = top_k;
        for (int i = 0; i < top_k; i++) {
            acc += w[i];
            if (acc >= top_p) {
                cut = i + 1;
                break;
            }
        }
        s = 0;
        for (int i = 0; i < cut; i++) s += w[i];
        for (int i = 0; i < cut; i++) w[i] /= s;
        top_k = cut;
    }
    for (int i = 0; i < top_k; i++) probs[i] = w[i];
    return top_k;
}

/* Build the same distribution q4_sample draws from: top-k (cap 128), temp
 * softmax, then top-p truncation. Returns the candidate count. */
int q4_dist_build(const float *logits, uint32_t n, float temp, int top_k,
                  float top_p, int32_t *ids, float *probs) {
    if (!logits || n == 0 || !ids || !probs) return 0;
    if (top_k <= 0 || top_k > (int)n) top_k = (int)n;
    if (top_k > 128) top_k = 128;
    int nc = top_k + q4_pen_extra();
    if (nc > 128) nc = 128;
    if (nc > (int)n) nc = (int)n;
    float w[128];
    int32_t sel[128];
    q4_topk(logits, n, (uint32_t)nc, sel, w, false);
    return dist_finish_k(sel, w, nc, top_k, temp, top_p, ids, probs);
}

int q4_dist_from_topk(const int32_t *ids_in, const float *logits, int k,
                      float temp, float top_p, int32_t *ids, float *probs) {
    if (!ids_in || !logits || !ids || !probs || k <= 0) return 0;
    return dist_finish_k(ids_in, logits, k, k, temp, top_p, ids, probs);
}

int q4_dist_from_topk_k(const int32_t *ids_in, const float *logits, int k,
                        int keep, float temp, float top_p, int32_t *ids,
                        float *probs) {
    if (!ids_in || !logits || !ids || !probs || k <= 0) return 0;
    return dist_finish_k(ids_in, logits, k, keep, temp, top_p, ids, probs);
}

int32_t q4_dist_sample(const int32_t *ids, const float *probs, int k,
                       unsigned *rng) {
    if (k <= 0) return 0;
    unsigned r = *rng * 1664525u + 1013904223u;
    *rng = r;
    float u = (r / 4294967296.0f);
    float acc = 0;
    for (int i = 0; i < k; i++) {
        acc += probs[i];
        if (u <= acc) return ids[i];
    }
    return ids[k - 1];
}

float q4_dist_prob(const int32_t *ids, const float *probs, int k, int32_t id) {
    for (int i = 0; i < k; i++)
        if (ids[i] == id) return probs[i];
    return 0.f;
}

int q4_spec_accept(const int32_t *tids, const float *tp, int tn,
                   const int32_t *dids, const float *dp, int dn,
                   int32_t draft, unsigned *rng, int32_t *resample) {
    float pd = q4_dist_prob(dids, dp, dn, draft);
    float pt = q4_dist_prob(tids, tp, tn, draft);
    if (pd <= 0.f) pd = 1e-9f;
    float acc_p = pt / pd;
    if (acc_p >= 1.f) return 1;
    unsigned r = *rng * 1664525u + 1013904223u;
    *rng = r;
    float u = r / 4294967296.0f;
    if (u <= acc_p) return 1;
    /* Rejected: sample from norm(max(0, p_target - p_draft)) over the target
     * support (draft-only ids carry zero target mass, so residual is zero
     * there). Standard speculative residual sampling. */
    float resid[Q4_DIST_MAX];
    int32_t rid[Q4_DIST_MAX];
    int nr = 0;
    float sum = 0;
    for (int i = 0; i < tn; i++) {
        float p = tp[i] - q4_dist_prob(dids, dp, dn, tids[i]);
        if (p > 0) {
            rid[nr] = tids[i];
            resid[nr] = p;
            sum += p;
            nr++;
        }
    }
    if (sum <= 0 || nr == 0) {
        /* Draft == target everywhere; accepted by the ratio test anyway. */
        if (resample) *resample = draft;
        return 0;
    }
    for (int i = 0; i < nr; i++) resid[i] /= sum;
    if (resample) *resample = q4_dist_sample(rid, resid, nr, rng);
    return 0;
}

int32_t q4_sample(const float *logits, uint32_t n, float temp, int top_k, float top_p,
                  unsigned *rng) {
    if (!logits || n == 0) return 0;
    if (temp <= 1e-5f) {
        int32_t b = 0;
        float v = logits[0];
        for (uint32_t i = 1; i < n; i++)
            if (logits[i] > v) {
                v = logits[i];
                b = (int32_t)i;
            }
        return b;
    }
    int32_t ids[Q4_DIST_MAX];
    float probs[Q4_DIST_MAX];
    int k = q4_dist_build(logits, n, temp, top_k, top_p, ids, probs);
    return q4_dist_sample(ids, probs, k, rng);
}
