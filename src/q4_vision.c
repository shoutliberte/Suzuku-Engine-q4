/* Images: the server's side of the vision path.
 *
 * The encoder (the model's mmproj: a 27-layer ViT plus the projector into
 * the text model's 2560-wide input) runs in a helper process on the CPU,
 * so its weights and work buffers live in DRAM and the GPU keeps every
 * byte for the experts.  The helper is `q4-vision` (tools/vision, llama.cpp
 * mtmd); Strata's `strata-vision` speaks the same protocol:
 *
 *   start:  <bin> --mmproj <f> --model <text model> [--threads N]
 *                 [--max-tokens N] [--min-tokens N]   ->  "READY <n_embd>"
 *   stdin:  ENC <image path> <output path>   ->  "OK <n> <nx> <ny> <ms>" | "ERR <msg>"
 *   output: int32 {'SVE1', n, nx, ny, n_embd}, float32 [n][n_embd]
 *
 * It starts at the first picture (text-only servers never pay for it) and
 * stays resident; encodings are kept in an LRU keyed by the image bytes,
 * since chat clients resend every picture of a conversation each turn.
 * Entries used by the current request are never evicted while it runs. */
#define _GNU_SOURCE
#include "q4.h"
#include "q4_vision.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint64_t key;
    int32_t n, nx, ny;
    float *rows;
    size_t bytes;
    uint64_t used; /* request serial of the last use */
    uint64_t hash; /* the engine's identity of the picture */
} vis_ent;

struct q4_vis {
    char bin[4096], mmproj[4096], model[4096], dir[256];
    int max_tokens, min_tokens, threads;
    uint32_t n_embd;
    pid_t pid;
    int to_fd, from_fd;
    char rbuf[4096];
    size_t rlen;
    int broken; /* failed to start: do not retry every request */
    vis_ent *ent;
    int n_ent, cap_ent;
    size_t bytes, budget;
    uint64_t serial, seq;
};

static int env_int(const char *k, int def) {
    const char *v = getenv(k);
    return v && v[0] ? atoi(v) : def;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

q4_vis *q4_vis_open(const char *model, uint32_t n_embd) {
    const char *mm = getenv("Q4_VISION_MMPROJ");
    if (!mm || !mm[0] || !strcmp(mm, "0") || !model || !model[0]) return NULL;
    q4_vis *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    snprintf(v->mmproj, sizeof(v->mmproj), "%s", mm);
    snprintf(v->model, sizeof(v->model), "%s", model);
    const char *b = getenv("Q4_VISION_BIN");
    if (b && b[0]) {
        snprintf(v->bin, sizeof(v->bin), "%s", b);
    } else {
        /* q4-vision next to the q4 binary */
        char self[4096];
        ssize_t L = readlink("/proc/self/exe", self, sizeof(self) - 1);
        self[L > 0 ? L : 0] = 0;
        char *sl = strrchr(self, '/');
        if (sl) *sl = 0;
        snprintf(v->bin, sizeof(v->bin), "%s/q4-vision", sl ? self : ".");
    }
    /* 1024: what a screenshot's small text needs.  Encode time on the CPU
     * grows faster than the token count (the ViT attends over 4 patches
     * per token): ~2 s at 300, ~16 s at ~1000 on a 16-core Zen 5. */
    v->max_tokens = env_int("Q4_VISION_MAX_TOKENS", 1024);
    v->min_tokens = env_int("Q4_VISION_MIN_TOKENS", 0);
    v->threads = env_int("Q4_VISION_THREADS", 0);
    v->budget = (size_t)env_int("Q4_VISION_CACHE_MIB", 256) << 20;
    v->n_embd = n_embd;
    v->pid = -1;
    v->to_fd = v->from_fd = -1;
    fprintf(stderr, "q4: vision on (encoder %s, mmproj %s, max %d tokens; "
                    "starts at the first picture)\n",
            v->bin, v->mmproj, v->max_tokens);
    return v;
}

static void vis_stop(q4_vis *v) {
    if (v->to_fd >= 0) {
        (void)!write(v->to_fd, "QUIT\n", 5);
        close(v->to_fd);
    }
    if (v->from_fd >= 0) close(v->from_fd);
    v->to_fd = v->from_fd = -1;
    if (v->pid > 0) {
        for (int i = 0; i < 50; i++) {
            if (waitpid(v->pid, NULL, WNOHANG) == v->pid) {
                v->pid = -1;
                break;
            }
            usleep(20000);
        }
        if (v->pid > 0) {
            kill(v->pid, SIGKILL);
            waitpid(v->pid, NULL, 0);
        }
    }
    v->pid = -1;
    v->rlen = 0;
}

void q4_vis_close(q4_vis *v) {
    if (!v) return;
    vis_stop(v);
    for (int i = 0; i < v->n_ent; i++) free(v->ent[i].rows);
    free(v->ent);
    if (v->dir[0]) rmdir(v->dir);
    free(v);
}

/* One line from the helper, waiting at most tmo seconds (0: forever). */
static int vis_line(q4_vis *v, char *out, size_t outn, double tmo) {
    const double t0 = now_s();
    for (;;) {
        char *nl = memchr(v->rbuf, '\n', v->rlen);
        if (nl) {
            size_t L = (size_t)(nl - v->rbuf);
            size_t c = L < outn - 1 ? L : outn - 1;
            memcpy(out, v->rbuf, c);
            out[c] = 0;
            memmove(v->rbuf, nl + 1, v->rlen - L - 1);
            v->rlen -= L + 1;
            return 0;
        }
        if (v->rlen == sizeof(v->rbuf)) v->rlen = 0; /* a runaway line */
        int wait_ms = -1;
        if (tmo > 0) {
            const double left = tmo - (now_s() - t0);
            if (left <= 0) return -1;
            wait_ms = (int)(left * 1000) + 1;
        }
        struct pollfd pf = {v->from_fd, POLLIN, 0};
        int r = poll(&pf, 1, wait_ms);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        ssize_t k = read(v->from_fd, v->rbuf + v->rlen, sizeof(v->rbuf) - v->rlen);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1; /* the helper exited */
        v->rlen += (size_t)k;
    }
}

static double mem_avail_gib(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    double g = -1;
    while (fgets(line, sizeof(line), f)) {
        unsigned long kb;
        if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) {
            g = (double)kb / (1024.0 * 1024.0);
            break;
        }
    }
    fclose(f);
    return g;
}

static int vis_start(q4_vis *v, char *err, size_t errn) {
    if (v->pid > 0) return 0;
    if (v->broken) {
        snprintf(err, errn, "the image encoder failed to start earlier "
                            "(see the server log)");
        return -1;
    }
    /* ~1.1 GiB resident, ~1.3 GiB while encoding, beside a ~45 GiB q4:
     * refuse rather than let the desktop pay for it, and stay clear of
     * serve.sh's watchdog, which kills q4 under Q4_SAFE_FLOOR (kB). */
    const char *sf = getenv("Q4_SAFE_FLOOR");
    const double floor_gib = sf && atol(sf) > 0 ? (double)atol(sf) / 1048576.0 : 3.0;
    const char *ng = getenv("Q4_VISION_NEED_GIB");
    const double need = ng && ng[0] ? atof(ng) : floor_gib + 2.0;
    const double av = mem_avail_gib();
    if (av >= 0 && av < need) {
        snprintf(err, errn, "not enough free RAM for the image encoder "
                            "(%.1f GiB available, %.0f needed)", av, need);
        return -1;
    }
    if (!v->dir[0]) {
        const char *base = access("/dev/shm", W_OK) == 0 ? "/dev/shm" : "/tmp";
        snprintf(v->dir, sizeof(v->dir), "%s/q4-vision-XXXXXX", base);
        if (!mkdtemp(v->dir)) {
            v->dir[0] = 0;
            snprintf(err, errn, "cannot create the image work directory");
            return -1;
        }
    }
    int in[2], out[2];
    if (pipe2(in, O_CLOEXEC) || pipe2(out, O_CLOEXEC)) {
        snprintf(err, errn, "pipe: %s", strerror(errno));
        return -1;
    }
    char mt[32], mn[32], th[32];
    snprintf(mt, sizeof(mt), "%d", v->max_tokens);
    snprintf(mn, sizeof(mn), "%d", v->min_tokens);
    snprintf(th, sizeof(th), "%d", v->threads);
    const char *argv[16];
    int a = 0;
    argv[a++] = v->bin;
    argv[a++] = "--mmproj";
    argv[a++] = v->mmproj;
    argv[a++] = "--model";
    argv[a++] = v->model;
    if (v->max_tokens > 0) argv[a++] = "--max-tokens", argv[a++] = mt;
    if (v->min_tokens > 0) argv[a++] = "--min-tokens", argv[a++] = mn;
    if (v->threads > 0) argv[a++] = "--threads", argv[a++] = th;
    argv[a] = NULL;
    const double t0 = now_s();
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, errn, "fork: %s", strerror(errno));
        close(in[0]), close(in[1]), close(out[0]), close(out[1]);
        return -1;
    }
    if (pid == 0) {
        dup2(in[0], 0);
        dup2(out[1], 1);
        /* nothing else of the server's: an inherited client socket kept
         * the connection open after the server closed it */
        closefrom(3);
        /* the encoder is CPU-only: keep any GPU runtime off the card */
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
        setenv("HIP_VISIBLE_DEVICES", "-1", 1);
        setenv("ROCR_VISIBLE_DEVICES", "-1", 1);
        signal(SIGPIPE, SIG_DFL);
        execv(v->bin, (char *const *)argv);
        fprintf(stderr, "q4: cannot run the image encoder %s: %s\n", v->bin,
                strerror(errno));
        _exit(127);
    }
    close(in[0]);
    close(out[1]);
    v->pid = pid;
    v->to_fd = in[1];
    v->from_fd = out[0];
    v->rlen = 0;
    char line[256];
    const int rc = vis_line(v, line, sizeof(line), env_int("Q4_VISION_READY_S", 300));
    unsigned emb = 0;
    if (rc || sscanf(line, "READY %u", &emb) != 1 || emb != v->n_embd) {
        if (rc) snprintf(err, errn, "the image encoder did not start (%s)", v->bin);
        else if (emb)
            snprintf(err, errn, "the image encoder's width %u is not the "
                                "model's %u (wrong mmproj?)", emb, v->n_embd);
        else snprintf(err, errn, "the image encoder did not start: %.200s", line);
        vis_stop(v);
        v->broken = 1;
        return -1;
    }
    fprintf(stderr, "q4: image encoder ready in %.1f s (pid %d)\n",
            now_s() - t0, (int)pid);
    return 0;
}

static uint64_t fnv64(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

void q4_vis_begin(q4_vis *v) {
    if (v) v->serial++;
}

static void vis_evict(q4_vis *v, size_t want) {
    while (v->bytes + want > v->budget && v->n_ent > 0) {
        int lo = -1;
        for (int i = 0; i < v->n_ent; i++)
            if (v->ent[i].used != v->serial &&
                (lo < 0 || v->ent[i].used < v->ent[lo].used))
                lo = i;
        if (lo < 0) return; /* everything left is this request's */
        v->bytes -= v->ent[lo].bytes;
        free(v->ent[lo].rows);
        v->ent[lo] = v->ent[--v->n_ent];
    }
}

static int write_file(const char *path, const uint8_t *d, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    size_t o = 0;
    while (o < n) {
        ssize_t k = write(fd, d + o, n - o);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) {
            close(fd);
            return -1;
        }
        o += (size_t)k;
    }
    return close(fd);
}

int q4_vis_encode(q4_vis *v, const uint8_t *data, size_t n, q4_image *out,
                  char *err, size_t errn) {
    if (!v || !data || !n || !out) return -1;
    const int32_t knobs[2] = {v->max_tokens, v->min_tokens};
    const uint64_t key = fnv64(knobs, sizeof(knobs), fnv64(data, n, 0xcbf29ce484222325ull));
    for (int i = 0; i < v->n_ent; i++)
        if (v->ent[i].key == key) {
            vis_ent *e = &v->ent[i];
            e->used = v->serial;
            *out = (q4_image){e->n, e->nx, e->ny, e->rows, e->hash};
            fprintf(stderr, "q4: image %zu B: cached (%d tokens %dx%d)\n", n,
                    e->n, e->nx, e->ny);
            return 0;
        }
    if (vis_start(v, err, errn)) return -1;
    char ip[512], op[512], line[512];
    const uint64_t id = ++v->seq;
    snprintf(ip, sizeof(ip), "%s/in-%llu", v->dir, (unsigned long long)id);
    snprintf(op, sizeof(op), "%s/out-%llu.sve", v->dir, (unsigned long long)id);
    if (write_file(ip, data, n)) {
        unlink(ip);
        snprintf(err, errn, "cannot write the image to %s", v->dir);
        return -1;
    }
    const double t0 = now_s();
    char cmd[1100];
    const int cl = snprintf(cmd, sizeof(cmd), "ENC %s %s\n", ip, op);
    int rc = -1;
    if (write(v->to_fd, cmd, (size_t)cl) == cl)
        rc = vis_line(v, line, sizeof(line), env_int("Q4_VISION_ENCODE_S", 300));
    unlink(ip);
    if (rc) {
        unlink(op);
        snprintf(err, errn, "the image encoder stopped or timed out");
        vis_stop(v); /* restarted at the next picture */
        return -1;
    }
    int on = 0, onx = 0, ony = 0;
    double ms = 0;
    if (strncmp(line, "OK ", 3) || sscanf(line + 3, "%d %d %d %lf", &on, &onx, &ony, &ms) < 3) {
        unlink(op);
        snprintf(err, errn, "the image could not be read: %.300s",
                 strncmp(line, "ERR ", 4) ? line : line + 4);
        return -1;
    }
    FILE *f = fopen(op, "rb");
    int32_t h[5] = {0};
    float *rows = NULL;
    const size_t cnt = (size_t)(on > 0 ? on : 0) * v->n_embd;
    bool ok = f && fread(h, sizeof(int32_t), 5, f) == 5 && h[0] == 0x31455653 &&
              h[1] == on && h[2] == onx && h[3] == ony && h[1] > 0 && h[2] > 0 &&
              h[3] > 0 && h[2] * h[3] == h[1] && (uint32_t)h[4] == v->n_embd &&
              (rows = malloc(cnt * sizeof(float))) != NULL &&
              fread(rows, sizeof(float), cnt, f) == cnt;
    if (f) fclose(f);
    unlink(op);
    if (!ok) {
        free(rows);
        snprintf(err, errn, "the image encoder wrote a bad embeddings file");
        return -1;
    }
    const size_t bytes = cnt * sizeof(float);
    vis_evict(v, bytes);
    if (v->n_ent == v->cap_ent) {
        int nc = v->cap_ent ? v->cap_ent * 2 : 16;
        vis_ent *ne = realloc(v->ent, (size_t)nc * sizeof(*ne));
        if (!ne) {
            free(rows);
            snprintf(err, errn, "out of memory");
            return -1;
        }
        v->ent = ne;
        v->cap_ent = nc;
    }
    /* the engine's identity: the grid and the embeddings themselves */
    const int32_t grid[3] = {on, onx, ony};
    const uint64_t hash = fnv64(rows, bytes, fnv64(grid, sizeof(grid), 0xcbf29ce484222325ull));
    v->ent[v->n_ent++] = (vis_ent){key, on, onx, ony, rows, bytes, v->serial, hash};
    v->bytes += bytes;
    *out = (q4_image){on, onx, ony, rows, hash};
    fprintf(stderr, "q4: image %zu B -> %d tokens (%dx%d) in %.2f s\n", n, on,
            onx, ony, now_s() - t0);
    return 0;
}

/* ---- image sources ---- */

static int b64v(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

static int b64_decode(const char *s, uint8_t **out, size_t *n) {
    const size_t L = strlen(s);
    uint8_t *o = malloc(L / 4 * 3 + 4);
    if (!o) return -1;
    size_t k = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < L; i++) {
        const int c = (unsigned char)s[i];
        if (c == '=' ) break;
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        const int d = b64v(c);
        if (d < 0) {
            free(o);
            return -1;
        }
        acc = (acc << 6) | (uint32_t)d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            o[k++] = (uint8_t)(acc >> bits);
        }
    }
    *out = o;
    *n = k;
    return k ? 0 : -1;
}

#define VIS_MAX_FILE (64u << 20)

int q4_vis_load(const char *src, int is_b64, uint8_t **out, size_t *n,
                char *err, size_t errn) {
    *out = NULL;
    *n = 0;
    if (!src || !src[0]) {
        snprintf(err, errn, "an image part without data");
        return -1;
    }
    if (is_b64) {
        if (b64_decode(src, out, n)) {
            snprintf(err, errn, "the image's base64 data could not be decoded");
            return -1;
        }
        return 0;
    }
    if (!strncmp(src, "data:", 5)) {
        const char *c = strchr(src, ',');
        if (!c || !memmem(src, (size_t)(c - src), ";base64", 7)) {
            snprintf(err, errn, "only base64 data: URLs are read");
            return -1;
        }
        if (b64_decode(c + 1, out, n)) {
            snprintf(err, errn, "the image's data: URL could not be decoded");
            return -1;
        }
        return 0;
    }
    if (!strncmp(src, "http://", 7) || !strncmp(src, "https://", 8)) {
        snprintf(err, errn, "image URLs are not fetched: send the picture as "
                            "a data: URL (base64) or a local file path");
        return -1;
    }
    const char *path = !strncmp(src, "file://", 7) ? src + 7 : src;
    if (path[0] != '/') {
        snprintf(err, errn, "an image must be a data: URL or an absolute "
                            "local file path");
        return -1;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || !S_ISREG(st.st_mode) ||
        st.st_size <= 0 || (uint64_t)st.st_size > VIS_MAX_FILE) {
        if (fd >= 0) close(fd);
        snprintf(err, errn, "the image file %.300s could not be read", path);
        return -1;
    }
    uint8_t *b = malloc((size_t)st.st_size);
    size_t o = 0;
    while (b && o < (size_t)st.st_size) {
        ssize_t k = read(fd, b + o, (size_t)st.st_size - o);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) break;
        o += (size_t)k;
    }
    close(fd);
    if (!b || o != (size_t)st.st_size) {
        free(b);
        snprintf(err, errn, "the image file %.300s could not be read", path);
        return -1;
    }
    *out = b;
    *n = o;
    return 0;
}
