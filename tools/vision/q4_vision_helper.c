/* q4-vision: the image encoder q4's server starts for pictures.
 *
 * llama.cpp's mtmd library with the model's mmproj file (vision encoder +
 * projector) turns an image into the rows the text model reads in place
 * of its <|image_pad|> cells; q4 places them and gives them their M-RoPE
 * positions.  CPU only: the encoder's weights and buffers stay in DRAM.
 *
 *   q4-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split>
 *             [--threads N] [--max-tokens N] [--min-tokens N]
 *             [--flash-attn on|off|auto]
 *
 * Resident: prints "READY <n_embd>", then per stdin line
 *   ENC <image path> <output path>  ->  "OK <n> <nx> <ny> <ms>" or "ERR <message>"
 *   QUIT
 * Output file: int32 {'SVE1', n, nx, ny, n_embd}, then float32 [n][n_embd],
 * row i at grid (x = i % nx, y = i / nx).  (The protocol and file are
 * Strata's strata-vision's, so either helper serves either engine.)
 * The text model is opened vocab-only. */
#define _GNU_SOURCE
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void quiet_log(enum ggml_log_level level, const char *text, void *u) {
    (void)u;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static void reply(const char *fmt, const char *a) {
    printf(fmt, a);
    fflush(stdout);
}

static int encode(mtmd_context *ctx, int n_embd, const char *img,
                  const char *out, char *err, size_t errn) {
    const double t0 = now_ms();
    struct mtmd_helper_bitmap_wrapper bw =
        mtmd_helper_bitmap_init_from_file(ctx, img, false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) {
        snprintf(err, errn, "the image could not be decoded (JPEG, PNG, BMP, "
                            "GIF, TGA and PNM are read)");
        return -1;
    }
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const char *marker = mtmd_default_marker();
    mtmd_input_text txt = {marker, strlen(marker), false, true};
    const mtmd_bitmap *bm[1] = {bw.bitmap};
    const mtmd_input_chunk *ic = NULL;
    int rc = -1;
    if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) {
        snprintf(err, errn, "the image could not be preprocessed");
        goto done;
    }
    for (size_t c = 0; c < mtmd_input_chunks_size(chunks); c++) {
        const mtmd_input_chunk *ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) ic = ch;
    }
    if (!ic) {
        snprintf(err, errn, "no image chunk");
        goto done;
    }
    if (mtmd_encode_chunk(ctx, ic) != 0) {
        snprintf(err, errn, "the vision encoder failed");
        goto done;
    }
    const mtmd_image_tokens *it = mtmd_input_chunk_get_tokens_image(ic);
    const int n = (int)mtmd_input_chunk_get_n_tokens(ic);
    /* the grid from the last token's decoder position */
    const struct mtmd_decoder_pos last =
        mtmd_image_tokens_get_decoder_pos(it, 0, (size_t)n - 1);
    const int nx = (int)last.x + 1, ny = (int)last.y + 1;
    if (nx * ny != n) {
        snprintf(err, errn, "the image grid is not rectangular (%d tokens)", n);
        goto done;
    }
    const float *embd = mtmd_get_output_embd(ctx);
    FILE *f = fopen(out, "wb");
    if (!f) {
        snprintf(err, errn, "cannot write %s", out);
        goto done;
    }
    const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
    const int ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
                   fwrite(embd, sizeof(float) * (size_t)n_embd, (size_t)n, f) == (size_t)n;
    if (fclose(f) != 0 || !ok) {
        snprintf(err, errn, "short write to %s", out);
        goto done;
    }
    printf("OK %d %d %d %.0f\n", n, nx, ny, now_ms() - t0);
    fflush(stdout);
    rc = 0;
done:
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
    return rc;
}

int main(int argc, char **argv) {
    const char *mmproj = NULL, *model = NULL;
    int threads = 0, max_tokens = 0, min_tokens = 0, fa_given = 0;
    enum llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (i + 1 >= argc) {
            fprintf(stderr, "%s needs a value\n", a);
            return 2;
        }
        const char *v = argv[++i];
        if (!strcmp(a, "--mmproj")) mmproj = v;
        else if (!strcmp(a, "--model")) model = v;
        else if (!strcmp(a, "--threads")) threads = atoi(v);
        else if (!strcmp(a, "--max-tokens")) max_tokens = atoi(v);
        else if (!strcmp(a, "--min-tokens")) min_tokens = atoi(v);
        else if (!strcmp(a, "--flash-attn")) {
            fa = !strcmp(v, "on")    ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                 : !strcmp(v, "off") ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                     : LLAMA_FLASH_ATTN_TYPE_AUTO;
            fa_given = 1;
        } else {
            fprintf(stderr, "unknown argument %s\n", a);
            return 2;
        }
    }
    if (!mmproj || !model) {
        fprintf(stderr, "usage: q4-vision --mmproj <mmproj.gguf> --model <model.gguf> "
                        "[--threads N] [--max-tokens N] [--min-tokens N] "
                        "[--flash-attn on|off|auto]\n");
        return 2;
    }
    llama_log_set(quiet_log, NULL);
    mtmd_helper_log_set(quiet_log, NULL);
    llama_backend_init();
    const double t0 = now_ms();
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    struct llama_model *text = llama_model_load_from_file(model, mp);
    if (!text) {
        reply("ERR cannot open the text model %s\n", model);
        return 1;
    }
    struct mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.print_timings = false;
    cp.warmup = false;
    /* ggml's tiled CPU flash attention needs the head size (72 here) to be
     * a multiple of the vector width: 16 floats under AVX-512, where it
     * falls back to a slow FP16-accumulating kernel (Strata #660). */
    if (!fa_given && ggml_cpu_has_avx512()) fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.flash_attn_type = fa;
    if (threads <= 0) {
        long c = sysconf(_SC_NPROCESSORS_ONLN) / 2; /* one per core */
        threads = c > 0 ? (int)c : 4;
    }
    cp.n_threads = threads;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    if (min_tokens > 0) cp.image_min_tokens = min_tokens;
    mtmd_context *ctx = mtmd_init_from_file(mmproj, text, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        reply("ERR cannot load the vision encoder %s\n", mmproj);
        return 1;
    }
    /* vocab-only: the width comes from the projector */
    int n_embd = 0;
    {
        struct gguf_init_params gp = {true, NULL};
        struct gguf_context *gg = gguf_init_from_file(mmproj, gp);
        const int64_t k = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
        if (k >= 0) n_embd = (int)gguf_get_val_u32(gg, k);
        if (gg) gguf_free(gg);
    }
    if (n_embd <= 0) {
        reply("ERR the vision encoder has no projection_dim%s\n", "");
        return 1;
    }
    fprintf(stderr, "q4-vision: loaded in %.1f s, CPU, %d threads, flash attention %s\n",
            (now_ms() - t0) / 1e3, threads,
            fa == LLAMA_FLASH_ATTN_TYPE_DISABLED ? "off"
            : fa == LLAMA_FLASH_ATTN_TYPE_ENABLED ? "on" : "auto");
    printf("READY %d\n", n_embd);
    fflush(stdout);

    char *line = NULL;
    size_t cap = 0;
    ssize_t L;
    while ((L = getline(&line, &cap, stdin)) >= 0) {
        while (L > 0 && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!strcmp(line, "QUIT")) break;
        /* "ENC <image> <out>": the output path is the last field */
        char *sp = strncmp(line, "ENC ", 4) ? NULL : strrchr(line + 4, ' ');
        if (!sp || sp == line + 4 || !sp[1]) {
            reply("ERR expected: ENC <image> <output>%s\n", "");
            continue;
        }
        *sp = 0;
        char err[512];
        if (encode(ctx, n_embd, line + 4, sp + 1, err, sizeof(err)))
            reply("ERR %s\n", err);
    }
    free(line);
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
