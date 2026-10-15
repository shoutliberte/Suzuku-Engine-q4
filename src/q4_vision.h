/* Images: the encoder helper and its cache (src/q4_vision.c). */
#ifndef Q4_VISION_H
#define Q4_VISION_H

#include <stddef.h>
#include <stdint.h>

typedef struct q4_vis q4_vis;

/* NULL unless Q4_VISION_MMPROJ names the encoder (vision off). */
q4_vis *q4_vis_open(const char *model, uint32_t n_embd);
void q4_vis_close(q4_vis *v);
/* Start of a request: its encodings stay cached until the next begin. */
void q4_vis_begin(q4_vis *v);
/* Image bytes -> rows (owned by the cache). 0 on success. */
int q4_vis_encode(q4_vis *v, const uint8_t *data, size_t n, q4_image *out,
                  char *err, size_t errn);
/* An image part's source: base64 (is_b64), a data: URL or a local path. */
int q4_vis_load(const char *src, int is_b64, uint8_t **out, size_t *n,
                char *err, size_t errn);

#endif
