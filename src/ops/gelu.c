/*
 * GELU activation for TFL_OP_GELU.
 *
 * Uses a 256-entry lookup table (one entry per INT8 value) computed at first
 * call so the hot path is a single table lookup per element.
 *
 * GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
 *
 *   inputs[0]  x  [...]  INT8  (s_in,  z_in)
 *   outputs[0] y  [...]  INT8  (s_out, z_out)
 */

#include <math.h>
#include <string.h>
#include "dna.h"
#include "model.h"
#include "ops.h"

#define SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEF      0.044715f

static inline float gelu_f(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline int8_t clamp_i8(int32_t v) {
    return v < -128 ? -128 : v > 127 ? 127 : (int8_t)v;
}

/*
 * Build the LUT for a given (s_in, z_in, s_out, z_out) combination.
 * lut[i] maps INT8 value (i - 128) to the quantized GELU output.
 * Index 0 → input value -128, index 255 → input value 127.
 */
static void build_lut(int8_t lut[256],
                      float s_in, int32_t z_in,
                      float s_out, int32_t z_out) {
    for (int i = 0; i < 256; i++) {
        int32_t q_in = i - 128;  /* signed int8 value */
        float x  = ((float)q_in - z_in) * s_in;
        float y  = gelu_f(x);
        int32_t q = (int32_t)roundf(y / s_out) + z_out;
        lut[i] = clamp_i8(q);
    }
}

int op_gelu(DnaModel *m, const Op *op) {
    Tensor *tin  = &m->tensors[op->inputs[0]];
    Tensor *tout = &m->tensors[op->outputs[0]];

    int8_t lut[256];
    build_lut(lut,
              tin->quant.scale[0],  tin->quant.zero_point[0],
              tout->quant.scale[0], tout->quant.zero_point[0]);

    const int8_t *src = (const int8_t *)tin->data;
    int8_t       *dst = (int8_t *)tout->data;
    int            n  = tin->n_elems;

    for (int i = 0; i < n; i++)
        dst[i] = lut[(uint8_t)src[i]];  /* (uint8_t) re-indexes -128..127 → 0..255 */

    return DNA_OK;
}
