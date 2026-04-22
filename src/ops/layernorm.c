/*
 * Layer normalization for TFL_OP_LAYER_NORM.
 *
 * TFLite may export LayerNorm as either:
 *   (a) A single LAYER_NORMALIZATION op  [handled here]
 *   (b) A sequence of MEAN/SUB/MUL/RSQRT ops [each handled individually]
 *
 * Inputs:
 *   inputs[0]  x       [..., C]  INT8   (s_in,  z_in)
 *   inputs[1]  gamma   [C]       FLOAT32 or INT8 scale
 *   inputs[2]  beta    [C]       FLOAT32 or INT8 offset
 * Output:
 *   outputs[0] y       [..., C]  INT8   (s_out, z_out)
 *
 * Normalization runs in float over the last (channel) dimension.
 * Mean and variance are computed per row in INT32 accumulators, then
 * converted to float for the normalization step.
 */

#include <math.h>
#include "dna.h"
#include "model.h"
#include "ops.h"

#define LN_EPS 1e-5f

static inline int8_t clamp_i8(int32_t v) {
    return v < -128 ? -128 : v > 127 ? 127 : (int8_t)v;
}

int op_layernorm(DnaModel *m, const Op *op) {
    Tensor *tx   = &m->tensors[op->inputs[0]];
    Tensor *tout = &m->tensors[op->outputs[0]];

    int C    = tx->shape[tx->ndim - 1];
    int rows = tx->n_elems / C;

    float s_in  = tx->quant.scale[0];
    int32_t z_in = tx->quant.zero_point[0];
    float s_out  = tout->quant.scale[0];
    int32_t z_out = tout->quant.zero_point[0];

    /* gamma and beta: optional, may be float or int8 */
    const float   *gamma_f = NULL;
    const float   *beta_f  = NULL;
    const int8_t  *gamma_q = NULL;
    const int8_t  *beta_q  = NULL;
    float gamma_scale = 1.0f, beta_scale = 1.0f;

    if (op->n_inputs >= 2 && op->inputs[1] >= 0) {
        Tensor *tg = &m->tensors[op->inputs[1]];
        if (tg->type == TFL_FLOAT32) gamma_f = (const float *)tg->data;
        else { gamma_q = (const int8_t *)tg->data; gamma_scale = tg->quant.scale[0]; }
    }
    if (op->n_inputs >= 3 && op->inputs[2] >= 0) {
        Tensor *tb = &m->tensors[op->inputs[2]];
        if (tb->type == TFL_FLOAT32) beta_f = (const float *)tb->data;
        else { beta_q = (const int8_t *)tb->data; beta_scale = tb->quant.scale[0]; }
    }

    const int8_t *src = (const int8_t *)tx->data;
    int8_t       *dst = (int8_t *)tout->data;

    for (int r = 0; r < rows; r++) {
        const int8_t *row = src + r * C;
        int8_t       *out = dst + r * C;

        /* Mean (in float, accumulated from dequantized values) */
        float mean = 0.0f;
        for (int c = 0; c < C; c++)
            mean += ((float)row[c] - z_in) * s_in;
        mean /= (float)C;

        /* Variance */
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = ((float)row[c] - z_in) * s_in - mean;
            var += d * d;
        }
        var /= (float)C;
        float inv_std = 1.0f / sqrtf(var + LN_EPS);

        /* Normalize, scale, shift, requantize */
        for (int c = 0; c < C; c++) {
            float x  = (((float)row[c] - z_in) * s_in - mean) * inv_std;
            float g  = gamma_f ? gamma_f[c]
                     : gamma_q ? ((float)gamma_q[c] * gamma_scale)
                     : 1.0f;
            float b  = beta_f  ? beta_f[c]
                     : beta_q  ? ((float)beta_q[c] * beta_scale)
                     : 0.0f;
            float y  = x * g + b;
            int32_t q = (int32_t)roundf(y / s_out) + z_out;
            out[c] = clamp_i8(q);
        }
    }
    return DNA_OK;
}
