/*
 * Softmax for TFL_OP_SOFTMAX.
 *
 * Dequantize → float softmax → requantize.
 * Attention matrices are small (typically ≤196×196 tokens), so float
 * arithmetic here is not a bottleneck.
 *
 *   inputs[0]  logits  [..., N]  INT8  (s_in,  z_in)
 *   outputs[0] probs   [..., N]  INT8  (s_out, z_out)
 *
 * Softmax is applied over the last dimension.
 */

#include <math.h>
#include <float.h>
#include "dna.h"
#include "model.h"
#include "ops.h"

static inline int8_t clamp_i8(int32_t v) {
    if (v < -128) return -128;
    if (v >  127) return  127;
    return (int8_t)v;
}

int op_softmax(DnaModel *m, const Op *op) {
    Tensor *tin  = &m->tensors[op->inputs[0]];
    Tensor *tout = &m->tensors[op->outputs[0]];

    int N = tin->shape[tin->ndim - 1];
    int rows = tin->n_elems / N;

    float s_in  = tin->quant.scale[0];
    int32_t z_in = tin->quant.zero_point[0];
    float s_out  = tout->quant.scale[0];
    int32_t z_out = tout->quant.zero_point[0];
    float beta = op->p.softmax.beta;

    const int8_t *src = (const int8_t *)tin->data;
    int8_t       *dst = (int8_t *)tout->data;

    for (int r = 0; r < rows; r++) {
        const int8_t *row_in  = src + r * N;
        int8_t       *row_out = dst + r * N;

        /* Find max for numerical stability */
        float maxv = -FLT_MAX;
        for (int i = 0; i < N; i++) {
            float f = ((float)row_in[i] - z_in) * s_in;
            if (f > maxv) maxv = f;
        }

        /* exp and sum */
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float f = ((float)row_in[i] - z_in) * s_in;
            /* reuse row_out as a float buffer isn't safe since it's int8;
             * compute in two passes */
            sum += expf((f - maxv) * beta);
        }

        /* Quantize output */
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < N; i++) {
            float f   = ((float)row_in[i] - z_in) * s_in;
            float p   = expf((f - maxv) * beta) * inv_sum;
            int32_t q = (int32_t)roundf(p / s_out) + z_out;
            row_out[i] = clamp_i8(q);
        }
    }
    return DNA_OK;
}
