/*
 * INT8 Conv2D for TFL_OP_CONV_2D.
 *
 * Used for patch embedding (e.g., 16×16 stride-16 kernel in DeiT, or small
 * conv stems in MobileViT/TinyViT).  Weight layout follows TFLite:
 *   weights  [out_ch, kH, kW, in_ch]  INT8  per-channel
 *   bias     [out_ch]                 INT32 (optional)
 *   input    [batch, H,  W,  in_ch]   INT8  per-tensor  (NHWC)
 *   output   [batch, oH, oW, out_ch]  INT8  per-tensor
 *
 * The inner loop is a direct (im2col-free) convolution; for a pure patch
 * embedding with stride == kernel size, this reduces to a single GEMM and is
 * fast enough.  General strided / padded conv is handled scalarly.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dna.h"
#include "model.h"
#include "ops.h"

static inline int8_t clamp_i8(int32_t v) {
    return v < -128 ? -128 : v > 127 ? 127 : (int8_t)v;
}

static int out_dim(int in, int k, int stride, TflPadding pad) {
    if (pad == TFL_PAD_SAME)  return (in + stride - 1) / stride;
    return (in - k) / stride + 1;
}

int op_conv2d(DnaModel *m, const Op *op) {
    Tensor *tin  = &m->tensors[op->inputs[0]];
    Tensor *tw   = &m->tensors[op->inputs[1]];
    Tensor *tout = &m->tensors[op->outputs[0]];

    int has_bias = (op->n_inputs >= 3 && op->inputs[2] >= 0);
    const int32_t *bias = has_bias
        ? (const int32_t *)m->tensors[op->inputs[2]].data : NULL;

    int B    = tin->shape[0];
    int iH   = tin->shape[1], iW = tin->shape[2], iC = tin->shape[3];
    int oC   = tw->shape[0],  kH = tw->shape[1],  kW = tw->shape[2];
    /* tw->shape[3] == iC */

    int sw   = op->p.conv2d.stride_w;
    int sh   = op->p.conv2d.stride_h;
    int oH   = out_dim(iH, kH, sh, op->p.conv2d.padding);
    int oW   = out_dim(iW, kW, sw, op->p.conv2d.padding);

    int pad_top = 0, pad_left = 0;
    if (op->p.conv2d.padding == TFL_PAD_SAME) {
        int ph = (oH - 1)*sh + kH - iH; if (ph < 0) ph = 0;
        int pw = (oW - 1)*sw + kW - iW; if (pw < 0) pw = 0;
        pad_top  = ph / 2;
        pad_left = pw / 2;
    }

    float s_a   = tin->quant.scale[0];
    int32_t za  = tin->quant.zero_point[0];
    float s_out = tout->quant.scale[0];
    int32_t z_out = tout->quant.zero_point[0];

    int n_scales = tw->quant.n_ch;
    float *eff = malloc((size_t)oC * sizeof(float));
    if (!eff) return DNA_ERR_OOM;
    for (int oc = 0; oc < oC; oc++) {
        float sw_c = tw->quant.scale[oc < n_scales ? oc : n_scales - 1];
        eff[oc] = s_a * sw_c / s_out;
    }

    const int8_t *inp = (const int8_t *)tin->data;
    const int8_t *w   = (const int8_t *)tw->data;
    int8_t       *out = (int8_t *)tout->data;

    for (int b = 0; b < B; b++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                for (int oc = 0; oc < oC; oc++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh*sh + kh - pad_top;
                        if (ih < 0 || ih >= iH) continue;
                        for (int kw2 = 0; kw2 < kW; kw2++) {
                            int iw = ow*sw + kw2 - pad_left;
                            if (iw < 0 || iw >= iW) continue;
                            for (int ic = 0; ic < iC; ic++) {
                                int32_t a_v = (int32_t)inp[((b*iH+ih)*iW+iw)*iC+ic] - za;
                                int32_t w_v = (int32_t)w[((oc*kH+kh)*kW+kw2)*iC+ic];
                                acc += a_v * w_v;
                            }
                        }
                    }
                    int idx = ((b*oH+oh)*oW+ow)*oC+oc;
                    out[idx] = clamp_i8((int32_t)roundf((float)acc * eff[oc]) + z_out);
                }
            }
        }
    }

    free(eff);
    return DNA_OK;
}
