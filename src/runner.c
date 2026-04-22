#include <stdio.h>
#include "dna.h"
#include "model.h"
#include "ops.h"

int dna_invoke(DnaModel *m) {
    for (int i = 0; i < m->n_ops; i++) {
        Op *op = &m->ops[i];
        int err = DNA_OK;

        switch (op->op) {
            case TFL_OP_FULLY_CONNECTED:
                err = op_gemm(m, op);
                break;

            case TFL_OP_BATCH_MATMUL:
                err = op_batchmatmul(m, op);
                break;

            case TFL_OP_SOFTMAX:
                err = op_softmax(m, op);
                break;

            case TFL_OP_LAYER_NORM:
                err = op_layernorm(m, op);
                break;

            case TFL_OP_GELU:
                err = op_gelu(m, op);
                break;

            case TFL_OP_CONV_2D:
                err = op_conv2d(m, op);
                break;

            case TFL_OP_ADD:
            case TFL_OP_MUL:
                err = op_elemwise(m, op);
                break;

            case TFL_OP_MEAN:
                err = op_mean(m, op);
                break;

            case TFL_OP_RESHAPE:
                err = op_reshape(m, op);
                break;

            case TFL_OP_TRANSPOSE:
                err = op_transpose(m, op);
                break;

            case TFL_OP_QUANTIZE:
                err = op_quantize(m, op);
                break;

            case TFL_OP_DEQUANTIZE:
                err = op_dequantize(m, op);
                break;

            case TFL_OP_RELU:
            case TFL_OP_RELU6: {
                /* In-place activation on the single input/output tensor */
                Tensor *t = &m->tensors[op->inputs[0]];
                Tensor *to = &m->tensors[op->outputs[0]];
                int8_t *d = (int8_t *)to->data;
                const int8_t *s = (const int8_t *)t->data;
                int8_t lo = (int8_t)t->quant.zero_point[0];
                /* ReLU6: upper bound at float 6.0 requantized */
                int8_t hi = 127;
                if (op->op == TFL_OP_RELU6) {
                    float f6 = 6.0f / to->quant.scale[0] + to->quant.zero_point[0];
                    hi = f6 > 127.0f ? 127 : (int8_t)(int)f6;
                }
                for (int j = 0; j < to->n_elems; j++) {
                    int8_t v = s[j];
                    if (v < lo) v = lo;
                    if (v > hi) v = hi;
                    d[j] = v;
                }
                break;
            }

            default:
                fprintf(stderr, "dna: unsupported op %d at node %d\n", op->op, i);
                return DNA_ERR_UNSUPPORTED;
        }

        if (err != DNA_OK) return err;
    }
    return DNA_OK;
}
