#pragma once
#include "model.h"

/* Each op function reads inputs and writes outputs via model tensor pointers.
 * Returns DNA_OK or an error code. */

int op_gemm        (DnaModel *m, const Op *op);
int op_batchmatmul (DnaModel *m, const Op *op);
int op_softmax     (DnaModel *m, const Op *op);
int op_layernorm   (DnaModel *m, const Op *op);
int op_gelu        (DnaModel *m, const Op *op);
int op_conv2d      (DnaModel *m, const Op *op);
int op_elemwise    (DnaModel *m, const Op *op);  /* ADD, MUL */
int op_mean        (DnaModel *m, const Op *op);
int op_reshape     (DnaModel *m, const Op *op);
int op_transpose   (DnaModel *m, const Op *op);
int op_quantize    (DnaModel *m, const Op *op);
int op_dequantize  (DnaModel *m, const Op *op);
