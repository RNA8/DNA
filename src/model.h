#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "tflite_schema.h"

/* ── Quantization ────────────────────────────────────────────────────────── */

/*
 * Per-channel weight quantization or per-tensor activation quantization.
 * n_ch == 1  →  per-tensor  (scale[0], zp[0])
 * n_ch  > 1  →  per-channel (scale[c], zp[c] for output channel c)
 */
typedef struct {
    float   *scale;
    int32_t *zero_point;
    int32_t  n_ch;
    int32_t  quant_dim;  /* which tensor dimension is the channel axis */
} Quant;

/* ── Tensor ──────────────────────────────────────────────────────────────── */

typedef struct {
    const char    *name;       /* points into mmap'd file */
    int32_t        shape[6];
    int32_t        ndim;
    TflTensorType  type;
    Quant          quant;
    void          *data;       /* weight: points into mmap; activation: scratch */
    int32_t        n_elems;
    bool           is_const;   /* true for weight tensors */
} Tensor;

static inline int32_t tensor_bytes(const Tensor *t) {
    int32_t bytes_per = (t->type == TFL_INT8 || t->type == TFL_UINT8) ? 1 :
                        (t->type == TFL_INT32 || t->type == TFL_FLOAT32) ? 4 :
                        (t->type == TFL_INT64) ? 8 : 1;
    return t->n_elems * bytes_per;
}

/* ── Op parameters ───────────────────────────────────────────────────────── */

typedef struct {
    TflActivation activation;
} GemmParams;

typedef struct {
    bool adj_lhs;
    bool adj_rhs;
} BatchMatMulParams;

typedef struct {
    TflPadding    padding;
    int32_t       stride_w, stride_h;
    TflActivation activation;
} Conv2DParams;

typedef struct {
    float beta;
} SoftmaxParams;

typedef struct {
    int32_t axes[4];
    int32_t n_axes;
    bool    keep_dims;
} ReduceParams;

/* ── Operator node ───────────────────────────────────────────────────────── */

typedef struct {
    TflOpCode  op;
    int32_t   *inputs;    /* tensor indices into Model.tensors */
    int32_t    n_inputs;
    int32_t   *outputs;
    int32_t    n_outputs;
    union {
        GemmParams      gemm;
        BatchMatMulParams bmm;
        Conv2DParams    conv2d;
        SoftmaxParams   softmax;
        ReduceParams    reduce;
    } p;
} Op;

/* ── Model ───────────────────────────────────────────────────────────────── */

struct DnaModel {
    Tensor    *tensors;
    int32_t    n_tensors;

    Op        *ops;
    int32_t    n_ops;

    int32_t   *inputs;    /* indices into tensors[] */
    int32_t    n_inputs;
    int32_t   *outputs;
    int32_t    n_outputs;

    /* backing memory */
    uint8_t   *file_data;
    size_t     file_size;
    void      *scratch;      /* activation buffers */
    size_t     scratch_size;

    /* heap blocks to free (quant arrays, op input/output index arrays) */
    void     **allocs;
    int32_t    n_allocs;
};
