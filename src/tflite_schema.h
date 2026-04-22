#pragma once
/*
 * TFLite FlatBuffer schema constants.
 *
 * Field indices match declaration order in schema.fbs (0-based).
 * Union fields occupy two consecutive indices: [N]=type, [N+1]=value.
 *
 * Source: tensorflow/lite/schema/schema.fbs (schema version 3)
 */

/* ── TensorType ──────────────────────────────────────────────────────────── */
typedef enum {
    TFL_FLOAT32  = 0,
    TFL_FLOAT16  = 1,
    TFL_INT32    = 2,
    TFL_UINT8    = 3,
    TFL_INT64    = 4,
    TFL_INT8     = 9,
    TFL_INT16    = 7,
    TFL_BOOL     = 6,
} TflTensorType;

/* ── BuiltinOperator ─────────────────────────────────────────────────────── */
typedef enum {
    TFL_OP_ADD              = 0,
    TFL_OP_CONV_2D          = 3,
    TFL_OP_DEQUANTIZE       = 6,
    TFL_OP_FULLY_CONNECTED  = 9,
    TFL_OP_MUL              = 18,
    TFL_OP_RELU             = 19,
    TFL_OP_RELU6            = 24,
    TFL_OP_RESHAPE          = 22,
    TFL_OP_SOFTMAX          = 25,
    TFL_OP_TRANSPOSE        = 39,
    TFL_OP_MEAN             = 40,
    TFL_OP_QUANTIZE         = 114,
    TFL_OP_BATCH_MATMUL     = 126,
    TFL_OP_GELU             = 137,
    TFL_OP_LAYER_NORM       = 149,
} TflOpCode;

/* ── ActivationFunctionType ──────────────────────────────────────────────── */
typedef enum {
    TFL_ACT_NONE   = 0,
    TFL_ACT_RELU   = 1,
    TFL_ACT_RELU6  = 3,
} TflActivation;

/* ── Padding ─────────────────────────────────────────────────────────────── */
typedef enum {
    TFL_PAD_SAME  = 0,
    TFL_PAD_VALID = 1,
} TflPadding;

/* ════════════════════════════════════════════════════════════════════════════
 * Field indices (field_id argument to fb_field / fb_fi32 / …)
 * ════════════════════════════════════════════════════════════════════════════ */

/* Model */
#define TFL_MODEL_operator_codes  1
#define TFL_MODEL_subgraphs       2
#define TFL_MODEL_buffers         4

/* OperatorCode */
#define TFL_OPCODE_deprecated_builtin  0   /* int8,  codes 0–126 */
#define TFL_OPCODE_custom_code         1
#define TFL_OPCODE_builtin_code        3   /* int32, authoritative */

/* SubGraph */
#define TFL_SG_tensors    0
#define TFL_SG_inputs     1
#define TFL_SG_outputs    2
#define TFL_SG_operators  3

/* Tensor */
#define TFL_TENSOR_shape         0
#define TFL_TENSOR_type          1   /* TflTensorType as int8 */
#define TFL_TENSOR_buffer        2   /* index into Model.buffers */
#define TFL_TENSOR_name          3
#define TFL_TENSOR_quantization  4

/* QuantizationParameters */
#define TFL_QUANT_scale              2
#define TFL_QUANT_zero_point         3
#define TFL_QUANT_quantized_dimension 6  /* after union fields 4,5 */

/* Buffer */
#define TFL_BUFFER_data  0

/* Operator */
#define TFL_OP_opcode_index   0
#define TFL_OP_inputs         1
#define TFL_OP_outputs        2
/* fields 3 (union type) and 4 (union value) = builtin_options */
#define TFL_OP_builtin_opts   4

/* FullyConnectedOptions */
#define TFL_FC_fused_activation  0

/* Conv2DOptions */
#define TFL_CONV_padding          0
#define TFL_CONV_stride_w         1
#define TFL_CONV_stride_h         2
#define TFL_CONV_fused_activation 3

/* BatchMatMulOptions */
#define TFL_BMM_adjoint_lhs  0
#define TFL_BMM_adjoint_rhs  1

/* SoftmaxOptions */
#define TFL_SOFTMAX_beta  0

/* ReducerOptions (MEAN) */
#define TFL_REDUCER_keep_dims  0
