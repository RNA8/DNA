#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define DNA_OK            0
#define DNA_ERR_IO       -1
#define DNA_ERR_FORMAT   -2
#define DNA_ERR_UNSUPPORTED -3
#define DNA_ERR_OOM      -4

/*
 * A runtime tensor.  Constant tensors (weights) are read-only; activation
 * tensors are read-write and owned by DnaModel.
 */
typedef struct {
    int8_t   *data;         /* INT8 values */
    int32_t   shape[6];     /* dimension sizes, outer→inner */
    int32_t   ndim;
    float     scale;        /* per-tensor activation scale */
    int32_t   zero_point;   /* per-tensor activation zero point */
    int32_t   n_elems;      /* total number of elements */
} DnaTensor;

typedef struct DnaModel DnaModel;

/* Load a quantized INT8 .tflite model.  Returns NULL on failure; call
 * dna_strerror() with the result of dna_last_error() for details. */
DnaModel  *dna_load(const char *path);
void       dna_free(DnaModel *m);

/* Number of graph inputs / outputs. */
int        dna_n_inputs(const DnaModel *m);
int        dna_n_outputs(const DnaModel *m);

/* Pointers to input/output tensors.  Fill input->data before invoking,
 * read output->data after.  Both are INT8; use scale/zero_point to convert
 * to/from float:
 *   quantize:   q = clamp(roundf(x / scale) + zero_point, -128, 127)
 *   dequantize: x = (q - zero_point) * scale
 */
DnaTensor *dna_input(DnaModel *m, int idx);
DnaTensor *dna_output(DnaModel *m, int idx);

/* Execute the graph.  Returns DNA_OK or an error code. */
int        dna_invoke(DnaModel *m);

int        dna_last_error(void);
const char *dna_strerror(int err);

#ifdef __cplusplus
}
#endif
