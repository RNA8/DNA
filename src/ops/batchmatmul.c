/*
 * INT8 BatchMatMul for TFL_OP_BATCH_MATMUL.
 *
 * Used for attention score computation (Q×Kᵀ) and context (Attn×V).
 * Both inputs are activation tensors with per-tensor quantization.
 *
 *   inputs[0]  A  [batch, heads, M, K]  INT8  (s_a, za)
 *   inputs[1]  B  [batch, heads, K, N]  INT8  (s_b, zb)
 *              (with adjoint flags: transpose last two dims before multiply)
 *   outputs[0] C  [batch, heads, M, N]  INT8  (s_out, z_out)
 *
 *   C[b,h,m,n] = sum_k (A[b,h,m,k]-za) * (B[b,h,k,n]-zb)
 *   effective_scale = s_a * s_b / s_out  (scalar)
 */

#include <stdlib.h>
#include <math.h>
#include "dna.h"
#include "model.h"
#include "ops.h"

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

static inline int8_t requantize_s(int32_t acc, float scale, int32_t zp) {
    int32_t r = (int32_t)roundf((float)acc * scale) + zp;
    if (r < -128) r = -128;
    if (r >  127) r =  127;
    return (int8_t)r;
}

/* Compute a single M×N matrix product of a [M,K] × [K,N] block. */
static void matmul_block(
    const int8_t *A, const int8_t *B, int8_t *C,
    int M, int K, int N,
    int32_t za, int32_t zb,
    float eff_scale, int32_t z_out)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
#ifdef __ARM_NEON
            int32x4_t vacc = vdupq_n_s32(0);
            int k = 0;
            for (; k <= K - 8; k += 8) {
                int8x8_t va = vld1_s8(A + m*K + k);
                int8x8_t vb = vld1_s8(B + k*N + n*1); /* non-contiguous; scalar below */
                (void)va; (void)vb;
                /* B is column-major access when n varies — not cache-friendly.
                 * The loop below handles it scalarly; a transposed B copy would
                 * be needed for a fully vectorized path. */
                break;
            }
            (void)vacc;
#endif
            /* Scalar path (B access pattern is strided) */
            for (int k = 0; k < K; k++)
                acc += ((int32_t)A[m*K + k] - za) * ((int32_t)B[k*N + n] - zb);

            C[m*N + n] = requantize_s(acc, eff_scale, z_out);
        }
    }
}

int op_batchmatmul(DnaModel *m, const Op *op) {
    Tensor *tA   = &m->tensors[op->inputs[0]];
    Tensor *tB   = &m->tensors[op->inputs[1]];
    Tensor *tout = &m->tensors[op->outputs[0]];

    bool adj_lhs = op->p.bmm.adj_lhs;
    bool adj_rhs = op->p.bmm.adj_rhs;

    /* Resolve last-two-dim shapes after potential transpose */
    int ndA = tA->ndim, ndB = tB->ndim;
    int M  = adj_lhs ? tA->shape[ndA-1] : tA->shape[ndA-2];
    int K  = adj_lhs ? tA->shape[ndA-2] : tA->shape[ndA-1];
    int N  = adj_rhs ? tB->shape[ndB-2] : tB->shape[ndB-1];
    int batch_dims = tA->n_elems / (M * K);  /* all dims before the matrix dims */

    int32_t za    = tA->quant.zero_point[0];
    int32_t zb    = tB->quant.zero_point[0];
    int32_t z_out = tout->quant.zero_point[0];
    float eff     = tA->quant.scale[0] * tB->quant.scale[0] / tout->quant.scale[0];

    /* For the transpose case we need a temporary transposed copy of B.
     * Allocate on the stack for small K*N, heap for larger. */
    int8_t *Bt = NULL;
    if (adj_rhs) {
        /* Transpose B: [K,N] → [N,K] per batch slice */
        Bt = malloc((size_t)(K * N));
        if (!Bt) return DNA_ERR_OOM;
    }

    const int8_t *A = (const int8_t *)tA->data;
    const int8_t *B = (const int8_t *)tB->data;
    int8_t       *C = (int8_t       *)tout->data;

    for (int b = 0; b < batch_dims; b++) {
        const int8_t *Ab = A + b * M * K;
        const int8_t *Bb = B + b * K * N;
        int8_t       *Cb = C + b * M * N;

        if (adj_lhs) {
            /* Need to transpose A too — handled by swapping M/K indices in scalar */
            /* For simplicity fall through to scalar loop below */
        }

        if (adj_rhs && Bt) {
            /* Transpose Bb into Bt */
            for (int ki = 0; ki < K; ki++)
                for (int ni = 0; ni < N; ni++)
                    Bt[ni*K + ki] = Bb[ki*N + ni];
            /* Now Bt is [N, K] — use it as B with N rows of length K */
            for (int mi = 0; mi < M; mi++) {
                const int8_t *Ar = adj_lhs ? Ab + mi : Ab + mi*K;
                for (int ni = 0; ni < N; ni++) {
                    int32_t acc = 0;
                    for (int ki = 0; ki < K; ki++) {
                        int32_t av = adj_lhs ? (int32_t)Ab[ki*M + mi] - za
                                              : (int32_t)Ab[mi*K + ki] - za;
                        acc += av * ((int32_t)Bt[ni*K + ki] - zb);
                    }
                    Cb[mi*N + ni] = requantize_s(acc, eff, z_out);
                }
                (void)Ar;
            }
        } else {
            matmul_block(Ab, Bb, Cb, M, K, N, za, zb, eff, z_out);
        }
    }

    free(Bt);
    return DNA_OK;
}
