/*
 * INT8 GEMM for TFL_OP_FULLY_CONNECTED.
 *
 * TFLite layout:
 *   inputs[0]  activation  [B, K]  INT8  per-tensor  (s_a, za)
 *   inputs[1]  weights     [N, K]  INT8  per-channel (s_w[n], 0)
 *   inputs[2]  bias        [N]     INT32 (optional; index == -1 if absent)
 *   outputs[0] result      [B, N]  INT8  per-tensor  (s_out, z_out)
 *
 * Accumulation:
 *   acc[b,n] = bias[n]  +  sum_k (a[b,k] - za) * w[n,k]
 *   out[b,n] = clamp( round(acc * s_a*s_w[n]/s_out) + z_out, -128, 127 )
 *
 * NEON paths:
 *   baseline  – vmull_s8 + vpadalq_s16  (ARMv8.0-A, Pi 4)
 *   dotprod   – vdotq_s32               (ARMv8.2-A+dotprod, Pi 5)
 *               compiled with __attribute__((target("+dotprod")))
 *               selected at runtime via HWCAP_ASIMDDP
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "dna.h"
#include "model.h"
#include "ops.h"

#ifdef __ARM_NEON
#  include <arm_neon.h>
#  include <sys/auxv.h>
#  include <asm/hwcap.h>
#endif

/* ── Requantize a single INT32 accumulator to INT8 ───────────────────────── */

static inline int8_t requantize(int32_t acc, float scale, int32_t zp_out) {
    float f = (float)acc * scale + (float)zp_out;
    int32_t r = (int32_t)roundf(f);
    if (r < -128) r = -128;
    if (r >  127) r =  127;
    return (int8_t)r;
}

/* ── Scalar inner kernel (reference / tail) ──────────────────────────────── */

static int32_t dot_scalar(const int8_t *a, const int8_t *b, int K) {
    int32_t acc = 0;
    for (int k = 0; k < K; k++) acc += (int32_t)a[k] * (int32_t)b[k];
    return acc;
}

/* ── NEON baseline: vmull_s8 → vpadalq_s16, 4 output channels at once ───── */

#ifdef __ARM_NEON
static void gemm_4ch_neon(
    const int8_t *a,
    const int8_t *w0, const int8_t *w1,
    const int8_t *w2, const int8_t *w3,
    int K,
    int32_t *acc0, int32_t *acc1, int32_t *acc2, int32_t *acc3)
{
    int32x4_t va0 = vdupq_n_s32(0), va1 = vdupq_n_s32(0);
    int32x4_t va2 = vdupq_n_s32(0), va3 = vdupq_n_s32(0);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        int8x8_t ia = vld1_s8(a + k);
        va0 = vpadalq_s16(va0, vmull_s8(ia, vld1_s8(w0 + k)));
        va1 = vpadalq_s16(va1, vmull_s8(ia, vld1_s8(w1 + k)));
        va2 = vpadalq_s16(va2, vmull_s8(ia, vld1_s8(w2 + k)));
        va3 = vpadalq_s16(va3, vmull_s8(ia, vld1_s8(w3 + k)));
    }
    *acc0 += vaddvq_s32(va0);
    *acc1 += vaddvq_s32(va1);
    *acc2 += vaddvq_s32(va2);
    *acc3 += vaddvq_s32(va3);
    for (; k < K; k++) {
        int32_t ia = (int32_t)a[k];
        *acc0 += ia * w0[k]; *acc1 += ia * w1[k];
        *acc2 += ia * w2[k]; *acc3 += ia * w3[k];
    }
}

/* Single-channel tail used when N % 4 != 0 */
static int32_t dot_neon(const int8_t *a, const int8_t *b, int K) {
    int32x4_t vacc = vdupq_n_s32(0);
    int k = 0;
    for (; k <= K - 8; k += 8)
        vacc = vpadalq_s16(vacc, vmull_s8(vld1_s8(a+k), vld1_s8(b+k)));
    int32_t s = vaddvq_s32(vacc);
    for (; k < K; k++) s += (int32_t)a[k] * b[k];
    return s;
}

/* ── dotprod path (Pi 5, Cortex-A76+) ───────────────────────────────────── */

__attribute__((target("+dotprod")))
static void gemm_4ch_dotprod(
    const int8_t *a,
    const int8_t *w0, const int8_t *w1,
    const int8_t *w2, const int8_t *w3,
    int K,
    int32_t *acc0, int32_t *acc1, int32_t *acc2, int32_t *acc3)
{
    int32x4_t va0 = vdupq_n_s32(0), va1 = vdupq_n_s32(0);
    int32x4_t va2 = vdupq_n_s32(0), va3 = vdupq_n_s32(0);
    int k = 0;
    for (; k <= K - 16; k += 16) {
        int8x16_t ia = vld1q_s8(a + k);
        va0 = vdotq_s32(va0, ia, vld1q_s8(w0 + k));
        va1 = vdotq_s32(va1, ia, vld1q_s8(w1 + k));
        va2 = vdotq_s32(va2, ia, vld1q_s8(w2 + k));
        va3 = vdotq_s32(va3, ia, vld1q_s8(w3 + k));
    }
    *acc0 += vaddvq_s32(va0);
    *acc1 += vaddvq_s32(va1);
    *acc2 += vaddvq_s32(va2);
    *acc3 += vaddvq_s32(va3);
    /* 8-element tail */
    for (; k <= K - 8; k += 8) {
        int8x8_t ia = vld1_s8(a + k);
        *acc0 += vaddvq_s32(vpadalq_s16(vdupq_n_s32(0), vmull_s8(ia, vld1_s8(w0+k))));
        *acc1 += vaddvq_s32(vpadalq_s16(vdupq_n_s32(0), vmull_s8(ia, vld1_s8(w1+k))));
        *acc2 += vaddvq_s32(vpadalq_s16(vdupq_n_s32(0), vmull_s8(ia, vld1_s8(w2+k))));
        *acc3 += vaddvq_s32(vpadalq_s16(vdupq_n_s32(0), vmull_s8(ia, vld1_s8(w3+k))));
    }
    for (; k < K; k++) {
        int32_t ia = (int32_t)a[k];
        *acc0 += ia * w0[k]; *acc1 += ia * w1[k];
        *acc2 += ia * w2[k]; *acc3 += ia * w3[k];
    }
}

__attribute__((target("+dotprod")))
static int32_t dot_dotprod(const int8_t *a, const int8_t *b, int K) {
    int32x4_t vacc = vdupq_n_s32(0);
    int k = 0;
    for (; k <= K - 16; k += 16)
        vacc = vdotq_s32(vacc, vld1q_s8(a+k), vld1q_s8(b+k));
    int32_t s = vaddvq_s32(vacc);
    for (; k <= K - 8; k += 8)
        s += vaddvq_s32(vpadalq_s16(vdupq_n_s32(0), vmull_s8(vld1_s8(a+k), vld1_s8(b+k))));
    for (; k < K; k++) s += (int32_t)a[k] * b[k];
    return s;
}

/* ── Runtime dispatch ────────────────────────────────────────────────────── */

typedef void (*fn4ch_t)(const int8_t*,
                        const int8_t*, const int8_t*,
                        const int8_t*, const int8_t*,
                        int, int32_t*, int32_t*, int32_t*, int32_t*);
typedef int32_t (*fndot_t)(const int8_t*, const int8_t*, int);

static fn4ch_t  g_4ch  = NULL;
static fndot_t  g_dot  = NULL;

static void init_dispatch(void) {
    if (g_4ch) return;
    unsigned long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_ASIMDDP) {
        g_4ch = gemm_4ch_dotprod;
        g_dot = dot_dotprod;
    } else {
        g_4ch = gemm_4ch_neon;
        g_dot = dot_neon;
    }
}

#else  /* non-ARM fallback */

static int32_t dot_neon(const int8_t *a, const int8_t *b, int K) {
    return dot_scalar(a, b, K);
}
static void init_dispatch(void) {}
#define g_dot dot_neon

#endif  /* __ARM_NEON */

/* ── Main GEMM entry point ───────────────────────────────────────────────── */

int op_gemm(DnaModel *m, const Op *op) {
    init_dispatch();

    Tensor *ta  = &m->tensors[op->inputs[0]];   /* activations [B, K] */
    Tensor *tw  = &m->tensors[op->inputs[1]];   /* weights     [N, K] */
    Tensor *tout = &m->tensors[op->outputs[0]]; /* output      [B, N] */

    int has_bias = (op->n_inputs >= 3 && op->inputs[2] >= 0);
    Tensor *tb  = has_bias ? &m->tensors[op->inputs[2]] : NULL;

    /* Infer B, K, N from tensor shapes.
     * TFLite keeps_num_dims may make the activation [B, K] or [..., K];
     * treat everything before the last dim as batch. */
    int K = ta->shape[ta->ndim - 1];
    int N = tw->shape[0];
    int B = ta->n_elems / K;

    const int8_t  *a   = (const int8_t  *)ta->data;
    const int8_t  *w   = (const int8_t  *)tw->data;
    const int32_t *bias = tb ? (const int32_t *)tb->data : NULL;
    int8_t        *out  = (int8_t *)tout->data;

    int32_t za     = ta->quant.zero_point[0];
    int32_t z_out  = tout->quant.zero_point[0];
    float   s_a    = ta->quant.scale[0];
    float   s_out  = tout->quant.scale[0];

    /* Per-channel effective scale: s_a * s_w[n] / s_out */
    int n_scales = tw->quant.n_ch;
    float *eff_scale = malloc((size_t)N * sizeof(float));
    if (!eff_scale) return DNA_ERR_OOM;
    for (int n = 0; n < N; n++) {
        float sw = tw->quant.scale[n < n_scales ? n : n_scales - 1];
        eff_scale[n] = s_a * sw / s_out;
    }

    /* Zero-point row correction: zp_offset[n] = za * sum_k w[n,k] */
    int32_t *zp_off = NULL;
    if (za != 0) {
        zp_off = malloc((size_t)N * sizeof(int32_t));
        if (!zp_off) { free(eff_scale); return DNA_ERR_OOM; }
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)w[n*K + k];
            zp_off[n] = za * s;
        }
    }

    for (int b = 0; b < B; b++) {
        const int8_t *ab = a + b * K;
        int8_t       *ob = out + b * N;

        int n = 0;
#ifdef __ARM_NEON
        for (; n <= N - 4; n += 4) {
            int32_t acc0 = bias ? bias[n]   : 0;
            int32_t acc1 = bias ? bias[n+1] : 0;
            int32_t acc2 = bias ? bias[n+2] : 0;
            int32_t acc3 = bias ? bias[n+3] : 0;

            g_4ch(ab,
                  w+(n+0)*K, w+(n+1)*K, w+(n+2)*K, w+(n+3)*K,
                  K, &acc0, &acc1, &acc2, &acc3);

            if (zp_off) {
                acc0 -= zp_off[n];   acc1 -= zp_off[n+1];
                acc2 -= zp_off[n+2]; acc3 -= zp_off[n+3];
            }
            ob[n]   = requantize(acc0, eff_scale[n],   z_out);
            ob[n+1] = requantize(acc1, eff_scale[n+1], z_out);
            ob[n+2] = requantize(acc2, eff_scale[n+2], z_out);
            ob[n+3] = requantize(acc3, eff_scale[n+3], z_out);
        }
#endif
        for (; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;
            acc += g_dot(ab, w + n*K, K);
            if (zp_off) acc -= zp_off[n];
            ob[n] = requantize(acc, eff_scale[n], z_out);
        }
    }

    /* Apply fused activation (ReLU / ReLU6) after requantize */
    if (op->p.gemm.activation == TFL_ACT_RELU ||
        op->p.gemm.activation == TFL_ACT_RELU6) {
        int8_t ceil6 = (op->p.gemm.activation == TFL_ACT_RELU6)
            ? requantize(6 << 0, s_out, z_out) : 127;
        for (int i = 0; i < B * N; i++) {
            if (out[i] < z_out) out[i] = (int8_t)z_out;
            if (out[i] > ceil6) out[i] = ceil6;
        }
    }

    free(eff_scale);
    free(zp_off);
    return DNA_OK;
}
