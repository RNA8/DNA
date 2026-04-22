#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#include "dna.h"
#include "model.h"
#include "fb.h"
#include "tflite_schema.h"

/* ── Error state ─────────────────────────────────────────────────────────── */

static int g_last_err = DNA_OK;

static int fail(int code) { g_last_err = code; return code; }

int        dna_last_error(void) { return g_last_err; }
const char *dna_strerror(int e) {
    switch (e) {
        case DNA_OK:              return "ok";
        case DNA_ERR_IO:          return "I/O error";
        case DNA_ERR_FORMAT:      return "invalid TFLite file";
        case DNA_ERR_UNSUPPORTED: return "unsupported op or format";
        case DNA_ERR_OOM:         return "out of memory";
        default:                  return "unknown error";
    }
}

/* ── Allocation tracker ──────────────────────────────────────────────────── */

static void *model_alloc(struct DnaModel *m, size_t n) {
    void *p = calloc(1, n);
    if (!p) return NULL;
    void **a = realloc(m->allocs, (size_t)(m->n_allocs + 1) * sizeof(void *));
    if (!a) { free(p); return NULL; }
    m->allocs = a;
    m->allocs[m->n_allocs++] = p;
    return p;
}

/* ── Quantization parsing ────────────────────────────────────────────────── */

static int parse_quant(struct DnaModel *m, const uint8_t *qt, Quant *q) {
    if (!qt) {
        /* No quantization info — treat as unquantized (scale=1, zp=0). */
        q->n_ch = 1;
        q->scale      = model_alloc(m, sizeof(float));
        q->zero_point = model_alloc(m, sizeof(int32_t));
        if (!q->scale || !q->zero_point) return fail(DNA_ERR_OOM);
        q->scale[0] = 1.0f;
        q->zero_point[0] = 0;
        return DNA_OK;
    }

    const uint8_t *fp;
    uint32_t n = 0;

    /* scale vector */
    fp = fb_field(qt, TFL_QUANT_scale);
    if (fp) {
        const uint8_t *sv = fb_vec(fp, &n);
        q->n_ch = (int32_t)n;
        q->scale = model_alloc(m, n * sizeof(float));
        if (!q->scale) return fail(DNA_ERR_OOM);
        for (uint32_t i = 0; i < n; i++)
            memcpy(&q->scale[i], sv + i * 4, 4);
    } else {
        q->n_ch = 1;
        q->scale = model_alloc(m, sizeof(float));
        if (!q->scale) return fail(DNA_ERR_OOM);
        q->scale[0] = 1.0f;
    }

    /* zero_point vector (stored as int64 in the schema) */
    fp = fb_field(qt, TFL_QUANT_zero_point);
    uint32_t nzp = 0;
    if (fp) {
        const uint8_t *zv = fb_vec(fp, &nzp);
        q->zero_point = model_alloc(m, (nzp ? nzp : 1) * sizeof(int32_t));
        if (!q->zero_point) return fail(DNA_ERR_OOM);
        for (uint32_t i = 0; i < nzp; i++) {
            int64_t v; memcpy(&v, zv + i * 8, 8);
            q->zero_point[i] = (int32_t)v;
        }
        if (nzp == 0) q->zero_point[0] = 0;
    } else {
        q->zero_point = model_alloc(m, sizeof(int32_t));
        if (!q->zero_point) return fail(DNA_ERR_OOM);
        q->zero_point[0] = 0;
    }

    q->quant_dim = fb_fi32(qt, TFL_QUANT_quantized_dimension, 0);
    return DNA_OK;
}

/* ── Tensor parsing ──────────────────────────────────────────────────────── */

static int parse_tensor(struct DnaModel *m,
                        const uint8_t *t_fb,
                        const uint8_t *buffers_fp, uint32_t n_buffers,
                        Tensor *t) {
    /* shape */
    const uint8_t *sfp = fb_field(t_fb, TFL_TENSOR_shape);
    t->ndim = 0;
    if (sfp) {
        uint32_t nd;
        const uint8_t *sv = fb_vec(sfp, &nd);
        t->ndim = (int32_t)nd;
        if (nd > 6) return fail(DNA_ERR_FORMAT);
        for (uint32_t i = 0; i < nd; i++)
            t->shape[i] = fb_i32(sv + i * 4);
    }

    /* elem count */
    t->n_elems = 1;
    for (int i = 0; i < t->ndim; i++) t->n_elems *= t->shape[i];

    /* type */
    t->type = (TflTensorType)fb_i8(t_fb, TFL_TENSOR_type, TFL_INT8);

    /* name */
    const uint8_t *nfp = fb_field(t_fb, TFL_TENSOR_name);
    t->name = nfp ? fb_str(nfp) : "";

    /* quantization */
    const uint8_t *qfp = fb_field(t_fb, TFL_TENSOR_quantization);
    const uint8_t *qt  = qfp ? fb_table(qfp) : NULL;
    int err = parse_quant(m, qt, &t->quant);
    if (err) return err;

    /* data: look up in the buffer list */
    uint32_t buf_idx = fb_fu32(t_fb, TFL_TENSOR_buffer, 0);
    t->is_const = false;
    t->data = NULL;

    if (buf_idx > 0 && buf_idx < n_buffers) {
        /* The buffers vector contains offsets to Buffer tables. Each element
         * in the vector is a 4-byte offset from its own position. */
        const uint8_t *buf_elem = buffers_fp + buf_idx * 4;
        const uint8_t *buf_tbl  = fb_table(buf_elem);
        const uint8_t *dfp      = fb_field(buf_tbl, TFL_BUFFER_data);
        if (dfp) {
            uint32_t dlen;
            const uint8_t *dptr = fb_vec(dfp, &dlen);
            if (dlen > 0) {
                t->data     = (void *)dptr;  /* points into mmap'd region */
                t->is_const = true;
            }
        }
    }

    return DNA_OK;
}

/* ── Operator parsing ────────────────────────────────────────────────────── */

static TflOpCode resolve_opcode(const uint8_t *opcodes_fp, uint32_t n_codes,
                                uint32_t idx) {
    if (idx >= n_codes) return (TflOpCode)-1;
    const uint8_t *oc = fb_table(opcodes_fp + idx * 4);
    /* Prefer field 3 (int32 builtin_code); fall back to field 0 (int8). */
    int32_t code = fb_fi32(oc, TFL_OPCODE_builtin_code, 0);
    if (code == 0)
        code = (int32_t)(int8_t)fb_i8(oc, TFL_OPCODE_deprecated_builtin, 0);
    return (TflOpCode)code;
}

static int parse_op(struct DnaModel *m,
                    const uint8_t *op_fb,
                    const uint8_t *opcodes_fp, uint32_t n_codes,
                    Op *op) {
    uint32_t oci = fb_fu32(op_fb, TFL_OP_opcode_index, 0);
    op->op = resolve_opcode(opcodes_fp, n_codes, oci);

    /* inputs */
    const uint8_t *ifp = fb_field(op_fb, TFL_OP_inputs);
    uint32_t ni = 0;
    const uint8_t *iv = ifp ? fb_vec(ifp, &ni) : NULL;
    op->n_inputs = (int32_t)ni;
    if (ni) {
        op->inputs = model_alloc(m, ni * sizeof(int32_t));
        if (!op->inputs) return fail(DNA_ERR_OOM);
        for (uint32_t i = 0; i < ni; i++)
            op->inputs[i] = fb_i32(iv + i * 4);
    }

    /* outputs */
    const uint8_t *ofp = fb_field(op_fb, TFL_OP_outputs);
    uint32_t no = 0;
    const uint8_t *ov = ofp ? fb_vec(ofp, &no) : NULL;
    op->n_outputs = (int32_t)no;
    if (no) {
        op->outputs = model_alloc(m, no * sizeof(int32_t));
        if (!op->outputs) return fail(DNA_ERR_OOM);
        for (uint32_t i = 0; i < no; i++)
            op->outputs[i] = fb_i32(ov + i * 4);
    }

    /* op-specific options */
    const uint8_t *opts_fp = fb_field(op_fb, TFL_OP_builtin_opts);
    const uint8_t *opts    = opts_fp ? fb_table(opts_fp) : NULL;

    switch (op->op) {
        case TFL_OP_FULLY_CONNECTED:
            op->p.gemm.activation = opts
                ? (TflActivation)fb_i8(opts, TFL_FC_fused_activation, TFL_ACT_NONE)
                : TFL_ACT_NONE;
            break;

        case TFL_OP_BATCH_MATMUL:
            op->p.bmm.adj_lhs = opts ? fb_bool(opts, TFL_BMM_adjoint_lhs, false) : false;
            op->p.bmm.adj_rhs = opts ? fb_bool(opts, TFL_BMM_adjoint_rhs, false) : false;
            break;

        case TFL_OP_CONV_2D:
            if (opts) {
                op->p.conv2d.padding    = (TflPadding)fb_i8(opts, TFL_CONV_padding, TFL_PAD_VALID);
                op->p.conv2d.stride_w   = fb_fi32(opts, TFL_CONV_stride_w, 1);
                op->p.conv2d.stride_h   = fb_fi32(opts, TFL_CONV_stride_h, 1);
                op->p.conv2d.activation = (TflActivation)fb_i8(opts, TFL_CONV_fused_activation, TFL_ACT_NONE);
            }
            break;

        case TFL_OP_SOFTMAX:
            op->p.softmax.beta = opts ? fb_f32(opts, TFL_SOFTMAX_beta, 1.0f) : 1.0f;
            break;

        case TFL_OP_MEAN:
            op->p.reduce.keep_dims = opts ? fb_bool(opts, TFL_REDUCER_keep_dims, false) : false;
            /* axes are passed as a separate input tensor, read at runtime */
            break;

        default:
            break;
    }

    return DNA_OK;
}

/* ── Scratch memory planning ─────────────────────────────────────────────── */

static int alloc_scratch(struct DnaModel *m) {
    /* Allocate each non-const activation tensor sequentially from a single
     * arena.  No lifetime analysis — simple but correct. */
    size_t total = 0;
    for (int i = 0; i < m->n_tensors; i++) {
        Tensor *t = &m->tensors[i];
        if (!t->is_const) {
            size_t sz = (size_t)tensor_bytes(t);
            /* 16-byte align each tensor */
            sz = (sz + 15) & ~(size_t)15;
            total += sz;
        }
    }

    if (total == 0) return DNA_OK;
    m->scratch = calloc(1, total);
    if (!m->scratch) return fail(DNA_ERR_OOM);
    m->scratch_size = total;

    uint8_t *ptr = m->scratch;
    for (int i = 0; i < m->n_tensors; i++) {
        Tensor *t = &m->tensors[i];
        if (!t->is_const) {
            t->data = ptr;
            size_t sz = (size_t)tensor_bytes(t);
            ptr += (sz + 15) & ~(size_t)15;
        }
    }
    return DNA_OK;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

DnaModel *dna_load(const char *path) {
    g_last_err = DNA_OK;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { fail(DNA_ERR_IO); return NULL; }

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); fail(DNA_ERR_IO); return NULL; }

    size_t fsz = (size_t)st.st_size;
    uint8_t *fdata = mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (fdata == MAP_FAILED) { fail(DNA_ERR_IO); return NULL; }

    /* Verify TFLite file identifier "TFL3" at bytes 4–7 */
    if (fsz < 8 || memcmp(fdata + 4, "TFL3", 4) != 0) {
        munmap(fdata, fsz);
        fail(DNA_ERR_FORMAT);
        return NULL;
    }

    struct DnaModel *m = calloc(1, sizeof(*m));
    if (!m) { munmap(fdata, fsz); fail(DNA_ERR_OOM); return NULL; }
    m->file_data = fdata;
    m->file_size = fsz;

    const uint8_t *root = fb_root(fdata);

    /* operator_codes vector */
    const uint8_t *oc_fp = fb_field(root, TFL_MODEL_operator_codes);
    uint32_t n_codes = 0;
    const uint8_t *oc_vec = oc_fp ? fb_vec(oc_fp, &n_codes) : NULL;

    /* buffers vector */
    const uint8_t *buf_fp = fb_field(root, TFL_MODEL_buffers);
    uint32_t n_buffers = 0;
    const uint8_t *buf_vec = buf_fp ? fb_vec(buf_fp, &n_buffers) : NULL;

    /* First (and only) subgraph */
    const uint8_t *sg_fp = fb_field(root, TFL_MODEL_subgraphs);
    if (!sg_fp) { dna_free(m); fail(DNA_ERR_FORMAT); return NULL; }
    uint32_t n_sg;
    const uint8_t *sg_vec = fb_vec(sg_fp, &n_sg);
    if (n_sg == 0) { dna_free(m); fail(DNA_ERR_FORMAT); return NULL; }
    const uint8_t *sg = fb_table(sg_vec);   /* first subgraph */

    /* ── Tensors ── */
    const uint8_t *tv_fp = fb_field(sg, TFL_SG_tensors);
    uint32_t n_tensors = 0;
    const uint8_t *tv   = tv_fp ? fb_vec(tv_fp, &n_tensors) : NULL;

    m->n_tensors = (int32_t)n_tensors;
    m->tensors   = calloc(n_tensors, sizeof(Tensor));
    if (!m->tensors) { dna_free(m); fail(DNA_ERR_OOM); return NULL; }

    for (uint32_t i = 0; i < n_tensors; i++) {
        const uint8_t *t_fb = fb_table(tv + i * 4);
        int err = parse_tensor(m, t_fb, buf_vec, n_buffers, &m->tensors[i]);
        if (err) { dna_free(m); return NULL; }
    }

    /* ── Operators ── */
    const uint8_t *ov_fp = fb_field(sg, TFL_SG_operators);
    uint32_t n_ops = 0;
    const uint8_t *ov   = ov_fp ? fb_vec(ov_fp, &n_ops) : NULL;

    m->n_ops = (int32_t)n_ops;
    m->ops   = calloc(n_ops, sizeof(Op));
    if (!m->ops) { dna_free(m); fail(DNA_ERR_OOM); return NULL; }

    for (uint32_t i = 0; i < n_ops; i++) {
        const uint8_t *op_fb = fb_table(ov + i * 4);
        int err = parse_op(m, op_fb, oc_vec, n_codes, &m->ops[i]);
        if (err) { dna_free(m); return NULL; }
    }

    /* ── Graph inputs / outputs ── */
    const uint8_t *inp_fp = fb_field(sg, TFL_SG_inputs);
    uint32_t n_in = 0;
    const uint8_t *inp_v = inp_fp ? fb_vec(inp_fp, &n_in) : NULL;
    m->n_inputs = (int32_t)n_in;
    m->inputs   = model_alloc(m, n_in * sizeof(int32_t));
    if (n_in && !m->inputs) { dna_free(m); fail(DNA_ERR_OOM); return NULL; }
    for (uint32_t i = 0; i < n_in; i++)
        m->inputs[i] = fb_i32(inp_v + i * 4);

    const uint8_t *out_fp = fb_field(sg, TFL_SG_outputs);
    uint32_t n_out = 0;
    const uint8_t *out_v = out_fp ? fb_vec(out_fp, &n_out) : NULL;
    m->n_outputs = (int32_t)n_out;
    m->outputs   = model_alloc(m, n_out * sizeof(int32_t));
    if (n_out && !m->outputs) { dna_free(m); fail(DNA_ERR_OOM); return NULL; }
    for (uint32_t i = 0; i < n_out; i++)
        m->outputs[i] = fb_i32(out_v + i * 4);

    /* ── Scratch activation memory ── */
    int serr = alloc_scratch(m);
    if (serr) { dna_free(m); return NULL; }

    return m;
}

void dna_free(struct DnaModel *m) {
    if (!m) return;
    if (m->file_data) munmap(m->file_data, m->file_size);
    free(m->scratch);
    free(m->tensors);
    free(m->ops);
    for (int i = 0; i < m->n_allocs; i++) free(m->allocs[i]);
    free(m->allocs);
    free(m);
}

int dna_n_inputs(const DnaModel *m)  { return m->n_inputs; }
int dna_n_outputs(const DnaModel *m) { return m->n_outputs; }

static DnaTensor tensor_to_public(const Tensor *t) {
    DnaTensor pub = {0};
    pub.data       = (int8_t *)t->data;
    pub.ndim       = t->ndim;
    pub.n_elems    = t->n_elems;
    pub.scale      = t->quant.n_ch > 0 ? t->quant.scale[0] : 1.0f;
    pub.zero_point = t->quant.n_ch > 0 ? t->quant.zero_point[0] : 0;
    for (int i = 0; i < t->ndim && i < 6; i++) pub.shape[i] = t->shape[i];
    return pub;
}

/* We return a pointer to a per-model public-tensor scratch so callers can
 * modify data in-place.  These are embedded in the DnaModel struct below. */
DnaTensor *dna_input(DnaModel *m, int idx) {
    if (idx < 0 || idx >= m->n_inputs) return NULL;
    Tensor *t = &m->tensors[m->inputs[idx]];
    /* Lazily expose via a small cache embedded in the model.
     * For simplicity, rebuild each call — the struct is 48 bytes. */
    static DnaTensor tmp;  /* not re-entrant, but single-threaded inference */
    tmp = tensor_to_public(t);
    tmp.data = (int8_t *)t->data;  /* user writes here */
    return &tmp;
}

DnaTensor *dna_output(DnaModel *m, int idx) {
    if (idx < 0 || idx >= m->n_outputs) return NULL;
    Tensor *t = &m->tensors[m->outputs[idx]];
    static DnaTensor tmp;
    tmp = tensor_to_public(t);
    return &tmp;
}
