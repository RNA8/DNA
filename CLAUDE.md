# CLAUDE.md — DNA (Deep Neural Acceleration)

Lightweight INT8 Vision Transformer inference library in C, targeting
Raspberry Pi 4 / 5 (ARM Cortex-A72 / A76).  Loads quantized `.tflite`
models and runs them using NEON SIMD with runtime dispatch for the
ARMv8.2-A dot-product extension.

License: GNU GPL v3

---

## Repository Layout

```
DNA/
├── include/
│   └── dna.h              Public API (load, invoke, tensor accessors)
├── src/
│   ├── fb.h               Header-only FlatBuffers binary reader
│   ├── tflite_schema.h    TFLite op codes, tensor types, field indices
│   ├── model.h            Internal IR: Tensor, Op, DnaModel structs
│   ├── model.c            TFLite flatbuffer parser + mmap loader
│   ├── ops.h              Internal op function declarations
│   ├── runner.c           Graph executor (dispatch loop over ops)
│   └── ops/
│       ├── gemm.c         INT8 FULLY_CONNECTED — NEON baseline + dotprod
│       ├── batchmatmul.c  INT8 BATCH_MATMUL (attention scores/context)
│       ├── softmax.c      SOFTMAX (dequant → float → requant)
│       ├── layernorm.c    LAYER_NORM (float normalization)
│       ├── gelu.c         GELU via 256-entry INT8 lookup table
│       ├── conv2d.c       CONV_2D (patch embedding, scalar)
│       └── elemwise.c     ADD, MUL, MEAN, RESHAPE, TRANSPOSE,
│                          QUANTIZE, DEQUANTIZE
├── Makefile
├── LICENSE                GNU GPL v3
└── README.md
```

---

## Build

```bash
make              # builds libdna.a
make clean
```

Compiler flags on aarch64: `-O3 -march=armv8-a+simd`.  The dotprod
kernel in `gemm.c` uses `__attribute__((target("+dotprod")))` and is
selected at runtime via `HWCAP_ASIMDDP`.  The library compiles and runs
on x86 (scalar fallback) for development.

---

## Public API (`include/dna.h`)

```c
DnaModel  *dna_load(const char *path);   // mmap + parse .tflite
void       dna_free(DnaModel *m);

int        dna_n_inputs(const DnaModel *m);
int        dna_n_outputs(const DnaModel *m);
DnaTensor *dna_input(DnaModel *m, int idx);   // write INT8 data here
DnaTensor *dna_output(DnaModel *m, int idx);  // read INT8 data here

int        dna_invoke(DnaModel *m);           // returns DNA_OK or error

// DnaTensor fields: data (int8_t*), shape[], ndim, scale, zero_point
// Quantize:    q = clamp(round(x / scale) + zero_point, -128, 127)
// Dequantize:  x = (q - zero_point) * scale
```

---

## Supported TFLite Ops

| Op | TFLite code | Notes |
|---|---|---|
| FULLY_CONNECTED | 9 | INT8 GEMM, per-channel weights, fused ReLU/ReLU6 |
| BATCH_MATMUL | 126 | Attention scores (Q×Kᵀ) and context (Attn×V) |
| SOFTMAX | 25 | Float computation over last dim |
| LAYER_NORM | 149 | Float normalization; also handles decomposed sequences |
| GELU | 137 | 256-entry LUT |
| CONV_2D | 3 | Patch embedding, VALID/SAME padding |
| ADD | 0 | Residual connections |
| MUL | 18 | |
| MEAN | 40 | Global average pool |
| RESHAPE | 22 | |
| TRANSPOSE | 39 | |
| QUANTIZE | 114 | float → INT8 |
| DEQUANTIZE | 6 | INT8 → float |
| RELU / RELU6 | 19, 24 | In-place |

---

## Quantization Model

- **Weights**: INT8, symmetric per-channel (`scale[c]`, `zero_point[c]=0`)
- **Activations**: INT8, asymmetric per-tensor (`scale`, `zero_point`)
- **Bias**: INT32, scale = `s_activation × s_weight[c]`
- **Accumulation**: INT32, then requantized via `round(acc × s_eff) + zp_out`
- **Target training**: PyTorch QAT → `ai_edge_torch` → `.tflite`

---

## NEON Strategy

| CPU | ISA | Instruction | Throughput |
|---|---|---|---|
| Cortex-A72 (Pi 4) | ARMv8.0-A | `vmull_s8` + `vpadalq_s16` | 8 MACs/cycle |
| Cortex-A76 (Pi 5) | ARMv8.2-A+dotprod | `vdotq_s32` | 16 MACs/cycle† |

†Per NEON lane; 4 lanes → 64 INT8 MACs per `vdotq_s32` instruction.

Runtime detection in `gemm.c`:
```c
getauxval(AT_HWCAP) & HWCAP_ASIMDDP  →  select dotprod path
```

---

## Adding a New Op

1. Add the TFLite opcode to `TflOpCode` in `src/tflite_schema.h`
2. Add a `params` field in the op union in `src/model.h` if needed
3. Parse the op options in `parse_op()` in `src/model.c`
4. Implement `int op_foo(DnaModel *m, const Op *op)` in `src/ops/`
5. Declare it in `src/ops.h`
6. Add a `case TFL_OP_FOO:` in the dispatch in `src/runner.c`
7. Update this file

---

## Known Limitations / Future Work

- `BATCH_MATMUL` uses a scalar inner loop when B (keys) is column-strided;
  a transposed-B packing pass would vectorize this fully
- `MEAN` uses an O(output × input) fallback; replace with a proper
  strided-reduction once axes are always the spatial dims [1,2]
- `CONV_2D` is scalar; patch embedding is called once per inference so
  this is not a bottleneck, but a NEON im2col+GEMM path would help for
  multi-scale conv stems (MobileViT)
- No multi-threading; the graph is single-threaded
- No memory re-use across activations (scratch = sum of all activation sizes)

---

## Development Workflow

- Branch: `claude/<feature>-<id>` off `master`
- Commits: short imperative messages
- Build check: `make` must succeed before pushing
- No external dependencies beyond libc and libm
