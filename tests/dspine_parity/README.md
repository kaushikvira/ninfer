# P0b — DSpine bridge-op parity (the P0 kill-switch)

Proves the fork's six DSpine bridge kernels are numerically faithful to Tencent's
reference. **The oracle is the COMPILED OP** (`/tmp/ninfer-dspine-port/build/p0a/lib/
libninfer_ops.so`, branch `feat/dspine-port` @ `1138e8b2`), run on GPU 1; the
**reference is Tencent's** `dspine_mlx` / `sglang_dspine` math, evaluated in NumPy
fp32 with BF16 storage at exactly the reference's rounding points. A previous
attempt failed by hand-transcribing the kernels into Python and comparing
model-vs-model; this harness does not do that — it calls the real kernels and
compares their raw output bytes to the reference.

## Verdict (measured, 2026-11-22, GPU 1, prod geometry)

**SOUND.** No DIVERGENT in the clean case. Every elementwise output is BIT-EXACT;
every reduction output is within the documented EQUIVALENT-ORDER band (reduction
order + the `rsqrtf`/`expf` intrinsic gap). Every strong mutant is DIVERGENT, so
the comparison has teeth.

| op | output | verdict | max abs delta | cases |
|---|---|---|---|---|
| `dspine_prepare_read` | `normalized` | **BIT-EXACT** | 0 ulp (bf16) | 3/3 |
| `dspine_prepare_read` | `rms` | EQUIVALENT-ORDER | 31–33 ulp (fp32) | 3/3 |
| `dspine_normalize_codes` | `codes` | **BIT-EXACT** | 0 ulp (bf16) | 3/3 |
| `dspine_message` | `message` | **BIT-EXACT** | 0 ulp (bf16) | 3/3 |
| `dspine_finish` | `residual_out` | **BIT-EXACT** / 1 ulp | 0–1 ulp (bf16) | 3/3 |
| `dspine_finish` | `normed` | **BIT-EXACT** / EQUIV-ORDER | 0–2 ulp (bf16) | 3/3 |
| `dspine_serial_scores` | `idx` | **BIT-EXACT** | 0 (int32) | 3/3 |
| `dspine_serial_chain` | `drafts` | **BIT-EXACT** | 0 (int32) | 3/3 |

`residual_out`'s occasional 1 ulp is a **signed-zero sign-bit artifact** of the
test data (a `±0.0` input), not a rounding defect; it is inside the `residual_out`
band of 2.

### Why the reduction outputs are not bit-exact (and why that is fine)

`rms` (prepare_read) and `normed` (finish) depend on a sum-of-squares reduction.
The kernel reduces with a 256-thread tree (`kernels.cu` `block_reduce_sum`); the
reference reduces with NumPy's pairwise sum. Both are fp32, so the results agree to
within the fp32 rounding scale (≤34 ulp for `rms`, ≤2 ulp for the derived `normed`).
The kernel also uses the `rsqrtf` intrinsic (≈2-ulp approximate) where the
reference uses a correctly-rounded `1/sqrt`; that gap is inside the band too. This
is the documented **EQUIVALENT-ORDER** category: the bits differ, the difference is
named (reduction order + intrinsic), and it is at the reference's own fp32 rounding
scale. The elementwise outputs — which carry the actual rounding points — are
BIT-EXACT, which is the real proof.

## Geometry

Prod DSpine dims with tiny batch/seq (per the task: do not port a "smaller"
variant): `hidden 5120`, `rank 512`, `code_size 1024`, `width 16` (15 drafts +
anchor), `batch 1`. **One deliberate deviation: `vocab 1024`** (prod ≈150k) — the
`serial_scores` tables (`head [V,H]`, `tg [V,R]`, `tsv [V,2R]`) would be ~770 MB at
prod vocab, beyond a small-allocation test. Vocab is a pure gather index (never
reduced over), so it does not affect the parity of the reduction math or the
rounding points.

## Layout gotchas (these are what the first attempt got wrong)

- **Tensor inputs are Fortran order** (first axis fastest: element `(h,c)` at
  offset `c*H+h`). Written with `tobytes(order='F')` (a bare `tobytes()` defaults
  to C-order even for F-contiguous arrays). The reference reshapes back with
  `order='F'`.
- **Weight tables (`codebook`/`tg`/`tsv`) and the fp8 `head` are C-order**
  (row-major: `data[row*k + col]`), the opposite of the Tensors.
- **`cand`/`idx` are `(16, drafts, batch)` = (slot, row, batch)**; the chain walk
  is `cand[j, i, b]` / `idx[j, i+1, b]` (the Tencent ref uses `(row, slot)` order).
- **The fixture's fp8 `head` `Weight` must carry `ndim = 2` and its `shape`/`padded_shape`**
  (P2b finding): it was built with `ndim = 0` and zeroed shapes, which no consumer read —
  `dspine_serial_scores` gates the head on qtype/layout only. The first op to *measure* the
  head (P2b's Q8 head gate, `ops/dspine_bridge/dspine_bridge.cpp::require_head`) took
  `padded_shape[1] = 0` as a refusal and died in `Tensor::data()`'s
  `runtime_check(notes_ != nullptr)`. A fixture `Weight` that only one op ever reads is not a
  weight; the geometry is filled in as `weight_view.cpp::native_weight` fills it.

## Build & run

Requires: `g++`, CUDA headers (`/usr/local/cuda`), the P0a build
(`/tmp/ninfer-dspine-port/build/p0a/lib/{libninfer_ops,libninfer_core}.so`),
NumPy. Runs on **GPU 1** (device index 1) with small allocations; it never touches
device 0 (prod).

```sh
cd tests/dspine_parity
./fixture/build.sh                      # builds fixture/dspine_fixture (one-time)
python3 run_parity.py --ops prepare_read,normalize_codes,message,finish,serial_scores,serial_chain \
                      --cases 3 --report /tmp/p0b_clean.json
```

`run_parity.py` generates fixed-seed dyadic test vectors (exact in bf16, so a
reduction-order difference cannot masquerade as a logic defect), writes them to
binary, invokes the fixture per op, and prints a per-case table with the max abs
delta in ulps. Exit 0 = no DIVERGENT.

### Teeth (mutants)

`--mutant NAME` computes the reference **with** a deliberate defect and compares it
to the compiled op. A sound op + working harness yields DIVERGENT, proving the
comparison can see that defect class. Each op has a strong mutant (all measured
DIVERGENT):

| op | strong mutant | defect | measured |
|---|---|---|---|
| `prepare_read` | `rms_no_sqrt` | drop the RMS sqrt/clamp | 2578 ulp |
| `normalize_codes` | `no_sqrt_c` | drop the `sqrt(C)` factor | 640 ulp |
| `message` | `sigmoid_bf16` | round the gate to bf16 before the value multiply (cast placement) | 1 ulp (band 0) |
| `finish` | `no_refine` | drop the `rms*delta` refinement term | 32817 ulp |
| `serial_scores` | `base_missing` | drop the `base_k` (H-dim) score term | 227/240 idx |
| `serial_chain` | `reversed` | consume the next row's index before reading the current row | 15/15 drafts |

Weak mutants (documented, inside the band, NOT the teeth): `finish/rounding_point`
(skip the intermediate bf16 round of `out`; 1–2 ulp) and `serial_scores/rms_missing`
(drop the rms factor; the H-dim base dominates the argmax so the idx is unchanged).

## Files

- `fixture/dspine_fixture.cpp` — C++ harness; calls the real ops, dumps raw bytes.
- `fixture/build.sh` — builds the fixture against the P0a build.
- `run_parity.py` — driver: vectors + fixture + compare + verdict table.
- `reference.py` — Tencent's op math in NumPy (each fn cites its reference lines).
- `mutants.py` — the deliberately-wrong variants (the teeth).
- `numeric.py` — bf16/fp32 ulp helpers + dyadic test-vector generators.

## References (rounding points)

- `dspine_mlx/dspine_mlx/bridge.py` — `DSpineCodeBridge.__call__` (bridge family).
- `dspine_mlx/dspine_mlx/sglang/kernels/reference.py` — `chain` / serial chain.
- `dspine_mlx/dspine_mlx/propose.py` — `RefinementTail.greedy` / `_chain`.
- `sglang_dspine/kernels/dspine.py` — triton `prepare_read`/`normalize_codes`/
  `feature_message`/`feature_finish`.
- Fork under test: `/tmp/ninfer-dspine-port/src/ops/dspine_bridge/kernels.cu`.
