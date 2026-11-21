# P2 — NVFP4 gather/dequant table parity (the D-20 / D-21b gap)

P0b proved the DSpine bridge ops faithful to Tencent's reference **with BF16 gather
tables**. The artifact we intend to serve (`…_dspine-nvfp4tables.ninfer`, 1.3157 GiB, the
D-14 sizing) ships **NVFP4** codebook / `tg` / `tsv`, and the dequant-on-gather branch
(`kernels.cu::table_elem_bf16`, template `Nvfp4=true`) had never been executed. This
directory measures it. The oracle is the compiled op of this tree
(`build/rtx5090-ops/lib/libninfer_ops.so`, preset `rtx5090-ops`, `sm_120a`, CUDA 13.1.115,
gcc 15.2) on GPU 1; the reference is the **P0b harness imported, not retyped**
(`tests/dspine_parity/{reference,numeric,mutants,run_parity}.py` — vectors, op math, ulp
helpers), fed with the CPU decode of the *same NVFP4 bytes* the device gathers.

No kernel was edited. The kernels are the suspects; Tencent's reference is the arbiter.

## Verdict (measured 2026-11-22, GPU 1 = RTX PRO 5000, ~25 GiB free window)

| op | table(s) | (a) device-NVFP4 vs CPU decode of the same bytes | (b) device NVFP4 vs device BF16 | teeth |
|---|---|---|---|---|
| `dspine_normalize_codes` | codebook `[V,C]` | **BIT-EXACT**, 0 ulp (1024/1024 anchor elements, 2/2 cases) | drift: max 16288 bf16 ulp, p99 16256 | 4/4 **DIVERGENT** (256 … 33793 ulp) |
| `dspine_serial_scores` (P0b vector scale) | `tg [V,R]`, `tsv [V,2R]` | **BIT-EXACT** idx, 240/240, 2/2 cases | 240/240 and 239/240 idx (1 argmax flip) | only `divisor_dropped` bites (232/227 idx) |
| `dspine_serial_scores` — `tgdom` (`rms`×1000, so the ranking term is the one the tables feed) | same | **BIT-EXACT** idx, 240/240, 2/2 cases | 27/240 and 25/240 idx flips | **4/4 DIVERGENT** (57–229 idx) |
| `dspine_serial_tail` | (no Weight — the scores table) | **format-invariant**: identical table in ⇒ byte-identical drafts/idx/proposal_q (3/3) | drift: 1, 6, 8 of 15 drafts; 24–34 of 240 idx; 224/240 of `proposal_q` | 4/4 mutants move the draws (6–14 of 15 drafts, 40–193 of 240 idx) ⇒ **teeth on this path** |

**Gather path: SOUND.** The three table-consuming entry points dequantize NVFP4 rows exactly
as the registered format says (E2M1 nibble × E4M3 scale ÷ divisor, rounded to BF16 once,
`nvfp4_scale_byte_offset` addressing), bit for bit, on the anchor rows, on the tg/tsv
gather, and the sampled tail is provably format-agnostic. The BF16→NVFP4 column is the
**expected quantization trade** (D-20: −0.85 GiB drafter ⇒ ~60k more device-KV tokens), not a
defect; at K1 (T=1.0, top_k=20 = no cut, fixed seed) it moves ~7–53 % of the sampled drafts,
which is why the A1 A/B (N≥3 repeats, a T=0 arm) is the only place its accept cost is
measured.

### Why the argmax teeth need the `tgdom` set

`serial_scores`' only output is an argmax, and at the P0b vector scale the H-dim term
(`base_k`, a 5120-wide dot through the FP8 head) dominates the R-dim term (`rms · m·tg`) that
is the *only* one tg/tsv feed — P0b recorded the same shape of effect
(`serial_scores/rms_missing` = weak mutant). A tg/tsv decode defect then flips 0–9 argmaxes
of 240, i.e. the comparison cannot see the defect class on that vector set. Scaling `rms` by
1000 (the op is linear in it; the same bytes go to the device and to the reference, so it is a
regime choice, not a tolerance) puts the ranking on the table term: `A` stays **BIT-EXACT**
and all four teeth bite (57–229 of 240 idx). The same defects also move 40–193 of 240 `idx`
and 6–14 of 15 drafts when they travel through the **FP32 score table** the bridge path hands
the sampled tail (`run_serial_tail.py`, which keeps per-`(sender, candidate)` scores instead
of collapsing them to an argmax). Read the two ops as one measurement.

## Geometry

Prod DSpine dims (`hidden 5120`, `rank 512`, `code_size 1024`, `width 16` = 15 drafts +
anchor, `batch 1`), **vocab 1024** as in P0b (a pure gather index; the prod 248 320 rows would
put tg+tsv at ~770 MB in BF16 and 1.0 GiB of NVFP4 planes for no extra gather coverage).
1024 is 8 × 128 rows and 1024/512 are multiples of 64, which the NVFP4
`block_scale_k16_m128x4_v1` tiling requires — the same divisibility the prod tables satisfy
(248320 = 1940 × 128; 1024 = 16 × 64; 512 = 8 × 64), so this is not a test-only geometry.

## Build & run

```sh
cmake --preset rtx5090-ops && cmaker --build build/rtx5090-ops --target ninfer_ops -j 32
cd tests/dspine_parity_nvfp4 && ./fixture/build.sh
# The window: the driver's --device is the CUDA index the *fixture* sees, so mask the
# card and use index 0 inside it. Never device 0 of the unmasked box (prod 5090).
CUDA_VISIBLE_DEVICES=1 ./run_nvfp4_parity.py --ops normalize_codes,serial_scores --cases 2 --device 0
CUDA_VISIBLE_DEVICES=1 ./run_serial_tail.py --cases 3 --device 0 --temperature 1.0 --top-k 20
```

The tail op's own parity vs the C++ host core and the committed Tencent-policy fixture is
`tests/ops/dspine` (`ninfer_dspine_serial_tail_op_test`), not this directory — **executed in
this window for the first time**: `OK 4 cases, 0 skipped, 2.93 s` on GPU 1 (reference parity,
CUDA-graph capture/replay, the wrong-`inv_temp` teeth, the geometry rejections). Its fixture
cases carry T ∈ {0, 0.05, 0.7, 0.9, 1.0 (with top_k 4), 1.2}; the D-23 **K1** configuration
(T = 1.0 *and* the no-cut top_k = 20) is not among them, so K1-proper is still an open arm
(everything above is at K1).

## Files

- `fixture/dspine_nvfp4_fixture.cpp` — P0b fixture CLI superset: `--nvtbl=NAME:CODES:SCALES:
  DIVISOR` builds the `Weight` the way `core/weight_view.cpp` materializes it (single payload,
  code plane at 0, swizzled scale plane at `align_up(code_bytes, 256)`, `group_size = group =
  16`, `ndim = 2`, `scale_dtype = FP8_E4M3FN`, `weight_scale` = the divisor, `input_scale > 0`)
  because `ops/common/host/operands.cpp::validate_nvfp4_weight` — the op's own gate — demands
  that shape. Plus the `serial_tail` case with the request policy on the command line
  (`--temp/--topk/--topp/--minp/--seed`).
- `nvfp4.py` — the encode side is the converter's own `NVFP4_MAXABS_DIVISOR_RNE_V1`
  (`tools/convert/quantization/nvfp4_maxabs.py`) plus the registered
  `swizzle_nvfp4_scales`; the decode side mirrors `kernels.cu::table_elem_bf16`,
  `nvfp4.cuh::decode_nvfp4_e2m1x2`, `fp8.cuh::decode_e4m3` (via the P0b `reference.py`) and
  `nvfp4.h::nvfp4_scale_byte_offset`. Also the four gather teeth.
- `run_nvfp4_parity.py` — `normalize_codes` / `serial_scores`: the P0b vectors, (a)/(b)/(c).
  `normalize_codes` is scored on the **anchor columns only** (`anchor_columns`): the codebook
  is read nowhere else, and the full array only dilutes the count.
- `run_serial_tail.py` — the tail: format-invariance control, BF16-vs-NVFP4 score tables,
  teeth. `score_table` is the table-level view of `reference.serial_scores`' `scores`
  expression (P0b's reference exposes only its argmax).

## Found in passing: the 1.3157 GiB artifact cannot be bound today (P2 integration, not a kernel gap)

The shipped `qwen3_8_27b_swift15_v3_dspine-nvfp4tables.ninfer` manifest carries

```text
dspine/bridge/codebook  format nvfp4  [248320, 1024]  block_scale_k16_m128x4_v1
dspine/bridge/tg        format nvfp4  [248320,  512]  block_scale_k16_m128x4_v1
dspine/bridge/tsv       format nvfp4  [248320, 1024]  block_scale_k16_m128x4_v1
```

while `src/models/common/dspine/weights.cpp:46,56,57` bind exactly those three with
`b.direct(name, shape)`, whose default is `format = QType::BF16` (`models/common/bindings.h:28`),
and `artifact/binder.cpp:49-53` throws `representation does not match its mathematical value
type` for any part whose format is not the requested one. So the D-20 artifact is rejected in
**binding**, before a `Weight` ever reaches the gather kernels that this harness just proved
sound. The last mile is to bind those three as NVFP4 (per-artifact: the BF16-tables build
binds BF16), gated on the same shape divisibility the format requires — which the prod shapes
satisfy (248320 = 1940 × 128; 1024 = 16 × 64; 512 = 8 × 64). Not exercised here: opening the
25 GiB artifact needs the engine build, which is outside a measurement window; the claim is
the binding path's code, plus the manifest above.

## Contract facts this harness pinned down (worth keeping)

- `dspine_serial_scores` **enforces** `tg.qtype == tsv.qtype` (`dspine_bridge.cpp`), so the
  header's "formats must match" is a gate, not a comment.
- `validate_nvfp4_weight` requires `n % 128 == 0`, `k % 64 == 0`, a positive finite
  `input_scale` *and* `weight_scale`, 16-byte-aligned non-overlapping planes inside
  `payload`. A gather table that fails any of these never reaches a kernel.
- The `--nvtbl` divisor is parsed with `strtof`: a numpy-scalar `repr` (`np.float32(2688.0)`)
  parses to **0** and the op then rejects the weight. The fixture now dies on a
  non-positive divisor instead of sending a dead table.
