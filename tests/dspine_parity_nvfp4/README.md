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

**This directory carries a second arm now** (`run_head_score_table.py`, the section starting
"P2 head-format bridge"): the proposal **head** format bridge (`q8_g32_fp16`, the tower's own
output head) and the **score-table producer** the sampled tail needs — the two gaps D-34 put
under the kernel-geometry blocker. There a kernel *was* edited, and the P0b head numerics are
held to bit-exactness by the `run_parity.py` arm above.

## Verdict (measured 2026-11-22, GPU 1 = RTX PRO 5000, ~25 GiB free window)

| op | table(s) | (a) device-NVFP4 vs CPU decode of the same bytes | (b) device NVFP4 vs device BF16 | teeth |
|---|---|---|---|---|
| `dspine_normalize_codes` | codebook `[V,C]` | **BIT-EXACT**, 0 ulp (1024/1024 anchor elements, 2/2 cases) | drift: max 16288 bf16 ulp, p99 16256 | 4/4 **DIVERGENT** (256 … 33793 ulp) |
| `dspine_serial_scores` (P0b vector scale) | `tg [V,R]`, `tsv [V,2R]` | **BIT-EXACT** idx, 240/240, 2/2 cases | 240/240 and 239/240 idx (1 argmax flip) | only `divisor_dropped` bites (232/227 idx) |
| `dspine_serial_scores` — `tgdom` (`rms`×1000, so the ranking term is the one the tables feed) | same | **BIT-EXACT** idx, 240/240, 2/2 cases | 27/240 and 25/240 idx flips | **4/4 DIVERGENT** (57–229 idx) |
| `dspine_serial_tail` | (no Weight — the scores table) | **format-invariant**: identical table in ⇒ byte-identical drafts/idx/proposal_q (3/3) | drift: 1, 6, 8 of 15 drafts; 24–34 of 240 idx; 224/240 of `proposal_q` | 4/4 mutants move the draws (6–14 of 15 drafts, 40–193 of 240 idx) ⇒ **teeth on this path** |

Reading the drift column: a bf16 ulp distance of ~16 000 is the signature of *one side being
zero* (the raw-bit metric spans ~256 steps per octave from zero to ≈24), i.e. NVFP4 quantizing
a small table entry to 0. On a P0b-shaped codebook `[1024, 1024]` the converter's profile
zeroes 18.0 % of the entries at a relative-Frobenius error of 0.095 — the 4-bit trade D-20
accepted for −0.85 GiB of drafter.

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
  expression (P0b's reference exposes only its argmax). Its 16 x 16 block is stored as
  `[sender, candidate]`, which is fine for a table it feeds the tail as an input; the table
  `dspine_serial_scores` *emits* (and the tail reads) is `[candidate, sender]` —
  `run_head_score_table.py::op_score_table` is that layout.
- `q8.py` — the tower-head arm: `encode_head` is the int8 / FP16 group quantizer
  (`integer.cuh::int8_fp16_scale`, `int8_quant_code`), `dequant_head` mirrors
  `embedding/kernel.cuh::embed_gather_q8_kernel` (FP16 scale per 32-column group, signed code
  at the `align_up(k, 128)` padded row stride, the product BF16-rounded), plus the four head
  teeth. Padded on purpose: prod's 5120 is a multiple of 128, so only a padded geometry can
  show a head read at the packed stride.
- `run_head_score_table.py` — **the head-format bridge and the score-table producer** (D-34's
  two gaps; see the next section). Two arms per case: the natural vectors and `headdom`
  (`rms` x 1e-3, the regime where the ranking *is* the head-term dot — the mirror of the
  `tgdom` arm above, for the same reason).

## P2 head-format bridge + score table (D-34: measured 2026-11-22, GPU 1 = RTX PRO 5000)

`dspine_serial_scores` took the proposal head as `FP8_E4M3FN_ROW_BF16`, but the head it scores
against **is the tower's** (`models/qwen3_5/definition.cpp:99` hands the DSpine `Weights`
`weights.text.output_head`), and the prod tower declares `q8_g32_fp16 [248320, 5120]`. And the
op computed the K x K score matrix, kept it in shared memory, and emitted only its argmax — so
the sampled tail, whose *input* is that FP32 `[batch, 16, 16, drafts]` table, had no producer.
Both are closed; the T=0.9 arm is no longer blocked twice over.

| arm | what is compared | verdict (prod 5120/512/1024, batch 1 and 2; padded 160/64/128; 2 cases x 2 arms) |
|---|---|---|
| `A-q8-gather` | device `idx` with a **Q8 head** vs the P0b reference fed the CPU decode of the same bytes | **BIT-EXACT** 240/240 (batch 1), 480/480 (batch 2), every case and arm |
| `A-score-table` | the emitted FP32 table vs the reference table (normwise: relative Frobenius ≤ 1e-3, pointwise ≤ 2e-2) | **EQUIVALENT-ORDER**, rel-Fro **1.3–1.5e-7**, max relative pointwise ≤ 1.0e-3 |
| `A-table-argmax` | `argmax(candidate axis)` of the emitted table vs the device `idx` | **BIT-EXACT** — the table is the matrix the argmax collapsed, in the layout the tail reads |
| `B-fp8-vs-q8` | device FP8-head `idx` vs device Q8-head `idx` (values of one matrix, two formats) | 0–2 of 240 argmax flips; table rel-Fro **6.8e-3** — the representation trade, priced not gated |
| `C-teeth` | reference-side head-decode defects: scale group offset / dropped sign / **packed (unpadded) code stride** / row-scale broadcast, and the rms·(m·tg) term dropped from the table | every kind bites; `packed_stride_read` only on a **padded** head (205–208 of 240 idx), which is why that geometry is in the suite |
| `D-compose` | device table → device `serial_tail`: **T=0.9, top_k 20, top_p 1.0, min_p 0.05** (prod) and T=0 | 15/15 drafts per row (30/30 at batch 2) with normalised finite `proposal_q` and every drawn id among the slot's 16 candidates; **T=0 == `serial_scores` + `serial_chain` BIT-EXACT** (D-24's free equivalence, now measured through the composed path) |
| `Z-gate` | four arms the op's contract must *refuse*: a Q8 head claiming `group = 16`, one claiming a BF16 scale plane, one with a padded column count that is not a whole number of groups, and the score table declared BF16 (only the `Weight` metadata is warped — never the planes) | **all four GATE-REJECTED, naming the field** (`q8 head group must be 32 (the G32 of the format)`, `… scale_dtype must be FP16`, `… whole 32-column groups in k and padded_shape[1]`, `scores must be FP32 […]`) — the acceptance of the tower's format is a decision of a gate that fires on the near-misses |

**Head format: how Q8 is gathered, with no copy.** `serial_scores_kernel` is templated on
`HeadQ8` and takes a `HeadRef {codes, scales, row_stride, scale_groups}`; per element it reads
the signed byte at `row * padded + h`, the FP16 scale of group `h / 32`, multiplies in FP32 and
rounds the product to BF16 — the addressing and the rounding point of
`embedding/kernel.cuh::embed_gather_q8_kernel`, which is the in-tree gather of this same
format. `dspine_bridge.cpp::require_head` accepts exactly `FP8_E4M3FN_ROW_BF16` (RowScale) and
`Q8_G32_FP16` (RowSplit, group 32, FP16 scales, `k % 32 == 0`, `padded_shape[1] % 32 == 0`, no
high plane) and names both sides in its message. The evidence that no copy is taken is in the
kernel (`head.codes` / `head.scales` are the weight's own planes, the row is indexed by
`cand_s[k]`) and in the memory accounting: prod's head is 1.18 GiB of codes + 76 MiB of FP16
scales, so a BF16 staging copy of it is +2.35 GiB, a row-scaled-FP8 copy +1.26 GiB, against the
**1.31568 GiB whole-drafter** budget D-14 sized (D-34's 3.83186 GiB load already counts that head
once, reused). Nothing in this change allocates.

**Score table: the shape of the output.** Decision: **an optional out-param on
`dspine_serial_scores`** (`Tensor* scores`, null = today's behaviour), not a sibling op. The
precedent is in this very file — `dspine_finish`'s optional `stash_raw` — and a sibling op
would re-run every 5120-wide head dot to recompute the `score_s` matrix the argmax already
read. Layout is FP32 `[batch, 16 (candidate), 16 (sender), drafts]`, first axis fastest, i.e.
`((row * 16 + sender) * 16 + candidate) * batch + b` — element-for-element the indexing
`serial_tail_kernel` already uses (`table[b + batch * (k + 16 * (sender + 16 * i))]`), and
`A-table-argmax` is the proof that the two agree. The buffer is caller-provided, the write is
`K*K = 256` fp32 per CTA (15 KiB for a prod round), no allocation, no host sync, so the graph
capture/replay contract of D-25 is untouched.

```sh
cmake --preset rtx5090-ops && cmake --build build/rtx5090-ops --target ninfer_ops -j 4
cd tests/dspine_parity_nvfp4 && ./fixture/build.sh
CUDA_VISIBLE_DEVICES=1 ./run_head_score_table.py --cases 2 --device 0                    # prod
CUDA_VISIBLE_DEVICES=1 ./run_head_score_table.py --cases 1 --device 0 --batch 2          # batch 2
CUDA_VISIBLE_DEVICES=1 ./run_head_score_table.py --cases 1 --device 0 \
    --hidden 160 --rank 64 --vocab 128 --pad 1                                           # padded stride
```

The head and its tables are the P0b vectors at the P0b seeds, so `B-fp8-vs-q8` compares two
representations of one matrix (the FP8 head's own dequantized values, re-quantized to Q8), not
two datasets. The Q8 head of the prod geometry is 5.0 MiB of codes + 0.32 MiB of scales: this is
a small-memory arm, and the device window it needs is the fixture process, not the artifact.

### Why the score table is judged normwise, and what `headdom` is for

The table is a matrix-family reduction (a 5120-term dot through the head, an R-term dot), and
the device order is fixed by the kernel (16 strided partial sums, then a sequential combine)
where NumPy reduces pairwise. The clean case measures 1.3e-7 relative Frobenius and up to 1.4e4
raw fp32 ulps of *pointwise* distance — cancellation inside a wide dot, invisible in norm, so
the ulp band that suits `rms` (a 5120-wide sum of squares) is the wrong yardstick here: the
op-development rule for this family is normwise with a gross pointwise ceiling, and the teeth
(`tg` term dropped: rel-Fro 4.3e-2) sit two to three orders above the band. The argmax rows
(`A-q8-gather`, `A-table-argmax`) stay at zero band, which is what makes a head defect
visible at all. The criterion's **non-finite rule is explicit**: a non-finite in the emitted
table is a failure in itself (`NON-FINITE`, exit 1) and not a distance — the op cannot see one
at call time without a host sync, so the harness is where the tail's `scores must be finite`
precondition gets its check on the producer side.

`headdom` (`rms` x 1e-3) is the counterpart of the `tgdom` arm above. `serial_scores`' ranking
is `base(head) + rms * (m · tg)`, and which term decides the argmax is a property of the
vectors, so at the natural scale `row_scale_broadcast` moves 41–48 of 240 argmaxes while
`sign_ignored` moves 224 — a tooth the natural arm cannot see is reported, not hidden, and the
arm where that term decides the ranking is where it must bite. `packed_stride_read` is inert by
construction at an unpadded head (the driver says so); it is the padded geometry that gives it
205–208 of 240.

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
- **The score table's 16 x 16 block is `[candidate, sender]`, first axis fastest** —
  `table[b + batch*(k + 16*(sender + 16*i))]` in `serial_tail_kernel`, `k` the candidate. A
  score-table builder that stores `[sender, candidate]` (as `run_serial_tail.py::score_table`
  does, harmlessly, because there the table is only an input) feeds every slot the *transposed*
  candidate set. Measured symptom in the first version of this harness: the table comparison
  came back at **2.2e9 raw fp32 ulps** (2 octaves of a float — not a numerics magnitude, the
  fingerprint of comparing the wrong elements), and only the layout fix brought it to the
  1e-7-of-the-matrix norm it now reports.
- `dspine_serial_scores` gates the head on `qtype`/`layout` **only** — so a fixture `Weight` with
  an empty `shape`/`padded_shape`/`ndim` passes it, and the first gate that reads the geometry
  (`require_head`) turns the empty `padded_shape[1]` into a dereference of `scales + 0` inside
  the kernel. Any new `Weight` a fixture builds must carry the geometry
  `weight_view.cpp::native_weight` would give it.
