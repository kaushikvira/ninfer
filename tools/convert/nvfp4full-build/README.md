# nvfp4full artifact build pipeline (fork-local, rebase-survivable)

Owner: kaushikvira. These scripts produced the published all-NVFP4 NInfer
artifacts (`kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3` and
`…swift-abliterated…`). Keep this directory when rebasing onto upstream —
upstream has no equivalent. Also mirrored on branch `contrib/nvfp4full-build`
(push it after rebases) and in the serving repo
(`v-llm-gateway/bench/build-swift-abliterated/`, `build-swift15/`).

## Pipeline (BF16 source -> published .ninfer)

```bash
PY=~/venvs/qquant/bin/python   # torch cu130 + llm-compressor + transformers + safetensors

# 1. all-NVFP4 quantization of the BF16 source (GPU, ~12 min + save on 5090)
$PY tools/convert/nvfp4full-build/quantize_all_nvfp4.py <BF16-src-dir> <out-hf-dir> [ncal=512]
#    - edit SRC/OUT at top of the script (or pass args after adapting)
#    - qconfig: nvfp4full_qconfig.json — W4A4 gs16 on ALL text projections
#      (self_attn q/k/v/o, linear_attn in_proj_qkv/z/out_proj, mlp gate/up/down),
#      lm_head FP8, vision ignored. Uses llm-compressor `nvfp4-pack-quantized`
#      + strategy tensor_group + dynamic "local" — REQUIRED so each module
#      stores weight_global_scale/input_global_scale (plain group strategy
#      drops the global scales and the converter then fails).

# 2. normalize packing-group divisors (CPU, ~3 min) — THE critical step
$PY tools/convert/nvfp4full-build/normalize_groups.py <out-hf-dir>
#    - engine native A4 route requires one contiguous fused parent per group
#      (GDN qkvz 16384 rows, attn qkgv 14336, mlp gate+up). llm-compressor
#      gives each module its own weight_global_scale -> converter refuses to
#      coalesce. Unify D=min(d_i), rescale E4M3 block scales by D/d_i (RNE,
#      shrink-only, no overflow). Verify 0 unnormalized groups afterwards.
#    - iterates ALL 64 layers (linear layers lack self_attn — do not derive
#      the layer list from self_attn names; that bug cost us one rebuild).

# 3. convert to v3 (CPU, ~1 min)
cd $NINFER_REPO && $PY -m tools.convert \
  --model <BF16-src-dir> \
  --source quantized=<out-hf-dir> \
  --source dflash2=<z-lab DFlash2 dir> \
  --recipe tools/convert/nvfp4full-build/recipe_swift_full.py \
  --components text,vision,mtp,dflash2 --proposal \
  --name <instance-name> \
  --out <artifact>.ninfer
#    - recipe_swift_full.py: text all-NVFP4 import_encoded (auto-fused after
#      step 2), token_embedding + output_head Q8 W8G32, vision official
#      _optional allocation, gdn a/b projections BF16. `--proposal` is
#      REQUIRED when serving with --lm-head-draft.

# 4. verify + publish
$PY -m tools.artifact.inspect <artifact>.ninfer     # check formats/objects
sha256sum <artifact>.ninfer > SHA256SUMS            # publish with the card
```

## History

- 2026-09-24: pipeline written for `d0xin/Swift-Qwen3.8-27B-Uncensored-BF16`
  (Swift-1.0) -> `kaushikvira/Qwen3.8-27B-swift-abliterated-nvfp4full-dflash2-NInfer-v3`
  (sha256 74c97213…). A/B: IFBench +1.3..+4.4pp vs cometkim nvfp4full profile.
- 2026-09-25: applied to `ukisai/Swift-1.5-Qwen3.8-27b` ->
  `kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3`
  (sha256 16f313c0…). A/B: IFBench +2.7..+4.3pp over the Swift-1.0 build,
  decode +8.1%. Adopted as production.
- Abliterated Swift-1.5 variant is pending (capture_direction.py +
  refusal-direction removal are prototyped in v-llm-gateway/build-swift15/).
