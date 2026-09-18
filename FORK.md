# Fork notes — kaushikvira/ninfer

This fork is **upstream `Neroued/ninfer` master `f76e19c0` (v3 model/weight
decoupling) + the carried commits below** (temporary upstream-PR ports that get
dropped when they merge, plus tooling/docs). The authoritative registry with
sources and drop triggers is [`PATCHES.md`](PATCHES.md).

> **v3 rebase note (2026-09-18):** rebasing onto upstream v3 dropped two carried
> commits. The C++ `nvfp4full` weights-profile registration (`2eb59dbc`) is now
> dead code — v3's loader is data-driven, so `nvfp4full` lives entirely in the
> artifact (upgraded to a v3 container with `tools/upgrade_ninfer_v2_to_v3.py`,
> which this fork extends to allow-list it). The PR #160 tile-contiguous-scales
> port (`7e8ad2e9`) is superseded by upstream `1d8587bc`.

## Model built with this fork

Grafted artifact (cometkim nvfp4full v1 + z-lab DFlash2 W8G32/BF16):
[kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer)

Sanitized serving profile (as used on our rig — KV auto/k8v4, DFlash2 K=7,
vision, host-KV 32 GiB, sampling defaults; **no secrets**):
[examples/ninfer-nvfp4full-grafted-dflash2.cfg.example](examples/ninfer-nvfp4full-grafted-dflash2.cfg.example)

## Carried commits vs upstream (`main` here)

| Commit | Type | What |
|---|---|---|
| `50e04987` | tooling | DFlash2 graft tool `tools/artifact/graft_dflash2_w8.py` + fork notes + sanitized serving example — builds DFlash2-capable artifacts (see the HF model card), no engine changes. Keep. |
| `9dcdce39` | engine (temp) | OpenAI Responses: accept `include=reasoning.encrypted_content` + `reasoning.summary` (cherry-pick of upstream PR #148, `193dab17`). Drop when #148 merges. |
| `25de38a5` | engine (temp) | skip summary/encrypted-only reasoning **input** Items (Inspect AI multi-turn; ours, derived from #148 which only handled the create side). Drop when #148 lands + upstream covers the input side. |
| `3ca1a001` | build (temp) | ccache + BuildKit cache mount for incremental docker builds (port of upstream PR #97, `03df31d5`). Drop when #97 merges. |
| `f26621be` | engine (temp) | `--image-token-budget N` per-image Vision-token ceiling (port of upstream PR #61, `eb413c76`), re-anchored onto v3's `processor_options`/`FrontendOptions` chain. Drop when #61 merges. |
| `cba96bcc` | build | curl in the runtime image (container healthcheck support). |
| `775f1e0b` | docs | patch registry + fork policy (`PATCHES.md`). |
| `3550b95f` | docs | `FORK.md` carried-commit delta. |
| *(tools)* | tooling (temp) | `tools/upgrade_ninfer_v2_to_v3.py`: allow-list `qwen3.8-27b/nvfp4full` so our artifact upgrades to a v3 container. Drop when upstream registers `nvfp4full`. |

Everything else is byte-identical to upstream `f76e19c0` (verify: `git log
origin/master..main` should show only the commits above).

## Graft tool

```bash
python3 -m tools.artifact.graft_dflash2_w8 \
  --artifact models/qwen3_8_27b_nvfp4full.ninfer \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b_nvfp4full-dflash2.ninfer
```

Appends the z-lab/Qwen3.8-27B-DFlash2 module (66 objects: W8G32_F16S matrices +
BF16 norms/conv bases/codebooks) to the cometkim nvfp4full v1 image; the base
tensors are copied byte-for-byte, MTP objects stay (validate-only under
`--spec dflash2`), and DFlash2 selects with:

```
--spec dflash2 --draft-tokens 7
```

Requires upstream master (or this fork) — DFlash2 binding landed upstream via
`385b30ce`.

## Build

```bash
docker build -t ninfer-master:local .
# or: cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
#      -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF \
#      && cmake --build build --parallel --target ninfer ninfer-serve
```

Requires CUDA 13.1, RTX 5090 (`sm_120a`), Linux.
