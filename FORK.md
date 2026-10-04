# Fork notes — kaushikvira/ninfer

This fork is **upstream `Neroued/ninfer` master + exactly one engine commit**, plus one
tooling commit for artifact grafting.

## Model built with this fork

Grafted artifact (cometkim nvfp4full v1 + z-lab DFlash2 W8G32/BF16):
[kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer)

Sanitized serving profile (as used on our rig — KV auto/k8v4, DFlash2 K=7,
vision, host-KV 32 GiB, sampling defaults; **no secrets**):
[examples/ninfer-nvfp4full-grafted-dflash2.cfg.example](examples/ninfer-nvfp4full-grafted-dflash2.cfg.example)

## Engine delta vs upstream (`main` here)

| Commit | Type | What |
|---|---|---|
| `2eb59dbc` | **engine (the +1)** | `feat: register qwen3.8-27b/nvfp4full weights profile (port of cometkim feat/qwen3.8-nvfp4full)` — 4 files: `package.h`, `package.cpp`, `bindings.cpp`, `variant.cpp`. Enables the cometkim `qwen3.8-27b/nvfp4full` weight profile (full-NVFP4 27B) on upstream master. |
| `4c520495` | tooling | `tools: graft z-lab DFlash2 module (W8G32/BF16) onto cometkim nvfp4full v1` — `tools/artifact/graft_dflash2_w8.py`, used to build DFlash2-capable artifacts (see the HF model card), no engine changes. |

Everything else is byte-identical to upstream `487f8977` (verify: `git log master..origin/master`).

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
