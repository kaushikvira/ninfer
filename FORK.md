# Fork notes — kaushikvira/ninfer

This fork is **upstream `Neroued/ninfer` master `487f8977` + the carried commits
below** (one own engine commit, one tooling commit, and temporary upstream-PR
ports that get dropped when they merge upstream). The authoritative registry
with sources and drop triggers is [`PATCHES.md`](PATCHES.md).

## Model built with this fork

Grafted artifact (cometkim nvfp4full v1 + z-lab DFlash2 W8G32/BF16):
[kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer)

Sanitized serving profile (as used on our rig — KV auto/k8v4, DFlash2 K=7,
vision, host-KV 32 GiB, sampling defaults; **no secrets**):
[examples/ninfer-nvfp4full-grafted-dflash2.cfg.example](examples/ninfer-nvfp4full-grafted-dflash2.cfg.example)

## Carried commits vs upstream (`main` here)

| Commit | Type | What |
|---|---|---|
| `2eb59dbc` | **engine (own)** | register `qwen3.8-27b/nvfp4full` weights profile (port of cometkim `feat/qwen3.8-nvfp4full`) — 4 files: `package.h`, `package.cpp`, `bindings.cpp`, `variant.cpp`. Enables the cometkim full-NVFP4 27B profile on upstream master. Keep. |
| `65bbf8b5` | tooling | DFlash2 graft tool `tools/artifact/graft_dflash2_w8.py` + fork notes + sanitized serving example — builds DFlash2-capable artifacts (see the HF model card), no engine changes. Keep. |
| `71b4c9b2` | engine (temp) | OpenAI Responses: accept `include=reasoning.encrypted_content` + `reasoning.summary` (cherry-pick of upstream PR #148, `193dab17`). Drop when #148 merges. |
| `bcc6261e` | engine (temp) | skip summary/encrypted-only reasoning **input** Items (Inspect AI multi-turn; ours, derived from #148 which only handled the create side). Drop when #148 lands + upstream covers the input side. |
| `e202c53b` | build (temp) | ccache + BuildKit cache mount for incremental docker builds (port of upstream PR #97, `03df31d5`). Drop when #97 merges. |
| `7e8ad2e9` | engine (temp) | NVFP4 TMA route reads activation scales tile-contiguous (port of upstream PR #160, `545f64b0`; +1–2.5% prefill on our box). Drop when #160 merges. |
| `4ac61fa2` | engine (temp) | `--image-token-budget N` per-image Vision-token ceiling + our validator fix (port of upstream PR #61, `eb413c76`). Drop when #61 merges. |
| `6e9e928a` | docs | patch registry + fork policy (`PATCHES.md`). |
| `8a42a465` | build | curl in the runtime image (container healthcheck support). |

Everything else is byte-identical to upstream `487f8977` (verify: `git log
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
