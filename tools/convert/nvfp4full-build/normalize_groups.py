"""Normalize llmcompressor NVFP4 global scales per packing group so the
ninfer converter can fuse attention parents (engine native A4 route requires
one contiguous parent region per fused group).

Dequant contract (converter): value = code * e4m3_scale / weight_global_scale.
For a group with members i and shared divisor D: scale_i' = scale_i * D / d_i
re-encoded to E4M3 (RNE) keeps values identical up to E4M3 re-encode noise.
We pick D = min(d_i) so scales only shrink (no E4M3 overflow).

Groups per layer L (HF names):
  attn: self_attn.{q,k,v}_proj       (engine group: query,key,gate,value; q covers query+gate)
  gdn:  linear_attn.{in_proj_qkv,in_proj_z}
  mlp:  mlp.{gate,up}_proj
"""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import safe_open, save_file

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/kv/models/swift-abliterated-nvfp4full-hf")

index = json.load(open(CKPT / "model.safetensors.index.json"))["weight_map"]
files = sorted(set(index.values()))

def tensor(name):
    f = index[name]
    with safe_open(CKPT / f, framework="pt") as fh:
        return fh.get_tensor(name)

def all_tensor_names():
    return set(index)

def group_members(layer, kind):
    p = f"model.language_model.layers.{layer}."
    if kind == "attn":
        return [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"]
    if kind == "gdn":
        return [p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z"]
    if kind == "mlp":
        return [p + "mlp.gate_proj", p + "mlp.up_proj"]

names = all_tensor_names()
layers = list(range(64))  # all layers: linear layers lack self_attn but have gdn groups

report = []
updates = {}  # tensor name -> new tensor
for layer in layers:
    for kind in ("attn", "gdn", "mlp"):
        members = [m for m in group_members(layer, kind) if m + ".weight_global_scale" in names]
        if len(members) < 2:
            continue
        gs = {m: float(tensor(m + ".weight_global_scale")) for m in members}
        D = min(gs.values())
        for m in members:
            d = gs[m]
            scale = tensor(m + ".weight_scale")  # e4m3 (n, k/16)
            s32 = scale.float()
            new = (s32 * (D / d)).to(torch.float8_e4m3fn)  # shrink-only: no E4M3 overflow
            # dequant-correctness guard: relative change of decoded scale
            rel = ((new.float() - s32 * (d / D)).abs() / (s32 * (d / D)).clamp_min(1e-30)).max().item()
            report.append((layer, kind, m.split(".")[-1], d, D, rel))
            updates[m + ".weight_scale"] = new
            updates[m + ".weight_global_scale"] = torch.tensor(D, dtype=torch.float32)
        # input globals: members read the same activation tensor -> unify to min
        ig = {m: float(tensor(m + ".input_global_scale")) for m in members
              if m + ".input_global_scale" in names}
        if ig:
            Di = min(ig.values())
            for m in ig:
                updates[m + ".input_global_scale"] = torch.tensor(Di, dtype=torch.float32)

# write back: group updates by shard file
by_file = {}
for name, t in updates.items():
    by_file.setdefault(index[name], {})[name] = t
backup = CKPT / "pre_normalize_backup"
backup.mkdir(exist_ok=True)
for fname, upd in by_file.items():
    import shutil
    shutil.copy2(CKPT / fname, backup / fname)
    with safe_open(CKPT / fname, framework="pt") as fh:
        data = {k: fh.get_tensor(k) for k in fh.keys()}
    data.update(upd)
    save_file(data, CKPT / fname, metadata={"format": "pt"})

worst = max(r[5] for r in report)
print(f"groups normalized: {len(report)}  worst scale re-encode rel error: {worst:.4%}")
print(f"updated {len(updates)} tensors across {len(by_file)} shards; backup in {backup}")
for r in report[:6]:
    print(f"  L{r[0]:2d} {r[1]:4s} {r[2]:12s} d={r[3]:.6g} D={r[4]:.6g}")
