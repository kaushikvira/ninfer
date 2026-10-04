"""All-NVFP4 quantization of d0xin Swift-Qwen3.8-27B-Uncensored-BF16.
Mirrors unsloth layout except attention/GDN also NVFP4 (nvfp4full-style).
Usage: qquant-python quantize_all_nvfp4.py [ncal=512]
"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SRC = sys.argv[1] if len(sys.argv) > 2 else "/home/kv/models/swift15-bf16"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/kv/models/swift15-nvfp4full-hf"
QCFG = json.load(open(os.path.join(os.path.dirname(__file__), "nvfp4full_qconfig.json")))
NCAL = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SEQ = 2048

print("=== target allocation ===", flush=True)
for g, v in QCFG["config_groups"].items():
    w = v["weights"]
    print(f"  {g}: w={w['num_bits']}b/{w['type']}/gs={w.get('group_size')} | {len(v['targets'])} patterns", flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu")
tok = AutoTokenizer.from_pretrained(SRC)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

recipe = QuantizationModifier(config_groups=QCFG["config_groups"], ignore=QCFG["ignore"])

t1 = time.time()
oneshot(
    model=model, tokenizer=tok,
    dataset="ultrachat-200k",
    splits={"calibration": f"train_sft[:{NCAL}]"},
    recipe=recipe,
    max_seq_length=SEQ, num_calibration_samples=NCAL,
    pipeline="sequential",  # onload one layer at a time -> fits 32GB
    output_dir=OUT, save_compressed=True,
)
print(f"DONE oneshot in {(time.time()-t1)/60:.1f} min -> {OUT}", flush=True)
