#!/usr/bin/env python3
"""Convert old HF model (vocab=50304) to proper format (vocab=50257) on RunPod.

The old model at hf-model-old has correct weight shapes (transformers layout)
but padded vocab (50304). Just trim embeddings + lm_head and save clean.
"""
import torch
import json
import shutil
from pathlib import Path
from safetensors.torch import save_file
from safetensors import safe_open

SRC_DIR = "/workspace/model/hf-model-old"
DST_DIR = "/workspace/model/hf-model"
ACTUAL_VOCAB = 50257

def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    print("Loading old model tensors...")
    tensors = {}
    with safe_open(str(src / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    print(f"Keys: {len(tensors)}")
    padded_vocab = tensors["transformer.wte.weight"].shape[0]
    print(f"Vocab: {padded_vocab} -> {ACTUAL_VOCAB}")

    # Trim vocab for wte and lm_head
    new_tensors = {}
    for key, t in tensors.items():
        if key in ("transformer.wte.weight", "lm_head.weight"):
            new_tensors[key] = t[:ACTUAL_VOCAB].clone().contiguous()
        else:
            new_tensors[key] = t.contiguous()

    print("\nSaving trimmed model.safetensors...")
    save_file(new_tensors, str(dst / "model.safetensors"))

    # Write clean config
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": ACTUAL_VOCAB,
        "n_positions": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_inner": 3072,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": None,
        "tie_word_embeddings": True,
        "scale_attn_weights": True,
        "use_cache": True,
        "dtype": "float32",
        "transformers_version": "5.5.0",
    }
    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")

    # Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        s = src / fname
        if s.exists():
            shutil.copy2(s, dst / fname)
            print(f"Copied: {fname}")

    # Verify
    print("\n=== Verification ===")
    with safe_open(str(dst / "model.safetensors"), framework="pt") as f:
        for key in sorted(f.keys())[:6]:
            print(f"  {key}: {f.get_tensor(key).shape}")
    print("  ...")

    sz = (dst / "model.safetensors").stat().st_size
    print(f"\nDone! {DST_DIR}")
    print(f"  Size: {sz / 1024 / 1024:.1f} MB")
    print(f"  Vocab: {ACTUAL_VOCAB}")

if __name__ == "__main__":
    main()
