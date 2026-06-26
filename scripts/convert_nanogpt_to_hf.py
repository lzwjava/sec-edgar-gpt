#!/usr/bin/env python3
"""Convert nanoGPT-layout safetensors to proper HuggingFace GPT2LMHeadModel format.

The source model has transposed linear weights and no biases (nanoGPT convention).
This script transposes the weights to transformers layout and adds zero biases.
"""
import torch
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DIR = "/workspace/model/hf-model"
DST_DIR = "/workspace/model/hf-model-converted"

def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    # Load source tensors
    print("Loading source tensors...")
    tensors = {}
    with safe_open(str(src / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Convert weights
    new_tensors = {}
    for key, tensor in tensors.items():
        if key == "transformer.wte.weight" or key == "transformer.wpe.weight" or key == "lm_head.weight":
            # Embedding and lm_head: keep as-is
            new_tensors[key] = tensor
        elif key.endswith(".weight"):
            # Linear layer weights: transpose from nanoGPT layout to transformers layout
            # nanoGPT: [out_features, in_features] -> transformers: [in_features, out_features]
            new_tensors[key] = tensor.t().contiguous()
        else:
            new_tensors[key] = tensor

    # Add missing biases (zeros)
    # Layer norm biases
    n_layer = 12
    n_embd = 768
    n_inner = 3072

    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        # ln_1 bias
        k = f"{prefix}.ln_1.bias"
        if k not in new_tensors:
            new_tensors[k] = torch.zeros(n_embd)
            print(f"  Added: {k}")

        # ln_2 bias
        k = f"{prefix}.ln_2.bias"
        if k not in new_tensors:
            new_tensors[k] = torch.zeros(n_embd)
            print(f"  Added: {k}")

        # attn.c_attn bias
        k = f"{prefix}.attn.c_attn.bias"
        if k not in new_tensors:
            new_tensors[k] = torch.zeros(3 * n_embd)  # 2304
            print(f"  Added: {k}")

        # attn.c_proj bias
        k = f"{prefix}.attn.c_proj.bias"
        if k not in new_tensors:
            new_tensors[k] = torch.zeros(n_embd)
            print(f"  Added: {k}")

        # mlp.c_fc bias
        k = f"{prefix}.mlp.c_fc.bias"
        if k not in new_tensors:
            new_tensors[k] = torch.zeros(n_inner)
            print(f"  Added: {k}")

        # mlp.c_proj bias
        k = f"{prefix}.mlp.c_proj.bias"
        if k not in new_tensors:
            new_tensors[k] = torch.zeros(n_embd)
            print(f"  Added: {k}")

    # transformer.ln_f bias
    k = "transformer.ln_f.bias"
    if k not in new_tensors:
        new_tensors[k] = torch.zeros(n_embd)
        print(f"  Added: {k}")

    # Save converted model
    print(f"\nSaving to {dst}...")
    save_file(new_tensors, str(dst / "model.safetensors"))

    # Copy config with proper transformers metadata
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": 50257,
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

    # Copy tokenizer files
    import shutil
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)
            print(f"  Copied: {fname}")

    # Verify shapes
    print("\nVerifying converted shapes...")
    with safe_open(str(dst / "model.safetensors"), framework="pt") as f:
        for key in sorted(f.keys())[:12]:
            print(f"  {key}: {f.get_tensor(key).shape}")

    print(f"\nDone! Converted model at {dst}")
    print(f"  Vocab size: 50257 (no padding)")
    print(f"  All biases: present (zeros)")
    print(f"  Linear weights: transposed to transformers layout")

if __name__ == "__main__":
    main()
