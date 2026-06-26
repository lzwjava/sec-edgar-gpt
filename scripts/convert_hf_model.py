#!/usr/bin/env python3
"""Convert HF model with nanoGPT layout to proper transformers layout.

Source: /workspace/model/hf-model-new (nanoGPT layout, vocab=50257, no biases)
Dest:   /workspace/model/hf-model (transformers layout, vocab=50257, with zero biases)
"""
import torch
import json
import shutil
from pathlib import Path
from safetensors.torch import save_file
from safetensors import safe_open

SRC_DIR = "/workspace/model/hf-model-new"
DST_DIR = "/workspace/model/hf-model"

def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    print("Loading source model tensors...")
    tensors = {}
    with safe_open(str(src / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    print(f"Keys: {len(tensors)}")

    # Check if weights need transposing
    # nanoGPT: c_attn.weight is [2304, 768], transformers: [768, 2304]
    sample_key = "transformer.h.0.attn.c_attn.weight"
    sample_shape = tensors[sample_key].shape
    print(f"Sample: {sample_key}: {sample_shape}")

    needs_transpose = sample_shape[0] > sample_shape[1]  # [2304, 768] -> True
    print(f"Needs transpose: {needs_transpose}")

    n_embd = 768
    n_inner = 3072
    n_layer = 12
    ACTUAL_VOCAB = 50257

    new_tensors = {}
    for key, t in tensors.items():
        if key == "transformer.wte.weight" or key == "transformer.wpe.weight" or key == "lm_head.weight":
            new_tensors[key] = t.contiguous()
        elif key.endswith(".weight") and needs_transpose:
            # Transpose linear weights: nanoGPT [out, in] -> transformers [in, out]
            new_tensors[key] = t.t().contiguous()
        else:
            new_tensors[key] = t.contiguous()

    # Add missing biases (zeros)
    for i in range(n_layer):
        sp = f"transformer.h.{i}"
        for name, size in [
            ("ln_1.bias", n_embd),
            ("ln_2.bias", n_embd),
            ("attn.c_attn.bias", 3 * n_embd),
            ("attn.c_proj.bias", n_embd),
            ("mlp.c_fc.bias", n_inner),
            ("mlp.c_proj.bias", n_embd),
        ]:
            k = f"{sp}.{name}"
            if k not in new_tensors:
                new_tensors[k] = torch.zeros(size)
                print(f"  Added: {k}")

    k = "transformer.ln_f.bias"
    if k not in new_tensors:
        new_tensors[k] = torch.zeros(n_embd)
        print(f"  Added: {k}")

    # Handle shared tensors (tie_word_embeddings=True means wte and lm_head share memory)
    # Clone lm_head to avoid shared memory issue
    if "lm_head.weight" in new_tensors and "transformer.wte.weight" in new_tensors:
        new_tensors["lm_head.weight"] = new_tensors["lm_head.weight"].clone()

    # Save
    print("\nSaving converted model.safetensors...")
    save_file(new_tensors, str(dst / "model.safetensors"))

    # Config
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": ACTUAL_VOCAB,
        "n_positions": 1024,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": 12,
        "n_inner": n_inner,
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
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        s = src / fname
        if s.exists():
            shutil.copy2(s, dst / fname)
            print(f"Copied: {fname}")

    # Verify
    print("\n=== Verification ===")
    with safe_open(str(dst / "model.safetensors"), framework="pt") as f:
        for key in sorted(f.keys())[:8]:
            print(f"  {key}: {f.get_tensor(key).shape}")
    print("  ...")

    sz = (dst / "model.safetensors").stat().st_size
    print(f"\nDone! {DST_DIR}")
    print(f"  Size: {sz / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
