#!/usr/bin/env python3
"""Convert nanoGPT checkpoint to proper HuggingFace GPT2LMHeadModel format.

Source: /mnt/data/nanoGPT/out-sec-edgar-124m/ckpt.pt (nanoGPT, bias=False, vocab=50304)
Dest:   /mnt/data/sec-edgar-gpt-124m-model/ (HF format, vocab_size=50257, with zero biases)
"""
import torch
import json
import shutil
from pathlib import Path
from safetensors.torch import save_file
from safetensors import safe_open

SRC_CKPT = "/mnt/data/nanoGPT/out-sec-edgar-124m/ckpt.pt"
SRC_TOKENIZER = "/mnt/data/sec-edgar-gpt-124m/tokenizer.json"
SRC_TOKENIZER_CFG = "/mnt/data/sec-edgar-gpt-124m/tokenizer_config.json"
SRC_GEN_CFG = "/mnt/data/sec-edgar-gpt-124m/generation_config.json"
DST_DIR = "/mnt/data/sec-edgar-gpt-124m-model"

ACTUAL_VOCAB = 50257

def main():
    dst = Path(DST_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    print("Loading nanoGPT checkpoint...")
    ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    cfg = ckpt.get("config", {})
    has_bias = cfg.get("bias", False)

    print(f"Config: bias={has_bias}, n_layer={cfg.get('n_layer')}, n_embd={cfg.get('n_embd')}")
    print(f"Checkpoint keys: {len(sd)}")

    padded_vocab = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 12)
    n_embd = cfg.get("n_embd", 768)
    n_inner = cfg.get("n_embd", 768) * 4  # 3072

    print(f"Vocab: {padded_vocab} -> {ACTUAL_VOCAB}")
    print(f"Layers: {n_layer}, Embd: {n_embd}, Inner: {n_inner}")

    new_sd = {}

    # Embeddings (trim vocab padding)
    new_sd["transformer.wte.weight"] = sd["transformer.wte.weight"][:ACTUAL_VOCAB].contiguous()
    new_sd["transformer.wpe.weight"] = sd["transformer.wpe.weight"].contiguous()

    for i in range(n_layer):
        sp = f"transformer.h.{i}"

        # Layer norms
        new_sd[f"{sp}.ln_1.weight"] = sd[f"{sp}.ln_1.weight"].contiguous()
        new_sd[f"{sp}.ln_1.bias"] = sd.get(f"{sp}.ln_1.bias", torch.zeros(n_embd))
        new_sd[f"{sp}.ln_2.weight"] = sd[f"{sp}.ln_2.weight"].contiguous()
        new_sd[f"{sp}.ln_2.bias"] = sd.get(f"{sp}.ln_2.bias", torch.zeros(n_embd))

        # Attention: nanoGPT [out, in] -> transformers GPT2Conv1D [in, out]
        w = sd[f"{sp}.attn.c_attn.weight"]  # [2304, 768]
        new_sd[f"{sp}.attn.c_attn.weight"] = w.t().contiguous()  # [768, 2304]
        new_sd[f"{sp}.attn.c_attn.bias"] = sd.get(f"{sp}.attn.c_attn.bias", torch.zeros(3 * n_embd))

        w = sd[f"{sp}.attn.c_proj.weight"]  # [768, 768]
        new_sd[f"{sp}.attn.c_proj.weight"] = w.t().contiguous()  # [768, 768]
        new_sd[f"{sp}.attn.c_proj.bias"] = sd.get(f"{sp}.attn.c_proj.bias", torch.zeros(n_embd))

        # MLP: same transpose
        w = sd[f"{sp}.mlp.c_fc.weight"]  # [3072, 768]
        new_sd[f"{sp}.mlp.c_fc.weight"] = w.t().contiguous()  # [768, 3072]
        new_sd[f"{sp}.mlp.c_fc.bias"] = sd.get(f"{sp}.mlp.c_fc.bias", torch.zeros(n_inner))

        w = sd[f"{sp}.mlp.c_proj.weight"]  # [768, 3072]
        new_sd[f"{sp}.mlp.c_proj.weight"] = w.t().contiguous()  # [3072, 768]
        new_sd[f"{sp}.mlp.c_proj.bias"] = sd.get(f"{sp}.mlp.c_proj.bias", torch.zeros(n_embd))

    # Final layer norm
    new_sd["transformer.ln_f.weight"] = sd["transformer.ln_f.weight"].contiguous()
    new_sd["transformer.ln_f.bias"] = sd.get("transformer.ln_f.bias", torch.zeros(n_embd))

    # lm_head (trim vocab, clone to avoid shared memory with wte)
    new_sd["lm_head.weight"] = sd["lm_head.weight"][:ACTUAL_VOCAB].clone().contiguous()

    # Save
    print("\nSaving model.safetensors...")
    save_file(new_sd, str(dst / "model.safetensors"))

    # Config
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": ACTUAL_VOCAB,
        "n_positions": cfg.get("n_positions", 1024),
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": cfg.get("n_head", 12),
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
    for src_path in [SRC_TOKENIZER, SRC_TOKENIZER_CFG, SRC_GEN_CFG]:
        p = Path(src_path)
        if p.exists():
            shutil.copy2(p, dst / p.name)
            print(f"Copied: {p.name}")

    # Verify
    print("\n=== Verification ===")
    with safe_open(str(dst / "model.safetensors"), framework="pt") as f:
        for key in sorted(f.keys()):
            t = f.get_tensor(key)
            print(f"  {key}: {t.shape}")

    sz = (dst / "model.safetensors").stat().st_size
    print(f"\nDone! {DST_DIR}")
    print(f"  Size: {sz / 1024 / 1024:.1f} MB")
    print(f"  Vocab: {ACTUAL_VOCAB} (trimmed from {padded_vocab})")
    print(f"  Biases: zeros (original training had bias=False)")
    print(f"  Weights: transposed to transformers GPT2Conv1D layout")

if __name__ == "__main__":
    main()
