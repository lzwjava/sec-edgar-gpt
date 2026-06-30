# SEC-EDGAR GPT-2 124M

A GPT-2 (124M) language model trained from scratch on SEC EDGAR filings (10-K, 10-Q, 8-K, etc.).

## Model Details

| Property | Value |
|---|---|
| Architecture | GPT-2 124M (12 layers, 12 heads, 768 hidden) |
| Parameters | 124,475,904 |
| Context Length | 1,024 tokens |
| Tokenizer | GPT-2 BPE (tiktoken) |
| Training Tokens | ~1.55B (1 epoch) |
| Training Steps | 47,000 |
| Validation Loss | 2.28 |
| Training Framework | [nanoGPT](https://github.com/karpathy/nanoGPT) |
| Training Hardware | NVIDIA RTX 4070 12GB |
| Training Time | ~8 hours |
| Bias | No (`bias=False`) |

## Training Data

SEC EDGAR filings sourced from the [SEC-EDGAR corpus on HuggingFace](https://huggingface.co/datasets/kapilrao/SEC-EDGAR), covering annual reports (10-K), quarterly reports (10-Q), current reports (8-K), and other filing types. Tokenized with GPT-2 BPE into ~1.55B tokens across 16 shards.

## Training Config

- Batch size: 4 × 1024 tokens, gradient accumulation 8 → effective batch 32,768 tokens/step
- Optimizer: GPT-3 style (AdamW, lr=6e-4, warmup=2000, cosine decay to 6e-5)
- No dropout, no weight bias

## Usage

### Recommended: native nanoGPT inference

This model was trained with [nanoGPT](https://github.com/karpathy/nanoGPT), and inference works best with the same native code. The checkpoint format, weight layout (`bias=False`), and tokenizer (`tiktoken`) all match directly — no conversion layer needed.

Copy `model.py` from nanoGPT into the same directory, then:

```python
import torch
import tiktoken
from model import GPTConfig, GPT

# Load checkpoint (nanoGPT format)
checkpoint = torch.load("ckpt.pt", map_location="cuda", weights_only=False)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
# Strip compilation prefix if present
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to("cuda")

# Tokenizer — same tiktoken encoding used during training
enc = tiktoken.get_encoding("gpt2")

prompt = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION"
start_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
x = torch.tensor(start_ids, dtype=torch.long, device="cuda")[None, ...]
y = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=200)
output = enc.decode(y[0].tolist())
print(output)
```

> **Tip:** For a full OpenAI-compatible API server using this approach, see [`server/server.py`](server/server.py).

### HuggingFace transformers (not recommended)

HuggingFace `transformers` can load this model, but there are known issues:

- **`bias=False` mismatch:** nanoGPT trains all linear layers without bias (`bias=False`). HuggingFace's `GPT2LMHeadModel` initialises with `bias=True` by default. The shapes match only because the HF conversion script pads the state dict — but you may get silent quality degradation or warnings.
- **Checkpoint format:** The raw checkpoint is saved in nanoGPT's format, not HuggingFace's. The HuggingFace Hub version goes through a conversion step that can introduce subtle mismatches.
- **Tokenizer differences:** HuggingFace wraps `GPT2Tokenizer` around the same BPE merges, but the `encode`/`decode` behaviour (special token handling, whitespace) can differ from the `tiktoken` library used during training. For best fidelity, use `tiktoken` directly.
- **`generate()` defaults:** HF's `model.generate()` defaults differ from nanoGPT's `generate()` — notably no `top_k` by default, different repetition penalty handling. Results will not be identical.

If you still want to try:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("lzwjava/sec-edgar-gpt")
tokenizer = GPT2Tokenizer.from_pretrained("lzwjava/sec-edgar-gpt")

prompt = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=200, temperature=0.8, do_sample=True, top_k=200)
print(tokenizer.decode(output[0]))
```

## Limitations

- Trained for only 1 epoch — coherent for ~200-500 tokens before repetitive loops
- No instruction tuning or RLHF — raw language model
- 124M parameters is small; don't expect state-of-the-art quality
- GPT-2 tokenizer may not handle all financial notation optimally

## Training Code

Trained with [nanoGPT](https://github.com/karpathy/nanoGPT). Training config available in the [source repo](https://github.com/lzwjava/sec-edgar-gpt).

## Development Notes

| Date | Topic | Notes |
|------|-------|-------|
| 2026-06-25 | [10-K Download Summary](notes/2026-06-25-sec-edgar-10-k-download-summary-en.md) | SEC EDGAR 10-K filing download process |
| 2026-06-25 | [Financial Pretraining Corpus](notes/2026-06-25-sec-edgar-financial-pretraining-corpus-en.md) | Corpus preparation and tokenization |
| 2026-06-26 | [GPT-2 on SEC-EDGAR Data](notes/2026-06-26-gpt-2-on-sec-edgar-data-en.md) | Paper structure and training overview |
| 2026-06-26 | [Training Loss Recovery](notes/2026-06-26-sec-edgar-training-loss-recovery-en.md) | Loss spike at 20k steps, recovery analysis |
| 2026-06-26 | [Prompt File Setup](notes/2026-06-26-sec-edgar-prompt-file-setup-en.md) | Inference prompt configuration |
| 2026-06-26 | [Model Quality Check](notes/2026-06-26-sec-edgar-model-quality-check-en.md) | Output quality evaluation |
| 2026-06-26 | [124M Generation Test](notes/2026-06-26-sec-edgar-124m-generation-test-en.md) | Generation samples across prompts |
| 2026-06-26 | [124M Generation Review](notes/2026-06-26-sec-edgar-124m-generation-review-en.md) | Detailed review of generated outputs |
| 2026-06-26 | [124M Upload](notes/2026-06-26-sec-edgar-gpt-124m-upload-en.md) | Model upload to HuggingFace |

## Citation

```bibtex
@misc{sec-edgar-gpt-124m,
  author = {Zhiwei Li},
  title = {SEC-EDGAR GPT-2 124M},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/lzwjava/sec-edgar-gpt}
}
```
