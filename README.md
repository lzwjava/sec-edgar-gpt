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

SEC EDGAR filings sourced from the [SEC-EDGAR corpus on HuggingFace](https://huggingface.co/datasets/eltorio/SEC-EDGAR-Corpus), covering annual reports (10-K), quarterly reports (10-Q), current reports (8-K), and other filing types. Tokenized with GPT-2 BPE into ~1.55B tokens across 16 shards.

## Training Config

- Batch size: 4 × 1024 tokens, gradient accumulation 8 → effective batch 32,768 tokens/step
- Optimizer: GPT-3 style (AdamW, lr=6e-4, warmup=2000, cosine decay to 6e-5)
- No dropout, no weight bias

## Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("lzwjava/sec-edgar-gpt")
tokenizer = GPT2Tokenizer.from_pretrained("lzwjava/sec-edgar-gpt")

prompt = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=200, temperature=0.8, do_sample=True)
print(tokenizer.decode(output[0]))
```

## Limitations

- Trained for only 1 epoch — coherent for ~200-500 tokens before repetitive loops
- No instruction tuning or RLHF — raw language model
- 124M parameters is small; don't expect state-of-the-art quality
- GPT-2 tokenizer may not handle all financial notation optimally

## Training Code

Trained with [nanoGPT](https://github.com/karpathy/nanoGPT). Training config available in the [source repo](https://github.com/lzwjava/sec-edgar-gpt).

## Citation

```bibtex
@misc{sec-edgar-gpt-124m,
  author = {Zhiwei Li},
  title = {SEC-EDGAR GPT-2 124M},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/lzwjava/sec-edgar-gpt}
}
```
