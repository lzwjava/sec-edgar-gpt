#!/usr/bin/env python3
"""OpenAI-compatible API server using native nanoGPT model (no transformers)."""
import os
import sys
import json
import torch
import time
import uuid
import datetime

# Add model directory to path so we can import model.py
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from model import GPTConfig, GPT

MODEL_DIR = os.path.dirname(__file__)
CKPT_PATH = os.path.join(MODEL_DIR, "ckpt.pt")
PROMPT_LOG = os.path.join(MODEL_DIR, "prompts.jsonl")
app = FastAPI()

print("Loading nanoGPT checkpoint...")
checkpoint = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
# Remove compiled model prefix if present
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to("cuda")

# Use tiktoken (GPT-2 BPE) — same as nanoGPT sample.py
import tiktoken
enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token  # 50256

print(f"Model loaded on GPU. Vocab: {gptconf.vocab_size}, params: {model.get_num_params()/1e6:.1f}M")

MIN_PROMPT_CHARS = 10
BLOCK_SIZE = gptconf.block_size  # 1024


def _log_interaction(req_id, req_model, endpoint, prompt_text, response_text, usage, latency_ms):
    """Log a full request/response pair to prompts.jsonl."""
    record = {
        "id": req_id,
        "model": req_model,
        "endpoint": endpoint,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "prompt": prompt_text,
        "response": response_text,
        "usage": usage,
        "latency_ms": round(latency_ms, 1),
    }
    try:
        with open(PROMPT_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # don't crash the server if logging fails


def _generate(input_ids, max_new_tokens, temperature, top_k):
    """Generate using native nanoGPT generate()."""
    return model.generate(input_ids, max_new_tokens, temperature=temperature, top_k=top_k)


class CompletionRequest(BaseModel):
    model: str = "sec-edgar-gpt-124m"
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.8
    top_k: int = 200
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "sec-edgar-gpt-124m"
    messages: List[ChatMessage]
    max_tokens: int = 1000
    temperature: float = 0.8
    top_k: int = 200
    stream: bool = False


@app.get("/")
async def index():
    html_path = os.path.join(MODEL_DIR, "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if len(req.prompt.strip()) < MIN_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt must be at least {MIN_PROMPT_CHARS} characters")
    start_ids = enc.encode(req.prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(start_ids, dtype=torch.long, device="cuda")[None, ...]
    t0 = time.time()
    y = _generate(x, req.max_tokens, req.temperature, req.top_k)
    latency = (time.time() - t0) * 1000
    gen_ids = y[0][len(start_ids):]
    text = enc.decode(gen_ids.tolist())
    result = {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(start_ids),
            "completion_tokens": len(gen_ids),
            "total_tokens": len(start_ids) + len(gen_ids),
        },
    }
    _log_interaction(
        req_id=result["id"], req_model=req.model, endpoint="completions",
        prompt_text=req.prompt, response_text=text,
        usage=result["usage"], latency_ms=latency,
    )
    return result


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    user_text = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    if len(user_text.strip()) < MIN_PROMPT_CHARS:
        raise HTTPException(400, f"Message must be at least {MIN_PROMPT_CHARS} characters")
    start_ids = enc.encode(user_text, allowed_special={"<|endoftext|>"})
    x = torch.tensor(start_ids, dtype=torch.long, device="cuda")[None, ...]
    t0 = time.time()
    y = _generate(x, req.max_tokens, req.temperature, req.top_k)
    latency = (time.time() - t0) * 1000
    gen_ids = y[0][len(start_ids):]
    text = enc.decode(gen_ids.tolist())
    result = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"message": {"role": "assistant", "content": text.strip()}, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(start_ids),
            "completion_tokens": len(gen_ids),
            "total_tokens": len(start_ids) + len(gen_ids),
        },
    }
    _log_interaction(
        req_id=result["id"], req_model=req.model, endpoint="chat_completions",
        prompt_text=user_text, response_text=text.strip(),
        usage=result["usage"], latency_ms=latency,
    )
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "sec-edgar-gpt-124m", "object": "model", "owned_by": "lzwjava"}]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
