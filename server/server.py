#!/usr/bin/env python3
"""OpenAI-compatible API server for GPT-2 using transformers + FastAPI."""
import torch, json, time, uuid, os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/workspace/model/hf-model"
app = FastAPI()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to("cuda")
model.eval()
EOS_ID = tokenizer.eos_token_id
print(f"Model loaded on GPU. Vocab: {tokenizer.vocab_size}, Model vocab: {model.config.vocab_size}")

SEC_SYSTEM = (
    "The following are excerpts from SEC EDGAR filings filed with the "
    "U.S. Securities and Exchange Commission by publicly traded companies.\n\n"
)

MIN_PROMPT_CHARS = 10

def _generate(input_ids, max_new_tokens, temperature, top_p):
    """Shared generate wrapper with proper attention_mask and pad_token_id."""
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=max((temperature or 0.7), 1e-7),
            top_p=top_p or 0.9,
            do_sample=True,
            pad_token_id=EOS_ID,
        )
    return output

class CompletionRequest(BaseModel):
    model: str = "sec-edgar-gpt-124m"
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "sec-edgar-gpt-124m"
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path, media_type="text/html")

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if len(req.prompt.strip()) < MIN_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt must be at least {MIN_PROMPT_CHARS} characters")
    prompt = SEC_SYSTEM + req.prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    output = _generate(input_ids, req.max_tokens, req.temperature, req.top_p)
    gen_ids = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(gen_ids),
            "total_tokens": input_ids.shape[1] + len(gen_ids),
        },
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Build prompt: prepend SEC system context, then messages
    user_text = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    if len(user_text.strip()) < MIN_PROMPT_CHARS:
        raise HTTPException(400, f"Message must be at least {MIN_PROMPT_CHARS} characters")
    prompt = SEC_SYSTEM + user_text + "\nassistant: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    output = _generate(input_ids, req.max_tokens, req.temperature, req.top_p)
    gen_ids = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Trim at first newline after a reasonable length to avoid run-on
    if "\n" in text[20:]:
        text = text[:text.index("\n", 20)]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"message": {"role": "assistant", "content": text.strip()}, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(gen_ids),
            "total_tokens": input_ids.shape[1] + len(gen_ids),
        },
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/models")
async def models():
    return {"data": [{"id": "sec-edgar-gpt-124m", "object": "model", "owned_by": "lzwjava"}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
