from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, time, logging

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------ Config ------------------
load_dotenv()
MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")  # small & compliant
MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS_CAP = int(os.getenv("MAX_NEW_TOKENS_CAP", "256"))

# Deterministic by default; flip to True later for creativity
DEFAULT_DO_SAMPLE = os.getenv("DO_SAMPLE", "false").lower() == "true"
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))

# ------------------ App ------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000","http://127.0.0.1:3000",
        "http://localhost:3001","http://127.0.0.1:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")
_pipe = None
_tok = None

def get_pipe():
    global _pipe, _tok
    if _pipe is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL)
        _pipe = pipeline("text-generation", model=mdl, tokenizer=_tok)
    return _pipe

def build_prompt(user_text: str) -> str:
    """
    Use the tokenizer's chat template if available (for *-Instruct/*-Chat models).
    Falls back to plain text otherwise.
    """
    global _tok
    if _tok is None:
        # ensure tokenizer loaded
        get_pipe()
    if hasattr(_tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        return _tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    # fallback: plain text with a clear instruction style
    return f"USER: {user_text}\nASSISTANT:"

# ------------------ Health & middleware ------------------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    resp = await call_next(request)
    ms = int((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} -> {resp.status_code} in {ms}ms")
    return resp

# ------------------ Schema ------------------
class Inp(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 80
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_new_tokens: Optional[int] = 24

class Out(BaseModel):
    completion: str
    elapsed_ms: int

# ------------------ Endpoint ------------------
@app.post("/generate", response_model=Out)
def generate(inp: Inp):
    text = (inp.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(text) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"prompt too long (>{MAX_PROMPT_CHARS} chars)")

    max_new = min(max(int(inp.max_new_tokens or 1), 1), MAX_NEW_TOKENS_CAP)
    do_sample = bool(DEFAULT_DO_SAMPLE if inp.do_sample is None else inp.do_sample)
    temperature = float(DEFAULT_TEMPERATURE if inp.temperature is None else inp.temperature)
    top_p = float(DEFAULT_TOP_P if inp.top_p is None else inp.top_p)
    min_new = max(1, min(int(inp.min_new_tokens or 1), max_new))

    # Build a proper chat-style prompt when possible
    prompt_text = build_prompt(text)

    start = time.time()
    pipe = get_pipe()
    res = pipe(
        prompt_text,
        max_new_tokens=max_new,
        min_new_tokens=min_new,      # ensure we get more than "." 🙂
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_full_text=False,      # ✅ only the continuation
    )
    elapsed_ms = int((time.time() - start) * 1000)
    completion = (res[0]["generated_text"] or "").strip()
    return Out(completion=completion, elapsed_ms=elapsed_ms)
