from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, time, logging, requests

from dotenv import load_dotenv
from openai import OpenAI

# Optional local/HF imports for laptop/local use
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------ Config ------------------
load_dotenv()

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")  # default when using OpenAI
MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS_CAP = int(os.getenv("MAX_NEW_TOKENS_CAP", "256"))

DEFAULT_DO_SAMPLE = os.getenv("DO_SAMPLE", "false").lower() == "true"
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional fallback if you ever want to use HF

# For local dev (ONLY used if no OPENAI_API_KEY/HF_TOKEN)
LOCAL_MODEL = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# ------------------ App ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # local dev
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
        # deployed frontend
        "https://my-llm-project.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")

# ------------------ Local pipeline (fallback) ------------------
_pipe = None
_tok = None

def get_pipe():
    """Local pipeline for development on your laptop."""
    global _pipe, _tok
    if _pipe is None:
        _tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        mdl = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL)
        _pipe = pipeline("text-generation", model=mdl, tokenizer=_tok)
    return _pipe

def build_prompt(user_text: str) -> str:
    """
    Builds a prompt; uses chat template if available (local path only).
    For OpenAI, we just send the plain text prompt_text below.
    """
    global _tok
    if _tok is None:
        try:
            get_pipe()
        except Exception:
            # If local pipeline isn't initialized (e.g., on Render), just return plain
            return f"USER: {user_text}\nASSISTANT:"
    if hasattr(_tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        return _tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"USER: {user_text}\nASSISTANT:"

# ------------------ OpenAI path ------------------
def call_openai(prompt_text: str, max_new: int, temperature: float, top_p: float) -> str:
    """
    Uses OpenAI Responses API (server-side).
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model=MODEL,                 # e.g., "gpt-4o-mini"
        input=prompt_text,           # single-string input
        max_output_tokens=max_new,   # caps new tokens
        temperature=temperature,
        top_p=top_p,
    )
    # SDK convenience:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()

    # Fallback: collect from structured output
    parts = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") == "output_text":
            parts.append(getattr(item, "text", ""))
    return "".join(parts).strip()

# ------------------ HF serverless path (optional fallback) ------------------
def call_hf_inference(prompt_text: str, max_new: int, temperature: float, top_p: float, min_new: int) -> str:
    url = f"https://api-inference.huggingface.co/models/{MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": max_new,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "return_full_text": False,
            "min_new_tokens": min_new,
        },
        "options": {"use_cache": True, "wait_for_model": True},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return (data[0]["generated_text"] or "").strip()
    if isinstance(data, dict) and "generated_text" in data:
        return (data["generated_text"] or "").strip()
    if isinstance(data, dict) and "error" in data:
        raise HTTPException(status_code=502, detail=data["error"])
    return str(data)

# ------------------ Health & middleware ------------------
@app.get("/health")
def health():
    if OPENAI_API_KEY:
        mode = "openai"
        model = MODEL
    elif HF_TOKEN:
        mode = "huggingface"
        model = MODEL
    else:
        mode = "local"
        model = LOCAL_MODEL
    return {"ok": True, "mode": mode, "model": model}

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
    temperature = float(DEFAULT_TEMPERATURE if inp.temperature is None else inp.temperature)
    top_p = float(DEFAULT_TOP_P if inp.top_p is None else inp.top_p)
    min_new = max(1, min(int(inp.min_new_tokens or 1), max_new))

    # For OpenAI we just pass the plain user text; for local/HF we build a chat-style prompt
    prompt_text = text if OPENAI_API_KEY else build_prompt(text)

    start = time.time()

    if OPENAI_API_KEY:
        completion = call_openai(prompt_text, max_new, temperature, top_p)
    elif HF_TOKEN:
        completion = call_hf_inference(prompt_text, max_new, temperature, top_p, min_new)
    else:
        pipe = get_pipe()
        res = pipe(
            prompt_text,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=True,  # local: allow sampling
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
        )
        completion = (res[0]["generated_text"] or "").strip()

    elapsed_ms = int((time.time() - start) * 1000)
    return Out(completion=completion, elapsed_ms=elapsed_ms)
