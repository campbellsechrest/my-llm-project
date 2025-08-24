from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, time, logging, requests

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

# Hugging Face Inference API (used on Render Free to avoid OOM)
HF_TOKEN = os.getenv("HF_TOKEN")
REMOTE_MODEL = MODEL  # reuse same model name remotely

# ------------------ App ------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://my-llm-project.vercel.app",  # live frontend (no trailing slash)
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")
_pipe = None
_tok = None

def get_pipe():
    """Local pipeline for development. Avoid on tiny cloud instances."""
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
        get_pipe()  # ensure tokenizer loaded when local
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

# ------------------ Remote HF inference ------------------
def call_hf_inference(
    prompt_text: str,
    max_new: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    min_new: int,
) -> str:
    """
    Call Hugging Face Inference API with simple retries when the model is loading.
    """
    url = f"https://api-inference.huggingface.co/models/{REMOTE_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": max_new,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "return_full_text": False,
            "min_new_tokens": min_new,
        },
        "options": {"use_cache": True, "wait_for_model": True},
    }

    retries = 5
    backoff = 1.5
    delay = 1.0
    last_err = None

    for _ in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            # 503 while the model spins up
            if r.status_code == 503:
                time.sleep(delay)
                delay *= backoff
                continue
            r.raise_for_status()
            data = r.json()
            # Typical HF responses:
            # [{"generated_text": "..."}] or {"generated_text": "..."}
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return (data[0]["generated_text"] or "").strip()
            if isinstance(data, dict) and "generated_text" in data:
                return (data["generated_text"] or "").strip()
            # Some servers return {"error": "..."} on failure
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(data["error"])
            # Fallback
            return str(data)
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= backoff

    raise HTTPException(status_code=502, detail=f"HF inference error: {last_err}")

# ------------------ Health & middleware ------------------
@app.get("/health")
def health():
    mode = "remote" if HF_TOKEN else "local"
    return {"ok": True, "model": MODEL, "mode": mode}

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

    prompt_text = build_prompt(text)
    start = time.time()

    if HF_TOKEN:
        # Remote inference (tiny memory footprint on Render Free)
        completion = call_hf_inference(prompt_text, max_new, do_sample, temperature, top_p, min_new)
    else:
        # Local pipeline (for your laptop; not for 512MB instances)
        pipe = get_pipe()
        res = pipe(
            prompt_text,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
        )
        completion = (res[0]["generated_text"] or "").strip()

    elapsed_ms = int((time.time() - start) * 1000)
    return Out(completion=completion, elapsed_ms=elapsed_ms)
