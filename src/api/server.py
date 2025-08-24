from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

app = FastAPI()
@app.get("/health")
def health():
    return {"ok": True}


_pipe = None

def get_pipe():
    global _pipe
    if _pipe is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL)
        _pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
    return _pipe

class Inp(BaseModel):
    prompt: str

class Out(BaseModel):
    completion: str

@app.post("/generate", response_model=Out)
def generate(inp: Inp):
    pipe = get_pipe()
    res = pipe(inp.prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"]
    return Out(completion=res)
