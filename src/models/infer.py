from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # small starter model

def get_pipe():
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL)
    return pipeline("text-generation", model=mdl, tokenizer=tok)

if __name__ == "__main__":
    pipe = get_pipe()
    prompt = "You are a helpful assistant. Summarize: Today I walked my dog and felt happy."
    out = pipe(prompt, max_new_tokens=60, do_sample=False)
    print(out[0]["generated_text"])
