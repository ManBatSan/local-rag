import os

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
generator = None


class GenRequest(BaseModel):
    prompt: str
    max_length: int = 128
    temperature: float = 0.7


def load_transformers_model(model_path: str):

    import importlib

    torch = importlib.import_module("torch")
    bnb = importlib.import_module("transformers").BitsAndBytesConfig
    AutoTokenizer = importlib.import_module("transformers").AutoTokenizer
    AutoModelForCausalLM = importlib.import_module("transformers").AutoModelForCausalLM
    pipeline = importlib.import_module("transformers").pipeline

    bnb_config = bnb(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
    )


def load_llamacpp_model(gguf_path: str):
    import importlib

    llama_cpp = importlib.import_module("llama_cpp")
    return llama_cpp.Llama(
        model_path=gguf_path, n_threads=8, n_ctx=2048 * 8, n_gpu_layers=100, seed=42
    )


@app.on_event("startup")
def startup():
    global generator
    use_llama = os.getenv("USE_LLAMACPP", "0") == "1"
    if use_llama:
        print("Using DeepSeek Model")
        model_name = os.getenv("MODEL_NAME")
        gguf = f"/models/{model_name}"
        generator = load_llamacpp_model(gguf)
    else:
        print("Using Llama 2 7B Model")
        model_name = os.getenv("MODEL_NAME")
        repo = f"/models/{model_name}"
        generator = load_transformers_model(repo)


@app.post("/generate")
def generate(req: GenRequest):
    if hasattr(generator, "create_completion"):
        # llama_cpp
        out = generator.create_completion(
            prompt=req.prompt,
            max_tokens=req.max_length,
            temperature=req.temperature,
        )
        print(out)
        text = out["choices"][0]["text"]
    else:
        # Transformers pipeline
        out = generator(
            req.prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            do_sample=True,
            num_return_sequences=1,
        )
        text = out[0]["generated_text"]
    return {"generated_text": text}


@app.get("/health")
def health():
    return {"status": "ok" if generator is not None else "loading"}
