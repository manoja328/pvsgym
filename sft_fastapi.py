import random
import os
from typing import List, Optional
import yaml
from types import SimpleNamespace
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="PVS Step Recommender API", version="1.0.0")

# ------------------------------
# Global state (loaded once)
# ------------------------------
TOKENIZER = None
MODEL = None
TEST_DATASET = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)

def load_model_and_tokenizer(path: str):
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        TOKENIZER = AutoTokenizer.from_pretrained(path, use_fast=True)
        # device_map="auto" lets HF place layers; dtype="auto" for mixed precision when available
        MODEL = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="auto")
        # Some models have no pad token id; fall back to eos
        if TOKENIZER.pad_token_id is None and TOKENIZER.eos_token_id is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        print("model and tokenizer loaded")
    return TOKENIZER, MODEL


def recommend_top_k_steps(model, tokenizer, prompt: str, top_k: int = 3):
    inputs = tokenizer(prompt, max_length=2048, truncation=True, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    stop_ids = {tokenizer.eos_token_id}
    for token in ["END"]:
        tok_id = tokenizer.convert_tokens_to_ids(token)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            stop_ids.add(tok_id)

    model.eval()
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=top_k,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=list(stop_ids),
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=128,
        )

    sequences = gen.sequences
    scores = gen.scores
    prompt_len = inputs["input_ids"].shape[1]

    suggestions_with_logprob = []
    for i in range(sequences.size(0)):
        gen_ids = sequences[i, prompt_len:]
        # Decode for display; keep raw text and also split first line as the command
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        total_logprob, token_count = 0.0, 0
        for t in range(min(len(scores), gen_ids.numel())):
            token_id = int(gen_ids[t].item())
            if token_id in stop_ids:
                break
            step_logits = scores[t][i]
            step_logprobs = F.log_softmax(step_logits, dim=-1)
            total_logprob += float(step_logprobs[token_id].item())
            token_count += 1

        length_norm_logprob = total_logprob / max(token_count, 1)
        suggestions_with_logprob.append({
            "log_prob": length_norm_logprob,
            "command": gen_text.split("\n")[0]
        })

    suggestions_with_logprob.sort(key=lambda x: x["log_prob"], reverse=True)
    return suggestions_with_logprob


# ------------------------------
# Pydantic models
# ------------------------------
class RecommendResponse(BaseModel):
    prompt: str
    top_k: int
    suggestions: List[dict]


class RecommendRequest(BaseModel):
    sequent: str
    prev_commands: List[str]
    top_k: Optional[int] = 3

# ------------------------------
# Startup: load config, model, and dataset
# ------------------------------
@app.on_event("startup")
def startup_event():
    # Allow overriding via env vars, else use YAML
    config_path = os.environ.get("PVS_API_CONFIG", "pvs_v5.yaml")
    config = load_config(config_path)

    save_path = os.environ.get("PVS_MODEL_PATH", getattr(config, 'save_path', None))
    if not save_path:
        raise RuntimeError("Model path not provided. Set PVS_MODEL_PATH or include save_path in config YAML.")

    load_model_and_tokenizer(save_path)


# ------------------------------
# Routes
# ------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/info")
def info():
    return {
        "model_name": getattr(MODEL.config, 'name_or_path', None),
        "vocab_size": getattr(MODEL.config, 'vocab_size', None),
        "eos_token_id": TOKENIZER.eos_token_id,
        "pad_token_id": TOKENIZER.pad_token_id,
        "device": str(MODEL.device),
    }

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    sequent = req.sequent.strip()
    prev_cmds = req.prev_commands or []
    prompt_lines = [f"Current Sequent:\n{sequent}\n"]
    for i, cmd in enumerate(prev_cmds):
        prompt_lines.append(f"Prev Command {i+1}: {cmd if cmd else 'None'}")
    prompt = "\n".join(prompt_lines) + "\nNext Command:\n"
    suggestions = recommend_top_k_steps(MODEL, TOKENIZER, prompt, top_k=req.top_k)
    return RecommendResponse(prompt=prompt, top_k=req.top_k, suggestions=suggestions)

    # if not prompt.strip():
    #     return JSONResponse(status_code=400, content={"error": "prompt must be a non-empty string"})

    # suggestions = recommend_top_k_steps(MODEL, TOKENIZER, prompt, top_k=top_k)
    # return RecommendResponse(prompt=prompt, top_k=top_k, suggestions=suggestions)


# ------------------------------
# Entrypoint for running with `python pvs_step_recommender_api.py`
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pvs_step_recommender_api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
