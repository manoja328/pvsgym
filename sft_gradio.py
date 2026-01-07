import random
import torch, os
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from datasets import Dataset
from sft_dataset import extract_rollouts, load_config

# Load model and tokenizer with caching and progress
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(path):
    with st.spinner("Loading model and tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="auto")
    return tokenizer, model

# Load and prepare dataset with caching and progress
@st.cache_data(show_spinner=True)
def load_dataset(path):
    with st.spinner("Loading and processing dataset..."):
        ds_train, ds_test = extract_rollouts(path)
    return ds_test

# Log probability scorer
def get_logprob(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt',  max_length=2048, truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs.logits
    labels = inputs['input_ids']
    log_probs = -F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none'
    ).view(labels.shape).sum(dim=1)
    return log_probs

# Completion ranker
def recommend_top_k_steps(model, tokenizer, prompt, top_k=3):
    inputs = tokenizer(prompt, max_length=2048, truncation=True, return_tensors='pt').to(model.device)

    stop_ids = {tokenizer.eos_token_id}
    # for token in ["END","\n"]:
    for token in ["END"]:
        END_ID = tokenizer.convert_tokens_to_ids(token)
        stop_ids.add(END_ID)

    model.eval()
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=top_k,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=list(stop_ids),
            output_scores=True,
            return_dict_in_generate=True,
        )

    sequences = gen.sequences
    scores = gen.scores
    prompt_len = inputs["input_ids"].shape[1]

    suggestions_with_logprob = []
    for i in range(sequences.size(0)):
        gen_text = tokenizer.decode(sequences[i, prompt_len:], skip_special_tokens=True).strip()
        gen_ids = sequences[i, prompt_len:]
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
        suggestions_with_logprob.append((gen_text, length_norm_logprob))

    suggestions_ranked = sorted(suggestions_with_logprob, key=lambda x: x[1], reverse=True)
    return suggestions_ranked

# WEBSOCKET = 111334
# @reqest.method("GET", "/recoommen")
# # def recommend(sequent: str, previous_comamnds: list):
#     result = model( seqent, prev_commands)
#     ## write  to websocket port
#     return websocket.send(json.dumps(result))


if __name__ == "__main__":

    # Streamlit UI
    st.title("PVS Step Recommender")

    config = load_config("pvs_v5.yaml")
    SAVE_PATH = config.save_path
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    NUM_SUGGESTIONS = 3
    # Initialize model, tokenizer, and dataset
    test_ds = load_dataset(config.rollouts_path)
    tokenizer, model = load_model_and_tokenizer(SAVE_PATH)

    if st.button("Sample Random Proof Step"):
        idx = random.randrange(len(test_ds))
        example = test_ds[idx]
        st.session_state["current_idx"] = idx
        st.session_state["example"] = example
        st.session_state["auto_recommend"] = True

    if "example" in st.session_state:
        example = st.session_state["example"]
        prompt = example["text"] + "\nNext Command:\n"
        st.text(f"Theorem: {example['id']}")
        st.text(f"True: {example['label']}")
        user_input = st.text_area("Input", prompt, height="content")

        # if st.button("Recommend Next Steps"):
        #     ranked_completions = recommend_top_k_steps(user_input)
        #     st.subheader("Top Suggestions")
        #     for i, (completion, score) in enumerate(ranked_completions):
        #         st.markdown(
        #             f"**Suggestion {i+1} (log prob: {score:.2f})**<br><br>{completion.replace(chr(10), '<br>')}",
        #             unsafe_allow_html=True
        #         )

        recommend_pressed = st.button("Recommend Next Steps")
        if recommend_pressed or st.session_state.get("auto_recommend", False):
            ranked_completions = recommend_top_k_steps(model, tokenizer, user_input, top_k=NUM_SUGGESTIONS)
            st.subheader("Top Suggestions")
            # Build table data
            table_data = [
                {
                    "Log_prob": round(score, 2),
                    "Command": completion.split("\n")[0],
                    "Output": completion.replace('\n', '\\n'),# Escaped for display
                }
                for i, (completion, score) in enumerate(ranked_completions)
            ]
            st.table(pd.DataFrame(table_data))




