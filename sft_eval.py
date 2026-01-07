import torch
import pandas as pd
import os, fire
from tqdm import tqdm
from sft_gradio import load_dataset, load_model_and_tokenizer, recommend_top_k_steps
from sft_dataset import extract_rollouts, load_config
from sft_evalstats import plot_confusion_matrix

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SUGGESTIONS = 3

def train(config_path="config.yaml"):
    config = load_config(config_path)
    SAVE_PATH = config.save_path

    CSV_FILE =  os.path.join(SAVE_PATH,"recommended_commands.csv")
    print("Saving all recommendations to ", CSV_FILE)

    # Initialize model, tokenizer, and dataset
    test_ds = load_dataset(config.rollouts_path)
    tokenizer, model = load_model_and_tokenizer(SAVE_PATH)

    all_results = []
    for idx in tqdm(range(len(test_ds))):
        example = test_ds[idx]
        prompt = example["text"] + "\nNext Command:\n"
        # print(f"Theorem: {example['id']}")
        # print(f"True: {example['label']}")
        ranked_completions = recommend_top_k_steps(model, tokenizer, prompt, top_k=NUM_SUGGESTIONS)
        # print("\nTop Guesses:")
        # for completion, score in ranked_completions:
        #     print(f"---- log_prob:{score:.2f} ---")
        #     print(completion)
        table_data = [
            {
                "log_prob": round(score, 2),
                "command": completion.split("\n")[0],
                "output": completion.replace('\n', '\\n'),# Escaped for display
            }
            for i, (completion, score) in enumerate(ranked_completions)
        ]
        example['llm'] = table_data
        all_results.append(example)
        # Convert all results to a DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(CSV_FILE, index=False)
    

    plot_confusion_matrix(csv_path=CSV_FILE)


if __name__ == "__main__":
    fire.Fire(train)

