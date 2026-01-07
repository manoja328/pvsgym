import os
import re
import json
import random

FILEDIR = "json-prooflite_june27"
OUTPUT_FILE = "prooflite_dataset.jsonl"
RANDOM_SEED = 42  # Set your desired seed here
val_split = 0.1   # 10% for validation
TRAIN_FILE = "prooflite_train_openai.jsonl"
VAL_FILE = "prooflite_val_openai.jsonl"

def convert_json_to_jsonl(input_dir, output_file):
    save_data = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(input_dir, filename)) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filename}: {e}")
            continue

        for entry in data['decls']:
            messages = [
                {"role": "user", "content": entry['decl'].strip()},
                {"role": "assistant", "content": entry['proof'].strip()}
            ]
            save_data.append({"messages": messages})

    random.shuffle(save_data)
    val_size = int(len(save_data) * val_split)
    val_data = save_data[:val_size]
    train_data = save_data[val_size:]

    print(f"Saved {len(train_data)} train and {len(val_data)} val entries.")

    with open(TRAIN_FILE, 'w', encoding='utf-8') as train_out:
        for entry in train_data:
            train_out.write(json.dumps(entry) + "\n")
    with open(VAL_FILE, 'w', encoding='utf-8') as val_out:
        for entry in val_data:
            val_out.write(json.dumps(entry) + "\n")

convert_json_to_jsonl(FILEDIR, OUTPUT_FILE)
