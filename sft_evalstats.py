import random
import pandas as pd
import re
from tqdm import tqdm
import os, fire
import ast  # In case the 'llm' column is stored as a string
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt

def get_command_type(command_str):
    """Extracts the first word from a command string like '(REPLACE -1 :DIR RL)'."""
    match = re.search(r'[A-Z]+', command_str)
    result = match.group(0) if match else ""
    return result

def plot_confusion_matrix(csv_path, topk = 3):
    df = pd.read_csv(csv_path)
    # Set top-k (should match what you used in generation)
    k = topk
    top1_exact = 0
    topk_exact = 0
    top1_type = 0
    topk_type = 0
    total = len(df)
    # For confusion matrix
    all_targets = []
    all_preds = []

    if isinstance(df.iloc[0]["llm"], str):
        df["llm"] = df["llm"].apply(ast.literal_eval)

    for _, row in tqdm(df.iterrows()):
        true_label = row["label"]
        true_type = get_command_type(true_label)
        predictions_full = [entry["command"] for entry in row["llm"]]
        predictions_type = [get_command_type(cmd) for cmd in predictions_full]

        all_targets.append(true_type)
        all_preds.append(predictions_type[0])  # Only top-1 prediction for CM

        # -- Exact Match --
        if true_label == predictions_full[0]:
            top1_exact += 1
        if true_label in predictions_full[:k]:
            topk_exact += 1

        # -- Command Type Match --
        if true_type == predictions_type[0]:
            top1_type += 1
        if true_type in predictions_type[:k]:
            topk_type += 1

    # Compute accuracy
    top1_exact_acc = top1_exact / total
    topk_exact_acc = topk_exact / total
    top1_type_acc = top1_type / total
    topk_type_acc = topk_type / total

    print(f"Total Examples: {total}\n")

    print(f"🔍 Exact Match Accuracy:")
    print(f"Top-1 Exact Match: {top1_exact_acc:.3f}")
    print(f"Top-{k} Exact Match: {topk_exact_acc:.3f}\n")

    print(f"🔍 Command Type Match Accuracy:")
    print(f"Top-1 Type Match: {top1_type_acc:.3f}")
    print(f"Top-{k} Type Match: {topk_type_acc:.3f}")


    # Step 1: Get top labels by true frequency
    topk = 30
    UNK = "__UNK__"
    true_counts = Counter(all_targets)
    top_20_labels = [label for label, _ in true_counts.most_common(topk)]
    allowed_labels = set(top_20_labels)

    filtered_pairs = [
        (t, p if p in allowed_labels else UNK)
        for t, p in zip(all_targets, all_preds)
        if t in allowed_labels
    ]

    filtered_targets = [t for t, p in filtered_pairs]
    filtered_preds = [p for t, p in filtered_pairs]

    # Step 4: Construct label list including UNK
    labels_top = sorted(set(filtered_targets + filtered_preds + [UNK]))
    label_to_idx = {label: idx for idx, label in enumerate(labels_top)}
    num_classes = len(labels_top)

    cm = confusion_matrix(
        [label_to_idx[t] for t in filtered_targets],
        [label_to_idx[p] for p in filtered_preds],
        labels=np.arange(num_classes)
    )

    # # Confusion matrix
    # labels_top = sorted(set(all_targets + all_preds))
    # label_to_idx = {label: idx for idx, label in enumerate(labels_top)}
    # num_classes = len(labels_top)
    # cm = confusion_matrix(
    #     [label_to_idx[t] for t in all_targets],
    #     [label_to_idx[p] for p in all_preds],
    #     labels=np.arange(num_classes)
    # )

    # Plotting
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_top, yticklabels=labels_top,
                cbar=False, square=True, linewidths=1, linecolor='gray')

    plt.xlabel("Predicted Command Type")
    plt.ylabel("True Command Type")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"confusion_top{topk}_unk.png", dpi=200)

    accuracy = np.trace(cm) / np.sum(cm)
    print(f"cm-Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # Load the saved file
    # SAVE_PATH = "pvs_sft4"
    # SAVE_PATH = "pvs_sft5"
    # SAVE_PATH = "pvs_sft25"
    # SAVE_PATH = "pvs_sftfull_roll2fixed"
    SAVE_PATH = "/workspace1/macharya/pvs_sftfull_roll2fixed_histfixed"
    CSV_FILE =  os.path.join(SAVE_PATH,"recommended_commands.csv")
    print("evaluating ", CSV_FILE)
    plot_confusion_matrix(CSV_FILE)
    # fire.Fire(plot_confusion_matrix)
