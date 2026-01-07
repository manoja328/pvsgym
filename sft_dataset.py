import json
import glob
import os
import random
import yaml
from tqdm import tqdm
from collections import defaultdict
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)

def save_jsonl(data, fpath):
    ## data is a list of dicts
    with open(fpath, 'w', encoding='utf-8') as val_out:
        for entry in data:
            val_out.write(json.dumps(entry) + "\n")
    print(f"Saved {len(data)} entries to {fpath}")

def parse_rollout_sequent(rollout_dir, save_path):
    all_examples = []
    files = glob.glob(os.path.join(rollout_dir, "*/*.log"))
    for path in tqdm(files, desc="Loading rollout files"):
        if not os.path.exists(path): continue
        with open(path) as f:
            steps = json.load(f)
            if not isinstance(steps, list) or not steps:
                continue
            # Use first step to get initial goal/state
            goal_line = "\n".join(steps[0]['sequents'])
            prev_sequent = goal_line
            # Build (state, action) examples from each step
            for curr in steps[1:]:
                example = {
                    "id": path.split("/")[-1].lstrip("rollout_").rstrip(".log"),
                    "folder": path.split("/")[-2],
                    "text": (
                        f"Current Sequent:\n{prev_sequent}"
                    ),
                    "label": curr["command"]
                }
                prev_sequent = "\n".join(s for s in curr['sequents'] if s != "QED")
                all_examples.append(example)

    save_jsonl(all_examples, save_path)


def parse_rollout_cmdhist(rollout_dir, save_path, topk = 3):
    all_examples = []
    files = glob.glob(os.path.join(rollout_dir, "*/*.log"))
    for path in tqdm(files, desc="Loading rollout files"):
        if not os.path.exists(path): continue
        with open(path) as f:
            steps = json.load(f)
            if not isinstance(steps, list) or not steps:
                continue
            cmd_hist_size = topk
            prev_commands = []
            for curr in steps[1:]:
                padded_prev_cmds = (["None"] * (cmd_hist_size - len(prev_commands))) + prev_commands[-cmd_hist_size:]
                prev_cmd_text = "\n".join(
                    f"Prev Command {i+1}: {cmd}" for i, cmd in enumerate(padded_prev_cmds)
                )
                example = {
                    "id": path.split("/")[-1].lstrip("rollout_").rstrip(".log"),
                    "folder": path.split("/")[-2],
                    "text": (
                        f"{prev_cmd_text}"
                    ),
                    "label": curr["command"]
                }
                prev_commands.append(curr["command"])
                all_examples.append(example)
    save_jsonl(all_examples, save_path)


def parse_rollout_full(rollout_dir, save_path, topk = 3):
    all_examples = []
    files = glob.glob(os.path.join(rollout_dir, "*/*.log"))
    for path in tqdm(files, desc="Loading rollout files"):
        if not os.path.exists(path): continue
        with open(path) as f:
            steps = json.load(f)
            if not isinstance(steps, list) or not steps:
                continue
            # Use first step to get initial goal/state
            goal_line = "\n".join(steps[0]['sequents'])
            prev_sequent = goal_line
            cmd_hist_size = topk
            prev_commands = []
            # Build (state, action) examples from each step
            for curr in steps[1:]:
                padded_prev_cmds = (["None"] * (cmd_hist_size - len(prev_commands))) + prev_commands[-cmd_hist_size:]
                prev_cmd_text = "\n".join(
                    f"Prev Command {i+1}: {cmd}" for i, cmd in enumerate(padded_prev_cmds)
                )
                example = {
                    "id": path.split("/")[-1].lstrip("rollout_").rstrip(".log"),
                    "folder": path.split("/")[-2],
                    "text": (
                        f"Current Sequent:\n{prev_sequent}\n\n"
                        f"{prev_cmd_text}"
                    ),
                    "label": curr["command"]
                }

                # Filter out "QED" from sequents before joining
                prev_sequent = "\n".join(s for s in curr['sequents'] if s != "QED")
                prev_commands.append(curr["command"])
                all_examples.append(example)
    save_jsonl(all_examples, save_path)


class Node:
    def __init__(self,id, val, parent = None):
        self.id = id
        self.val = val
        self.parent = parent

    def __repr__(self):
        return f"({self.id}){self.val}"
    
def get_kparents(node, k = -1):
    ret = []
    while node:
        ret.append(node.id)
        node = node.parent
        if k == 0:
            break
        k = k - 1
    return ret

def process_commands(commands,  k = 3):
    nodedict = {}
    nodes = {}
    for nodec, name in enumerate(commands):
        node = Node(nodec, name)
        if nodec == 0: 
            prev = None
        elif node.val in nodedict:
            prev = nodedict[node.val]
        else:
            parent_val = '.'.join(name.split(".")[:-1])
            prev = nodedict[parent_val]
        node.parent = prev
        nodedict[node.val] = node
        nodes[nodec] = node
    # print(nodes)

    node_parents = {}
    for nid, node in nodes.items():
        parents = get_kparents(node, k)[1:] ## return indices
        node_parents[nid] = parents
        # print(nid, node.val, parents)
    return node_parents


def parse_rollout_full_histfix(rollout_dir, save_path, topk = 3):
    all_examples = []
    files = glob.glob(os.path.join(rollout_dir, "*/*.log"))
    for path in tqdm(files, desc="Loading rollout files"):
        if not os.path.exists(path): continue
        with open(path) as f:
            steps = json.load(f)
            if not isinstance(steps, list) or not steps:
                continue

            # Use first step to get initial goal/state
            goal_line = "\n".join(steps[0]['sequents'])
            proof_labels = [ step['label'] for step in steps[1:] ] # first step is just goal sequent
            node_to_parentids = process_commands(proof_labels)
            ## now build G .. and get chains ... 
            prev_sequent = goal_line
            cmd_hist_size = 3
            node_id = 0
            for curr in steps[1:]:
                hist_ids = node_to_parentids[node_id][::-1] ## reverse chronological
                prev_commands = [ steps[nid+1]['command'] for  nid  in  hist_ids]
                padded_prev_cmds = (["None"] * (cmd_hist_size - len(prev_commands))) + prev_commands[-cmd_hist_size:]
                prev_cmd_text = "\n".join(
                    f"Prev Command {i+1}: {cmd}" for i, cmd in enumerate(padded_prev_cmds)
                )
                example = {
                    "id": path.split("/")[-1].lstrip("rollout_").rstrip(".log"),
                    "folder": path.split("/")[-2],
                    "text": (
                        f"Current Sequent:\n{prev_sequent}\n\n"
                        f"{prev_cmd_text}"
                    ),
                    "label": curr["command"]
                }
                prev_sequent = "\n".join(s for s in curr['sequents'] if s != "QED")
                prev_commands.append(curr["command"])
                all_examples.append(example)
                node_id +=1
    save_jsonl(all_examples, save_path)


def extract_rollouts(rollout_path, test_size = 0.05):
    folder_to_examples = defaultdict(list)
    with open(rollout_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            folder = example["folder"]
            folder_to_examples[folder].append(example)

    all_folders = list(folder_to_examples.keys())
    train_folders, test_folders = train_test_split(
        all_folders, test_size = test_size, random_state=42
    )

    def collect(folders):
        return [ex for folder in folders for ex in folder_to_examples[folder]]

    train_examples = collect(train_folders)
    test_examples = collect(test_folders)
    ds_train = Dataset.from_list(train_examples)
    ds_test = Dataset.from_list(test_examples)
    print(f"📊 Split result: {len(ds_train)} train | {len(ds_test)} test examples")
    return ds_train, ds_test

if __name__ == "__main__":
    # parse_rollout_file = parse_rollout_cmdhist("rollouts2", "rollouts2_cmdhist_only.jsonl")
    # parse_rollout_file = parse_rollout_sequent("rollouts2", "rollouts2_sequent_only.jsonl")
    # parse_rollout_file = parse_rollout_full("rollouts2", "rollouts2_cmdhist_plus_seq.jsonl")
    # parse_rollout_file = parse_rollout_full("rollouts2_fixed", "rollouts2fixed_cmdhist_plus_seq.jsonl")
    # ds_train, ds_test = extract_rollouts("rollouts2fixed_cmdhist_plus_seq.jsonl")
    # ds_train, ds_test = extract_rollouts("rollouts2_cmdhist_plus_seq.jsonl")
    ## fix history from same branch or parent
    parse_rollout_file = parse_rollout_full_histfix("rollouts2_fixed", "rollouts2fixed_cmdhist_plus_seq_histfixed.jsonl")
    ds_train, ds_test = extract_rollouts("rollouts2fixed_cmdhist_plus_seq_histfixed.jsonl")

    # Saved 210706 entries to rollouts2fixed_cmdhist_plus_seq.jsonl
    # 📊 Split result: 202579 train | 8127 test examples
    # 📊 Split result: 430527 train | 17580 test examples