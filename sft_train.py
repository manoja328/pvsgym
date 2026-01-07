import json
import os
import random
from collections import defaultdict
import numpy as np
import glob, fire
import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from sft_dataset import extract_rollouts, load_config
os.environ["WANDB_PROJECT"] = "expmath-sft"

# ------------------------------------------
# 🔍 Helper: Check if we're in DDP mode
# ------------------------------------------
def is_distributed():
    return dist.is_available() and dist.is_initialized()

def print_trainable_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔍 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

def analyze_token_lengths(dataset, tokenizer, n_samples=1000):
    stats = defaultdict(list)
    # Optionally subsample to speed things up
    n_samples = min(n_samples, len(dataset))
    subset = dataset.select(range(n_samples))
    for ex in subset:
        # Tokenize each field
        prompt_len = len(tokenizer.encode(ex["text"], add_special_tokens=False))
        stats["text"].append(prompt_len)

    # Print summary
    for key, values in stats.items():
        arr = np.array(values)
        print(f"\n🔍 {key} lengths:")
        print(f"   avg = {arr.mean():.1f}, max = {arr.max()}, 95th percentile = {np.percentile(arr,95):.1f}")

    return stats

def train(
    config_path="config.yaml",
    bs=8,
    testrun=False,
    use_peft_lora=False,
    use_fsdp=False,
    use_deepspeed=False,
    ds_config=None,
):
    config = load_config(config_path)
    setup_ddp()

    rank = int(os.environ.get("RANK", 0)) if is_distributed() else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if is_distributed() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed() else 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    SAVE_PATH = config.save_path
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(ex):
        return tokenizer(ex["text"], padding=True, truncation=True, max_length=512)

    def convert_to_dapt_format(example):
        eos_token = tokenizer.eos_token
        full_text = example["text"] + "\nNext Command:\n" + example["label"] + eos_token
        return {"text": full_text}

    # Load and format dataset
    ds_train, ds_test =  extract_rollouts(config.rollouts_path)
    analyze_token_lengths(ds_train, tokenizer)

    if testrun:
        print("🔥 Subsample for quick test runs")
        ds_train = ds_train.select(range(min(500, len(ds_train))))
        ds_test  = ds_test.select(range(min(500, len(ds_test))))

    formatted_train = ds_train.map(convert_to_dapt_format, num_proc = 8)  # keep only "text"
    formatted_test = ds_test.map(convert_to_dapt_format, num_proc = 8 )  # keep only "text"
    remove_cols = formatted_train.column_names
    tokenized_train = formatted_train.map(tok_fn, batched=True, num_proc=8, remove_columns = remove_cols)
    tokenized_eval = formatted_test.map(tok_fn, batched=True, num_proc=8, remove_columns = remove_cols)

    print("\n🔍 Inspecting random samples from training data:\n")
    for _ in range(5):
        idx = random.randrange(len(ds_train))
        sample = formatted_train[idx]
        tokenized_sample = tokenized_train[idx]
        print(sample["text"])
        print(tokenized_sample)
        print("-" * 60)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 🔀 Decide whether to enable FSDP
    if use_fsdp and is_distributed():
        fsdp_mode = "full_shard"
        fsdp_cfg={
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            # "fsdp_min_num_params": 1e7,
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_use_orig_params": True,
            "activation_checkpointing": True,
        }
        print("✅ FSDP enabled: full_shard auto_wrap on LlamaDecoderLayer")
    else:
        fsdp_mode = None
        fsdp_cfg = None
        if use_fsdp and not is_distributed():
            print("⚠️ use_fsdp=True but not in distributed mode; FSDP will be ignored.")
        else:
            print("⏹ FSDP disabled; using plain DDP or single-GPU.")

    # ------------------------------------------
    # 🤖 Model loading (DDP-safe vs. Sharded)
    # ------------------------------------------
    if is_distributed():
        if dist.get_rank() == 0:
            print("🚀 Running in DDP mode (multi-GPU data parallel)")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=None, ## Model loading (FSDP/DDP-safe)
            dtype=torch.bfloat16,  # or torch.float16 if you prefer
        ).to(device)
    else:
        print("🧠 Running in single-process mode with model sharding (device_map='auto')")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            dtype=torch.bfloat16,  # or torch.float16 if you prefer
        )

    deepspeed_config = ds_config if use_deepspeed else None

    args = TrainingArguments(
        output_dir=SAVE_PATH,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True
        gradient_checkpointing=False, ## for fdsp this needs to be off
        learning_rate=1e-5,
        num_train_epochs=1,  # or set max_steps for token-target budgeting
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        logging_steps=20,
        eval_steps=1000,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=3,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        bf16=True,
        report_to="wandb",
        # 🔥 FSDP toggle
        deepspeed=deepspeed_config,
        # fsdp=fsdp_mode,
        # fsdp_config=fsdp_cfg,
        fsdp=None,
        fsdp_config=None,
    )

    # LoRA config
    # https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
    lora_config = None
    if use_peft_lora:
        rank = 64
        lora_config = LoraConfig(
            r=rank,
            lora_alpha= 2 * rank,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            modules_to_save = ["embed_tokens", "lm_head"], ## List of modules apart from LoRA layers to be set as trainable
        )

        # model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()

    print_trainable_summary(model)
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        # data_collator=collator,
        peft_config=lora_config,
    )

    trainer.train()

    # --------------------------
    # 💾 Saving final model
    # --------------------------
    # Let accelerate sync all processes
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    # # FSDP-specific safe saving (only if enabled)
    # if getattr(trainer, "is_fsdp_enabled", False):
    #     # Set to save a full state dict on rank 0
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    #     if trainer.accelerator.is_main_process:
    #         trainer.save_model(SAVE_PATH)
    #         tokenizer.save_pretrained(SAVE_PATH)
    #         print(f"✅ [FSDP] Training complete. Model saved to: {SAVE_PATH}")
    # else:
    #     # Normal / DeepSpeed / DDP saving
    #     if (not is_distributed()) or dist.get_rank() == 0:
    #         trainer.save_model(SAVE_PATH)
    #         tokenizer.save_pretrained(SAVE_PATH)
    #         print(f"✅ Training complete. Model saved to: {SAVE_PATH}")

    # Normal / DeepSpeed / DDP saving
    if (not is_distributed()) or dist.get_rank() == 0:
        trainer.save_model(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"✅ Training complete. Model saved to: {SAVE_PATH}")

    if is_distributed():
        dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(train)
    ## python <file> --- Model loads with device_map="auto" and runs with model sharding
    ## torchrun --nproc_per_node=2 <file> --- Model loads normally and runs with full data parallelism

