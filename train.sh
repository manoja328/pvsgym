# python pvs_RL_train_dapt.py
# python pvs_RL_train.py

CONFIG="pvs_v5fixed.yaml"

## Model loads with device_map="auto" and runs with model sharding
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# python sft_train.py --config_path $CONFIG  --bs 32

###  Model loads normally and runs with full data parallelism
# torchrun --nproc_per_node=4 sft_train.py --config_path $CONFIG  --bs 4

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# CONFIG_FILE="ddp_config.yaml"
# accelerate launch \
#   --num_machines 1 \
#   --num_processes 4 \
#   --multi_gpu \
#   --mixed_precision bf16 \
#   --config_file "$CONFIG_FILE" \
#    sft_train.py \
#    --config_path "$CONFIG" \
#    --bs 32 \
#    --use_peft_lora True


# export CUDA_VISIBLE_DEVICES=4,5,6,7
# CONFIG_FILE="fsdp_config.yaml"
# accelerate launch \
#   --num_machines 1 \
#   --num_processes 4 \
#   --multi_gpu \
#   --mixed_precision bf16 \
#   --config_file "$CONFIG_FILE" \
#    sft_train.py \
#    --config_path "$CONFIG" \
#    --bs 1 \
#    --use_peft_lora False


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# CONFIG="pvs_v6.yaml"
# CONFIG_FILE="fsdp_config2.yaml"
# accelerate launch \
#   --num_machines 1 \
#   --num_processes 4 \
#   --multi_gpu \
#   --mixed_precision bf16 \
#   --config_file "$CONFIG_FILE" \
#    sft_train.py \
#    --config_path "$CONFIG" \
#    --bs 1 \
#    --use_peft_lora False \
#    --use_fsdp True

## this worked on newgpu with 48gb gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3   # or 4,5,6,7 on your node
CONFIG="pvs_v6.yaml"
torchrun --nproc_per_node=4 sft_train.py \
  --config_path "$CONFIG" \
  --bs 8 \
  --use_peft_lora False \
  --testrun False \
  --use_fsdp False \
  --use_deepspeed True \
  --ds_config "deepspeed_cf.json"


# accelerate launch \
#   --num_machines 1 \
#   --num_processes 8 \
#   --multi_gpu \
#   --mixed_precision bf16 \
#   --config_file "$CONFIG_FILE" \
#    sft_train.py \
#    --config_path "$CONFIG" \
#    --bs 1 \
#    --use_peft_lora False


## eval
# CUDA_VISIBLE_DEVICES=0 python sft_eval.py --config_path $CONFIG

## hosting in stremalit
# CUDA_VISIBLE_DEVICES=0 streamlit run sft_gradio.py  --server.port 7861