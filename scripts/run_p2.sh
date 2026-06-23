#!/bin/bash
# Stage 3 — information-gain-guided GRPO that learns *when* to fold to latent.
# Multi-GPU via Accelerate + FSDP (edit --num_processes for your GPU count).
set -e
cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch \
    --num_processes=2 \
    --main_process_port 29778 \
    --mixed_precision=bf16 \
    --use_fsdp \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=Qwen2DecoderLayer \
    --fsdp_sharding_strategy=SHARD_GRAD_OP \
    --fsdp_state_dict_type=FULL_STATE_DICT \
    compression/train_grpo_latent.py \
    --checkpoint ./checkpoints/phase15/best \
    --output_dir ./checkpoints/phase2 \
    --num_steps 600 \
    --batch_size 12 \
    --group_size 8 \
    --temperature 0.7 \
    --max_new_tokens 1024 \
    --seed 42
