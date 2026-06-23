#!/bin/bash
# Stage 2 — curriculum SFT that installs the latent-feedback ability.
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
    compression/train_sft_latent.py \
    --checkpoint ./checkpoints/phase0_base \
    --sft_data ./checkpoints/phase15_sft_data/sft_training_data.pt \
    --output_dir ./checkpoints/phase15 \
    --lr_base 1e-6 \
    --lr_adapter 1e-4 \
    --adapter_loss_weight 3.0 \
    --batch_size 16 \
    --grad_accum 4 \
    --num_steps 1000 \
    --warmup_steps 50 \
    --eval_every 100 \
    --save_every 100 \
    --max_grad_norm 1.0 \
    --seed 42
