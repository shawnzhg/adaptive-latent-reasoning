#!/bin/bash
# Stage 1 — data preparation for latent compression.
# Initializes the base model + latent adapter, then builds the
# information-gain / counterfactual curriculum data used for SFT.
set -e
cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (Optional) initialize base model + latent adapter + special token:
# python compression/init_model.py
# (Optional) calibrate the information-gain signal on sampled trajectories:
# python compression/run_calibration.py --checkpoint ./checkpoints/phase0_base \
#     --num_problems 1000 --num_trajectories 16 \
#     --output ./checkpoints/phase1/calibration_stats.json --seed 42

python compression/build_curriculum_data.py \
    --checkpoint ./checkpoints/phase0_base \
    --raw_chunks_dir ./checkpoints/phase1/raw_chunks \
    --output_dir ./checkpoints/phase15_sft_data \
    --epsilon 0.1 \
    --max_skip_ratio 0.12 \
    --min_segment_len 3 \
    --target_skip_quantile 0.2
