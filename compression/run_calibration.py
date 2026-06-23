"""
KVIG calibration entry script.

Run this script after Phase 1 training to perform KVIG calibration:
1. Load the Phase 1 checkpoint
2. Randomly select 500 problems from the MATH training set
3. Generate 16 trajectories per problem
4. Compute per-step KVIG for each trajectory
5. Statistical analysis + signal validation
6. Output calibration constants: d_eff_threshold, T_ref, KVIG_mean, KVIG_std

Usage:
    python run_calibration.py --checkpoint ./checkpoints/phase1/final
    python run_calibration.py --checkpoint ./checkpoints/phase1/best
"""

import os
import sys
import argparse
import logging
import torch
import random
import numpy as np
import json

from config import SPARKConfig, get_config
from model_utils import load_checkpoint, setup_model_for_phase1
from data_utils import MathProblemDataset
from calibration import MCIGCalibrator, apply_calibration_to_config
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("calibration")


def parse_args():
    parser = argparse.ArgumentParser(description="KVIG Calibration")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Phase 1 checkpoint")
    parser.add_argument("--num_problems", type=int, default=500,
                        help="Number of calibration problems")
    parser.add_argument("--num_trajectories", type=int, default=16,
                        help="Trajectories per problem")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for calibration results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Config
    config = get_config()
    config.calibration.num_problems = args.num_problems
    config.calibration.num_trajectories_per_problem = args.num_trajectories
    if args.output:
        config.calibration.calibration_output_path = args.output
    config.seed = args.seed

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    logger.info("=" * 70)
    logger.info("KVIG Calibration")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Problems: {config.calibration.num_problems}")
    logger.info(f"Trajectories/problem: {config.calibration.num_trajectories_per_problem}")
    logger.info("=" * 70)

    # ========================================
    # 1. Load the Phase 1 checkpoint
    # ========================================
    logger.info("Loading Phase 1 checkpoint...")

    if os.path.exists(os.path.join(args.checkpoint, "config.json")):
        # Load from the checkpoint
        model, tokenizer, skip_token_id, skip_adapter, extra_state = load_checkpoint(
            args.checkpoint, config
        )
    else:
        # If nothing was saved, set up from scratch
        logger.info("No saved checkpoint found, setting up from scratch...")
        model, tokenizer, skip_token_id, skip_adapter = setup_model_for_phase1(config)

    model.eval()
    logger.info("Model loaded and set to eval mode")

    # ========================================
    # 2. Prepare calibration data
    # ========================================
    logger.info("Preparing calibration data...")
    dataset = MathProblemDataset(config.data, tokenizer, seed=config.seed)

    # Select calibration problems from the MATH dataset
    calibration_problems = dataset.get_calibration_subset(
        n=config.calibration.num_problems,
        source="math",
    )
    logger.info(f"Selected {len(calibration_problems)} MATH problems for calibration")

    # ========================================
    # 3. Run calibration
    # ========================================
    calibrator = MCIGCalibrator(config)
    results = calibrator.run_calibration(
        model=model,
        tokenizer=tokenizer,
        calibration_problems=calibration_problems,
    )

    # ========================================
    # 4. Apply the results to the config
    # ========================================
    if results["validation_passed"]:
        updated_config = apply_calibration_to_config(results, config.kvig)

        # Save the updated config
        config_save_path = os.path.join(
            os.path.dirname(config.calibration.calibration_output_path),
            "kvig_config_calibrated.json",
        )
        with open(config_save_path, "w") as f:
            json.dump({
                "d_eff_threshold": updated_config.d_eff_threshold,
                "t_ref": updated_config.t_ref,
                "mcig_mean": updated_config.kvig_mean, # Note: the updated_config.kvig_mean name is kept unchanged
                "mcig_std": updated_config.kvig_std,
                "alpha": updated_config.alpha,
                "beta": updated_config.beta,
            }, f, indent=2)
        logger.info(f"Calibrated KVIG config saved to {config_save_path}")
    else:
        logger.error("Calibration FAILED. Please check Phase 1 training quality.")
        logger.error("Possible actions:")
        logger.error("  1. Train Phase 1 for more steps")


        
        logger.error("  2. Increase calibration data (--num_problems)")
        logger.error("  3. Check if model can solve math problems at all")

    # ========================================
    # 5. Print summary
    # ========================================
    logger.info("=" * 70)
    logger.info("Calibration Summary")
    logger.info("=" * 70)
    logger.info(f"  Status: {'PASSED' if results['validation_passed'] else 'FAILED'}")
    logger.info(f"  d_eff_threshold: {results['d_eff_threshold']:.4f}")
    logger.info(f"  T_ref: {results['t_ref']:.1f} tokens")
    logger.info(f"  MCIG_mean: {results['mcig_mean']:.6f}")
    logger.info(f"  MCIG_std: {results['mcig_std']:.6f}")
    logger.info(f"  AUC: {results['auc']:.4f}")
    logger.info(f"  p-value: {results['p_value']:.2e}")
    logger.info(f"  Correct mean MCIG: {results['correct_mean_mcig']:.6f}")
    logger.info(f"  Incorrect mean MCIG: {results['incorrect_mean_mcig']:.6f}")
    logger.info(f"  α: {results['alpha']:.3f}, β: {results['beta']:.3f}")
    logger.info("=" * 70)

    if results["validation_passed"]:
        logger.info("Next step: Phase 1.5 SFT warmup (train_phase15.py)")
    else:
        logger.info("Fix Phase 1 before proceeding to Phase 1.5")

    return results


if __name__ == "__main__":
    main()