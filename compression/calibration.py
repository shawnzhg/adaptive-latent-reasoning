"""
KVIG calibration module.

Run KVIG baseline calibration after Phase 1 training:
1. Generate calibration data: 500 problems x 16 trajectories = 8000 trajectories
2. Compute per-step KVIG for each trajectory
3. Statistical analysis: correct group vs incorrect group
4. Validate the KVIG signal's effectiveness (AUC >= 0.65, p < 0.001)
5. Output key constants: d_eff_threshold, T_ref, KVIG_mean, KVIG_std

If validation fails, automatically search for better alpha, beta parameters.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
import logging
import json
import os
import time
import gc

import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
import optuna

from config import SPARKConfig, CalibrationConfig, KVIGConfig, ModelConfig
from information_gain import MCIGComputer
from rollout import generate_trajectories
from data_utils import build_prompt

logger = logging.getLogger(__name__)


class MCIGCalibrator:
    """
    MCIG calibrator.
    """

    def __init__(self, config: SPARKConfig):
        self.config = config
        self.cal_config = config.calibration
        # Note: since config.py is unchanged, we still pass config.kvig here
        self.mcig_computer = MCIGComputer(config.kvig, config.model)

    def run_calibration(
        self,
        model,
        tokenizer,
        calibration_problems: List[Dict],
    ) -> Dict:
        """
        Run the full calibration flow.

        Args:
            model: the checkpoint from completed Phase 1 training
            tokenizer: tokenizer
            calibration_problems: list of calibration problems

        Returns:
            calibration_results: {
                "d_eff_threshold": float,
                "t_ref": float,
                "kvig_mean": float,
                "kvig_std": float,
                "auc": float,
                "p_value": float,
                "correct_mean_mcig": float,
                "incorrect_mean_mcig": float,
                "d_eff_median": float,
                "d_eff_mean": float,
                "alpha": float,
                "beta": float,
                "num_correct_trajectories": int,
                "num_incorrect_trajectories": int,
                "validation_passed": bool,
            }
        """
        logger.info("=" * 60)
        logger.info("Starting KVIG Calibration")
        logger.info(f"  Problems: {len(calibration_problems)}")
        logger.info(f"  Trajectories per problem: {self.cal_config.num_trajectories_per_problem}")
        logger.info("=" * 60)

        cal_start = time.time()


        # ========================================
        # Step 1 & 2: Resumable generation and KVIG computation
        # ========================================
        trajectories_data, mcig_results = self._generate_and_compute_mcig_with_resume(
            model, tokenizer, calibration_problems
        )
        
        num_correct = sum(1 for t in trajectories_data if t["is_correct"])
        num_incorrect = len(trajectories_data) - num_correct
        logger.info(f"  Total Valid Trajectories: {len(trajectories_data)}")
        logger.info(f"  Correct: {num_correct}, Incorrect: {num_incorrect}")

        if num_correct < 50 or num_incorrect < 50:
            logger.warning("Too few correct/incorrect trajectories for reliable calibration!")

        logger.info("Step 3: Statistical analysis and validation...")
        stats = self._analyze_statistics(trajectories_data, mcig_results)

        passed = self._validate(stats)

        if not passed:
            logger.warning("Calibration validation FAILED. (MCIG uses analytical metrics, skipping search. Proceeding with caution...)")

        self._export_final_sft_data(trajectories_data, mcig_results)

        results = {
            "d_eff_threshold": stats["d_eff_median"],
            "t_ref": stats["correct_mean_length"],
            "mcig_mean": stats["overall_mean_mcig"],
            "mcig_std": stats["overall_std_mcig"],
            "auc": stats["auc"],
            "p_value": stats["p_value"],
            "correct_mean_mcig": stats["correct_mean_mcig"],
            "incorrect_mean_mcig": stats["incorrect_mean_mcig"],
            "d_eff_median": stats["d_eff_median"],
            "d_eff_mean": stats["d_eff_mean"],
            "alpha": getattr(self.mcig_computer, 'alpha_energy', 0.0),
            "beta": getattr(self.mcig_computer, 'decay', 0.0),
            "num_correct_trajectories": num_correct,
            "num_incorrect_trajectories": num_incorrect,
            "validation_passed": passed,
        }

        cal_time = time.time() - cal_start
        logger.info(f"Calibration completed in {cal_time / 60:.1f} minutes")
        self._log_results(results)

        # Save results
        self._save_results(results)

        return results


    @torch.no_grad()
    def _compute_all_mcig(
        self,
        model,
        trajectories: List[Dict],
        log_prefix: str = ""   
    ) -> List[Dict]:
        model.eval()
        mcig_results = []
        for i, traj in enumerate(trajectories):
            full_ids = traj["full_ids"].to(model.device)
            attention_mask = traj["full_attention_mask"].to(model.device)
            prompt_length = traj["prompt_length"]

            result = self.mcig_computer.compute_trajectory_from_model(
                model=model,
                input_ids=full_ids,
                attention_mask=attention_mask,
                prompt_length=prompt_length,
                # Note: MCIG does not actually need d_eff_threshold_override; passing it is absorbed by **kwargs, so leave it as-is
                d_eff_threshold_override=None,
            )
            mcig_results.append(result)
            if (i + 1) % 100 == 0:
                logger.info(f"  {log_prefix} Computed MCIG for {i+1}/{len(trajectories)} trajectories")
        return mcig_results


    def _export_final_sft_data(self, trajectories, mcig_results):
        final_path = os.path.join(
            os.path.dirname(self.cal_config.calibration_output_path),
            "calibration_raw_data.pt"
        )
        logger.info(f"Step 5: Exporting full dataset for Phase 1.5 SFT to {final_path}...")
        
        sft_dataset = []
        for traj, mcig_res in zip(trajectories, mcig_results):
            sft_dataset.append({
                "prompt_length": traj["prompt_length"],
                "response_length": traj["response_length"],
                "full_ids": traj["full_ids"], 
                "is_correct": traj["is_correct"],
                "mcig_values": mcig_res["mcig_values"],
                "mean_mcig": mcig_res["mean_mcig"],
                "d_eff_values": mcig_res.get("d_eff_values", []) # guard against KeyError
            })

        torch.save(sft_dataset, final_path)
        logger.info("  Final SFT data successfully compiled and saved.")


    def _generate_and_compute_mcig_with_resume(
        self, model, tokenizer, problems: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        [Enhanced] Batch-size-independent resumable generation.
        """
        raw_data_dir = os.path.join(
            os.path.dirname(self.cal_config.calibration_output_path), 
            "raw_chunks"
        )
        os.makedirs(raw_data_dir, exist_ok=True)

        # 1. Scan the directory, load all existing chunks, keyed by problem index (idx)
        completed_data = {} # problem_idx -> (trajectory_list, mcig_list)

        chunk_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.pt')]
        if chunk_files:
            logger.info(f"Scanning {len(chunk_files)} existing chunks for resume...")
            for f in chunk_files:
                try:
                    # Parse the original start index from the filename for progress alignment (optional)
                    # The key part is loading the data content
                    path = os.path.join(raw_data_dir, f)
                    data = torch.load(path, map_location="cpu")

                    # Assume we locate problems by their question string or by order in the original list
                    # The safest is to require the chunk order to match the current problems list
                    # In the original code, problems were sliced in order
                    import re
                    match = re.match(r"chunk_(\d+)_(\d+)\.pt", f)
                    if match:
                        start_idx = int(match.group(1))
                        end_idx = int(match.group(2))

                        trajs = data["trajectories"]
                        # Backward-compatible with a possible old key (kvig_results)
                        mcigs = data.get("mcig_results", data.get("kvig_results"))

                        num_problems_in_chunk = end_idx - start_idx

                        # ==========================================================
                        # Core fix: compatibility with the old flattened format (flat list).
                        # If the list length is far larger than the problem count, it is old flat data
                        # (e.g. 1024 trajectories for 64 problems).
                        # ==========================================================
                        if len(trajs) > num_problems_in_chunk:
                            group_size = len(trajs) // num_problems_in_chunk
                            # Re-fold into a nested list (List[List[Dict]])
                            trajs = [trajs[i * group_size : (i + 1) * group_size] for i in range(num_problems_in_chunk)]
                            mcigs = [mcigs[i * group_size : (i + 1) * group_size] for i in range(num_problems_in_chunk)]

                        # Map back to global indices
                        for i in range(len(trajs)):
                            completed_data[start_idx + i] = (trajs[i], mcigs[i])
                except Exception as e:
                    logger.warning(f"Failed to load chunk {f}: {e}")

        all_trajectories = [None] * len(problems)
        all_mcig_results = [None] * len(problems)

        # Fill in the already-loaded data
        for idx, (t_list, m_list) in completed_data.items():
            if idx < len(problems):
                all_trajectories[idx] = t_list
                all_mcig_results[idx] = m_list

        # 2. Iterate with the new batch_size, processing only the missing indices
        batch_size = 32 # suggest lowering to 8 or 16 to prevent OOM

        for batch_start in range(0, len(problems), batch_size):
            batch_end = min(batch_start + batch_size, len(problems))

            # Check whether this batch is fully completed
            if all(all_trajectories[i] is not None for i in range(batch_start, batch_end)):
                logger.info(f"  [Skip] Problems {batch_start}-{batch_end} already completed.")
                continue

            logger.info(f"Processing missing problems in range {batch_start}-{batch_end}...")

            # Find the problems actually missing in this batch
            missing_indices = [i for i in range(batch_start, batch_end) if all_trajectories[i] is None]
            batch_problems = [problems[i] for i in missing_indices]

            if not batch_problems:
                continue

            # 3. Run generation and computation
            grouped = generate_trajectories(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_problems,
                group_size=self.cal_config.num_trajectories_per_problem,
                temperature=self.cal_config.temperature,
                max_new_tokens=self.config.model.max_new_tokens,
                data_config=self.config.data,
            )

            # Flatten the results
            batch_trajs_flat = []
            for group in grouped:
                batch_trajs_flat.extend(group)

            batch_mcig = self._compute_all_mcig(model, batch_trajs_flat, log_prefix=f"[{batch_start}-{batch_end}]")

            # 4. Convert back to grouped form, fill the master table, and persist
            # Assume each problem corresponds to a fixed number of trajectories (group_size)
            gs = self.cal_config.num_trajectories_per_problem
            for i, p_idx in enumerate(missing_indices):
                start = i * gs
                end = (i + 1) * gs

                prob_trajs = batch_trajs_flat[start:end]
                prob_mcigs = batch_mcig[start:end]

                # Store in memory
                all_trajectories[p_idx] = prob_trajs
                all_mcig_results[p_idx] = prob_mcigs

                # Move to CPU
                for t in prob_trajs:
                    if isinstance(t.get("full_ids"), torch.Tensor):
                        t["full_ids"] = t["full_ids"].cpu()
                    if isinstance(t.get("full_attention_mask"), torch.Tensor):
                        t["full_attention_mask"] = t["full_attention_mask"].cpu()

            # Save the current chunk (use a new filename to avoid overwriting the old large chunk)
            chunk_file = os.path.join(raw_data_dir, f"chunk_{batch_start}_{batch_end}.pt")
            torch.save({
                "trajectories": [all_trajectories[i] for i in range(batch_start, batch_end)],
                "mcig_results": [all_mcig_results[i] for i in range(batch_start, batch_end)]
            }, chunk_file)
            
            logger.info(f"  [Save] New chunk {batch_start}-{batch_end} saved.")
            torch.cuda.empty_cache()
            gc.collect()

        # Finally, flatten List[List[Dict]] back to the originally expected return format
        final_trajs = []
        final_mcigs = []
        for t_group, m_group in zip(all_trajectories, all_mcig_results):
            final_trajs.extend(t_group)
            final_mcigs.extend(m_group)

        return final_trajs, final_mcigs

    def _analyze_statistics(
        self,
        trajectories: List[Dict],
        mcig_results: List[Dict],
    ) -> Dict:
        """
        Statistical analysis.

        Computes:
        - mean_KVIG difference between correct and incorrect groups
        - t-test p-value
        - AUC (using mean_KVIG to distinguish correct/incorrect)
        - d_eff distribution
        - T_ref (average length of correct trajectories)
        """
        correct_mean_mcigs = []
        incorrect_mean_mcigs = []
        all_mean_mcigs = []
        all_d_effs = []
        correct_lengths = []
        labels = []  # 1 = correct, 0 = incorrect

        for i, (traj, mcig_res) in enumerate(zip(trajectories, mcig_results)):
            mean_mcig = mcig_res["mean_mcig"]
            all_mean_mcigs.append(mean_mcig)
            all_d_effs.extend(mcig_res.get("d_eff_values", [])) # guard against KeyError

            if traj["is_correct"]:
                correct_mean_mcigs.append(mean_mcig)
                correct_lengths.append(traj["response_length"])
                labels.append(1)
            else:
                incorrect_mean_mcigs.append(mean_mcig)
                labels.append(0)

        # t-test
        if len(correct_mean_mcigs) > 1 and len(incorrect_mean_mcigs) > 1:
            t_stat, p_value = scipy_stats.ttest_ind(
                correct_mean_mcigs,
                incorrect_mean_mcigs,
                alternative="two-sided",  # the correct group should be higher
            )
        else:
            t_stat, p_value = 0.0, 1.0

        # AUC
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, all_mean_mcigs)
        else:
            auc = 0.5

        # d_eff statistics
        d_eff_array = np.array(all_d_effs)
        d_eff_valid = d_eff_array[np.isfinite(d_eff_array)]

        # Overall KVIG statistics
        all_mcigs_flat = []
        for mcig_res in mcig_results:
            all_mcigs_flat.extend(mcig_res["mcig_values"])

        stats = {
            "correct_mean_mcig": np.mean(correct_mean_mcigs) if correct_mean_mcigs else 0.0,
            "incorrect_mean_mcig": np.mean(incorrect_mean_mcigs) if incorrect_mean_mcigs else 0.0,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "auc": float(auc),
            "d_eff_median": float(np.median(d_eff_valid)) if len(d_eff_valid) > 0 else 5.0,
            "d_eff_mean": float(np.mean(d_eff_valid)) if len(d_eff_valid) > 0 else 5.0,
            "d_eff_std": float(np.std(d_eff_valid)) if len(d_eff_valid) > 0 else 1.0,
            "overall_mean_mcig": float(np.mean(all_mcigs_flat)) if all_mcigs_flat else 0.0,
            "overall_std_mcig": float(np.std(all_mcigs_flat)) if all_mcigs_flat else 1.0,
            "correct_mean_length": float(np.mean(correct_lengths)) if correct_lengths else 300.0,
            "cohen_d": self._cohens_d(correct_mean_mcigs, incorrect_mean_mcigs),
        }

        return stats

    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Compute the Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((m1 - m2) / pooled_std)

    def _validate(self, stats: Dict) -> bool:
        """
        Validate the KVIG signal's effectiveness.

        All of the following must hold:
        1. p < 0.001 (the correct group's mean_KVIG is significantly higher than the incorrect group's)
        2. AUC >= 0.65
        3. d_eff median > 3
        """
        cal_cfg = self.cal_config

        checks = {
            "p_value": stats["p_value"] < cal_cfg.min_p_value_threshold,
            "auc": stats["auc"] >= cal_cfg.min_auc,
            "d_eff_median": stats["d_eff_median"] > cal_cfg.min_d_eff_median,
        }

        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  Validation {name}: {status} "
                       f"(value={stats.get(name, 'N/A')}, "
                       f"threshold={getattr(cal_cfg, f'min_{name}' if f'min_{name}' in dir(cal_cfg) else name, 'N/A')})")

        all_passed = all(checks.values())

        if all_passed:
            logger.info("  All calibration checks PASSED")
        else:
            logger.warning("  Some calibration checks FAILED")

        return all_passed

    def _log_results(self, results: Dict):
        """Print the calibration results."""
        logger.info("=" * 60)
        logger.info("KVIG Calibration Results:")
        logger.info(f"  d_eff_threshold (d_eff median): {results['d_eff_threshold']:.4f}")
        logger.info(f"  T_ref (avg correct length): {results['t_ref']:.1f}")
        logger.info(f"  MCIG_mean: {results.get('mcig_mean', 0.0):.6f}")
        logger.info(f"  MCIG_std: {results.get('mcig_std', 0.0):.6f}")
        logger.info(f"  AUC: {results['auc']:.4f}")
        logger.info(f"  p-value: {results['p_value']:.2e}")
        logger.info(f"  Correct group mean MCIG: {results['correct_mean_mcig']:.6f}")
        logger.info(f"  Incorrect group mean MCIG: {results['incorrect_mean_mcig']:.6f}")
        logger.info(f"  α: {results['alpha']:.3f}, β: {results['beta']:.3f}")
        logger.info(f"  Validation: {'PASSED' if results['validation_passed'] else 'FAILED'}")
        logger.info("=" * 60)

    def _save_results(self, results: Dict):
        """Save the calibration results to a JSON file."""
        output_path = self.cal_config.calibration_output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ensure all values are JSON serializable
        serializable = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Calibration results saved to {output_path}")


def load_calibration_results(path: str) -> Dict:
    """Load the calibration results."""
    with open(path, "r") as f:
        return json.load(f)


def apply_calibration_to_config(
    calibration_results: Dict,
    kvig_config: KVIGConfig,
) -> KVIGConfig:
    """
    Apply the calibration results to the KVIG config.

    Args:
        calibration_results: calibration output
        kvig_config: KVIG config

    Returns:
        the updated KVIG config
    """
    kvig_config.d_eff_threshold = calibration_results["d_eff_threshold"]
    kvig_config.t_ref = calibration_results["t_ref"]
    kvig_config.kvig_mean = calibration_results["mcig_mean"] 
    kvig_config.kvig_std = calibration_results["mcig_std"]
    kvig_config.alpha = calibration_results["alpha"]
    kvig_config.beta = calibration_results["beta"]

    logger.info("Calibration results applied to KVIG config:")
    logger.info(f"  d_eff_threshold = {kvig_config.d_eff_threshold}")
    logger.info(f"  T_ref = {kvig_config.t_ref}")
    logger.info(f"  α = {kvig_config.alpha}, β = {kvig_config.beta}")

    return kvig_config