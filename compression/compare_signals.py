#!/usr/bin/env python3
"""
Multi-Signal Comparison experiment.

Goal: on existing Phase 1 calibration data, compare several internal signals
for their effectiveness as a skip reward. No need to regenerate trajectories;
just re-run a forward pass to extract information at different levels.

Signal list:
  1. KVIG        - existing, loaded directly from raw_chunks
  2. CosSim      - cos(h_t, h_{t-1}), hidden-state cosine similarity (1-cos = change magnitude)
  3. AttnEntropy - entropy of the last-layer attention weights
  4. PolicyEntropy - entropy of the next-token logit distribution (ReLU of H_{t-1} - H_t = causal entropy drop)
  5. MCIG        - trajectory curvature kappa_t x causal entropy drop grad H_t x hidden-state norm ||h_t||

For each signal it computes:
  - Token-level CV (coefficient of variation)
  - Skip Safety (whether the answer survives dropping the lowest-20% tokens)
  - Trajectory-level AUC (distinguishing correct/incorrect)
  - Semantic analysis (what the low/high-signal tokens are)

Usage:
    python compare_signals.py \
        --checkpoint /path/to/phase1/best \
        --raw_chunks_dir /path/to/raw_chunks \
        --output_dir ./signal_comparison_results \
        --max_trajectories 500 \
        --batch_size 4

    # Quick test (small amount of data):
    python compare_signals.py \
        --checkpoint ./checkpoints/phase1/best \
        --raw_chunks_dir ./checkpoints/phase1/raw_chunks \
        --output_dir ./signal_comparison_results \
        --max_trajectories 50 \
        --batch_size 2
"""

import os
import sys
import argparse
import logging
import json
import glob
import time
import gc
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from data_utils import extract_model_answer, check_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("signal_compare")


# ============================================================
# Part 1: Data loading
# ============================================================

def load_raw_chunks(raw_chunks_dir: str, max_trajectories: int = -1) -> Tuple[List[Dict], List[Dict]]:
    """
    Load all saved trajectories and KVIG results from the raw_chunks directory.

    raw_chunks format (produced by calibration.py):
    {
        "trajectories": [
            {
                "full_ids": tensor (seq_len,),
                "full_attention_mask": tensor (seq_len,),
                "prompt_length": int,
                "response_length": int,
                "is_correct": bool,
                "predicted_answer": str,
                "ground_truth": str,
                ...
            }, ...
        ],
        "kvig_results": [
            {
                "kvig_values": list[float],  # per-token
                "mean_kvig": float,
                "d_eff_values": list[float],
            }, ...
        ]
    }
    """
    chunk_files = sorted(glob.glob(os.path.join(raw_chunks_dir, "chunk_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {raw_chunks_dir}")
    
    logger.info(f"Found {len(chunk_files)} chunk files in {raw_chunks_dir}")
    
    all_trajectories = []
    all_kvig_results = []
    
    for cf in chunk_files:
        try:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            all_trajectories.extend(data["trajectories"])
            all_kvig_results.extend(data["kvig_results"])
        except Exception as e:
            logger.warning(f"Failed to load {cf}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_trajectories)} trajectories total")
    
    # Limit the count
    if max_trajectories > 0 and len(all_trajectories) > max_trajectories:
        # Proportional sampling, preserving the correct/incorrect ratio
        correct_idx = [i for i, t in enumerate(all_trajectories) if t["is_correct"]]
        incorrect_idx = [i for i, t in enumerate(all_trajectories) if not t["is_correct"]]
        
        n_correct = len(correct_idx)
        n_incorrect = len(incorrect_idx)
        ratio = n_correct / max(n_correct + n_incorrect, 1)
        
        n_sample_correct = int(max_trajectories * ratio)
        n_sample_incorrect = max_trajectories - n_sample_correct
        
        np.random.seed(42)
        sampled_correct = np.random.choice(correct_idx, min(n_sample_correct, n_correct), replace=False).tolist()
        sampled_incorrect = np.random.choice(incorrect_idx, min(n_sample_incorrect, n_incorrect), replace=False).tolist()
        
        sampled_idx = sorted(sampled_correct + sampled_incorrect)
        all_trajectories = [all_trajectories[i] for i in sampled_idx]
        all_kvig_results = [all_kvig_results[i] for i in sampled_idx]
        
        logger.info(f"Sampled {len(all_trajectories)} trajectories "
                    f"({len(sampled_correct)} correct, {len(sampled_incorrect)} incorrect)")
    
    n_correct = sum(1 for t in all_trajectories if t["is_correct"])
    n_incorrect = len(all_trajectories) - n_correct
    logger.info(f"Final dataset: {n_correct} correct, {n_incorrect} incorrect")
    
    return all_trajectories, all_kvig_results


# ============================================================
# Part 2: Signal computer
# ============================================================

class MultiSignalComputer:
    """
    Run a single forward pass, extract all intermediate representations
    needed by the signals, then compute the 5 signals.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def compute_all_signals(
        self,
        full_ids: torch.Tensor,       # (seq_len,)
        attention_mask: torch.Tensor,  # (seq_len,)
        prompt_length: int,
        kvig_values: Optional[List[float]] = None,  # existing KVIG, optional
    ) -> Dict[str, np.ndarray]:
        """
        Compute all signals for a single trajectory.

        Returns:
            {
                "kvig": np.array (resp_len,),
                "cossim": np.array (resp_len,),        # 1 - cos(h_t, h_{t-1})
                "attn_entropy": np.array (resp_len,),
                "policy_entropy": np.array (resp_len,),
                "causal_entropy_drop": np.array (resp_len,),  # ReLU(H_{t-1} - H_t)
                "mcig": np.array (resp_len,),
            }
        """
        input_ids = full_ids.unsqueeze(0).to(self.device)
        attn_mask = attention_mask.unsqueeze(0).to(self.device)
        
        resp_len = full_ids.shape[0] - prompt_length
        if resp_len <= 2:
            # Too short, return all zeros
            empty = np.zeros(max(resp_len, 1))
            return {name: empty.copy() for name in
                    ["kvig", "cossim", "attn_entropy", "policy_entropy",
                     "causal_entropy_drop", "mcig"]}

        # --- Single forward pass, extract all intermediate information ---
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
        
        # logits: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        
        # hidden states: last layer (1, seq_len, d_model)
        last_hidden = outputs.hidden_states[-1][0]  # (seq_len, d_model)

        # attention weights: last layer (1, num_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)

        # Slice the response part
        h_resp = last_hidden[prompt_length:]      # (resp_len, d)
        logits_resp = logits[prompt_length:]       # (resp_len, vocab)

        # Also need the last prompt position (as h_{-1} and logits_{-1})
        prev_idx = max(0, prompt_length - 1)
        h_prev_first = last_hidden[prev_idx]  # (d,)
        logits_prev_first = logits[prev_idx]  # (vocab,)

        # --- Signal 1: KVIG (use directly if available) ---
        if kvig_values is not None and len(kvig_values) == resp_len:
            sig_kvig = np.array(kvig_values, dtype=np.float32)
        else:
            # Fallback: use cossim as a proxy
            sig_kvig = np.zeros(resp_len, dtype=np.float32)

        # --- Signal 2: Cosine Similarity Change ---
        # cossim(t) = 1 - cos(h_t, h_{t-1})
        # High value = h changes a lot = high information gain = important token
        # Low value = h barely changes = redundant token
        sig_cossim = np.zeros(resp_len, dtype=np.float32)
        h_all = torch.cat([h_prev_first.unsqueeze(0), h_resp], dim=0)  # (resp_len+1, d)
        for t in range(resp_len):
            cos_val = F.cosine_similarity(
                h_all[t+1].unsqueeze(0).float(), h_all[t].unsqueeze(0).float()
            ).item()
            sig_cossim[t] = 1.0 - cos_val  # change magnitude

        # --- Signal 3: Attention Entropy (converted to Focus / concentration) ---
        # Remove causal-mask position bias, compute normalized entropy and invert to a focus score
        # High focus (large value) = concentrated attention = key node
        sig_attn_entropy = np.zeros(resp_len, dtype=np.float32)
        for t in range(resp_len):
            pos = prompt_length + t
            if pos < last_attn.shape[-1]:
                attn_dist = last_attn[:, pos, :pos+1].float().clamp(min=1e-10)
                head_entropies = -(attn_dist * attn_dist.log()).sum(dim=-1)
                mean_h = head_entropies.mean().item()
                
                max_possible_entropy = np.log(pos + 1)
                normalized_entropy = mean_h / max_possible_entropy if max_possible_entropy > 0 else 0.0
                # Math correction: 1.0 - normalized entropy = focus (larger means more concentrated and more critical, consistent with higher_is_important=True)
                sig_attn_entropy[t] = max(0.0, min(1.0, 1.0 - normalized_entropy))

        # --- Signal 4: Policy Entropy & strict causal entropy drop ---
        sig_policy_entropy = np.zeros(resp_len, dtype=np.float32)
        
        all_logits = torch.cat([logits_prev_first.unsqueeze(0), logits_resp], dim=0).float()
        log_probs = F.log_softmax(all_logits, dim=-1)
        probs = log_probs.exp()
        entropies = -(probs * log_probs).sum(dim=-1).cpu().numpy()  # (resp_len+1,)
        
        sig_policy_entropy = entropies[:-1].astype(np.float32)
        
        # Pure zero-inflation (ReLU), creating an absolute filler-text basin
        raw_drop = entropies[:-1] - entropies[1:]
        sig_causal_drop = F.relu(torch.tensor(raw_drop)).numpy().astype(np.float32)

        # --- Signal 5: Dense-MCIG (Additive OR-Logic & RL-Dense) ---
        # Keep the variable name as sig_mcig so the dict lookup does not fail
        sig_mcig = np.zeros(resp_len, dtype=np.float32)

        if resp_len >= 2:
            # 1. Geodesic curvature (guards against latent-space distortion)
            h_all_norm = F.normalize(h_all.float(), p=2, dim=-1)
            delta_h_norm = h_all_norm[1:] - h_all_norm[:-1]

            cos_t0 = F.cosine_similarity(h_all_norm[1].unsqueeze(0), h_all_norm[0].unsqueeze(0), eps=1e-8).item()

            # Add .cpu().numpy() to copy the Tensor back to memory
            kappa_vec = (1.0 - F.cosine_similarity(delta_h_norm[1:], delta_h_norm[:-1], dim=-1, eps=1e-8)).cpu().numpy()

            # 2. Symmetric information shock (JSD; whether converging or diverging, any drastic distribution change marks a key node)
            P_prev = probs[:-1]
            P_curr = probs[1:]
            M = 0.5 * (P_prev + P_curr)
            kl_prev_M = torch.sum(P_prev * (torch.log(P_prev + 1e-10) - torch.log(M + 1e-10)), dim=-1)
            kl_curr_M = torch.sum(P_curr * (torch.log(P_curr + 1e-10) - torch.log(M + 1e-10)), dim=-1)
            jsd_vec = (0.5 * (kl_prev_M + kl_curr_M)).cpu().numpy()

            # 3. Absolute log-energy shock (guards against global inflation, sensitive in both directions)
            h_norms = h_all.float().norm(dim=-1).cpu().numpy()
            log_energy_diff = np.abs(np.log(h_norms[1:] + 1e-8) - np.log(h_norms[:-1] + 1e-8))

            # 4. Pure additive fusion (OR logic), producing a dense gradient (Dense Reward)
            # Special case for the first step
            sig_mcig[0] = max(jsd_vec[0], max(log_energy_diff[0] * 2.0, max(0.0, 1.0 - cos_t0)))

            # Vectorized max-pooling for subsequent steps
            sig_mcig[1:] = np.maximum(kappa_vec, np.maximum(jsd_vec[1:], log_energy_diff[1:] * 2.0))
            #sig_mcig[0] = jsd_vec[0] + log_energy_diff[0] + max(0.0, 1.0 - cos_t0)
            #sig_mcig[1:] = kappa_vec + jsd_vec[1:] + log_energy_diff[1:]

        # ==========================================
        # Momentum envelope (still needed; protects featureless inertial payloads such as multi-digit numbers)
        # ==========================================
        decay_factor = 0.8
        if len(sig_mcig) > 0:
            for t in range(1, len(sig_mcig)):
                sig_mcig[t] = max(sig_mcig[t], sig_mcig[t-1] * decay_factor)

        # --- Free GPU memory ---
        del outputs, logits, last_hidden, last_attn, h_resp, logits_resp
        del h_all, all_logits, log_probs, probs
        if resp_len >= 2:
            del h_all_norm, delta_h_norm
        torch.cuda.empty_cache()

        
        result = {
            "kvig": sig_kvig,
            "cossim": sig_cossim,
            "attn_entropy": sig_attn_entropy,
            "policy_entropy": sig_policy_entropy,
            "causal_entropy_drop": sig_causal_drop,
            "mcig": sig_mcig,
        }
        
        for key in result:
            result[key] = np.nan_to_num(result[key], nan=0.0, posinf=0.0, neginf=0.0)
            
        return result



# ============================================================
# Part 3: Diagnostic evaluator
# ============================================================

class SignalDiagnostics:
    """Run standardized diagnostics for each signal."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def evaluate_signal(
        self,
        signal_name: str,
        all_signals: List[np.ndarray],     # signal values per trajectory
        all_trajectories: List[Dict],       # corresponding trajectory metadata
        skip_ratio: float = 0.2,
        higher_is_important: bool = True,   # True = high value is an important token
    ) -> Dict:
        """
        Run full diagnostics for a single signal.

        Args:
            signal_name: signal name
            all_signals: list of per-token signal values per trajectory
            all_trajectories: trajectory metadata
            skip_ratio: drop the lowest X% of tokens
            higher_is_important:
                True  -> high signal = important (KVIG, cossim, attn_entropy, causal_drop, mcig)
                False -> low signal = important (policy_entropy: low entropy = certain = important)

        Returns:
            diagnostic result dict
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating signal: {signal_name}")
        logger.info(f"  Direction: {'higher=important' if higher_is_important else 'lower=important'}")
        logger.info(f"{'='*60}")
        
        all_cv = []
        skip_safe = 0
        skip_total = 0
        
        # Trajectory-level: collect the mean signal per trajectory
        correct_means = []
        incorrect_means = []
        all_means = []
        labels = []

        # Semantic analysis: collect statistics on low/high-signal tokens
        low_token_counter = defaultdict(int)
        high_token_counter = defaultdict(int)
        
        for i, (sig, traj) in enumerate(zip(all_signals, all_trajectories)):
            if len(sig) < 3:
                continue
            
            # -- Token-level CV --
            mean_s = np.mean(sig)
            std_s = np.std(sig)
            cv = std_s / (abs(mean_s) + 1e-8)
            all_cv.append(cv)

            # -- Trajectory-level mean --
            all_means.append(mean_s)
            labels.append(1 if traj["is_correct"] else 0)
            if traj["is_correct"]:
                correct_means.append(mean_s)
            else:
                incorrect_means.append(mean_s)
            
            # -- Skip Safety (upgraded to dynamic-threshold adaptive pruning) --
            if "response_ids" in traj and traj["response_ids"] is not None:
                response_ids = traj["response_ids"]
            else:
                response_ids = traj["full_ids"][traj["prompt_length"]:]

            if len(sig) > 5 and traj["is_correct"] and len(response_ids) == len(sig):
                tokens = self.tokenizer.convert_ids_to_tokens(response_ids.tolist())
                
                # ==========================================
                # Core upgrade: dynamic threshold based on the signal landscape.
                # Instead of rigidly cutting 20%, cut all "below 10% of the mean" plateau noise.
                # ==========================================
                sig_mean = np.mean(sig)
                if higher_is_important:
                    dynamic_thresh = sig_mean * 0.1  # threshold: 10% of the mean
                    candidate_skip_idx = np.where(sig < dynamic_thresh)[0]
                else:
                    # For signals like policy entropy where lower is more important
                    dynamic_thresh = sig_mean * 1.5
                    candidate_skip_idx = np.where(sig > dynamic_thresh)[0]

                # Keep the sorting so we can process from least to next-least important
                if higher_is_important:
                    sorted_idx = np.argsort(sig)
                else:
                    sorted_idx = np.argsort(sig)[::-1]

                # Cap the max skip at 75% to avoid deleting everything in extreme cases
                max_allow_skip = int(len(sig) * 0.75)

                # ==========================================
                # Core fix: syntax-and-number protection barrier.
                # The real Skip Adapter emits format symbols, so offline testing should not delete them.
                # ==========================================
                actual_skip_idx = set()
                protected_chars = set(['\\', '{', '}', '[', ']', '(', ')', '=', '+', '-', '*', '/', '^', '_', '$'])

                # Only choose from candidates judged "filler" by the dynamic threshold
                candidate_set = set(candidate_skip_idx.tolist())

                for idx in sorted_idx:
                    if idx not in candidate_set:
                        continue  # above the dynamic threshold: never delete

                    tok_str = tokens[idx]
                    if any(c.isdigit() for c in tok_str) or any(c in protected_chars for c in tok_str):
                        continue
                    
                    actual_skip_idx.add(idx)
                    if len(actual_skip_idx) >= max_allow_skip:
                        break
                
                # Record this trajectory's actual skip ratio (for final statistics)
                actual_ratio = len(actual_skip_idx) / len(sig)
                if not hasattr(self, "accumulated_skip_ratios"):
                    self.accumulated_skip_ratios = defaultdict(list)
                self.accumulated_skip_ratios[signal_name].append(actual_ratio)
                
                keep_idx = [j for j in range(len(sig)) if j not in actual_skip_idx]
                
                if len(keep_idx) > 0:
                    kept_ids = response_ids[keep_idx]
                    kept_text = self.tokenizer.decode(kept_ids, skip_special_tokens=True)

                    kept_answer = extract_model_answer(kept_text)
                    reference = traj.get("ground_truth", "")
                    if not reference:
                        reference = traj.get("predicted_answer", "")

                    still_correct = False
                    if reference:
                        if kept_answer and check_answer(kept_answer, reference):
                            still_correct = True
                        else:
                            ref_clean = reference.strip()
                            if len(ref_clean) > 0 and ref_clean in kept_text:
                                still_correct = True
                            else:
                                # Fallback strategy C: GSM8K final-number extraction
                                import re
                                nums = re.findall(r'-?\d+\.?\d*', kept_text)
                                if nums:
                                    try:
                                        if abs(float(nums[-1].replace(',', '')) - float(
                                                ref_clean.replace(',', ''))) < 1e-6:
                                            still_correct = True
                                    except:
                                        pass

                    skip_total += 1
                    if still_correct:
                        skip_safe += 1
            
            # -- Semantic statistics (sample the first 200 trajectories) --
            if i < 200 and len(response_ids) == len(sig):
                tokens = self.tokenizer.convert_ids_to_tokens(response_ids.tolist())
                n_show = min(5, len(sig) // 4)
                if n_show > 0:
                    if higher_is_important:
                        low_idx = np.argsort(sig)[:n_show]
                        high_idx = np.argsort(sig)[-n_show:]
                    else:
                        low_idx = np.argsort(sig)[-n_show:]   # the "unimportant" ones
                        high_idx = np.argsort(sig)[:n_show]    # the "important" ones
                    
                    for j in low_idx:
                        if j < len(tokens):
                            low_token_counter[tokens[j]] += 1
                    for j in high_idx:
                        if j < len(tokens):
                            high_token_counter[tokens[j]] += 1
        
        # -- Compute summary metrics --
        mean_cv = float(np.mean(all_cv)) if all_cv else 0.0
        
        # AUC
        if len(set(labels)) > 1 and len(all_means) == len(labels):
            try:
                auc = roc_auc_score(labels, all_means)
            except Exception:
                auc = 0.5
        else:
            auc = 0.5
        
        # t-test
        if len(correct_means) > 1 and len(incorrect_means) > 1:
            t_stat, p_value = scipy_stats.ttest_ind(
                correct_means, incorrect_means, alternative="two-sided"
            )
        else:
            t_stat, p_value = 0.0, 1.0
        
        # Skip safety rate
        skip_safe_rate = skip_safe / max(skip_total, 1)
        
        # Cohen's d
        if len(correct_means) > 1 and len(incorrect_means) > 1:
            pooled_std = np.sqrt(
                ((len(correct_means)-1)*np.std(correct_means, ddof=1)**2 +
                 (len(incorrect_means)-1)*np.std(incorrect_means, ddof=1)**2) /
                (len(correct_means) + len(incorrect_means) - 2)
            )
            cohens_d = (np.mean(correct_means) - np.mean(incorrect_means)) / max(pooled_std, 1e-8)
        else:
            cohens_d = 0.0
        
        # Top semantic tokens
        top_low = sorted(low_token_counter.items(), key=lambda x: -x[1])[:15]
        top_high = sorted(high_token_counter.items(), key=lambda x: -x[1])[:15]

        # -- Print results --
        actual_avg_skip = np.mean(self.accumulated_skip_ratios[signal_name]) if hasattr(self, "accumulated_skip_ratios") and signal_name in self.accumulated_skip_ratios else skip_ratio
        logger.info(f"\n--- {signal_name} Results ---")
        logger.info(f"  Token-level CV:    {mean_cv:.3f}")
        logger.info(f"  Real Skip Ratio:   {actual_avg_skip:.1%} (Dynamic)")
        logger.info(f"  Skip Safety:       {skip_safe}/{skip_total} = {skip_safe_rate:.1%}")
        logger.info(f"  Trajectory AUC:    {auc:.4f}")
        logger.info(f"  p-value:           {p_value:.2e}")
        logger.info(f"  Cohen's d:         {cohens_d:.4f}")
        logger.info(f"  Correct mean:      {np.mean(correct_means):.6f}" if correct_means else "  Correct mean: N/A")
        logger.info(f"  Incorrect mean:    {np.mean(incorrect_means):.6f}" if incorrect_means else "  Incorrect mean: N/A")
        logger.info(f"  Top REDUNDANT tokens: {[t[0] for t in top_low[:10]]}")
        logger.info(f"  Top IMPORTANT tokens: {[t[0] for t in top_high[:10]]}")
        
        return {
            "signal_name": signal_name,
            "mean_cv": mean_cv,
            "skip_safe": skip_safe,
            "skip_total": skip_total,
            "skip_safe_rate": skip_safe_rate,
            "auc": auc,
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "correct_mean": float(np.mean(correct_means)) if correct_means else None,
            "incorrect_mean": float(np.mean(incorrect_means)) if incorrect_means else None,
            "top_redundant_tokens": top_low[:15],
            "top_important_tokens": top_high[:15],
            "all_cv": [float(x) for x in all_cv],
            "higher_is_important": higher_is_important,
        }
    


# ============================================================
# Part 4: Main flow
# ============================================================

def compute_signals_for_all_trajectories(
    model,
    tokenizer,
    trajectories: List[Dict],
    kvig_results: List[Dict],
    device: str = "cuda",
    batch_size: int = 1,  # process one at a time to avoid OOM
) -> Dict[str, List[np.ndarray]]:
    """
    Compute all signals for all trajectories.

    Returns:
        {
            "kvig": [np.array, np.array, ...],       # one array per trajectory
            "cossim": [np.array, ...],
            "attn_entropy": [np.array, ...],
            "policy_entropy": [np.array, ...],
            "causal_entropy_drop": [np.array, ...],
            "mcig": [np.array, ...],
        }
    """
    computer = MultiSignalComputer(model, tokenizer, device)
    
    all_signals = {
        "kvig": [],
        "cossim": [],
        "attn_entropy": [],
        "policy_entropy": [],
        "causal_entropy_drop": [],
        "mcig": [],
    }
    
    start_time = time.time()
    
    for i, (traj, kvig_res) in enumerate(zip(trajectories, kvig_results)):
        try:
            signals = computer.compute_all_signals(
                full_ids=traj["full_ids"],
                attention_mask=traj["full_attention_mask"],
                prompt_length=traj["prompt_length"],
                kvig_values=kvig_res.get("kvig_values", None),
            )
            
            for name in all_signals:
                all_signals[name].append(signals[name])
        
        except Exception as e:
            logger.warning(f"Failed on trajectory {i}: {e}")
            # Insert an empty array as a placeholder
            resp_len = traj.get("response_length", 1)
            for name in all_signals:
                all_signals[name].append(np.zeros(resp_len))
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(trajectories) - i - 1)
            logger.info(f"  Computed {i+1}/{len(trajectories)} trajectories "
                       f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
        
        # Periodically free GPU memory
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    total_time = time.time() - start_time
    logger.info(f"All signals computed in {total_time:.0f}s "
               f"({total_time/len(trajectories):.1f}s/traj)")
    
    return all_signals


def generate_comparison_report(results: Dict[str, Dict], output_dir: str):
    """Generate the comparison report."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Multi-Signal Comparison Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Header
    header = f"{'Signal':<22} {'CV':>6} {'SkipSafe':>10} {'AUC':>7} {'Cohen_d':>9} {'p-value':>12}"
    report_lines.append(header)
    report_lines.append("-" * 80)
    
    # Sort by skip safety
    sorted_results = sorted(results.values(), key=lambda x: -x["skip_safe_rate"])
    
    for r in sorted_results:
        line = (f"{r['signal_name']:<22} "
                f"{r['mean_cv']:>6.3f} "
                f"{r['skip_safe']}/{r['skip_total']:>4} ({r['skip_safe_rate']:.0%})"
                f"{r['auc']:>7.4f} "
                f"{r['cohens_d']:>9.4f} "
                f"{r['p_value']:>12.2e}")
        report_lines.append(line)
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("RANKING (for skip reward usage):")
    report_lines.append("=" * 80)
    
    # Composite score: skip_safety * 0.5 + CV_normalized * 0.3 + AUC_bonus * 0.2
    scores = {}
    max_cv = max(r["mean_cv"] for r in sorted_results) if sorted_results else 1.0
    for r in sorted_results:
        cv_norm = r["mean_cv"] / max(max_cv, 1e-8)
        auc_bonus = max(0, r["auc"] - 0.5) * 2  # 0~1 scale
        score = r["skip_safe_rate"] * 0.5 + cv_norm * 0.3 + auc_bonus * 0.2
        scores[r["signal_name"]] = score
    
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranked, 1):
        r = results[name]
        report_lines.append(
            f"  #{rank} {name:<20} score={score:.3f}  "
            f"(safety={r['skip_safe_rate']:.0%}, CV={r['mean_cv']:.3f}, AUC={r['auc']:.4f})"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("SEMANTIC ANALYSIS (top redundant/important tokens per signal):")
    report_lines.append("=" * 80)
    
    for r in sorted_results:
        report_lines.append(f"\n  [{r['signal_name']}]")
        redundant_tokens = [t[0] for t in r["top_redundant_tokens"][:10]]
        important_tokens = [t[0] for t in r["top_important_tokens"][:10]]
        report_lines.append(f"    Redundant: {redundant_tokens}")
        report_lines.append(f"    Important: {important_tokens}")
    
    # Recommendation
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATION:")
    report_lines.append("=" * 80)
    
    best_name = ranked[0][0] if ranked else "N/A"
    best_result = results.get(best_name, {})
    report_lines.append(f"  Best signal for Phase 1.5 SFT data: {best_name}")
    report_lines.append(f"  Best signal for Phase 2 skip reward: {best_name}")
    if len(ranked) >= 2:
        second_name = ranked[1][0]
        report_lines.append(f"  Ensemble candidate: {best_name} + {second_name}")
    
    report_text = "\n".join(report_lines)

    # Print
    print("\n" + report_text)

    # Save
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Multi-Signal Comparison")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Phase 1 model checkpoint path")
    parser.add_argument("--raw_chunks_dir", type=str, required=True,
                        help="Directory with raw_chunks/*.pt from calibration")
    parser.add_argument("--output_dir", type=str, default="./signal_comparison_results",
                        help="Output directory")
    parser.add_argument("--max_trajectories", type=int, default=500,
                        help="Max trajectories to process (-1 for all)")
    parser.add_argument("--skip_ratio", type=float, default=0.2,
                        help="Fraction of tokens to skip in safety test")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for forward pass (1 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved signals if available")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ========================================
    # Step 1: Load calibration data
    # ========================================
    logger.info("Step 1: Loading raw calibration data...")
    trajectories, kvig_results = load_raw_chunks(
        args.raw_chunks_dir, max_trajectories=args.max_trajectories
    )
    
    # ========================================
    # Step 2: Load model
    # ========================================
    signals_cache = os.path.join(args.output_dir, "all_signals.pt")
    
    if args.resume and os.path.exists(signals_cache):
        logger.info("Step 2: Loading cached signals (--resume)...")
        cached = torch.load(signals_cache, map_location="cpu", weights_only=False)
        all_signals = cached["all_signals"]
        # Ensure trajectory counts match
        min_len = min(len(trajectories), len(all_signals["kvig"]))
        trajectories = trajectories[:min_len]
        kvig_results = kvig_results[:min_len]
        for k in all_signals:
            all_signals[k] = all_signals[k][:min_len]
        logger.info(f"Loaded {min_len} cached signal vectors")
    else:
        logger.info("Step 2: Loading model for forward passes...")

        # Dynamically import project modules
        checkpoint_parent = os.path.dirname(os.path.dirname(args.checkpoint))
        project_root = os.path.dirname(checkpoint_parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from config import get_config
        config = get_config()
        
        from model_utils import load_checkpoint, setup_model_for_phase1
        if os.path.exists(os.path.join(args.checkpoint, "config.json")):
            model, tokenizer, skip_token_id, skip_adapter, _ = load_checkpoint(
                args.checkpoint, config)
        else:
            model, tokenizer, skip_token_id, skip_adapter = setup_model_for_phase1(config)
        model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model loaded on {device}")
        
        # ========================================
        # Step 3: Compute all signals
        # ========================================
        logger.info("Step 3: Computing all signals (this is the main computation)...")
        all_signals = compute_signals_for_all_trajectories(
            model=model,
            tokenizer=tokenizer,
            trajectories=trajectories,
            kvig_results=kvig_results,
            device=device,
        )
        
        # Cache signals (to avoid recomputation)
        logger.info(f"Caching signals to {signals_cache}...")
        torch.save({"all_signals": all_signals}, signals_cache)

        # Free the model's GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Reload tokenizer (if the model was freed)
        # tokenizer is already in memory, no need to reload

    # ========================================
    # Step 4: Run diagnostics for each signal
    # ========================================
    logger.info("\nStep 4: Running diagnostics for all signals...")

    # Need the tokenizer
    if "tokenizer" not in dir() or tokenizer is None:
        from transformers import AutoTokenizer
        # Load the tokenizer from the checkpoint
        from config import get_config
        config = get_config()
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name_or_path, trust_remote_code=True
        )
    
    diagnostics = SignalDiagnostics(tokenizer)
    
    # Define signals and their direction
    signal_configs = {
        "kvig":                 {"higher_is_important": True},
        "cossim":               {"higher_is_important": True},   # high change = important
        "attn_entropy":         {"higher_is_important": True},   # high entropy = integrates many sources = important
        "policy_entropy":       {"higher_is_important": True},  # low entropy = high certainty = possibly important
        "causal_entropy_drop":  {"higher_is_important": True},   # large drop = causally critical
        "mcig":                 {"higher_is_important": True},   # composite metric
    }
    
    all_results = {}
    for signal_name, config_dict in signal_configs.items():
        if signal_name in all_signals and len(all_signals[signal_name]) > 0:
            result = diagnostics.evaluate_signal(
                signal_name=signal_name,
                all_signals=all_signals[signal_name],
                all_trajectories=trajectories,
                skip_ratio=args.skip_ratio,
                **config_dict,
            )
            all_results[signal_name] = result
    
    # ========================================
    # Step 5: Generate the comparison report
    # ========================================
    logger.info("\nStep 5: Generating comparison report...")
    generate_comparison_report(all_results, args.output_dir)

    # Save detailed results as JSON
    json_results = {}
    for name, r in all_results.items():
        json_r = {k: v for k, v in r.items() if k != "all_cv"}
        json_r["all_cv_mean"] = r["mean_cv"]
        json_r["all_cv_std"] = float(np.std(r["all_cv"])) if r["all_cv"] else 0.0
        # Convert token tuples to a serializable format
        json_r["top_redundant_tokens"] = [(t, int(c)) for t, c in r["top_redundant_tokens"]]
        json_r["top_important_tokens"] = [(t, int(c)) for t, c in r["top_important_tokens"]]
        json_results[name] = json_r
    
    json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed results saved to {json_path}")

    logger.info("\nMulti-signal comparison complete!")
    logger.info(f"All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()