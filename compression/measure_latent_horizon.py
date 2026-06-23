import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import logging
import argparse
from tqdm import tqdm

from config import get_config
from model_utils import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lrh_syntax")

class LRHMeasurer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.think_token_id = self.tokenizer.convert_tokens_to_ids("<SKIP>")
        if self.think_token_id == self.tokenizer.unk_token_id:
            logger.warning("<THINK> token not found; using <extra_id_0> instead for testing.")
            self.think_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def _is_critical_token(self, token_str):
        """
        Syntax-aware core: decide whether the current token is a critical
        anchor that must remain explicit.
        Mirrors the E (Execution) and T (Transition) idea from ThinKV.
        """
        # Strip tokenizer-specific prefix characters (handles Llama's ' ' and Qwen/GPT's 'Ġ')
        clean_str = token_str.replace('Ġ', '').replace(' ', '').replace('Ċ', '').strip().lower()

        if not clean_str:
            return False

        # 1. Contains digits (core of Execution)
        if any(char.isdigit() for char in clean_str):
            return True

        # 2. Math operators and formatting symbols (core of Execution)
        math_symbols = {'+', '-', '=', '*', '/', '^', '_', '{', '}', '(', ')', '[', ']', '\\', '|', '<', '>'}
        if any(sym in clean_str for sym in math_symbols):
            return True

        # 3. Logical connectives and key verbs (mirrors ThinKV's Transition)
        transitions = {
            'so', 'but', 'wait', 'then', 'if', 'let', 'thus', 'therefore', 
            'actually', 'now', 'sqrt', 'frac', 'pi', 'suppose', 'find', 'verify'
        }
        if clean_str in transitions:
            return True
            
        return False

    @torch.no_grad()
    def compute_answer_likelihood(self, input_ids, attention_mask, answer_start_idx):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_log_probs = -loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view(shift_labels.size())
        
        answer_log_probs = token_log_probs[0, answer_start_idx - 1:]
        return answer_log_probs.mean().item()

    def create_k_step_latent_sequence(self, input_ids, prompt_length, answer_start_idx, k):
        """
        Syntax-aware masking: keep critical tokens; for non-critical tokens,
        keep one fallback anchor every K steps.
        """
        masked_ids = input_ids.clone()
        cot_length = answer_start_idx - prompt_length

        if cot_length <= 0:
            return masked_ids

        # Get the token strings of the CoT segment
        cot_ids = input_ids[0, prompt_length:answer_start_idx].tolist()
        cot_tokens = self.tokenizer.convert_ids_to_tokens(cot_ids)
        
        last_anchor_pos = -1
        explicit_count = 0
        
        for i in range(cot_length):
            token_str = cot_tokens[i]
            is_critical = self._is_critical_token(token_str)
            
            # Critical symbol, or a forced anchor to prevent consecutive latents from exceeding K
            if is_critical or (i - last_anchor_pos >= k):
                last_anchor_pos = i
                explicit_count += 1
            else:
                masked_ids[0, prompt_length + i] = self.think_token_id
                
        return masked_ids, explicit_count / cot_length

    def run_measurement(self, trajectories_path, num_samples=100, k_values=[1, 2, 3, 4, 5, 6, 8, 10]):
        logger.info(f"Loading trajectories from {trajectories_path}")
        data = torch.load(trajectories_path, map_location="cpu")
        correct_trajs = [t for t in data if t.get("is_correct", True)]
        
        if len(correct_trajs) > num_samples:
            random.seed(42)
            correct_trajs = random.sample(correct_trajs, num_samples)

        results = {k: [] for k in k_values}
        explicit_ratios = {k: [] for k in k_values}
        
        self.model.eval()
        
        for idx, traj in enumerate(tqdm(correct_trajs, desc="Measuring Syntax-Aware LRH")):
            input_ids = traj["full_ids"].to(self.model.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            prompt_length = traj["prompt_length"]
            seq_len = input_ids.size(1)
            answer_start_idx = max(prompt_length, seq_len - 25) 
            
            orig_ll = self.compute_answer_likelihood(
                input_ids, torch.ones_like(input_ids), answer_start_idx
            )
            
            for k in k_values:
                if k == 1:
                    results[k].append(0.0)
                    explicit_ratios[k].append(1.0)
                    continue
                    
                masked_ids, ratio = self.create_k_step_latent_sequence(
                    input_ids, prompt_length, answer_start_idx, k
                )
                
                masked_ll = self.compute_answer_likelihood(
                    masked_ids, torch.ones_like(masked_ids), answer_start_idx
                )
                
                results[k].append(masked_ll - orig_ll)
                explicit_ratios[k].append(ratio)

        logger.info("="*60)
        logger.info("Syntax-Aware LRH Results (Threshold: Delta LL > -0.15)")
        logger.info("="*60)
        
        lrh = 1
        for k in k_values:
            deltas = results[k]
            pass_rate = sum(1 for d in deltas if d > -0.15) / len(deltas)
            mean_delta = np.mean(deltas)
            mean_ratio = np.mean(explicit_ratios[k])
            
            logger.info(f"K = {k:2d} | Pass Rate: {pass_rate*100:5.1f}% | Delta LL: {mean_delta:7.4f} | Explicit Token Ratio: {mean_ratio*100:4.1f}%")
            
            if pass_rate >= 0.70:
                lrh = max(lrh, k)
                
        logger.info("="*60)
        logger.info(f"Syntax-Aware Estimated LRH = {lrh}")
        logger.info("="*60)
        
        return lrh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Syntax-Aware LRH")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    config = get_config()
    model, tokenizer, _, _, _ = load_checkpoint(args.checkpoint, config)
    
    measurer = LRHMeasurer(model, tokenizer, config)
    measurer.run_measurement(
        trajectories_path=args.data_path,
        num_samples=args.num_samples,
        k_values=[1, 2, 3, 4, 5, 6, 8, 10, 15]
    )