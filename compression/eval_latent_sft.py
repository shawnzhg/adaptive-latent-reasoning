"""
Phase 1.5 Quick Validation (Unified with data_utils).

Verify whether the model retains its baseline reasoning ability after
Phase 1.5 SFT training.
Acceptance criterion: GSM8K >= 89.5% (Phase 1 was 90.5%, allowing a 1% drop).

Generation follows the Golden Rule:
  token(t) == <SKIP> -> input(t) = Adapter(h_{t-1})
  token(t) == normal -> input(t) = E(token(t))

Usage:
    python eval_phase15.py --checkpoint ./checkpoints/phase15/best
    python eval_phase15.py --checkpoint ./checkpoints/phase15/best --n_samples 500
"""

import os
import sys
import argparse
import logging
import random
import time

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_adapter import SkipAdapter
from config import DataConfig
# Use the unified data and evaluation utilities throughout
from data_utils import GSM8KEvalDataset, extract_model_answer, check_answer, build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_p15")


# ============================================================
# Golden Rule autoregressive generation (low-level logic preserved as-is)
# ============================================================

@torch.no_grad()
def generate_with_skip(
    model,
    skip_adapter,
    tokenizer,
    prompt_ids: torch.Tensor,
    skip_token_id: int,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    k_max: int = 8,
) -> dict:
    """
    Autoregressive generation following the Golden Rule.

    Golden Rule:
      predicted <SKIP> -> next-step input = Adapter(h_last)
      predicted normal token -> next-step input = E(token)
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]

    # Process the prompt
    out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    last_h = out.hidden_states[-1][0, -1, :]  # (d,)

    generated_ids = []
    n_skips = 0
    consecutive_skips = 0

    for step in range(max_new_tokens):
        logits = out.logits[0, -1, :]

        if temperature <= 0:
            next_token = logits.argmax().item()
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        if next_token == tokenizer.eos_token_id:
            break

        pos = torch.tensor([[prompt_len + step]], device=device)

        if next_token == skip_token_id:
            consecutive_skips += 1
            n_skips += 1

            if consecutive_skips > k_max:
                # Over the limit: mask <SKIP>, take the next-best token
                logits[skip_token_id] = -float("inf")
                if temperature <= 0:
                    next_token = logits.argmax().item()
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                consecutive_skips = 0
                generated_ids.append(next_token)
                emb = model.get_input_embeddings()(
                    torch.tensor([[next_token]], device=device)
                )
                out = model(
                    inputs_embeds=emb, position_ids=pos,
                    past_key_values=past_kv, use_cache=True,
                    output_hidden_states=True,
                )
            else:
                # Valid <SKIP>: use Adapter(h_last) as the next-step input
                z_skip = skip_adapter(last_h)
                out = model(
                    inputs_embeds=z_skip.unsqueeze(0).unsqueeze(0),
                    position_ids=pos,
                    past_key_values=past_kv, use_cache=True,
                    output_hidden_states=True,
                )
        else:
            consecutive_skips = 0
            generated_ids.append(next_token)
            emb = model.get_input_embeddings()(
                torch.tensor([[next_token]], device=device)
            )
            out = model(
                inputs_embeds=emb, position_ids=pos,
                past_key_values=past_kv, use_cache=True,
                output_hidden_states=True,
            )

        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][0, -1, :]

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "text": gen_text,
        "n_skips": n_skips,
        "n_explicit": len(generated_ids),
        "total_steps": len(generated_ids) + n_skips,
    }


# ============================================================
# GSM8K evaluation (refactored to call data_utils)
# ============================================================

def evaluate_gsm8k(
    model, skip_adapter, tokenizer, skip_token_id, data_config,
    n_samples=200, temperature=0.0,
) -> dict:
    """GSM8K evaluation (unified framework)."""

    model.eval()
    skip_adapter.eval()

    # Use the unified Dataset loader
    try:
        eval_dataset = GSM8KEvalDataset(data_config, tokenizer)
        problems = eval_dataset.problems
    except Exception as e:
        logger.error(f"Failed to load GSM8K using data_utils: {e}")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "avg_skips": 0, "skip_ratio": 0}

    indices = list(range(min(n_samples, len(problems))))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:n_samples]

    correct = 0
    total = 0
    skip_counts = []
    explicit_counts = []

    for i, idx in enumerate(indices):
        item = problems[idx]
        question = item["question"]
        gold = item["answer"]  # data_utils already extracts and cleans the reference answer

        # Use the unified prompt builder (v3_simple)
        prompt = build_prompt(question, tokenizer, data_config)
        prompt_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        result = generate_with_skip(
            model, skip_adapter, tokenizer, prompt_ids,
            skip_token_id, max_new_tokens=512, temperature=temperature,
        )

        # Use the unified answer-extraction and tolerance-comparison logic
        pred = extract_model_answer(result["text"])
        is_correct = check_answer(pred, gold)

        if is_correct:
            correct += 1
        total += 1
        skip_counts.append(result["n_skips"])
        explicit_counts.append(result["n_explicit"])

        if (i + 1) % 50 == 0:
            logger.info(
                f"  [{i+1}/{n_samples}] acc={correct/total:.4f}, "
                f"avg_skips={np.mean(skip_counts):.1f}, "
                f"avg_explicit={np.mean(explicit_counts):.0f}"
            )

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_skips": float(np.mean(skip_counts)) if skip_counts else 0,
        "avg_explicit": float(np.mean(explicit_counts)) if explicit_counts else 0,
        "skip_ratio": float(np.mean([
            s / max(s + e, 1) for s, e in zip(skip_counts, explicit_counts)
        ])) if skip_counts else 0,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Phase 1.5 checkpoint (e.g. ./checkpoints/phase15/best)")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy (deterministic)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Phase 1.5 Validation (Unified Evaluator)")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Samples: {args.n_samples}")
    logger.info("=" * 60)

    # Instantiate the data config (so data_utils works correctly)
    data_config = DataConfig()

    # -- Load model --
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    d_model = model.config.hidden_size
    skip_token_id = tokenizer.convert_tokens_to_ids("<SKIP>")
    logger.info(f"  d_model={d_model}, skip_token_id={skip_token_id}")

    # -- Load Adapter --
    skip_adapter = SkipAdapter(
        hidden_size=d_model, bottleneck_ratio=4
    ).to(dtype=torch.bfloat16, device="cuda")

    adapter_path = os.path.join(args.checkpoint, "skip_adapter.pt")
    if os.path.exists(adapter_path):
        skip_adapter.load_state_dict(
            torch.load(adapter_path, map_location="cuda", weights_only=False)
        )
        logger.info(f"  Adapter loaded from {adapter_path}")
    else:
        logger.warning(f"  No adapter found at {adapter_path}")
    skip_adapter.eval()

    # -- GSM8K evaluation --
    logger.info("\nEvaluating GSM8K...")
    t0 = time.time()
    gsm_results = evaluate_gsm8k(
        model, skip_adapter, tokenizer, skip_token_id, data_config,
        n_samples=args.n_samples, temperature=args.temperature,
    )
    elapsed = time.time() - t0

    # -- Results --
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1.5 VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  GSM8K Accuracy:     {gsm_results['accuracy']:.4f} "
                f"({gsm_results['correct']}/{gsm_results['total']})")
    logger.info(f"  Avg <SKIP>/response: {gsm_results['avg_skips']:.1f}")
    logger.info(f"  Avg explicit tokens: {gsm_results['avg_explicit']:.0f}")
    logger.info(f"  Skip ratio:          {gsm_results['skip_ratio']:.1%}")
    logger.info(f"  Time:                {elapsed:.0f}s")

    # -- Acceptance check --
    PHASE1_BASELINE = 0.905
    TOLERANCE = 0.01
    threshold = PHASE1_BASELINE - TOLERANCE

    if gsm_results["accuracy"] >= threshold:
        logger.info(f"\n  PASSED: {gsm_results['accuracy']:.4f} >= {threshold:.4f}")
        logger.info("  -> Ready for Phase 2 GRPO training")
    else:
        logger.warning(f"\n  FAILED: {gsm_results['accuracy']:.4f} < {threshold:.4f}")
        logger.warning("  -> Phase 1.5 may have degraded reasoning ability")
        logger.warning("  -> Consider: fewer steps, lower lr_base, or check SFT data quality")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()