"""
Trajectory generation + log-prob computation.

FSDP compatibility strategy:
  - generate():
    1) summon_full_params restores the full parameters
    2) temporarily replace every FSDP(layer) with the bare layer -> forward bypasses FSDP hooks
    3) each rank generates independently, triggering no all_gather -> no deadlock
  - compute_log_probs: uses the FSDP-wrapped model (call counts are already synced across ranks)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from contextlib import contextmanager, nullcontext
from transformers import AutoTokenizer
import logging

from config import DataConfig
from data_utils import extract_model_answer, check_answer, build_prompt

logger = logging.getLogger(__name__)


# ============================================================
# FSDP utility functions
# ============================================================

def _is_fsdp_model(model) -> bool:
    """Detect whether the model is wrapped by FSDP."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            return True
        inner = getattr(model, "module", None)
        if inner is not None and isinstance(inner, FSDP):
            return True
        return False
    except ImportError:
        return False


def _get_fsdp_module(model):
    """Get the outermost FSDP module."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        return model
    inner = getattr(model, "module", None)
    if inner is not None and isinstance(inner, FSDP):
        return inner
    return model


def _get_inner_model(model):
    """
    Get the innermost original CausalLM model (e.g. Qwen2ForCausalLM).
    Only strips the top-level wrappers (Accelerate -> FSDP -> module).
    Note: the inner decoder layers may still be wrapped by FSDP.
    """
    inner = model
    for _ in range(10):
        if hasattr(inner, '_fsdp_wrapped_module'):
            inner = inner._fsdp_wrapped_module
            continue
        if hasattr(inner, 'module'):
            inner = inner.module
            continue
        break
    return inner


def _get_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


# ============================================================
# Core FSDP fix: temporarily unwrap all FSDP wrappers during generation
# ============================================================

@contextmanager
def _fsdp_generate_context(model):
    """
    FSDP-safe generate context manager.

    Problem: Accelerate FSDP wraps not only the outermost model but also each
    decoder layer separately:

        FSDP(Qwen2ForCausalLM)
          -> model.layers = [
               FSDP(Qwen2DecoderLayer_0),   <- every layer has its own FSDP!
               FSDP(Qwen2DecoderLayer_1),
               ...
             ]

    In generate(), each generated token does one forward pass, and each forward
    through every FSDP(layer) triggers an all_gather (collective communication).
    Different ranks generate sequences of different lengths -> different all_gather
    counts -> NCCL deadlock.

    Fix:
    1) summon_full_params: restore all parameters to full shape (collective op, all ranks sync)
    2) temporarily replace every FSDP(child) in the module tree with child._fsdp_wrapped_module
       -> forward bypasses FSDP.forward() -> no all_gather triggered
    3) each rank can independently generate sequences of different lengths
    4) restore the FSDP wrappers on exit
    5) exit summon_full_params (collective op, all ranks sync)
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fsdp_module = _get_fsdp_module(model)

    # Step 1: summon_full_params (collective op -- all ranks must enter together)
    with FSDP.summon_full_params(fsdp_module, writeback=False):

        # Step 2: get the inner CausalLM model
        inner_model = _get_inner_model(model)

        # Step 3: recursively replace all FSDP submodules
        swapped = []  # [(parent_module, attr_name, original_fsdp_child)]

        def _swap_fsdp_children(module):
            """Recursively walk the module tree, replacing every FSDP(child) with child itself."""
            for name, child in list(module.named_children()):
                if isinstance(child, FSDP):
                    # Take out the original module inside FSDP
                    unwrapped_child = child._fsdp_wrapped_module
                    # Record it so we can restore later
                    swapped.append((module, name, child))
                    # Replace by operating directly on the _modules dict
                    # This works for nn.Module, nn.ModuleList, and nn.ModuleDict
                    module._modules[name] = unwrapped_child
                    # Continue recursing into the unwrapped submodules (they may also contain FSDP)
                    _swap_fsdp_children(unwrapped_child)
                else:
                    _swap_fsdp_children(child)

        _swap_fsdp_children(inner_model)

        if swapped:
            logger.debug(
                f"Temporarily unwrapped {len(swapped)} FSDP sub-modules for generation"
            )

        try:
            # Step 4: yield the clean model (no FSDP hooks)
            yield inner_model
        finally:
            # Step 5: restore all FSDP wrappers (in reverse order)
            for parent, name, original_fsdp in reversed(swapped):
                parent._modules[name] = original_fsdp

            if swapped:
                logger.debug(f"Restored {len(swapped)} FSDP sub-modules")


# ============================================================
# Trajectory generation
# ============================================================

@torch.no_grad()
def generate_trajectories(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[Dict],
    group_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    data_config: Optional[DataConfig] = None,
    blocked_token_ids: Optional[List[int]] = None,
) -> List[List[Dict]]:
    """Generate multiple trajectories for a batch of prompts.

    Args:
        blocked_token_ids: list of token IDs forbidden from being generated (e.g. <SKIP>)
    """
    model.eval()
    if data_config is None:
        data_config = DataConfig()

    grouped_trajectories = []
    is_fsdp = _is_fsdp_model(model)

    if is_fsdp:
        # FSDP: use the safe generation context (summon_full_params + unwrap submodule FSDP)
        ctx = _fsdp_generate_context(model)
    else:
        # Single-GPU / DDP: use the model directly
        ctx = nullcontext(model)

    with ctx as gen_model:
        # Disable autocast to avoid bf16 NaN accumulation
        with torch.amp.autocast('cuda', enabled=False):
            device = _get_device(gen_model)

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # ==========================================================
            # Core speedup change: enable left padding for matrix-parallel generation.
            # Push batch_size * group_size sequences onto the GPU at once.
            # ==========================================================
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"

            prompt_texts = [build_prompt(p["question"], tokenizer, data_config) for p in prompts]
            encoded = tokenizer(
                prompt_texts, 
                return_tensors="pt", 
                padding=True, 
                add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Record each prompt's true length (excluding left padding)
            prompt_lengths = attention_mask.sum(dim=1).tolist()
            input_seq_len = input_ids.shape[1]

            try:
                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=group_size, # generate M problems x N trajectories at once
                )
                if blocked_token_ids:
                    gen_kwargs["bad_words_ids"] = [[t] for t in blocked_token_ids]
                    
                outputs = gen_model.generate(**gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM during batched generation! Please reduce batch_size in calibration.py.")
                raise
            finally:
                tokenizer.padding_side = original_padding_side

            # Parse the batched output
            idx = 0
            for p_idx, prompt_data in enumerate(prompts):
                ground_truth = prompt_data["answer"]
                actual_prompt_len = prompt_lengths[p_idx]

                # Strip left padding to get the clean original prompt_ids
                clean_prompt_ids = input_ids[p_idx, -actual_prompt_len:]

                trajectories = []
                for g in range(group_size):
                    full_padded_ids = outputs[idx]
                    idx += 1

                    # Slice the newly generated response (after the input matrix's max length)
                    response_ids = full_padded_ids[input_seq_len:]

                    # Truncate at EOS
                    eos_pos = (response_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        cut = eos_pos[0].item() + 1
                        response_ids = response_ids[:cut]

                    # Assemble the true full_ids (with all padding removed)
                    full_ids = torch.cat([clean_prompt_ids, response_ids])
                    
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    predicted_answer = extract_model_answer(response_text)
                    is_correct = check_answer(predicted_answer, ground_truth)
                    
                    trajectories.append({
                        "prompt_ids": clean_prompt_ids.cpu(),
                        "prompt_length": actual_prompt_len,
                        "response_ids": response_ids.cpu(),
                        "response_text": response_text,
                        "full_ids": full_ids.cpu(),
                        "full_attention_mask": torch.ones_like(full_ids).cpu(),
                        "predicted_answer": predicted_answer,
                        "ground_truth": ground_truth,
                        "is_correct": is_correct,
                        "response_length": len(response_ids),
                    })
                grouped_trajectories.append(trajectories)

    model.train()
    return grouped_trajectories


# ============================================================
# Log-prob computation -- single trajectory
# Uses the FSDP-wrapped model here (call counts already synced across ranks)
# The FSDP forward hook automatically handles all_gather/reduce_scatter
# ============================================================

def compute_log_probs_single(
    model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Single-trajectory log probs (with gradient)."""
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    device = _get_device(model)
    full_ids = full_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits
    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    response_ids = full_ids[0, prompt_length:]
    response_log_probs = log_probs_all[0, prompt_length - 1:-1, :]

    log_probs = response_log_probs.gather(
        dim=-1,
        index=response_ids.unsqueeze(-1),
    ).squeeze(-1)

    return log_probs


@torch.no_grad()
def compute_log_probs_single_no_grad(
    model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Single-trajectory log probs (no gradient, used for the ref model)."""
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    device = _get_device(model)
    full_ids = full_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits
    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    response_ids = full_ids[0, prompt_length:]
    response_log_probs = log_probs_all[0, prompt_length - 1:-1, :]

    log_probs = response_log_probs.gather(
        dim=-1,
        index=response_ids.unsqueeze(-1),
    ).squeeze(-1)

    return log_probs


# ============================================================
# Evaluation
# ============================================================

def evaluate_accuracy(
    model,
    tokenizer: AutoTokenizer,
    problems: List[Dict],
    data_config: DataConfig,
    temperature: float = 0.0,
    max_problems: int = 500,
    max_new_tokens: int = 512,
    **kwargs,
) -> Dict:
    """Evaluate Pass@1."""
    model.eval()

    if len(problems) > max_problems:
        import random
        problems = random.sample(problems, max_problems)

    correct = 0
    total = 0
    is_fsdp = _is_fsdp_model(model)

    if is_fsdp:
        ctx = _fsdp_generate_context(model)
    else:
        ctx = nullcontext(model)

    with ctx as gen_model:
        with torch.amp.autocast('cuda', enabled=False):
            device = _get_device(gen_model)

            for i, problem in enumerate(problems):
                question = problem["question"]
                ground_truth = problem["answer"]

                prompt_text = build_prompt(question, tokenizer, data_config)
                encoded = tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    if temperature > 0:
                        gen_kwargs.update(
                            do_sample=True, temperature=temperature, top_p=0.95
                        )
                    else:
                        gen_kwargs.update(do_sample=False, top_k=None, top_p=None)

                    try:
                        outputs = gen_model.generate(**encoded, **gen_kwargs)
                    except RuntimeError as e:
                        if "inf" in str(e).lower() or "nan" in str(e).lower():
                            logger.warning(
                                f"NaN/Inf in eval problem {i}, skipping"
                            )
                            total += 1
                            continue
                        raise

                response_ids = outputs[0, encoded["input_ids"].shape[1]:]
                response_text = tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                predicted = extract_model_answer(response_text)

                if check_answer(predicted, ground_truth):
                    correct += 1
                total += 1

                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Eval: {i+1}/{len(problems)}, acc={correct/total:.4f}"
                    )

    model.train()
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Evaluation: {correct}/{total} = {accuracy:.4f}")

    return {"accuracy": accuracy, "correct": correct, "total": total}