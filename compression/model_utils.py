"""
Model utilities

Responsibilities:
1. Load the Qwen-2.5-7B-Instruct base model
2. Vocabulary extension: add the <SKIP> token
3. Semantic initialization of the <SKIP> embedding
4. Attach the Skip Adapter module
5. Freeze/unfreeze parameter management
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import logging

from config import ModelConfig, SPARKConfig
from latent_adapter import SkipAdapter

logger = logging.getLogger(__name__)


def load_base_model(
    config: ModelConfig,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model and tokenizer

    Returns:
        model: Qwen-2.5-7B-Instruct
        tokenizer: the corresponding tokenizer
    """
    torch_dtype = getattr(torch, dtype)

    logger.info(f"Loading model: {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch 2.4 native SDPA, no extra installation needed
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",  # GRPO requires left padding
    )

    # Ensure a pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Model loaded. Vocab size: {len(tokenizer)}, "
                f"Hidden size: {model.config.hidden_size}")

    return model, tokenizer


def add_skip_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ModelConfig,
) -> int:
    """
    Add the <SKIP> token to the vocabulary and initialize its embedding

    Initialization strategy:
    1. Find the embeddings of semantically related tokens ("skip", "pass", "omit", "therefore")
    2. Take their arithmetic mean e_mean
    3. Add small random noise: e_SKIP = e_mean + N(0, (0.01 x ||e_mean||)^2 I)

    Args:
        model: base model
        tokenizer: tokenizer
        config: model configuration

    Returns:
        skip_token_id: the token ID of <SKIP>
    """
    # Step 1: add <SKIP> to the vocabulary
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [config.skip_token_str]
    })
    assert num_added == 1, f"Expected to add 1 token, got {num_added}"

    skip_token_id = tokenizer.convert_tokens_to_ids(config.skip_token_str)
    logger.info(f"Added {config.skip_token_str} with token_id = {skip_token_id}")

    # Step 2: Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Step 3: collect embeddings of semantically related tokens
    embedding_layer = model.get_input_embeddings()
    semantic_embeddings = []

    for word in config.skip_semantic_tokens:
        # tokenization may produce multiple subwords; take the first one
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) > 0:
            emb = embedding_layer.weight.data[token_ids[0]].clone()
            semantic_embeddings.append(emb)
            logger.info(f"  Semantic token '{word}' -> id={token_ids[0]}, "
                        f"norm={emb.norm().item():.4f}")

    if len(semantic_embeddings) == 0:
        logger.warning("No semantic tokens found, using random initialization")
        return skip_token_id

    # Step 4: compute the mean
    e_mean = torch.stack(semantic_embeddings).mean(dim=0)
    mean_norm = e_mean.norm().item()

    # Step 5: add noise
    noise_std = config.skip_embedding_noise_scale * mean_norm
    noise = torch.randn_like(e_mean) * noise_std
    e_skip = e_mean + noise

    # Step 6: write the embedding
    with torch.no_grad():
        embedding_layer.weight.data[skip_token_id] = e_skip

    # Also update lm_head (the output layer) if weights are tied
    output_layer = model.get_output_embeddings()
    if output_layer is not None and not _is_tied(model):
        with torch.no_grad():
            output_layer.weight.data[skip_token_id] = e_skip

    logger.info(f"<SKIP> embedding initialized: norm={e_skip.norm().item():.4f}, "
                f"noise_std={noise_std:.6f}")

    return skip_token_id


def _is_tied(model) -> bool:
    """Check whether the embedding and lm_head share weights"""
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if output_emb is None:
        return False
    return input_emb.weight.data_ptr() == output_emb.weight.data_ptr()


def attach_skip_adapter(
    model: AutoModelForCausalLM,
    config: ModelConfig,
) -> SkipAdapter:
    """
    Create and attach the Skip Adapter to the model

    The Skip Adapter exists as a standalone module and does not modify the original
    model structure. Access it via model.skip_adapter.

    Args:
        model: base model
        config: model configuration

    Returns:
        skip_adapter: the created Skip Adapter instance
    """
    # Change 1: remove the bottleneck_ratio parameter, enabling the full-dimension residual network
    adapter = SkipAdapter(
        hidden_size=config.hidden_size,
    )

    # Move the adapter to the same device and precision as the model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    adapter = adapter.to(device=device, dtype=dtype)

    # Attach it to the model
    model.skip_adapter = adapter

    param_info = adapter.get_param_count()

    # Change 2: remove the reference to bottleneck_dim in the log, print the full dimension instead
    logger.info(f"Skip Adapter attached: {param_info['total_M']:.2f}M params "
                f"(full_dim={config.hidden_size})")

    return adapter


def get_last_hidden_state_before_norm(
    model: AutoModelForCausalLM,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[tuple] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[tuple]]:
    """
    Forward pass that returns the last-layer hidden state (before RMSNorm) and the logits

    For the Qwen2 architecture:
        model.model(...) returns last_hidden_state (after all Transformer blocks but before the final RMSNorm)
        model.model.norm(...) is the final RMSNorm
        model.lm_head(...) is the output projection

    Args:
        model: model
        input_ids: token IDs (batch_size, seq_len)
        inputs_embeds: directly provided embedding (used for the <SKIP> path)
        attention_mask: attention mask
        past_key_values: KV cache
        position_ids: position IDs
        use_cache: whether to return the KV cache

    Returns:
        h_last: hidden state before RMSNorm (batch_size, seq_len, hidden_size)
        logits: output logits (batch_size, seq_len, vocab_size)
        new_past_key_values: updated KV cache (if use_cache=True)
    """
    # Get the Transformer block output (before RMSNorm)
    outputs = model.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=use_cache,
        output_hidden_states=False,
    )

    h_last = outputs.last_hidden_state  # before RMSNorm

    # Manually apply RMSNorm + lm_head to obtain the logits
    h_normed = model.model.norm(h_last)
    logits = model.lm_head(h_normed)

    new_past = outputs.past_key_values if use_cache else None

    return h_last, logits, new_past


def setup_model_for_phase1(config: SPARKConfig):
    """
    Full model preparation for Phase 1

    Execution order:
    1. Load the base model
    2. Add the <SKIP> token and initialize its embedding
    3. Attach the Skip Adapter (W_down zero-initialized)
    4. Return the prepared model, tokenizer, and skip_token_id

    Note: <SKIP> is not enabled during Phase 1; these components are merely pre-installed.

    Returns:
        model: prepared model
        tokenizer: tokenizer
        skip_token_id: the token ID of <SKIP>
        skip_adapter: the Skip Adapter instance
    """
    # 1. Load the base model
    model, tokenizer = load_base_model(
        config.model,
        dtype=config.dtype,
        device=config.device,
    )

    # 2. Add the <SKIP> token
    skip_token_id = add_skip_token(model, tokenizer, config.model)

    # 3. Attach the Skip Adapter
    skip_adapter = attach_skip_adapter(model, config.model)

    # 4. Set the model to training mode
    model.train()

    # 5. The Skip Adapter is frozen during Phase 1 (no gradients needed)
    for param in skip_adapter.parameters():
        param.requires_grad = False
    logger.info("Skip Adapter frozen for Phase 1 (will be unfrozen in Phase 1.5)")

    # 6. The <SKIP> embedding is also frozen during Phase 1
    # (in practice the model rarely generates it; freezing is just an extra safeguard)
    # Note: since the embedding layer participates in training as a whole, we exclude
    # the <SKIP> gradient via a hook. This is handled in grpo_trainer.py.

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params / 1e9:.2f}B, "
                f"Trainable: {trainable_params / 1e9:.2f}B")

    return model, tokenizer, skip_token_id, skip_adapter


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    skip_adapter: SkipAdapter,
    step: int,
    output_dir: str,
    extra_state: Optional[dict] = None,
):
    """Save a checkpoint (model + tokenizer + Skip Adapter + extra state)"""
    import os
    save_path = os.path.join(output_dir, f"checkpoint-step-{step}")
    os.makedirs(save_path, exist_ok=True)

    # Save the model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Separately save the Skip Adapter (since it is not part of the original model structure)
    adapter_path = os.path.join(save_path, "skip_adapter.pt")
    torch.save(skip_adapter.state_dict(), adapter_path)

    # Save the extra state
    if extra_state is not None:
        state_path = os.path.join(save_path, "training_state.pt")
        torch.save(extra_state, state_path)

    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    config: SPARKConfig,
):
    """Load a checkpoint"""
    torch_dtype = getattr(torch, config.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        device_map=config.device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        padding_side="left",
    )

    # Rebuild the Skip Adapter
    skip_adapter = attach_skip_adapter(model, config.model)

    # Load the Skip Adapter weights
    import os
    adapter_path = os.path.join(checkpoint_path, "skip_adapter.pt")
    if os.path.exists(adapter_path):
        try:
            # Attempt to load, suppressing the weights_only warning
            state_dict = torch.load(adapter_path, map_location=config.device, weights_only=True)
            skip_adapter.load_state_dict(state_dict, strict=True)
            logger.info("Skip Adapter weights loaded successfully.")
        except RuntimeError as e:
            # Catch size mismatches caused by an architecture upgrade (e.g., removing the bottleneck)
            logger.warning("="*60)
            logger.warning("Architecture Mismatch Detected in Skip Adapter!")
            logger.warning("This is EXPECTED if you upgraded to the v2 Full-Dimension Adapter.")
            logger.warning("Discarding old weights and using fresh residual initialization.")
            logger.warning("="*60)

    skip_token_id = tokenizer.convert_tokens_to_ids(config.model.skip_token_str)

    # Load the extra state
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    extra_state = None
    if os.path.exists(state_path):
        extra_state = torch.load(state_path, map_location=config.device)

    return model, tokenizer, skip_token_id, skip_adapter, extra_state
