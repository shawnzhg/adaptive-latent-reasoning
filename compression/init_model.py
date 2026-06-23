"""
Fast initialization of the 1.5B base model (skipping Phase 1).
"""
import os
import torch
import logging
from config import get_config
from model_utils import setup_model_for_phase1

logging.basicConfig(level=logging.INFO)

def main():
    config = get_config()
    print("=" * 60)
    print(f"Initializing Phase 0 Base from: {config.model.model_name_or_path}")
    print("=" * 60)
    
    # Get the model with <SKIP> and the Adapter attached
    model, tokenizer, skip_token_id, skip_adapter = setup_model_for_phase1(config)

    output_dir = "./checkpoints/phase0_base"
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # Core fix: save with HuggingFace's native method to ensure config.json is generated
    # =========================================================
    print("Saving HuggingFace standard model...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Save the Adapter
    print("Saving Skip Adapter...")
    torch.save(skip_adapter.state_dict(), os.path.join(output_dir, "skip_adapter.pt"))

    print(f"\nPhase 0 Base successfully saved to {output_dir}")
    print("   You can now run train_phase15.py safely!")

if __name__ == "__main__":
    main()