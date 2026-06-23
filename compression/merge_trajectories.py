import os
import torch
import glob

def merge_chunks():
    # directory where the chunks are saved
    raw_data_dir = "./checkpoints/phase1/raw_chunks"
    # output path for the merged result
    final_path = "./checkpoints/phase1/calibration_raw_data.pt"

    chunk_files = glob.glob(os.path.join(raw_data_dir, "chunk_*.pt"))
    if not chunk_files:
        print(f"Error: No chunks found in {raw_data_dir}")
        return

    print(f"[*] Found {len(chunk_files)} chunk files. Merging...")

    sft_dataset = []
    for f in chunk_files:
        # load each chunk
        chunk_data = torch.load(f, map_location="cpu", weights_only=False)
        trajectories = chunk_data["trajectories"]
        kvig_results = chunk_data["kvig_results"]

        # extract the fields needed for visualization and SFT training
        for traj, kvig_res in zip(trajectories, kvig_results):
            sft_dataset.append({
                "prompt_length": traj["prompt_length"],
                "response_length": traj["response_length"],
                "full_ids": traj["full_ids"], 
                "is_correct": traj["is_correct"],
                "kvig_values": kvig_res["kvig_values"],
                "mean_kvig": kvig_res["mean_kvig"],
                "d_eff_values": kvig_res["d_eff_values"]
            })
            
    print(f"[*] Successfully extracted {len(sft_dataset)} trajectories.")

    # save the final file
    torch.save(sft_dataset, final_path)
    print(f"[OK] Final dataset saved to: {final_path}")
    print("[*] You can now run offline_viz.py!")

if __name__ == "__main__":
    merge_chunks()