from huggingface_hub import hf_hub_download
import os

target_dir = "./models"
os.makedirs(target_dir, exist_ok=True)

print("Starting download of v1-5-pruned.ckpt from Hugging Face...")
path = hf_hub_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    filename="v1-5-pruned.ckpt",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)
print(f"Successfully downloaded to: {path}")
