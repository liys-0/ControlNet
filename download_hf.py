import os
import shutil
from huggingface_hub import hf_hub_download

filename = "v1-5-pruned.ckpt"
target_path = "./models/v1-5-pruned.ckpt"

repos_to_try = ["stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]
downloaded_path = None

for repo_id in repos_to_try:
    try:
        print(f"Trying to download {filename} from {repo_id}...")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        break
    except Exception as e:
        print(f"Failed: {e}")

if downloaded_path:
    print(f"Copying to {target_path}...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(downloaded_path, target_path)
    print("Download and copy complete.")
else:
    print("Failed to download from all attempted repositories.")
    exit(1)
