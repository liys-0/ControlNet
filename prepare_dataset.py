import os
import cv2
import numpy as np
import json
import glob
from tqdm import tqdm
import shutil


def prepare_dataset():
    input_base = "/home/lys/projects/cadd/amd/POC_Dataset/patches"
    avalon_dir = os.path.join(input_base, "avalon")
    pfib_dir = os.path.join(input_base, "pfib")
    gt_dir = os.path.join(input_base, "gt")

    output_base = "/home/lys/projects/POC_Dataset/for_ControlNet"
    source_dir = os.path.join(output_base, "source")
    target_dir = os.path.join(output_base, "target")
    json_path = os.path.join(output_base, "prompt.json")

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    avalon_files = glob.glob(os.path.join(avalon_dir, "*.png"))

    json_lines = []

    for avalon_path in tqdm(avalon_files, desc="Processing images"):
        filename = os.path.basename(avalon_path)

        gt_path = os.path.join(gt_dir, filename)
        pfib_path = os.path.join(pfib_dir, filename)

        if not os.path.exists(gt_path) or not os.path.exists(pfib_path):
            continue

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None or np.any(gt_img > 0):
            continue

        avalon_img = cv2.imread(avalon_path)
        if avalon_img is None:
            continue

        edge_map = cv2.Canny(avalon_img, 100, 200)
        edge_map_3c = np.stack([edge_map] * 3, axis=-1)

        out_source_path = os.path.join(source_dir, filename)
        cv2.imwrite(out_source_path, edge_map_3c)

        out_target_path = os.path.join(target_dir, filename)
        shutil.copy2(pfib_path, out_target_path)

        json_lines.append(
            {
                "source": f"source/{filename}",
                "target": f"target/{filename}",
                "prompt": "normal pfib",
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        for line in json_lines:
            f.write(json.dumps(line) + "\n")

    print(
        f"Dataset generation complete. Saved {len(json_lines)} valid image pairs to {output_base}"
    )


if __name__ == "__main__":
    prepare_dataset()
