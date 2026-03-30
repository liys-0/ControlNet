import os
import cv2
import numpy as np
import json
import glob
from tqdm import tqdm
import shutil


def prepare_defect_dataset():
    input_base = "/home/lys/projects/cadd/amd/POC_Dataset/patches"
    pfib_dir = os.path.join(input_base, "pfib")
    gt_dir = os.path.join(input_base, "gt")

    output_base = "/home/lys/projects/POC_Dataset/for_ControlNet_defect"
    source_dir = os.path.join(output_base, "source")
    target_dir = os.path.join(output_base, "target")
    json_path = os.path.join(output_base, "prompt.json")

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    pfib_files = glob.glob(os.path.join(pfib_dir, "*.png"))

    json_lines = []

    for pfib_path in tqdm(pfib_files, desc="Processing defect images"):
        filename = os.path.basename(pfib_path)

        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            continue

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if gt_img is None or not np.any(gt_img > 0):
            continue

        pfib_img = cv2.imread(pfib_path)
        if pfib_img is None:
            continue

        edge_map = cv2.Canny(pfib_img, 100, 200)
        edge_map_3c = np.stack([edge_map] * 3, axis=-1)

        out_source_path = os.path.join(source_dir, filename)
        cv2.imwrite(out_source_path, edge_map_3c)

        out_target_path = os.path.join(target_dir, filename)
        shutil.copy2(pfib_path, out_target_path)

        json_lines.append(
            {
                "source": f"source/{filename}",
                "target": f"target/{filename}",
                "prompt": "defect pfib",
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        for line in json_lines:
            f.write(json.dumps(line) + "\n")

    print(
        f"Dataset generation complete. Saved {len(json_lines)} defect image pairs to {output_base}"
    )


if __name__ == "__main__":
    prepare_defect_dataset()
