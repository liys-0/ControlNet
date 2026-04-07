import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset


class DefectDataset(Dataset):
    def __init__(self, base_dir="/home/lys/projects/POC_Dataset/for_ControlNet_defect"):
        self.data = []
        self.base_dir = base_dir
        json_path = os.path.join(base_dir, "prompt.json")

        if not os.path.exists(json_path):
            print(
                f"Warning: {json_path} not found. Please run prepare_defect_dataset.py first."
            )
            return

        with open(json_path, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source = cv2.imread(os.path.join(self.base_dir, source_filename))
        target = cv2.imread(os.path.join(self.base_dir, target_filename))

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        if "grayscale" not in prompt.lower():
            prompt = prompt + ", grayscale, monochrome"

        return dict(jpg=target, txt=prompt, hint=source)
