import argparse
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/homes/yusha/POC_Dataset/for_ControlNet/",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="./models/control_sd15_canny.pth",
        help="Path to resume model",
    )
    parser.add_argument(
        "--model_yaml",
        type=str,
        default="./models/cldm_v15.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--logger_freq", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--sd_locked", action="store_true", default=True, help="Lock SD weights"
    )
    parser.add_argument(
        "--unlock_sd", action="store_false", dest="sd_locked", help="Unlock SD weights"
    )
    parser.add_argument("--only_mid_control", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./output")
    return parser.parse_args()


args = parse_args()

# Configs
resume_path = args.resume_path
batch_size = args.batch_size
logger_freq = args.logger_freq
learning_rate = args.learning_rate
sd_locked = args.sd_locked
only_mid_control = args.only_mid_control


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(args.model_yaml).cpu()
model.load_state_dict(load_state_dict(resume_path, location="cpu"), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(data_dir=args.data_dir)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=32,
    callbacks=[logger],
    accumulate_grad_batches=4,
    max_steps=args.max_steps,
    default_root_dir=args.output_dir,
)


# Train!
trainer.fit(model, dataloader)
