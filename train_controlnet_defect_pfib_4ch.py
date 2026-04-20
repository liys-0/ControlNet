import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_defect_4ch import DefectDataset4Ch
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/homes/yusha/POC_Dataset/for_ControlNet_defect",
    help="Base directory containing prompt.json and images",
)
parser.add_argument(
    "--mask_dir",
    type=str,
    default=None,
    help="Directory containing mask images for masked loss",
)
parser.add_argument(
    "--resume_path",
    type=str,
    default="./models/control_sd15_ini_4ch.ckpt",
    help="Path to initialized 4-channel ControlNet model"
)
parser.add_argument("--model_config", type=str, default="./models/cldm_v15_4ch.yaml")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--logger_freq", type=int, default=300)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument(
    "--sd_locked", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True
)
parser.add_argument(
    "--only_mid_control",
    type=lambda x: str(x).lower() in ["true", "1", "yes"],
    default=False,
)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="./models/controlnet_defect_4ch")
parser.add_argument("--num_workers", type=int, default=4)

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "training_args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

model = create_model(args.model_config).cpu()

if os.path.exists(args.resume_path):
    model.load_state_dict(load_state_dict(args.resume_path, location="cpu"), strict=False)
    print(f"Loaded 4-channel base model from {args.resume_path}")
else:
    print(f"Warning: {args.resume_path} not found. You might need to run tool_add_control_4ch.py first.")

model.learning_rate = args.learning_rate
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control

# For full ControlNet training, we do NOT lock `model.control_model.parameters()`
# ControlLDM handles training the ControlNet based on `sd_locked=True`
# (meaning SD Unet is locked, but ControlNet is optimized)

dataset = DefectDataset4Ch(base_dir=args.dataset_dir, mask_dir=args.mask_dir)
if len(dataset) == 0:
    print("Dataset is empty. Exiting...")
    exit(1)

dataloader = DataLoader(
    dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True
)
logger = ImageLogger(batch_frequency=args.logger_freq)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.save_dir,
    filename="controlnet-epoch{epoch:02d}",
    every_n_epochs=args.save_freq,
    save_top_k=-1, # Save all checkpoints
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=args.gpus,
    num_nodes=args.num_nodes,
    precision=32,
    callbacks=[logger, checkpoint_callback],
    accumulate_grad_batches=args.accumulate_grad_batches,
    max_epochs=args.max_epochs,
    default_root_dir=args.save_dir
)

print("Starting 4-channel ControlNet training from scratch...")
trainer.fit(model, dataloader)
