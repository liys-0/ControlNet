import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_defect_edge import DefectDatasetEdge
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
torch.set_float32_matmul_precision('high')
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
    default="./models/control_v11p_sd15_canny.pth",
    help="Path to pretrained 3-channel edge ControlNet (e.g. Canny or HED)"
)
parser.add_argument("--model_config", type=str, default="./models/cldm_v15.yaml")
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
parser.add_argument("--save_dir", type=str, default="./models/controlnet_defect_edge")
parser.add_argument("--num_workers", type=int, default=4)
# Added 2026-04-30: full Lightning resume after disk-full crash at epoch 83.
# --ckpt_path restores optimizer state, LR scheduler, epoch counter, and global_step
# so training continues exactly where it stopped (not a fresh fine-tune).
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="Lightning checkpoint to resume from (restores optimizer/epoch/global_step). "
         "When set, --resume_path pretrained weights are overwritten by this checkpoint."
)
# Cap on-disk checkpoints. Original code hard-coded save_top_k=-1 which kept every
# epoch and filled the 1.8 TB disk (188 GB of ckpts) — that's what caused the crash.
parser.add_argument(
    "--save_top_k",
    type=int,
    default=3,
    help="How many epoch checkpoints to keep on disk. -1 keeps all (will fill disk)."
)

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "training_args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

model = create_model(args.model_config).cpu()

if os.path.exists(args.resume_path):
    pretrained_weights = load_state_dict(args.resume_path, location="cpu")
    
    target_key = "control_model.input_hint_block.0.weight"
    if target_key in pretrained_weights:
        old_weight = pretrained_weights[target_key]
        if old_weight.shape[1] == 3:
            new_weight = torch.zeros((16, 4, 3, 3), dtype=old_weight.dtype)
            new_weight[:, :3, :, :] = old_weight
            pretrained_weights[target_key] = new_weight
            print(f"Modified '{target_key}' from {old_weight.shape} to {new_weight.shape} (4th channel initialized to zero).")

    model.load_state_dict(pretrained_weights, strict=False)
    print(f"Loaded pretrained model from {args.resume_path}")
else:
    print(f"Warning: {args.resume_path} not found.")
    print("Please download a pretrained ControlNet (like Canny) and place it there to avoid random initialization collapse.")

model.learning_rate = args.learning_rate
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control

# For full ControlNet training, we do NOT lock `model.control_model.parameters()`
# ControlLDM handles training the ControlNet based on `sd_locked=True`
# (meaning SD Unet is locked, but ControlNet is optimized)

dataset = DefectDatasetEdge(base_dir=args.dataset_dir, mask_dir=args.mask_dir)
if len(dataset) == 0:
    print("Dataset is empty. Exiting...")
    exit(1)

dataloader = DataLoader(
    dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True
)
logger = ImageLogger(batch_frequency=args.logger_freq)

# save_top_k>0 requires a `monitor` metric in Lightning, otherwise it raises
# MisconfigurationException. We track train/loss_epoch (already logged by ControlLDM)
# and keep the K best-loss ckpts. save_last=True separately maintains a `last.ckpt`
# pointing at the most recent epoch — that's the file to use for resume after a crash.
# auto_insert_metric_name=False stops Lightning from prefixing "train/loss_epoch=" into
# the filename (the slash breaks paths); we embed the value manually instead.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.save_dir,
    filename="controlnet-epoch{epoch:02d}-{train/loss_epoch:.4f}",
    every_n_epochs=args.save_freq,
    save_top_k=args.save_top_k,
    save_last=True,
    monitor="train/loss_epoch",
    mode="min",
    auto_insert_metric_name=False,
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

print("Starting 3-channel Edge ControlNet training with soft mask loss...")
# ckpt_path=None → fresh run; ckpt_path=<file> → resume optimizer/epoch/step state.
trainer.fit(model, dataloader, ckpt_path=args.ckpt_path)