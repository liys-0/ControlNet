from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_defect import DefectDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
import os
from lora import inject_trainable_lora, extract_lora_up_down
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
    default="./lightning_logs/version_5616/checkpoints/epoch=24-step=5000.ckpt",
)
parser.add_argument(
    "--fallback_path", type=str, default="./models/control_sd15_ini.ckpt"
)
parser.add_argument("--model_config", type=str, default="./models/cldm_v15.yaml")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--logger_freq", type=int, default=300)
parser.add_argument("--learning_rate", type=float, default=1e-4)
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
parser.add_argument("--lora_rank", type=int, default=32)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--lora_save_dir", type=str, default="./models/lora_defect")
parser.add_argument("--num_workers", type=int, default=0)

args = parser.parse_args()

os.makedirs(args.lora_save_dir, exist_ok=True)
with open(os.path.join(args.lora_save_dir, "training_args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

resume_path = args.resume_path
batch_size = args.batch_size
logger_freq = args.logger_freq
learning_rate = args.learning_rate
sd_locked = args.sd_locked
only_mid_control = args.only_mid_control

model = create_model(args.model_config).cpu()

if os.path.exists(resume_path):
    # model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.load_state_dict(load_state_dict(resume_path, location="cpu"), strict=False)
    print(f"Loaded normal pfib ControlNet from {resume_path}")
else:
    print(f"Warning: {resume_path} not found. Loading default fallback if possible...")
    fallback_path = args.fallback_path
    if os.path.exists(fallback_path):
        # model.load_state_dict(load_state_dict(fallback_path, location="cpu"))
        model.load_state_dict(
            load_state_dict(fallback_path, location="cpu"), strict=False
        )

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

for param in model.control_model.parameters():
    param.requires_grad = False

print("Injecting LoRA into UNet...")
lora_params = inject_trainable_lora(model.model.diffusion_model, rank=args.lora_rank)


def configure_lora_optimizers():
    opt = torch.optim.AdamW(lora_params, lr=model.learning_rate)
    return opt


model.configure_optimizers = configure_lora_optimizers


class SaveLoRACallback(pl.Callback):
    def __init__(self, save_path, save_freq=1):
        self.save_path = save_path
        self.save_freq = save_freq
        os.makedirs(self.save_path, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_freq == 0:
            lora_dict = extract_lora_up_down(pl_module.model.diffusion_model)
            save_file = os.path.join(
                self.save_path, f"lora_epoch_{trainer.current_epoch}.ckpt"
            )
            torch.save(lora_dict, save_file)
            print(f"Saved LoRA weights to {save_file}")


dataset = DefectDataset(base_dir=args.dataset_dir, mask_dir=args.mask_dir)
if len(dataset) == 0:
    print("Dataset is empty. Exiting...")
    exit(1)

dataloader = DataLoader(
    dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True
)
logger = ImageLogger(batch_frequency=logger_freq)
lora_saver = SaveLoRACallback(args.lora_save_dir, save_freq=args.save_freq)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=args.gpus,
    num_nodes=args.num_nodes,
    precision=32,
    callbacks=[logger, lora_saver],
    accumulate_grad_batches=args.accumulate_grad_batches,
    max_epochs=args.max_epochs,
)

print("Starting LoRA training...")
trainer.fit(model, dataloader)
