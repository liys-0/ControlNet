from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_defect import DefectDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
import os
from lora import inject_trainable_lora, extract_lora_up_down

resume_path = "./models/control_sd15_normal_pfib.ckpt"
batch_size = 1
logger_freq = 300
learning_rate = 1e-4
sd_locked = True
only_mid_control = False

model = create_model("./models/cldm_v15.yaml").cpu()

if os.path.exists(resume_path):
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    print(f"Loaded normal pfib ControlNet from {resume_path}")
else:
    print(f"Warning: {resume_path} not found. Loading default fallback if possible...")
    fallback_path = "./models/control_sd15_ini.ckpt"
    if os.path.exists(fallback_path):
        model.load_state_dict(load_state_dict(fallback_path, location="cpu"))

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

for param in model.control_model.parameters():
    param.requires_grad = False

print("Injecting LoRA into UNet...")
lora_params = inject_trainable_lora(model.model.diffusion_model, rank=32)


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


dataset = DefectDataset()
if len(dataset) == 0:
    print("Dataset is empty. Exiting...")
    exit(1)

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
lora_saver = SaveLoRACallback("./models/lora_defect", save_freq=1)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=32,
    callbacks=[logger, lora_saver],
    accumulate_grad_batches=4,
    max_epochs=20,
)

print("Starting LoRA training...")
trainer.fit(model, dataloader)
