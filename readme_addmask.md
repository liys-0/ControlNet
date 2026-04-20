# Training a Defect LoRA with a 4-Channel ControlNet Mask Condition

When training a LoRA to generate defect regions on normal background edge maps (like Avalon edge maps or PFIB images), the LoRA often learns the global noise and texture of the dataset, resulting in a messy background.

To fix this, we can pass a mask region directly as a condition to the ControlNet. By expanding the ControlNet input from 3 channels (RGB) to 4 channels (RGB + Mask), the model explicitly learns *where* the defect should be placed relative to the background structure.

The following files and steps have been set up to support this 4-channel conditioning approach.

## 1. Architecture Update: 4-Channel Config
Created `models/cldm_v15_4ch.yaml`.
The only change from the base `cldm_v15.yaml` is updating the ControlNet's input layer to accept 4 channels:
```yaml
    control_stage_config:
      target: cldm.cldm.ControlNet
      params:
        image_size: 32 # unused
        in_channels: 4
        hint_channels: 4 # Changed from 3 to 4
```

## 2. Base Model Initialization
To train with 4 channels, we need a base ControlNet checkpoint that has a 4-channel input convolution layer. We generated a new base model using `tool_add_control_4ch.py`:
```bash
./venv/bin/python tool_add_control_4ch.py models/v1-5-pruned.ckpt models/control_sd15_ini_4ch.ckpt
```
This script adds the 4th channel initialized with zeros so that it doesn't break the pre-trained Stable Diffusion weights.

## 3. Revised Dataset Loader (`tutorial_dataset_defect_4ch.py`)
A new dataset loader was created to construct the 4-channel input. It stacks the 1-channel mask to the end of the 3-channel (RGB) normal image:

```python
# Scale values to [0, 1]
source = source.astype(np.float32) / 255.0
mask = mask.astype(np.float32) / 255.0

# Add channel dimension to mask (H, W) -> (H, W, 1)
mask_expanded = np.expand_dims(mask, axis=-1)

# Concatenate source (RGB) with mask (1 channel) to form a 4-channel image
hint_4ch = np.concatenate([source, mask_expanded], axis=-1)

# The mask_tensor is also kept for the custom loss masking in ddpm.py
mask_tensor = torch.tensor(mask, dtype=torch.float32)

return dict(jpg=target, txt=prompt, hint=hint_4ch, mask=mask_tensor)
```

## 4. Revised Training Script (`train_lora_defect_pfib_4ch.py`)
A new training script was created that wires everything together:
- Uses `tutorial_dataset_defect_4ch.py`
- Points to `--model_config ./models/cldm_v15_4ch.yaml`
- Defaults to `--fallback_path ./models/control_sd15_ini_4ch.ckpt`
- Outputs to a new folder: `--lora_save_dir ./models/lora_defect_4ch`

*Note: The default `resume_path` was removed so it doesn't accidentally try to load an incompatible 3-channel checkpoint and crash on a shape mismatch.*

## How to Start Training

Run the new training script with your virtual environment:

```bash
./venv/bin/python train_lora_defect_pfib_4ch.py
```

### What happens during training:
1. **Input:** The ControlNet receives a 4-channel conditioning image (RGB Avalon background + Mask region).
2. **Generation:** Stable Diffusion generates the output, and ControlNet dictates the background structure while explicitly defining exactly where the defect should be.
3. **Loss Isolation:** The previously implemented custom `# Inject Mask Logic` in `ddpm.py` isolates the training loss strictly to the mask area. This forces the LoRA weights to ONLY learn the "defect texture" and leaves the background clean.