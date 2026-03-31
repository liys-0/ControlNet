# ControlNet + LoRA for Defect Image Generation

This guide explains how to use the custom LoRA scripts added to this repository to train and generate defect images on normal pfib (plasma focused ion beam) images.

## 1. Preparing the Dataset
You can generate your defect dataset by running the existing preparation script. Ensure that the paths inside the script point correctly to your image patches.
```bash
python prepare_defect_dataset.py
```
This generates the `prompt.json`, `source/` (edges), and `target/` (defective pfib) in your designated dataset directory. The script uses Canny edge detection to extract the conditioning structure from the pfib images.

## 2. Training the LoRA
You have already trained a normal pfib ControlNet (e.g. `control_sd15_normal_pfib.ckpt`). The training script `train_lora_normal_pfib.py` will load this ControlNet, freeze the weights of both the SD Base Model and the ControlNet, inject trainable LoRA layers into the CrossAttention modules, and fine-tune only the LoRA parameters on your defect dataset.

### Steps to Train:
1. Open `train_lora_normal_pfib.py`.
2. Update the `resume_path` variable to point to your previously trained Normal PFIB ControlNet (`.ckpt` file).
3. Run the training script:
```bash
python train_lora_normal_pfib.py
```
The script will save LoRA checkpoints at the end of each epoch in `./models/lora_defect/`.

## 3. Generating Defect Images
Once your LoRA model has been trained, you can use `generate_lora_normal_pfib.py` to generate defect pfib images conditioned on your input "normal pfib" images.

### Example usage:
```bash
python generate_lora_normal_pfib.py \
    --input_image "/path/to/your/normal_pfib.png" \
    --output_image "simulated_defect.png" \
    --controlnet_path "./models/control_sd15_normal_pfib.ckpt" \
    --lora_path "./models/lora_defect/lora_epoch_19.ckpt" \
    --prompt "defect pfib" \
    --num_samples 4
```

### Options:
- `--input_image`: The image you want to extract condition edges from.
- `--raw_control_image`: If you already have pre-computed edges/normals and want to use them directly without computing Canny on `input_image`.
- `--controlnet_path`: The path to your trained ControlNet.
- `--lora_path`: The path to the trained LoRA.
- `--prompt`: Your text prompt.
- `--strength`: How strongly the ControlNet influences the image.
- `--low_threshold` / `--high_threshold`: Canny edge thresholds.

## Code Overview
- **`lora.py`**: A lightweight, standalone LoRA injection module designed for the PyTorch Lightning UNet implementation in this repository. It finds all `CrossAttention` layers and wraps them with `LoRALinearLayer`s.
- **`tutorial_dataset_defect.py`**: A custom `Dataset` class for PyTorch that reads your prepared defect dataset.
- **`train_lora_normal_pfib.py`**: The main training loop utilizing PyTorch Lightning.
- **`generate_lora_normal_pfib.py`**: The inference script.
