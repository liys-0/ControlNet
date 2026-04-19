#!/bin/bash
#SBATCH --job-name=lora_test
#SBATCH --output=slurm_output/lora_test_%j.out
#SBATCH --error=slurm_output/lora_test_%j.err 
#SBATCH --gpus=1
#SBATCH --qos=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liys@a-star.edu.sg

# Create slurm log directory if it doesn't exist
mkdir -p slurm

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate virtual environment
source venv/bin/activate

# Run the LoRA training script
# trained on avalon dat

# Inside test_lora_pfib.slurm
for Lora_Weight in 0.1 0.2 0.3; do
    # Test Defect
    python generate_lora_normal_pfib.py \
        --input_image "/homes/yusha/POC_Dataset/for_ControlNet_defect/source/3_patch_r01_c01.png" \
        --output_image "/homes/yusha/POC_Dataset/for_ControlNet_defect/generated/3_patch_r01_c01_defect_${Lora_Weight}.png" \
        --controlnet_path "/homes/yusha/ControlNet/output/lightning_logs/version_6398/checkpoints/epoch=24-step=5000.ckpt" \
        --lora_path "/homes/yusha/ControlNet/output_lora/lora_epoch_99.ckpt" \
        --lora_weight $Lora_Weight \
        --prompt "normal pfib background, trigger_word_defect, extreme heavy film grain, tv static, electron noise, uniform noise, grainy, flat lighting, low contrast, washed out, gray" \
        --n_prompt "smooth, denoised, plastic, fluid, melted, clear, high quality, directional lighting" \
        --scale 3.5 \
        --seed 42


    # Test Normal
    python generate_lora_normal_pfib.py \
        --input_image "/homes/yusha/POC_Dataset/for_ControlNet_defect/source/3_patch_r01_c01.png" \
        --output_image "/homes/yusha/POC_Dataset/for_ControlNet_defect/generated/3_patch_r01_c01_normal_${Lora_Weight}.png" \
        --controlnet_path "/homes/yusha/ControlNet/output/lightning_logs/version_6398/checkpoints/epoch=24-step=5000.ckpt" \
        --lora_path "/homes/yusha/ControlNet/output_lora/lora_epoch_99.ckpt" \
        --prompt "normal pfib background, trigger_word_defect, extreme heavy film grain, tv static, electron noise, uniform noise, grainy, flat lighting, low contrast, washed out, gray" \
        --n_prompt "smooth, denoised, plastic, fluid, melted, clear, high quality, directional lighting, shadows, deep blacks, glowing whites, high contrast, sharp edges" \
        --lora_weight $Lora_Weight \
        --scale 3.5 \
        --seed 42
done