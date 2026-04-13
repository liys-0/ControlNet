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

#--lora_path "/homes/yusha/ControlNet/lightning_logs/version_5626/checkpoints/epoch=19-step=740.ckpt" \

python generate_lora_normal_pfib.py \
    --input_image "/homes/yusha/POC_Dataset/for_ControlNet_defect/source/3_patch_r01_c02.png" \
    --output_image "/homes/yusha/POC_Dataset/for_ControlNet_defect/generated/3_patch_r01_c02_generated.png" \
    --controlnet_path "/homes/yusha/ControlNet/lightning_logs/version_5616/checkpoints/epoch=23-step=5000.ckpt" \
    --raw_control_image "/homes/yusha/POC_Dataset/for_ControlNet_defect/source/3_patch_r01_c02.png" \
    --num_samples 5

echo "Job finished at $(date)"
