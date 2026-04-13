#!/bin/bash
#SBATCH --job-name=lora_pfib_train
#SBATCH --output=slurm_output/lora_pfib_train_%j.out
#SBATCH --error=slurm_output/lora_pfib_train_%j.err 
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

# need to set up the data path for the training script 
# need to set up the latest version of the trained model using normal pfib data 
# Run the LoRA training script
python train_lora_normal_pfib.py \
    --mask_dir "/home/lys/projects/POC_Dadaset/20251208/patches/gt" \
    --lora_save_dir "./output_lora" \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --lora_rank 64

echo "Job finished at $(date)"
