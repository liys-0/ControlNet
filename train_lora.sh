#!/bin/bash
#SBATCH --job-name=lora_pfib_train
#SBATCH --output=slurm/lora_pfib_train_%j.out
#SBATCH --error=slurm/lora_pfib_train_%j.err 
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
python train_lora_normal_pfib.py

echo "Job finished at $(date)"
