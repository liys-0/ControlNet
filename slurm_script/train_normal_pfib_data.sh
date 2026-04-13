#!/bin/bash
#SBATCH --job-name=controlnet_train
#SBATCH --output=slurm_output/controlnet_train_%j.out
#SBATCH --error=slurm_output/controlnet_train_%j.err 
#SBATCH --gpus=1
#SBATCH --qos=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liys@a-star.edu.sg

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate virtual environment
source venv/bin/activate

python tutorial_train.py

echo "Job finished at $(date)"