#!/bin/bash
#SBATCH --job-name=controlnet_edge_train
#SBATCH --output=slurm_output/controlnet_edge_train_%j.out
#SBATCH --error=slurm_output/controlnet_edge_train_%j.err 
#SBATCH --gpus=1
#SBATCH --qos=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liys@a-star.edu.sg

# Create slurm log directory if it doesn't exist
mkdir -p slurm_output

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate virtual environment
source ../venv/bin/activate || source venv/bin/activate || true
cd ..

python train_controlnet_defect_edge.py \
    --dataset_dir "/homes/yusha/POC_Dataset/for_ControlNet_defect" \
    --save_dir "./models/controlnet_defect_edge" \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 100 \
    --gpus 1 \
    --num_workers 4

echo "Job finished at $(date)"