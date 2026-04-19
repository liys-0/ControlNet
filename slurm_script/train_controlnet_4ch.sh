#!/bin/bash
#SBATCH --job-name=controlnet_pfib_train_4ch
#SBATCH --output=slurm_output/controlnet_pfib_train_4ch_%j.out
#SBATCH --error=slurm_output/controlnet_pfib_train_4ch_%j.err 
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

# Ensure 4-channel base model exists
if [ ! -f "models/control_sd15_ini_4ch.ckpt" ]; then
    echo "Base model not found. Initializing 4-channel model from SD 1.5..."
    python tool_add_control_4ch.py models/v1-5-pruned.ckpt models/control_sd15_ini_4ch.ckpt
fi

# Run the 4-channel full ControlNet training script
python train_controlnet_defect_pfib_4ch.py \
    --dataset_dir "/homes/yusha/POC_Dataset/for_ControlNet_all" \
    --mask_dir "/home/lys/projects/POC_Dadaset/20251208/patches/gt" \
    --save_dir "./models/controlnet_defect_4ch" \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 100

echo "Job finished at $(date)"
