#!/bin/bash
#SBATCH --job-name=cn_test_4ch
#SBATCH --output=slurm_output/cn_test_4ch_%j.out
#SBATCH --error=slurm_output/cn_test_4ch_%j.err 
#SBATCH --gpus=1
#SBATCH --qos=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liys@a-star.edu.sg

mkdir -p slurm_output

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

source ../venv/bin/activate || source venv/bin/activate || true
cd ..

CHECKPOINT_PATH="./models/controlnet_defect_4ch/controlnet-epoch00.ckpt"

python test_headless_4ch.py \
  --input_folder "/homes/yusha/POC_Dataset/for_ControlNet_defect/source" \
  --mask_folder "/home/lys/projects/POC_Dadaset/20251208/patches/gt" \
  --output_folder "./test_output_4ch" \
  --checkpoint "$CHECKPOINT_PATH" \
  --prompt "defect pfib" \
  --num_samples 5

echo "Job finished at $(date)"
