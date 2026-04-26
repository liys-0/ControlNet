#!/bin/bash
#SBATCH --job-name=controlnet_edge_test
#SBATCH --output=slurm_output/controlnet_edge_test_%j.out
#SBATCH --error=slurm_output/controlnet_edge_test_%j.err 
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

python test_controlnet_defect_edge_mask.py \
    --model_path "./models/controlnet_defect_edge_mask/lightning_logs/version_0/checkpoints/last.ckpt" \
    --resume_path "./models/control_sd15_canny.pth" \
    --test_dir "/homes/yusha/POC_Dataset/for_ControlNet_defect" \
    --mask_dir "/home/lys/projects/POC_Dadaset/20251208/patches/gt" \
    --output_dir "./test_results/defect_edge_mask"

echo "Job finished at $(date)"