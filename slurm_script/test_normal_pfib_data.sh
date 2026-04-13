#!/bin/bash
#SBATCH --job-name=cn_test
#SBATCH --output=slurm_output/cn_test_%j.out
#SBATCH --error=slurm_output/cn_test_%j.err 
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

#python test_my_pfib_model.py
##--input_folder "/homes/yusha/POC_Dataset/for_ControlNet/source" \
##--input_folder "/homes/yusha/POC_Dataset/20251208/patches/avalon" \
python test_headless.py \
  --input_folder "/homes/yusha/POC_Dataset/for_ControlNet_defect/source" \
  --output_folder "/homes/yusha/POC_Dataset/for_ControlNet/5626_generated_results_from_edge_map" \
  --checkpoint "./lightning_logs/version_5626/checkpoints/epoch=19-step=740.ckpt" \
  --prompt "normal pfib" \
  --raw_control_image False \
  --num_samples 5

echo "Job finished at $(date)"