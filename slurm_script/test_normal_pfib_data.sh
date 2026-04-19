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

python test_headless.py \
  --input_folder "/homes/yusha/POC_Dataset/for_ControlNet_defect/source_edgemap_overlay_avalon_defect_edge" \
  --output_folder "/homes/yusha/POC_Dataset/for_ControlNet_defect/6488_generated_results_from_edge_map" \
  --checkpoint "/homes/yusha/ControlNet/output/run_6488/lightning_logs/version_6488/checkpoints/epoch=39-step=10000.ckpt" \
  --prompt "normal pfib" \
  --raw_control_image False \
  --num_samples 5

echo "Job finished at $(date)"