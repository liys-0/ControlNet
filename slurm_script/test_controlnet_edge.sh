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

python test_controlnet_defect_edge.py \
    --model_path "./models/controlnet_defect_edge/lightning_logs/version_0/checkpoints/last.ckpt" \
    --test_dir "/homes/yusha/POC_Dataset/for_ControlNet_defect" \
    --output_dir "./test_results/defect_edge"

echo "Job finished at $(date)"