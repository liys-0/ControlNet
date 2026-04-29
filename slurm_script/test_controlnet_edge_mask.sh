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

#ource ../venv/bin/activate || source venv/bin/activate || true
source /homes/yusha/ControlNet/venv/bin/activate

python test_controlnet_defect_edge_mask.py \
    --model_path "./models/controlnet_defect_edge_mask/run270403_avalon_not_normalized_mask/controlnet-epochepoch=75-v1.ckpt" \
    --test_dir "/homes/yusha/POC_Dataset/for_ControlNet_all/test_4ch_edge_map" \
    --mask_dir "/homes/yusha/POC_Dataset/for_ControlNet_all/test_4ch_masks" \
    --output_dir "/homes/yusha/POC_Dataset/for_ControlNet_all/gen_4ch_avalon_defect_image_epoc75" \
    --num_samples 10



echo "Job finished at $(date)"