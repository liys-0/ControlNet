#!/bin/bash
#SBATCH --job-name=bash 
#SBATCH --output=slurm_output/controlnet_edge_mask_train_%j.out
#SBATCH --error=slurm_output/controlnet_edge_mask_train_%j.err 
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
#source ../venv/bin/activate || source venv/bin/activate || true

#!/bin/bash

source /homes/yusha/ControlNet/venv/bin/activate

DATASET_DIR="/homes/yusha/POC_Dataset/for_ControlNet_all/for_avalon_train"
MASK_DIR="/homes/yusha/POC_Dataset/20251208/patches/gt"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory $DATASET_DIR does not exist."
    exit 1
fi

if [ ! -d "$MASK_DIR" ]; then
    echo "Error: Mask directory $MASK_DIR does not exist."
    exit 1
fi

PROMPT_FILE="$DATASET_DIR/prompt.json"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: prompt.json not found in $DATASET_DIR"
    exit 1
fi

MISSING_MASKS=0

while IFS= read -r line; do
    if [[ "$line" =~ \"target\":\ *\"([^\"]+)\" ]]; then
        target_filename="${BASH_REMATCH[1]}"
        mask_name=$(basename "$target_filename")
        
        mask_name="${mask_name#defect_}"
        mask_name="${mask_name#normal_}"
        
        mask_path="$MASK_DIR/$mask_name"
        if [ ! -f "$mask_path" ]; then
            echo "Warning: Missing mask for target image '$target_filename' at expected path: $mask_path"
            MISSING_MASKS=1
        fi
    fi
done < "$PROMPT_FILE"

if [ "$MISSING_MASKS" -eq 1 ]; then
    echo "Error: One or more images are missing their corresponding masks. Stopping."
    exit 1
fi

echo "All images have their corresponding masks. Starting training..."

srun python train_controlnet_defect_edge_mask.py \
    --dataset_dir "$DATASET_DIR" \
    --mask_dir "$MASK_DIR" \
    --resume_path "./models/control_sd15_ini.pth" \
    --save_dir "./models/controlnet_defect_edge_mask/run270402_avalon_not_normalized_mask/" \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 100 \
    --gpus 1 \
    --num_workers 4

echo "Job finished at $(date)"