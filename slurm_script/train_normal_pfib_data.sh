#!/bin/bash
#SBATCH --job-name=controlnet_train
#SBATCH --output=slurm_output/controlnet_train_%j.out
#SBATCH --error=slurm_output/controlnet_train_%j.err 
#SBATCH --gpus=1
#SBATCH --qos=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liys@a-star.edu.sg

# ==============================================================================
# Usage Instructions
# ==============================================================================
# You can run this script directly or via sbatch, and override default parameters:
# 
# Example:
#   sbatch slurm_script/train_normal_pfib_data.sh \
#       --data_dir /path/to/dataset/ \
#       --batch_size 2 \
#       --learning_rate 1e-4 \
#       --output_dir ./my_custom_output_dir
#
# Available Arguments:
#   --data_dir      Path to the training dataset containing prompt.json (default: /homes/yusha/POC_Dataset/for_ControlNet/)
#   --resume_path   Path to the pretrained ControlNet model to resume from
#   --model_yaml    Path to the model configuration YAML file
#   --batch_size    Batch size for training (default: 1)
#   --logger_freq   Frequency of logging images to TensorBoard/ImageLogger (default: 300)
#   --learning_rate Learning rate (default: 1e-5)
#   --max_steps     Maximum training steps (default: 5000)
#   --output_dir    Directory to save checkpoints, logs, and argument configurations
# ==============================================================================

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate virtual environment
source venv/bin/activate

DATA_DIR="/homes/yusha/POC_Dataset/for_ControlNet/"
RESUME_PATH="./models/control_sd15_canny.pth"
MODEL_YAML="./models/cldm_v15.yaml"
BATCH_SIZE=1
LOGGER_FREQ=300
LEARNING_RATE=1e-5
MAX_STEPS=5000
OUTPUT_DIR="./output/run_${SLURM_JOB_ID:-local}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift ;;
        --resume_path) RESUME_PATH="$2"; shift ;;
        --model_yaml) MODEL_YAML="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --logger_freq) LOGGER_FREQ="$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        --max_steps) MAX_STEPS="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Saving outputs to $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Arguments: DATA_DIR=$DATA_DIR RESUME_PATH=$RESUME_PATH MODEL_YAML=$MODEL_YAML BATCH_SIZE=$BATCH_SIZE LOGGER_FREQ=$LOGGER_FREQ LEARNING_RATE=$LEARNING_RATE MAX_STEPS=$MAX_STEPS OUTPUT_DIR=$OUTPUT_DIR" > "$OUTPUT_DIR/bash_args.txt"

python tutorial_train.py \
    --data_dir "$DATA_DIR" \
    --resume_path "$RESUME_PATH" \
    --model_yaml "$MODEL_YAML" \
    --batch_size "$BATCH_SIZE" \
    --logger_freq "$LOGGER_FREQ" \
    --learning_rate "$LEARNING_RATE" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR"

echo "Job finished at $(date)"