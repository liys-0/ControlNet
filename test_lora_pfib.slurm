#!/bin/bash
#SBATCH --job-name=test_lora_pfib      # Job name
#SBATCH --output=logs/test_lora_%j.out # Output log file
#SBATCH --error=logs/test_lora_%j.err  # Error log file
#SBATCH --partition=gpu                # Partition/Queue name
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=8              # Request 8 CPU cores
#SBATCH --mem=32G                      # Request 32GB RAM
#SBATCH --time=04:00:00                # Set a time limit (e.g., 4 hours)

# Create logs directory if it doesn't exist
mkdir -p logs

# Set working directory to your ControlNet folder
cd /home/lys/projects/controlnet || exit 1

# Load modules if required on your cluster (e.g., CUDA)
# module load cuda/11.8

echo "Starting evaluation/generation with LoRA..."
echo "Date: $(date)"
echo "Host: $(hostname)"

# Ensure you have your environment set up properly.
# Replace with your actual path or conda initialization if needed.
source venv/bin/activate

# Optional: Set the huggingface cache or model cache offline if internet is restricted
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Execute the test script
python generate_lora_normal_pfib.py

echo "Finished generation!"
echo "Date: $(date)"
