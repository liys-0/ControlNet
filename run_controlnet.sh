#!/bin/bash
#SBATCH --job-name=controlnet       # Job name
#SBATCH --output=logs/cnet_%j.log   # Standard output and error log (%j = job ID)
#SBATCH --partition=gpu             # Partition/queue name (change if your cluster uses a different name like 'gpu', 'a100', etc)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (typically 1 for single-node PyTorch)
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --gres=gpu:1                # Request 1 GPU (change to gpu:a100:1 etc if you need a specific GPU type)
#SBATCH --mem=32G                   # Memory per node
#SBATCH --time=24:00:00             # Time limit hrs:min:sec

echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# ---------------------------------------------------------------------
# ENVIRONMENT ACTIVATION
# Uncomment the section that matches how you installed the environment
# ---------------------------------------------------------------------

# --- OPTION A: If you used PIP (venv) ---
# source venv/bin/activate

# --- OPTION B: If you used CONDA ---
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate control
# ---------------------------------------------------------------------

# Verify GPU is visible to PyTorch
python -c "import torch; print(f'PyTorch GPU available: {torch.cuda.is_available()}')"

# ---------------------------------------------------------------------
# RUN SCRIPT
# Replace 'tutorial_train.py' with whatever script you want to run.
# For example, if you want to run a Gradio app without a web browser 
# opening locally, you might need to run it with share=True in the script.
# ---------------------------------------------------------------------

echo "Starting execution..."
python tutorial_train.py

echo "Job finished on $(date)"
