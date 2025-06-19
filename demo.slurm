#!/bin/bash
#SBATCH --job-name=<your_job_name> # Replace with your actual job name
#SBATCH --account=PGS0218
#SBATCH --gpus=1
#SBATCH --partition=quad
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/demo_slurm.out

# === Activate Conda Envi ===
module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate gpu2 

# === Go to your dir ===
cd Lora_persona # s Replace with your actual script directory
# From below, you can put what
# === Just priting things ===
echo "====== ENVIRONMENT INFO ======"
echo "Hostname : $(hostname)"
echo "Working dir : $(pwd)"
echo "Date : $(date)"
echo "Python : $(which python)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"

# Example: Running a Python script
python3 /users/PGS0218/julina/projects/LoRA_persona/mypersonality/run_lora.py # Replace with your actual script
