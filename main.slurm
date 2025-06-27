#!/bin/bash
#SBATCH --job-name=dynamic_job # Generic name, will be refined by the log file
#SBATCH --account=PGS0218
#SBATCH --gpus=1
#SBATCH --partition=quad
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#
# NOTE: The #SBATCH --output directive is removed.
# We will handle logging dynamically inside the script.
#

# --- 1. SCRIPT ARGUMENT VALIDATION ---
if [[ -z "$1" ]]; then
    echo "Error: Missing script choice." >&2
    echo "Usage: sbatch $0 <script_number>" >&2
    echo "Example: sbatch $0 1" >&2
    exit 1
fi

SCRIPT_CHOICE=$1

# --- 2. DYNAMIC LOG FILE SETUP ---
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/script_${SCRIPT_CHOICE}-job_${SLURM_JOB_ID}.out"
mkdir -p "$LOG_DIR"
exec > "$LOG_FILE" 2>&1

# --- 3. SELECT PYTHON SCRIPT TO RUN ---
PYTHON_SCRIPT=""
case $SCRIPT_CHOICE in
    # 1)
    #     echo "Selected script #1: run_lora.py"
    #     PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/1_run_finetune_lora_MTL.py" ;;
    # 
    2)
        echo "Selected script #2: Another script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/2_run_finetune_roberta.py" ;;
    3)
        echo "Selected script #3: A third script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/3_run_zero_shot.py" ;;
    4)
        echo "Selected script #4: A fourth script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/4_run_ml.py" ;;
    # 5)
    #     echo "Selected script #5: A fifth script"
    #     PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/1.1_run_finetune_BiLSTM_lora.py" ;;
    6)
        echo "Selected script #6: A sixth script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/2.1_run_finetune_full_BiLSTM.py" ;;
    7)
        echo "Selected script #7: A seventh script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/2.2_run_finetune_full_MLP.py" ;;
    8)
        echo "Selected script #8: A eight script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/1.3_run_finetune_lora_MTL_GCN.py" ;;
    9)
        echo "Selected script #9: A ninth script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/1_run_finetune_lora_MTL.py" ;;
    10)
        echo "Selected script #10: A Tenth script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/1.2_run_finetune_lora_MTL_MLP.py" ;;
    11)
        echo "Selected script #10: A 11 script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/1.6_hyperparameter_tuning_lora_MTL_MLP.py" ;;
    12)
        echo "Selected script #10: A 12 script"
        PYTHON_SCRIPT="/users/PGS0218/julina/projects/LoRA_persona/mypersonality/5_run_prompt_tuning.py" ;;
 
    *)
        echo "Error: Invalid script choice '${SCRIPT_CHOICE}'."
        echo "Please choose a number defined in the script's case statement."
        exit 1
        ;;
esac


# === Activate Conda Envi ===
module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate gpu2 

# === Go to your dir ===
cd /users/PGS0218/julina/projects/LoRA_persona 

# === Just priting things ===
echo "====== JOB INFO ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Initial Job Name: $SLURM_JOB_NAME"
echo "Chosen Script Number: $SCRIPT_CHOICE"
echo "Executing Python Script: $PYTHON_SCRIPT"
echo "Logging to file: $(pwd)/$LOG_FILE"
echo "------------------------------"
echo "====== ENVIRONMENT INFO ======"
echo "Hostname : $(hostname)"
echo "Working dir : $(pwd)"
echo "Date : $(date)"
echo "Python : $(which python)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"

# --- FIX: CONTROL LIBRARY THREADING ---
# Prevent libraries (NumPy, Scikit-learn) from conflicting with SLURM's CPU allocation.
# This is a very common cause of jobs freezing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "====== THREADING CONTROL ======"
echo "OMP_NUM_THREADS is set to $OMP_NUM_THREADS"

# --- 4. EXECUTE THE SELECTED SCRIPT ---
echo "====== SCRIPT EXECUTION START ======"
python3 -u "$PYTHON_SCRIPT"  # --- FIX: Use python -u for unbuffered output --- This ensures print() statements appear in the log file immediately.
echo "====== SCRIPT EXECUTION FINISHED ======"