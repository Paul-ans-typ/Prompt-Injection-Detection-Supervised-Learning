#!/bin/bash
#SBATCH --job-name=qlora_injection
#SBATCH --output=logs/qlora_%j.out
#SBATCH --error=logs/qlora_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1          # 1x A100 80GB — handles up to 32B in 4-bit
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00            # 12 hours — increase for 32B model

# -----------------------------------------------------------------------
# Usage:
#   sbatch scripts/slurm_qlora.sh                          # 7B (default)
#   sbatch --export=MODEL=/path/to/Qwen2.5-32B-Instruct \
#          --export=BATCH=4,ACCUM=8 scripts/slurm_qlora.sh # 32B
# -----------------------------------------------------------------------

# Default values (override via --export or editing here)
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-8}
EPOCHS=${EPOCHS:-3}
MAX_LEN=${MAX_LEN:-256}

echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Model:       $MODEL"
echo "Batch size:  $BATCH  x  accum $ACCUM  =  effective $(($BATCH * $ACCUM))"
echo "Epochs:      $EPOCHS"
echo "Started:     $(date)"
echo "============================================"

# Load modules (adjust to Sol's actual module names)
module purge
module load anaconda3
module load cuda/12.1    # or whichever CUDA version Sol provides

# Activate your conda environment
conda activate prompt_env

# Make sure logs directory exists
mkdir -p logs

# Move to project root
cd $SLURM_SUBMIT_DIR

# Run training
python src/train_qlora.py \
    --model        "$MODEL" \
    --epochs       $EPOCHS \
    --batch-size   $BATCH \
    --grad-accum   $ACCUM \
    --max-length   $MAX_LEN \
    --lora-r       16 \
    --lora-alpha   32

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
