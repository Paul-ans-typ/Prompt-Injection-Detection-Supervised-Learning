#!/bin/bash
# =============================================================================
# SLURM job script — QLoRA fine-tuning for Qwen2.5-32B-Instruct on Sol HPC
#
# Usage:
#   sbatch scripts/slurm_qlora_32b.sh
#
# Before submitting:
#   1. Edit the USER CONFIG block below (PROJECT_DIR, NETID, MODEL_PATH)
#   2. Make sure model weights are already on scratch:
#        scp -r ./models/Qwen2.5-32B-Instruct <netid>@sol.asu.edu:/scratch/<netid>/
#   3. Make sure data/processed/ is already on Sol:
#        scp -r ./data/processed <netid>@sol.asu.edu:/scratch/<netid>/project/data/
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM directives
# -----------------------------------------------------------------------------
#SBATCH --job-name=qlora_32b
#SBATCH --output=logs/qlora_32b_%j.out
#SBATCH --error=logs/qlora_32b_%j.err

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Request exactly one A100 80GB — the 40GB variant will OOM on 32B
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb

# 32B 4-bit base (~20 GB) + activations + optimizer + data buffers
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16         # more workers for tokenization + data loading

# 32B at batch 2 + accum 16 on ~500k samples takes ~30-40 hours
# Sol's max wall time is typically 7 days; set conservatively at 48h first
#SBATCH --time=48:00:00

# Email notifications — edit or remove if not wanted
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUREMAIL@asu.edu

# -----------------------------------------------------------------------------
# USER CONFIG — edit these before submitting
# -----------------------------------------------------------------------------
NETID="yournetid"
PROJECT_DIR="/scratch/${NETID}/project"
MODEL_PATH="/scratch/${NETID}/Qwen2.5-32B-Instruct"   # local weights, no internet needed
CONDA_ENV="prompt_env"

# Training hyperparameters
EPOCHS=3
BATCH=2          # per-device; 2 is the safe ceiling for 32B on A100 80GB
ACCUM=16         # effective batch = 2 x 16 = 32
MAX_LEN=256
LORA_R=16
LORA_ALPHA=32

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
echo "============================================================"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : ${SLURM_NODELIST}"
echo "  Model    : ${MODEL_PATH}"
echo "  Batch    : ${BATCH} x accum ${ACCUM} = effective $((BATCH * ACCUM))"
echo "  Epochs   : ${EPOCHS}"
echo "  Started  : $(date)"
echo "============================================================"

module purge
module load anaconda3/2022.10     # adjust version to what Sol provides: module avail anaconda3
module load cuda/12.1             # adjust to what Sol provides:          module avail cuda

# Properly source conda so that 'conda activate' works inside SLURM
# (plain 'conda activate' silently fails without this)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Python  : $(which python)"
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# -----------------------------------------------------------------------------
# Redirect ALL HuggingFace / Torch caches to scratch
# Home directories on Sol are quota-limited (~10 GB) — scratch is not
# -----------------------------------------------------------------------------
SCRATCH_CACHE="/scratch/${NETID}/.cache"
mkdir -p "${SCRATCH_CACHE}"

export HF_HOME="${SCRATCH_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_CACHE}/huggingface/hub"
export HF_DATASETS_CACHE="${SCRATCH_CACHE}/huggingface/datasets"
export TORCH_HOME="${SCRATCH_CACHE}/torch"

# -----------------------------------------------------------------------------
# Force offline mode — Sol has no outbound internet
# Without these, transformers will hang trying to reach huggingface.co
# -----------------------------------------------------------------------------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# -----------------------------------------------------------------------------
# Prevent tokenizer parallelism warning / deadlock with DataLoader workers
# -----------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false

# bitsandbytes needs to find the CUDA libraries — point it at the loaded module
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
# Ensure output directories exist
# -----------------------------------------------------------------------------
cd "${PROJECT_DIR}"
mkdir -p logs
mkdir -p results/qlora
mkdir -p models/qlora

# -----------------------------------------------------------------------------
# Checkpoint resume support
# If the job was killed and resubmitted, resume from the last checkpoint
# rather than starting over.
# -----------------------------------------------------------------------------
CKPT_DIR="${PROJECT_DIR}/models/qlora/$(basename ${MODEL_PATH})/checkpoints"
RESUME_ARG=""

if [ -d "${CKPT_DIR}" ]; then
    # Find the most recent checkpoint directory
    LAST_CKPT=$(ls -td "${CKPT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    if [ -n "${LAST_CKPT}" ]; then
        echo "Resuming from checkpoint: ${LAST_CKPT}"
        # Pass the checkpoint path via an env var that train_qlora.py reads
        export RESUME_FROM_CHECKPOINT="${LAST_CKPT}"
    fi
fi

# -----------------------------------------------------------------------------
# Print GPU memory before launch
# -----------------------------------------------------------------------------
echo ""
echo "GPU memory before training:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
echo ""

# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------
python src/train_qlora.py \
    --model        "${MODEL_PATH}" \
    --epochs       ${EPOCHS} \
    --batch-size   ${BATCH} \
    --grad-accum   ${ACCUM} \
    --max-length   ${MAX_LEN} \
    --lora-r       ${LORA_R} \
    --lora-alpha   ${LORA_ALPHA}

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Post-run summary
# -----------------------------------------------------------------------------
echo ""
echo "GPU memory after training:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

echo ""
echo "============================================================"
echo "  Finished  : $(date)"
echo "  Exit code : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
