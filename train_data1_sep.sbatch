#!/bin/bash
#SBATCH --job-name="train-data1-sep"
#SBATCH --partition=gpu_HIGH
#SBATCH --time 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-95%4
#SBATCH -o 'outfile_train_data1_sep.txt'
#SBATCH -e 'error_train_data1_sep.txt'

set -euo pipefail

# Make output unbuffered for real-time streaming
export PYTHONUNBUFFERED=1

JOB_ID=${SLURM_JOB_ID:-$$}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
WORKDIR=${SLURM_SUBMIT_DIR:-$(pwd)}
LOGDIR="$WORKDIR/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/train_${JOB_ID}_${TASK_ID}.log"

# Set TMPDIR to a directory in scratch to avoid filling up /tmp on the node
export TMPDIR="$WORKDIR/tmp"
mkdir -p "$TMPDIR"

# Create a shared virtualenv for the job (reused across jobs to save space)
VENV_DIR="$WORKDIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv at $VENV_DIR"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found in PATH. If your cluster uses modules, consider: module load python" >&2
  fi
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"

  echo "Upgrading pip and installing requirements..."
  python -m pip install --upgrade pip wheel setuptools

  # Install torch with CUDA if needed; adjust index-url for your cluster.
  # export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
  # pip install --no-cache-dir torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
  pip install --no-cache-dir torch torchvision torchaudio
  pip install --no-cache-dir pytorch-metric-learning scikit-learn pandas numpy
else
  echo "Using existing virtualenv at $VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi

echo "Starting training — streaming logs to stdout and $LOGFILE"
cd "$WORKDIR"

CONFIG_DIR="$WORKDIR/config_sep1"
CONFIG_FILE="$CONFIG_DIR/run_${TASK_ID}.json"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config not found for array task $TASK_ID: $CONFIG_FILE" >&2
  echo "Expected configs named: $CONFIG_DIR/run_<ARRAY_ID>.json" >&2
  exit 1
fi

RUN_TAG="data1sep_array${TASK_ID}_$(basename "$CONFIG_FILE" .json)"
SAVE_DIR="$WORKDIR/runs_data1sep/array_${TASK_ID}"
mkdir -p "$SAVE_DIR"

# Use stdbuf to ensure line-buffered output so `tee` updates in real time
stdbuf -oL -eL python -u separate_training.py \
  --config "$CONFIG_FILE" \
  --run-name "$RUN_TAG" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "$LOGFILE"

# Deactivate virtualenv if active
deactivate || true

echo "Job $JOB_ID finished. Logs: $LOGFILE"
