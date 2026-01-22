#!/bin/bash
#SBATCH --job-name="qot-tuning"
#SBATCH --partition=gpu
#SBATCH --time 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -o 'outfile_tune_test.txt'
#SBATCH -e 'error_tune_test.txt'

set -euo pipefail

# Make output unbuffered for real-time streaming
export PYTHONUNBUFFERED=1

JOB_ID=${SLURM_JOB_ID:-$$}
WORKDIR=${SLURM_SUBMIT_DIR:-$(pwd)}
LOGDIR="$WORKDIR/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/tune_${JOB_ID}.log"

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

echo "Starting tuning — streaming logs to stdout and $LOGFILE"
cd "$WORKDIR"

# Use stdbuf to ensure line-buffered output so `tee` updates in real time
stdbuf -oL -eL python -u separate_training.py \
  --mode joint \
  --data1 datasets/cleaned_lightpath_dataset.csv \
  --target1 datasets/cleaned_lightpath_target.csv \
  --device cuda \
  --tune \
  --tune-space tune_space_full_test.json \
  2>&1 | tee -a "$LOGFILE"

# Deactivate virtualenv if active
deactivate || true

echo "Job $JOB_ID finished. Logs: $LOGFILE"
