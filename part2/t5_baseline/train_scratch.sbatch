#!/usr/bin/env bash
# Extra credit: T5-small trained from scratch, as a batch job on Torch.
# Submit from part2/ (create the log folder once first: mkdir -p logs):
#   sbatch --job-name=<name> t5_baseline/train_scratch.sbatch <train_baseline.py arguments>
#SBATCH --account=torch_pr_672_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash t5_baseline/run.sh "$@"
echo "Finished $(date)"
