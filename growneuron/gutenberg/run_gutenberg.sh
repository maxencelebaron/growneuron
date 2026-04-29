#!/bin/bash

#SBATCH --job-name=gradmax-gutenberg
#SBATCH --partition=tau
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --exclude=margpu007,margpu008,margpu018,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --array=1-3
#SBATCH --output=logs/gradmax_gutenberg_%A_%a.out
#SBATCH --error=logs/gradmax_gutenberg_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python growneuron/gutenberg/main.py \
  --data_dir="${HOME}/data/datasets" \
  --output_dir="growneuron/gutenberg/runs/seed${SLURM_ARRAY_TASK_ID}" \
  --config=growneuron/gutenberg/configs/grow_all_at_once.py \
  --config.grow_type=add_gradmax \
  --config.seed=${SLURM_ARRAY_TASK_ID}
