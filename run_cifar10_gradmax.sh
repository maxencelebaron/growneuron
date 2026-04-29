#!/bin/bash
#SBATCH --job-name=gradmax-cifar10

#SBATCH --partition=tau
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --exclude=margpu007,margpu018,margpu024,margpu025,margpu026,margpu027,margpu029

#SBATCH --output=logs/gradmax_cifar10_%j.out
#SBATCH --error=logs/gradmax_cifar10_%j.err


set -euo pipefail

# Keep TFDS cache on scratch.
export TFDS_DATA_DIR=/scratch/${USER}/tfds
mkdir -p "$TFDS_DATA_DIR"

python growneuron/cifar/main.py \
  --download_data \
  --output_dir="growneuron/cifar/runs" \
  --config=growneuron/cifar/configs/grow_all_at_once.py \
  --config.grow_type=add_gradmax
