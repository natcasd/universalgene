#!/bin/bash
#SBATCH --job-name=contrastive3            # appears in `squeue`
#SBATCH --partition=gpu      # change to your GPU/CPU partition
#SBATCH --exclude=gpu[2607-2609]
#SBATCH --gres=gpu:2                     # number / type of GPUs per run
#SBATCH --cpus-per-task=12                # adjust to match DataLoader workers
#SBATCH --mem=192G                        # or whatever your data need
#SBATCH --time=24:00:00                  # wall‑clock limit
#SBATCH --output=logs/%x_%A_%a.out       # one log per seed

# module load anaconda
# source activate universalgene
source ../pytorch.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python contrastive/train_contrastive.py \
    --batch_size 64 \
    --n_layers 6 \
    --n_heads 4 \
    --d_model 256 \
    --epochs 10 \
    --cls_token \
    --nworkers 6 \
    --lr 1e-3
