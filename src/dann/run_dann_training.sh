#!/bin/bash
#SBATCH --job-name=dann_training            # appears in `squeue`
#SBATCH --partition=3090-gcondo      # change to your GPU/CPU partition
#SBATCH --exclude=gpu[2607-2609]
#SBATCH --gres=gpu:4                     # number / type of GPUs per run
#SBATCH --cpus-per-task=8                # adjust to match DataLoader workers
#SBATCH --mem=192G                        # or whatever your data need
#SBATCH --time=24:00:00                  # wall‑clock limit
#SBATCH --output=logs/%x_%A_%a.out       # one log per seed

#SBATCH --mail-user=winston_y_li@brown.edu   # <‑‑ change me
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS     # BEGIN|END|FAIL|ALL|TIME_LIMIT, etc.

# ---------- 1.  Environment ----------

# module purge
# module load cuda/12.2                    # or your site’s CUDA module
source ../../pytorch.venv/bin/activate

# ---------- 2.  Pick the seed ----------
# SEED=${SLURM_ARRAY_TASK_ID}              

# ---------- 3.  Run your code ----------
srun python dann_training.py -m dann -bs 128
# srun python dann_training.py -m transformer_dann -bs 128
# srun python dann_training.py -m cdan -bs 128