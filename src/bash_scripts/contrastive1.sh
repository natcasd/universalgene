#! /bin/bash
#SBATCH --job-name=contrastive1
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-user=nathan_depiero@brown.edu 
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS     

module load anaconda
source activate universalgene

srun python contrastive/train_contrastive.py \
    --batch_size 64 \
    --n_layers 3 \
    --n_heads 4 \
    --d_model 256 \
    --epochs 10 \
    --cls_token \
    --nworkers 6
