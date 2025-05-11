#! /bin/bash
#SBATCH --job-name=contrastive6
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --mail-user=nathan_depiero@brown.edu 
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS     

module load anaconda
source activate universalgene

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python contrastive/train_contrastive.py \
    --batch_size 264 \
    --n_layers 8 \
    --n_heads 2 \
    --d_model 512 \
    --epochs 10 \
    --nworkers 6 \
    --lr 1e-5 \
    --projection \
    --temperature 0.07 \
    --multiply_by_expr \
    --train_path data/tabula_muris/preprocessed_reduced/tm_adata_train.pkl \
    --val_path data/tabula_muris/preprocessed_reduced/tm_adata_test.pkl \