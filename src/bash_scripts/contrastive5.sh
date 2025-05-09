#! /bin/bash
#SBATCH --job-name=contrastive2
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
    --batch_size 256 \
    --n_layers 6 \
    --n_heads 4 \
    --d_model 512 \
    --epochs 10 \
    --cls_token \
    --nworkers 6 \
    --lr 1e-4 \
    --train_path data/tabula_muris/preprocessed_reduced/tm_adata_train.pkl \
    --val_path data/tabula_muris/preprocessed_reduced/tm_adata_test.pkl \