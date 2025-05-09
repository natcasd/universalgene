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
    --batch_size 8192 \
    --n_layers 6 \
    --d_model 512 \
    --epochs 5 \
    --nworkers 6 \
    --lr 1e-4 \
    --encoder_type dense