#!/bin/bash
#SBATCH -n 8
#SBATCH --mem=192G
#SBATCH -t 12:00:00

module load anaconda
source activate universalgene

python contrast_benchmark.py

