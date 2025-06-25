#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --time 96:00:00
#SBATCH --cpus-per-task 8
#SBATCH --mem 256000
#SBATCH --job-name TUH_prep
#SBATCH --output /gpfs/space/projects/BetterMedicine/illia/DL_Registration/outputs/tuh_prep_%j.out
#SBATCH -A revvity
#SBATCH --exclude=falcon3

source activate CTREG

python /gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/common/ocs/lightning_pipeline/train.py