#!/bin/bash
#SBATCH --job-name=gat
#SBATCH --account=IscrC_SISTER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --output=log/slurm_%j.out
#SBATCH --error=log/slurm_%j.err

module load python/3.11.7
module load cuda/12.2
source $WORK/tripteshb/venv/bin/activate

cd $WORK/tripteshb/SL-neural-network-for-temporal-graph

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

!python -u models/gat.py -d wikipedia --gpu 0 --prefix gat --n_epoch 100