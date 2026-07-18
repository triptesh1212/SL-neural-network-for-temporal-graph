#!/bin/bash
#SBATCH --job-name=sl_tgat
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

python3 models/sl_rnn.py -d wikipedia --zeta_real 0.04 --zeta_imag 0.5 --nu_real 1.0 --nu_imag 0.0 --sl_dt 1.0 --n_layer 2 --n_epoch 20