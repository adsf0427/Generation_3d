#!/bin/bash
#SBATCH --account=def-rjliao
#SBATCH --gres=gpu:a100:1        # Number of GPUs per node (specifying v100l gpu)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # CPU cores per MPI process
#SBATCH --mem=64G                 # memory per node
#SBATCH --time=0-08:00            # time (DD-HH:MM)
#SBATCH --output=./slurm_log/%x-%j.out
#SBATCH --mail-user=adsf_zsx@qq.com
#SBATCH --mail-type=ALL

module load python/3.8 cuda/11.4
source ~/zsx/bin/activate
python train_generation.py --category car --model output/train_generation/2023-01-08-03-47-00/epoch_1199.pth
