#!/bin/bash
#SBATCH --account=def-rjliao
#SBATCH --gres=gpu:a100:2        # Number of GPUs per node (specifying v100l gpu)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8        # CPU cores per MPI process
#SBATCH --mem=64G                 # memory per node
#SBATCH --time=0-18:00            # time (DD-HH:MM)
#SBATCH --output=./slurm_log/%x-%j.out
#SBATCH --mail-user=adsf_zsx@qq.com
#SBATCH --mail-type=ALL

module load python/3.8 cuda/11.4
source ~/zsx/bin/activate
export NCCL_BLOCKING_WAIT=1

cd $SLURM_TMPDIR
unzip -q /home/zyliang/scratch/ShapeNetCore.v2.PC15k.zip

cd /lustre07/scratch/zyliang/3dGeneration
export MAIN_NODE=$(hostname)

srun python train_generation.py --category car --dist_url tcp://$MAIN_NODE:3456 --uncondition False --dataroot "$SLURM_TMPDIR/ShapeNetCore.v2.PC15k/" \
--distribution_type multi

