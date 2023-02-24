salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --cpus-per-task=32 --mem=32G --mail-user=adsf_zsx@qq.com --mail-type=ALL
salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --gres=gpu:a100:1 --cpus-per-task=24 --mem=64G --mail-user=adsf_zsx@qq.com --mail-type=ALL
module load python/3.8 cuda/11.4
source ~/zsx/bin/activate 

cd $SLURM_TMPDIR
unzip -q /home/zyliang/scratch/ShapeNetCore.v2.PC15k.zip


cd /lustre07/scratch/zyliang/3dGeneration

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_generation.py --category all --uncondition False --dataroot "$SLURM_TMPDIR/ShapeNetCore.v2.PC15k/" \
--distribution_type multi

CUDA_VISIBLE_DEVICES=4 python train_generation_simple.py --category all --uncondition False --dataroot "../ShapeNetCore.v2.PC15k/" \
--distribution_type multi --model output/train_generation_simple/2023-02-21-15-24-28/epoch_20.pth


export NCCL_BLOCKING_WAIT=1

