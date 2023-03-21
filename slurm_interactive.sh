salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --cpus-per-task=32 --mem=32G --mail-user=adsf_zsx@qq.com --mail-type=ALL
salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --gres=gpu:a100:4 --cpus-per-task=24 --mem=64G --mail-user=adsf_zsx@qq.com --mail-type=ALL
module load python/3.8 cuda/11.4
source ~/zsx/bin/activate 

cd $SLURM_TMPDIR
unzip -q /home/zyliang/scratch/ShapeNetCore.v2.PC15k.zip


cd /lustre07/scratch/zyliang/3dGeneration

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_generation.py --category all --uncondition False --dataroot "$SLURM_TMPDIR/ShapeNetCore.v2.PC15k/" \
--distribution_type multi


OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 \
 train_generation_simple.py --category all --uncondition False --bs 4 --dataroot "../ShapeNetCore.v2.PC15k/" \
--parameterization "x0" \
--distribution_type multi --clip_ckpt "/home/internc/CLIP_PC/lightning_logs/version_2/checkpoints/epoch=50-step=113781.ckpt"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_generation_simple.py --category all --uncondition False --bs 4 --dataroot "../ShapeNetCore.v2.PC15k/" \
--parameterization "x0" \
--clip_ckpt "/home/internc/CLIP_PC/lightning_logs/version_2/checkpoints/epoch=50-step=113781.ckpt" \
--distribution_type single


torchrun --nproc_per_node 8 
OMP_NUM_THREADS=16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train_generation_simple.py --category all --no-uncondition --bs 64 --use_transformer --dataroot "../ShapeNetCore.v2.PC15k/" \
--parameterization "eps" \
--model "output/train_generation_simple/2023-02-27-22-27-51/epoch_240.pth" \
--distribution_type multi

CUDA_VISIBLE_DEVICES=7 \
python test.py --no-uncondition --use_transformer \
--parameterization "eps" --manualSeed 360 \
--model "output/train_generation_simple/2023-03-06-14-37-18/epoch_500.pth"

export NCCL_BLOCKING_WAIT=1

