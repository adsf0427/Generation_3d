salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --cpus-per-task=32 --mem=32G --mail-user=adsf_zsx@qq.com --mail-type=ALL
salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --gres=gpu:a100:1 --cpus-per-task=24 --mem=64G --mail-user=adsf_zsx@qq.com --mail-type=ALL
module load python/3.8 cuda/11.4
module load cuda/11.4
source ~/zsx/bin/activate 
python train_generation.py --category chair --model output/pretrained/chair_1799.pth

export NCCL_BLOCKING_WAIT=1

