#!/bin/bash
#SBATCH --job-name=supervised_MAPS_imagenet_mix_50_50_ImageNet
#SBATCH --time=24:00:00
#SBATCH --partition=gpu2
#SBATCH --output=results/%j.out
#SBATCH --no-requeue
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate MT

nvidia-smi


# 50/50 (MAPS/ImageNet) => 50% MAPS, 50% ImageNet
srun python train_supervised_MAPS_imagenet_mix.py \
  --maps_root /work/dldevel/kalinowski/MAPS_training/dataset_new_multiple \
  --imagenet_train_root /work/dldevel/kalinowski/datasets/train \
  --imagenet_index_cache /work/dldevel/kalinowski/datasets/cache \
  --imagenet_wnids n02676566,n02504458,n02690373,n02701002,n07753592,n01443537,n03452741,n03761084,n03782006,n04266014,n07745940,n04507155 \
  --imagenet_val_root /work/dldevel/kalinowski/datasets/test \
  --maps_train_ratio 0.50 \
  --model resnet18 \
  --epochs 50 \
  --device cuda \
  --batch_size 128 \
  --num_workers 0 \
  --seed 5 \
  --no_pretrained \
  --lr 0.1 \
  --momentum 0.9 \
  --wd 1e-4 \
  --cosine \
  --amp \
  --save_path sup_maps_resnet18_50_img50_seed_5.pt

