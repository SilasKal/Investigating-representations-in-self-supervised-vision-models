#!/bin/bash
#SBATCH -A eehpc-dev-2026d01-092g
#SBATCH --job-name=MAPS_training_action_ssl_camera_elevation
#SBATCH -t 48:00:00
#SBATCH -p normal-a100-40
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --output=results/%j.out

ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

srun python train.py \
  --mode train \
  --dataset MAPSInstanceSplitDataset \
  --data_root dataset_new_multiple \
  --model resnet18 \
  --pretrained False \
  --modules classic,action,linear_eval \
  --device cuda \
  --num_devices 1 \
  --contrast combined \
  --n_epochs 50 \
  --batch_size 128 \
  --lrate 1e-4 \
  --test_every 5 \
  --save_model True \
  --save_every 10 \
  --maps_pair_mode next \
  --maps_test_instance_index 4 \
  --name action_ssl_resnet18_MAPS_camera_elevation_5 \
  --action_indices "camera_elevation"
