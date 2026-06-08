#!/bin/bash
#SBATCH -A eehpc-dev-2026d01-092g
#SBATCH --job-name=MAPS_generation
#SBATCH -t 48:00:00
#SBATCH -p normal-a100-40
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --output=results/%j.out

ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

#blender --version
#which blender

srun python create_dataset_function_one_param_2.py