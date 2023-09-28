#!/bin/bash

#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu-long
#SBATCH --gpus=6
#SBATCH --gpus-per-task=6
#SBATCH --time=23:00:00
#SBATCH --array=0-30
#SBATCH -o slurm-%j.out


source activate roads_tf
srun -n1 python singleDF_KratzertModel_Implementation.py --set_index=$SLURM_ARRAY_TASK_ID
