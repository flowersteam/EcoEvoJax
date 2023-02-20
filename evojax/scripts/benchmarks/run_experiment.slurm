#!/bin/bash

#SBATCH -A imi@v100
#SBATCH --time=03:00:00
#SBATCH --input=none
#SBATCH --mail-user=gautier.hamon@inria.fr
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH -o run_experiment.slurm.out
#SBATCH -e run_experiment.slurm.err
#SBATCH --array=0-1



python train.py -config configs/OpenES/gridworld_recipes${SLURM_ARRAY_TASK_ID}.yaml




