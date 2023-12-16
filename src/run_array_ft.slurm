#!/bin/bash

#SBATCH --job-name=ftchbpn
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=100GB
#SBATCH --time=2-00:00:00
#SBATCH --array=1-8

#SBATCH --output=/scratch/craj/lets_be_fair/runs/out/guidance.%j.out.txt
#SBATCH --error=/scratch/craj/lets_be_fair/runs/err/guidance.%j.err.txt

#SBATCH --mail-type=END 
#SBATCH --mail-user=craj@gmu.edu

nvidia-smi

# Read arguments from the config file
readarray -t ARGS < ft_args.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

echo $ARGUMENTS

# Execute the script with the arguments
python ft_all.py $ARGUMENTS