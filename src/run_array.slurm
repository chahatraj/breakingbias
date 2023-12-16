#!/bin/bash

#SBATCH --job-name=evalch
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40GB
#SBATCH --time=2-00:00:00
#SBATCH --array=1-9

#SBATCH --output=/scratch/craj/lets_be_fair/runs/out/cappr.%j.out.txt
#SBATCH --error=/scratch/craj/lets_be_fair/runs/err/cappr.%j.err.txt

#SBATCH --mail-type=END 
#SBATCH --mail-user=craj@gmu.edu

nvidia-smi

# Read arguments from the config file
readarray -t ARGS < gen_args.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

echo $ARGUMENTS

# Execute the script with the arguments
python gen_cappr.py $ARGUMENTS