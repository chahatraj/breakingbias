#!/bin/bash

#SBATCH --job-name=evalyi3
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
# #SBATCH --nodelist=gpu006

#SBATCH --mem=100GB
#SBATCH --time=3-00:00:00

#SBATCH --mail-type=END 
#SBATCH --mail-user=craj@gmu.edu

#SBATCH --output=/scratch/craj/lets_be_fair/runs/out/yi.%j.out.txt
#SBATCH --error=/scratch/craj/lets_be_fair/runs/err/yi.%j.err.txt

nvidia-smi 
# source activate ~/pytorch-env/bin/activate
python gen_cappr.py --model_name yi --llama_size 34 --use_peft 0 --case 0 --prompt_type 3
