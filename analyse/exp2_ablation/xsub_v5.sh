#!/bin/bash
#SBATCH --job-name=exp2_abl
#SBATCH --output=/home/zhaozhimiao/xs/clothes/cloth_new/analyse/exp2_ablation/out_v5.txt
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1
cd /home/zhaozhimiao/xs/clothes/cloth_new/analyse/exp2_ablation
~/anaconda3/envs/HPA/bin/python run_ablation_v5.py
