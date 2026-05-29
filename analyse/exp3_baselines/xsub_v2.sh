#!/bin/bash
#SBATCH --job-name=exp3_bsl
#SBATCH --output=/home/zhaozhimiao/xs/clothes/cloth_new/analyse/exp3_baselines/out_v2.txt
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1
cd /home/zhaozhimiao/xs/clothes/cloth_new/analyse/exp3_baselines
~/anaconda3/envs/HPA/bin/python run_baselines_v2.py
