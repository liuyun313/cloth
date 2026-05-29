#!/bin/bash
#SBATCH --job-name=exp1_grid
#SBATCH --output=/home/zhaozhimiao/xs/clothes/cloth_new/analyse/exp1_sensitivity/out_grid.txt
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1
cd /home/zhaozhimiao/xs/clothes/cloth_new/analyse/exp1_sensitivity
~/anaconda3/envs/HPA/bin/python run_grid.py
