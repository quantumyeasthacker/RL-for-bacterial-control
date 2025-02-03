#!/bin/bash
#SBATCH --job-name=RL_wandb_wAct
#SBATCH --error=RL_wandb_wAct.err
#SBATCH --output=RL_wandb_wAct.out
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=jkratz@andrew.cmu.edu
#SBATCH --nodelist=c05

python3 run_w_wandb.py