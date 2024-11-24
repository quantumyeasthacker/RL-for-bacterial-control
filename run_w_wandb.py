from MLP_full_fluc_drug_cost_par_eval import CDQL
import wandb
# import torch
from joblib import Parallel, delayed
# must be python environment 3.9.1

def train():
    # Initialize W&B run
    with wandb.init() as run:
        config = run.config
        # torch.manual_seed(0)
        c = CDQL(T=config.kn0_T)
        c.train(sweep_var=config.kn0_T)

def par_func(sweep_id, train, count):
    wandb.agent(sweep_id, function=train, count=count)
    wandb.finish()

if __name__ == "__main__":
    wandb.login()

    sweep_config = {
        'method': 'grid', # Can be 'grid', 'bayes', or 'random'
        'parameters': {
            'kn0_T': {'values': [6, 12, 24]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="antibioticRL-periodSweep_wAct")
    # Ensure this is the correct sweep ID by accessing
    # https://wandb.ai/{USERNAME}/{PROJECT_NAME}/sweeps/{SWEEP_ID}

    count = 2
    Parallel(n_jobs=-1)(delayed(par_func)(sweep_id, train, count) for _ in range(count))