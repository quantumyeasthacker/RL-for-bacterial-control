"""Evaluate a trained constant-nutrient (logpop-experiment) MLP checkpoint by rolling out the
greedy policy many times and saving each rollout's info log, for the relative-performance metric.

The env is reconstructed to EXACTLY match run_w_wandb_single_constenv_logpop.py training
(k_n0_observation=False, b_observation=True, obs_type given, k_n0 constant), so the saved
checkpoint's Q-networks load and behave as trained. reward_type is irrelevant here (rewards are
not used) and left at the env default. max_pop=inf so rollouts run the full decision horizon
(mirrors eval_trained_agents_generalized_mutate.py).

Each rollout's `info` is pickled as trial_<k>tcbk.pkl, the format load_logger_data_new expects.

Usage:
    python eval_constenv_logpop_policy.py checkpoint_dir out_dir n_trials \
        antibiotic nutrient delay obs_type [n_jobs]

Positional args:
    checkpoint_dir : str   path to an episode_<n> checkpoint dir (holds q_1,q_2,q_target_1,q_target_2)
    out_dir        : str   directory to write trial_<k>tcbk.pkl files into (created if missing)
    n_trials       : int   number of greedy rollouts to run
    antibiotic     : float drug level for the "on" action (b_actions = [0, a]); 3.72 here
    nutrient       : float constant nutrient k_n0; 2.0 here
    delay          : int   observation delay-embed length; 30 here
    obs_type       : str   "growth_rate" or "log10_pop" (must match training; growth_rate here)
Optional:
    n_jobs         : int   parallel workers (default 10)

Output (under out_dir/):
    trial_0tcbk.pkl ... trial_<n_trials-1>tcbk.pkl   (pickled env info dict per rollout)
"""

import os
import sys
import pickle

import numpy as np
from joblib import Parallel, delayed

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL


MAIN = __name__ == "__main__"

if MAIN:
    checkpoint_dir = sys.argv[1]
    out_dir = sys.argv[2]
    n_trials = int(sys.argv[3])
    antibiotic_value = float(sys.argv[4])
    nutrient_value = float(sys.argv[5])
    delay_embed_len = int(sys.argv[6])
    obs_type = sys.argv[7].lower()
    n_jobs = int(sys.argv[8]) if len(sys.argv) > 8 else 10
    num_decisions = 300

    os.makedirs(out_dir, exist_ok=True)

    # env identical to training (run_w_wandb_single_constenv_logpop.py), except max_pop=inf so
    # trajectories run the full horizon; reward_type left at default (unused during eval).
    cell_config = CellConfig()   # default: no mutation, as in training
    env_config = EnvConfig(
        k_n0_observation=False,     # nutrient constant -> not observed (matches training)
        b_observation=True,         # drug history observed (matches training)
        k_n0_constant=nutrient_value,
        delay_embed_len=delay_embed_len,
        b_actions=[0, antibiotic_value],
        obs_type=obs_type,
        max_pop=np.inf,
    )
    env = ConstantNutrientEnv(env_config, cell_config)

    c = CDQL(env, buffer_size=1_000_000, batch_size=512, use_gpu=False)
    c.load_data(checkpoint_dir, False)   # loads q_1,q_2,q_target_1,q_target_2 from checkpoint_dir

    def run_trial(_):
        # eval_step resets the env and rolls out the deterministic (greedy) policy
        _, _, _, _, info = c.eval_step(num_decisions=num_decisions)
        return info

    infos = Parallel(n_jobs=n_jobs)(delayed(run_trial)(i) for i in range(n_trials))
    for k, info in enumerate(infos):
        with open(os.path.join(out_dir, "trial_%dtcbk.pkl" % k), "wb") as f:
            pickle.dump(info, f)

    print("Done: wrote %d trials to %s" % (n_trials, out_dir))
    sys.exit(0)
