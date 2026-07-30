"""Constant-antibiotic baseline for the constant-nutrient env: hold the drug ON (b = antibiotic)
every decision step and save each rollout's info log. This is the P_constant reference used by the
relative-performance metric (log P_constant - log P_policy), mirroring the constant baseline in
simulation_test_mutate.py but with no mutation and no mutation-rate folder tokens.

Each rollout's `info` is pickled as trial_<k>tcbk.pkl, the format load_logger_data_new expects.

Usage:
    python sim_baseline_constenv.py out_dir n_trials antibiotic nutrient delay

Positional args:
    out_dir    : str   directory to write trial_<k>tcbk.pkl files into (created if missing)
    n_trials   : int   number of constant-drug rollouts to run
    antibiotic : float drug level held ON every step (b_actions = [0, a]); 3.72 here
    nutrient   : float constant nutrient k_n0; 2.0 here
    delay      : int   observation delay-embed length; 30 here (affects warm-up only)

Output (under out_dir/):
    trial_0tcbk.pkl ... trial_<n_trials-1>tcbk.pkl   (pickled env info dict per rollout)
"""

import os
import sys
import pickle

import numpy as np

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv


MAIN = __name__ == "__main__"

if MAIN:
    out_dir = sys.argv[1]
    n_trials = int(sys.argv[2])
    antibiotic_value = float(sys.argv[3])
    nutrient_value = float(sys.argv[4])
    delay_embed_len = int(sys.argv[5])
    num_decisions = 300

    os.makedirs(out_dir, exist_ok=True)

    cell_config = CellConfig()   # default: no mutation (matches the trained model's env)
    env_config = EnvConfig(
        k_n0_observation=False,
        b_observation=True,
        k_n0_constant=nutrient_value,
        delay_embed_len=delay_embed_len,
        b_actions=[0, antibiotic_value],
        max_pop=np.inf,
    )
    env = ConstantNutrientEnv(env_config, cell_config)

    decisions = [1] * num_decisions   # action index 1 -> b = antibiotic_value (drug always ON)

    for k in range(n_trials):
        env.reset()
        info = None
        for decision in decisions:
            _, _, terminated, truncated, info = env.step(decision)
            if terminated or truncated:
                break
        with open(os.path.join(out_dir, "trial_%dtcbk.pkl" % k), "wb") as f:
            pickle.dump(info, f)

    print("Done: wrote %d baseline trials to %s" % (n_trials, out_dir))
    sys.exit(0)
