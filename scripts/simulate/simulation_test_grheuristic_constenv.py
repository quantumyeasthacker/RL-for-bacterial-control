"""Growth-rate-threshold heuristic controller on the CONSTANT-nutrient env (constant env n=2).

A hysteresis controller on the population growth rate g -- the SAME signal the RL agent receives
as its population observation, computed per decision step as
    g = (log N_t - log N_{t-1}) / delta_t
from consecutive population counts in the sim log (log entry layout:
[t, k_n0, b, num_cells, U_ave, phi_R_ave, phi_S_ave]; N is index 3). The rule (band low < high):
    - antibiotic is APPLIED at the start of control (first decision = ON)
    - it is REMOVED once g falls below G_LOW   (population declining enough)
    - it is APPLIED again once g rises above G_HIGH (population growing too fast)
    - this on/off cycle repeats until the end of the simulation.
This is the growth-rate analog of simulation_test_phiSheuristic_constenv.py.

Each rollout's `info` is pickled as trial_<k>tcbk.pkl (the format load_logger_data_new expects).
This is the "policy" side of the relative-performance metric
(rel. perf. = log P_constant - log P_heuristic); the constant-drug baseline (P_constant) is the
same one produced by sim_baseline_constenv.py. The env config mirrors that baseline exactly
(k_n0_observation=False, b_observation=True, max_pop=inf).

Usage:
    python simulation_test_grheuristic_constenv.py out_dir n_trials antibiotic nutrient delay \
        [start_index] [g_high] [g_low]

Positional args:
    out_dir     : str   directory to write trial_<k>tcbk.pkl files into (created if missing)
    n_trials    : int   number of heuristic rollouts to run in this call
    antibiotic  : float drug level applied when ON (b_actions = [0, a]); 3.72 here
    nutrient    : float constant nutrient k_n0; 2.0 here (the "n=2" condition)
    delay       : int   observation delay-embed length; 30 here (affects warm-up only)
    start_index : int   (optional, default 0) trial-file numbering offset for Slurm-array blocks:
                        task t passes start_index = t*n_trials.
    g_high      : float (optional, default G_HIGH=0.5) re-apply antibiotic once g rises above this.
    g_low       : float (optional, default G_LOW=-0.5) remove antibiotic once g falls below this.

Output (under out_dir/):
    trial_<start_index>tcbk.pkl ... trial_<start_index + n_trials - 1>tcbk.pkl
    (pickled env info dict per rollout)
"""

import os
import sys
import pickle

import numpy as np

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv


MAIN = __name__ == "__main__"

# default growth-rate hysteresis thresholds (band low < high)
G_HIGH = 0.5    # re-apply antibiotic once g rises above this
G_LOW = -0.5    # remove antibiotic once g falls below this


def choose_next_action(current_action, g, g_high, g_low):
    """Hysteresis rule on the growth rate g.

    current_action: 1 = antibiotic ON, 0 = OFF (the action just applied).
    Returns the action to apply on the next decision step.
    """
    if current_action == 1 and g < g_low:
        return 0  # population declining enough -> stop dosing
    if current_action == 0 and g > g_high:
        return 1  # population growing too fast -> resume dosing
    return current_action


if MAIN:
    out_dir = sys.argv[1]
    n_trials = int(sys.argv[2])
    antibiotic_value = float(sys.argv[3])
    nutrient_value = float(sys.argv[4])
    delay_embed_len = int(sys.argv[5])
    start_index = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    g_high = float(sys.argv[7]) if len(sys.argv) > 7 else G_HIGH
    g_low = float(sys.argv[8]) if len(sys.argv) > 8 else G_LOW
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

    for k in range(n_trials):
        env.reset()
        action = 1  # antibiotic is always applied at the start of control
        info = None
        for _ in range(num_decisions):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            # per-step growth rate from the last two population counts (== the agent's obs)
            log = info["log"]
            dt = info["delta_t"]
            N_t = log[-1][3] if log[-1][3] > 0 else 1e-5
            N_prev = log[-2][3] if log[-2][3] > 0 else 1e-5
            g = (np.log(N_t) - np.log(N_prev)) / dt
            action = choose_next_action(action, g, g_high, g_low)
        with open(os.path.join(out_dir, "trial_%dtcbk.pkl" % (start_index + k)), "wb") as f:
            pickle.dump(info, f)

    print("Done: wrote %d growth-rate-heuristic trials (indices %d..%d, g_high=%g g_low=%g) to %s"
          % (n_trials, start_index, start_index + n_trials - 1, g_high, g_low, out_dir))
    sys.exit(0)
