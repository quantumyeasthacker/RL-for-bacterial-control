"""phi_S-threshold heuristic controller on the CONSTANT-nutrient env (constant env n=2).

Same hysteresis rule as simulation_test_phiSheuristic_varenv.py, but on ConstantNutrientEnv
(fixed nutrient k_n0) instead of the variable-nutrient env:
    - antibiotic is APPLIED at the start of control (first decision = ON)
    - it is REMOVED once phi_S_ave rises above PHIS_HIGH (default 0.2)
    - it is APPLIED again once phi_S_ave falls below PHIS_LOW (default 0.03)
    - this on/off cycle repeats until the end of the simulation.
phi_S_ave is read once per decision step from the sim log
    (log entry layout: [t, k_n0, b, num_cells, U_ave, phi_R_ave, phi_S_ave]),
i.e. info["log"][-1][6] after each env.step().

Each rollout's `info` is pickled as trial_<k>tcbk.pkl, the format load_logger_data_new
expects. This is the "policy" side of the relative-performance metric
(rel. perf. = log P_constant - log P_heuristic); the constant-drug baseline (P_constant) is
produced separately by sim_baseline_constenv.py. The env config mirrors that baseline exactly
(k_n0_observation=False, b_observation=True, max_pop=inf) so the two are directly comparable;
obs_type/reward_type do not affect the population dynamics the heuristic controls.

Usage:
    python simulation_test_phiSheuristic_constenv.py out_dir n_trials antibiotic nutrient delay [start_index]

Positional args:
    out_dir     : str   directory to write trial_<k>tcbk.pkl files into (created if missing)
    n_trials    : int   number of heuristic rollouts to run in this call
    antibiotic  : float drug level applied when ON (b_actions = [0, a]); 3.72 here
    nutrient    : float constant nutrient k_n0; 2.0 here (the "n=2" condition)
    delay       : int   observation delay-embed length; 30 here (affects warm-up only)
    start_index : int   (optional, default 0) trial-file numbering offset, so a Slurm array can
                        run disjoint blocks in parallel: task t passes start_index = t*n_trials
                        and writes trial_<t*n_trials> ... trial_<t*n_trials + n_trials - 1>.
    phiS_high   : float (optional, default PHIS_HIGH=0.2)  remove antibiotic once phi_S_ave > this.
    phiS_low    : float (optional, default PHIS_LOW=0.03)  re-apply antibiotic once phi_S_ave < this.

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

# default phi_S hysteresis thresholds (match simulation_test_phiSheuristic_varenv.py)
PHIS_HIGH = 0.2   # remove antibiotic once phi_S_ave rises above this
PHIS_LOW = 0.03   # re-apply antibiotic once phi_S_ave falls below this


def choose_next_action(current_action, phi_S, phiS_high, phiS_low):
    """Hysteresis rule on phi_S_ave.

    current_action: 1 = antibiotic ON, 0 = OFF (the action just applied).
    Returns the action to apply on the next decision step.
    """
    if current_action == 1 and phi_S > phiS_high:
        return 0  # cells are well protected -> stop dosing, let phi_S decay
    if current_action == 0 and phi_S < phiS_low:
        return 1  # defenses have decayed -> resume dosing
    return current_action


if MAIN:
    out_dir = sys.argv[1]
    n_trials = int(sys.argv[2])
    antibiotic_value = float(sys.argv[3])
    nutrient_value = float(sys.argv[4])
    delay_embed_len = int(sys.argv[5])
    start_index = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    phiS_high = float(sys.argv[7]) if len(sys.argv) > 7 else PHIS_HIGH
    phiS_low = float(sys.argv[8]) if len(sys.argv) > 8 else PHIS_LOW
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
            phi_S = info["log"][-1][6]           # population-average stress fraction
            action = choose_next_action(action, phi_S, phiS_high, phiS_low)
        with open(os.path.join(out_dir, "trial_%dtcbk.pkl" % (start_index + k)), "wb") as f:
            pickle.dump(info, f)

    print("Done: wrote %d phi_S-heuristic trials (indices %d..%d, phiS_high=%g phiS_low=%g) to %s"
          % (n_trials, start_index, start_index + n_trials - 1, phiS_high, phiS_low, out_dir))
    sys.exit(0)
