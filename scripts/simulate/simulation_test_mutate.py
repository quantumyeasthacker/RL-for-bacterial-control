"""Fixed-protocol (non-agent) baseline rollouts under cell mutation.

Applies a hardcoded decision sequence -- "constant" = antibiotic on for every decision --
to the constant- or variable-nutrient env, and pickles one info dict per rollout. Used as
the reference P_constant in the relative-performance metric
(rel. perf. = log P_constant - log P_policy).

Usage:
    python simulation_test_mutate.py \
        half_period antibiotic_value eval_env eval_variable rep results_dir mutate_prob \
        [max_pop] [num_decisions]

Positional args:
    half_period      : int   folder label only for the "constant" protocol (use 0)
    antibiotic_value : float drug level applied when ON (b_actions = [0, a])
    eval_env         : str   "constenv" or "varenv"
    eval_variable    : str   constenv: constant k_n0; varenv: switching period T
    rep              : int   batch index; this call writes trials rep*10 .. rep*10+9
    results_dir      : str   parent output directory
    mutate_prob      : float per-division mutation probability (CellConfig mutate=True)
Optional positional args:
    max_pop          : float population cap that TRUNCATES a rollout, default EnvConfig's
                       1e11. Pass "inf" to disable it so every rollout runs the full 300
                       decisions. This MUST match the agent eval it will be compared against
                       -- eval_trained_agents_*.py use max_pop = inf, and a cap censors the
                       baseline's final population (biasing log P_constant downward) in
                       exactly the runaway trials that a mutation sweep is measuring.
    num_decisions    : int, default 300. Decision steps per rollout (each delta_t = 0.2 h, so
                       300 -> 60 h of control after the 12 h warm-up). MUST match the agent eval
                       this baseline is paired with; the output path does not encode it, so run a
                       non-default horizon into its own results_dir.

Output (under results_dir/):
    a<antibiotic>_<eval_env>_<eval_variable>_mutprob<rate>_value_check/constant_<half_period>/
        trial_<n>tcbk.pkl  (n = rep*10 .. rep*10+9), pickled env info dict per rollout
"""

import os
import numpy as np
import sys
import pickle

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv, VariableNutrientEnv


MAIN = __name__ == "__main__"

if MAIN:

    half_period = int(sys.argv[1])
    initialize_app = "constant" #sys.argv[2]
    antibiotic_value = float(sys.argv[2])
    eval_env = sys.argv[3]
    eval_variable = sys.argv[4]
    rep = int(sys.argv[5])
    results_dir = sys.argv[6]
    mutate_prob = float(sys.argv[7])
    # population cap that truncates a rollout; "inf" disables it. Must match the agent eval
    # this baseline is compared against (see the module docstring).
    max_pop = float(sys.argv[8]) if len(sys.argv) > 8 else EnvConfig.max_pop
    # decision steps per rollout. MUST match the agent eval this baseline is compared against
    # (eval_trained_agents_generalized_mutate.py's num_decisions), since the relative-performance
    # metric time-averages the population over the decision phase.
    num_decisions = int(sys.argv[9]) if len(sys.argv) > 9 else 300

    cell_config = CellConfig(mutate=True, mutate_prob=mutate_prob)

    # is this section below dead? 
    # env_config = EnvConfig(
    #     delay_embed_len = 30,
    #     b_actions = [0, antibiotic_value],
    #     max_pop = np.inf,
    #     k_n0_mean = 2.55,
    # )

    if eval_env == "constenv":
        env_config = EnvConfig(
            k_n0_constant = float(eval_variable),
            b_actions = [0, antibiotic_value],
            max_pop = max_pop
        )
        env = ConstantNutrientEnv(env_config, cell_config)
    elif eval_env == "varenv":
        env_config = EnvConfig(
            b_actions = [0, antibiotic_value],
            T_k_n0 = int(eval_variable),
            k_n0_mean = 2.55,
            sigma_kn0 = 0.1,
            max_pop = max_pop
        )
        env = VariableNutrientEnv(env_config, cell_config)

    folder_name=f"{results_dir}/a{antibiotic_value:.2f}_{eval_env}_{eval_variable}_mutprob{mutate_prob}_value_check/{initialize_app}_{half_period}/"
    os.makedirs(folder_name, exist_ok=True)

    if initialize_app == "low":
        decisions = ([0] * half_period + [1] * half_period) * (num_decisions // 2 // half_period + 1)
    elif initialize_app == "high":
        decisions = ([1] * half_period + [0] * half_period) * (num_decisions // 2 // half_period + 1)
    elif initialize_app == "constant":
        decisions = [1] * num_decisions
    elif initialize_app == "constant_low":
        decisions = [0] * num_decisions

    decisions = decisions[:num_decisions]

    num_of_reps = 10
    for i in range(num_of_reps):
        env.reset()
        for decision in decisions:
            _, _, terminated, truncated, info = env.step(decision)
            if terminated or truncated:
                break
        fname="trial_%d"%int(rep*num_of_reps+i)

        with open(os.path.join(folder_name,str(fname)+'tcbk.pkl'), "wb") as f:
            pickle.dump(info, f)

    print("Done")
    sys.exit(0)