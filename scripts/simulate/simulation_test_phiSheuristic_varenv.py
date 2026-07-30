import os
# import matplotlib.pyplot as plt
# import copy
import numpy as np
import sys
import json
import pickle

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, VariableNutrientEnv


# ---------------------------------------------------------------------------
# phi_S-threshold heuristic controller (variable-nutrient env)
#
# Instead of an open-loop pulsing schedule, the antibiotic is switched by a
# hysteresis rule on the population-average stress-protein fraction, phi_S:
#   - antibiotic is APPLIED at the start of control (first decision = ON)
#   - it is REMOVED once phi_S_ave rises above PHIS_HIGH (default 0.2)
#   - it is APPLIED again once phi_S_ave falls below PHIS_LOW (default 0.03)
#   - this on/off cycle repeats until the end of the simulation.
# phi_S_ave is read once per decision step from the sim log
#   (log entry layout: [t, k_n0, b, num_cells, U_ave, phi_R_ave, phi_S_ave]),
# i.e. info["log"][-1][6] after each env.step().
#
# The extinction fraction (fraction of the trials whose final population == 0)
# is assessed over the trials run in this call and written to a per-rep summary.
#
# Usage:
#   python simulation_test_phiSheuristic_varenv.py \
#       <antibiotic_value> <T_k_n0> <rep> <results_dir> [cell_param] [cell_pct]
#
# positional args:
#   1 antibiotic_value  (float)  antibiotic level applied when ON
#   2 T_k_n0            (int)    nutrient oscillation period passed to EnvConfig
#   3 rep              (int)    replicate-batch index (offsets trial numbering)
#   4 results_dir      (str)    root output directory
# optional CellConfig sweep args (env-parameter sensitivity sweep):
#   5 cell_param       (str)    name of a CellConfig field to override
#   6 cell_pct         (int)    percent change; field := default * (1 + pct/100)
# When cell_param/cell_pct are given, one CellConfig field is overridden and the
# folder name gains a "_{param}_{+pct}pct" token.
#
# Output layout:
#   <results_dir>/a{antibiotic_value:.2f}_T{T_k_n0}{sweep_tag}_phiSheuristic/
#     trial_{rep*num_of_reps + i}tcbk.pkl   pickled `info` dict per trial (env log)
#     extinction_summary_rep{rep}.json      {n_trials, n_extinct, extinction_frac,
#                                            phiS_high, phiS_low, antibiotic_value,
#                                            T_k_n0, rep, trial_ids, extinct_flags}
# ---------------------------------------------------------------------------


MAIN = __name__ == "__main__"

# phi_S hysteresis thresholds
PHIS_HIGH = 0.2   # remove antibiotic once phi_S_ave rises above this
PHIS_LOW = 0.03   # re-apply antibiotic once phi_S_ave falls below this


def choose_next_action(current_action, phi_S):
    """Hysteresis rule on phi_S_ave.

    current_action: 1 = antibiotic ON, 0 = OFF (the action just applied).
    Returns the action to apply on the next decision step.
    """
    if current_action == 1 and phi_S > PHIS_HIGH:
        return 0  # cells are well protected -> stop dosing, let phi_S decay
    if current_action == 0 and phi_S < PHIS_LOW:
        return 1  # defenses have decayed -> resume dosing
    return current_action


if MAIN:
    antibiotic_value = float(sys.argv[1])
    T_k_n0 = int(sys.argv[2])
    rep = int(sys.argv[3])
    results_dir = sys.argv[4]
    cell_param = sys.argv[5] if len(sys.argv) > 5 else None
    cell_pct = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    num_decisions = 300

    if cell_param is not None:
        default_val = getattr(CellConfig(), cell_param)
        new_val = default_val * (1 + cell_pct / 100)
        cell_config = CellConfig(**{cell_param: new_val})
        sweep_tag = f"_{cell_param}_{cell_pct:+d}pct"
        print(f"CellConfig override: {cell_param} {default_val} -> {new_val} ({cell_pct:+d}%)")
    else:
        cell_config = CellConfig()
        sweep_tag = ""

    env_config = EnvConfig(
        delay_embed_len = 30,
        b_actions = [0, antibiotic_value],
        max_pop = np.inf,
        T_k_n0 = T_k_n0,
        k_n0_mean = 2.55,
        sigma_kn0 = 0.1
    )

    # T_k_n0: Optional[Union[float, None]] = None # 6
    # k_n0_mean: Optional[Union[float, None]] = None # 2.55
    # sigma_kn0: Optional[Union[float, None]] = None # 0.1

    env = VariableNutrientEnv(env_config, cell_config)

    folder_name = f"{results_dir}/a{antibiotic_value:.2f}_T{T_k_n0}{sweep_tag}_phiSheuristic/"
    os.makedirs(folder_name, exist_ok=True)

    num_of_reps = 10
    trial_ids = []
    extinct_flags = []
    for i in range(num_of_reps):
        env.reset()

        action = 1  # antibiotic is always applied at the start of control
        info = None
        for _ in range(num_decisions):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            # phi_S_ave from the most recent log entry drives the next decision
            phi_S = info["log"][-1][6]
            action = choose_next_action(action, phi_S)

        trial_id = int(rep * num_of_reps + i)
        fname = "trial_%d" % trial_id

        with open(os.path.join(folder_name, str(fname) + 'tcbk.pkl'), "wb") as f:
            pickle.dump(info, f)

        # extinction: final population count == 0 (log layout index 3 = num_cells)
        extinct = int(info["log"][-1][3] == 0)
        trial_ids.append(trial_id)
        extinct_flags.append(extinct)

    extinction_frac = float(np.mean(extinct_flags)) if extinct_flags else float("nan")

    summary = {
        "n_trials": len(extinct_flags),
        "n_extinct": int(np.sum(extinct_flags)),
        "extinction_frac": extinction_frac,
        "phiS_high": PHIS_HIGH,
        "phiS_low": PHIS_LOW,
        "antibiotic_value": antibiotic_value,
        "T_k_n0": T_k_n0,
        "rep": rep,
        "trial_ids": trial_ids,
        "extinct_flags": extinct_flags,
    }
    with open(os.path.join(folder_name, f"extinction_summary_rep{rep}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Extinction fraction (rep {rep}, {len(extinct_flags)} trials): {extinction_frac:.3f}")
    print("Done")
    sys.exit(0)
