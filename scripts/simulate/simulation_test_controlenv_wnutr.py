"""
Famine-protocol timing-sensitivity simulation (control-nutrient env).

The baseline "Famine" protocol reduces the nutrient concentration to its lowest
value AND applies the antibiotic at the same instant (offset = 0). This script
sweeps a timing OFFSET between the two interventions to test how sensitive the
protocol's performance (extinction fraction) is to the nutrient reduction being
staggered relative to the antibiotic.

Timescale
    One decision step is delta_t = 0.2 h = 12 min (EnvConfig.delta_t).
    Offsets are expressed in whole decision steps and swept over
        OFFSETS = [-2, -1, 0, +1, +2]  (i.e. -24, -12, 0, +12, +24 min).
    Sign convention (offset = t_nutrient_reduced - t_antibiotic_applied):
        offset > 0 : nutrient reduced AFTER antibiotic
                     -> during the |offset|-step window: antibiotic ON, nutrient HIGH (feast)
        offset < 0 : nutrient reduced BEFORE antibiotic
                     -> during the |offset|-step window: nutrient LOW, antibiotic OFF
        offset = 0 : simultaneous (baseline famine)

Per rep
    reset()  -> 12 h (60-step) warm-up at a RANDOM nutrient from k_n0_actions, b = 0
    control  -> stagger window of |offset| steps, then both-on famine for the rest
                of num_decisions steps (nutrient = min(k_n0_actions), b = max(b_actions)).
    Reps are paired across offsets via a per-rep RNG seed, so the only thing that
    differs between offsets for a given trial index is the stagger timing
    (set PAIR_SEEDS = False to restore fully independent, unseeded reps).

Usage
    python simulation_test_controlenv_wnutr.py <manual_protocol> <antibiotic_value> \
        <nutrient_range> <rep> <results_dir>

    manual_protocol   : "Famine" (feast->famine) or "Feast" (antibiotic-only)
    antibiotic_value  : float, the "on" antibiotic level (b = [0, antibiotic_value])
    nutrient_range    : underscore-joined nutrient actions, e.g. "1_3"
    rep               : int batch index; trial ids are rep*num_of_reps + i so array
                        jobs can be aggregated
    results_dir       : output root

    Optional environment-variable overrides (used e.g. for cheap smoke tests):
        NUM_REPS       : reps per offset per invocation (default 10)
        NUM_DECISIONS  : control-phase decision steps (default 300)
        OFFSETS        : comma-separated offsets in decision steps (default "-2,-1,0,1,2")

Output layout
    <results_dir>/a<ab:.2f>_n<nutrient_range>_value_check/<manual_protocol>/
        offset<+d>steps/                      # one folder per offset (e.g. offset+1steps)
            trial_<k>tcbk.pkl                 # pickled env `info` (info["log"] trajectory),
                                              # same format the plotting code expects
        extinction_summary_rep<rep>.pkl       # dict summarizing this batch:
            {
              "offsets_steps":   [-2,-1,0,1,2],
              "offsets_min":     [-24,-12,0,12,24],
              "delta_t":         0.2,
              "num_of_reps":     <N>,
              "rep":             <rep>,
              "extinct_counts":  {offset: int, ...},   # reps that reached extinction
              "extinction_frac": {offset: float, ...}, # extinct_count / num_of_reps
              "final_num_cells": {offset: [float, ...]},# final population per rep
            }
"""

import os
import numpy as np
import sys
import pickle

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ControlNutrientEnv


# timing offsets between nutrient reduction and antibiotic application, in decision steps
OFFSETS = [-2, -1, 0, 1, 2]
# pair reps across offsets by seeding each trial index identically (keeps the random
# warm-up nutrient random, but identical for the same trial across offsets)
PAIR_SEEDS = True
BASE_SEED = 0


def run_famine_offset(env, k_n0_low, k_n0_high, b_on, b_off, offset, num_decisions):
    """Run one control episode (after reset) with a staggered nutrient/antibiotic switch.

    Returns (info, extinct) where extinct is True if the population reached 0.
    """
    window = abs(offset)
    for step in range(num_decisions):
        if step < window:
            if offset > 0:
                # nutrient reduced AFTER antibiotic: antibiotic on, nutrient still high
                k_n0, b = k_n0_high, b_on
            else:
                # nutrient reduced BEFORE antibiotic: nutrient low, antibiotic off
                k_n0, b = k_n0_low, b_off
        else:
            # both-on famine
            k_n0, b = k_n0_low, b_on

        _, _, terminated, truncated, info = env.step_hardcode(k_n0, b)
        if terminated or truncated:
            break

    extinct = env.sim_cells.true_num_cells == 0
    return info, extinct


MAIN = __name__ == "__main__"

if MAIN:
    manual_protocol = sys.argv[1]
    antibiotic_value = float(sys.argv[2])
    nutrient_range = sys.argv[3]
    rep = int(sys.argv[4])
    results_dir = sys.argv[5]

    num_decisions = int(os.environ.get("NUM_DECISIONS", 300))
    num_of_reps = int(os.environ.get("NUM_REPS", 10))
    offsets = [int(o) for o in os.environ["OFFSETS"].split(",")] if os.environ.get("OFFSETS") else OFFSETS

    cell_config = CellConfig()
    k_n0_actions = [float(nutr) for nutr in nutrient_range.split('_')]
    env_config = EnvConfig(
        k_n0_actions = k_n0_actions,
        b_actions = [0, antibiotic_value],
        num_actions = len(k_n0_actions) * 2,
        max_pop = np.inf,
    )

    env = ControlNutrientEnv(env_config, cell_config)

    # feast (high) vs famine (low) nutrient, and antibiotic on/off levels
    k_n0_low = min(env_config.k_n0_actions)
    k_n0_high = max(env_config.k_n0_actions)
    b_on = max(env_config.b_actions)
    b_off = min(env_config.b_actions)

    if manual_protocol not in ("Famine", "Feast"):
        raise ValueError(f"unknown manual_protocol: {manual_protocol}")
    if manual_protocol == "Feast":
        # antibiotic-only control: nutrient stays high, so the "low" target is also high
        k_n0_low = k_n0_high

    base_folder = f"{results_dir}/a{antibiotic_value:.2f}_n{nutrient_range}_value_check/{manual_protocol}/"

    extinct_counts = {}
    extinction_frac = {}
    final_num_cells = {}

    for offset in offsets:
        offset_folder = os.path.join(base_folder, f"offset{offset:+d}steps")
        os.makedirs(offset_folder, exist_ok=True)

        n_extinct = 0
        finals = []
        for i in range(num_of_reps):
            trial_index = rep * num_of_reps + i
            if PAIR_SEEDS:
                np.random.seed(BASE_SEED + trial_index)

            env.reset()
            info, extinct = run_famine_offset(
                env, k_n0_low, k_n0_high, b_on, b_off, offset, num_decisions
            )

            n_extinct += int(extinct)
            finals.append(float(env.sim_cells.true_num_cells))

            fname = "trial_%d" % trial_index
            with open(os.path.join(offset_folder, str(fname) + 'tcbk.pkl'), "wb") as f:
                pickle.dump(info, f)

        extinct_counts[offset] = n_extinct
        extinction_frac[offset] = n_extinct / num_of_reps
        final_num_cells[offset] = finals
        print(f"offset {offset:+d} steps ({offset * env_config.delta_t * 60:+.0f} min): "
              f"extinction fraction {extinction_frac[offset]:.3f} ({n_extinct}/{num_of_reps})")

    summary = {
        "offsets_steps": offsets,
        "offsets_min": [o * env_config.delta_t * 60 for o in offsets],
        "delta_t": env_config.delta_t,
        "num_of_reps": num_of_reps,
        "rep": rep,
        "extinct_counts": extinct_counts,
        "extinction_frac": extinction_frac,
        "final_num_cells": final_num_cells,
    }
    os.makedirs(base_folder, exist_ok=True)
    with open(os.path.join(base_folder, f"extinction_summary_rep{rep}.pkl"), "wb") as f:
        pickle.dump(summary, f)

    print("Done")
    sys.exit(0)
