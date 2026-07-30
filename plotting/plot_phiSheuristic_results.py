# %%
# plot_phiSheuristic_results.py
#
# Aggregates and visualizes the phi_S-threshold heuristic simulation on the
# variable-nutrient env (scripts/simulate/sim_phiSheuristic_varenv.sbatch, which runs
# simulation_test_phiSheuristic_varenv.py). The controller switches the antibiotic on a
# hysteresis rule on the population-average stress fraction phi_S:
#   ON at start of control -> OFF once phi_S_ave > 0.2 -> ON once phi_S_ave < 0.03 -> ...
#
# Usage:
#   Run from the plotting/ directory (so `from utils import ...` resolves):
#       cd RL-for-bacterial-control-package/plotting && python plot_phiSheuristic_results.py
#   Point RESULTS_DIR below at the sbatch's output root (must match its RESULTS_DIR).
#
# Does three things:
#   1. Aggregation: pools the per-rep extinction_summary_rep*.json files into one overall
#      extinction fraction (n_extinct_total / n_trials_total) and writes
#      <folder>/extinction_aggregate.json. Falls back to counting the trial pkls directly
#      (final population == 0) if the per-rep summaries are absent.
#   2. 4-panel stacked trajectory figure (Antibiotic / Nutrient / phi_S / Population),
#      same coloring/shadowing convention as plot_trajectory_4panel.py. Dotted guides mark
#      the phi_S switch thresholds (0.2, 0.03) in the phi_S panel.
#   3. Bar chart comparing the heuristic's measured extinction fraction to our previous RL
#      agent, which achieved 1.0 (hard-coded -- that run's data is not available).
#
# Output layout (under FIG_BASE):
#   figures_jpg/phiSheuristic/a<ab:.2f>_T<T>_phiSheuristic_4panel.jpg        (+ figures_pdf/...)
#   figures_jpg/phiSheuristic/a<ab:.2f>_T<T>_phiSheuristic_extinction_bar.jpg (+ figures_pdf/...)
#   <RESULTS_DIR>/a<ab:.2f>_T<T>_phiSheuristic/extinction_aggregate.json

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

from utils import load_logger_data_new, ANTIBIOTIC_COLOR, POP_SIZE_COLOR


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ----- run settings (must match sim_phiSheuristic_varenv.sbatch) -----
RESULTS_DIR = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/phiSheuristic_varenv")
FIG_BASE = RESULTS_DIR
ANTIBIOTIC = 3.72
T_KN0 = 12

# phi_S switch thresholds (must match simulation_test_phiSheuristic_varenv.py)
PHIS_HIGH = 0.2
PHIS_LOW = 0.03

# our previous RL agent achieved extinction_frac = 1.0 (data not available -> hard-coded)
RL_AGENT_FRAC = 1.0

# colors: teal (heuristic) vs orange (RL agent) -- CVD-safe blue/orange pair; bars are also
# labelled on the axis and with their value, so identity never depends on color alone.
HEURISTIC_COLOR = ANTIBIOTIC_COLOR  # "#216d87"
RL_COLOR = "#c8791f"
NUT_COLOR = "#548c6a"

# geometry (must match the sim; see plot_trajectory_4panel.py)
delta_t = 0.2
warm_up_embed = 60 + 1
num_decisions = 300
sim_length = num_decisions + warm_up_embed
max_pop = int(1e11)
N_SHADOW = 20

FOLDER = RESULTS_DIR / f"a{ANTIBIOTIC:.2f}_T{T_KN0}_phiSheuristic"


# %% ----- ----- ----- 1. aggregate extinction fraction ----- ----- ----- %% #
def aggregate_extinction(folder):
    """Pool per-rep JSON summaries into one overall extinction fraction.

    Returns (extinction_frac, n_extinct, n_trials, source_str).
    Falls back to counting the trial pkls directly if no summaries are present.
    """
    summary_files = sorted(glob.glob(str(folder / "extinction_summary_rep*.json")))
    if summary_files:
        n_extinct = 0
        n_trials = 0
        reps = []
        for sf in summary_files:
            with open(sf) as f:
                s = json.load(f)
            n_extinct += int(s["n_extinct"])
            n_trials += int(s["n_trials"])
            reps.append(s["rep"])
        frac = n_extinct / n_trials if n_trials else float("nan")
        return frac, n_extinct, n_trials, f"{len(summary_files)} per-rep summaries (reps {sorted(reps)})"

    # fallback: count trial pkls directly (final population == 0)
    tcbk_list, *_ = load_logger_data_new(folder, sim_length, max_pop, n_trials=1000)
    extinct = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    n_extinct = int(np.sum(extinct))
    n_trials = len(extinct)
    frac = n_extinct / n_trials if n_trials else float("nan")
    return frac, n_extinct, n_trials, f"{n_trials} trial pkls (no summaries found)"


# %% ----- ----- ----- 2. 4-panel trajectory figure ----- ----- ----- %% #
def plot_single_4panel(loaded_logger, out_name, color_nut,
                       more_antibiotic=True, more_nutrient=True):
    """4-panel stacked trajectory plot: antibiotic, nutrient, phi_S, P."""
    tcbk_list, t, b, k_n0, cell_array, (_, max_id, _) = loaded_logger

    # highlight the longest-surviving trajectory among the first N_SHADOW
    n_show = min(N_SHADOW, len(tcbk_list))
    max_id = 0
    for j in range(n_show):
        if len(tcbk_list[j][0]) > len(tcbk_list[max_id][0]):
            max_id = j

    figure, ax = plt.subplots(4, 1)
    figure.subplots_adjust(hspace=.0)

    # --- panel 0: antibiotic (bang-bang schedule differs per trial) ---
    a = 0
    if more_antibiotic:
        for j in range(n_show):
            ax[a].plot(tcbk_list[j][0], tcbk_list[j][2], color='gray', alpha=0.3, linewidth=1)
    ax[a].plot(tcbk_list[max_id][0], tcbk_list[max_id][2], color=ANTIBIOTIC_COLOR)
    ax[a].set_ylabel('Antibiotic')
    ax[a].get_xaxis().set_ticks([])

    # --- panel 1: nutrient ---
    a += 1
    if more_nutrient:
        for j in range(n_show):
            ax[a].plot(tcbk_list[j][0], tcbk_list[j][3], color='gray', alpha=0.3, linewidth=1)
    ax[a].plot(tcbk_list[max_id][0], tcbk_list[max_id][3], color=color_nut)
    ax[a].set_ylabel('Nutrient')
    ax[a].get_xaxis().set_ticks([])

    # --- panel 2: phi_S only, with the switch thresholds marked ---
    a += 1
    for j in range(n_show):
        ax[a].plot(tcbk_list[j][0, :-1], tcbk_list[j][5, :-1], color='gray', alpha=0.3, linewidth=1)
    ax[a].plot(tcbk_list[max_id][0, :-1], tcbk_list[max_id][5, :-1], color="black")
    # phi_S switch thresholds (the rule that drives the antibiotic panel above)
    for thr in (PHIS_HIGH, PHIS_LOW):
        ax[a].axhline(thr, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax[a].set_ylabel(r'$\phi_S$')
    ax[a].set_ylim(bottom=-0.1, top=0.45)
    ax[a].get_xaxis().set_ticks([])

    x_end = tcbk_list[max_id][0, -2]
    ax[a].annotate(r'$\phi_S$', xy=(x_end, tcbk_list[max_id][5, -2]),
                   xytext=(4, 0), textcoords='offset points',
                   va='center', ha='left', clip_on=False)

    # --- panel 3: population ---
    a += 1
    for j in range(n_show):
        ax[a].plot(tcbk_list[j][0], tcbk_list[j][1], color='gray', alpha=0.3, linewidth=1)
    ax[a].plot(tcbk_list[max_id][0], tcbk_list[max_id][1], color=POP_SIZE_COLOR)
    ax[a].set_ylabel(r'$P$')
    ax[a].set_xlabel('Time (h)')
    ax[a].set_yscale('log')

    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    figure.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(figure)


# %% ----- ----- ----- 3. extinction-fraction bar chart ----- ----- ----- %% #
def plot_extinction_bar(heuristic_frac, out_name):
    """Two-bar comparison: phi_S heuristic (measured) vs previous RL agent (1.0)."""
    labels = [r'$\phi_S$ heuristic', 'RL agent']
    values = [heuristic_frac, RL_AGENT_FRAC]
    colors = [HEURISTIC_COLOR, RL_COLOR]

    figure, ax = plt.subplots(figsize=(3.2, 3.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62, zorder=3)

    # direct value labels (secondary encoding -> identity never color-alone)
    for rect, v in zip(bars, values):
        ax.annotate(f'{v:.2f}', xy=(rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Extinction fraction')
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color='0.85', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    figure.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(figure)


# %% ----- ----- ----- run ----- ----- ----- %% #
if __name__ == "__main__":
    # 1. aggregate
    frac, n_ext, n_tot, source = aggregate_extinction(FOLDER)
    agg = {
        "extinction_frac": frac,
        "n_extinct": n_ext,
        "n_trials": n_tot,
        "source": source,
        "antibiotic_value": ANTIBIOTIC,
        "T_k_n0": T_KN0,
        "phiS_high": PHIS_HIGH,
        "phiS_low": PHIS_LOW,
        "rl_agent_frac": RL_AGENT_FRAC,
    }
    with open(FOLDER / "extinction_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Overall extinction fraction: {frac:.3f}  ({n_ext}/{n_tot})  [from {source}]")

    # 2. 4-panel trajectory figure
    loaded_logger = load_logger_data_new(FOLDER, sim_length, max_pop, n_trials=n_tot)
    tag = f"a{ANTIBIOTIC:.2f}_T{T_KN0}_phiSheuristic"
    plot_single_4panel(loaded_logger,
                       FIG_BASE / "figures_jpg" / "phiSheuristic" / f"{tag}_4panel.jpg", NUT_COLOR)
    plot_single_4panel(loaded_logger,
                       FIG_BASE / "figures_pdf" / "phiSheuristic" / f"{tag}_4panel.pdf", NUT_COLOR)
    print("Wrote 4-panel trajectory figure")

    # 3. extinction bar chart
    plot_extinction_bar(frac,
                        FIG_BASE / "figures_jpg" / "phiSheuristic" / f"{tag}_extinction_bar.jpg")
    plot_extinction_bar(frac,
                        FIG_BASE / "figures_pdf" / "phiSheuristic" / f"{tag}_extinction_bar.pdf")
    print("Wrote extinction bar chart")
    print("Done")
