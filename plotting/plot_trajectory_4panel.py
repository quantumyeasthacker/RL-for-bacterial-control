# %%
# plot_trajectory_4panel.py
#
# Usage:
#   Run from the plotting/ directory (so `from utils import ...` resolves), e.g.
#       cd RL-for-bacterial-control-package/plotting && python plot_trajectory_4panel.py
#   or cell-by-cell in an interactive session (# %% markers).
#   Reads the varenv env-parameter sensitivity sweep produced by
#   scripts/simulate/sim_varenv_cellsweep.sbatch. Point RESULTS_DIR below at that
#   sweep's output root (must match the sbatch's RESULTS_DIR).
#
# Output layout:
#   For every swept config (CellConfig param in {alpha, beta, sigma} x percent level in
#   {-20,-10,0,+10,+20}, at fixed a=3.72, T=12, always-apply), writes one 4-panel stacked
#   trajectory figure to figures_jpg/var_nutrient_4panel_cellsweep/ and figures_pdf/...
#   Panels, top -> bottom, sharing the x-axis (Time, h):
#       1. Antibiotic
#       2. Nutrient
#       3. phi_R and phi_S combined in one panel (both black, solid; the highlighted
#          trajectory for each is labelled in-figure at its right end)
#       4. Population P (log scale)
#   In every panel the highlighted (surviving/longest) trial is drawn solid and ~20
#   other replicates are drawn as faint gray "shadow" lines, matching plot_single().
#
# This covers the varenv-with-simulation (no-agent) case only.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import load_logger_data_new, ANTIBIOTIC_COLOR, POP_SIZE_COLOR
from pathlib import Path


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# root holding the sim_varenv_cellsweep.sbatch output (must match its RESULTS_DIR)
RESULTS_DIR = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/varenv_cellsweep")
# where the figures go (jpg + pdf subfolders are created under here)
FIG_BASE = RESULTS_DIR

# fixed run settings (must match the sbatch)
ANTIBIOTIC = 3.72
T_KN0 = 12
INIT_APP = "constant"
HALF_PERIOD = 0

# the sweep grid (must match the sbatch)
SWEEP_PARAMS = ["alpha", "beta", "sigma"]
SWEEP_PCTS = [-20, -10, 0, 10, 20]

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

N_SHADOW = 20  # number of replicate "shadow" trajectories per panel


# %% ----- ----- ----- ----- 4-panel plotting function ----- ----- ----- ----- %% #
def plot_single_4panel(
        loaded_logger,
        out_name,
        color_nut,
        more_antibiotic=False,
        more_nutrient=False,
):
    """4-panel stacked trajectory plot: antibiotic, nutrient, (phi_R + phi_S), P.

    Same coloring/shadowing convention as utils.plot_single, except phi_R and phi_S
    share a single panel. Both are drawn solid black; an in-figure text label sits at
    the right end of each highlighted curve to tell them apart.
    """
    tcbk_list, t, b, k_n0, cell_array, (_, max_id, _) = loaded_logger

    # pick the longest trajectory among the first N_SHADOW as the highlighted one
    max_id = 0
    for j in range(N_SHADOW):
        if len(tcbk_list[j][0]) > len(tcbk_list[max_id][0]):
            max_id = j

    figure, ax = plt.subplots(4, 1)
    figure.subplots_adjust(hspace=.0)

    # --- panel 0: antibiotic ---
    ax_num = 0
    if more_antibiotic:
        for j in range(N_SHADOW):
            ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][2], color='gray', alpha=0.3, linewidth=1)
    ax[ax_num].plot(tcbk_list[max_id][0], tcbk_list[max_id][2], color=ANTIBIOTIC_COLOR)
    ax[ax_num].set_ylabel('Antibiotic')
    ax[ax_num].get_xaxis().set_ticks([])

    # --- panel 1: nutrient ---
    ax_num += 1
    if more_nutrient:
        for j in range(N_SHADOW):
            ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][3], color='gray', alpha=0.3, linewidth=1)
    ax[ax_num].plot(tcbk_list[max_id][0], tcbk_list[max_id][3], color=color_nut)
    ax[ax_num].set_ylabel('Nutrient')
    ax[ax_num].get_xaxis().set_ticks([])

    # --- panel 2: phi_R and phi_S combined ---
    ax_num += 1
    for j in range(N_SHADOW):  # shadows for both curves
        ax[ax_num].plot(tcbk_list[j][0, :-1], tcbk_list[j][4, :-1], color='gray', alpha=0.3, linewidth=1)
        ax[ax_num].plot(tcbk_list[j][0, :-1], tcbk_list[j][5, :-1], color='gray', alpha=0.3, linewidth=1)
    ax[ax_num].plot(tcbk_list[max_id][0, :-1], tcbk_list[max_id][4, :-1], color="black")
    ax[ax_num].plot(tcbk_list[max_id][0, :-1], tcbk_list[max_id][5, :-1], color="black")
    ax[ax_num].set_ylabel(r'$\phi_R,\ \phi_S$')
    ax[ax_num].set_ylim(bottom=-0.1, top=0.45)
    ax[ax_num].get_xaxis().set_ticks([])

    # in-figure labels at the right end of each highlighted curve
    x_end = tcbk_list[max_id][0, -2]
    ax[ax_num].annotate(r'$\phi_R$', xy=(x_end, tcbk_list[max_id][4, -2]),
                        xytext=(4, 0), textcoords='offset points',
                        va='center', ha='left', clip_on=False)
    ax[ax_num].annotate(r'$\phi_S$', xy=(x_end, tcbk_list[max_id][5, -2]),
                        xytext=(4, 0), textcoords='offset points',
                        va='center', ha='left', clip_on=False)

    # --- panel 3: population ---
    ax_num += 1
    for j in range(N_SHADOW):
        ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][1], color='gray', alpha=0.3, linewidth=1)
    ax[ax_num].plot(tcbk_list[max_id][0], tcbk_list[max_id][1], color=POP_SIZE_COLOR)
    ax[ax_num].set_ylabel(r'$P$')
    ax[ax_num].set_xlabel('Time (h)')
    ax[ax_num].set_yscale('log')

    out_path = os.path.dirname(out_name)
    os.makedirs(out_path, exist_ok=True)

    figure.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(figure)


# %% ----- ----- ----- ----- varenv sim env-parameter sweep (no agent) ----- ----- ----- ----- %% #
n_trials = 100  # trials per config (rep 0..9 x num_of_reps=10 in the sbatch)

for param in SWEEP_PARAMS:
    for pct in SWEEP_PCTS:
        sweep_tag = f"{param}_{pct:+d}pct"  # matches simulation_test_varenv.py folder token
        folder_name = (
            RESULTS_DIR
            / f"a{ANTIBIOTIC:.2f}_T{T_KN0}_{sweep_tag}_value_check"
            / f"{INIT_APP}_{HALF_PERIOD}"
        )

        try:
            loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials)
        except (IndexError, FileNotFoundError):
            print(f"Skipping {sweep_tag}: no data found in {folder_name}")
            continue

        out_jpg = FIG_BASE / "figures_jpg" / "var_nutrient_4panel_cellsweep" / f"a{ANTIBIOTIC:.2f}_T{T_KN0}_{sweep_tag}.jpg"
        plot_single_4panel(loaded_logger, out_jpg, "#548c6a", True, True)

        out_pdf = FIG_BASE / "figures_pdf" / "var_nutrient_4panel_cellsweep" / f"a{ANTIBIOTIC:.2f}_T{T_KN0}_{sweep_tag}.pdf"
        plot_single_4panel(loaded_logger, out_pdf, "#548c6a", True, True)

        print(f"Plotted {sweep_tag}")

# %%
