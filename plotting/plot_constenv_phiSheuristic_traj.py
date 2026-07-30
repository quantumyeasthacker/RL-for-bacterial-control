"""3-panel stacked trajectory figure for the phi_S-threshold heuristic on constant env n=2
(rollouts from simulation_test_phiSheuristic_constenv.py).

Panels, top -> bottom, sharing the x-axis (Time, h):
    1. Antibiotic (bang-bang schedule; differs per trial)
    2. phi_S  (population-average stress fraction) with the 0.2 / 0.03 switch thresholds
    3. Population P (log scale)
The nutrient panel of the varenv 4-panel figure is dropped here because the nutrient is
constant (k_n0=2). Same coloring/shadowing convention as plot_phiSheuristic_results.py:
the longest-surviving trajectory among the first N_SHADOW is drawn solid, the rest faint gray.

Input:
    <BASE>/phiS_heuristic/trial_*tcbk.pkl   (from simulation_test_phiSheuristic_constenv.py)

Output:
    plotting/figures_jpg/constenv_phiSheuristic_n2_3panel.jpg
    plotting/figures_pdf/constenv_phiSheuristic_n2_3panel.pdf
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import load_logger_data_new, ANTIBIOTIC_COLOR, POP_SIZE_COLOR

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ----- run settings (match sim_phiSheuristic_constenv.sbatch) -----
# subdir / figure tag / thresholds come from the environment so a threshold sweep renders each
# combo separately; the defaults reproduce the original 0.2/0.03 run.
HEUR_SUBDIR = os.environ.get("HEUR_SUBDIR", "phiS_heuristic")
FIG_TAG = os.environ.get("FIG_TAG", "")

BASE = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/constenv_logpop_runs/relperf_n2")
FOLDER = BASE / HEUR_SUBDIR

# phi_S switch thresholds (match simulation_test_phiSheuristic_constenv.py)
PHIS_HIGH = float(os.environ.get("PHIS_HIGH", "0.2"))
PHIS_LOW = float(os.environ.get("PHIS_LOW", "0.03"))

# geometry (match the sim)
warm_up_embed = 61
num_decisions = 300
sim_length = num_decisions + warm_up_embed
max_pop = int(1e11)
N_SHADOW = 20
N_TRIALS = 100


def plot_single_3panel(loaded_logger, out_name):
    """3-panel stacked trajectory plot: antibiotic, phi_S, P."""
    tcbk_list = loaded_logger[0]

    # highlight the longest-surviving trajectory among the first N_SHADOW
    n_show = min(N_SHADOW, len(tcbk_list))
    max_id = 0
    for j in range(n_show):
        if len(tcbk_list[j][0]) > len(tcbk_list[max_id][0]):
            max_id = j

    figure, ax = plt.subplots(3, 1)
    figure.subplots_adjust(hspace=.0)

    # --- panel 0: antibiotic (bang-bang schedule differs per trial) ---
    a = 0
    for j in range(n_show):
        ax[a].plot(tcbk_list[j][0], tcbk_list[j][2], color='gray', alpha=0.3, linewidth=1)
    ax[a].plot(tcbk_list[max_id][0], tcbk_list[max_id][2], color=ANTIBIOTIC_COLOR)
    ax[a].set_ylabel('Antibiotic')
    ax[a].get_xaxis().set_ticks([])

    # --- panel 1: phi_S only, with the switch thresholds marked ---
    a += 1
    for j in range(n_show):
        ax[a].plot(tcbk_list[j][0, :-1], tcbk_list[j][5, :-1], color='gray', alpha=0.3, linewidth=1)
    ax[a].plot(tcbk_list[max_id][0, :-1], tcbk_list[max_id][5, :-1], color="black")
    for thr in (PHIS_HIGH, PHIS_LOW):
        ax[a].axhline(thr, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax[a].set_ylabel(r'$\phi_S$')
    ax[a].set_ylim(bottom=-0.1, top=0.45)
    ax[a].get_xaxis().set_ticks([])
    x_end = tcbk_list[max_id][0, -2]
    ax[a].annotate(r'$\phi_S$', xy=(x_end, tcbk_list[max_id][5, -2]),
                   xytext=(4, 0), textcoords='offset points',
                   va='center', ha='left', clip_on=False)

    # --- panel 2: population ---
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


if __name__ == "__main__":
    loaded_logger = load_logger_data_new(str(FOLDER) + "/", sim_length, max_pop, n_trials=N_TRIALS)
    PLOT_DIR = Path(__file__).resolve().parent
    plot_single_3panel(loaded_logger, PLOT_DIR / "figures_jpg" / f"constenv_phiSheuristic_n2_3panel{FIG_TAG}.jpg")
    plot_single_3panel(loaded_logger, PLOT_DIR / "figures_pdf" / f"constenv_phiSheuristic_n2_3panel{FIG_TAG}.pdf")
    print(f"Wrote 3-panel trajectory figure (subdir={HEUR_SUBDIR}, tag='{FIG_TAG}')")
