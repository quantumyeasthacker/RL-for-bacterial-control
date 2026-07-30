# %%
# plot_2panel_survext.py
#
# Usage:
#   Run from the plotting/ directory (so `from utils import ...` resolves):
#       cd RL-for-bacterial-control-package/plotting && python plot_2panel_survext.py
#   Reads the varenv cell-parameter sweep output under RESULTS_DIR.
#
# Output layout:
#   For each config in CONFIGS, a 2-panel stacked figure (shared Time axis):
#       top:    Antibiotic  (original teal; the surviving trajectory highlighted, gray shadows)
#       bottom: Population P (log scale; gray shadows, PLUS two highlighted trajectories:
#                 - the survivor drawn in green (POP_SIZE_COLOR), and
#                 - the longest-surviving-before-extinction trajectory drawn in red)
#   Written next to the 4-panel figures as <config>_2panel_survext.jpg.
#
# This is a trimmed 2-panel variant of plot_trajectory_4panel.py's plot for the varenv
# no-agent simulation sweep.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

from utils import load_logger_data_new, ANTIBIOTIC_COLOR, POP_SIZE_COLOR


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

RESULTS_DIR = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/varenv_cellsweep")
FIG_DIR = RESULTS_DIR / "figures_jpg" / "var_nutrient_4panel_cellsweep"

# fixed run settings (match the sweep)
ANTIBIOTIC = 3.72
T_KN0 = 12
INIT_APP = "constant"
HALF_PERIOD = 0

num_decisions = 300
warm_up_embed = 61
sim_length = num_decisions + warm_up_embed   # 361
max_pop = int(1e11)
N_SHADOW = 20

EXTINCT_COLOR = "crimson"

CONFIGS = ["beta_-20pct", "beta_+0pct"]


def _positive(t, p):
    """Return (t, p) keeping only points with p > 0 (log-axis safe; drops the pop=0 death point)."""
    m = np.asarray(p) > 0
    return np.asarray(t)[m], np.asarray(p)[m]


def plot_two_panel_survext(loaded_logger, out_name):
    tcbk_list, *_ = loaded_logger

    lengths = [t.shape[1] for t in tcbk_list]
    finals = [t[1, -1] for t in tcbk_list]

    # green survivor: reaches full length, pick the most clearly thriving (max final pop)
    survivors = [i for i, L in enumerate(lengths) if L >= sim_length]
    green_id = max(survivors, key=lambda i: finals[i])
    # red: among trajectories that go extinct (terminate early), the one that lasts longest
    extincts = [i for i, L in enumerate(lengths) if L < sim_length]
    red_id = max(extincts, key=lambda i: lengths[i])

    figure, ax = plt.subplots(2, 1)
    figure.subplots_adjust(hspace=.0)

    # --- top panel: antibiotic (original teal, survivor highlighted) ---
    for j in range(N_SHADOW):
        ax[0].plot(tcbk_list[j][0], tcbk_list[j][2], color='gray', alpha=0.3, linewidth=1)
    ax[0].plot(tcbk_list[green_id][0], tcbk_list[green_id][2], color=ANTIBIOTIC_COLOR)
    ax[0].set_ylabel('Antibiotic')
    ax[0].get_xaxis().set_ticks([])

    # --- bottom panel: population (green survivor + red extinct) ---
    for j in range(N_SHADOW):
        ts, ps = _positive(tcbk_list[j][0], tcbk_list[j][1])
        ax[1].plot(ts, ps, color='gray', alpha=0.3, linewidth=1)
    ts, ps = _positive(tcbk_list[green_id][0], tcbk_list[green_id][1])
    ax[1].plot(ts, ps, color=POP_SIZE_COLOR, linewidth=1.8, label='survival')
    ts, ps = _positive(tcbk_list[red_id][0], tcbk_list[red_id][1])
    ax[1].plot(ts, ps, color=EXTINCT_COLOR, linewidth=1.8, label='extinction')
    ax[1].set_ylabel(r'$P$')
    ax[1].set_xlabel('Time (h)')
    ax[1].set_yscale('log')
    ax[1].legend(frameon=False, fontsize=8, loc='lower left')

    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    figure.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(figure)
    print(f"{os.path.basename(out_name)}: survivor=trial_{green_id} (final pop {finals[green_id]:.3g}), "
          f"extinct=trial_{red_id} (survived {lengths[red_id]}/{sim_length} steps)")


# %%
for cfg in CONFIGS:
    param, pct = cfg.split("_")  # e.g. "beta", "-20pct"
    folder = RESULTS_DIR / f"a{ANTIBIOTIC:.2f}_T{T_KN0}_{cfg}_value_check" / f"{INIT_APP}_{HALF_PERIOD}"
    loaded = load_logger_data_new(folder, sim_length, max_pop, 100)
    out_name = FIG_DIR / f"a{ANTIBIOTIC:.2f}_T{T_KN0}_{cfg}_2panel_survext.jpg"
    plot_two_panel_survext(loaded, out_name)

# %%
