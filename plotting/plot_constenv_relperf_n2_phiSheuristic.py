"""Relative-performance (two-bar) comparison for the phi_S-threshold HEURISTIC on constant env
n=2, styled after plot_constenv_relperf_n2_alt.py. The bars are:
    "phi_S heuristic" : the phi_S hysteresis controller (measured from its rollout pkls)
    "exp growth rate" : the learned growth_rate-reward agent, entered as a FIXED reference
                        (REF_VALUE +/- REF_ERR; placeholder until real results are available).

Metric (identical to plot_constenv_relperf_n2_alt.py):
    per-trial value = log10( mean over the decision phase of cell population )
    rel. perf.      = log P_constant - log P_policy      (higher = policy beats constant drug)
P_constant is the constant-drug baseline; the heuristic's value + error are recomputed from its
rollout pkls. P_constant is shared with the RL relperf figure (same baseline_constant dir).

Input:
    <BASE>/phiS_heuristic/trial_*tcbk.pkl      (from simulation_test_phiSheuristic_constenv.py)
    <BASE>/baseline_constant/trial_*tcbk.pkl   (from sim_baseline_constenv.py; P_constant)

Output:
    plotting/figures_pdf/constenv_relperf_n2_phiSheuristic.pdf
    plotting/figures_jpg/constenv_relperf_n2_phiSheuristic.jpg
    (bars: "phi_S heuristic" = measured, "exp growth rate" = fixed reference)
"""

# %%
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import load_logger_data_new

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# %% ----- experiment definition ----- %% #
# The heuristic subdir, figure-name tag, and thresholds come from the environment so a threshold
# sweep can render each combo separately; the defaults reproduce the original 0.2/0.03 run.
HEUR_SUBDIR = os.environ.get("HEUR_SUBDIR", "phiS_heuristic")
FIG_TAG = os.environ.get("FIG_TAG", "")
PHIS_HIGH = os.environ.get("PHIS_HIGH", "0.2")
PHIS_LOW = os.environ.get("PHIS_LOW", "0.03")

BASE = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/constenv_logpop_runs/relperf_n2")
POLICY_DIR = BASE / HEUR_SUBDIR
BASELINE_DIR = BASE / "baseline_constant"

POLICY_LABEL = r"$\phi_S$ heuristic"   # measured heuristic bar
POLICY_COLOR = "#216d87"               # teal, matching the phi_S-heuristic figures
# fixed reference bar (learned growth_rate-reward agent); label overridable via env (e.g. "agent")
REF_LABEL = os.environ.get("REF_LABEL", "exp growth rate")
# PLACEHOLDER value/error until the real growth_rate-reward agent results are available -- update
# both (kept identical to plot_constenv_relperf_n2_alt.py's REF_VALUE/REF_ERR).
REF_VALUE = 3.4
REF_ERR = 0.1
REF_COLOR = "#8c8c8c"

# analysis params (match plot_constenv_relperf_n2_alt.py) #
warm_up_embed = 61                  # drop warm-up(60) + init(1)
sim_length = 300 + warm_up_embed    # 361
max_pop = int(1e11)
N_TRIALS = 100


def per_trial_log_cell(folder: Path):
    _, _, _, _, cell_array, _ = load_logger_data_new(
        str(folder) + "/", sim_length, max_pop, n_trials=N_TRIALS, RS=False
    )
    return np.log10(cell_array[:, warm_up_embed:].mean(axis=1))


# %% ----- load + compute ----- %% #
base_vals = per_trial_log_cell(BASELINE_DIR)
pol_vals = per_trial_log_cell(POLICY_DIR)
P_constant = base_vals.mean()
P_policy = pol_vals.mean()
rel = P_constant - P_policy
sem = np.sqrt(pol_vals.var() / len(pol_vals) + base_vals.var() / len(base_vals))
print(f"log P_constant = {P_constant:.4f}  log P_policy(phi_S heuristic) = {P_policy:.4f}")
print(f"'{POLICY_LABEL}' (heuristic): rel. perf. = {rel:.4f} +/- {sem:.4f}  (n={len(pol_vals)})")
print(f"'{REF_LABEL}' (reference): {REF_VALUE} +/- {REF_ERR}")

# %% ----- bar chart ----- %% #
labels = [POLICY_LABEL, REF_LABEL]
ys = [rel, REF_VALUE]
errs = [sem, REF_ERR]
colors = [POLICY_COLOR, REF_COLOR]

fig, ax = plt.subplots(figsize=(5, 4.5))
xs = np.arange(len(labels))
ax.bar(xs, ys, yerr=errs, capsize=5, color=colors,
       edgecolor="black", linewidth=0.8, width=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel(r"Relative performance, $\log P_{constant}-\log P_{policy}$", fontsize=11)
ax.set_title(r"constant env n=2  ($\phi_S{>}%s$ off, ${<}%s$ on;  N=%d)"
             % (PHIS_HIGH, PHIS_LOW, N_TRIALS), fontsize=10)
ax.tick_params(labelsize=11)
fig.tight_layout()

PLOT_DIR = Path(__file__).resolve().parent
for sub, ext in (("figures_pdf", "pdf"), ("figures_jpg", "jpg")):
    out_dir = PLOT_DIR / sub
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir / f"constenv_relperf_n2_phiSheuristic{FIG_TAG}.{ext}", dpi=600, bbox_inches="tight")
    print(f"saved {out_dir / f'constenv_relperf_n2_phiSheuristic{FIG_TAG}.{ext}'}")

# %%
