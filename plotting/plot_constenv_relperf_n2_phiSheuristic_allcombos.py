"""Combined relative-performance bar chart for ALL phi_S-threshold heuristic combos on constant
env n=2, plus the learned agent reference bar. One bar per (phiS_high, phiS_low) combo, sorted by
performance, so the combos can be compared directly on a single axis.

Metric (identical to plot_constenv_relperf_n2_phiSheuristic.py):
    per-trial value = log10( mean over the decision phase of cell population )
    rel. perf.      = log P_constant - log P_policy      (higher = policy beats constant drug)
P_constant is the shared constant-drug baseline (baseline_constant); each combo's value + error
are recomputed from its rollout pkls. The agent bar is a fixed reference (REF_VALUE +/- REF_ERR;
placeholder until the real growth_rate-reward agent results are available).

Input (per combo, produced by sim_phiSheuristic_constenv.sbatch):
    <BASE>/<subdir>/trial_*tcbk.pkl
    <BASE>/baseline_constant/trial_*tcbk.pkl    (P_constant)

Output:
    plotting/figures_pdf/constenv_relperf_n2_phiSheuristic_allcombos.pdf
    plotting/figures_jpg/constenv_relperf_n2_phiSheuristic_allcombos.jpg
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
BASE = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/constenv_logpop_runs/relperf_n2")
BASELINE_DIR = BASE / "baseline_constant"

# every phi_S-heuristic combo evaluated so far: (trial subdir, phiS_high, phiS_low)
COMBOS = [
    ("phiS_heuristic",            0.2,  0.03),
    ("phiS_heuristic_h0.22_l0.02", 0.22, 0.02),
    ("phiS_heuristic_h0.2_l0.02",  0.2,  0.02),
    ("phiS_heuristic_h0.18_l0.02", 0.18, 0.02),
    ("phiS_heuristic_h0.18_l0.01", 0.18, 0.01),
    ("phiS_heuristic_h0.16_l0.02", 0.16, 0.02),
]

HEUR_COLOR = "#216d87"   # teal, matching the phi_S-heuristic figures
REF_LABEL = "agent"      # learned growth_rate-reward agent (fixed reference bar)
REF_VALUE = 3.4          # PLACEHOLDER until real growth_rate-reward results are available
REF_ERR = 0.1
REF_COLOR = "#8c8c8c"

# analysis params (match plot_constenv_relperf_n2_phiSheuristic.py) #
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
P_constant = base_vals.mean()

results = []  # (label, rel, sem)
for subdir, high, low in COMBOS:
    pol_vals = per_trial_log_cell(BASE / subdir)
    rel = P_constant - pol_vals.mean()
    sem = np.sqrt(pol_vals.var() / len(pol_vals) + base_vals.var() / len(base_vals))
    label = f"{high:g}/{low:g}"
    results.append((label, rel, sem))
    print(f"  {label:>10s}  (n={len(pol_vals):3d})  rel. perf. = {rel:.4f} +/- {sem:.4f}")

# sort heuristic combos best -> worst; agent reference bar goes last
results.sort(key=lambda r: r[1], reverse=True)
print(f"log P_constant = {P_constant:.4f}   '{REF_LABEL}' reference = {REF_VALUE} +/- {REF_ERR}")

labels = [r[0] for r in results] + [REF_LABEL]
ys = [r[1] for r in results] + [REF_VALUE]
errs = [r[2] for r in results] + [REF_ERR]
colors = [HEUR_COLOR] * len(results) + [REF_COLOR]

# %% ----- bar chart ----- %% #
fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 1.5, 4.5))
xs = np.arange(len(labels))
bars = ax.bar(xs, ys, yerr=errs, capsize=4, color=colors,
              edgecolor="black", linewidth=0.8, width=0.72)

# direct value labels above each bar (few bars -> exact values aid the comparison)
for rect, v, e in zip(bars, ys, errs):
    ax.annotate(f"{v:.2f}", xy=(rect.get_x() + rect.get_width() / 2, v + e),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel(r"Relative performance, $\log P_{constant}-\log P_{policy}$", fontsize=11)
ax.set_xlabel(r"$\phi_S$ heuristic thresholds (high/low)", fontsize=11)
ax.set_title("constant env n=2  (N=%d trials/condition)" % N_TRIALS, fontsize=11)
ax.tick_params(labelsize=10)
ax.set_ylim(0, max(ys) + max(errs) + 0.35)
fig.tight_layout()

PLOT_DIR = Path(__file__).resolve().parent
for sub, ext in (("figures_pdf", "pdf"), ("figures_jpg", "jpg")):
    out_dir = PLOT_DIR / sub
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir / f"constenv_relperf_n2_phiSheuristic_allcombos.{ext}",
                dpi=600, bbox_inches="tight")
    print(f"saved {out_dir / f'constenv_relperf_n2_phiSheuristic_allcombos.{ext}'}")

# %%
