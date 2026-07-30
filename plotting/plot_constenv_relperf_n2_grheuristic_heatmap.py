"""Relative-performance HEATMAP for the growth-rate-threshold heuristic sweep on constant env n=2.
One cell per (g_high, g_low) combo over the sweep grid; cells with g_low >= g_high are invalid
(no hysteresis band) and left blank.

Metric (identical to the phi_S-heuristic relperf scripts):
    per-trial value = log10( mean over the decision phase of cell population )
    rel. perf.      = log P_constant - log P_policy      (higher = policy beats constant drug)
P_constant is the shared constant-drug baseline (baseline_constant); each combo's value is
recomputed from its rollout pkls. The learned-agent reference (fixed placeholder REF_VALUE) is
noted in the title for scale.

Input (per combo, produced by sim_grheuristic_constenv.sbatch):
    <BASE>/gr_heuristic_h<high>_l<low>/trial_*tcbk.pkl
    <BASE>/baseline_constant/trial_*tcbk.pkl        (P_constant)

Output:
    plotting/figures_pdf/constenv_relperf_n2_grheuristic_heatmap.pdf
    plotting/figures_jpg/constenv_relperf_n2_grheuristic_heatmap.jpg
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

# sweep grid (must match submit_grheuristic_constenv_sweep.sh)
HIGHS = [0.0, 0.5, 1.0]        # re-apply antibiotic once g > high
LOWS = [-1.0, -0.5, 0.0]      # remove antibiotic once g < low

REF_LABEL = "agent"           # learned growth_rate-reward agent (fixed reference)
REF_VALUE = 3.4               # PLACEHOLDER until real growth_rate-reward results are available

# analysis params (match plot_constenv_relperf_n2_phiSheuristic.py) #
warm_up_embed = 61                  # drop warm-up(60) + init(1)
sim_length = 300 + warm_up_embed    # 361
max_pop = int(1e11)
N_TRIALS = 100


def subdir(high, low):
    return f"gr_heuristic_h{high:g}_l{low:g}"


def per_trial_log_cell(folder: Path):
    _, _, _, _, cell_array, _ = load_logger_data_new(
        str(folder) + "/", sim_length, max_pop, n_trials=N_TRIALS, RS=False
    )
    return np.log10(cell_array[:, warm_up_embed:].mean(axis=1))


# %% ----- load + compute grid ----- %% #
base_vals = per_trial_log_cell(BASELINE_DIR)
P_constant = base_vals.mean()

# M[i, j] = rel. perf. for HIGHS[i] x LOWS[j]  (NaN = invalid band or missing data)
M = np.full((len(HIGHS), len(LOWS)), np.nan)
for i, high in enumerate(HIGHS):
    for j, low in enumerate(LOWS):
        if low >= high:
            continue  # invalid: no hysteresis band
        folder = BASE / subdir(high, low)
        try:
            pol_vals = per_trial_log_cell(folder)
        except (IndexError, FileNotFoundError):
            print(f"  missing data for high={high} low={low} ({folder.name}); leaving blank")
            continue
        M[i, j] = P_constant - pol_vals.mean()
        print(f"  high={high:>4g} low={low:>4g}  (n={len(pol_vals):3d})  rel. perf. = {M[i, j]:.4f}")

print(f"log P_constant = {P_constant:.4f}   '{REF_LABEL}' reference = {REF_VALUE}")

# %% ----- heatmap ----- %% #
Mm = np.ma.masked_invalid(M)
cmap = mpl.colormaps["cividis"].copy()
cmap.set_bad("0.9")   # invalid/missing cells -> light gray

fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(Mm, cmap=cmap, origin="upper", aspect="auto")

ax.set_xticks(range(len(LOWS)))
ax.set_xticklabels([f"{l:g}" for l in LOWS])
ax.set_yticks(range(len(HIGHS)))
ax.set_yticklabels([f"{h:g}" for h in HIGHS])
ax.set_xlabel(r"$g_{low}$  (remove drug once $g < g_{low}$)", fontsize=11)
ax.set_ylabel(r"$g_{high}$  (apply drug once $g > g_{high}$)", fontsize=11)
ax.set_title(r"growth-rate heuristic, constant env n=2 (N=%d)" "\n"
             r"rel. perf. $\log P_{constant}-\log P_{policy}$   ('%s' ref = %.1f)"
             % (N_TRIALS, REF_LABEL, REF_VALUE), fontsize=10)

# annotate each valid cell with its value, in a contrasting ink
vmin, vmax = Mm.min(), Mm.max()
for i in range(len(HIGHS)):
    for j in range(len(LOWS)):
        if np.ma.is_masked(Mm[i, j]):
            continue
        frac = (M[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if frac < 0.5 else "black")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("relative performance", fontsize=10)
fig.tight_layout()

PLOT_DIR = Path(__file__).resolve().parent
for sub, ext in (("figures_pdf", "pdf"), ("figures_jpg", "jpg")):
    out_dir = PLOT_DIR / sub
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir / f"constenv_relperf_n2_grheuristic_heatmap.{ext}",
                dpi=600, bbox_inches="tight")
    print(f"saved {out_dir / f'constenv_relperf_n2_grheuristic_heatmap.{ext}'}")

# %%
