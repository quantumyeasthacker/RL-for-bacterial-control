"""Alternate (two-bar) version of plot_constenv_relperf_n2.py: shows ONLY the final checkpoint
(episode_399) of a3.72_n2.00_delay30_growth_rate_rep0 alongside the fixed reference bar, with
the bars relabeled "log pop size" (episode_399) and "exp growth rate" (reference).

Metric is identical to plot_constenv_relperf_n2.py:
    per-trial value = log10( mean over the decision phase of cell population )
    rel. perf.      = log P_constant - log P_policy      (higher = policy beats constant drug)
episode_399's value + error are recomputed from the rollout pkls; the reference bar is a fixed
constant (REF_VALUE +/- REF_ERR).

Input (produced by run_relperf_n2.sbatch):
    <BASE>/policy/episode_399/trial_*tcbk.pkl
    <BASE>/baseline_constant/trial_*tcbk.pkl

Output:
    plotting/figures_pdf/constenv_relperf_n2_alt.pdf
    plotting/figures_jpg/constenv_relperf_n2_alt.jpg
    (bars: "log pop size" = episode_399, "exp growth rate" = fixed reference)
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
POLICY_DIR = BASE / "policy" / "episode_399"
BASELINE_DIR = BASE / "baseline_constant"

# The two bars compare REWARD FUNCTIONS (reward_type), not observation type. This model
# (a3.72_n2.00_delay30_growth_rate_rep0) was trained with reward_type=log10_pop -> "log pop size".
# The reference is the original growth_rate-reward model -> "exp growth rate".
POLICY_LABEL = "log pop size"       # episode_399 bar (reward_type=log10_pop)
POLICY_COLOR = "#548c6a"
REF_LABEL = "exp growth rate"       # fixed reference bar (reward_type=growth_rate model)
# PLACEHOLDER value/error until the real growth_rate-reward results are available -- update both.
REF_VALUE = 3.4
REF_ERR = 0.1
REF_COLOR = "#8c8c8c"

# analysis params (match plot_constenv_relperf_n2.py) #
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
print(f"log P_constant = {P_constant:.4f}  log P_policy(episode_399) = {P_policy:.4f}")
print(f"'{POLICY_LABEL}' (episode_399): rel. perf. = {rel:.4f} +/- {sem:.4f}  (n={len(pol_vals)})")
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
ax.set_title("constant env n=2  (N=%d trials/condition)" % N_TRIALS, fontsize=10)
ax.tick_params(labelsize=11)
fig.tight_layout()

PLOT_DIR = Path(__file__).resolve().parent
for sub, ext in (("figures_pdf", "pdf"), ("figures_jpg", "jpg")):
    out_dir = PLOT_DIR / sub
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir / f"constenv_relperf_n2_alt.{ext}", dpi=600, bbox_inches="tight")
    print(f"saved {out_dir / f'constenv_relperf_n2_alt.{ext}'}")

plt.show()
# %%
