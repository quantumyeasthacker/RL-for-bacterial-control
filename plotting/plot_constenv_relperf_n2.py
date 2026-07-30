"""Relative performance of the ORIGINAL constenv model a3.72_n2.00_delay30_growth_rate_rep0
(lr=1e-4) on the constant env n=2, for two checkpoints (episode_340 and episode_399), vs the
constant-antibiotic baseline. Single-config analogue of plot_mutate_gru.py.

Metric (identical formula to plot_mutate_gru.py / plot_mutation.py):
    per-trial value = log10( mean over the decision phase of cell population )
    P               = mean of per-trial value over trials
    rel. perf.      = log P_constant - log P_policy      (higher = policy beats constant drug)
The decision phase drops the first warm_up_embed = warm_up(60)+init_len(1) = 61 logged points.

Error bar per checkpoint: propagated across-trial std,
    sqrt( std(policy per-trial)^2 / n_pol + std(baseline per-trial)^2 / n_base ),
i.e. the SEM of the difference of the two trial means.

Input (produced by run_relperf_n2.sbatch):
    <BASE>/policy/episode_399/trial_*tcbk.pkl
    <BASE>/policy/episode_340/trial_*tcbk.pkl
    <BASE>/baseline_constant/trial_*tcbk.pkl

A third, hand-supplied reference bar (REF_VALUE +/- REF_ERR, default 3.4 +/- 0.2, the original-
model value) is appended for comparison; it is a fixed constant, not computed from data.

Output:
    plotting/figures_pdf/constenv_relperf_n2.pdf
    plotting/figures_jpg/constenv_relperf_n2.jpg
    (bars: episode_340, final episode_399, and the fixed reference)
    prints the numeric rel. perf. and the underlying log P values.
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
POLICY_DIRS = {
    "episode_340": BASE / "policy" / "episode_340",
    "episode_399": BASE / "policy" / "episode_399",
}
BASELINE_DIR = BASE / "baseline_constant"

CKPT_ORDER = ["episode_340", "episode_399"]
CKPT_LABELS = {"episode_340": "episode_340", "episode_399": "final (episode_399)"}
CKPT_COLORS = {"episode_340": "#a7c82f", "episode_399": "#548c6a"}

# fixed reference value for comparison (original-model value supplied by hand, not computed here)
REF_LABEL = "original model\n(reference)"
REF_VALUE = 3.4
REF_ERR = 0.2
REF_COLOR = "#8c8c8c"

# analysis params (match plot_mutate_gru.py) #
warm_up = 60
init_len = 1
warm_up_embed = warm_up + init_len          # 61: drop warm-up + init phase
num_decisions = 300
sim_length = num_decisions + warm_up_embed   # 361
max_pop = int(1e11)
N_TRIALS = 100


def per_trial_log_cell(folder: Path):
    """Per-trial log10(time-averaged cell population) over the decision phase. Returns 1D array."""
    _, _, _, _, cell_array, _ = load_logger_data_new(
        str(folder) + "/", sim_length, max_pop, n_trials=N_TRIALS, RS=False
    )
    return np.log10(cell_array[:, warm_up_embed:].mean(axis=1))


# %% ----- load + compute ----- %% #
base_vals = per_trial_log_cell(BASELINE_DIR)
P_constant = base_vals.mean()
print(f"baseline (constant a=3.72): log P_constant = {P_constant:.4f}  "
      f"(n={len(base_vals)}, trial std={base_vals.std():.4f})")

results = {}   # ckpt -> (rel_perf, sem, P_policy)
for ck in CKPT_ORDER:
    pol_vals = per_trial_log_cell(POLICY_DIRS[ck])
    P_policy = pol_vals.mean()
    rel = P_constant - P_policy
    sem = np.sqrt(pol_vals.var() / len(pol_vals) + base_vals.var() / len(base_vals))
    results[ck] = (rel, sem, P_policy)
    print(f"{ck}: log P_policy = {P_policy:.4f}  ->  rel. perf. = {rel:.4f} +/- {sem:.4f}  "
          f"(n={len(pol_vals)})")

# %% ----- bar chart ----- %% #
# two computed checkpoint bars + one hand-supplied reference bar (3.4 +/- 0.2)
labels = [CKPT_LABELS[ck] for ck in CKPT_ORDER] + [REF_LABEL]
ys = [results[ck][0] for ck in CKPT_ORDER] + [REF_VALUE]
errs = [results[ck][1] for ck in CKPT_ORDER] + [REF_ERR]
colors = [CKPT_COLORS[ck] for ck in CKPT_ORDER] + [REF_COLOR]

fig, ax = plt.subplots(figsize=(6, 4.5))
xs = np.arange(len(labels))
ax.bar(xs, ys, yerr=errs, capsize=5, color=colors,
       edgecolor="black", linewidth=0.8, width=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel(r"Relative performance, $\log P_{constant}-\log P_{policy}$", fontsize=11)
ax.set_title("Original model a3.72_n2.00_delay30_growth_rate_rep0 (lr=1e-4)\n"
             f"constant env n=2  (N={N_TRIALS} trials/condition)", fontsize=9)
ax.tick_params(labelsize=11)
fig.tight_layout()

PLOT_DIR = Path(__file__).resolve().parent
for sub, ext in (("figures_pdf", "pdf"), ("figures_jpg", "jpg")):
    out_dir = PLOT_DIR / sub
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir / f"constenv_relperf_n2.{ext}", dpi=600, bbox_inches="tight")
    print(f"saved {out_dir / f'constenv_relperf_n2.{ext}'}")

plt.show()
# %%
