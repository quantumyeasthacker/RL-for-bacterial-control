"""Relative performance of the proteome-context agents vs the eval-time mutation probability.

Companion to:
    train_context_varenv_mutate.sbatch      (5 context configs x 3 reps; job 11572255)
    eval_context_varenv_mutate.sbatch       (agent eval: P_policy;      job 11578146)
    sim_baseline_constant_context.sbatch    (constant-antibiotic baseline: P_constant)

Metric (mirrors plot_mutate_gru.py / plot_mutation.py):
    For each (context config, rep, mutation rate):
        eval_log_cell = mean over trials of log10( time-averaged cell population )   (P_policy)
        sim_log_cell  = same, for the constant-antibiotic baseline                   (P_constant)
        log_diff      = sim_log_cell - eval_log_cell        (relative performance; higher = better)
    One point per (config, rate) = mean of log_diff over the 3 training reps.
    Error bars = std across those 3 reps (the independent-seed spread; reps are unseeded, so
    this is the run-to-run variability the effect must beat to be real).

Unlike plot_mutate_gru.py this averages over REPS, not over eval-configs: the context sweep
was evaluated in-distribution only (varenv T=6), with replication instead of breadth.

BASELINE NOTE: this reads the context experiment's own baseline_constant, which was generated
with max_pop = inf to match the agent evals (which all use max_pop = np.inf). The older
rnn_varenv_encdec_mutate_runs baseline used EnvConfig's default 1e11 cap, which truncates
runaway rollouts and understates log P_constant in 3% (rate 0) to 34% (rate 1) of trials --
a bias along the very axis plotted here. Do not substitute that baseline.

Two panels, each <= 4 series so color alone stays comfortable:
    (a) assay-frequency dose-response : no-context vs ctx10 / ctx50 / ctx200
    (b) staleness-input ablation      : no-context vs ctx50 vs ctx50-no-age

The script loads however many trial_*tcbk.pkl files exist per cell, so it runs on partial data
and again once the full eval/baseline runs complete.

Output:
    plotting/figures_pdf/context_mutate_relperf.pdf
    plotting/figures_jpg/context_mutate_relperf.jpg
    plotting/figures_jpg/context_mutate_relperf.csv   (the plotted numbers)
"""

# %%
import os
import re
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import load_logger_data_new

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# %% ----- ----- ----- ----- experiment definition ----- ----- ----- ----- %% #
RUN_BASE = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/context_varenv_mutate_runs")
EVAL_BASE = RUN_BASE / "results_eval"           # agent eval output (eval sbatch default)
BASELINE_BASE = RUN_BASE / "baseline_constant"  # constant-antibiotic baseline, max_pop = inf

ANTIBIOTIC = 3.72
TRAINED_ENV = "T6"
DELAY = 30
TRAIN_MUTPROB = "0.1"
EVAL_ENV, EVAL_VAR = "varenv", "6"
CHECKPOINT = "episode_399"
BASELINE_HALF_PERIOD = 0
REPS = [0, 1, 2]

# context configs: (ctx_tag as it appears in the trial name, legend label)
CONFIGS = [
    ("",             "no context"),
    ("_ctx10",       "context, refresh 10"),
    ("_ctx50",       "context, refresh 50"),
    ("_ctx200",      "context, refresh 200"),
    ("_ctx50noage",  "context, refresh 50 (no age)"),
]

# Categorical slots 1-5 in fixed order (validated: worst adjacent CVD dE 9.1, normal-vision
# dE 19.6 on the light surface). Assigned in the sweep's semantic order so the legend reads
# control -> increasingly stale -> ablation; the ORDER carries the meaning, not the hue.
COLORS = {
    "":            "#2a78d6",   # slot 1 blue
    "_ctx10":      "#eb6834",   # slot 2 orange
    "_ctx50":      "#1baf7a",   # slot 3 aqua
    "_ctx200":     "#eda100",   # slot 4 yellow
    "_ctx50noage": "#e87ba4",   # slot 5 magenta
}
# secondary encoding (required at >=4 series): distinct marker per series, and the ablation
# dashed so it reads as a variant of the ctx50 condition it is compared against.
MARKERS = {"": "o", "_ctx10": "s", "_ctx50": "^", "_ctx200": "D", "_ctx50noage": "v"}
LINESTYLES = {"": "-", "_ctx10": "-", "_ctx50": "-", "_ctx200": "-", "_ctx50noage": "--"}

PANELS = [
    ("Assay-frequency dose-response", ["", "_ctx10", "_ctx50", "_ctx200"]),
    ("Staleness-input ablation",      ["", "_ctx50", "_ctx50noage"]),
]

# chart chrome (light surface)
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# mutation rates: reproduce the sbatch folder tokens EXACTLY (str(float(f"{x:.4f}")))
MUTPROBS = [float(f"{x:.4f}") for x in np.linspace(0, 1, 10)]

# %% ----- ----- ----- ----- analysis params (match plot_mutate_gru.py) ----- ----- ----- ----- %% #
delta_t = 0.2
init_len = 1
warm_up = 60
warm_up_embed = warm_up + init_len            # 61: drop warm-up + init phase
num_decisions = 300
sim_length = num_decisions + warm_up_embed    # 361
N_TRIALS_EVAL = 10                            # eval_step reps per eval cell
N_TRIALS_BASE = 100                           # baseline rollouts per cell
# Diagnostic only: both arms run uncapped (max_pop = inf), so this never truncates anything
# here; it is reported so a capped dataset cannot be pasted in unnoticed.
BLOWUP_THRESHOLD = 1e11


# %% ----- ----- ----- ----- helpers ----- ----- ----- ----- %% #
def available_n_trials(folder: Path, cap: int) -> int:
    """Highest contiguous trial index present (+1), capped. 0 if none/missing."""
    if not folder.is_dir():
        return 0
    files = glob.glob(str(folder / "trial_*tcbk.pkl"))
    idxs = [int(re.search(r"trial_(\d+)tcbk", os.path.basename(f)).group(1)) for f in files]
    return min(max(idxs) + 1, cap) if idxs else 0


def log_cell_stats(folder: Path, cap: int):
    """(mean, std, n, frac_blowup) of per-trial log10(time-averaged population) over the
    decision phase. Returns (nan, nan, 0, nan) if no trials are present."""
    n = available_n_trials(folder, cap)
    if n == 0:
        return np.nan, np.nan, 0, np.nan
    _, _, _, _, cell_array, _ = load_logger_data_new(
        str(folder) + "/", sim_length, np.inf, n_trials=n, RS=False
    )
    decision_phase = cell_array[:, warm_up_embed:]
    per_trial = np.log10(decision_phase.mean(axis=1))
    frac_blowup = float((decision_phase.max(axis=1) >= BLOWUP_THRESHOLD).mean())
    return per_trial.mean(), per_trial.std(), n, frac_blowup


def trial_name(ctx_tag: str, rep: int) -> str:
    return (f"a{ANTIBIOTIC:.2f}_{TRAINED_ENV}_delay{DELAY}"
            f"_mutprob{TRAIN_MUTPROB}_MLP{ctx_tag}_rep{rep}")


def eval_folder(ctx_tag: str, rep: int, mutprob: float) -> Path:
    return EVAL_BASE / f"{trial_name(ctx_tag, rep)}_{EVAL_ENV}_{EVAL_VAR}_mutprob{mutprob}" / CHECKPOINT


def baseline_folder(mutprob: float) -> Path:
    name = f"a{ANTIBIOTIC:.2f}_{EVAL_ENV}_{EVAL_VAR}_mutprob{mutprob}_value_check"
    return BASELINE_BASE / name / f"constant_{BASELINE_HALF_PERIOD}"


# %% ----- ----- ----- ----- load + compute relative performance ----- ----- ----- ----- %% #
# baseline (P_constant) is config- and rep-independent: one value per mutation rate
baseline_log, baseline_blowup = {}, {}
for mp in MUTPROBS:
    bf = baseline_folder(mp)
    mean, _, n, blow = log_cell_stats(bf, N_TRIALS_BASE)
    if n == 0:
        print(f"WARNING: missing baseline {bf}")
    else:
        print(f"  baseline mutprob={mp:<7} n={n:<4} logP={mean:6.3f}  blow-up frac={blow:.2f}")
    baseline_log[mp] = mean
    baseline_blowup[mp] = blow

# agent eval (P_policy) -> per-rep log_diff
# diffs[ctx_tag][mutprob] = list of per-rep log_diff (len up to len(REPS))
diffs = {tag: {mp: [] for mp in MUTPROBS} for tag, _ in CONFIGS}
for tag, label in CONFIGS:
    for rep in REPS:
        for mp in MUTPROBS:
            ef = eval_folder(tag, rep, mp)
            mean, _, n, _ = log_cell_stats(ef, N_TRIALS_EVAL)
            if n == 0:
                print(f"WARNING: missing eval {ef}")
                continue
            sim = baseline_log[mp]
            if np.isnan(mean) or np.isnan(sim):
                continue
            diffs[tag][mp].append(sim - mean)   # log_diff = log P_constant - log P_policy

# %% ----- ----- ----- ----- plot ----- ----- ----- ----- %% #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

for ax, (panel_title, tags) in zip(axes, PANELS):
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    for tag in tags:
        label = dict(CONFIGS)[tag]
        xs, ys, errs = [], [], []
        for mp in MUTPROBS:
            d = diffs[tag][mp]
            if len(d) == 0:
                continue
            xs.append(mp)
            ys.append(np.mean(d))
            errs.append(np.std(d))
        if not xs:
            print(f"No data for {label}; skipping.")
            continue
        ax.errorbar(xs, ys, yerr=errs, marker=MARKERS[tag], linestyle=LINESTYLES[tag],
                    capsize=3, linewidth=2, markersize=6, color=COLORS[tag], label=label,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.8, zorder=3)

    ax.set_xlabel("Eval-time mutation probability", fontsize=12, color=INK_2)
    ax.set_title(panel_title, fontsize=11, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(labelsize=10, colors=MUTED)
    # legend is always present (identity is never carried by color alone); three of the five
    # slots sit below 3:1 on the light surface, so the visible legend is also the relief.
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_2, loc="best")

axes[0].set_ylabel(r"Rel. performance,  $\log P_{constant}-\log P_{policy}$",
                   fontsize=12, color=INK_2)
fig.suptitle(
    f"Slow proteome context under mutation  ({EVAL_ENV} T={EVAL_VAR}, {CHECKPOINT}, "
    f"trained at mutate_prob={TRAIN_MUTPROB}; error bars = std over {len(REPS)} training reps)",
    fontsize=10, color=INK_2, y=1.02)
fig.tight_layout()

# %% ----- ----- ----- ----- save ----- ----- ----- ----- %% #
PLOT_DIR = Path(__file__).resolve().parent
for sub, ext in (("figures_pdf", "pdf"), ("figures_jpg", "jpg")):
    out_dir = PLOT_DIR / sub
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir / f"context_mutate_relperf.{ext}", dpi=600, bbox_inches="tight")
    print(f"saved {out_dir / f'context_mutate_relperf.{ext}'}")

csv_path = PLOT_DIR / "figures_jpg" / "context_mutate_relperf.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["context_config", "mutate_prob", "n_reps",
                "relperf_mean", "relperf_std", "baseline_logP", "baseline_blowup_frac"])
    for tag, label in CONFIGS:
        for mp in MUTPROBS:
            d = diffs[tag][mp]
            if len(d) == 0:
                continue
            w.writerow([label, mp, len(d), f"{np.mean(d):.6f}", f"{np.std(d):.6f}",
                        f"{baseline_log[mp]:.6f}", f"{baseline_blowup[mp]:.4f}"])
print(f"saved {csv_path}")

plt.show()
# %%
