"""Plot the average relative performance of the trained GRU agent as a function of the
eval-time mutation probability, for two checkpoints, with std error bars.

Companion to:
    eval_generalized_mutate_gru.sbatch   (agent eval: P_policy)
    sim_baseline_constant_mutate.sbatch  (constant-antibiotic baseline: P_constant)

Metric (mirrors plot_mutation.py):
    For each (eval-config, mutation-rate, checkpoint):
        eval_log_cell = mean over trials of log10( time-averaged cell population )   (P_policy)
        sim_log_cell  = same, for the constant-antibiotic baseline                   (P_constant)
        log_diff      = sim_log_cell - eval_log_cell        (relative performance; higher=better)
    "Average relative performance" averages log_diff over the 4 eval-configs
    (constenv c=1, c=3; varenv T=6, T=12), giving one curve per checkpoint.

Error bars: std of log_diff across the 4 eval-configs at each mutation rate (the spread of
the quantity being averaged). Set ERRBAR = "trial" below to instead use the pooled across-
trial std of eval_log_cell (matching plot_mutation.py's eval_log_cell_std).

The script loads however many trial_*tcbk.pkl files exist per cell, so it runs on partial
data and again after the full 100-trial eval/baseline runs complete.

Output:
    plotting/figures_pdf/mutate_gru_relperf.pdf
    plotting/figures_jpg/mutate_gru_relperf.jpg
"""

# %%
import os
import re
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import load_logger_data_new

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# %% ----- ----- ----- ----- experiment definition ----- ----- ----- ----- %% #
RUN_BASE = Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/rnn_varenv_encdec_mutate_runs")
EVAL_BASE = RUN_BASE / "results_eval"          # agent eval output (eval sbatch default)
BASELINE_BASE = RUN_BASE / "baseline_constant" # constant-antibiotic baseline (sim sbatch)

ANTIBIOTIC = 3.72
AGENT_TRIAL = "a3.72_T6_delay5_mutprob0.1_GRU_rep0"   # agent-eval folder prefix (trial_name)
BASELINE_HALF_PERIOD = 0                              # "constant" policy folder label

CHECKPOINTS = ["episode_250", "episode_399"]          # 26th and final
CKPT_LABELS = {"episode_250": "26th ckpt (episode_250)",
               "episode_399": "final ckpt (episode_399)"}
CKPT_COLORS = {"episode_250": "#a7c82f", "episode_399": "#548c6a"}

# eval configs averaged into each curve: (eval_env, eval_variable)
EVAL_CONFIGS = [("constenv", "1"), ("constenv", "3"), ("varenv", "6"), ("varenv", "12")]

# mutation rates: reproduce the sbatch folder tokens EXACTLY (str(float(f"{x:.4f}")))
MUTPROBS = [float(f"{x:.4f}") for x in np.linspace(0, 1, 10)]

ERRBAR = "config"   # "config" -> std across the 4 configs; "trial" -> pooled across-trial std

# %% ----- ----- ----- ----- analysis params (match plot_mutation.py) ----- ----- ----- ----- %% #
delta_t = 0.2
init_len = 1
warm_up = 60
warm_up_embed = warm_up + init_len          # 61: drop warm-up + init phase
num_decisions = 300
sim_length = num_decisions + warm_up_embed   # 361
max_pop = int(1e11)
N_TRIALS = 100


# %% ----- ----- ----- ----- helpers ----- ----- ----- ----- %% #
def available_n_trials(folder: Path) -> int:
    """Highest contiguous trial index present (+1), capped at N_TRIALS. 0 if none/missing."""
    if not folder.is_dir():
        return 0
    files = glob.glob(str(folder / "trial_*tcbk.pkl"))
    idxs = [int(re.search(r"trial_(\d+)tcbk", os.path.basename(f)).group(1)) for f in files]
    return min(max(idxs) + 1, N_TRIALS) if idxs else 0


def log_cell_stats(folder: Path):
    """(mean, std, n) of per-trial log10(time-averaged cell population) over the decision phase.
    Returns (nan, nan, 0) if no trials are present."""
    n = available_n_trials(folder)
    if n == 0:
        return np.nan, np.nan, 0
    _, _, _, _, cell_array, _ = load_logger_data_new(
        str(folder) + "/", sim_length, max_pop, n_trials=n, RS=False
    )
    per_trial = np.log10(cell_array[:, warm_up_embed:].mean(axis=1))
    return per_trial.mean(), per_trial.std(), n


def eval_folder(eval_env, eval_var, mutprob, checkpoint) -> Path:
    name = f"{AGENT_TRIAL}_{eval_env}_{eval_var}_mutprob{mutprob}"
    return EVAL_BASE / name / checkpoint


def baseline_folder(eval_env, eval_var, mutprob) -> Path:
    name = f"a{ANTIBIOTIC:.2f}_{eval_env}_{eval_var}_mutprob{mutprob}_value_check"
    return BASELINE_BASE / name / f"constant_{BASELINE_HALF_PERIOD}"


# %% ----- ----- ----- ----- load + compute relative performance ----- ----- ----- ----- %% #
# baseline (P_constant) is checkpoint-independent: one value per (config, mutprob)
baseline_log = {}   # (config_idx, mutprob) -> sim_log_cell
for ci, (env, var) in enumerate(EVAL_CONFIGS):
    for mp in MUTPROBS:
        bf = baseline_folder(env, var, mp)
        mean, _, n = log_cell_stats(bf)
        if n == 0:
            print(f"WARNING: missing baseline {bf}")
        baseline_log[(ci, mp)] = mean

# agent eval (P_policy) and log_diff, per (checkpoint, config, mutprob)
# curves[ckpt][mutprob] = list of per-config log_diff (len up to 4)
# trial_std[ckpt][mutprob] = list of per-config eval std (for ERRBAR == "trial")
curves = {ck: {mp: [] for mp in MUTPROBS} for ck in CHECKPOINTS}
trial_std = {ck: {mp: [] for mp in MUTPROBS} for ck in CHECKPOINTS}
for ck in CHECKPOINTS:
    for ci, (env, var) in enumerate(EVAL_CONFIGS):
        for mp in MUTPROBS:
            ef = eval_folder(env, var, mp, ck)
            mean, std, n = log_cell_stats(ef)
            if n == 0:
                print(f"WARNING: missing eval {ef}")
            sim = baseline_log[(ci, mp)]
            if np.isnan(mean) or np.isnan(sim):
                continue
            curves[ck][mp].append(sim - mean)   # log_diff = P_constant - P_policy
            trial_std[ck][mp].append(std)

# %% ----- ----- ----- ----- plot: 2 curves (one per checkpoint) ----- ----- ----- ----- %% #
fig, ax = plt.subplots(figsize=(6, 4.5))
for ck in CHECKPOINTS:
    xs, ys, errs = [], [], []
    for mp in MUTPROBS:
        diffs = curves[ck][mp]
        if len(diffs) == 0:
            continue
        xs.append(mp)
        ys.append(np.mean(diffs))
        if ERRBAR == "trial":
            errs.append(np.sqrt(np.mean(np.square(trial_std[ck][mp]))))  # pooled trial std
        else:
            errs.append(np.std(diffs))                                   # std across configs
    if not xs:
        print(f"No data for {ck}; skipping curve.")
        continue
    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, linewidth=1.5,
                color=CKPT_COLORS[ck], label=CKPT_LABELS[ck])

ax.set_xlabel("Mutation probability", fontsize=14)
ax.set_ylabel(r"Avg. relative performance, $\log P_{constant}-\log P_{policy}$", fontsize=12)
ax.tick_params(labelsize=12)
err_note = "std across configs" if ERRBAR == "config" else "pooled trial std"
ax.set_title(f"GRU agent {AGENT_TRIAL}\n(avg over {len(EVAL_CONFIGS)} eval envs; error bars = {err_note})",
             fontsize=10)
ax.legend(fontsize=11)
fig.tight_layout()

PLOT_DIR = Path(__file__).resolve().parent
for sub, ext in (("figures_pdf", "pdf"), ("figures_jpg", "jpg")):
    out_dir = PLOT_DIR / sub
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir / f"mutate_gru_relperf.{ext}", dpi=600, bbox_inches="tight")
    print(f"saved {out_dir / f'mutate_gru_relperf.{ext}'}")

plt.show()
# %%
