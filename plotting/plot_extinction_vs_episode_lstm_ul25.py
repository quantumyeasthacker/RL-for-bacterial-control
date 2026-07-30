# %%
"""Plot mean extinction fraction vs. training episode for the LSTM encoder-decoder (unroll 25) model.

Standalone counterpart to plot_extinction_vs_episode.py, for the single already-trained model
a3.72_T6_delay1_LSTM_encdec_rep0 evaluated at train_unroll_len=25 (its ul=25 checkpoints are the
ones that survived on disk; no retraining was needed).

Consumes the per-checkpoint eval output produced by
scripts/eval/eval_rnn_varenv_lstm_encdec_ul25.sbatch: for each saved checkpoint (episode_<n>),
that sbatch writes 100 trial_*tcbk.pkl rollouts under
"{model_dir}_eval/a3.72_T6_delay1_LSTM_encdec_ul25_rep0/episode_<n>/".

For every checkpoint this script loads those rollouts, computes the extinction fraction (fraction
of trials whose final cell count is 0, matching plot_gen.py: extinction = 1 if tcbk[1, -1] == 0
else 0), and plots that mean fraction against the training-episode index, with a +/- 1 std band.

Run with the RL_bact env python (the pkls were written with its numpy):
    /storage/home/hcoda1/0/jkratz3/r-jkratz3-0/envs/RL_bact/bin/python \
        plot_extinction_vs_episode_lstm_ul25.py

Output (written under plotting/figures_jpg and plotting/figures_pdf):
    extinction_vs_episode_lstm_encdec_ul25.jpg / .pdf
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

from utils import load_logger_data_new


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# nutrient-style palette reused from plot_gen.py / utils.py
COLOR_LIST = ["#dec60c", "#548c6a", "#a7c82f", "#5e6b75"]

# %% ----- ----- ----- ----- config ----- ----- ----- ----- %% #
# Loader params. Extinction only uses the final cell count (tcbk[1, -1]); SIM_LENGTH just bounds
# the cell-trajectory padding in load_logger_data_new and does not affect the extinction fraction.
SIM_LENGTH = 1000
MAX_POP = int(1e11)
N_TRIALS = 100          # trials per checkpoint written by the sbatch (rep_eval 0..9 x 10)

# The single LSTM encoder-decoder model, evaluated at train_unroll_len=25.
MODELS = [
    {
        "label": "delay1 LSTM (encoder_decoder), unroll 25",
        "eval_dir": Path("/storage/project/r-sbanerjee347-0/jkratz3/pnas_rl/rnn_varenv_encdec_runs/results_eval/"
                         "a3.72_T6_delay1_LSTM_encdec_ul25_rep0"),
        "color": COLOR_LIST[0],
    },
]

OUT_NAME = "extinction_vs_episode_lstm_encdec_ul25"
FIG_DIR = Path(__file__).resolve().parent


# %% ----- ----- ----- ----- helpers ----- ----- ----- ----- %% #
def episode_dirs_sorted(eval_dir):
    """Yield (episode_int, path) for each episode_<n> checkpoint dir, ascending."""
    eps = []
    for name in os.listdir(eval_dir):
        m = re.fullmatch(r"episode_(\d+)", name)
        if m and (eval_dir / name).is_dir():
            eps.append((int(m.group(1)), eval_dir / name))
    return sorted(eps, key=lambda x: x[0])


def n_trials_present(folder):
    """Highest trial index + 1 present in folder (load_logger_data_new skips gaps)."""
    idxs = [int(m.group(1)) for f in os.listdir(folder)
            if (m := re.fullmatch(r"trial_(\d+)tcbk\.pkl", f))]
    return max(idxs) + 1 if idxs else 0


def extinction_stats(folder):
    """Mean and std of the per-trial extinction indicator (1 if final cell count == 0)."""
    n = min(N_TRIALS, n_trials_present(folder))
    if n == 0:
        return None
    tcbk_list, *_ = load_logger_data_new(folder, SIM_LENGTH, MAX_POP, n, RS=False)
    if not tcbk_list:
        return None
    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    return np.mean(extinction), np.std(extinction), len(extinction)


# %% ----- ----- ----- ----- gather + plot ----- ----- ----- ----- %% #
fig, ax = plt.subplots(figsize=(8, 6))

for model in MODELS:
    eval_dir = model["eval_dir"]
    if not eval_dir.is_dir():
        print(f"WARNING: eval dir not found, skipping: {eval_dir}")
        continue

    episodes, mean_ext, std_ext = [], [], []
    for ep_int, ep_path in episode_dirs_sorted(eval_dir):
        stats = extinction_stats(ep_path)
        if stats is None:
            print(f"WARNING: no trials in {ep_path}, skipping")
            continue
        m, s, n = stats
        episodes.append(ep_int)
        mean_ext.append(m)
        std_ext.append(s)
    if not episodes:
        print(f"WARNING: no usable checkpoints for {model['label']}")
        continue

    episodes = np.array(episodes)
    mean_ext = np.array(mean_ext)
    std_ext = np.array(std_ext)

    ax.plot(episodes, mean_ext, marker='o', ms=4, color=model["color"], label=model["label"])
    # extinction fraction is bounded to [0, 1]; clip the std band to that range
    ax.fill_between(episodes,
                    np.clip(mean_ext - std_ext, 0, 1),
                    np.clip(mean_ext + std_ext, 0, 1),
                    color=model["color"], alpha=0.2)

ax.set_xlabel("Training episode")
ax.set_ylabel("Mean extinction fraction")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Extinction fraction vs. training episode (antibiotic conc. = 3.72)")
ax.legend(loc="lower right", fontsize=10, title="model")
fig.tight_layout()

(FIG_DIR / "figures_jpg").mkdir(exist_ok=True)
(FIG_DIR / "figures_pdf").mkdir(exist_ok=True)
fig.savefig(FIG_DIR / "figures_jpg" / f"{OUT_NAME}.jpg", dpi=600, bbox_inches='tight')
fig.savefig(FIG_DIR / "figures_pdf" / f"{OUT_NAME}.pdf", dpi=600, bbox_inches='tight')
print(f"Saved: {FIG_DIR / 'figures_jpg' / (OUT_NAME + '.jpg')}")

# %%
