#!/usr/bin/env python
"""Aggregate + visualize the bacterial-parameter-mismatch eval outputs (run_all_bactparam.py).

Walks results/eval_bactparam/, computes per-episode control-performance metrics from each
info['log'], aggregates over the 10 reps (and over conditions) per (group, alpha, beta, sigma),
and writes summary CSVs + figures showing how control degrades as the true bacterial physiology
is mis-specified relative to what the agent was trained on:
  * alpha x beta heatmaps (at sigma=1.0) per env group -- localize where control transitions,
  * an alpha/beta-ratio collapse scatter -- tests whether the drug's net effect is governed by
    the ratio (points from different (alpha, beta) pairs at equal ratio should coincide), and
  * a sigma marginal line plot (at alpha=beta=1.0).

Metrics (per episode, control phase = after warm-up), identical to analyze_noisy.py:
  log10_final  -- log10 final population (lower = better control)
  log10_max    -- log10 peak population during control
  burden       -- mean log10 population over control (time-averaged bacterial load)
  cum_reward   -- (ln N_final - ln N_start)/dt, the agent's own objective (lower = better)
  extinct      -- population driven to 0
  drug_frac    -- fraction of decisions with drug on

Run from repo root:  python scripts/eval/analyze_bactparam.py
"""
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

EVAL_ROOT = "results/eval_bactparam"
OUT = os.path.join(EVAL_ROOT, "analysis")

GROUP_FROM_DIR = {
    "results_delay_30_record_generalized": "gen",
    "results_delay_30_record_constenv": "const",
    "results_delay_30_record_varenv": "var",
    "results_delay_30_record_controlenv": "control",
}
GROUP_ORDER = ["const", "var", "control", "gen->const", "gen->var"]

TAG_RE = re.compile(r"a([\d.]+)_b([\d.]+)_s([\d.]+)")

# Full control-phase horizon (decisions after warm-up); the evals fix --num-decisions 300,
# so a non-extinct episode logs exactly this many control steps. Used to pad early-terminating
# (extinct) trajectories out to a common length for the paper-consistent mean-population metric.
NUM_DECISIONS = 300


def episode_metrics(info):
    L = np.asarray(info["log"], dtype=float)
    w = int(info["warm_up"])
    dt = float(info["delta_t"])
    t, b, N = L[:, 0], L[:, 2], L[:, 3]
    N_final = N[-1]
    N_start = N[w] if len(N) > w else N[0]
    ctrl = N[w + 1:]
    drug = b[w + 1:]
    Nf, Ns = max(N_final, 1.0), max(N_start, 1.0)
    # log_mean_pop: mean population the way the main figures report it (plotting/plot_gen.py
    # eval_log_cell). Hold the last logged value out to the full horizon -- extinction logs a
    # final 0, so extinct steps pull the *linear* time-mean toward 0 -- then log10. max_pop is
    # inf in these evals so the only early stop is extinction (last value 0), where last-point
    # padding equals zero padding. Floored at 1 like the other log metrics to stay finite.
    if ctrl.size:
        padded = np.pad(ctrl, (0, max(0, NUM_DECISIONS - ctrl.size)), mode="edge")[:NUM_DECISIONS]
        log_mean_pop = float(np.log10(max(padded.mean(), 1.0)))
    else:
        log_mean_pop = np.log10(Nf)
    return dict(
        extinct=float(N_final == 0),
        ext_time=(float(t[-1]) if N_final == 0 else np.nan),
        log10_final=np.log10(Nf),
        log10_max=(np.log10(np.clip(ctrl, 1, None).max()) if ctrl.size else np.log10(Nf)),
        log_mean_pop=log_mean_pop,
        burden=(float(np.mean(np.log10(np.clip(ctrl, 1, None)))) if ctrl.size else np.log10(Ns)),
        cum_reward=(np.log(Nf) - np.log(Ns)) / dt,
        drug_frac=(float(np.mean(drug > 0)) if drug.size else 0.0),
        n_decisions=len(L) - 1 - w,
    )


def parse_and_load(path):
    rel = os.path.relpath(path, EVAL_ROOT)
    env_group, eval_suffix, episode, tag, _ = rel.split(os.sep)
    env_type = GROUP_FROM_DIR[env_group]
    m = TAG_RE.search(tag)
    alpha_mult, beta_mult, sigma_mult = float(m.group(1)), float(m.group(2)), float(m.group(3))
    rep = int(re.search(r"_rep(\d+)", eval_suffix).group(1))

    if env_type == "gen":
        mm = re.search(r"_(constenv|varenv)_([\d.]+)$", eval_suffix)
        group = "gen->const" if mm.group(1) == "constenv" else "gen->var"
        condition = mm.group(2)
    elif env_type == "const":
        group, condition = "const", re.search(r"_n([\d.]+)_delay", eval_suffix).group(1)
    elif env_type == "var":
        group, condition = "var", re.search(r"_T(\d+)_delay", eval_suffix).group(1)
    else:
        group, condition = "control", re.search(r"_n([\d_]+)_b", eval_suffix).group(1)

    meta = dict(group=group, condition=condition, alpha_mult=alpha_mult,
                beta_mult=beta_mult, sigma_mult=sigma_mult, rep=rep)
    try:
        info = pickle.load(open(path, "rb"))
        m = episode_metrics(info)
    except Exception as e:  # keep going; flag bad files
        return {**meta, "error": str(e)}
    m.update(meta)
    return m


def heatmap_grid_ab(env_summary, value, title, fname, fmt=".2f", cmap="viridis", center=None):
    """alpha (rows) x beta (cols) heatmap per env group, at sigma=1.0."""
    df = env_summary[env_summary.sigma_mult == 1.0]
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), squeeze=False)
    for ax, g in zip(axes[0], groups):
        piv = (df[df.group == g]
               .pivot(index="alpha_mult", columns="beta_mult", values=value)
               .sort_index(ascending=False))
        sns.heatmap(piv, ax=ax, annot=True, fmt=fmt, cmap=cmap, center=center,
                    cbar=True, linewidths=.5, annot_kws={"size": 8})
        ax.set_title(g, fontsize=11)
        ax.set_xlabel("beta multiplier")
        ax.set_ylabel("alpha multiplier")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


def ratio_collapse(env_summary, value, ylabel, fname):
    """Scatter metric vs alpha/beta ratio (sigma=1.0). If the drug's net effect is governed by
    the ratio, points from different (alpha, beta) pairs at equal ratio coincide onto one curve."""
    df = env_summary[env_summary.sigma_mult == 1.0].copy()
    df["ratio"] = df.alpha_mult / df.beta_mult
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for g in groups:
        s = df[df.group == g].sort_values("ratio")
        ax.plot(s.ratio, s[value], "o", alpha=.7, label=g)
    ax.set_xlabel("alpha / beta  multiplier ratio  (drug damage / repair)")
    ax.set_ylabel(ylabel)
    ax.axvline(1.0, color="k", lw=.8, ls="--", alpha=.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{ylabel} vs alpha/beta ratio  (collapse -> ratio governs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


def sigma_lines(env_summary, value, ylabel, fname):
    """metric vs sigma multiplier (at alpha=beta=1.0) per env group."""
    df = env_summary[(env_summary.alpha_mult == 1.0) & (env_summary.beta_mult == 1.0)]
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for g in groups:
        s = df[df.group == g].sort_values("sigma_mult")
        ax.plot(s.sigma_mult, s[value], "o-", label=g)
    ax.set_xlabel("sigma multiplier (damage-noise strength), alpha=beta=1.0")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{ylabel} vs damage-noise strength", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = glob.glob(os.path.join(EVAL_ROOT, "*", "*", "*", "a*_b*_s*", "*tcbk.pkl"))
    print(f"found {len(files)} episode pkls; loading ...")
    rows = Parallel(n_jobs=24, verbose=1)(delayed(parse_and_load)(f) for f in files)
    df = pd.DataFrame(rows)
    bad = df[df.get("error").notna()] if "error" in df else df.iloc[0:0]
    if len(bad):
        print(f"WARNING: {len(bad)} files failed to load")
    df = df[df.get("error").isna()] if "error" in df else df

    df.to_csv(os.path.join(OUT, "episode_metrics.csv"), index=False)

    metrics = ["log10_final", "log10_max", "log_mean_pop", "burden", "cum_reward", "extinct", "drug_frac", "ext_time"]
    agg = {m: "mean" for m in metrics}
    agg["rep"] = "count"
    keys = ["alpha_mult", "beta_mult", "sigma_mult"]

    by_cond = (df.groupby(["group", "condition"] + keys).agg(agg)
               .rename(columns={"rep": "n_episodes"}).reset_index())
    by_cond.to_csv(os.path.join(OUT, "summary_by_condition.csv"), index=False)

    by_env = (df.groupby(["group"] + keys).agg(agg)
              .rename(columns={"rep": "n_episodes"}).reset_index())

    # degradation relative to the nominal (1,1,1) baseline, per group
    base = (by_env[(by_env.alpha_mult == 1.0) & (by_env.beta_mult == 1.0) & (by_env.sigma_mult == 1.0)]
            .set_index("group")[["log10_final", "log_mean_pop", "burden", "cum_reward"]])
    by_env["d_log10_final"] = by_env.apply(lambda r: r.log10_final - base.loc[r.group, "log10_final"], axis=1)
    by_env["d_log_mean_pop"] = by_env.apply(lambda r: r.log_mean_pop - base.loc[r.group, "log_mean_pop"], axis=1)
    by_env["d_burden"] = by_env.apply(lambda r: r.burden - base.loc[r.group, "burden"], axis=1)
    by_env["d_cum_reward"] = by_env.apply(lambda r: r.cum_reward - base.loc[r.group, "cum_reward"], axis=1)
    by_env.to_csv(os.path.join(OUT, "summary_by_env.csv"), index=False)

    # figures: alpha x beta grid (sigma=1)
    heatmap_grid_ab(by_env, "log10_final", "Final population  log10(N_final)  [lower = better control]",
                    "heat_log10_final.png", fmt=".2f", cmap="rocket_r")
    heatmap_grid_ab(by_env, "d_log10_final", "Degradation vs nominal:  log10(N_final) - baseline",
                    "heat_delta_log10_final.png", fmt="+.2f", cmap="RdBu_r", center=0)
    heatmap_grid_ab(by_env, "log_mean_pop", "Mean population  log10(mean_t N)  [paper convention, lower = better]",
                    "heat_log_mean_pop.png", fmt=".2f", cmap="rocket_r")
    heatmap_grid_ab(by_env, "d_log_mean_pop", "Degradation vs nominal:  log10(mean_t N) - baseline",
                    "heat_delta_log_mean_pop.png", fmt="+.2f", cmap="RdBu_r", center=0)
    heatmap_grid_ab(by_env, "extinct", "Extinction fraction", "heat_extinct.png", fmt=".2f", cmap="viridis")
    heatmap_grid_ab(by_env, "drug_frac", "Drug-on fraction", "heat_drug_frac.png", fmt=".2f", cmap="mako")
    # ratio hypothesis + sigma marginal
    ratio_collapse(by_env, "log10_final", "log10(N_final)", "ratio_log10_final.png")
    ratio_collapse(by_env, "log_mean_pop", "log10(mean_t N)  [paper convention]",
                   "ratio_log_mean_pop.png")
    ratio_collapse(by_env, "extinct", "extinction fraction", "ratio_extinct.png")
    sigma_lines(by_env, "log10_final", "log10(N_final)", "sigma_log10_final.png")

    # text headline: nominal vs the adversarial alpha x beta corner (alpha 0.8, beta 1.2)
    adv = (0.8, 1.2)
    print("\n=== headline: log10(final pop), nominal (1,1) vs adversarial corner "
          "(alpha %.1f, beta %.1f) ===" % adv)
    print(f"{'group':12s} {'nominal':>8s} {'adv':>8s} {'delta':>8s}  {'mp_nom':>8s} {'mp_adv':>8s}"
          f"  {'ext_nom':>8s} {'ext_adv':>8s}   (mp = log_mean_pop)")
    g_env = by_env[by_env.sigma_mult == 1.0]
    for g in GROUP_ORDER:
        s = g_env[g_env.group == g]
        if not len(s):
            continue
        c = s[(s.alpha_mult == 1.0) & (s.beta_mult == 1.0)]
        w = s[(s.alpha_mult == adv[0]) & (s.beta_mult == adv[1])]
        if not len(c) or not len(w):
            continue
        c, w = c.iloc[0], w.iloc[0]
        print(f"{g:12s} {c.log10_final:8.2f} {w.log10_final:8.2f} {w.log10_final-c.log10_final:+8.2f}"
              f"  {c.log_mean_pop:8.2f} {w.log_mean_pop:8.2f}"
              f"  {c.extinct:8.2f} {w.extinct:8.2f}")
    print(f"\nwrote CSVs + figures to {OUT}/")


if __name__ == "__main__":
    main()
