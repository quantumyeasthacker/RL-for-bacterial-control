#!/usr/bin/env python
"""Fair bacterial-parameter-mismatch analysis: learned policy vs. the best fixed protocol run
on the SAME perturbed bacteria.

analyze_bactparam.py scores the RL agent against its own nominal (1,1,1) run, which conflates two
effects: the agent coping worse with a shifted bacterium vs. the drug simply being intrinsically
weaker on that bacterium (any controller would look worse). This script removes that confound by
loading the fixed-protocol baselines (run_all_baseline_bactparam.py) evaluated on the identical
(alpha, beta, sigma) cells and reporting, per cell, the learned policy's ADVANTAGE over the best
non-learned protocol facing the same physiology:

    advantage(a,b,s) = min_fixed  log10 mean-pop[fixed protocol]  -  log10 mean-pop[RL]

If a parameter shift only weakens the drug, both terms move together and the advantage is
preserved -- the statement the mismatch experiment is really trying to make.

Baselines depend only on (env dynamics, nutrient condition), so the const/var baselines are shared
between the specialized (const/var) and generalized (gen->const/gen->var) RL groups via GROUP_TO_
FAMILY; the join key is (family, condition, a, b, s). "Best fixed" excludes the no_drug growth
ceiling: const/var -> {constant, pulse_h*}, control -> {feast, famine}.

Metrics come from analyze_bactparam.episode_metrics (imported), so RL and baseline pkls are scored
through one code path. log_mean_pop is the paper-convention mean population (lower = better).

Outputs (results/eval_bactparam/analysis_fair/):
  baseline_summary.csv    per (family, condition, policy, a, b, s): mean metrics over reps
  fair_comparison.csv     per (group, condition, a, b, s): RL vs best-fixed + advantage
  heat_advantage.png      alpha x beta advantage heatmap per group (sigma=1)
  heat_advantage_ext.png  alpha x beta extinction-advantage heatmap (RL - best fixed)
  marginal_<group>.png    alpha- and beta-marginal overlays: RL vs each baseline (lines move
                          together = "drug got weaker", RL line diverging = RL-specific change)

Run from repo root (after both eval sweeps have populated their trees):
  python scripts/eval/analyze_bactparam_fair.py
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

# reuse the RL analyzer's per-episode metric + RL loader so scoring can't drift
from analyze_bactparam import episode_metrics, parse_and_load, EVAL_ROOT, GROUP_ORDER

BASELINE_ROOT = "results/eval_bactparam_baseline"
OUT = os.path.join(EVAL_ROOT, "analysis_fair")

# RL group -> baseline env family (const/var baselines are shared with the gen evals)
GROUP_TO_FAMILY = {
    "const": "const", "gen->const": "const",
    "var": "var", "gen->var": "var",
    "control": "control",
}
# protocols that count as a "fixed strategy facing the same bacteria" (exclude the no_drug ceiling)
FIXED_POLICIES = {
    "const": lambda pol: pol == "constant" or pol.startswith("pulse_h"),
    "var": lambda pol: pol == "constant",
    "control": lambda pol: pol in ("feast", "famine"),
}
KEYS = ["alpha_mult", "beta_mult", "sigma_mult"]


def load_baseline_file(path):
    rel = os.path.relpath(path, BASELINE_ROOT)
    family_env, cond, policy, tag, _ = rel.split(os.sep)
    family = family_env[:-3]  # strip "env"
    m = re.search(r"a([\d.]+)_b([\d.]+)_s([\d.]+)", tag)
    meta = dict(family=family, condition=cond, policy=policy,
                alpha_mult=float(m.group(1)), beta_mult=float(m.group(2)),
                sigma_mult=float(m.group(3)))
    try:
        info = pickle.load(open(path, "rb"))
        return {**meta, **episode_metrics(info)}
    except Exception as e:
        return {**meta, "error": str(e)}


def load_rl():
    files = glob.glob(os.path.join(EVAL_ROOT, "*", "*", "*", "a*_b*_s*", "*tcbk.pkl"))
    print(f"RL: {len(files)} pkls")
    rows = Parallel(n_jobs=24, verbose=0)(delayed(parse_and_load)(f) for f in files)
    df = pd.DataFrame(rows)
    df = df[df.get("error").isna()] if "error" in df else df
    agg = {"log_mean_pop": "mean", "log10_final": "mean", "extinct": "mean", "rep": "count"}
    return (df.groupby(["group", "condition"] + KEYS).agg(agg)
            .rename(columns={"rep": "n"}).reset_index())


def load_baselines():
    files = glob.glob(os.path.join(BASELINE_ROOT, "*", "*", "*", "a*_b*_s*", "*tcbk.pkl"))
    print(f"baselines: {len(files)} pkls")
    rows = Parallel(n_jobs=24, verbose=0)(delayed(load_baseline_file)(f) for f in files)
    df = pd.DataFrame(rows)
    df = df[df.get("error").isna()] if "error" in df else df
    agg = {"log_mean_pop": "mean", "log10_final": "mean", "extinct": "mean"}
    return (df.groupby(["family", "condition", "policy"] + KEYS).agg(agg).reset_index())


def best_fixed(bl):
    """For each (family, condition, a, b, s), pick the fixed protocol with the lowest mean-pop
    (best suppression) and return its mean-pop + that protocol's extinction fraction + name."""
    mask = bl.apply(lambda r: FIXED_POLICIES[r.family](r.policy), axis=1)
    fx = bl[mask].copy()
    idx = fx.groupby(["family", "condition"] + KEYS)["log_mean_pop"].idxmin()
    best = fx.loc[idx, ["family", "condition"] + KEYS +
                  ["policy", "log_mean_pop", "extinct"]].reset_index(drop=True)
    return best.rename(columns={"policy": "best_policy",
                                "log_mean_pop": "base_log_mean_pop",
                                "extinct": "base_extinct"})


def build_comparison(rl, best):
    rl = rl.copy()
    rl["family"] = rl["group"].map(GROUP_TO_FAMILY)
    m = rl.merge(best, on=["family", "condition"] + KEYS, how="inner")
    m["advantage"] = m["base_log_mean_pop"] - m["log_mean_pop"]      # + = RL suppresses more
    m["ext_advantage"] = m["extinct"] - m["base_extinct"]           # + = RL extinguishes more
    return m


def heat_advantage(cmp, value, title, fname, center=0.0, fmt="+.2f", cmap="RdBu_r"):
    # average the per-cell advantage over the nutrient conditions in each group (a group like
    # gen->const spans k_n0 in {1..4}), so one alpha x beta cell maps to one number.
    df = (cmp[cmp.sigma_mult == 1.0]
          .groupby(["group", "alpha_mult", "beta_mult"], as_index=False)[value].mean())
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), squeeze=False)
    for ax, g in zip(axes[0], groups):
        piv = (df[df.group == g].pivot(index="alpha_mult", columns="beta_mult", values=value)
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


def marginal_overlays(rl, bl, cmp):
    """Per group: log_mean_pop vs alpha (beta=sigma=1) and vs beta (alpha=sigma=1), overlaying the
    RL policy against every baseline protocol on the same bacteria. Lines translating together =
    the drug's intrinsic potency changed; the RL line pulling away from the fixed protocols =
    a policy-specific effect."""
    groups = [g for g in GROUP_ORDER if g in cmp["group"].unique()]
    pairs = sorted({(g, c) for g in groups
                    for c in cmp[cmp.group == g]["condition"].unique()})
    for g, cond in pairs:
        fam = GROUP_TO_FAMILY[g]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), squeeze=True)
        for ax, axis, other in ((axes[0], "alpha_mult", "beta_mult"),
                                (axes[1], "beta_mult", "alpha_mult")):
            r = rl[(rl.group == g) & (rl.condition == cond) &
                   (rl[other] == 1.0) & (rl.sigma_mult == 1.0)].sort_values(axis)
            ax.plot(r[axis], r.log_mean_pop, "o-", color="#569122", lw=2.2, ms=6,
                    label="learned (RL)", zorder=5)
            b = bl[(bl.family == fam) & (bl.condition == cond) &
                   (bl[other] == 1.0) & (bl.sigma_mult == 1.0)]
            for pol, s in b.groupby("policy"):
                s = s.sort_values(axis)
                style = ":" if pol == "no_drug" else "-"
                ax.plot(s[axis], s.log_mean_pop, style, marker=".", alpha=.75, label=pol)
            ax.set_xlabel(f"{axis.replace('_mult', '')} multiplier "
                          f"({other.replace('_mult', '')}=sigma=1)")
            ax.set_ylabel("log10 mean population  (lower = better)")
            ax.grid(alpha=.3)
        axes[1].legend(fontsize=7, ncol=2, loc="best", framealpha=.9)
        fig.suptitle(f"{g}  (condition {cond}):  learned vs fixed protocols on the same bacteria",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe = f"{g.replace('->', '_')}_{str(cond).replace('.', 'p')}"
        fig.savefig(os.path.join(OUT, f"marginal_{safe}.png"), dpi=130)
        plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rl = load_rl()
    bl = load_baselines()
    bl.to_csv(os.path.join(OUT, "baseline_summary.csv"), index=False)

    best = best_fixed(bl)
    cmp = build_comparison(rl, best)
    cmp.to_csv(os.path.join(OUT, "fair_comparison.csv"), index=False)

    heat_advantage(cmp, "advantage",
                   "Learned advantage over best fixed protocol on the SAME bacteria\n"
                   "log10 mean-pop[best fixed] - log10 mean-pop[RL]   (>0: RL suppresses more)",
                   "heat_advantage.png")
    heat_advantage(cmp, "ext_advantage",
                   "Extinction advantage: P(extinct)[RL] - P(extinct)[best fixed]  (>0: RL better)",
                   "heat_advantage_ext.png", fmt="+.2f")
    marginal_overlays(rl, bl, cmp)

    # headline: nominal vs adversarial corner (alpha 0.8, beta 1.2), advantage preserved?
    # averaged over each group's nutrient conditions (see heat_advantage).
    adv = (0.8, 1.2)
    g_cmp = (cmp[cmp.sigma_mult == 1.0]
             .groupby(["group", "alpha_mult", "beta_mult"], as_index=False)
             [["advantage", "log_mean_pop", "base_log_mean_pop"]].mean())
    print("\n=== advantage over best fixed protocol: nominal (1,1) vs adversarial "
          "(alpha %.1f, beta %.1f) ===" % adv)
    print(f"{'group':12s} {'adv_nom':>8s} {'adv_corner':>11s}   "
          f"{'RL_mp_nom':>9s} {'base_mp_nom':>11s}   {'RL_mp_cor':>9s} {'base_mp_cor':>11s}")
    for g in GROUP_ORDER:
        s = g_cmp[g_cmp.group == g]
        c = s[(s.alpha_mult == 1.0) & (s.beta_mult == 1.0)]
        w = s[(s.alpha_mult == adv[0]) & (s.beta_mult == adv[1])]
        if not len(c) or not len(w):
            continue
        c, w = c.iloc[0], w.iloc[0]
        print(f"{g:12s} {c.advantage:8.2f} {w.advantage:11.2f}   "
              f"{c.log_mean_pop:9.2f} {c.base_log_mean_pop:11.2f}   "
              f"{w.log_mean_pop:9.2f} {w.base_log_mean_pop:11.2f}")
    print(f"\nwrote CSVs + figures to {OUT}/")


if __name__ == "__main__":
    main()
