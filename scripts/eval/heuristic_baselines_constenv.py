"""Adaptive-heuristic baselines for the constant-nutrient (antibiotic-only) figure.

Beyond fixed-dose and periodic pulsing, we benchmark the learned policy against two
*adaptive* heuristics drawn (in form only) from the adaptive-therapy literature. Each
keeps a control archetype and drops the mechanism it was originally designed around:

  - threshold : reactive threshold feedback -- apply drug when the measured population
                P >= Omega, withhold below. A closed-loop rule keyed off RAW LOAD.
                (borrows the "treat while symptomatic / load is high" idea.)
  - frontload : open-loop front-loaded schedule -- apply drug for the first fraction
                tau of the horizon, then withhold. Keyed off the CLOCK, not the state.
                (borrows the "hit hardest early / decreasing dose" idea.)

Each heuristic has one free knob (Omega, tau), swept over a family of members. Following
the anti-strawman rule we report each family's most favorable member -- by default the best
member on the eval seed block (--holdout instead ranks on a disjoint tuning block and reports
the winner on the eval block, to defuse "tuned on the test metric").

With --sweep_eval every member is also evaluated on the eval block, which yields the full
sweep curves (mean log10 population vs. the family's knob) rather than just the winner.

Reference policies (not adaptive heuristics): no_drug (upper bound), constant (fixed dose),
pulsing (our hard-coded on/off baseline, swept over half-periods), learned (our DQN, read
from logged eval trajectories; the best training rep per nutrient is selected, matching the
min-selection idiom of plotting/plot_constant_record.py).

Metric matches Fig 2C (plotting/plot_gen.py): per trial, the population is zero-padded to the
full horizon (a trial that goes extinct counts as population 0 for the remaining steps -- the
same expand_and_fill convention as plotting/utils.py:load_logger_data_new), averaged over the
post-warm-up window, then log10; averaged over trials (lower = better suppression). This
credits extinction, so it differs from a truncate-at-death average wherever trials go extinct
(here, c=3). Also report extinction fraction.

    micromamba activate rl-bacterial-control
    python scripts/eval/heuristic_baselines_constenv.py --quick
    python scripts/eval/heuristic_baselines_constenv.py --nutrients 1,2,3 --sweep_eval --n_jobs 24
    python scripts/eval/heuristic_baselines_constenv.py --plot_only
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import pickle

import numpy as np
from joblib import Parallel, delayed

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv

ANTIBIOTIC = 3.72
DELAY = 30
WARM_UP = 60  # post-warm-up window starts at index WARM_UP + 1
LEARNED_DECISIONS = 300  # decisions in the logged learned eval episodes (fixes their horizon)
TUNE_SEED0 = 0        # tuning seed block
EVAL_SEED0 = 100_000  # eval seed block (disjoint from tuning)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEARNED_BASE = os.path.join(_REPO, "results/eval/results_delay_30_record_constenv_eval")
LEARNED_EPISODE = "episode_399"
LEARNED_REPS = range(5)

OUTDIR = os.path.join(_REPO, "results", "eval_heuristic_baselines")
NPZ = os.path.join(OUTDIR, "heuristic_baselines_constenv.npz")

FAMILIES = {
    "no_drug":   ("no drug",            "#bbbbbb"),
    "constant":  ("constant (fixed)",   "#cf7171"),
    "pulsing":   ("pulsing",            "#1f77b4"),
    "threshold": ("threshold feedback", "#e08214"),
    "frontload": ("front-load",         "#8073ac"),
    "learned":   ("learned (RL)",       "#569122"),
}
# family -> (x-axis label, whether the knob axis is log-scaled)
KNOB_AXIS = {
    "pulsing":   ("pulsing half-period $h$ (decisions)", False),
    "threshold": ("threshold $\\Omega$ (population)", True),
    "frontload": ("front-load fraction $\\tau$", False),
}


def make_env(nutrient):
    return ConstantNutrientEnv(
        EnvConfig(
            k_n0_observation=False,
            b_observation=True,
            k_n0_constant=nutrient,
            delay_embed_len=DELAY,
            b_actions=[0, ANTIBIOTIC],
            max_pop=np.inf,
        ),
        CellConfig(),
    )


# --- policies: action(t, pop, N) -> 0 (no drug) or 1 (drug) ---------------------

def pol_constant(t, pop, N):
    return 1


def pol_nodrug(t, pop, N):
    return 0


def make_pulsing(half):
    def f(t, pop, N):
        return 1 if (t // half) % 2 == 0 else 0
    return f


def make_threshold(omega):
    """Reactive threshold feedback: treat while population is above the threshold."""
    def f(t, pop, N):
        return 1 if pop >= omega else 0
    return f


def make_frontload(tau):
    """Front-loaded schedule: drug for the first fraction tau of the horizon, then off."""
    def f(t, pop, N):
        return 1 if t < tau * N else 0
    return f


def run_episode(policy_fn, num_decisions, seed, nutrient):
    np.random.seed(seed)
    env = make_env(nutrient)
    with contextlib.redirect_stdout(io.StringIO()):  # silence fsolve init prints
        _, info = env.reset()
    pop = env.sim_cells.true_num_cells
    for t in range(num_decisions):
        action = policy_fn(t, pop, num_decisions)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, terminated, truncated, info = env.step(action)
        pop = env.sim_cells.true_num_cells
        if terminated or truncated:
            break
    return np.asarray(info["log"], dtype=float)


def summarize(log, full_len, warm_up=WARM_UP):
    """Per-trial metrics from a log array [t, nutr, drug, pop, U, phiR, phiS].

    The population is zero-padded to the full horizon (`full_len`) before time-averaging,
    matching the paper's loader (plotting/utils.py: load_logger_data_new -> expand_and_fill,
    which pads every trajectory to sim_length with zeros). A trial that goes extinct early
    therefore counts as population 0 for the remaining steps, so the metric credits
    extinction -- this is the convention used by Fig 2C (plotting/plot_gen.py).
    """
    pop = log[:, 3]
    padded = np.zeros(full_len)
    n = min(len(pop), full_len)
    padded[:n] = pop[:n]
    post = padded[warm_up + 1:]
    m = float(post.mean())
    return {
        "mean_log_pop": float(np.log10(m)) if m > 0 else float(np.log10(1e-5)),
        "final_pop": float(pop[-1]),
        "extinct": int(pop[-1] == 0),
    }


def run_policy(policy_fn, n_reps, num_decisions, n_jobs, seed0, nutrient):
    logs = Parallel(n_jobs=n_jobs)(
        delayed(run_episode)(policy_fn, num_decisions, seed0 + i, nutrient)
        for i in range(n_reps)
    )
    full_len = num_decisions + WARM_UP + 1
    return logs, [summarize(lg, full_len) for lg in logs]


def aggregate(summaries):
    mlp = np.array([s["mean_log_pop"] for s in summaries])
    ext = np.array([s["extinct"] for s in summaries])
    return {
        "mean_log_pop": float(mlp.mean()),
        "mean_log_pop_std": float(mlp.std()),
        "mean_log_pop_sem": float(mlp.std() / np.sqrt(len(mlp))),
        "extinct_frac": float(ext.mean()),
        "n": len(summaries),
    }


def _rep_summaries(nutrient, rep):
    d = os.path.join(LEARNED_BASE, f"a{ANTIBIOTIC:.2f}_n{nutrient:.2f}_delay{DELAY}_rep{rep}",
                     LEARNED_EPISODE)
    pkls = sorted(glob.glob(os.path.join(d, "*tcbk.pkl")))
    if not pkls:
        return None, None
    logs = [np.asarray(pickle.load(open(p, "rb"))["log"], dtype=float) for p in pkls]
    full_len = LEARNED_DECISIONS + WARM_UP + 1
    return logs, [summarize(lg, full_len) for lg in logs]


def load_learned(nutrient):
    """Load logged learned-policy trajectories; select the best training rep.

    Mirrors the min-selection idiom the paper's plotting uses. Also returns every rep's
    score so the across-seed spread stays visible (it is what best-rep selection hides).
    """
    per_rep = {}
    for rep in LEARNED_REPS:
        logs, summ = _rep_summaries(nutrient, rep)
        if summ is not None:
            per_rep[rep] = (aggregate(summ), logs, summ)
    if not per_rep:
        raise FileNotFoundError(f"No learned eval pkls for nutrient {nutrient} in {LEARNED_BASE}")
    best = min(per_rep, key=lambda r: per_rep[r][0]["mean_log_pop"])
    agg, logs, _ = per_rep[best]
    spread = {r: per_rep[r][0]["mean_log_pop"] for r in per_rep}
    return agg, logs[0], {"best_rep": int(best), "per_rep_mean_log_pop": spread}


def eval_family(members, n_tune, n_eval, num_decisions, n_jobs, nutrient, holdout, sweep_eval):
    """Sweep `members`; select the winner and (optionally) evaluate every member.

    holdout    : rank on the tuning block, report the winner on the disjoint eval block.
    sweep_eval : additionally evaluate every member on the eval block -> sweep curves.
    """
    tune = {}
    if holdout:
        for name, fn in members.items():
            _, ts = run_policy(fn, n_tune, num_decisions, n_jobs, TUNE_SEED0, nutrient)
            tune[name] = aggregate(ts)["mean_log_pop"]

    ev = {}
    if sweep_eval or not holdout:
        for name, fn in members.items():
            elogs, esumm = run_policy(fn, n_eval, num_decisions, n_jobs, EVAL_SEED0, nutrient)
            ev[name] = (aggregate(esumm), elogs[0])

    if holdout:
        chosen = min(tune, key=lambda n: tune[n])
        selection = "tune_mean_log_pop"
    else:
        chosen = min(ev, key=lambda n: ev[n][0]["mean_log_pop"])
        selection = "eval_mean_log_pop"

    if chosen not in ev:  # holdout without sweep_eval: only the winner needs an eval run
        elogs, esumm = run_policy(members[chosen], n_eval, num_decisions, n_jobs,
                                  EVAL_SEED0, nutrient)
        ev[chosen] = (aggregate(esumm), elogs[0])

    sweep = {"chosen": chosen, "selection": selection,
             "members": {n: {**({"tune_mean_log_pop": tune[n]} if n in tune else {}),
                             **({"eval": dict(ev[n][0])} if n in ev else {})}
                         for n in members}}
    return chosen, ev[chosen][0], ev[chosen][1], sweep


def build_members(omegas, taus, pulses):
    return {
        "pulsing":   {f"pulsing(h={h})": make_pulsing(h) for h in pulses},
        "threshold": {f"threshold(Omega={om:.0e})": make_threshold(om) for om in omegas},
        "frontload": {f"frontload(tau={ta:g})": make_frontload(ta) for ta in taus},
    }


def knob_of(name):
    """Numeric knob value parsed back out of a member name, for the sweep x-axis."""
    return float(name.split("=")[1].rstrip(")"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--nutrients", type=str, default="1,2,3", help="constant nutrient values c")
    ap.add_argument("--n_eval", type=int, default=100, help="eval seed-block reps (reported)")
    ap.add_argument("--n_tune", type=int, default=20, help="tuning seed-block reps (knob selection)")
    ap.add_argument("--num_decisions", type=int, default=300)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--omegas", type=str, default="1e2,3e2,1e3,3e3,1e4,3e4,1e5,3e5,1e6")
    ap.add_argument("--taus", type=str, default="0.25,0.5,0.75,0.9")
    ap.add_argument("--pulse_halves", type=str, default="3,5,10,15,20,25,30,40")
    ap.add_argument("--sweep_eval", action="store_true",
                    help="evaluate every swept member (not just the winner) -> sweep curves")
    ap.add_argument("--holdout", action="store_true",
                    help="rank swept members on a disjoint tuning block, then report the "
                         "winner on the eval block (default: report the best eval member)")
    ap.add_argument("--plot_only", action="store_true", help="re-plot from saved npz")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    if args.plot_only:
        d = np.load(NPZ, allow_pickle=True)
        plot_all(d["by_nutrient"][0], OUTDIR)
        return

    if args.quick:
        args.nutrients, args.n_eval, args.n_tune = "2", 2, 1
        args.num_decisions, args.n_jobs = 20, 4
        args.omegas, args.pulse_halves, args.taus = "1e4,1e5", "20", "0.9"
        args.sweep_eval = True

    holdout = args.holdout
    nutrients = [float(x) for x in args.nutrients.split(",")]
    omegas = [float(x) for x in args.omegas.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    pulses = [int(x) for x in args.pulse_halves.split(",")]

    by_nutrient = {}
    for nutrient in nutrients:
        print(f"\n########## constant nutrient c = {nutrient:g} ##########", flush=True)
        results, sample_logs, sweeps = {}, {}, {}

        for name, fn in (("no_drug", pol_nodrug), ("constant", pol_constant)):
            elogs, esumm = run_policy(fn, args.n_eval, args.num_decisions, args.n_jobs,
                                      EVAL_SEED0, nutrient)
            results[name] = aggregate(esumm)
            sample_logs[name] = elogs[0]
            print(f"  {name:24s} {results[name]['mean_log_pop']:+.2f}", flush=True)

        for fam, members in build_members(omegas, taus, pulses).items():
            chosen, agg, slog, sweep = eval_family(
                members, args.n_tune, args.n_eval, args.num_decisions, args.n_jobs,
                nutrient, holdout, args.sweep_eval)
            results[chosen] = agg
            sample_logs[chosen] = slog
            sweeps[fam] = sweep
            print(f"  [{fam}] chosen={chosen} -> {agg['mean_log_pop']:+.2f}", flush=True)

        try:
            lagg, llog, lmeta = load_learned(nutrient)
            results["learned (ours)"] = lagg
            sample_logs["learned (ours)"] = llog
            sweeps["learned_meta"] = lmeta
            print(f"  learned (rep{lmeta['best_rep']}) -> {lagg['mean_log_pop']:+.2f}", flush=True)
        except FileNotFoundError as e:
            print("  WARNING:", e, flush=True)

        by_nutrient[nutrient] = {"results": results, "sample_logs": sample_logs, "sweeps": sweeps}

    sel = "holdout knob-selection" if holdout else "best-on-eval"
    print(f"\n=== summary: mean log10 population (lower = better) | {sel} ===")
    for nutrient, d in by_nutrient.items():
        res, sw = d["results"], d["sweeps"]
        order = (["learned (ours)"]
                 + [sw[f]["chosen"] for f in ("pulsing", "threshold", "frontload") if f in sw]
                 + ["constant", "no_drug"])
        print(f"  --- c = {nutrient:g} ---")
        for name in order:
            if name in res:
                r = res[name]
                print(f"    {name:26s} {r['mean_log_pop']:+.2f} +/- {r['mean_log_pop_std']:.2f}"
                      f" (sem {r['mean_log_pop_sem']:.3f}) | extinct {r['extinct_frac']:.2f}"
                      f" (n={r['n']})")

    np.savez(NPZ, by_nutrient=np.array([by_nutrient], dtype=object))
    print(f"\nsaved -> {NPZ}")
    plot_all(by_nutrient, OUTDIR)


def plot_all(by_nutrient, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nutrients = sorted(by_nutrient)
    fams = [f for f in ("pulsing", "threshold", "frontload")
            if any(f in by_nutrient[n]["sweeps"] for n in nutrients)]

    # ---- sweep grid: relative performance log10(P_constant) - log10(P), Fig 2C style ----
    # higher = better: how much a swept member beats fixed dosing. Constant is the y = 0
    # line; the learned policy is the dashed reference.
    fig, axes = plt.subplots(len(fams), len(nutrients),
                             figsize=(4.3 * len(nutrients), 3.1 * len(fams)),
                             squeeze=False)
    for i, fam in enumerate(fams):
        xlabel, logx = KNOB_AXIS[fam]
        for j, nutrient in enumerate(nutrients):
            ax = axes[i][j]
            d = by_nutrient[nutrient]
            res = d["results"]
            sweep = d["sweeps"].get(fam)
            if sweep is None:
                continue
            c0 = res["constant"]["mean_log_pop"] if "constant" in res else 0.0
            pts = [(knob_of(m), c0 - v["eval"]["mean_log_pop"], v["eval"]["mean_log_pop_std"])
                   for m, v in sweep["members"].items() if "eval" in v]
            if pts:
                pts.sort()
                x, y, e = (np.array(t) for t in zip(*pts))
                ax.fill_between(x, y - e, y + e, color=FAMILIES[fam][1], alpha=0.18, lw=0)
                ax.plot(x, y, "o-", color=FAMILIES[fam][1], lw=1.6, ms=4,
                        label=FAMILIES[fam][0])
                kb = knob_of(sweep["chosen"])
                ysel = dict((knob_of(m), c0 - v["eval"]["mean_log_pop"])
                            for m, v in sweep["members"].items() if "eval" in v)[kb]
                ax.plot([kb], [ysel], "^", ms=11, color=FAMILIES[fam][1], mec="k", mew=0.6,
                        zorder=5, label="selected")
            ax.axhline(0, color="#cf7171", ls=":", lw=1.5, label="constant")
            if "learned (ours)" in res:
                ax.axhline(c0 - res["learned (ours)"]["mean_log_pop"], color="#569122",
                           ls="--", lw=1.5, label="learned (RL)")
            if logx:
                ax.set_xscale("log")
            if i == 0:
                ax.set_title(f"constant nutrient $c$ = {nutrient:g}", fontsize=11)
            if j == 0:
                ax.set_ylabel(r"$\log_{10}P_{\mathrm{constant}} - \log_{10}P$")
            ax.set_xlabel(xlabel, fontsize=9)
            if i == 0 and j == len(nutrients) - 1:
                ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle("Adaptive-heuristic sweeps vs. learned policy (higher = better control)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(outdir, "heuristic_baselines_sweeps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")

    # ---- best-member bar figures: one per metric, same structure ----
    def members_of(d):
        res, sw = d["results"], d["sweeps"]
        chosen = ["no_drug", "constant"] + [sw[f]["chosen"] for f in fams if f in sw]
        if "learned (ours)" in res:
            chosen.append("learned (ours)")
        fam_of = [next((f for f in FAMILIES if c.startswith(f)), "learned") for c in chosen]
        labels = [FAMILIES[f][0] for f in fam_of]
        cols = [FAMILIES[f][1] for f in fam_of]
        return chosen, labels, cols, np.arange(len(chosen))[::-1]

    # mean log10 population (suppression, lower = better)
    fig, axes = plt.subplots(1, len(nutrients), figsize=(4.6 * len(nutrients), 3.6),
                             squeeze=False)
    for j, nutrient in enumerate(nutrients):
        res = by_nutrient[nutrient]["results"]
        chosen, labels, cols, ypos = members_of(by_nutrient[nutrient])
        ax = axes[0][j]
        vals = [res[c]["mean_log_pop"] for c in chosen]
        errs = [res[c]["mean_log_pop_std"] for c in chosen]
        ax.barh(ypos, vals, xerr=errs, color=cols, edgecolor="k", capsize=4, height=0.65)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels if j == 0 else [""] * len(labels), fontsize=9)
        ax.axvline(res["constant"]["mean_log_pop"], color="#cf7171", ls=":", lw=1.5)
        ax.set_xlabel("mean log$_{10}$ population  (lower = better)")
        ax.set_title(f"$c$ = {nutrient:g}", fontsize=11)
    fig.suptitle("Best member of each family vs. learned policy — population suppression",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(outdir, "heuristic_baselines_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")

    # extinction probability (higher = better) -- same structure as the suppression figure
    fig, axes = plt.subplots(1, len(nutrients), figsize=(4.6 * len(nutrients), 3.6),
                             squeeze=False)
    for j, nutrient in enumerate(nutrients):
        res = by_nutrient[nutrient]["results"]
        chosen, labels, cols, ypos = members_of(by_nutrient[nutrient])
        ax = axes[0][j]
        ext = [res[c]["extinct_frac"] for c in chosen]
        ax.barh(ypos, ext, color=cols, edgecolor="k", height=0.65)
        for y, e in zip(ypos, ext):
            ax.text(min(e + 0.02, 0.9), y, f"{e:.2f}", va="center", fontsize=8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels if j == 0 else [""] * len(labels), fontsize=9)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("extinction probability  (higher = better)")
        ax.set_title(f"$c$ = {nutrient:g}", fontsize=11)
    fig.suptitle("Best member of each family vs. learned policy — extinction probability",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(outdir, "heuristic_baselines_extinction.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
