#!/usr/bin/env python
"""Characterize the TRUE growth-rate and log-population scales from the already-saved clean
baselines, then express the injected measurement noise as a fraction of those scales.

This produces the "what the noise level sigma means physically" appendix of
results/eval_noisy/analysis/analysis_report.md. It is the companion to analyze_noisy.py:
analyze_noisy.py measures how the controllers *degrade* under noise/lag; this script measures
the *physical scale* of the true signals the noise is added to, so sigma can be reported as a
percentage of the growth-rate and log-pop the policy actually reads.

No re-simulation: it reads only the clean baselines (noise0.0_lag0/) that run_all_noisy.py
already wrote, restricted to EXACTLY the paper configs that run_all_noisy.py enumerates (65
base runs, last training episode, control phase only).

Noise model (src/rlBacterialControl/envs/envs.py, BaseEnv.observation):
    N_meas = N_true * exp(eps),  eps ~ N(0, sigma).
  - log-pop observation noise std     = sigma
  - growth-rate observation noise std = sqrt(2)*sigma/delta_t   [from (eps_t - eps_{t-1})/delta_t]

Run from the repo root:  python scripts/eval/analyze_noise_baseline_from_saved.py
"""
import os
import glob
import pickle
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

OUT_DIR = "results/eval_noisy"
RESULTS_ROOT = "results/train"
DELTA_T = 0.2  # hours per decision step (matches the trained configs)

# reuse the driver's enumeration + path helpers so the set of runs matches run_all_noisy exactly
_spec = importlib.util.spec_from_file_location(
    "run_all_noisy", os.path.join(HERE, "run_all_noisy.py"))
ran = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ran)


def load_baseline_runs():
    """Return [(env_type, eval_suffix, [pkl paths])] for every paper-config clean baseline."""
    runs = []
    for env_type in ran.ENV_ORDER:
        results_dir = os.path.join(RESULTS_ROOT, ran.ENV_DIRS[env_type])
        for run in ran.enumerate_runs(env_type, results_dir):
            trial_dir = os.path.join(results_dir, run["trial_name"])
            try:
                episode = ran.resolve_episode(trial_dir, "last")
            except FileNotFoundError:
                continue
            eval_out = ran.eval_out_dir(OUT_DIR, results_dir, run["eval_suffix"], episode, 0.0, 0)
            pkls = sorted(glob.glob(os.path.join(eval_out, "trial_*tcbk.pkl")))
            if pkls:
                runs.append((env_type, run["eval_suffix"], pkls))
    return runs


def gr_logN(info):
    """True per-step growth rate (1/hr) and log-population (natural log) from one episode log,
    tagged with the drug that produced each step and whether the step is in the control phase."""
    log = np.array(info["log"])
    dt = info["delta_t"]
    warm = int(info["warm_up"])
    N = log[:, 3].astype(float)
    b = log[:, 2].astype(float)
    lnN = np.log(np.clip(N, 1e-5, None))
    gr = np.diff(lnN) / dt          # gr[i] is the growth over the step ending at index i+1
    b_step = b[1:]                  # drug applied producing that step
    idx = np.arange(1, len(N))
    ctrl = idx >= warm              # control phase (agent acting, past warm-up)
    return dict(gr=gr, lnN=lnN[1:], b=b_step, ctrl=ctrl)


FLOOR = np.log(1e-5)  # -11.5; the extinction clip in envs.observation/reward


def summarize(GR, LNN, B, CTRL):
    """Growth-rate + log-pop statistics over the control phase, with the single extinction
    spike into/out of the 1e-5 floor filtered out (it is not a real growth rate and would
    otherwise dominate the mean|.| and std)."""
    keep = CTRL & (LNN > FLOOR + 1e-6)
    gr_c, lnN_c, b_c = GR[keep], LNN[keep], B[keep]
    on = b_c > 0
    off = ~on
    return dict(
        n=int(keep.sum()),
        gr_mean=np.mean(gr_c), gr_std=np.std(gr_c),
        gr_absmean=np.mean(np.abs(gr_c)), gr_absstd=np.std(np.abs(gr_c)),
        gr_off=np.mean(gr_c[off]) if off.any() else np.nan,   # drug-free (max sustained) growth
        gr_on=np.mean(gr_c[on]) if on.any() else np.nan,      # drug-on killing rate
        frac_on=on.mean(),
        lnN_mean=np.mean(lnN_c), lnN_std=np.std(lnN_c),
        lnN_min=np.min(lnN_c), lnN_max=np.max(lnN_c),
    )


def analyze_noise_baseline_from_saved(delta_t=DELTA_T):
    """Load the clean baselines, print the true-scale table and the noise-vs-scale table,
    and return (per_env_stats, all_stats) for programmatic use."""
    runs = load_baseline_runs()
    print(f"loaded {len(runs)} paper-config runs (noise0.0_lag0 baselines)\n")

    # concatenate every control-phase step per env type
    per_env = {}
    for env_type, _suffix, pkls in runs:
        GR, LNN, B, CTRL = [], [], [], []
        for p in pkls:
            d = gr_logN(pickle.load(open(p, "rb")))
            GR.append(d["gr"]); LNN.append(d["lnN"]); B.append(d["b"]); CTRL.append(d["ctrl"])
        per_env.setdefault(env_type, {"GR": [], "LNN": [], "B": [], "CTRL": []})
        per_env[env_type]["GR"].append(np.concatenate(GR))
        per_env[env_type]["LNN"].append(np.concatenate(LNN))
        per_env[env_type]["B"].append(np.concatenate(B))
        per_env[env_type]["CTRL"].append(np.concatenate(CTRL))

    def row(name, s):
        return (f"{name:8} {s['gr_mean']:+8.3f} {s['gr_std']:7.3f} "
                f"{s['gr_absmean']:9.3f} {s['gr_absstd']:8.3f} "
                f"{s['gr_off']:+9.3f} {s['gr_on']:+8.3f} | "
                f"{s['lnN_mean']:8.2f} {s['lnN_std']:7.2f} "
                f"[{s['lnN_min']:5.1f},{s['lnN_max']:5.1f}]")

    print("=== TRUE growth-rate & log-pop scale, CONTROL PHASE, per env type ===")
    print(f"{'env':8} {'gr_mean':>8} {'gr_std':>7} {'|gr|mean':>9} {'|gr|std':>8} "
          f"{'gr_off_m':>9} {'gr_on_m':>8} | {'lnN_mean':>8} {'lnN_std':>7} {'lnN_rng':>14}")

    allGR, allLNN, allB, allCTRL = [], [], [], []
    per_env_stats = {}
    for env_type in ran.ENV_ORDER:
        if env_type not in per_env:
            continue
        GR = np.concatenate(per_env[env_type]["GR"])
        LNN = np.concatenate(per_env[env_type]["LNN"])
        B = np.concatenate(per_env[env_type]["B"])
        CTRL = np.concatenate(per_env[env_type]["CTRL"])
        allGR.append(GR); allLNN.append(LNN); allB.append(B); allCTRL.append(CTRL)
        s = summarize(GR, LNN, B, CTRL)
        per_env_stats[env_type] = s
        print(row(env_type, s))

    GR = np.concatenate(allGR); LNN = np.concatenate(allLNN)
    B = np.concatenate(allB); CTRL = np.concatenate(allCTRL)
    s = summarize(GR, LNN, B, CTRL)
    print(row("ALL", s))

    # reference scales for the noise-% table (from the pooled "ALL" stats)
    gr_mean_ref = abs(s['gr_mean'])   # |mean growth rate| (small -> large %)
    gr_std_ref = s['gr_std']          # SD of the growth-rate signal the agent must read
    gr_absmean_ref = s['gr_absmean']  # typical |growth rate|
    gr_absstd_ref = s['gr_absstd']    # SD of |growth rate|
    lnN_mean_ref = s['lnN_mean']
    lnN_std_ref = s['lnN_std']

    print("\n=== Injected noise vs those scales ===")
    print("growth-rate noise SD = sqrt(2)*sigma/delta_t (effective sigma on gr); "
          "log-pop noise SD = sigma")
    print(f"gr refs (1/hr): |gr_mean|={gr_mean_ref:.3f}, gr_std={gr_std_ref:.3f}, "
          f"|gr|_mean={gr_absmean_ref:.3f}, |gr|_std={gr_absstd_ref:.3f}")
    print(f"lnN refs (nat log): lnN_mean={lnN_mean_ref:.2f}, lnN_std={lnN_std_ref:.2f}\n")
    hdr = (f"{'sigma':>6} | {'gr_noiseSD':>10} | {'%gr_mean':>9} {'%gr_std':>8} "
           f"{'%|gr|mean':>9} {'%|gr|std':>9} | {'%lnN_mean':>9} {'%lnN_std':>9} | "
           f"{'count_err':>9}")
    print(hdr)
    print("-" * len(hdr))
    for sig in [0.05, 0.10, 0.20, 0.40]:
        grn = np.sqrt(2) * sig / delta_t     # effective sigma on the growth-rate observation
        count_err = np.exp(sig) - 1          # multiplicative count error (~sigma for small sigma)
        print(f"{sig:>6} | {grn:>10.3f} | "
              f"{100*grn/gr_mean_ref:>8.1f}% {100*grn/gr_std_ref:>7.1f}% "
              f"{100*grn/gr_absmean_ref:>8.1f}% {100*grn/gr_absstd_ref:>8.1f}% | "
              f"{100*sig/lnN_mean_ref:>8.1f}% {100*sig/lnN_std_ref:>8.1f}% | "
              f"{100*count_err:>8.1f}%")

    return per_env_stats, s


if __name__ == "__main__":
    os.chdir(REPO)
    analyze_noise_baseline_from_saved()
