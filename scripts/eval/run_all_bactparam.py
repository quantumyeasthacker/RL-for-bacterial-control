#!/usr/bin/env python
"""Run ALL trained models through the bacterial-parameter-mismatch eval.

Discovers every trained-model trial under --results-root and drives
`eval_trained_agents_bactparam.py` over a grid of (alpha, beta, sigma) multipliers, processing
env types in the order gen, const, var, control. The generalized agent is expanded across the
same eval conditions as the clean gen eval (constenv k_n0 in {1,2,3,4}; varenv T in {6,12,18,24}).

Default grid (29 configs):
  * a 5x5 alpha x beta grid at sigma=1.0 -- {0.8,0.9,1.0,1.1,1.2} x {0.8,0.9,1.0,1.1,1.2}
    (this subsumes the per-parameter alpha and beta marginals along its centre cross), plus
  * a sigma marginal at alpha=beta=1.0 -- sigma in {0.8,0.9,1.1,1.2}.
The (1.0,1.0,1.0) baseline is the grid centre, run first, and reproduces the clean eval.

The alpha x beta *grid* (not just corners) is deliberate: heatmaps localize where control
degrades, and constant-alpha/beta contours test whether the drug's net effect is governed by
the ratio -- neither is answerable from corner points. To find a boundary that sits outside
+/-20%, widen --alpha-mults / --beta-mults (or edit the DEFAULT_* constants below).

Key features:
  * resumable -- a run whose output pkls already exist is skipped (use --force to redo),
  * parallel  -- runs --jobs eval processes at once (each eval is single-threaded),
  * --dry-run -- list/count what would run without executing,
  * manifest  -- appends every run's status to results/eval_bactparam/run_manifest.csv.

Examples (run from the repo root, with the env python):
  # see the plan and scale without running anything
  python scripts/eval/run_all_bactparam.py --dry-run

  # run the full grid, 24 parallel workers
  python scripts/eval/run_all_bactparam.py --jobs 24

  # only the constant + variable envs, a single setting, quick smoke test
  python scripts/eval/run_all_bactparam.py --env-types const var \
      --alpha-mults 0.8 --beta-mults 1.2 --sigma-mults \
      --num-decisions 30 --num-reps-eval 2
"""
import os
import re
import sys
import csv
import time
import argparse
import importlib.util
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(HERE, "eval_trained_agents_bactparam.py")

# import the eval module so the driver and the eval script share the episode resolver + dir tag
_spec = importlib.util.spec_from_file_location("eval_bactparam", EVAL_SCRIPT)
eval_bactparam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_bactparam)
resolve_episode = eval_bactparam.resolve_episode
param_tag = eval_bactparam.param_tag

# env_type -> results/train subfolder (processed in this order)
ENV_DIRS = {
    "gen": "results_delay_30_record_generalized",
    "const": "results_delay_30_record_constenv",
    "var": "results_delay_30_record_varenv",
    "control": "results_delay_30_record_controlenv",
}
ENV_ORDER = ["gen", "const", "var", "control"]

# generalized agent is evaluated on these specific conditions (mirrors the clean gen eval)
GEN_EVAL_CONDS = [("constenv", v) for v in ("1.00", "2.00", "3.00", "4.00")] \
               + [("varenv", v) for v in ("6", "12", "18", "24")]

# default multiplier grids (edit or override on the CLI to widen past +/-20%)
DEFAULT_ALPHA_MULTS = [0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_BETA_MULTS = [0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_SIGMA_MULTS = [0.8, 0.9, 1.1, 1.2]  # sigma-only marginal; 1.0 is already in the alpha x beta grid

# subset of trained models to evaluate, mirroring the paper's reported configs (None = no filter).
CONST_NUTRIENTS = {1.0, 2.0, 3.0}      # constant-nutrient k_n0 values to include
VAR_PERIODS = {"12"}                   # variable-nutrient T_k_n0 periods to include
CONTROL_NRANGES = {"1_3"}              # control nutrient action sets to include
CONTROL_OBS_FILTER = ("True", "True")  # control (b_obs, k_obs): the paper uses both True
GEN_EPISODES = {"500"}                 # generalized agent: training-episode counts to include

# trial-name parsers (the dir basename is the trial_name produced by the training scripts)
RE_CONST = re.compile(r"^a([\d.]+)_n([\d.]+)_delay(\d+)_rep(\d+)$")
RE_VAR = re.compile(r"^a([\d.]+)_T(\d+)_delay(\d+)_rep(\d+)$")
RE_CONTROL = re.compile(r"^a([\d.]+)_n(.+)_b(True|False)_k(True|False)_delay(\d+)_rep(\d+)$")
RE_GEN = re.compile(r"^a([\d.]+)_(const.+_T[\d_]+)_delay(\d+)_episodes(\d+)_rep(\d+)$")


def build_grid(alpha_mults, beta_mults, sigma_mults):
    """(alpha, beta, sigma) multiplier tuples: 5x5 alpha x beta at sigma=1, plus a sigma marginal
    at alpha=beta=1. The (1,1,1) baseline is de-duped and placed first so baselines finish early."""
    grid = [(ma, mb, 1.0) for ma in alpha_mults for mb in beta_mults]
    grid += [(1.0, 1.0, ms) for ms in sigma_mults]
    ordered, seen = [], set()
    for t in [(1.0, 1.0, 1.0)] + grid:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def enumerate_runs(env_type, results_dir):
    """Return a list of base runs (one per model, expanded for gen) as dicts:
       {trial_name, eval_suffix, cli}  where cli is the env-specific + per-trial flag list."""
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for trial in sorted(os.listdir(results_dir)):
        if not os.path.isdir(os.path.join(results_dir, trial)):
            continue

        if env_type == "const":
            m = RE_CONST.match(trial)
            if not m:
                continue
            a, n, d, r = m.groups()
            if CONST_NUTRIENTS is not None and float(n) not in CONST_NUTRIENTS:
                continue
            cli = ["--antibiotic", a, "--nutrient", n, "--delay", d, "--rep-run", r]
            runs.append(dict(trial_name=trial, eval_suffix=trial, cli=cli))

        elif env_type == "var":
            m = RE_VAR.match(trial)
            if not m:
                continue
            a, T, d, r = m.groups()
            if VAR_PERIODS is not None and T not in VAR_PERIODS:
                continue
            cli = ["--antibiotic", a, "--nutrient", T, "--delay", d, "--rep-run", r]
            runs.append(dict(trial_name=trial, eval_suffix=trial, cli=cli))

        elif env_type == "control":
            m = RE_CONTROL.match(trial)
            if not m:
                continue
            a, nrange, bobs, kobs, d, r = m.groups()
            if CONTROL_OBS_FILTER is not None and (bobs, kobs) != CONTROL_OBS_FILTER:
                continue
            if CONTROL_NRANGES is not None and nrange not in CONTROL_NRANGES:
                continue
            cli = ["--antibiotic", a, "--nutrient-range", nrange, "--b-obs", bobs,
                   "--k-obs", kobs, "--delay", d, "--rep-run", r]
            runs.append(dict(trial_name=trial, eval_suffix=trial, cli=cli))

        elif env_type == "gen":
            m = RE_GEN.match(trial)
            if not m:
                continue
            a, trained_env, d, eps, r = m.groups()
            if GEN_EPISODES is not None and eps not in GEN_EPISODES:
                continue
            for eval_env, eval_var in GEN_EVAL_CONDS:
                cli = ["--antibiotic", a, "--trained-env", trained_env, "--delay", d,
                       "--episodes", eps, "--rep-run", r,
                       "--eval-env", eval_env, "--eval-variable", eval_var]
                runs.append(dict(trial_name=trial,
                                 eval_suffix=f"{trial}_{eval_env}_{eval_var}",
                                 cli=cli))
    return runs


def eval_out_dir(out_dir, results_dir, eval_suffix, episode, ma, mb, ms):
    """Mirror the path the eval script writes to (kept in sync with eval_trained_agents_bactparam.py)."""
    env_group = os.path.basename(results_dir.rstrip("/"))
    return f"{out_dir}/{env_group}/{eval_suffix}/{episode}/{param_tag(ma, mb, ms)}/"


def is_done(eval_out, rep_eval, num_reps_eval):
    """A run is complete iff its last expected pkl exists (they are written in order)."""
    last = f"trial_{rep_eval * num_reps_eval + num_reps_eval - 1}tcbk.pkl"
    return os.path.exists(os.path.join(eval_out, last))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results/train", help="dir holding the per-env-type folders")
    p.add_argument("--out-dir", default="results/eval_bactparam", help="output root (default results/eval_bactparam)")
    p.add_argument("--env-types", nargs="+", choices=ENV_ORDER, default=ENV_ORDER,
                   help="which env types to run, in this order (default: gen const var control)")
    p.add_argument("--alpha-mults", type=float, nargs="+", default=DEFAULT_ALPHA_MULTS,
                   help="alpha multipliers for the alpha x beta grid")
    p.add_argument("--beta-mults", type=float, nargs="+", default=DEFAULT_BETA_MULTS,
                   help="beta multipliers for the alpha x beta grid")
    p.add_argument("--sigma-mults", type=float, nargs="*", default=DEFAULT_SIGMA_MULTS,
                   help="sigma-only marginal multipliers at alpha=beta=1 (pass with no values to skip)")
    p.add_argument("--episode", default="last", help="episode to load, or 'last' (default)")
    p.add_argument("--rep-eval", type=int, default=0)
    p.add_argument("--num-decisions", type=int, default=300)
    p.add_argument("--num-reps-eval", type=int, default=10)
    p.add_argument("--jobs", type=int, default=16, help="parallel eval processes (default 16)")
    p.add_argument("--force", action="store_true", help="re-run even if outputs already exist")
    p.add_argument("--dry-run", action="store_true", help="list/count tasks without running")
    p.add_argument("--limit", type=int, default=None, help="cap number of tasks (for testing)")
    args = p.parse_args()

    grid = build_grid(args.alpha_mults, args.beta_mults, args.sigma_mults)  # baseline first

    # build the full task list, in env order
    tasks = []          # each: dict(env_type, results_dir, eval_suffix, cli, ma, mb, ms, eval_out)
    skipped = 0
    per_env_counts = {}
    for env_type in args.env_types:
        results_dir = os.path.join(args.results_root, ENV_DIRS[env_type])
        base_runs = enumerate_runs(env_type, results_dir)
        n_env_tasks = 0
        for run in base_runs:
            trial_dir = os.path.join(results_dir, run["trial_name"])
            try:
                episode = resolve_episode(trial_dir, args.episode)
            except FileNotFoundError:
                continue
            for ma, mb, ms in grid:
                eval_out = eval_out_dir(args.out_dir, results_dir, run["eval_suffix"], episode, ma, mb, ms)
                if not args.force and is_done(eval_out, args.rep_eval, args.num_reps_eval):
                    skipped += 1
                    continue
                tasks.append(dict(env_type=env_type, results_dir=results_dir,
                                  eval_suffix=run["eval_suffix"], cli=run["cli"],
                                  ma=ma, mb=mb, ms=ms, eval_out=eval_out))
                n_env_tasks += 1
        per_env_counts[env_type] = (len(base_runs), n_env_tasks)

    if args.limit is not None:
        tasks = tasks[:args.limit]

    # summary
    print("=" * 72)
    print(f"alpha x beta grid: {args.alpha_mults} x {args.beta_mults}")
    print(f"sigma marginal:    {args.sigma_mults}")
    print(f"total settings:    {len(grid)}   (baseline {param_tag(1.0, 1.0, 1.0)} first)")
    print(f"reps/eval:  {args.num_reps_eval}   decisions: {args.num_decisions}")
    for env_type in args.env_types:
        nb, nt = per_env_counts.get(env_type, (0, 0))
        print(f"  {env_type:8s}: {nb:3d} base runs (x grid) -> {nt:4d} to run")
    print(f"to run: {len(tasks)}   already done (skipped): {skipped}")
    est_min = len(tasks) * 3.4 / max(args.jobs, 1)
    print(f"est wall time @ {args.jobs} jobs: ~{est_min:.0f} min (~{est_min/60:.1f} h)  [~3.4 min/run]")
    print("=" * 72)

    if args.dry_run or not tasks:
        if args.dry_run:
            for t in tasks[:20]:
                print(f"  [{t['env_type']}] {t['eval_suffix']} {param_tag(t['ma'], t['mb'], t['ms'])}")
            if len(tasks) > 20:
                print(f"  ... (+{len(tasks) - 20} more)")
        return

    # manifest
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "run_manifest.csv")
    new_manifest = not os.path.exists(manifest_path)
    manifest_lock = threading.Lock()
    manifest_f = open(manifest_path, "a", newline="")
    manifest_w = csv.writer(manifest_f)
    if new_manifest:
        manifest_w.writerow(["env_type", "eval_suffix", "alpha_mult", "beta_mult", "sigma_mult",
                             "status", "seconds", "eval_out"])

    def build_cmd(t):
        return [sys.executable, EVAL_SCRIPT, t["env_type"],
                "--results-dir", t["results_dir"], "--out-dir", args.out_dir,
                "--alpha-mult", str(t["ma"]), "--beta-mult", str(t["mb"]), "--sigma-mult", str(t["ms"]),
                "--episode", args.episode, "--rep-eval", str(args.rep_eval),
                "--num-decisions", str(args.num_decisions),
                "--num-reps-eval", str(args.num_reps_eval)] + t["cli"]

    def run_one(t):
        t0 = time.time()
        proc = subprocess.run(build_cmd(t), capture_output=True, text=True)
        secs = time.time() - t0
        ok = proc.returncode == 0
        with manifest_lock:
            manifest_w.writerow([t["env_type"], t["eval_suffix"], t["ma"], t["mb"], t["ms"],
                                 "ok" if ok else "FAIL", f"{secs:.0f}", t["eval_out"]])
            manifest_f.flush()
        return t, ok, secs, proc.stderr

    done = 0
    fails = 0
    start = time.time()
    print(f"running {len(tasks)} tasks with {args.jobs} workers ...")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futures):
            t, ok, secs, err = fut.result()
            done += 1
            if not ok:
                fails += 1
                tail = "\n    ".join(err.strip().splitlines()[-3:]) if err else "(no stderr)"
                print(f"  FAIL [{t['env_type']}] {t['eval_suffix']} {param_tag(t['ma'], t['mb'], t['ms'])}\n    {tail}")
            if done % 20 == 0 or done == len(tasks):
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(tasks) - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done}/{len(tasks)}] {fails} fail | "
                      f"elapsed {elapsed/60:.0f}m | eta {eta:.0f}m")
    manifest_f.close()
    print(f"DONE: {done} tasks, {fails} failures. manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
