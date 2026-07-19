#!/usr/bin/env python
"""Run ALL fixed reference protocols over the bacterial-parameter-mismatch grid.

Companion driver to run_all_bactparam.py (which drives the *trained* agents). It sweeps
baseline_policies_bactparam.py over the SAME (alpha, beta, sigma) grid so that, for every
perturbed bacterium the RL eval visits, we also have the paper's hand-designed protocols run on
the identical physiology. That is what makes the comparison fair: when a parameter shift makes
the drug intrinsically weaker, the baselines degrade too, and analyze_bactparam_fair.py measures
the learned policy's *advantage over the best fixed protocol* per cell rather than against the
nominal null.

Protocols depend only on (env dynamics, nutrient condition) -- never on a trained model -- so the
constant/variable baselines are computed once per nutrient condition and reused by the generalized
agent's constenv/varenv evaluations (analyze_bactparam_fair joins on the condition). The nutrient
conditions therefore cover the union the RL eval needs:
  const  : k_n0 in {1,2,3,4}   (const models use 1,2,3; gen->const adds 4)
  var    : T   in {6,12,18,24}  (var model uses 12; gen->var uses all four)
  control: nutrient-range 1_3

Same grid as run_all_bactparam (29 settings: 5x5 alpha x beta at sigma=1 + a sigma marginal).
Resumable (skip cells whose pkls exist; --force to redo), parallel (--jobs), --dry-run, and a
manifest under the output root.

Examples (repo root, env python):
  python scripts/eval/run_all_baseline_bactparam.py --dry-run
  python scripts/eval/run_all_baseline_bactparam.py --jobs 24
  # quick smoke test: one condition, one cell, few reps
  python scripts/eval/run_all_baseline_bactparam.py --families const \
      --const-nutrients 2 --alpha-mults 0.8 --beta-mults 1.2 --sigma-mults \
      --num-decisions 30 --num-reps-eval 2
"""
import os
import sys
import csv
import time
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # let the sibling eval scripts import by module name
from baseline_policies_bactparam import family_policies  # noqa: E402
from eval_trained_agents_bactparam import param_tag       # noqa: E402
from run_all_bactparam import build_grid                  # noqa: E402

RUNNER = os.path.join(HERE, "baseline_policies_bactparam.py")

FAMILY_ORDER = ["const", "var", "control"]
DEFAULT_CONST_NUTRIENTS = [1.0, 2.0, 3.0, 4.0]
DEFAULT_VAR_PERIODS = [6, 12, 18, 24]
DEFAULT_CONTROL_RANGES = ["1_3"]

DEFAULT_ALPHA_MULTS = [0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_BETA_MULTS = [0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_SIGMA_MULTS = [0.8, 0.9, 1.1, 1.2]


def condition_tag(family, cond):
    if family == "const":
        return f"{float(cond):.2f}"
    if family == "var":
        return str(int(cond))
    return str(cond)  # control range


def condition_flag(family, cond):
    if family == "const":
        return ["--nutrient", str(float(cond))]
    if family == "var":
        return ["--period", str(int(cond))]
    return ["--nutrient-range", str(cond)]  # control


def conditions_for(family, args):
    if family == "const":
        return args.const_nutrients
    if family == "var":
        return args.var_periods
    return args.control_ranges


def eval_out_dir(out_dir, family, cond, policy, ma, mb, ms):
    """Mirror the path baseline_policies_bactparam.py writes to."""
    return f"{out_dir}/{family}env/{condition_tag(family, cond)}/{policy}/{param_tag(ma, mb, ms)}/"


def is_done(eval_out, rep_eval, num_reps_eval):
    last = f"trial_{rep_eval * num_reps_eval + num_reps_eval - 1}tcbk.pkl"
    return os.path.exists(os.path.join(eval_out, last))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results/eval_bactparam_baseline")
    p.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=FAMILY_ORDER)
    p.add_argument("--const-nutrients", type=float, nargs="+", default=DEFAULT_CONST_NUTRIENTS)
    p.add_argument("--var-periods", type=int, nargs="+", default=DEFAULT_VAR_PERIODS)
    p.add_argument("--control-ranges", nargs="+", default=DEFAULT_CONTROL_RANGES)
    p.add_argument("--alpha-mults", type=float, nargs="+", default=DEFAULT_ALPHA_MULTS)
    p.add_argument("--beta-mults", type=float, nargs="+", default=DEFAULT_BETA_MULTS)
    p.add_argument("--sigma-mults", type=float, nargs="*", default=DEFAULT_SIGMA_MULTS)
    p.add_argument("--antibiotic", type=float, default=3.72)
    p.add_argument("--delay", type=int, default=30)
    p.add_argument("--num-decisions", type=int, default=300)
    p.add_argument("--num-reps-eval", type=int, default=10)
    p.add_argument("--rep-eval", type=int, default=0)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    grid = build_grid(args.alpha_mults, args.beta_mults, args.sigma_mults)  # baseline (1,1,1) first

    tasks = []
    skipped = 0
    per_family_counts = {}
    for family in args.families:
        n_family = 0
        for cond in conditions_for(family, args):
            for policy in family_policies(family):
                for ma, mb, ms in grid:
                    eval_out = eval_out_dir(args.out_dir, family, cond, policy, ma, mb, ms)
                    if not args.force and is_done(eval_out, args.rep_eval, args.num_reps_eval):
                        skipped += 1
                        continue
                    tasks.append(dict(family=family, cond=cond, policy=policy,
                                      ma=ma, mb=mb, ms=ms, eval_out=eval_out))
                    n_family += 1
        per_family_counts[family] = n_family

    if args.limit is not None:
        tasks = tasks[:args.limit]

    print("=" * 72)
    print(f"alpha x beta grid: {args.alpha_mults} x {args.beta_mults}")
    print(f"sigma marginal:    {args.sigma_mults}")
    print(f"total settings:    {len(grid)}   (baseline {param_tag(1.0, 1.0, 1.0)} first)")
    print(f"reps/eval:  {args.num_reps_eval}   decisions: {args.num_decisions}")
    for family in args.families:
        conds = conditions_for(family, args)
        pols = family_policies(family)
        print(f"  {family:8s}: {len(conds)} conds x {len(pols)} policies x {len(grid)} cells "
              f"-> {per_family_counts.get(family, 0)} to run")
    print(f"to run: {len(tasks)}   already done (skipped): {skipped}")
    print("=" * 72)

    if args.dry_run or not tasks:
        for t in tasks[:20]:
            print(f"  [{t['family']}] cond={t['cond']} {t['policy']} "
                  f"{param_tag(t['ma'], t['mb'], t['ms'])}")
        if len(tasks) > 20:
            print(f"  ... (+{len(tasks) - 20} more)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "run_manifest.csv")
    new_manifest = not os.path.exists(manifest_path)
    manifest_lock = threading.Lock()
    manifest_f = open(manifest_path, "a", newline="")
    manifest_w = csv.writer(manifest_f)
    if new_manifest:
        manifest_w.writerow(["family", "condition", "policy", "alpha_mult", "beta_mult",
                             "sigma_mult", "status", "seconds", "eval_out"])

    def build_cmd(t):
        return [sys.executable, RUNNER, t["family"],
                "--policy", t["policy"], "--out-dir", args.out_dir,
                "--alpha-mult", str(t["ma"]), "--beta-mult", str(t["mb"]),
                "--sigma-mult", str(t["ms"]), "--antibiotic", str(args.antibiotic),
                "--delay", str(args.delay), "--num-decisions", str(args.num_decisions),
                "--num-reps-eval", str(args.num_reps_eval), "--rep-eval", str(args.rep_eval),
                ] + condition_flag(t["family"], t["cond"])

    def run_one(t):
        t0 = time.time()
        proc = subprocess.run(build_cmd(t), capture_output=True, text=True)
        secs = time.time() - t0
        ok = proc.returncode == 0
        with manifest_lock:
            manifest_w.writerow([t["family"], condition_tag(t["family"], t["cond"]), t["policy"],
                                 t["ma"], t["mb"], t["ms"], "ok" if ok else "FAIL",
                                 f"{secs:.0f}", t["eval_out"]])
            manifest_f.flush()
        return t, ok, secs, proc.stderr

    done = fails = 0
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
                print(f"  FAIL [{t['family']}] cond={t['cond']} {t['policy']} "
                      f"{param_tag(t['ma'], t['mb'], t['ms'])}\n    {tail}")
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
