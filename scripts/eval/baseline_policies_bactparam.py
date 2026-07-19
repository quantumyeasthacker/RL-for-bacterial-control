"""Run fixed (non-learned) reference protocols on mis-specified bacteria.

Companion to eval_trained_agents_bactparam.py. That script applies a *trained* agent to a
bacterium whose physiology (alpha/beta/sigma) is scaled away from nominal; this script runs the
paper's hand-designed protocols on the *same* perturbed bacterium, so the learned policy can be
judged against a controller that faces the identical physiology rather than against the nominal
null. Changing the cell parameters changes how the drug bites regardless of who is steering, so
a fair "did the RL policy degrade?" question needs these per-cell references, not the (1,1,1)
baseline.

The perturbation and the on-disk tag come from eval_trained_agents_bactparam.py (imported), so
the dynamics and directory naming can never drift from the RL eval. Protocols require no trained
model -- they depend only on (env dynamics, nutrient condition) -- so a baseline computed for the
constant-nutrient env at k_n0=2 is reused verbatim for the generalized agent evaluated on that
same constenv condition (the driver exploits this; nothing here is gen-specific).

Protocols per env family (paper set only):
  const : no_drug (growth ceiling), constant (drug always on), pulse_h{H} (on/off half-period H).
  var   : no_drug, constant.  (nutrient fluctuates uncontrollably, so a fixed pulse phase is not
          a meaningful reference and is deliberately omitted.)
  control: feast (max nutrient + drug), famine (min nutrient + drug), no_drug (max nutrient, off).

Outputs mirror the RL tree but are keyed by protocol instead of a trained episode:
  results/eval_bactparam_baseline/{family}env/{condition}/{policy}/a{ma}_b{mb}_s{ms}/trial_N tcbk.pkl
The pickled info dict is exactly what the envs emit, so analyze_bactparam's episode_metrics reads
baseline and RL pkls through the same code path.

Examples (run from repo root, with the env python):
  # constant-nutrient references at k_n0=2, beta +20% (harder to kill), the pulsing family
  python scripts/eval/baseline_policies_bactparam.py const --nutrient 2.0 \
      --policy pulse_h20 --beta-mult 1.2

  # control-env feast protocol on the 1_3 nutrient set, alpha -20%
  python scripts/eval/baseline_policies_bactparam.py control --nutrient-range 1_3 \
      --policy feast --alpha-mult 0.8
"""
import os
import io
import argparse
import contextlib
import pickle

import numpy as np

from rlBacterialControl.envs.envs import (
    EnvConfig, ConstantNutrientEnv, VariableNutrientEnv, ControlNutrientEnv,
)
# single source of truth for the perturbation + on-disk tag, shared with the RL eval
from eval_trained_agents_bactparam import make_cell_config, param_tag

# pulsing half-periods swept for the const family (decisions); best member is picked downstream.
PULSE_HALVES = [3, 5, 10, 15, 20, 25, 30, 40]


def condition_tag(family, args):
    """String naming the nutrient condition, matched to how analyze_bactparam parses the RL
    eval_suffix so baseline and RL rows join exactly (const '2.00', var '12', control '1_3')."""
    if family == "const":
        return f"{args.nutrient:.2f}"
    if family == "var":
        return str(int(args.period))
    return args.nutrient_range  # control


def family_policies(family):
    """Names of the fixed protocols for an env family (the driver enumerates these)."""
    if family == "const":
        return ["no_drug", "constant"] + [f"pulse_h{h}" for h in PULSE_HALVES]
    if family == "var":
        return ["no_drug", "constant"]
    return ["no_drug", "feast", "famine"]  # control


def make_env(family, args, cell_cfg):
    """Build the env for a family with the SAME EnvConfig the RL eval uses, minus the trained
    model. delay_embed_len/warm_up/max_pop mirror eval_trained_agents_bactparam.build."""
    a, d = args.antibiotic, args.delay
    if family == "const":
        cfg = EnvConfig(
            k_n0_observation=False, b_observation=True, k_n0_constant=args.nutrient,
            delay_embed_len=d, b_actions=[0, a], max_pop=np.inf,
        )
        return ConstantNutrientEnv(cfg, cell_cfg)
    if family == "var":
        cfg = EnvConfig(
            k_n0_observation=False, b_observation=True, delay_embed_len=d,
            b_actions=[0, a], T_k_n0=int(args.period), k_n0_mean=2.55, sigma_kn0=0.1,
            max_pop=np.inf,
        )
        return VariableNutrientEnv(cfg, cell_cfg)
    # control
    k_n0_actions = [float(x) for x in args.nutrient_range.split("_")]
    cfg = EnvConfig(
        k_n0_observation=False, b_observation=True, delay_embed_len=d,
        b_actions=[0, a], k_n0_actions=k_n0_actions, num_actions=len(k_n0_actions) * 2,
        k_n0_mean=2.55, sigma_kn0=0.1, max_pop=np.inf,
    )
    return ControlNutrientEnv(cfg, cell_cfg)


def drug_schedule(policy, num_decisions):
    """0/1 antibiotic action per decision for the const/var families."""
    if policy == "no_drug":
        return [0] * num_decisions
    if policy == "constant":
        return [1] * num_decisions
    if policy.startswith("pulse_h"):
        h = int(policy.split("pulse_h")[1])
        pattern = ([0] * h + [1] * h) * (num_decisions // (2 * h) + 1)
        return pattern[:num_decisions]
    raise SystemExit(f"unknown drug-schedule policy {policy!r}")


def control_setpoint(policy, args):
    """(k_n0, b) held for the whole episode for a control-env protocol (matches the paper's
    Feast/Famine simulator: nutrient pinned high/low, drug on; no_drug is the growth ceiling)."""
    k_n0_actions = [float(x) for x in args.nutrient_range.split("_")]
    a = args.antibiotic
    if policy == "feast":
        return max(k_n0_actions), a
    if policy == "famine":
        return min(k_n0_actions), a
    if policy == "no_drug":
        return max(k_n0_actions), 0.0
    raise SystemExit(f"unknown control policy {policy!r}")


def run_policy(env, family, policy, args):
    """Run one protocol for num_reps_eval episodes, returning a list of info dicts."""
    infos = []
    if family in ("const", "var"):
        decisions = drug_schedule(policy, args.num_decisions)
    else:
        k_n0, b = control_setpoint(policy, args)
    for i in range(args.num_reps_eval):
        with contextlib.redirect_stdout(io.StringIO()):  # silence fsolve init prints
            env.reset()
            if family in ("const", "var"):
                for act in decisions:
                    _, _, terminated, truncated, info = env.step(act)
                    if terminated or truncated:
                        break
            else:
                for _ in range(args.num_decisions):
                    _, _, terminated, truncated, info = env.step_hardcode(k_n0, b)
                    if terminated or truncated:
                        break
        infos.append(info)
    return infos


def main():
    p = argparse.ArgumentParser(description="Run fixed protocols on mis-specified bacteria.")
    p.add_argument("family", choices=["const", "var", "control"])
    p.add_argument("--policy", required=True, help="protocol name; see family_policies()")
    p.add_argument("--alpha-mult", type=float, default=1.0)
    p.add_argument("--beta-mult", type=float, default=1.0)
    p.add_argument("--sigma-mult", type=float, default=1.0)
    p.add_argument("--antibiotic", type=float, default=3.72)
    p.add_argument("--delay", type=int, default=30)
    p.add_argument("--num-decisions", type=int, default=300)
    p.add_argument("--num-reps-eval", type=int, default=10)
    p.add_argument("--rep-eval", type=int, default=0, help="output-file numbering offset")
    p.add_argument("--out-dir", default="results/eval_bactparam_baseline")
    # const / var / control condition
    p.add_argument("--nutrient", type=float, help="const: k_n0 value")
    p.add_argument("--period", type=float, help="var: T_k_n0 period")
    p.add_argument("--nutrient-range", help='control: nutrient action set, e.g. "1_3"')
    args = p.parse_args()

    if args.policy not in family_policies(args.family):
        raise SystemExit(f"policy {args.policy!r} is not valid for family {args.family!r}; "
                         f"choose from {family_policies(args.family)}")

    cell_cfg = make_cell_config(args.alpha_mult, args.beta_mult, args.sigma_mult)
    env = make_env(args.family, args, cell_cfg)

    cond = condition_tag(args.family, args)
    tag = param_tag(args.alpha_mult, args.beta_mult, args.sigma_mult)
    out = f"{args.out_dir}/{args.family}env/{cond}/{args.policy}/{tag}/"
    os.makedirs(out, exist_ok=True)
    print(f"[{args.family}] cond={cond} policy={args.policy} {tag} -> {out}")

    infos = run_policy(env, args.family, args.policy, args)
    for i, info in enumerate(infos):
        fname = f"trial_{args.rep_eval * args.num_reps_eval + i}"
        with open(os.path.join(out, fname + "tcbk.pkl"), "wb") as f:
            pickle.dump(info, f)
    print("Done")


if __name__ == "__main__":
    main()
