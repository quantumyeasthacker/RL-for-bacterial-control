"""Apply an already-trained agent to an env with imperfect (noisy / lagged) measurements.

The agent and its weights are untouched: it was trained on perfect sensing and is here
evaluated on a copy of the same env whose *observations* are corrupted by
  - measurement noise: log-normal multiplicative error on the population count, propagated
    into the growth-rate observation (--noise), and/or
  - time lag: the growth-rate measurement reaches the agent --lag decision steps late.
The underlying cell dynamics and the reward/log are NOT corrupted, so info['log'] still
records the true trajectory the (mis-)informed agent actually produced. Note that for the
control env the nutrient (k_n0) is the agent's own *action*, not a measurement, so it stays
clean -- only the bacterial growth-rate measurement is corrupted (true for every env type).

Loads the same trained-model folders as the clean eval scripts (same trial_name layout) and
writes to 'results/eval_noisy/{env_group}/{trial}/{episode}/noise{n}_lag{l}/' (override the root
with --out-dir; env_group is the source folder's basename) so sweeps never collide. --episode
defaults to 'last' (auto-detect the final saved episode_N). Running with --noise 0 --lag 0
reproduces the clean baseline through the exact same code path.

Examples (run from the repo root):

  # constant-nutrient agent (n=1.0), 15% count noise, lag of 3 steps, last episode
  python scripts/eval/eval_trained_agents_noisy.py const \
      --results-dir results/train/results_delay_30_record_constenv \
      --nutrient 1.0 --rep-run 0 --noise 0.15 --lag 3

  # variable-nutrient agent (T=12)
  python scripts/eval/eval_trained_agents_noisy.py var \
      --results-dir results/train/results_delay_30_record_varenv \
      --nutrient 12 --rep-run 0 --noise 0.15 --lag 3

  # control agent (nutrient action set 1_2_3_4, observes k_n0)
  python scripts/eval/eval_trained_agents_noisy.py control \
      --results-dir results/train/results_delay_30_record_controlenv \
      --nutrient-range 1_2_3_4 --b-obs True --k-obs True --rep-run 0 --noise 0.15 --lag 3

  # generalized agent, evaluated on a constant env at k_n0=2.0
  python scripts/eval/eval_trained_agents_noisy.py gen \
      --results-dir results/train/results_delay_30_record_generalized \
      --trained-env const1_4_T6_24 --episodes 400 \
      --eval-env constenv --eval-variable 2.0 --rep-run 0 --noise 0.15 --lag 3
"""
import os
import argparse
import pickle
import numpy as np

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import (
    EnvConfig, ConstantNutrientEnv, VariableNutrientEnv, ControlNutrientEnv,
)
from rlBacterialControl.agent.MLP_full import CDQL


def resolve_episode(trial_dir, episode):
    """Return the episode sub-folder to load. 'last'/'auto' picks the highest-numbered
    episode_N dir (the final saved model); otherwise the literal value is used as-is."""
    if episode not in ("last", "auto"):
        return episode
    eps = []
    for name in os.listdir(trial_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(trial_dir, name)):
            try:
                eps.append((int(name.split("_")[1]), name))
            except ValueError:
                continue
    if not eps:
        raise FileNotFoundError(f"No episode_* sub-folders found in {trial_dir}")
    return max(eps)[1]


def require(args, names):
    missing = [f"--{n.replace('_', '-')}" for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"env '{args.env_type}' requires: {', '.join(missing)}")


def build(args):
    """Return (env, trial_name, eval_suffix) for the requested env type.

    trial_name -> folder holding the trained model (matches the training scripts).
    eval_suffix -> sub-path under '{results_dir}_eval_noisy/' for the outputs.
    The measurement-corruption knobs (meas_noise_std, meas_lag) are the only thing
    that differs from the clean eval; everything else mirrors the clean eval scripts.
    """
    a, d, r = args.antibiotic, args.delay, args.rep_run
    noise_lag = dict(meas_noise_std=args.noise, meas_lag=args.lag)

    if args.env_type == "const":
        require(args, ["nutrient"])
        trial_name = "a%.2f_n%.2f_delay%d_rep%d" % (a, args.nutrient, d, r)
        cfg = EnvConfig(
            k_n0_observation=False, b_observation=True, k_n0_constant=args.nutrient,
            delay_embed_len=d, b_actions=[0, a], max_pop=np.inf, **noise_lag,
        )
        return ConstantNutrientEnv(cfg, CellConfig()), trial_name, trial_name

    if args.env_type == "var":
        require(args, ["nutrient"])
        trial_name = f"a{a:.2f}_T{int(args.nutrient)}_delay{d}_rep{r}"
        cfg = EnvConfig(
            k_n0_observation=False, b_observation=True, delay_embed_len=d,
            b_actions=[0, a], T_k_n0=int(args.nutrient), k_n0_mean=2.55, sigma_kn0=0.1,
            max_pop=np.inf, **noise_lag,
        )
        return VariableNutrientEnv(cfg, CellConfig()), trial_name, trial_name

    if args.env_type == "control":
        require(args, ["nutrient_range"])
        b_obs = args.b_obs == "True"
        k_obs = args.k_obs == "True"
        trial_name = f"a{a:.2f}_n{args.nutrient_range}_b{b_obs}_k{k_obs}_delay{d}_rep{r}"
        k_n0_actions = [float(x) for x in args.nutrient_range.split("_")]
        cfg = EnvConfig(
            k_n0_observation=k_obs, b_observation=b_obs, delay_embed_len=d,
            b_actions=[0, a], k_n0_actions=k_n0_actions, num_actions=len(k_n0_actions) * 2,
            k_n0_mean=2.55, sigma_kn0=0.1, max_pop=np.inf, **noise_lag,
        )
        return ControlNutrientEnv(cfg, CellConfig()), trial_name, trial_name

    if args.env_type == "gen":
        require(args, ["trained_env", "episodes", "eval_env", "eval_variable"])
        trial_name = f"a{a:.2f}_{args.trained_env}_delay{d}_episodes{args.episodes}_rep{r}"
        eval_suffix = f"{trial_name}_{args.eval_env}_{args.eval_variable}"
        if args.eval_env == "constenv":
            cfg = EnvConfig(
                k_n0_observation=False, b_observation=True,
                k_n0_constant=float(args.eval_variable), delay_embed_len=d,
                b_actions=[0, a], max_pop=np.inf, **noise_lag,
            )
            env = ConstantNutrientEnv(cfg, CellConfig())
        else:  # varenv
            cfg = EnvConfig(
                k_n0_observation=False, b_observation=True, delay_embed_len=d,
                b_actions=[0, a], T_k_n0=int(args.eval_variable), k_n0_mean=2.55,
                sigma_kn0=0.1, max_pop=np.inf, **noise_lag,
            )
            env = VariableNutrientEnv(cfg, CellConfig())
        return env, trial_name, eval_suffix

    raise SystemExit(f"unknown env_type {args.env_type!r}")


def main():
    p = argparse.ArgumentParser(description="Eval a trained agent under noisy/lagged measurements.")
    p.add_argument("env_type", choices=["const", "var", "control", "gen"])
    p.add_argument("--results-dir", required=True, help="dir holding the trained-model trial folders")
    p.add_argument("--noise", type=float, required=True, help="meas_noise_std (log-normal count noise); 0 = off")
    p.add_argument("--lag", type=int, required=True, help="meas_lag (decision steps the measurement is delayed); 0 = off")
    p.add_argument("--episode", default="last", help="episode_N to load, or 'last' to auto-detect (default)")
    p.add_argument("--antibiotic", type=float, default=3.72)
    p.add_argument("--delay", type=int, default=30, help="delay_embed_len")
    p.add_argument("--rep-run", type=int, default=0, help="training replicate to load")
    p.add_argument("--rep-eval", type=int, default=0, help="eval-batch offset for output file naming")
    p.add_argument("--num-decisions", type=int, default=300)
    p.add_argument("--num-reps-eval", type=int, default=10)
    p.add_argument("--out-dir", default="results/eval_noisy", help="root dir for outputs (default results/eval_noisy)")
    # const / var
    p.add_argument("--nutrient", type=float, help="const: k_n0 value ; var: T_k_n0 period")
    # control
    p.add_argument("--nutrient-range", help='control: nutrient action set, e.g. "1_2_3_4"')
    p.add_argument("--b-obs", default="True", help="control: b_observation ('True'/'False')")
    p.add_argument("--k-obs", default="False", help="control: k_n0_observation ('True'/'False')")
    # gen
    p.add_argument("--trained-env", help='gen: trained-env tag, e.g. "const1_4_T6_24"')
    p.add_argument("--episodes", type=int, help="gen: total training episodes in the trial name")
    p.add_argument("--eval-env", choices=["constenv", "varenv"], help="gen: env to evaluate on")
    p.add_argument("--eval-variable", help="gen: k_n0 (constenv) or T period (varenv)")
    args = p.parse_args()

    env, trial_name, eval_suffix = build(args)

    trial_dir = f"{args.results_dir}/{trial_name}"
    episode = resolve_episode(trial_dir, args.episode)
    folder_name = f"{trial_dir}/{episode}/"
    # mirror the source env-type folder under the output root so models and their
    # noisy evals stay parallel, e.g. results/eval_noisy/results_delay_30_record_constenv/...
    env_group = os.path.basename(args.results_dir.rstrip("/"))
    eval_out = f"{args.out_dir}/{env_group}/{eval_suffix}/{episode}/noise{args.noise}_lag{args.lag}/"
    os.makedirs(eval_out, exist_ok=True)
    print(f"Loading {folder_name} | noise={args.noise} lag={args.lag} -> {eval_out}")

    c = CDQL(env, buffer_size=1_000_000, batch_size=512, use_gpu=False)
    c.load_data(folder_name, False)
    for i_eval in range(args.num_reps_eval):
        _, _, _, _, info = c.eval_step(num_decisions=args.num_decisions)
        fname = f"trial_{args.rep_eval * args.num_reps_eval + i_eval}"
        with open(os.path.join(eval_out, fname + "tcbk.pkl"), "wb") as f:
            pickle.dump(info, f)
    print("Done")


if __name__ == "__main__":
    main()