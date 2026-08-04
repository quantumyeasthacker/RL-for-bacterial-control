"""Single training run for any of the antibiotic-control experiments, logged to wandb.

This replaces the nine run_w_wandb_single_*.py scripts with one entry point. Every experiment
shares the same skeleton -- parse args, build CellConfig/EnvConfig, build the MLP or RNN agent,
train, log to wandb -- and differs only in which env class it instantiates, which EnvConfig
fields it sets, its wandb project, and its trial-name format. Those four differences live in
one `_ExpSpec` per experiment (see EXPERIMENTS at the bottom); everything else is written once.

Trial names and wandb project names are byte-identical to the old per-experiment scripts, so
new runs land in the same output directories and the same wandb projects as existing results.

Usage:
    python train.py <experiment> [flags]

    python train.py --help                    # list experiments
    python train.py constenv-mutate --help    # flags for one experiment

Experiments (old script -> subcommand):
    run_w_wandb_single_constenv.py                  -> constenv
    run_w_wandb_single_constenv_logpop.py           -> constenv-logpop
    run_w_wandb_single_constenv_mutate.py           -> constenv-mutate
    run_w_wandb_single_controlenv.py                -> controlenv
    run_w_wandb_single_generalized.py               -> generalized
    run_w_wandb_single_generalized_holdout_const.py -> generalized-holdout-const
    run_w_wandb_single_generalized_holdout_var.py   -> generalized-holdout-var
    run_w_wandb_single_varenv.py                    -> varenv
    run_w_wandb_single_varenv_mutate.py             -> varenv-mutate

Common flags (all experiments):
    --antibiotic FLOAT    drug level for the "on" action (b_actions = [0, a])   [required]
    --delay-embed INT     observation delay-embed length (use 1 for the RNN agent) [required]
    --rep INT             replicate index                                       [required]
    --results-dir PATH    parent output directory                               [required]
    --episodes INT        training episodes, default 400
    --num-decisions INT   decision steps per episode, default 300
    (episodes/num-decisions are lowered only for smoke tests; leave at the defaults for real runs)

Agent flags (all experiments except constenv-logpop, which is MLP-only):
    --agent {MLP,RNN}     default MLP; RNN is the recurrent RNN_full / r2d2 agent
    --rnn-type {LSTM,GRU} recurrent cell, default LSTM        (only used when --agent RNN)
    --net-arch {rnn,encoder_decoder}  default rnn             (only used when --agent RNN)
    --train-unroll-len INT  R2D2 unroll length, default 20    (only used when --agent RNN)
    --batch-size INT      default 512
    --buffer-size FLOAT   default 1e6  (accepts 1e4-style values)
    --learning-rate FLOAT Q-network learning rate, default 1e-4

Context flags (constenv-mutate, varenv-mutate only):
    The agent optionally observes the slow proteome context (population-average phi_S or
    phiS_max), re-measured only every --context-freq decision steps and held fixed in between.
    This is an env-side observation block, so it applies to the MLP and RNN agents alike.
    --context-freq INT    decision steps between re-measurements; 0 (default) disables the
                          context block entirely (original observation)
    --no-context-age      ablate the normalized age of the latched context. Without the age the
                          refresh is an unobservable jump and the augmented state is not Markov.
    --context-signal {phi_S,phiS_max,noise,noise_phiSmax}   default phi_S.
                          "phi_S" = the fast stress-protein fraction; "phiS_max" = the evolvable
                          ceiling mutation acts on; "noise"/"noise_phiSmax" = INFORMATION-FREE
                          CONTROLS drawn from the corresponding marginal and latched/aged
                          identically, which separate "the signal is informative" from "the extra
                          input units helped". Tagged phiSmax / rand / randPhiSmax in the name.

Output (under <results-dir>/<trial_name>/):
    episode_<n>/{q_1,q_2,q_target_1,q_target_2} checkpoints, Eval/*.jpg, reward_Q_loss.jpg,
    and wandb logging of all config params + eval metrics.
    trial_name is per-experiment; see the `name` lambda of each _ExpSpec below.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import wandb

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import (
    EnvConfig,
    ConstantNutrientEnv,
    VariableNutrientEnv,
    ControlNutrientEnv,
    GeneralizedAgentEnv,
)
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl.agent.RNN_full import CDQL as CDQL_RNN
from rlBacterialControl import wandb_auth


# ----------------------------------------------------------------------------------------- #
# experiment specification
# ----------------------------------------------------------------------------------------- #

@dataclass
class _ExpSpec:
    """Everything that distinguishes one experiment from another.

    project   : wandb project the run is logged to
    env_cls   : env class instantiated as env_cls(env_config, cell_config)
    add_args  : adds the experiment's own flags to its subparser
    configure : args -> (CellConfig, EnvConfig); the experiment-specific env setup
    name      : args -> trial name (also the output subdirectory and the wandb run name)
    extra     : args -> experiment-specific wandb config keys (beyond the common/agent ones)
    num_evals : evaluation episodes per eval round
    mlp_only  : True for experiments with no RNN variant (no agent flags on the subparser)
    """
    project: str
    env_cls: type
    configure: Callable
    name: Callable
    add_args: Callable = lambda p: None
    extra: Callable = lambda a: {}
    num_evals: int = 5
    mlp_only: bool = False


def _int_list(s):
    """Parse the underscore-joined int lists the generalized experiments use ("10_20_30")."""
    return [int(v) for v in s.split("_")]


def _float_list(s):
    """Parse the underscore-joined float lists the generalized/control experiments use."""
    return [float(v) for v in s.split("_")]


# ----------------------------------------------------------------------------------------- #
# shared pieces
# ----------------------------------------------------------------------------------------- #

def _agent_tag(args):
    """Agent half of the trial name: MLP, LSTM, GRU, or e.g. LSTM_encdec."""
    tag = args.rnn_type if args.agent == "RNN" else "MLP"
    if args.agent == "RNN" and args.net_arch == "encoder_decoder":
        tag += "_encdec"
    return tag


def _context_tag(args):
    """Context half of the trial name; empty when the context block is off.

    The context is an env-side observation block, so it tags independently of the agent.
    """
    if not args.context_freq:
        return ""
    tag = f"_ctx{args.context_freq}" + ("" if args.context_age else "noage")
    if args.context_signal == "noise":
        tag += "rand"            # control arm which lacks dynamic readout, only gives average
    elif args.context_signal == "noise_phiSmax":
        tag += "randPhiSmax"     # same control, calibrated to the phiS_max marginal instead
    elif args.context_signal == "phiS_max":
        tag += "phiSmax"         # tracks phiS_max_ave, i.e. the mutating value directly
    return tag


def _context_env_kwargs(args):
    """The four EnvConfig fields the context block needs, shared by the mutate experiments."""
    return dict(
        context_observation=args.context_freq > 0,
        # the env requires a period >= 1 even when the context block is switched off
        context_update_freq=max(args.context_freq, 1),
        context_age_observation=args.context_age,
        context_signal=args.context_signal,
    )


def _context_wandb(args):
    on = args.context_freq > 0
    return {"context_observation": on,
            "context_update_freq": args.context_freq if on else None,
            "context_age_observation": args.context_age if on else None,
            "context_signal": args.context_signal if on else None}


def _build_agent(env, args, use_gpu=False):
    """The one MLP-vs-RNN branch, shared by every experiment."""
    common = dict(buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  train_freq=1,
                  gradient_steps=1,
                  use_gpu=use_gpu,
                  learning_rate=args.learning_rate)
    if args.agent == "RNN":
        return CDQL_RNN(env,
                        rnn_type=args.rnn_type,
                        train_unroll_len=args.train_unroll_len,
                        net_arch=args.net_arch,
                        **common)
    return CDQL(env, **common)


def _agent_wandb(args):
    return {"agent_type": args.agent,
            "rnn_type": args.rnn_type if args.agent == "RNN" else None,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "train_unroll_len": args.train_unroll_len if args.agent == "RNN" else None,
            "net_arch": args.net_arch if args.agent == "RNN" else None,
            "learning_rate": args.learning_rate}


# ----------------------------------------------------------------------------------------- #
# per-experiment definitions
# ----------------------------------------------------------------------------------------- #

def _add_nutrient(p):
    p.add_argument("--nutrient", type=float, required=True,
                   help="constant nutrient concentration k_n0")


def _add_mutate(p):
    p.add_argument("--mutate-prob", type=float, required=True,
                   help="per-division mutation probability (CellConfig mutate=True)")
    p.add_argument("--context-freq", type=int, default=0,
                   help="decision steps between context re-measurements; 0 disables the block")
    p.add_argument("--context-age", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    p.add_argument("--no-context-age", dest="context_age", action="store_false",
                   help="ablate the normalized age of the latched context")
    p.add_argument("--context-signal", default="phi_S",
                   choices=["phi_S", "phiS_max", "noise", "noise_phiSmax"],
                   help="what the context block carries (default phi_S)")


def _add_period(p):
    p.add_argument("--period", type=int, required=True, dest="T_k_n0",
                   help="nutrient oscillation period T_k_n0")


def _add_generalized(p):
    p.add_argument("--constant-nutrient", required=True,
                   help="underscore-joined constant nutrient levels, e.g. 2.0_2.55_3.0")
    p.add_argument("--periods", required=True,
                   help="underscore-joined oscillation periods, e.g. 10_20_30")


# --- constant nutrient ---------------------------------------------------------------------

def _constenv(args):
    return CellConfig(), EnvConfig(
        k_n0_observation=False,        # nutrient is constant -> not observed
        b_observation=True,
        k_n0_constant=args.nutrient,
        delay_embed_len=args.delay_embed,
        b_actions=[0, args.antibiotic],
    )


def _constenv_logpop(args):
    # reward/cost is log10(population) in both observation variants of this experiment
    return CellConfig(), EnvConfig(
        k_n0_observation=False,        # nutrient is constant -> not observed
        b_observation=True,            # antibiotic (drug) history IS observed, as before
        k_n0_constant=args.nutrient,
        delay_embed_len=args.delay_embed,
        b_actions=[0, args.antibiotic],
        obs_type=args.obs_type,
        reward_type="log10_pop",
    )


def _constenv_mutate(args):
    cell = CellConfig(mutate=True, mutate_prob=args.mutate_prob)
    return cell, EnvConfig(
        k_n0_observation=False,        # nutrient is constant -> not observed
        b_observation=True,
        delay_embed_len=args.delay_embed,
        b_actions=[0, args.antibiotic],
        k_n0_constant=args.nutrient,
        # ConstantNutrientEnv.reset() forces k_n0_init to k_n0_constant (with a warning) if they
        # disagree; set it here so the run starts already at the operating nutrient level.
        k_n0_init=args.nutrient,
        **_context_env_kwargs(args),
    )


# --- variable nutrient ---------------------------------------------------------------------

def _varenv(args):
    return CellConfig(), EnvConfig(
        k_n0_observation=False,
        b_observation=True,
        delay_embed_len=args.delay_embed,
        b_actions=[0, args.antibiotic],
        T_k_n0=args.T_k_n0,
        k_n0_mean=2.55,
        sigma_kn0=0.1,
    )


def _varenv_mutate(args):
    cell = CellConfig(mutate=True, mutate_prob=args.mutate_prob)
    return cell, EnvConfig(
        k_n0_observation=False,
        b_observation=True,
        delay_embed_len=args.delay_embed,
        b_actions=[0, args.antibiotic],
        T_k_n0=args.T_k_n0,
        k_n0_mean=2.55,
        sigma_kn0=0.1,
        **_context_env_kwargs(args),
    )


def _varenv_name(args):
    # For RNN agents the trial name encodes train_unroll_len (ul<n>) so runs that differ only in
    # unroll length get distinct output dirs and cannot overwrite each other's checkpoints.
    # MLP naming is unchanged (train_unroll_len does not apply).
    stem = f"a{args.antibiotic:.2f}_T{args.T_k_n0}_delay{args.delay_embed}_{_agent_tag(args)}"
    if args.agent == "RNN":
        stem += f"_ul{args.train_unroll_len}"
    return f"{stem}_rep{args.rep}"


# --- nutrient control ----------------------------------------------------------------------

def _controlenv(args):
    k_n0_actions = _float_list(args.nutrient_range)
    return CellConfig(), EnvConfig(
        b_observation=args.b_observation,
        k_n0_observation=args.k_n0_observation,
        delay_embed_len=args.delay_embed,
        k_n0_actions=k_n0_actions,
        b_actions=[0, args.antibiotic],
        num_actions=len(k_n0_actions) * 2,
    )


def _add_controlenv(p):
    p.add_argument("--nutrient-range", required=True,
                   help="underscore-joined selectable nutrient levels, e.g. 2.0_2.55_3.0")
    p.add_argument("--b-observation", type=lambda s: s == "True", required=True,
                   help="True/False: observe the antibiotic history")
    p.add_argument("--k-n0-observation", type=lambda s: s == "True", required=True,
                   help="True/False: observe the nutrient level")


# --- generalized agent ---------------------------------------------------------------------

def _generalized(args, **holdout):
    return CellConfig(), EnvConfig(
        k_n0_observation=False,
        b_observation=True,
        k_n0_constant=_float_list(args.constant_nutrient),
        delay_embed_len=args.delay_embed,
        b_actions=[0, args.antibiotic],
        T_k_n0=_int_list(args.periods),
        k_n0_mean=2.55,
        sigma_kn0=0.1,
        **holdout,
    )


def _generalized_name(args, trained_env):
    return (f"a{args.antibiotic:.2f}_{trained_env}_delay{args.delay_embed}"
            f"_episodes{args.episodes}_rep{args.rep}")


EXPERIMENTS = {
    "constenv": _ExpSpec(
        project="antibioticRL-constant-nutrient-delays-record",
        env_cls=ConstantNutrientEnv,
        add_args=_add_nutrient,
        configure=_constenv,
        name=lambda a: "a%.2f_n%.2f_delay%d_rep%d" % (
            a.antibiotic, a.nutrient, a.delay_embed, a.rep),
        extra=lambda a: {"nutrient_value": a.nutrient},
    ),

    "constenv-logpop": _ExpSpec(
        project="antibioticRL-constant-nutrient-logpop",
        env_cls=ConstantNutrientEnv,
        add_args=lambda p: (_add_nutrient(p), p.add_argument(
            "--obs-type", default="log10_pop", choices=["log10_pop", "growth_rate"],
            help="bacterial observation signal (default log10_pop)")),
        configure=_constenv_logpop,
        # learning rate is encoded in the trial name so runs that differ only in lr get distinct
        # output dirs / wandb names and cannot overwrite each other's checkpoints.
        name=lambda a: "a%.2f_n%.2f_delay%d_%s_lr%.0e_rep%d" % (
            a.antibiotic, a.nutrient, a.delay_embed, a.obs_type, a.learning_rate, a.rep),
        extra=lambda a: {"nutrient_value": a.nutrient,
                         "obs_type": a.obs_type,
                         "reward_type": "log10_pop"},
        mlp_only=True,
    ),

    "constenv-mutate": _ExpSpec(
        project="antibioticRL-constenv-mutate",
        env_cls=ConstantNutrientEnv,
        add_args=lambda p: (_add_nutrient(p), _add_mutate(p)),
        configure=_constenv_mutate,
        name=lambda a: (f"a{a.antibiotic:.2f}_n{a.nutrient:.2f}_delay{a.delay_embed}"
                        f"_mutprob{a.mutate_prob}_{_agent_tag(a)}{_context_tag(a)}_rep{a.rep}"),
        extra=lambda a: {"nutrient_value": a.nutrient,
                         "mutate_prob": a.mutate_prob,
                         **_context_wandb(a)},
        num_evals=10,
    ),

    "controlenv": _ExpSpec(
        project="antibioticRL-control-nutrient-delay-30-record",
        env_cls=ControlNutrientEnv,
        add_args=_add_controlenv,
        configure=_controlenv,
        name=lambda a: (f"a{a.antibiotic:.2f}_n{a.nutrient_range}_b{a.b_observation}"
                        f"_k{a.k_n0_observation}_delay{a.delay_embed}_rep{a.rep}"),
        extra=lambda a: {"nutrient_range": a.nutrient_range,
                         "b_observation": a.b_observation,
                         "k_n0_observation": a.k_n0_observation},
    ),

    "generalized": _ExpSpec(
        project="antibioticRL-generalized-agent-delays-record",
        env_cls=GeneralizedAgentEnv,
        add_args=_add_generalized,
        configure=_generalized,
        name=lambda a: _generalized_name(
            a, f"const{a.constant_nutrient}_T{a.periods}"),
        extra=lambda a: {"constant_nutrient": _float_list(a.constant_nutrient),
                         "T_k_n0": _int_list(a.periods)},
    ),

    "generalized-holdout-const": _ExpSpec(
        project="antibioticRL-generalized-agent-holdout_const",
        env_cls=GeneralizedAgentEnv,
        add_args=lambda p: (_add_generalized(p), p.add_argument(
            "--holdout", required=True,
            help="underscore-joined nutrient interval held out of training, e.g. 2.4_2.7")),
        configure=lambda a: _generalized(a, hold_out_range_const=_float_list(a.holdout)),
        name=lambda a: _generalized_name(
            a, f"const{a.constant_nutrient}_hold{a.holdout}_T{a.periods}"),
        extra=lambda a: {"constant_nutrient": _float_list(a.constant_nutrient),
                         "T_k_n0": _int_list(a.periods),
                         "const_holdout": _float_list(a.holdout)},
    ),

    "generalized-holdout-var": _ExpSpec(
        project="antibioticRL-generalized-agent-holdout_var",
        env_cls=GeneralizedAgentEnv,
        add_args=lambda p: (_add_generalized(p), p.add_argument(
            "--holdout", required=True,
            help="underscore-joined period interval held out of training, e.g. 20_30")),
        configure=lambda a: _generalized(a, hold_out_range_var=_int_list(a.holdout)),
        name=lambda a: _generalized_name(
            a, f"const{a.constant_nutrient}_T{a.periods}_hold{a.holdout}"),
        extra=lambda a: {"constant_nutrient": _float_list(a.constant_nutrient),
                         "T_k_n0": _int_list(a.periods),
                         "T_holdout": _int_list(a.holdout)},
    ),

    "varenv": _ExpSpec(
        project="antibioticRL-varenv-nutrient-delay-30-record",
        env_cls=VariableNutrientEnv,
        add_args=_add_period,
        configure=_varenv,
        name=_varenv_name,
        extra=lambda a: {"T_k_n0": a.T_k_n0},
    ),

    "varenv-mutate": _ExpSpec(
        project="antibioticRL-varenv-mutate",
        env_cls=VariableNutrientEnv,
        add_args=lambda p: (_add_period(p), _add_mutate(p)),
        configure=_varenv_mutate,
        name=lambda a: (f"a{a.antibiotic:.2f}_T{a.T_k_n0}_delay{a.delay_embed}"
                        f"_mutprob{a.mutate_prob}_{_agent_tag(a)}{_context_tag(a)}_rep{a.rep}"),
        extra=lambda a: {"T_k_n0": a.T_k_n0,
                         "mutate_prob": a.mutate_prob,
                         **_context_wandb(a)},
        num_evals=10,
    ),
}


# ----------------------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------------------- #

def _parse_args(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--antibiotic", type=float, required=True,
                        help='drug level for the "on" action (b_actions = [0, a])')
    common.add_argument("--delay-embed", type=int, required=True,
                        help="observation delay-embed length (use 1 for the RNN agent)")
    common.add_argument("--rep", type=int, required=True, help="replicate index")
    common.add_argument("--results-dir", required=True, help="parent output directory")
    common.add_argument("--episodes", type=int, default=400, help="training episodes")
    common.add_argument("--num-decisions", type=int, default=300,
                        help="decision steps per episode")

    # hyperparameters every agent takes, MLP-only experiments included
    agent = argparse.ArgumentParser(add_help=False)
    agent.add_argument("--batch-size", type=int, default=512)
    # float()-then-int() so 1e6-style values are accepted
    agent.add_argument("--buffer-size", type=lambda s: int(float(s)), default=1_000_000,
                       help="replay buffer size, default 1e6 (accepts 1e4-style values)")
    agent.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Q-network learning rate, default 1e-4")

    # only for experiments that have an RNN variant (see _ExpSpec.mlp_only)
    rnn = argparse.ArgumentParser(add_help=False)
    rnn.add_argument("--agent", default="MLP", type=str.upper, choices=["MLP", "RNN"],
                     help="MLP (default) or RNN (recurrent RNN_full / r2d2 agent)")
    rnn.add_argument("--rnn-type", default="LSTM", type=str.upper, choices=["LSTM", "GRU"],
                     help="recurrent cell, default LSTM (only used when --agent RNN)")
    rnn.add_argument("--net-arch", default="rnn", type=str.lower,
                     choices=["rnn", "encoder_decoder"],
                     help="default rnn (only used when --agent RNN)")
    rnn.add_argument("--train-unroll-len", type=int, default=20,
                     help="R2D2 unroll length, default 20 (only used when --agent RNN)")

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Single antibiotic-control training run, logged to wandb.")
    subs = parser.add_subparsers(dest="experiment", required=True, metavar="experiment")
    for name, spec in EXPERIMENTS.items():
        parents = [common, agent] if spec.mlp_only else [common, agent, rnn]
        sub = subs.add_parser(name, parents=parents, help=spec.project)
        spec.add_args(sub)

    args = parser.parse_args(argv)
    if EXPERIMENTS[args.experiment].mlp_only:
        # MLP-only experiments still go through _build_agent/_agent_wandb, which read these.
        args.agent, args.rnn_type, args.net_arch, args.train_unroll_len = \
            "MLP", "LSTM", "rnn", 20
    return args


def main(argv=None):
    args = _parse_args(argv)
    spec = EXPERIMENTS[args.experiment]

    ## ----- wandb setting ----- ##
    trial_name = spec.name(args)
    folder_name = f"{args.results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": args.antibiotic,
                    "delay_embed_len": args.delay_embed,
                    "rep": args.rep,
                    "episodes": args.episodes,
                    "num_decisions": args.num_decisions,
                    **_agent_wandb(args),
                    **spec.extra(args)}

    # credentials come from ~/.config/wandb_rl/credentials.json (or $WANDB_CREDENTIALS),
    # else $WANDB_API_KEY/$WANDB_ENTITY, else the local ~/.netrc from `wandb login`.
    wandb_auth.init(project=spec.project,
                    dir=folder_name,
                    name=str(trial_name),
                    config=wandb_config,
                    settings=wandb.Settings(symlink=False))

    ## ----- RL setting ----- ##
    cell_config, env_config = spec.configure(args)
    env = spec.env_cls(env_config, cell_config)
    c = _build_agent(env, args, use_gpu=False)

    ## ----- RL training ----- ##
    c.train(episodes=args.episodes,
            num_decisions=args.num_decisions,
            num_evals=spec.num_evals,
            folder_name=folder_name)

    wandb.finish()
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
