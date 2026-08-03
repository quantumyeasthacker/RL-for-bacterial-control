"""Evaluate a trained agent under mutation, on either the constant- or variable-nutrient
environment, sweeping the EVAL-time mutation probability.

Usage:
    python eval_trained_agents_generalized_mutate.py \\
        antibiotic_value trained_env delay_embed_len eval_env eval_variable \\
        rep_run rep_eval results_dir episodes training_episode mutate_prob \\
        [agent] [rnn_type] [net_arch] [train_mutprob] [context_update_freq] [context_age] \\
        [context_signal] [num_decisions]

Positional arguments:
    antibiotic_value : float   antibiotic dose used for the b=1 action
    trained_env      : str     trained-model env tag in the folder name, e.g. "T6"
    delay_embed_len  : int     observation delay-embed length (use 1 for the RNN agent)
    eval_env         : str     "constenv" or "varenv" -- environment to evaluate in
    eval_variable    : str     constenv: constant k_n0 value; varenv: switching period T
    rep_run          : int     training replicate index (selects the trained-model folder)
    rep_eval         : int     eval batch index (offsets the saved trial_* filenames)
    results_dir      : str     directory holding the trained-model folders
    episodes         : int     episode count baked into the MLP folder name (only used to
                               build the MLP folder name; IGNORED when agent == "RNN")
    training_episode : str     episode sub-folder of the checkpoint to load (e.g. episode_399)
    mutate_prob      : float   EVAL-time per-division mutation probability (CellConfig);
                               this is the quantity being swept, independent of training

Optional arguments (default to the original MLP behaviour, so existing calls are unchanged):
    agent            : "MLP" (default) or "RNN" (recurrent RNN_full / r2d2 agent)
    rnn_type         : "LSTM" (default) or "GRU"  (only used when agent == "RNN")
    net_arch         : "rnn" (default) or "encoder_decoder"  (only used when agent == "RNN")
    train_mutprob    : str, the TRAINING mutation probability baked into the trained-model
                       folder name (e.g. "0.1"). REQUIRED when agent == "RNN". Both mutate
                       training scripts tag folders as
                       "a{ab}_{env}_delay{d}_mutprob{train_mutprob}_{agent_tag}{ctx_tag}_rep{rep}".
                       Distinct from `mutate_prob` (the eval-time rate swept independently of
                       how the model was trained).
                       For agent == "MLP" this also SELECTS THE FOLDER-NAMING SCHEME:
                         omitted -> "a{ab}_{env}_delay{d}_episodes{episodes}_rep{rep}"
                                    (models from run_w_wandb_single_generalized.py)
                         given   -> "a{ab}_{env}_delay{d}_mutprob{p}_MLP{ctx_tag}_rep{rep}"
                                    (models from run_w_wandb_single_varenv_mutate.py)
    context_update_freq : int, default 0. The slow proteome-context setting the model was
                       TRAINED with. Must match: it sets the observation width, hence the
                       network input size that load_data() restores. 0 = no context block.
    context_age      : int, default 1. Whether the trained model also observed the normalized
                       context age. Must match training (1 = yes, 0 = ablated).
    context_signal   : str, default "phi_S". WHAT the context block carried during training:
                         "phi_S"        the population-average stress-protein fraction
                         "phiS_max"     the population-average evolvable ceiling (the trait
                                        mutation actually acts on)
                         "noise"        the information-free control (independent draw,
                                        calibrated to phi_S's marginal)
                         "noise_phiSmax" the information-free control calibrated to phiS_max's
                                        marginal -- use this one against a "phiS_max" arm
                       MUST match training. All variants share the same observation width, so a
                       mismatch does NOT raise: it silently loads the wrong model (the ctx_tag
                       resolves to a different folder) and/or feeds the wrong signal.
    num_decisions    : int, default 300. Decision steps per eval rollout (each is delta_t = 0.2 h,
                       so 300 -> 60 h of control after the 12 h warm-up). MUST match the
                       constant-antibiotic baseline it is compared against -- the relative-
                       performance metric time-averages the population over the decision phase,
                       so differing horizons are not comparable. Note the eval OUTPUT PATH does
                       not encode this, so run a non-default horizon into its own results_dir
                       (e.g. via symlinked model folders) or it will overwrite the 300-step data.

    The RNN options (rnn_type, net_arch) must match how the model was trained, since they
    determine the network architecture that load_data() restores, and (with train_mutprob)
    they select the trained-model folder. agent_tag = rnn_type, with "_encdec" appended when
    net_arch == "encoder_decoder". ctx_tag = "_ctx{freq}", plus "noage" when context_age == 0,
    plus "rand"/"randPhiSmax"/"phiSmax" for context_signal "noise"/"noise_phiSmax"/"phiS_max";
    "" when context_update_freq == 0.

Output (written under
        "{results_dir}_eval/{trial_name}_{eval_env}_{eval_variable}_mutprob{mutate_prob}/{training_episode}/"):
    trial_<n>tcbk.pkl : pickled eval-trajectory info dict, one per eval rep
                        (n = rep_eval * num_of_reps_eval + i_eval)
    NB: for an RNN model trial_name already carries the TRAINING "mutprob..." token, so the
    eval-output folder name ends with two mutprob tokens -- the first is the training rate
    (in trial_name), the trailing one is the eval rate being swept.
"""

import os
import sys
import pickle
import numpy as np

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv, VariableNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl.agent.RNN_full import CDQL as CDQL_RNN


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    trained_env = sys.argv[2]
    delay_embed_len = int(sys.argv[3])
    eval_env = sys.argv[4]
    eval_variable = sys.argv[5]
    rep_run = int(sys.argv[6])
    rep_eval = int(sys.argv[7])
    results_dir = sys.argv[8]
    episodes = int(sys.argv[9])
    training_episode = sys.argv[10]
    mutate_prob = float(sys.argv[11])

    ## ----- optional agent selection (backward compatible: defaults to MLP) ----- ##
    agent_type = sys.argv[12].upper() if len(sys.argv) > 12 else "MLP"
    rnn_type = sys.argv[13].upper() if len(sys.argv) > 13 else "LSTM"
    net_arch = sys.argv[14].lower() if len(sys.argv) > 14 else "rnn"  # "rnn" or "encoder_decoder"
    train_mutprob = sys.argv[15] if len(sys.argv) > 15 else None
    # context observation the model was TRAINED with; must match, since it sets the network
    # input width that load_data() restores (0 = the model saw no context block).
    context_update_freq = int(sys.argv[16]) if len(sys.argv) > 16 else 0
    context_age = bool(int(sys.argv[17])) if len(sys.argv) > 17 else True
    context_signal = sys.argv[18] if len(sys.argv) > 18 else "phi_S"  # can also be "noise" or "phiS_max"
    # decision steps per eval rollout. MUST match the constant-antibiotic baseline this eval is
    # compared against (simulation_test_mutate.py's num_decisions): the relative-performance
    # metric time-averages the population over the decision phase, so a policy evaluated over a
    # different horizon than its baseline is not comparable.
    num_decisions = int(sys.argv[19]) if len(sys.argv) > 19 else 300

    context_observation = context_update_freq > 0
    # mirrors the training script's tagging: "_ctx<freq>" plus "noage" when the age was ablated
    ctx_tag = ""
    if context_observation:
        ctx_tag = f"_ctx{context_update_freq}" + ("" if context_age else "noage")
        if context_signal == "noise":
            ctx_tag += "rand"   # control arm which lacks dynamic readout, only gives average
        elif context_signal == "noise_phiSmax":
            ctx_tag += "randPhiSmax"  # same control, calibrated to the phiS_max marginal instead
        elif context_signal == "phiS_max":
            ctx_tag += "phiSmax"    # tracking phiS_max_ave instead of phiS_ave, thus directly tracking mutating value

    ## ----- wandb setting ----- ##
    if agent_type == "RNN":
        # Mirror the RNN-mutate training script's folder tagging (mutprob, rnn_type, _encdec).
        if train_mutprob is None:
            raise SystemExit("train_mutprob (argv[15]) is required when agent == 'RNN'")
        agent_tag = rnn_type
        if net_arch == "encoder_decoder":
            agent_tag += "_encdec"
        trial_name = f"a{antibiotic_value:.2f}_{trained_env}_delay{delay_embed_len}_mutprob{train_mutprob}_{agent_tag}{ctx_tag}_rep{rep_run}"
    elif train_mutprob is not None:
        # MLP model from run_w_wandb_single_varenv_mutate.py, which tags folders with the
        # training mutation rate and an "MLP" agent token rather than "episodes<n>".
        trial_name = f"a{antibiotic_value:.2f}_{trained_env}_delay{delay_embed_len}_mutprob{train_mutprob}_MLP{ctx_tag}_rep{rep_run}"
    else:
        # MLP model from run_w_wandb_single_generalized.py (original naming, unchanged)
        trial_name = f"a{antibiotic_value:.2f}_{trained_env}_delay{delay_embed_len}_episodes{episodes}_rep{rep_run}"
    folder_name = f"{results_dir}/{trial_name}/{training_episode}/"

    eval_out = f"{results_dir}_eval/{trial_name}_{eval_env}_{eval_variable}_mutprob{mutate_prob}/{training_episode}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    k_n0_observation = False
    b_observation = True
    use_gpu = False

    cell_config = CellConfig(mutate=True, mutate_prob=mutate_prob)

    if eval_env == "constenv":
        env_config = EnvConfig(
            k_n0_observation = k_n0_observation,
            b_observation = b_observation,
            k_n0_constant = float(eval_variable),
            delay_embed_len = delay_embed_len,
            b_actions = [0, antibiotic_value],
            max_pop = np.inf,
            context_observation = context_observation,
            context_update_freq = max(context_update_freq, 1), # env requires >= 1 even when off
            context_age_observation = context_age,
            context_signal = context_signal
        )
        env = ConstantNutrientEnv(env_config, cell_config)
    elif eval_env == "varenv":
        env_config = EnvConfig(
            k_n0_observation = k_n0_observation,
            b_observation = b_observation,
            delay_embed_len = delay_embed_len,
            b_actions = [0, antibiotic_value],
            T_k_n0 = int(eval_variable),
            k_n0_mean = 2.55,
            sigma_kn0 = 0.1,
            max_pop = np.inf,
            context_observation = context_observation,
            context_update_freq = max(context_update_freq, 1), # env requires >= 1 even when off
            context_age_observation = context_age,
            context_signal = context_signal
        )
        env = VariableNutrientEnv(env_config, cell_config)

    if agent_type == "RNN":
        c = CDQL_RNN(env,
                     use_gpu = use_gpu,
                     rnn_type = rnn_type,
                     net_arch = net_arch)
    else:
        c = CDQL(env,
                 use_gpu = use_gpu)

    ## ----- RL evaluating ----- ##
    num_of_reps_eval = 10
    c.load_data(folder_name, False)
    for i_eval in range(num_of_reps_eval):
        _, _, _, _, info = c.eval_step(num_decisions=num_decisions)
        fname=f"trial_{rep_eval*num_of_reps_eval+i_eval}"
        with open(os.path.join(eval_out, str(fname)+'tcbk.pkl'), "wb") as f:
            pickle.dump(info, f)
    print("Done")
    sys.exit(0)
