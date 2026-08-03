"""Evaluate a trained agent on the variable-nutrient environment.

Usage:
    python eval_trained_agents_varenv.py \\
        antibiotic_value T_k_n0 delay_embed_len rep_run rep_eval results_dir training_episode \\
        [agent] [rnn_type] [net_arch] [train_unroll_len] [eval_out_base]

Positional arguments:
    antibiotic_value : float   antibiotic dose used for the b=1 action
    T_k_n0           : int     nutrient-switching period of the eval env
    delay_embed_len  : int     observation delay-embed length (use 1 for the RNN agent)
    rep_run          : int     training replicate index (selects the trained-model folder)
    rep_eval         : int     eval batch index (offsets the saved trial_* filenames)
    results_dir      : str     directory holding the trained-model folders
    training_episode : str     episode sub-folder of the checkpoint to load

Optional arguments (default to the original MLP behaviour, so existing calls are unchanged):
    agent            : "MLP" (default) or "RNN" (recurrent RNN_full / r2d2 agent)
    rnn_type         : "LSTM" (default) or "GRU"  (only used when agent == "RNN")
    net_arch         : "rnn" (default) or "encoder_decoder"  (only used when agent == "RNN")
    train_unroll_len : int, default 20. Selects the RNN model folder (trial_name encodes
                       "ul<n>"); must match how the model was trained. Ignored for MLP.
    eval_out_base    : str, output base dir for eval results. Defaults to "{results_dir}_eval".
                       Pass this to send eval output somewhere other than next to the model
                       (e.g. to pool evals of models that live in different results_dirs).

    NB: this script has no proteome-context options on purpose. Context models are produced
    only by run_w_wandb_single_varenv_mutate.py, whose folders carry a "mutprob" token that the
    trial_name below does not build; they are evaluated by eval_trained_agents_generalized_mutate.py.
    The models this script serves come from run_w_wandb_single_varenv.py, which has no context.

    The RNN options (rnn_type, net_arch, train_unroll_len) must match how the model was
    trained: rnn_type/net_arch determine the network architecture that load_data() restores,
    and all three select the trained-model folder, which the RNN training script tags with the
    agent type and unroll length, e.g. "..._LSTM_ul20_rep0" or "..._LSTM_encdec_ul5_rep0".

    MLP folder naming is resolved against BOTH schemes, newest first, since the trainer gained
    an agent token when the MLP/RNN switch was added:
        "a{ab}_T{T}_delay{d}_MLP_rep{r}"   current  (run_w_wandb_single_varenv.py today)
        "a{ab}_T{T}_delay{d}_rep{r}"       legacy   (models trained before that switch)
    Whichever exists under results_dir/<name>/<training_episode> is used, so both old and new
    result trees work unchanged; if neither exists the script exits naming both candidates.

Output (written under "{eval_out_base}/{trial_name}/{training_episode}/",
        eval_out_base defaulting to "{results_dir}_eval"):
    trial_<n>tcbk.pkl : pickled eval-trajectory info dict, one per eval rep
                        (n = rep_eval * num_of_reps_eval + i_eval)
"""

import os
import sys
import pickle
import numpy as np

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, VariableNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl.agent.RNN_full import CDQL as CDQL_RNN


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    T_k_n0 = int(sys.argv[2])
    delay_embed_len = int(sys.argv[3])
    rep_run = int(sys.argv[4])
    rep_eval = int(sys.argv[5])
    results_dir = sys.argv[6]
    training_episode = sys.argv[7]

    ## ----- optional agent selection (backward compatible: defaults to MLP) ----- ##
    agent_type = sys.argv[8].upper() if len(sys.argv) > 8 else "MLP"
    rnn_type = sys.argv[9].upper() if len(sys.argv) > 9 else "LSTM"
    net_arch = sys.argv[10].lower() if len(sys.argv) > 10 else "rnn"  # "rnn" or "encoder_decoder"
    # train_unroll_len selects the RNN model folder (trial_name encodes ul<n>); it must
    # match how the model was trained. Ignored for MLP. Defaults to the training default (20).
    train_unroll_len = int(sys.argv[11]) if len(sys.argv) > 11 else 20
    eval_out_base = sys.argv[12] if len(sys.argv) > 12 else f"{results_dir}_eval"

    ## ----- wandb setting ----- ##
    if agent_type == "RNN":
        # Mirror the RNN training script's folder tagging (rnn_type, "_encdec" suffix).
        agent_tag = rnn_type
        if net_arch == "encoder_decoder":
            agent_tag += "_encdec"
        trial_name = f"a{antibiotic_value:.2f}_T{T_k_n0}_delay{delay_embed_len}_{agent_tag}_ul{train_unroll_len}_rep{rep_run}"
    else:
        # MLP folder naming changed when the MLP/RNN switch was added to
        # run_w_wandb_single_varenv.py: it now writes an agent token ("..._MLP_rep{r}"), while
        # models trained before that carry none ("..._rep{r}"). Accept both -- prefer the
        # current scheme, fall back to the legacy one -- so old and new result trees are
        # equally evaluable. Only the folder NAME differs; the checkpoints are identical.
        _base = f"a{antibiotic_value:.2f}_T{T_k_n0}_delay{delay_embed_len}"
        _candidates = [f"{_base}_MLP_rep{rep_run}",   # current naming
                       f"{_base}_rep{rep_run}"]       # legacy naming (pre MLP/RNN switch)
        for _cand in _candidates:
            if os.path.isdir(os.path.join(results_dir, _cand, training_episode)):
                trial_name = _cand
                break
        else:
            raise SystemExit(
                "no trained-model folder found under %r for either MLP naming scheme:\n  %s"
                % (results_dir, "\n  ".join(_candidates)))
    folder_name = f"{results_dir}/{trial_name}/{training_episode}/"
    
    eval_out = f"{eval_out_base}/{trial_name}/{training_episode}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    k_n0_observation = False
    b_observation = True
    use_gpu = False

    cell_config = CellConfig()
    env_config = EnvConfig(
        k_n0_observation = k_n0_observation,
        b_observation = b_observation,
        delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        T_k_n0 = T_k_n0,
        k_n0_mean = 2.55,
        sigma_kn0 = 0.1,
        max_pop = np.inf,
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
        _, _, _, _, info = c.eval_step(num_decisions=300)
        fname=f"trial_{rep_eval*num_of_reps_eval+i_eval}"
        with open(os.path.join(eval_out, str(fname)+'tcbk.pkl'), "wb") as f:
            pickle.dump(info, f)
    print("Done")
    sys.exit(0)