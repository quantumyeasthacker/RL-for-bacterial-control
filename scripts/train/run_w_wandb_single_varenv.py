"""Single variable-nutrient training run, logged to wandb.

Usage:
    python run_w_wandb_single_varenv.py \
        antibiotic_value T_k_n0 delay_embed_len rep results_dir \
        [agent] [rnn_type] [batch_size] [buffer_size] [train_unroll_len]

Positional args:
    antibiotic_value : float   drug level for the "on" action (b_actions = [0, a])
    T_k_n0           : int     nutrient oscillation period
    delay_embed_len  : int     observation delay-embed length (use 1 for the RNN agent)
    rep              : int     replicate index
    results_dir      : str     parent output directory
Optional positional args:
    agent            : "MLP" (default) or "RNN" (recurrent RNN_full / r2d2 agent)
    rnn_type         : "LSTM" (default) or "GRU"  (only used when agent == "RNN")
    batch_size       : int, default 512
    buffer_size      : int, default 1_000_000     (accepts 1e4-style values)
    train_unroll_len : int, default 20  (R2D2 unroll length; only used when agent == "RNN")
    net_arch         : "rnn" (default) or "encoder_decoder"  (only used when agent == "RNN")

Output (under results_dir/<trial_name>/):
    episode_<n>/{q_1,q_2,q_target_1,q_target_2} checkpoints, Eval/*.jpg, reward_Q_loss.jpg,
    and wandb logging of all config params + eval metrics.
"""

import os
import sys
import wandb

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, VariableNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl.agent.RNN_full import CDQL as CDQL_RNN
from rlBacterialControl import wandb_auth


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    T_k_n0 = int(sys.argv[2])
    delay_embed_len = int(sys.argv[3])
    rep = int(sys.argv[4])
    results_dir = sys.argv[5]
    # optional trailing args (see module docstring)
    agent_type = sys.argv[6].upper() if len(sys.argv) > 6 else "MLP"
    rnn_type = sys.argv[7].upper() if len(sys.argv) > 7 else "LSTM"
    batch_size = int(sys.argv[8]) if len(sys.argv) > 8 else 512
    buffer_size = int(float(sys.argv[9])) if len(sys.argv) > 9 else 1_000_000  # accepts 1e4-style values
    train_unroll_len = int(sys.argv[10]) if len(sys.argv) > 10 else 20
    net_arch = sys.argv[11].lower() if len(sys.argv) > 11 else "rnn"  # "rnn" or "encoder_decoder"

    ## ----- wandb setting ----- ##
    agent_tag = rnn_type if agent_type == "RNN" else "MLP"
    if agent_type == "RNN" and net_arch == "encoder_decoder":
        agent_tag += "_encdec"
    # For RNN agents the trial name encodes train_unroll_len (ul<n>) so runs that differ
    # only in unroll length get distinct output dirs and cannot overwrite each other's
    # checkpoints. MLP naming is unchanged (train_unroll_len does not apply).
    if agent_type == "RNN":
        trial_name = f"a{antibiotic_value:.2f}_T{T_k_n0}_delay{delay_embed_len}_{agent_tag}_ul{train_unroll_len}_rep{rep}"
    else:
        trial_name = f"a{antibiotic_value:.2f}_T{T_k_n0}_delay{delay_embed_len}_{agent_tag}_rep{rep}"
    folder_name = f"{results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": antibiotic_value,
                    "T_k_n0": T_k_n0,
                    "delay_embed_len": delay_embed_len,
                    "rep": rep,
                    "agent_type": agent_type,
                    "rnn_type": rnn_type if agent_type == "RNN" else None,
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "train_unroll_len": train_unroll_len if agent_type == "RNN" else None,
                    "net_arch": net_arch if agent_type == "RNN" else None}
    
    # credentials come from ~/.config/wandb_rl/credentials.json (or $WANDB_CREDENTIALS),
    # else $WANDB_API_KEY/$WANDB_ENTITY, else the local ~/.netrc from `wandb login`.
    wandb_auth.init(project="antibioticRL-varenv-nutrient-delay-30-record",
                    dir=folder_name,
                    name=str(trial_name),
                    config=wandb_config,
                    settings=wandb.Settings(symlink=False))
    
    ## ----- RL setting ----- ##
    k_n0_observation = False
    b_observation = True
    use_gpu = False

    cell_config = CellConfig()
    env_config = EnvConfig(
        # num_cells_init = 60,
        # threshold = 50,
        # delta_t = 0.2,
        k_n0_observation = k_n0_observation,
        b_observation = b_observation,
        delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        T_k_n0 = T_k_n0,
        k_n0_mean = 2.55,
        sigma_kn0 = 0.1
    )

    env = VariableNutrientEnv(env_config, cell_config)
    if agent_type == "RNN":
        c = CDQL_RNN(env,
                     buffer_size = buffer_size,
                     batch_size = batch_size,
                     train_freq = 1,
                     gradient_steps = 1,
                     use_gpu = use_gpu,
                     rnn_type = rnn_type,
                     train_unroll_len = train_unroll_len,
                     net_arch = net_arch)
    else:
        c = CDQL(env,
                 buffer_size = buffer_size,
                 batch_size = batch_size,
                 train_freq = 1,
                 gradient_steps = 1,
                 use_gpu = use_gpu)

    ## ----- RL training ----- ##
    c.train(episodes=400,
            num_decisions=300,
            num_evals=5,
            folder_name=folder_name)
    
    wandb.finish()
    print("Done")
    sys.exit(0)