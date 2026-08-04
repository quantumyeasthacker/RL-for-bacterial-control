"""Single constant-nutrient training run with cell mutation, logged to wandb.

The constant-nutrient counterpart of run_w_wandb_single_varenv_mutate.py: identical agent,
observation blocks and context machinery, but the nutrient is held at a fixed k_n0 instead of
oscillating with period T_k_n0. The cells mutate (phiS_max can change at each division event),
with mutate_prob as a required positional arg.

Optionally the agent also observes the slow proteome context (population-average phi_S or
phiS_max, re-measured only every context_update_freq decision steps and held fixed in between).
This is an env-side observation block, so it applies to the MLP and RNN agents alike.

Usage:
    python run_w_wandb_single_constenv_mutate.py \
        antibiotic_value nutrient_value delay_embed_len rep results_dir mutate_prob \
        [agent] [rnn_type] [batch_size] [buffer_size] [train_unroll_len] [net_arch] \
        [context_update_freq] [context_age] [episodes] [num_decisions] [context_signal]

Positional args:
    antibiotic_value : float   drug level for the "on" action (b_actions = [0, a])
    nutrient_value   : float   constant nutrient concentration k_n0
    delay_embed_len  : int     observation delay-embed length (use 1 for the RNN agent)
    rep              : int     replicate index
    results_dir      : str     parent output directory
    mutate_prob      : float   per-division mutation probability (CellConfig mutate=True)
Optional positional args:
    agent            : "MLP" (default) or "RNN" (recurrent RNN_full / r2d2 agent)
    rnn_type         : "LSTM" (default) or "GRU"  (only used when agent == "RNN")
    batch_size       : int, default 512
    buffer_size      : int, default 1_000_000     (accepts 1e4-style values)
    train_unroll_len : int, default 20  (R2D2 unroll length; only used when agent == "RNN")
    net_arch         : "rnn" (default) or "encoder_decoder"  (only used when agent == "RNN")
    context_update_freq : int, default 0   decision steps between context re-measurements;
                       0 disables the context block entirely (original observation)
    context_age      : int, default 1      1 = also observe the normalized age of the latched
                       context, 0 = ablate it (only used when context_update_freq > 0)
    episodes         : int, default 400    training episodes
    num_decisions    : int, default 300    decision steps per episode
    (episodes/num_decisions are lowered only for smoke tests; leave at the defaults for real runs)
    context_signal   : str, default "phi_S". "phi_S" = the fast stress-protein fraction;
                       "phiS_max" = the evolvable ceiling mutation acts on; "noise"/"noise_phiSmax"
                       = INFORMATION-FREE CONTROLS drawn from the corresponding marginal and
                       latched/aged identically, which separate "the signal is informative" from
                       "the extra input units helped". Tagged in the trial name as
                       phiSmax / rand / randPhiSmax.

Output (under results_dir/<trial_name>/):
    episode_<n>/{q_1,q_2,q_target_1,q_target_2} checkpoints, Eval/*.jpg, reward_Q_loss.jpg,
    and wandb logging of all config params + eval metrics.
    trial_name = a<antibiotic>_n<nutrient>_delay<d>_mutprob<p>_<agent_tag>[_ctx<freq>[noage][phiSmax|rand|randPhiSmax]]_rep<rep>
"""

import os
import sys
import wandb

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl.agent.RNN_full import CDQL as CDQL_RNN
from rlBacterialControl import wandb_auth


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    nutrient_value   = float(sys.argv[2])
    delay_embed_len  = int(sys.argv[3])
    rep              = int(sys.argv[4])
    results_dir      = sys.argv[5]
    mutate_prob      = float(sys.argv[6])          # required
    # optional trailing args
    agent_type       = sys.argv[7].upper()  if len(sys.argv) > 7  else "MLP"
    rnn_type         = sys.argv[8].upper()  if len(sys.argv) > 8  else "LSTM"
    batch_size       = int(sys.argv[9])     if len(sys.argv) > 9  else 512
    buffer_size      = int(float(sys.argv[10])) if len(sys.argv) > 10 else 1_000_000
    train_unroll_len = int(sys.argv[11])    if len(sys.argv) > 11 else 20
    net_arch         = sys.argv[12].lower() if len(sys.argv) > 12 else "rnn"
    context_update_freq = int(sys.argv[13]) if len(sys.argv) > 13 else 0
    context_age      = bool(int(sys.argv[14])) if len(sys.argv) > 14 else True
    episodes         = int(sys.argv[15])    if len(sys.argv) > 15 else 400
    num_decisions    = int(sys.argv[16])    if len(sys.argv) > 16 else 300
    context_signal   = sys.argv[17]         if len(sys.argv) > 17 else "phi_S"

    context_observation = context_update_freq > 0

    ## ----- wandb setting ----- ##
    agent_tag = rnn_type if agent_type == "RNN" else "MLP"
    if agent_type == "RNN" and net_arch == "encoder_decoder":
        agent_tag += "_encdec"
    # the context is an env-side observation block, so it tags independently of the agent
    ctx_tag = ""
    if context_observation:
        ctx_tag = f"_ctx{context_update_freq}" + ("" if context_age else "noage")
        if context_signal == "noise":
            ctx_tag += "rand"   # control arm which lacks dynamic readout, only gives average
        elif context_signal == "noise_phiSmax":
            ctx_tag += "randPhiSmax"  # same control, calibrated to the phiS_max marginal instead
        elif context_signal == "phiS_max":
            ctx_tag += "phiSmax"    # tracking phiS_max_ave instead of phiS_ave, thus directly tracking mutating value
    trial_name = f"a{antibiotic_value:.2f}_n{nutrient_value:.2f}_delay{delay_embed_len}_mutprob{mutate_prob}_{agent_tag}{ctx_tag}_rep{rep}"
    folder_name = f"{results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": antibiotic_value,
                    "nutrient_value": nutrient_value,
                    "delay_embed_len": delay_embed_len,
                    "rep": rep,
                    "agent_type": agent_type,
                    "rnn_type": rnn_type if agent_type == "RNN" else None,
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "train_unroll_len": train_unroll_len if agent_type == "RNN" else None,
                    "net_arch": net_arch if agent_type == "RNN" else None,
                    "mutate_prob": mutate_prob,
                    "context_observation": context_observation,
                    "context_update_freq": context_update_freq if context_observation else None,
                    "context_age_observation": context_age if context_observation else None,
                    "context_signal": context_signal if context_observation else None,
                    "episodes": episodes,
                    "num_decisions": num_decisions}

    # credentials come from ~/.config/wandb_rl/credentials.json (or $WANDB_CREDENTIALS),
    # else $WANDB_API_KEY/$WANDB_ENTITY, else the local ~/.netrc from `wandb login`.
    wandb_auth.init(project="antibioticRL-constenv-mutate",
                    dir=folder_name,
                    name=str(trial_name),
                    config=wandb_config,
                    settings=wandb.Settings(symlink=False))

    ## ----- RL setting ----- ##
    k_n0_observation = False   # nutrient is constant -> not observed
    b_observation = True
    use_gpu = False

    cell_config = CellConfig(mutate=True, mutate_prob=mutate_prob)
    env_config = EnvConfig(
        k_n0_observation = k_n0_observation,
        b_observation = b_observation,
        delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        k_n0_constant = nutrient_value,
        # ConstantNutrientEnv.reset() forces k_n0_init to k_n0_constant (with a warning) if they
        # disagree; set it here so the run starts already at the operating nutrient level.
        k_n0_init = nutrient_value,
        context_observation = context_observation,
        # the env requires a period >= 1 even when the context block is switched off
        context_update_freq = max(context_update_freq, 1),
        context_age_observation = context_age,
        context_signal = context_signal
    )

    env = ConstantNutrientEnv(env_config, cell_config)
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
    c.train(episodes=episodes,
            num_decisions=num_decisions,
            num_evals=10,
            folder_name=folder_name)

    wandb.finish()
    print("Done")
    sys.exit(0)
