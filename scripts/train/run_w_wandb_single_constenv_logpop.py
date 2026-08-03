"""Single constant-nutrient MLP training run using population-size (log10) signals, logged to wandb.

This is a variant of run_w_wandb_single_constenv.py for the "log population size vs. growth rate"
comparison. The agent is the default MLP (CDQL). Two observation variants are supported via the
obs_type arg:
    obs_type = "log10_pop"    -> the bacterial observation block is log10(population count)
    obs_type = "growth_rate"  -> the bacterial observation block is the finite-difference
                                 log-growth rate (the original behavior)
In BOTH variants the antibiotic (drug) history is included in the observation, exactly as before
(b_observation = True), and the reward/cost is log10(population count) (reward_type = "log10_pop").

Usage:
    python run_w_wandb_single_constenv_logpop.py \
        antibiotic_value nutrient_value delay_embed_len rep results_dir [obs_type] [learning_rate]

Positional args:
    antibiotic_value : float   drug level for the "on" action (b_actions = [0, a])
    nutrient_value   : float   constant nutrient concentration k_n0
    delay_embed_len  : int     observation delay-embed length
    rep              : int     replicate index
    results_dir      : str     parent output directory
Optional positional args:
    obs_type         : "log10_pop" (default) or "growth_rate"  (bacterial observation signal)
    learning_rate    : float, default 1e-4   Q-network learning rate (accepts 1e-5-style values)

Output (under results_dir/<trial_name>/):
    episode_<n>/{q_1,q_2,q_target_1,q_target_2} checkpoints, Eval/*.jpg, reward_Q_loss.jpg,
    and wandb logging of all config params + eval metrics.
    trial_name = a<antibiotic>_n<nutrient>_delay<d>_<obs_type>_lr<learning_rate>_rep<rep>
    (learning_rate is encoded in the name so lr variants do not overwrite each other)
"""

import os
import sys
import wandb

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl import wandb_auth


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    nutrient_value = float(sys.argv[2])
    delay_embed_len = int(sys.argv[3])
    rep = int(sys.argv[4])
    results_dir = sys.argv[5]
    # observation signal for the bacterial block: "log10_pop" (default) or "growth_rate"
    obs_type = sys.argv[6].lower() if len(sys.argv) > 6 else "log10_pop"
    assert obs_type in ("log10_pop", "growth_rate"), f"unknown obs_type: {obs_type}"
    # Q-network learning rate (accepts 1e-5-style values)
    learning_rate = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-4
    # reward/cost is log10(population) in both variants of this experiment
    reward_type = "log10_pop"

    ## ----- wandb setting ----- ##
    # learning rate is encoded in the trial name so runs that differ only in lr get distinct
    # output dirs / wandb names and cannot overwrite each other's checkpoints.
    lr_tag = "lr%.0e" % learning_rate  # e.g. 1e-05 -> "lr1e-05"
    trial_name = "a%.2f_n%.2f_delay%d_%s_%s_rep%d" % (
        antibiotic_value, nutrient_value, delay_embed_len, obs_type, lr_tag, rep)
    folder_name = f"{results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": antibiotic_value,
                    "nutrient_value": nutrient_value,
                    "delay_embed_len": delay_embed_len,
                    "rep": rep,
                    "agent_type": "MLP",
                    "obs_type": obs_type,
                    "reward_type": reward_type,
                    "learning_rate": learning_rate}

    # credentials come from ~/.config/wandb_rl/credentials.json (or $WANDB_CREDENTIALS),
    # else $WANDB_API_KEY/$WANDB_ENTITY, else the local ~/.netrc from `wandb login`.
    wandb_auth.init(project="antibioticRL-constant-nutrient-logpop",
                    dir=folder_name,
                    name=str(trial_name),
                    config=wandb_config,
                    settings=wandb.Settings(symlink=False))

    ## ----- RL setting ----- ##
    k_n0_observation = False   # nutrient is constant -> not observed
    b_observation = True       # antibiotic (drug) history IS observed, as before
    use_gpu = False

    cell_config = CellConfig()
    env_config = EnvConfig(
        k_n0_observation = k_n0_observation,
        b_observation = b_observation,
        k_n0_constant = nutrient_value,
        delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        obs_type = obs_type,
        reward_type = reward_type
    )

    env = ConstantNutrientEnv(env_config, cell_config)
    c = CDQL(env,
             buffer_size = 1_000_000,
             batch_size = 512,
             train_freq = 1,
             gradient_steps = 1,
             use_gpu = use_gpu,
             learning_rate = learning_rate)

    ## ----- RL training ----- ##
    c.train(episodes=400,
            num_decisions=300,
            num_evals=5,
            folder_name=folder_name)

    wandb.finish()
    print("Done")
    sys.exit(0)
