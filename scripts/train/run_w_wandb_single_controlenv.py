import os
import sys
import wandb

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ControlNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl.agent.RNN_full import CDQL as CDQL_RNN
from rlBacterialControl import wandb_auth


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    nutrient_range = sys.argv[2]
    delay_embed_len = int(sys.argv[3])
    rep = int(sys.argv[4])
    results_dir = sys.argv[5]
    b_observation = sys.argv[6] == "True"
    k_n0_observation = sys.argv[7] == "True"
    # optional trailing arg: "MLP" (default) or "RNN" to train the recurrent RNN_full agent
    agent_type = sys.argv[8].upper() if len(sys.argv) > 8 else "MLP"

    ## ----- wandb setting ----- ##
    trial_name = f"a{antibiotic_value:.2f}_n{nutrient_range}_b{b_observation}_k{k_n0_observation}_delay{delay_embed_len}_rep{rep}"
    folder_name = f"{results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": antibiotic_value,
                    "nutrient_range": nutrient_range,
                    "delay_embed_len": delay_embed_len,
                    "b_observation": b_observation,
                    "k_n0_observation": k_n0_observation,
                    "rep": rep,
                    "agent_type": agent_type}
    
    # credentials come from ~/.config/wandb_rl/credentials.json (or $WANDB_CREDENTIALS),
    # else $WANDB_API_KEY/$WANDB_ENTITY, else the local ~/.netrc from `wandb login`.
    wandb_auth.init(project="antibioticRL-control-nutrient-delay-30-record",
                    dir=folder_name,
                    name=str(trial_name),
                    config=wandb_config,
                    settings=wandb.Settings(symlink=False))
    
    ## ----- RL setting ----- ##
    # k_n0_observation = False
    # b_observation = True
    use_gpu = False
    k_n0_actions = [float(nutr) for nutr in nutrient_range.split('_')]

    cell_config = CellConfig()
    env_config = EnvConfig(
        # num_cells_init = 60,
        # threshold = 50,
        # delta_t = 0.2,
        b_observation = b_observation,
        k_n0_observation = k_n0_observation,
        delay_embed_len = delay_embed_len,
        k_n0_actions = k_n0_actions,
        b_actions = [0, antibiotic_value],
        num_actions = len(k_n0_actions) * 2,
    )

    env = ControlNutrientEnv(env_config, cell_config)
    if agent_type == "RNN":
        c = CDQL_RNN(env,
                     buffer_size = 1_000_000,
                     batch_size = 512,
                     train_freq = 1,
                     gradient_steps = 1,
                     use_gpu = use_gpu,
                     rnn_type = "LSTM",  # recurrent cell for RNN_full
                     train_unroll_len = 20)   # R2D2 unroll length
    else:
        c = CDQL(env,
                 buffer_size = 1_000_000,
                 batch_size = 512,
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