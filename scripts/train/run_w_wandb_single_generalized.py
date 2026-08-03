import os
import sys
import wandb

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, GeneralizedAgentEnv
from rlBacterialControl.agent.MLP_full import CDQL
from rlBacterialControl.agent.RNN_full import CDQL as CDQL_RNN
from rlBacterialControl import wandb_auth


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    constant_nutrient_string = sys.argv[2]
    constant_nutrient = [float(i_n) for i_n in constant_nutrient_string.split("_")]
    T_k_n0_string = sys.argv[3]
    T_k_n0 = [int(i_T) for i_T in T_k_n0_string.split("_")]
    delay_embed_len = int(sys.argv[4])
    rep = int(sys.argv[5])
    results_dir = sys.argv[6]
    episodes = int(sys.argv[7]) # 400
    # optional trailing arg: "MLP" (default) or "RNN" to train the recurrent RNN_full agent
    agent_type = sys.argv[8].upper() if len(sys.argv) > 8 else "MLP"

    ## ----- wandb setting ----- ##
    trained_env = f"const{constant_nutrient_string}_T{T_k_n0_string}"
    trial_name = f"a{antibiotic_value:.2f}_{trained_env}_delay{delay_embed_len}_episodes{episodes}_rep{rep}"
    folder_name = f"{results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": antibiotic_value,
                    "constant_nutrient": constant_nutrient,
                    "T_k_n0": T_k_n0,
                    "delay_embed_len": delay_embed_len,
                    "episodes": episodes,
                    "rep": rep,
                    "agent_type": agent_type}
    
    # credentials come from ~/.config/wandb_rl/credentials.json (or $WANDB_CREDENTIALS),
    # else $WANDB_API_KEY/$WANDB_ENTITY, else the local ~/.netrc from `wandb login`.
    wandb_auth.init(project="antibioticRL-generalized-agent-delays-record",
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
        k_n0_constant = constant_nutrient,
        delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        T_k_n0 = T_k_n0,
        k_n0_mean = 2.55,
        sigma_kn0 = 0.1
    )

    env = GeneralizedAgentEnv(env_config, cell_config)
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
    c.train(episodes=episodes,
            num_decisions=300,
            num_evals=5,
            folder_name=folder_name)
    
    wandb.finish()
    print("Done")
    sys.exit(0)