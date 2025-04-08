import os
import sys
import wandb

from envs.cell_model import CellConfig
from envs.envs import EnvConfig, GeneralizedAgentEnv
from agent.MLP_full import CDQL


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    constant_nutrient = [float(i_n) for i_n in sys.argv[2].split("_")]
    T_k_n0 = [int(i_T) for i_T in sys.argv[3].split("_")]
    delay_embed_len = int(sys.argv[4])
    rep = int(sys.argv[5])
    results_dir = sys.argv[6]

    ## ----- wandb setting ----- ##
    trial_name = f"a{antibiotic_value:.2f}_const{constant_nutrient}_T{T_k_n0}_delay{delay_embed_len}_rep{rep}"
    folder_name = f"{results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": antibiotic_value,
                    "constant_nutrient": constant_nutrient,
                    "T_k_n0": T_k_n0,
                    "delay_embed_len": delay_embed_len,
                    "rep": rep}
    
    wandb.login(key = "a566d3654abf3ddf6060c24afc0e67fb4dd30c7a")
    wandb.init(project="antibioticRL-zihang-generalized-agent-delay-30",
               dir=folder_name,
               name=str(trial_name),
               config=wandb_config,
               settings=wandb.Settings(symlink=False))
    
    ## ----- RL setting ----- ##
    k_n0_observation = False
    b_observation = True
    use_gpu = False
    warm_up = 37

    cell_config = CellConfig()
    env_config = EnvConfig(
        # num_cells_init = 60,
        # threshold = 50,
        # delta_t = 0.2,
        k_n0_observation = k_n0_observation,
        b_observation = b_observation,
        k_n0_constant = constant_nutrient,
        warm_up = warm_up, delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        T_k_n0 = T_k_n0,
        k_n0_mean = 2.55,
        sigma_kn0 = 0.1
    )

    env = GeneralizedAgentEnv(env_config, cell_config)
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