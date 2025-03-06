import os
from cell_model import CellConfig
from envs import EnvConfig, ConstantNutrientEnv
from MLP_full import CDQL
import sys
import wandb


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    nutrient_value = float(sys.argv[2])
    delay_embed_len = int(sys.argv[3])
    rep = int(sys.argv[4])
    results_dir = sys.argv[5]

    ## ----- wandb setting ----- ##
    trial_name = "a%.2f_n%.2f_delay%d_rep%d"%(antibiotic_value, nutrient_value, delay_embed_len, rep)
    folder_name = f"{results_dir}/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"antibiotic_value": antibiotic_value,
                    "nutrient_value": nutrient_value,
                    "delay_embed_len": delay_embed_len,
                    "rep": rep}
    
    wandb.login(key = "a566d3654abf3ddf6060c24afc0e67fb4dd30c7a")
    wandb.init(project="antibioticRL-zihang-constant-nutrient-test",
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
        k_n0_constant = nutrient_value,
        warm_up = warm_up, delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value]
    )

    env = ConstantNutrientEnv(env_config, cell_config)
    c = CDQL(env,
             buffer_size = 1_000_000,
             batch_size = 512,
             train_freq = 1,
             gradient_steps = 1,
             use_gpu = use_gpu)
    
    ## ----- RL training ----- ##
    c.train(episodes=300,
            num_decisions=200,
            num_evals=5,
            folder_name=folder_name)
    
    wandb.finish()
    print("Done")
    sys.exit(0)