import os
from cell_model import CellConfig
from envs import EnvConfig, ConstantNutrientEnv
from MLP_full import CDQL
import sys
import pickle


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    nutrient_value = float(sys.argv[2])
    delay_embed_len = int(sys.argv[3])
    rep = int(sys.argv[4])
    
    ## ----- wandb setting ----- ##
    trial_name = "a%.2f_n%.2f_delay%d_rep%d"%(antibiotic_value, nutrient_value, delay_embed_len, rep)
    folder_name = f"results_panelty/{trial_name}/"
    
    eval_out = f"results_eval_panelty/{trial_name}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    k_n0_observation = True
    b_observation = True
    use_gpu = False
    warm_up = 37 - delay_embed_len

    cell_config = CellConfig()
    env_config = EnvConfig(k_n0_observation = k_n0_observation,
                           b_observation = b_observation,
                           k_n0_init = nutrient_value,
                           warm_up = warm_up, delay_embed_len = delay_embed_len,
                           b_actions = [0, antibiotic_value])

    env = ConstantNutrientEnv(env_config, cell_config)
    c = CDQL(env,
             buffer_size = 1_000_000,
             batch_size = 512,
             use_gpu = use_gpu)
        
    ## ----- RL evaluating ----- ##
    c.load_data(folder_name, False)
    for i_eval in range(100):
        _, _, _, _, info = c.eval_step(num_decisions=200)
        fname="trial_%d"%int(i_eval)
        with open(os.path.join(eval_out, str(fname)+'tcbk.pkl'), "wb") as f:
            pickle.dump(info, f)
    print("Done")
    sys.exit(0)