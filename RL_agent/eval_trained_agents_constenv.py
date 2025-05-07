import os
import sys
import pickle

from envs.cell_model import CellConfig
from envs.envs import EnvConfig, ConstantNutrientEnv
from agent.MLP_full import CDQL


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    nutrient_value = float(sys.argv[2])
    delay_embed_len = int(sys.argv[3])
    rep_run = int(sys.argv[4])
    rep_eval = int(sys.argv[5])
    results_dir = sys.argv[6]
    training_episode = sys.argv[7] 
    
    ## ----- wandb setting ----- ##
    trial_name = "a%.2f_n%.2f_delay%d_rep%d"%(antibiotic_value, nutrient_value, delay_embed_len, rep_run)
    folder_name = f"{results_dir}/{trial_name}/{training_episode}/"
    
    eval_out = f"{results_dir}_eval/{trial_name}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    k_n0_observation = False
    b_observation = True
    use_gpu = False

    cell_config = CellConfig()
    env_config = EnvConfig(k_n0_observation = k_n0_observation,
                           b_observation = b_observation,
                           k_n0_constant = nutrient_value,
                           delay_embed_len = delay_embed_len,
                           b_actions = [0, antibiotic_value])

    env = ConstantNutrientEnv(env_config, cell_config)
    c = CDQL(env,
             buffer_size = 1_000_000,
             batch_size = 512,
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