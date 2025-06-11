import os
import sys
import pickle
import numpy as np

from envs.cell_model import CellConfig
from envs.envs import EnvConfig, ConstantNutrientEnv, VariableNutrientEnv
from agent.MLP_full import CDQL


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    trained_env = sys.argv[2]
    delay_embed_len = int(sys.argv[3])
    eval_env = sys.argv[4]
    eval_variable = sys.argv[5]
    rep_run = int(sys.argv[6])
    rep_eval = int(sys.argv[7])
    results_dir = sys.argv[8]
    episodes = int(sys.argv[9])
    training_episode = sys.argv[10]

    ## ----- wandb setting ----- ##
    trial_name = f"a{antibiotic_value:.2f}_{trained_env}_delay{delay_embed_len}_rep{rep_run}"
    folder_name = f"{results_dir}/{trial_name}/{training_episode}/"
    
    eval_out = f"results_delay_30_record_special_gen_eval/{trial_name}_{eval_env}_{eval_variable}/{training_episode}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    k_n0_observation = False
    b_observation = True
    use_gpu = False

    cell_config = CellConfig()
    if eval_env == "constenv":
        env_config = EnvConfig(
            k_n0_observation = k_n0_observation,
            b_observation = b_observation,
            k_n0_constant = float(eval_variable),
            delay_embed_len = delay_embed_len,
            b_actions = [0, antibiotic_value],
            max_pop = np.inf,
        )
        env = ConstantNutrientEnv(env_config, cell_config)
    elif eval_env == "varenv":
        env_config = EnvConfig(
            k_n0_observation = k_n0_observation,
            b_observation = b_observation,
            delay_embed_len = delay_embed_len,
            b_actions = [0, antibiotic_value],
            T_k_n0 = int(eval_variable),
            k_n0_mean = 2.55,
            sigma_kn0 = 0.1,
            max_pop = np.inf,
        )
        env = VariableNutrientEnv(env_config, cell_config)
        
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