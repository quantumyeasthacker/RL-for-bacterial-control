import os
import sys
import pickle

from envs.cell_model import CellConfig
from envs.envs import EnvConfig, ControlNutrientEnv
from agent.MLP_full import CDQL


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    nutrient_range = sys.argv[2]
    delay_embed_len = int(sys.argv[3])
    rep_run = int(sys.argv[4])
    rep_eval = int(sys.argv[5])
    results_dir = sys.argv[6]
    b_observation = sys.argv[7] == "True"
    k_n0_observation = sys.argv[8] == "True"
    training_episode = sys.argv[9]

    ## ----- wandb setting ----- ##
    trial_name = f"a{antibiotic_value:.2f}_n{nutrient_range}_b{b_observation}_k{k_n0_observation}_delay{delay_embed_len}_rep{rep_run}"
    folder_name = f"{results_dir}/{trial_name}/{training_episode}/"
    
    eval_out = f"{results_dir}_eval/{trial_name}/{training_episode}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    # k_n0_observation = True
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