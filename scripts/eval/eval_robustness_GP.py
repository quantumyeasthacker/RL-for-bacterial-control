import os
import sys
import pickle
import numpy as np

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv, VariableNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- parameter loading ----- ##
    antibiotic_value = float(sys.argv[1])
    trained_env = sys.argv[2]
    delay_embed_len = int(sys.argv[3])
    rep_run = int(sys.argv[4])
    rep_eval = int(sys.argv[5])
    results_dir = sys.argv[6]
    episodes = int(sys.argv[7])
    training_episode = sys.argv[8]
    ls = float(sys.argv[9])
    a = float(sys.argv[10])
    noise_process = "GP"

    ## ----- wandb setting ----- ##
    trial_name = f"a{antibiotic_value:.2f}_{trained_env}_delay{delay_embed_len}_episodes{episodes}_rep{rep_run}"
    folder_name = f"{results_dir}/{trial_name}/{training_episode}/"

    eval_out = f"{results_dir}_eval/{trial_name}_{noise_process}_ls{ls}_amp{a}/{training_episode}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    k_n0_observation = False
    b_observation = True
    use_gpu = False

    cell_config = CellConfig()

    env_config = EnvConfig(
        k_n0_observation = k_n0_observation,
        b_observation = b_observation,
        delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        k_n0_mean = 2.55,
        noise_process = noise_process,
        ls = ls,
        a = a,
        max_pop = np.Inf
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