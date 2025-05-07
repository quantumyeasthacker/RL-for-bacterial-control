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
    delay_embed_len = int(sys.argv[2])
    rep_run = int(sys.argv[3])
    rep_eval = int(sys.argv[4])
    results_dir = sys.argv[5]

    ## ----- wandb setting ----- ##
    trial_name = f"a{antibiotic_value:.2f}_delay{delay_embed_len}_rep{rep_run}"
    # folder_name = f"{results_dir}/{trial_name}/"
    folder_name = "/Users/Josiah/Documents/OSPool/feb24_nutr_drug_control2act/Fluc_nutrient/"

    eval_out = f"{results_dir}_eval/{trial_name}/"
    os.makedirs(eval_out, exist_ok=True)

    ## ----- RL setting ----- ##
    k_n0_observation = True
    b_observation = True
    use_gpu = False
    warm_up = 37

    cell_config = CellConfig()
    env_config = EnvConfig(
        k_n0_observation = k_n0_observation,
        b_observation = b_observation,
        warm_up = warm_up, delay_embed_len = delay_embed_len,
        b_actions = [0, antibiotic_value],
        k_n0_actions = [1.0, 4.0],
        k_n0_mean = 2.55,
        sigma_kn0 = 0.1,
        num_actions=4
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