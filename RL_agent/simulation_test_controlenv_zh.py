import os
# import matplotlib.pyplot as plt
# import copy
import numpy as np
import sys
import pickle

from envs.cell_model import CellConfig
from envs.envs import EnvConfig, ControlNutrientEnv


MAIN = __name__ == "__main__"

if MAIN:
    manual_protocol = sys.argv[1]
    antibiotic_value = float(sys.argv[2])
    nutrient_range = sys.argv[3]
    rep = int(sys.argv[4])
    results_dir = sys.argv[5]

    num_decisions = 300

    cell_config = CellConfig()
    k_n0_actions = [float(nutr) for nutr in nutrient_range.split('_')]
    env_config = EnvConfig(
        k_n0_actions = k_n0_actions,
        b_actions = [0, antibiotic_value],
        num_actions = len(k_n0_actions) * 2,
        max_pop = np.inf,
    )

    env = ControlNutrientEnv(env_config, cell_config)

    folder_name=f"{results_dir}/a{antibiotic_value:.2f}_n{nutrient_range}_value_check/{manual_protocol}/"
    os.makedirs(folder_name, exist_ok=True)

    if manual_protocol == "Famine":
        k_n0 = min(env_config.k_n0_actions)
        b = max(env_config.b_actions)
    elif manual_protocol == "Feast":
        k_n0 = max(env_config.k_n0_actions)
        b = max(env_config.b_actions)
    
    num_of_reps = 10
    for i in range(num_of_reps):
        env.reset()
        for _ in range(num_decisions):
            _, _, terminated, truncated, info = env.step_hardcode(k_n0, b)
            if terminated or truncated:
                break
        fname="trial_%d"%int(rep*num_of_reps+i)
        
        with open(os.path.join(folder_name, str(fname)+'tcbk.pkl'), "wb") as f:
            pickle.dump(info, f)

    print("Done")
    sys.exit(0)