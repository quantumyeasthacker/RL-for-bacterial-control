import os
# import matplotlib.pyplot as plt
# import copy
import numpy as np
import sys
import pickle

from envs.cell_model import CellConfig
from envs.envs import EnvConfig, VariableNutrientEnv


MAIN = __name__ == "__main__"

if MAIN:
    ## ----- antibiotic/nutrient value check ----- ##
    # init_pop_size = 1000
    # sample_size = 100
        
    ## ----- non-monotonic pulsing behavior ----- ##
    # half_period = 0
    # initialize_app = "constant_low"
    # antibiotic_value = 3.7
    # nutrient_value = 0.5
    # rep = 0

    half_period = int(sys.argv[1])
    initialize_app = sys.argv[2]
    antibiotic_value = float(sys.argv[3])
    T_k_n0 = int(sys.argv[4])
    rep = int(sys.argv[5])
    results_dir = sys.argv[6]

    num_decisions = 300

    cell_config = CellConfig()
    env_config = EnvConfig(
        delay_embed_len = 30,
        b_actions = [0, antibiotic_value],
        max_pop = np.inf,
        T_k_n0 = T_k_n0,
        k_n0_mean = 2.55,
        sigma_kn0 = 0.1
    )
                           

    # T_k_n0: Optional[Union[float, None]] = None # 6
    # k_n0_mean: Optional[Union[float, None]] = None # 2.55
    # sigma_kn0: Optional[Union[float, None]] = None # 0.1

    env = VariableNutrientEnv(env_config, cell_config)

    folder_name=f"{results_dir}/a{antibiotic_value:.2f}_T{T_k_n0}_value_check/{initialize_app}_{half_period}/"
    os.makedirs(folder_name, exist_ok=True)

    if initialize_app == "low":
        decisions = ([0] * half_period + [1] * half_period) * (num_decisions // 2 // half_period + 1)
    elif initialize_app == "high":
        decisions = ([1] * half_period + [0] * half_period) * (num_decisions // 2 // half_period + 1)
    elif initialize_app == "constant":
        decisions = [1] * num_decisions
    elif initialize_app == "constant_low":
        decisions = [0] * num_decisions

    decisions = decisions[:num_decisions]
    
    num_of_reps = 10
    for i in range(num_of_reps):
        env.reset()
        for decision in decisions:
            _, _, terminated, truncated, info = env.step(decision)
            if terminated or truncated:
                break
        fname="trial_%d"%int(rep*num_of_reps+i)
        
        with open(os.path.join(folder_name,str(fname)+'tcbk.pkl'), "wb") as f:
            pickle.dump(info, f)

    print("Done")
    sys.exit(0)