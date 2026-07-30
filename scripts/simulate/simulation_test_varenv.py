import os
# import matplotlib.pyplot as plt
# import copy
import numpy as np
import sys
import pickle

from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, VariableNutrientEnv


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

    # positional args:
    #   1 half_period  2 initialize_app  3 antibiotic_value  4 T_k_n0  5 rep  6 results_dir
    # optional CellConfig sweep args (added for the env-parameter sensitivity sweep):
    #   7 cell_param   8 cell_pct
    # When cell_param/cell_pct are given, one CellConfig field is overridden to
    # default * (1 + cell_pct/100) and the folder name gains a "_{param}_{+pct}pct" token.
    half_period = int(sys.argv[1])
    initialize_app = sys.argv[2]
    antibiotic_value = float(sys.argv[3])
    T_k_n0 = int(sys.argv[4])
    rep = int(sys.argv[5])
    results_dir = sys.argv[6]
    cell_param = sys.argv[7] if len(sys.argv) > 7 else None
    cell_pct = int(sys.argv[8]) if len(sys.argv) > 8 else 0

    num_decisions = 300

    if cell_param is not None:
        default_val = getattr(CellConfig(), cell_param)
        new_val = default_val * (1 + cell_pct / 100)
        cell_config = CellConfig(**{cell_param: new_val})
        sweep_tag = f"_{cell_param}_{cell_pct:+d}pct"
        print(f"CellConfig override: {cell_param} {default_val} -> {new_val} ({cell_pct:+d}%)")
    else:
        cell_config = CellConfig()
        sweep_tag = ""
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

    folder_name=f"{results_dir}/a{antibiotic_value:.2f}_T{T_k_n0}{sweep_tag}_value_check/{initialize_app}_{half_period}/"
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