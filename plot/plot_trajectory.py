# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
# from scipy.optimize import curve_fit
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist

from utils import expand_and_fill, estimate_frequency_fft, down_edge_detection, load_logger_data_new, plot_single, plot_single_varenv, plot_single_separate
from pathlib import Path


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EPS = 1e-6
# COLOR_LIST = ["#dec60c", "#a7c82f", "#548c6a"]
# BASE_PATH = Path("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/")
BASE_PATH = Path("/mnt/d/bacterial_adaptation/working_folder")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %% ----- ----- ----- ----- constant sim ----- ----- ----- ----- %% #
color_map = {"1.00": "#dec60c", "2.00": "#a7c82f", "3.00": "#548c6a", "4.00": "#116E60"}
env_type = "constenv"
param_file = BASE_PATH / "param_space" / f"param_non_monotonic_pulsing_{env_type}.txt"
sim_folder = BASE_PATH / f"results_sim_{env_type}"

n_trials = 100

with open(param_file, "r") as f:
    param_sim = f.readlines()

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

for param in param_sim:
    half_period = int(param[0])
    initialize_app = param[1]
    antibiotic_value = float(param[2])
    nutrient_value = str(param[3])
    # if nutrient_value == "4.00":
    #     continue
    if nutrient_value != "2.00":
        continue
    if initialize_app != "constant":
        continue

    folder_name = sim_folder / f"a{param[2]}_n{param[3]}_value_check" / f"{param[1]}_{param[0]}/"
    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials)

    out_name = BASE_PATH / "figures_jpg" / "constant_nutrient_separate" / f"a{param[2]}_n{param[3]}_value_check" / f"{param[1]}_{param[0]}.jpg"
    plot_single_separate(loaded_logger, out_name, color_map[nutrient_value], warm_up_embed = warm_up_embed)
    
    out_name = BASE_PATH / "figures_pdf" / "constant_nutrient_separate" / f"a{param[2]}_n{param[3]}_value_check" / f"{param[1]}_{param[0]}.pdf"
    plot_single_separate(loaded_logger, out_name, color_map[nutrient_value], warm_up_embed = warm_up_embed)


# %% ----- ----- ----- ----- constant eval ----- ----- ----- ----- %% #
color_map = {"1.00": "#dec60c", "2.00": "#a7c82f", "3.00": "#548c6a"}
env_type = "constenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250309_constant_app" / eval_folder

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

for param in param_agent:
    antibiotic_value = float(param[0])
    nutrient_value = str(param[1])
    training_episode = param[4]
    if nutrient_value != "2.00":
        continue

    folder_name = eval_folder / f"a{param[0]}_n{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval)

    out_name = BASE_PATH / "figures_jpg" / "constant_nutrient" / "eval" / f"a{param[0]}_n{param[1]}_delay{param[2]}_rep{param[3]}.jpg"
    plot_single(loaded_logger, out_name, color_map[nutrient_value], True)

    out_name = BASE_PATH / "figures_pdf" / "constant_nutrient" / "eval" / f"a{param[0]}_n{param[1]}_delay{param[2]}_rep{param[3]}.pdf"
    plot_single(loaded_logger, out_name, color_map[nutrient_value], True)

# %% ----- ----- ----- ----- varenv sim (constant) ----- ----- ----- ----- %% #
colors_baby_blue = ["#89CFF0", "#60BEEB", "#38AEE6", "#1B99D4", "#167CAC", "#115E83", "#0B415A"]
colors_baby_blue = colors_baby_blue[::-2]
env_type = "varenv"
param_file = BASE_PATH / "param_space" / f"param_non_monotonic_pulsing_{env_type}.txt"
sim_folder = BASE_PATH / f"results_sim_{env_type}"

n_trials = 100

with open(param_file, "r") as f:
    param_sim = f.readlines()

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

for param in param_sim:
    # half_period = int(param[0])
    initialize_app = param[1]
    antibiotic_value = float(param[2])
    # if param[3] != "12":
    #     continue
    # if initialize_app != "constant":
    #     continue

    folder_name = sim_folder / f"a{param[2]}_T{param[3]}_value_check" / f"{param[1]}_{param[0]}/"
    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials)

    out_name = BASE_PATH / "figures_jpg" / "var_nutrient" / f"a{param[2]}_T{param[3]}_value_check" / f"{param[1]}_{param[0]}.jpg"
    plot_single(loaded_logger, out_name, "#548c6a", True, True)
    # plot_single_separate(loaded_logger, out_name, colors_baby_blue[0], False, True, warm_up_embed = warm_up_embed)

    # out_name = BASE_PATH / "figures_pdf" / "var_nutrient_separate" / f"a{param[2]}_T{param[3]}_value_check" / f"{param[1]}_{param[0]}.pdf"
    # plot_single_separate(loaded_logger, out_name, colors_baby_blue[0], False, True, warm_up_embed = warm_up_embed)

# %% ----- ----- ----- ----- varenv eval ----- ----- ----- ----- %% #
colors_baby_blue = ["#89CFF0", "#60BEEB", "#38AEE6", "#1B99D4", "#167CAC", "#115E83", "#0B415A"]
colors_baby_blue = colors_baby_blue[::-2]
env_type = "varenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250325_varenv" / eval_folder

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

for param in param_agent:
    antibiotic_value = float(param[0])
    training_episode = param[4]
    if param[1] != "12":
        continue
    
    folder_name = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval)

    out_name = BASE_PATH / "figures_jpg" / "var_nutrient" / "eval" / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}.jpg"
    plot_single_varenv(loaded_logger, out_name, colors_baby_blue, n_trials_eval)
    # plot_single(loaded_logger, out_name, "#548c6a", True, True)

    # out_name = BASE_PATH / "figures_pdf" / "var_nutrient" / "eval" / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}.pdf"
    # plot_single_varenv(loaded_logger, out_name, colors_baby_blue, n_trials_eval)

# %% ----- ----- ----- ----- specialized agents generalizability check in use ----- ----- ----- ----- %% #
# colors_baby_blue = ["#89CFF0", "#60BEEB", "#38AEE6", "#1B99D4", "#167CAC", "#115E83", "#0B415A"]
# colors_baby_blue = colors_baby_blue[::-2]
default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

env_type = "special_gen"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval_inuse.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250421_special_gen" / eval_folder

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

for param in param_agent:
    antibiotic_value = float(param[0])
    total_episodes = int(param[6])
    training_episode = param[7]
    folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_rep{param[5]}_{param[3]}_{param[4]}"
    folder_name = eval_folder / folder_name / training_episode

    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval)

    out_name = BASE_PATH / "figures_jpg" / env_type / f"a{param[0]}_{param[1]}_delay{param[2]}_rep{param[5]}" / f"{param[3]}_{param[4]}.jpg"
    plot_single_varenv(loaded_logger, out_name, default_color_list, n_trials_eval)

    # out_name = BASE_PATH / "figures_pdf" / env_type / f"a{param[0]}_{param[1]}_delay{param[2]}_rep{param[5]}" / f"{param[3]}_{param[4]}.pdf"
    # plot_single_varenv(loaded_logger, out_name, default_color_list, n_trials_eval)

# %%
# %% ----- ----- ----- ----- specialized agents generalizability check self ----- ----- ----- ----- %% #
# colors_baby_blue = ["#89CFF0", "#60BEEB", "#38AEE6", "#1B99D4", "#167CAC", "#115E83", "#0B415A"]
# colors_baby_blue = colors_baby_blue[::-2]
default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

env_type = "special_gen"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval_self.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250421_special_gen" / eval_folder

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

for param in param_agent:
    antibiotic_value = float(param[0])
    total_episodes = int(param[6])
    training_episode = param[7]
    folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_rep{param[5]}_{param[3]}_{param[4]}"
    folder_name = eval_folder / folder_name / training_episode

    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval)

    out_name = BASE_PATH / "figures_jpg" / env_type / f"a{param[0]}_{param[1]}_delay{param[2]}_rep{param[5]}" / f"{param[3]}_{param[4]}.jpg"
    plot_single_varenv(loaded_logger, out_name, default_color_list, n_trials_eval)

    # out_name = BASE_PATH / "figures_pdf" / env_type / f"a{param[0]}_{param[1]}_delay{param[2]}_rep{param[5]}" / f"{param[3]}_{param[4]}.pdf"
    # plot_single_varenv(loaded_logger, out_name, default_color_list, n_trials_eval)

# %%
