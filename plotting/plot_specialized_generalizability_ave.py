# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
# from scipy.optimize import curve_fit
import matplotlib as mpl
from utils import expand_and_fill, estimate_frequency_fft, down_edge_detection, load_logger_data_new, get_best_row_extinct_rate
from pathlib import Path


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EPS = 1e-6
# default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLOR_LIST = ["#dec60c", "#a7c82f", "#548c6a"]

# BASE_PATH = Path("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/")
BASE_PATH = Path("/home/zihangw/BacteriaAdaptation")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %% test
# with open("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/20250309_constant_app/archive_length300/results_sim_constenv/a3.72_n0.50_value_check/constant_0/trial_0tcbk.pkl", "rb") as f:
#     info = pickle.load(f)

# %% ----- ----- ----- ----- constant sim ----- ----- ----- ----- %% #
env_type = "constenv"
param_file = BASE_PATH / "param_space" / f"param_non_monotonic_pulsing_{env_type}.txt"
sim_folder = BASE_PATH / f"results_sim_{env_type}"
# sim_folder = BASE_PATH / "20250309_constant_app" / sim_folder

n_trials = 100

with open(param_file, "r") as f:
    param_sim = f.readlines()

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

df_constant_sim = pd.DataFrame(param_sim)
df_constant_sim.columns = ["half_period", "initialize_app", "antibiotic_value", "nutrient_value"]
df_constant_sim["inst_env"] = env_type
df_constant_sim["inst_variable"] = df_constant_sim["nutrient_value"]
df_constant_sim["inst_combination"] = df_constant_sim["inst_env"] + "_" + df_constant_sim["inst_variable"]

df_constant_sim["half_period"] = df_constant_sim["half_period"].astype(float)
df_constant_sim["antibiotic_value"] = df_constant_sim["antibiotic_value"].astype(float)
df_constant_sim["nutrient_value"] = df_constant_sim["nutrient_value"].astype(float)

final_cell_list = []
final_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_sim:
    half_period = int(param[0])
    initialize_app = param[1]
    antibiotic_value = float(param[2])
    folder_name = sim_folder / f"a{param[2]}_n{param[3]}_value_check" / f"{param[1]}_{param[0]}/"

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials, False)
    # cell_ave = np.mean(cell_array, axis=0)
    if half_period == 0:
        freq = 0
    else:
        freq = 1 / (half_period * 2 * delta_t)
    freq_list.append(freq)

    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))

    # freq_param_list = []
    # for tcbk in tcbk_list:
    #     b = tcbk[2,warm_up_embed:]
    #     freq = [estimate_frequency_fft(b == antibiotic_value, sampling_unit=delta_t)]
    #     if not np.isnan(freq):
    #         freq_param_list += [freq]

    # if len(freq_param_list) == 0:
    #     freq_list += [0]
    # else:
    #     freq_list += [np.mean(freq_param_list)]
    final_cell_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean())
    final_cell_std_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std())

df_constant_sim["sim_log_cell"] = final_cell_list
df_constant_sim["sim_log_cell_std"] = final_cell_std_list
# df_constant_sim["sim_log_cell"] = np.log10(df_constant_sim["sim_final_cell"])
df_constant_sim["sim_freq"] = freq_list
df_constant_sim["extinction_frac"] = extinction_frac_list
df_constant_sim["extinction_rate"] = extinction_rate_list

df_constant_sim_cst_app = df_constant_sim[df_constant_sim["initialize_app"] == "constant"]

# %% ----- ----- ----- ----- constant eval ----- ----- ----- ----- %% #
env_type = "constenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250309_constant_app" / eval_folder

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_constant_eval = pd.DataFrame(param_agent)
df_constant_eval.columns = ["antibiotic_value", "nutrient_value", "delay_embed_len", "rep", "training_episode"]
df_constant_eval["inst_env"] = env_type
df_constant_eval["inst_variable"] = df_constant_eval["nutrient_value"]
df_constant_eval["inst_combination"] = df_constant_eval["inst_env"] + "_" + df_constant_eval["inst_variable"]

df_constant_eval["antibiotic_value"] = df_constant_eval["antibiotic_value"].astype(float)
df_constant_eval["nutrient_value"] = df_constant_eval["nutrient_value"].astype(float)
df_constant_eval["delay_embed_len"] = df_constant_eval["delay_embed_len"].astype(int)
df_constant_eval["rep"] = df_constant_eval["rep"].astype(int)
df_constant_eval["training_episode_int"] = df_constant_eval["training_episode"].map(lambda x: int(x.split("_")[-1]))

eval_cell_list = []
eval_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_agent:
    antibiotic_value = float(param[0])
    training_episode = param[4]
    folder_name = eval_folder / f"a{param[0]}_n{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
    # cell_ave = np.mean(cell_array, axis=0)
    
    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))

    freq_param_list = []
    for tcbk in tcbk_list:
        b = tcbk[2,warm_up_embed:]
        freq = [estimate_frequency_fft(b == antibiotic_value, sampling_unit=delta_t)]
        if not np.isnan(freq):
            freq_param_list += [freq]
    if len(freq_param_list) == 0:
        freq_list += [0]
    else:
        freq_list += [np.mean(freq_param_list)]
    
    eval_cell_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean()]
    eval_cell_std_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std()]

df_constant_eval["eval_log_cell"] = eval_cell_list
df_constant_eval["eval_log_cell_std"] = eval_cell_std_list
# df_constant_eval["eval_log_cell"] = np.log10(df_constant_eval["eval_final_cell"])
df_constant_eval["eval_freq"] = freq_list
df_constant_eval["extinction_frac"] = extinction_frac_list
df_constant_eval["extinction_rate"] = extinction_rate_list

# %% ----- ----- ----- ----- varenv sim ----- ----- ----- ----- %% #
env_type = "varenv"
param_file = BASE_PATH / "param_space" / f"param_non_monotonic_pulsing_{env_type}.txt"
sim_folder = BASE_PATH / f"results_sim_{env_type}"
# sim_folder = BASE_PATH / "20250325_varenv" / sim_folder

n_trials = 100

with open(param_file, "r") as f:
    param_sim = f.readlines()

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]
param_sim = [x for x in param_sim if x[1] == "constant"]

df_varenv_sim = pd.DataFrame(param_sim)
df_varenv_sim.columns = ["half_period", "initialize_app", "antibiotic_value", "T_k0"]
df_varenv_sim["inst_env"] = env_type
df_varenv_sim["inst_variable"] = df_varenv_sim["T_k0"]
df_varenv_sim["inst_combination"] = df_varenv_sim["inst_env"] + "_" + df_varenv_sim["inst_variable"]

df_varenv_sim["half_period"] = df_varenv_sim["half_period"].astype(float)
df_varenv_sim["antibiotic_value"] = df_varenv_sim["antibiotic_value"].astype(float)
df_varenv_sim["T_k0"] = df_varenv_sim["T_k0"].astype(int)

final_cell_list = []
final_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_sim:
    # half_period = int(param[0])
    initialize_app = param[1]
    antibiotic_value = float(param[2])
    folder_name = sim_folder / f"a{param[2]}_T{param[3]}_value_check" / f"{param[1]}_{param[0]}/"

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials, False)
    # cell_ave = np.mean(cell_array, axis=0)

    if half_period == 0:
        freq = 0
    else:
        freq = 1 / (half_period * 2 * delta_t)
    freq_list.append(freq)

    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))

    # freq_param_list = []
    # for tcbk in tcbk_list:
    #     b = tcbk[2,warm_up_embed:]
    #     freq = [estimate_frequency_fft(b == antibiotic_value, sampling_unit=delta_t)]
    #     if not np.isnan(freq):
    #         freq_param_list += [freq]

    # if len(freq_param_list) == 0:
    #     freq_list += [0]
    # else:
    #     freq_list += [np.mean(freq_param_list)]

    final_cell_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean())
    final_cell_std_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std())

df_varenv_sim["sim_log_cell"] = final_cell_list
df_varenv_sim["sim_log_cell_std"] = final_cell_std_list
# df_varenv_sim["sim_log_cell"] = np.log10(df_varenv_sim["sim_final_cell"])
df_varenv_sim["sim_freq"] = freq_list
df_varenv_sim["extinction_frac"] = extinction_frac_list
df_varenv_sim["extinction_rate"] = extinction_rate_list

df_varenv_sim_cst_app = df_varenv_sim[df_varenv_sim["initialize_app"] == "constant"]

# %% ----- ----- ----- ----- varenv eval ----- ----- ----- ----- %% #
env_type = "varenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250325_varenv" / eval_folder

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_varenv_eval = pd.DataFrame(param_agent)
df_varenv_eval.columns = ["antibiotic_value", "T_k0", "delay_embed_len", "rep", "training_episode"]
df_varenv_eval["inst_env"] = env_type
df_varenv_eval["inst_variable"] = df_varenv_eval["T_k0"]
df_varenv_eval["inst_combination"] = df_varenv_eval["inst_env"] + "_" + df_varenv_eval["inst_variable"]

df_varenv_eval["antibiotic_value"] = df_varenv_eval["antibiotic_value"].astype(float)
df_varenv_eval["T_k0"] = df_varenv_eval["T_k0"].astype(int)
df_varenv_eval["delay_embed_len"] = df_varenv_eval["delay_embed_len"].astype(int)
df_varenv_eval["rep"] = df_varenv_eval["rep"].astype(int)
df_varenv_eval["training_episode_int"] = df_varenv_eval["training_episode"].map(lambda x: int(x.split("_")[-1]))

eval_cell_list = []
eval_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_agent:
    antibiotic_value = float(param[0])
    training_episode = param[4]
    folder_name = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
    # cell_ave = np.mean(cell_array, axis=0)

    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))

    freq_param_list = []
    for tcbk in tcbk_list:
        b = tcbk[2,warm_up_embed:]
        freq = [estimate_frequency_fft(b == antibiotic_value, sampling_unit=delta_t)]
        if not np.isnan(freq):
            freq_param_list += [freq]

    if len(freq_param_list) == 0:
        freq_list += [0]
    else:
        freq_list += [np.mean(freq_param_list)]
    
    eval_cell_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean()]
    eval_cell_std_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std()]

df_varenv_eval["eval_log_cell"] = eval_cell_list
df_varenv_eval["eval_log_cell_std"] = eval_cell_std_list
# df_varenv_eval["eval_log_cell"] = np.log10(df_varenv_eval["eval_final_cell"])
df_varenv_eval["eval_freq"] = freq_list
df_varenv_eval["extinction_frac"] = extinction_frac_list
df_varenv_eval["extinction_rate"] = extinction_rate_list

# %% ----- ----- ----- ----- special_gen eval ----- ----- ----- ----- %% #
env_type = "special_gen"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval_inuse.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250421_special_gen" / eval_folder

n_trials_eval = 100

# running_list = ["a3.72_const1_4_T6_24_delay30_episodes600_rep0"]

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_special_gen_eval = pd.DataFrame(param_agent)
df_special_gen_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode", "results_dir"]
df_special_gen_eval["inst_combination"] = df_special_gen_eval["inst_env"] + "_" + df_special_gen_eval["inst_variable"]

df_special_gen_eval["antibiotic_value"] = df_special_gen_eval["antibiotic_value"].astype(float)
df_special_gen_eval["delay_embed_len"] = df_special_gen_eval["delay_embed_len"].astype(int)
df_special_gen_eval["rep"] = df_special_gen_eval["rep"].astype(int)
df_special_gen_eval["episodes"] = df_special_gen_eval["episodes"].astype(int)

eval_cell_list = []
eval_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_agent:
    antibiotic_value = float(param[0])
    total_episodes = int(param[6])
    training_episode = param[7]
    folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_rep{param[5]}_{param[3]}_{param[4]}"
    folder_name = eval_folder / folder_name / training_episode
    # print(len(os.listdir(folder_name)))

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
    # cell_ave = np.mean(cell_array, axis=0)

    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))

    freq_param_list = []
    for tcbk in tcbk_list:
        b = tcbk[2,warm_up_embed:]
        freq = [estimate_frequency_fft(b == antibiotic_value, sampling_unit=delta_t)]
        if not np.isnan(freq):
            freq_param_list += [freq]

    if len(freq_param_list) == 0:
        freq_list += [0]
    else:
        freq_list += [np.mean(freq_param_list)]
    
    eval_cell_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean()]
    eval_cell_std_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std()]

df_special_gen_eval["eval_log_cell"] = eval_cell_list
df_special_gen_eval["eval_log_cell_std"] = eval_cell_std_list
# df_special_gen_eval["eval_log_cell"] = np.log10(df_special_gen_eval["eval_final_cell"])
df_special_gen_eval["eval_freq"] = freq_list
df_special_gen_eval["extinction_frac"] = extinction_frac_list
df_special_gen_eval["extinction_rate"] = extinction_rate_list

# %% ----- ----- ----- ----- generalized eval ----- ----- ----- ----- %% #
env_type = "generalized"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250421_generalized" / eval_folder

n_trials_eval = 100

# running_list = ["a3.72_const1_4_T6_24_delay30_episodes600_rep0"]

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_generalized_eval = pd.DataFrame(param_agent)
df_generalized_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode"]
df_generalized_eval["inst_combination"] = df_generalized_eval["inst_env"] + "_" + df_generalized_eval["inst_variable"]

df_generalized_eval["antibiotic_value"] = df_generalized_eval["antibiotic_value"].astype(float)
df_generalized_eval["delay_embed_len"] = df_generalized_eval["delay_embed_len"].astype(int)
df_generalized_eval["rep"] = df_generalized_eval["rep"].astype(int)
df_generalized_eval["episodes"] = df_generalized_eval["episodes"].astype(int)

# df_generalized_eval = df_generalized_eval.loc[df_generalized_eval["episodes"] == 400].reset_index(drop=True) ##### temporary

eval_cell_list = []
eval_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_agent:
    antibiotic_value = float(param[0])
    total_episodes = int(param[6])
    training_episode = param[7]
    folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_episodes{param[6]}_rep{param[5]}_{param[3]}_{param[4]}"
    # if total_episodes != 400: ##### temporary
    #     continue
    # if folder_name in running_list:
    #     continue
    folder_name = eval_folder / folder_name / training_episode
    # print(len(os.listdir(folder_name)))

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
    # cell_ave = np.mean(cell_array, axis=0)

    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))
    
    freq_param_list = []
    for tcbk in tcbk_list:
        b = tcbk[2,warm_up_embed:]
        freq = [estimate_frequency_fft(b == antibiotic_value, sampling_unit=delta_t)]
        if not np.isnan(freq):
            freq_param_list += [freq]

    if len(freq_param_list) == 0:
        freq_list += [0]
    else:
        freq_list += [np.mean(freq_param_list)]
    
    eval_cell_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean()]
    eval_cell_std_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std()]

df_generalized_eval["eval_log_cell"] = eval_cell_list
df_generalized_eval["eval_log_cell_std"] = eval_cell_std_list
# df_generalized_eval["eval_log_cell"] = np.log10(df_generalized_eval["eval_final_cell"])
df_generalized_eval["eval_freq"] = freq_list
df_generalized_eval["extinction_frac"] = extinction_frac_list
df_generalized_eval["extinction_rate"] = extinction_rate_list

# %% ----- ----- ----- ----- gen v.s. special ----- ----- ----- ----- %% #
df_constant_eval_app = df_constant_eval.groupby(
    ["antibiotic_value", "nutrient_value", "delay_embed_len", "training_episode", "inst_combination"]
)["eval_log_cell"].mean().reset_index()
df_constant_eval_app = df_constant_eval_app.merge(df_constant_sim_cst_app[["inst_combination", "sim_log_cell"]], on="inst_combination")
df_constant_eval_app["log_diff"] = df_constant_eval_app["sim_log_cell"] - df_constant_eval_app["eval_log_cell"]

df_varenv_eval_app = df_varenv_eval.groupby(
    ["antibiotic_value", "T_k0", "delay_embed_len", "training_episode", "inst_combination"]
)["eval_log_cell"].mean().reset_index()
df_varenv_eval_app = df_varenv_eval_app.merge(df_varenv_sim_cst_app[["inst_combination", "sim_log_cell"]], on="inst_combination")
df_varenv_eval_app["log_diff"] = df_varenv_eval_app["sim_log_cell"] - df_varenv_eval_app["eval_log_cell"]

df_sim_cst_app = pd.concat([df_varenv_sim_cst_app, df_constant_sim_cst_app])
df_eval_app = pd.concat([df_constant_eval_app, df_varenv_eval_app], ignore_index=True)

df_eval_app_trained = df_eval_app.copy()
df_eval_app_trained["trained_env"] = np.where(
    df_eval_app_trained['nutrient_value'].notna(),
    'n' + df_eval_app_trained['nutrient_value'].map('{:.2f}'.format),
    'T' + df_eval_app_trained['T_k0'].map('{:.0f}'.format)
)

# %%
# df_special_gen_eval
# df_constant_sim_cst_app
# df_varenv_sim_cst_app

# df_constant_eval_app_selected = df_constant_eval_app.copy()
# df_constant_eval_app_selected["trained_env"] = df_constant_eval_app_selected["nutrient_value"].map(lambda x: f"n{x:.2f}")
# df_constant_eval_app_selected = df_constant_eval_app_selected[["trained_env", "rep"]]

# df_varenv_eval_app_selected = df_varenv_eval_app.copy()
# df_varenv_eval_app_selected["trained_env"] = df_varenv_eval_app_selected["T_k0"].map(lambda x: f"T{x}")
# df_varenv_eval_app_selected = df_varenv_eval_app_selected[["trained_env", "rep"]]

# # concatenate the selected dataframes with no index
# df_special_gen_selected = pd.concat([df_constant_eval_app_selected, df_varenv_eval_app_selected], ignore_index=True)

# df_special_gen_eval_app = df_special_gen_eval.merge(df_special_gen_selected[['trained_env', 'rep']], on=['trained_env', 'rep'])

df_special_gen_eval_app = df_special_gen_eval.groupby(
    ["antibiotic_value", "trained_env", "delay_embed_len", "training_episode", "inst_combination"]
)["eval_log_cell"].mean().reset_index()


df_special_gen_eval_app["inst_combination"] = df_special_gen_eval_app["inst_combination"].str.replace(r'^(constenv_\d+)$', r'\1.00', regex=True)

df_special_gen_eval_app = df_special_gen_eval_app.merge(df_sim_cst_app[["inst_combination", "sim_log_cell"]], on="inst_combination")
df_special_gen_eval_app["log_diff"] = df_special_gen_eval_app["sim_log_cell"] - df_special_gen_eval_app["eval_log_cell"]

# %%
df_generalized_eval_app = df_generalized_eval[df_generalized_eval["episodes"] == 400].reset_index(drop=True)
# sum_df = df_generalized_eval_app.groupby(["episodes", "rep"])["eval_log_cell"].sum().reset_index()
# min_rep = sum_df.loc[sum_df["eval_log_cell"].idxmin()]
# if isinstance(min_rep, pd.Series):
#     min_rep = min_rep.to_frame().T
# df_generalized_eval_app = df_generalized_eval_app.merge(min_rep[['episodes', 'rep']], on=['episodes', 'rep'])

df_generalized_eval_app = df_generalized_eval_app.groupby(
    ["antibiotic_value", "trained_env", "delay_embed_len", "training_episode", "inst_combination"]
)["eval_log_cell"].mean().reset_index()

df_generalized_eval_app["inst_combination"] = df_generalized_eval_app["inst_combination"].str.replace(r'^(constenv_\d+)$', r'\1.00', regex=True)

df_generalized_eval_app = df_generalized_eval_app.merge(df_sim_cst_app[["inst_combination", "sim_log_cell"]], on="inst_combination")
df_generalized_eval_app["log_diff"] = df_generalized_eval_app["sim_log_cell"] - df_generalized_eval_app["eval_log_cell"]

# %% ----- ----- ----- ----- plot gen v.s. special (bar) ----- ----- ----- ----- %% #
df1 = df_special_gen_eval_app[["trained_env", "inst_combination", "log_diff"]].copy()
df1["source"] = "Single-env agent " + df1["trained_env"]
# df1["color"] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# COLOR_LIST = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
df2 = df_eval_app_trained[["trained_env", "inst_combination", "log_diff"]].copy()
# df2["source"] = "Specialized Agents"
df2["source"] = "Single-env agent " + df2["trained_env"]
# df2["color"] = "#17becf"
df3 = df_generalized_eval_app[["trained_env", "inst_combination", "log_diff"]].copy()
df3["source"] = "Multi-env agent"

combined_df = pd.concat([df1, df2, df3], ignore_index=True)
combined_df = combined_df.sort_values(by=['inst_combination', 'source'])

#####
select_sources = ["Single-env agent n1.00", "Single-env agent n3.00", "Single-env agent T6", "Single-env agent T12", "Multi-env agent"]
select_inst_comb = ["constenv_1.00", "constenv_3.00", "varenv_6", "varenv_12"]
combined_df = combined_df.loc[(combined_df['source'].isin(select_sources)) & (combined_df["inst_combination"].isin(select_inst_comb))].reset_index(drop=True)
combined_df['source'] = combined_df['source'].astype('category')

inst_order = ["Env n1.00", "Env n3.00", "Env T6", "Env T12",]
agent_order = ["Single-env agent n1.00", "Single-env agent n3.00",
               "Single-env agent T6", "Single-env agent T12",
               "Multi-env agent"]

custom_palette = {
    "Single-env agent n1.00": "#edf5d7",
    "Single-env agent n3.00": "#C9F7CF",
    "Single-env agent T6": "#d3e6f4",
    "Single-env agent T12": "#B5AAF7",
    "Multi-env agent": "#dac4cd"
}

box_list = [i*5 for i in range(4)] + [i+16 for i in range(4)]

#####

# text replacement for inst_combination
mapping = {"constenv_1.00": "Env n1.00", "constenv_2.00": "Env n2.00", "constenv_3.00": "Env n3.00", "constenv_4.00": "Env n4.00",
           "varenv_6": "Env T6", "varenv_12": "Env T12", "varenv_18": "Env T18", "varenv_24": "Env T24"}
combined_df['inst_combination'] = combined_df['inst_combination'].replace(mapping)
# plot_order = np.unique(combined_df['inst_combination'])


combined_df["inst_combination"] = pd.Categorical(
    combined_df["inst_combination"],
    categories=inst_order,
    ordered=True
)
combined_df["source"] = pd.Categorical(
    combined_df["source"],
    categories=agent_order,
    ordered=True
)
combined_df = combined_df.sort_values(by=['inst_combination', 'source'])

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=combined_df,
    x='inst_combination',
    y='log_diff',
    order=inst_order,
    gap=0.15,
    hue='source',
    errorbar=('ci', None),  # disable automatic CI bars
    err_kws={'linewidth': 1},
    capsize=0.1,
    palette=custom_palette,
)
ax.axhline(0, color='gray', linewidth=1, linestyle='--')
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.2f', padding=3)
for i, patch in enumerate(ax.patches):
    print(patch.get_label())
    print(patch.get_height())

    if i not in box_list:
        continue
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax.set_title('Generalizability of Specialized Agents')
ax.set_xlabel('Evaluation environment')
# ax.set_ylabel(r'$\Delta \log(population\ size)$')
# ax.set_ylabel(r'$\log(P_{constant})-\log(P_{pulsing})$')
ax.set_ylabel(r'Relative performance')
ax.legend(title="Agent Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(BASE_PATH / "figures_jpg" / "special_gen_mean.jpg", dpi=600, bbox_inches='tight')
fig.savefig(BASE_PATH / "figures_pdf" / "special_gen_mean.pdf", dpi=600, bbox_inches='tight')
