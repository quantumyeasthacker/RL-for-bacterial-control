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
COLOR_LIST = ["#dec60c", "#a7c82f", "#548c6a", "#5e6b75"]

# BASE_PATH = Path("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/")
BASE_PATH = Path("/home/zihangw/BacteriaAdaptation")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %% ----- ----- ----- ----- generalized eval 30 ----- ----- ----- ----- %% #
# env_type = "generalized"
# param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
# eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# # eval_folder = BASE_PATH / "20250421_generalized" / eval_folder

# n_trials_eval = 100

# # running_list = ["a3.72_const1_4_T6_24_delay30_episodes600_rep0"]

# with open(param_file, "r") as f:
#     param_agent = f.readlines()

# param_agent = [x.strip() for x in param_agent]
# param_agent = [x.split(" ") for x in param_agent]

# param_agent = [x for x in param_agent if x[6] == "400"] # number episodes == 400

# df_generalized_eval = pd.DataFrame(param_agent)
# df_generalized_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode"]
# df_generalized_eval["inst_combination"] = df_generalized_eval["inst_env"] + "_" + df_generalized_eval["inst_variable"]

# df_generalized_eval["antibiotic_value"] = df_generalized_eval["antibiotic_value"].astype(float)
# df_generalized_eval["delay_embed_len"] = df_generalized_eval["delay_embed_len"].astype(int)
# df_generalized_eval["rep"] = df_generalized_eval["rep"].astype(int)
# df_generalized_eval["episodes"] = df_generalized_eval["episodes"].astype(int)

# # df_generalized_eval = df_generalized_eval.loc[df_generalized_eval["episodes"] == 400].reset_index(drop=True) ##### temporary

# eval_cell_list = []
# eval_cell_std_list = []
# freq_list = []
# extinction_frac_list = []
# extinction_rate_list = []
# for param in param_agent:
#     antibiotic_value = float(param[0])
#     total_episodes = int(param[6])
#     training_episode = param[7]
#     folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_episodes{param[6]}_rep{param[5]}_{param[3]}_{param[4]}"
#     # if total_episodes != 400: ##### temporary
#     #     continue
#     # if folder_name in running_list:
#     #     continue
#     folder_name = eval_folder / folder_name / training_episode
#     # print(len(os.listdir(folder_name)))

#     tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
#     # cell_ave = np.mean(cell_array, axis=0)

#     extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
#     extinction_frac_list.append(np.mean(extinction))

#     extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
#     extinction_rate_list.append(np.mean(extinction_rate))
    
#     freq_param_list = []
#     for tcbk in tcbk_list:
#         b = tcbk[2,warm_up_embed:]
#         freq = [estimate_frequency_fft(b == antibiotic_value, sampling_unit=delta_t)]
#         if not np.isnan(freq):
#             freq_param_list += [freq]

#     if len(freq_param_list) == 0:
#         freq_list += [0]
#     else:
#         freq_list += [np.mean(freq_param_list)]
    
#     eval_cell_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean()]
#     eval_cell_std_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std()]

# df_generalized_eval["eval_log_cell"] = eval_cell_list
# df_generalized_eval["eval_log_cell_std"] = eval_cell_std_list
# # df_generalized_eval["eval_log_cell"] = np.log10(df_generalized_eval["eval_final_cell"])
# df_generalized_eval["eval_freq"] = freq_list
# df_generalized_eval["extinction_frac"] = extinction_frac_list
# df_generalized_eval["extinction_rate"] = extinction_rate_list

# %% ----- ----- ----- ----- generalized eval else ----- ----- ----- ----- %% #
env_type = "generalized"
param_file = BASE_PATH / "param_space" / f"param_agent_delays_{env_type}_eval_more.txt"
eval_folder = BASE_PATH / f"results_delays_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250421_generalized" / eval_folder

n_trials_eval = 100

# running_list = ["a3.72_const1_4_T6_24_delay30_episodes600_rep0"]

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_generalized_eval_else = pd.DataFrame(param_agent)
df_generalized_eval_else.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode"]
df_generalized_eval_else["inst_combination"] = df_generalized_eval_else["inst_env"] + "_" + df_generalized_eval_else["inst_variable"]

df_generalized_eval_else["antibiotic_value"] = df_generalized_eval_else["antibiotic_value"].astype(float)
df_generalized_eval_else["delay_embed_len"] = df_generalized_eval_else["delay_embed_len"].astype(int)
df_generalized_eval_else["rep"] = df_generalized_eval_else["rep"].astype(int)
df_generalized_eval_else["episodes"] = df_generalized_eval_else["episodes"].astype(int)

# df_generalized_eval_else = df_generalized_eval_else.loc[df_generalized_eval_else["episodes"] == 400].reset_index(drop=True) ##### temporary

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

df_generalized_eval_else["eval_log_cell"] = eval_cell_list
df_generalized_eval_else["eval_log_cell_std"] = eval_cell_std_list
# df_generalized_eval_else["eval_log_cell"] = np.log10(df_generalized_eval_else["eval_final_cell"])
df_generalized_eval_else["eval_freq"] = freq_list
df_generalized_eval_else["extinction_frac"] = extinction_frac_list
df_generalized_eval_else["extinction_rate"] = extinction_rate_list


# %% ave extinction rate v.s. history length
# df_embed = df_generalized_eval_else.copy()
# # df_embed = pd.concat([df_generalized_eval, df_generalized_eval_else], ignore_index=True)
# df_embed = df_embed.sort_values(by=["delay_embed_len", "rep", "inst_combination"]).reset_index(drop=True)

# sum_df = df_embed.groupby(["delay_embed_len", "rep"])["extinction_rate"].mean().reset_index()
# gen_rep = sum_df.loc[sum_df.groupby(["delay_embed_len"])["extinction_rate"].idxmax()]

# if isinstance(gen_rep, pd.Series):
#     gen_rep = gen_rep.to_frame().T

# # df_embed_app = df_embed.merge(gen_rep[['delay_embed_len', 'rep']], on=['delay_embed_len', 'rep'])
# # df_embed_app["inst_combination"] = df_embed_app["inst_combination"].str.replace(r'^(constenv_\d+)$', r'\1.00', regex=True)

# plt.figure(figsize=(8, 6))
# plt.plot(gen_rep["delay_embed_len"], gen_rep["extinction_rate"], marker='o')
# plt.xlabel("History Length")
# plt.ylabel("Average extinction rate")

# plt.savefig(BASE_PATH / "figures_pdf" / "extinction_rate_vs_history_length.pdf", bbox_inches='tight')
# plt.savefig(BASE_PATH / "figures_jpg" / "extinction_rate_vs_history_length.jpg", bbox_inches='tight')

# plt.show()

# %% ave extinction rate v.s. episodes
df_embed = df_generalized_eval_else.copy()
df_embed = df_embed.sort_values(by=["delay_embed_len", "training_episode", "inst_combination"]).reset_index(drop=True)

sum_df = df_embed.groupby(["delay_embed_len", "training_episode"])["extinction_rate"].mean().reset_index()
sum_df["training_episode_int"] = sum_df["training_episode"].apply(lambda x: int(x.split("_")[1]))

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(data=sum_df, x="training_episode_int", y="extinction_rate",
                hue="delay_embed_len", marker='o', ax=ax)
plt.xlabel("Training Episodes")
plt.ylabel("Average Extinction Rate")
plt.legend(title="History Length")
plt.savefig(BASE_PATH / "figures_pdf" / "extinction_rate_vs_training_episodes.pdf", bbox_inches='tight')
plt.savefig(BASE_PATH / "figures_jpg" / "extinction_rate_vs_training_episodes.jpg", bbox_inches='tight')

# %%
