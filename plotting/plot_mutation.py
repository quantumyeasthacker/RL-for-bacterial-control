# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
# from scipy.optimize import curve_fit
import matplotlib as mpl
from utils import expand_and_fill, estimate_frequency_fft, down_edge_detection, load_logger_data_new
from pathlib import Path


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EPS = 1e-6
# default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLOR_LIST = ["#dec60c", "#a7c82f", "#548c6a"]

BASE_PATH = Path("/home/zihangw/BacteriaAdaptation/jk_agents/mutation/")
BASE_PATH_PLOT = Path("/home/zihangw/BacteriaAdaptation/")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %% ----- ----- ----- ----- constant sim ----- ----- ----- ----- %% #
# env_type = "constenv"
param_file = BASE_PATH / f"param_simtest_tgt.txt"
sim_folder = BASE_PATH / f"results_delay_30_record_generalized_simtest"

n_trials = 100

with open(param_file, "r") as f:
    param_sim = f.readlines()

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

df_constant_sim = pd.DataFrame(param_sim)
df_constant_sim.columns = ["half_period", "antibiotic_value", "inst_env", "inst_variable", "mutprob"]
df_constant_sim["inst_combination"] = df_constant_sim["inst_env"] + " " + df_constant_sim["inst_variable"]  + ' mutprob ' + df_constant_sim['mutprob']

df_constant_sim["half_period"] = df_constant_sim["half_period"].astype(float)
df_constant_sim["antibiotic_value"] = df_constant_sim["antibiotic_value"].astype(float)

final_cell_list = []
final_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_sim:

    half_period = int(param[0])
    initialize_app = "constant"
    antibiotic_value = float(param[1])
    folder_name = sim_folder / f"a{antibiotic_value}_{param[2]}_{param[3]}_mutprob{param[4]}_value_check" / f"{initialize_app}_{half_period}/"

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials, False)

    if half_period == 0:
        freq = 0
    else:
        freq = 1 / (half_period * 2 * delta_t)
    freq_list.append(freq)

    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))

    final_cell_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean())
    final_cell_std_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std())

df_constant_sim["sim_log_cell"] = final_cell_list
df_constant_sim["sim_log_cell_std"] = final_cell_std_list
# df_constant_sim["sim_log_cell"] = np.log10(df_constant_sim["sim_final_cell"])
df_constant_sim["sim_freq"] = freq_list
df_constant_sim["extinction_frac"] = extinction_frac_list
df_constant_sim["extinction_rate"] = extinction_rate_list

df_constant_sim_cst_app = df_constant_sim


# %% ----- ----- ----- ----- generalized eval ----- ----- ----- ----- %% #
env_type = "generalized"
param_file = BASE_PATH / f"param_agent_delay_30_{env_type}_eval_tgt.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"

n_trials_eval = 100

# running_list = ["a3.72_const1_4_T6_24_delay30_episodes600_rep0"]

with open(param_file, "r") as f:
    param_agent = f.readlines()
print(param_agent)
param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_generalized_eval = pd.DataFrame(param_agent)
df_generalized_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode", "mutprob"]

df_generalized_eval["inst_variable"] = df_generalized_eval["inst_variable"].astype(float).astype(int).astype(str)
# df_generalized_eval["inst_combination"] = df_generalized_eval["inst_env"] + " " + df_generalized_eval["inst_variable"] + ' mutprob ' + df_generalized_eval['mutprob']
df_generalized_eval["inst_combination"] = df_generalized_eval["inst_env"] + " " + df_generalized_eval["inst_variable"] + ' mutprob ' + df_generalized_eval['mutprob']

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
    # antibiotic_value = float(param[0])
    # trial_name = param[1]
    # delay_embed_len = int(param[2])
    # eval_env = param[3]
    # eval_variable = param[4]
    # rep_run = int(param[5])
    # total_episodes = int(param[6])
    # training_episode = param[7]
    # mutate_prob = float(param[8])

    # folder_name = f"{trial_name}_{eval_env}_{eval_variable}_mutprob{mutate_prob}"

    antibiotic_value = float(param[0])
    total_episodes = int(param[6])
    training_episode = param[7]
    folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_episodes{param[6]}_rep{param[5]}_{param[3]}_{param[4]}_mutprob{param[8]}"
    
    # if total_episodes != 400: ##### temporary
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

# %% ----- ----- ----- ----- new agent ----- ----- ----- ----- %% #
param_file = BASE_PATH / f"param_mutprob_agent_eval.txt"
eval_folder = BASE_PATH / f"gen_nonstat_mut_zh_eval"

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()
param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_new_agent_eval = pd.DataFrame(param_agent)
df_new_agent_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode", "mutprob"]

df_new_agent_eval["inst_variable"] = df_new_agent_eval["inst_variable"].astype(float).astype(int).astype(str)

df_new_agent_eval["inst_combination"] = df_new_agent_eval["inst_env"] + " " + df_new_agent_eval["inst_variable"] + ' mutprob ' + df_new_agent_eval['mutprob']

df_new_agent_eval["antibiotic_value"] = df_new_agent_eval["antibiotic_value"].astype(float)
df_new_agent_eval["delay_embed_len"] = df_new_agent_eval["delay_embed_len"].astype(int)
df_new_agent_eval["rep"] = df_new_agent_eval["rep"].astype(int)
df_new_agent_eval["episodes"] = df_new_agent_eval["episodes"].astype(int)

eval_cell_list = []
eval_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_agent:
    antibiotic_value = float(param[0])
    trial_name = param[1]
    # delay_embed_len = int(param[2])
    eval_env = param[3]
    eval_variable = param[4]
    # rep_run = int(param[5])
    # total_episodes = int(param[6])
    training_episode = param[7]
    mutate_prob = float(param[8])

    folder_name = f"{trial_name}_{eval_env}_{eval_variable}_mutprob{mutate_prob}"

    # antibiotic_value = float(param[0])
    # total_episodes = int(param[6])
    # training_episode = param[7]
    # folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_episodes{param[6]}_rep{param[5]}_{param[3]}_{param[4]}_mutprob{param[8]}"
    
    folder_name = eval_folder / folder_name / training_episode
    
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

df_new_agent_eval["eval_log_cell"] = eval_cell_list
df_new_agent_eval["eval_log_cell_std"] = eval_cell_std_list
# df_new_agent_eval["eval_log_cell"] = np.log10(df_new_agent_eval["eval_final_cell"])
df_new_agent_eval["eval_freq"] = freq_list
df_new_agent_eval["extinction_frac"] = extinction_frac_list
df_new_agent_eval["extinction_rate"] = extinction_rate_list
# df_new_agent_eval = df_new_agent_eval.loc[df_new_agent_eval["episodes"] == 400].reset_index(drop=True) ##### temporary


# %% ----- ----- ----- ----- gen v.s. sim ----- ----- ----- ----- %% #

# import re
# def replace_dot00(text):
#     return re.sub(r'\b(\d+)\.00\b', lambda m: str(int(m.group(1))), text)

sum_df = df_generalized_eval.groupby(["episodes", "trained_env", "rep"])["eval_log_cell"].sum().reset_index()
# min_rep = sum_df.loc[sum_df.groupby(["episodes", "trained_env", "mutprob"])["eval_log_cell"].idxmin()]
min_rep = sum_df.loc[sum_df["eval_log_cell"].idxmin()]

if isinstance(min_rep, pd.Series):
    min_rep = min_rep.to_frame().T
df_generalized_eval_app = df_generalized_eval.merge(min_rep[['episodes', "trained_env", 'rep']], on=['episodes', 'rep', "trained_env"])
# df_generalized_eval_app['inst_combination'] = df_generalized_eval_app['inst_combination'].apply(replace_dot00)

df_sim_cst_app = df_constant_sim_cst_app

df_generalized_eval_app = df_generalized_eval_app.merge(df_sim_cst_app[["inst_combination", "sim_log_cell"]], on="inst_combination")
df_generalized_eval_app["log_diff"] = df_generalized_eval_app["sim_log_cell"] - df_generalized_eval_app["eval_log_cell"]

# df_generalized_eval_app.sort_values(by='trained_env')

df_new_agent_eval_app = df_new_agent_eval.merge(df_sim_cst_app[["inst_combination", "sim_log_cell"]], on="inst_combination")
df_new_agent_eval_app["log_diff"] = df_new_agent_eval_app["sim_log_cell"] - df_new_agent_eval_app["eval_log_cell"]

# %% ----- ----- ----- ----- plot gen v.s. sim (bar) ----- ----- ----- ----- %% #
# df1 = df_generalized_eval_app[["inst_combination", "trained_env", "log_diff", "eval_log_cell_std", "mutprob", "extinction_frac", "extinction_rate"]].copy()
# df1["source"] = "Generalized Agents " + df1["inst_combination"]

# df2 = df_new_agent_eval[["inst_combination", "trained_env", "rep", "log_diff", "eval_log_cell_std", "mutprob", "extinction_frac", "extinction_rate"]].copy()
# df2["source"] = "New Agents " + df2["inst_combination"] + df2["rep"].astype(str)

# # df1["color"] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# # COLOR_LIST = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# fig, ax = plt.subplots(figsize=(10, 6))

# df1.rename(columns={'eval_log_cell_std': 'std'}, inplace=True)

# sns.barplot(
#     data=df1,
#     x='inst_combination',
#     y='log_diff',
#     hue='source',
#     errorbar=('ci', None),  # disable automatic CI bars
#     err_kws={'linewidth': 1},
#     capsize=0.1
# )
# ax.axhline(0, color='gray', linewidth=1, linestyle='--')
# # for container in ax.containers:
# #     ax.bar_label(container, fmt='%.2f', padding=3)
# for patch in ax.patches:
#     patch.set_edgecolor('black')
#     patch.set_linewidth(1)


# # ax.set_title('Generalized Agent Performance Against Mutating Cells')
# ax.set_xlabel('')
# ax.set_ylabel(r'$\log(P_{constant})-\log(P_{pulsing})$')

# ax.legend_.remove()
# ax.set_xticks(df1['inst_combination'])
# # ax.set_xticklabels(combined_df['env_comb'])
# # group_pos = np.linspace(1,len(combined_df['cell_comb'])-2.5,len(combined_df['cell_comb'].drop_duplicates()))
# # group_names = combined_df['cell_comb'].drop_duplicates()
# # for pos,name in zip(group_pos, group_names):
# #     ax.text(pos, -5.5, name, ha='center', va='top', rotation=55)
# plt.xticks(rotation=90)
# fig.tight_layout()

# # Fig_PATH = BASE_PATH / "figures_pdf"
# # os.makedirs(Fig_PATH, exist_ok=True)
# # fig.savefig(BASE_PATH / "figures_pdf" / "gen_mutate_bar_fastmut.pdf", dpi=600, bbox_inches='tight')


# %% ----- ----- ----- ----- plot gen v.s. mutate (scatter) ----- ----- ----- ----- %% #
mean_df = df_generalized_eval_app.groupby(["episodes", "trained_env", "rep", "mutprob"])["log_diff"].mean().reset_index()
mean_df = mean_df.sort_values(by='mutprob')
mean_df['mutprob'] = mean_df['mutprob'].astype(float)

mean_df_new_agents = df_new_agent_eval_app.copy()
# .loc[df_new_agent_eval_app["training_episode"] == "episode_499"].copy()
mean_df_new_agents = mean_df_new_agents.groupby(["trained_env", "training_episode", "mutprob"])["log_diff"].mean().reset_index()
mean_df_new_agents['mutprob'] = mean_df_new_agents['mutprob'].astype(float)

fig = plt.figure()
plt.scatter(mean_df['mutprob'], mean_df['log_diff'], label="multi-env agent", color='k')
plt.plot(mean_df['mutprob'], mean_df['log_diff'], color='k')
for trained_env in mean_df_new_agents['trained_env'].unique():
    for training_episode in mean_df_new_agents['training_episode'].unique():
        subset = mean_df_new_agents[(mean_df_new_agents['trained_env'] == trained_env) & (mean_df_new_agents['training_episode'] == training_episode)]
        plt.scatter(subset['mutprob'], subset['log_diff'], label=f"{trained_env} {training_episode}")
        plt.plot(subset['mutprob'], subset['log_diff'])
plt.xlabel('Mutation probability', fontsize=14)
plt.ylabel('Agent Performance, $\log P_{constant}-\log P_{policy}$', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()
plt.show()

Fig_PATH = BASE_PATH_PLOT / "figures_pdf"
os.makedirs(Fig_PATH, exist_ok=True)
fig.savefig(Fig_PATH  / "mutprob_new_agent.pdf", dpi=600, bbox_inches='tight')

Fig_PATH = BASE_PATH_PLOT / "figures_jpg"
os.makedirs(Fig_PATH, exist_ok=True)
fig.savefig(Fig_PATH / "mutprob_new_agent.jpg", dpi=600, bbox_inches='tight')

# %%
