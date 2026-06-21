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

BASE_PATH = Path("/home/zihangw/BacteriaAdaptation/jk_agents/episodic/")
BASE_PATH_PLOT = Path("/home/zihangw/BacteriaAdaptation/")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %% ----- ----- ----- ----- constant sim ----- ----- ----- ----- %% #
env_type = "constenv"
param_file = BASE_PATH / f"param_simtest_robust_tgt.txt"
sim_folder = BASE_PATH / f"results_generalized_simtest"

n_trials = 50

with open(param_file, "r") as f:
    param_sim = f.readlines()

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

df_constant_sim = pd.DataFrame(param_sim)
df_constant_sim.columns = ["half_period", "antibiotic_value", "inst_env", "inst_variable", "rep", "phiSmax", "phiRmax"]
df_constant_sim["inst_combination"] = "phiS " + df_constant_sim['phiSmax'] + " phiR " + df_constant_sim['phiRmax'] + " " + df_constant_sim["inst_env"] + " " + df_constant_sim["inst_variable"]

df_constant_sim = df_constant_sim.drop(columns=["rep"])
df_constant_sim = df_constant_sim.drop_duplicates().reset_index(drop=True)

param_sim = df_constant_sim.values.tolist()

final_cell_list = []
final_cell_std_list = []
freq_list = []
extinction_frac_list = []
extinction_rate_list = []
for param in param_sim:

    half_period = int(param[0])
    initialize_app = "constant"
    antibiotic_value = float(param[1])
    folder_name = sim_folder / f"a{antibiotic_value}_{param[2]}_{param[3]}_phiR{param[5]}_phiS{param[4]}_value_check" / f"{initialize_app}_{half_period}/"

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
param_file = BASE_PATH / f"param_gen_agent_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_generalized_eval"

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()
param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_generalized_eval = pd.DataFrame(param_agent)
df_generalized_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode", "phiSmax", "phiRmax"]

df_generalized_eval["inst_variable"] = df_generalized_eval["inst_variable"].astype(float).astype(int).astype(str)

df_generalized_eval["inst_combination"] = "phiS " + df_generalized_eval['phiSmax'] + " phiR " + df_generalized_eval['phiRmax'] + " " + df_generalized_eval["inst_env"] + " " + df_generalized_eval["inst_variable"]

df_generalized_eval["antibiotic_value"] = df_generalized_eval["antibiotic_value"].astype(float)
df_generalized_eval["delay_embed_len"] = df_generalized_eval["delay_embed_len"].astype(int)
df_generalized_eval["rep"] = df_generalized_eval["rep"].astype(int)
df_generalized_eval["episodes"] = df_generalized_eval["episodes"].astype(int)

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
    
    phiS = float(param[8])
    phiR = float(param[9])

    folder_name = f"{trial_name}_{eval_env}_{eval_variable}_phiS{phiS}_phiR{phiR}"

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

df_generalized_eval["eval_log_cell"] = eval_cell_list
df_generalized_eval["eval_log_cell_std"] = eval_cell_std_list
# df_generalized_eval["eval_log_cell"] = np.log10(df_generalized_eval["eval_final_cell"])
df_generalized_eval["eval_freq"] = freq_list
df_generalized_eval["extinction_frac"] = extinction_frac_list
df_generalized_eval["extinction_rate"] = extinction_rate_list
# df_generalized_eval = df_generalized_eval.loc[df_generalized_eval["episodes"] == 400].reset_index(drop=True) ##### temporary

# %% ----- ----- ----- ----- new agent ----- ----- ----- ----- %% #
param_file = BASE_PATH / f"param_epi_agent_eval.txt"
eval_folder = BASE_PATH / f"gen_rand_init_zh_eval"

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()
param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_new_agent_eval = pd.DataFrame(param_agent)
df_new_agent_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "inst_env", "inst_variable", "rep", "episodes", "training_episode", "phiSmax", "phiRmax"]

df_new_agent_eval["inst_variable"] = df_new_agent_eval["inst_variable"].astype(float).astype(int).astype(str)
df_new_agent_eval["inst_combination"] = "phiS " + df_new_agent_eval['phiSmax'] + " phiR " + df_new_agent_eval['phiRmax'] + " " + df_new_agent_eval["inst_env"] + " " + df_new_agent_eval["inst_variable"]

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
    
    phiS = float(param[8])
    phiR = float(param[9])

    folder_name = f"{trial_name}_{eval_env}_{eval_variable}_phiS{phiS}_phiR{phiR}"

    folder_name = eval_folder / folder_name / training_episode
    
    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
    
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

# %%
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

# %%
env_type_plot = "varenv"
df1 = df_generalized_eval_app[["inst_combination", "inst_env", "inst_variable", "trained_env", "log_diff", "eval_log_cell_std", "phiRmax", "phiSmax", "extinction_frac", "extinction_rate"]].copy()
df1["source"] = "Generalized Agents " + df1["inst_combination"]
df1['env_comb'] = df1['inst_env'] + " " + df1['inst_variable']
df1['cell_comb'] = "phiS " + df1['phiSmax'] + " phiR " + df1['phiRmax']
# df1["color"] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# COLOR_LIST = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

combined_df = df1.loc[df1["inst_env"] == env_type_plot].reset_index(drop=True)
combined_df.rename(columns={'eval_log_cell_std': 'std'}, inplace=True)
# combined_df = combined_df.drop_duplicates(subset='source').reset_index(drop=True) ##### temporary due to repeating analysis for sim

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=combined_df,
    x='inst_combination',
    y='log_diff',
    hue='source',
    errorbar=('ci', None),  # disable automatic CI bars
    err_kws={'linewidth': 1},
    capsize=0.1
)
ax.axhline(0, color='gray', linewidth=1, linestyle='--')
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.2f', padding=3)
for patch in ax.patches:
    patch.set_edgecolor('black')
    patch.set_linewidth(1)


ax.set_title('Generalized Agent Performance Against Changes in Cell Parameters')
ax.set_xlabel('')
ax.set_ylabel(r'$\log(P_{constant})-\log(P_{pulsing})$')
plt.ylim(-3,4.5)

ax.legend_.remove()
ax.set_xticks(combined_df['inst_combination'])
ax.set_xticklabels(combined_df['env_comb'])
group_pos = np.linspace(1,len(combined_df['cell_comb'])-2.5,len(combined_df['cell_comb'].drop_duplicates()))
group_names = combined_df['cell_comb'].drop_duplicates()
for pos,name in zip(group_pos, group_names):
    ax.text(pos, -5.5, name, ha='center', va='top', rotation=55)
plt.xticks(rotation=80)
fig.tight_layout()
fig.savefig(BASE_PATH_PLOT / f"figures_pdf/randinit_generalized_agent_{env_type_plot}.pdf")

# %%
df2 = df_new_agent_eval_app[["inst_combination", "inst_env", "inst_variable", "trained_env", "training_episode", "log_diff", "eval_log_cell_std", "phiRmax", "phiSmax", "extinction_frac", "extinction_rate"]].copy()
df2["source"] = "New Agents " + df2["inst_combination"]
df2['env_comb'] = df2['inst_env'] + " " + df2['inst_variable']
df2['cell_comb'] = "phiS " + df2['phiSmax'] + " phiR " + df2['phiRmax']

train_env_2 = df2["trained_env"].unique()
train_episode_2 = df2["training_episode"].unique()

for trained_env in train_env_2:
    for training_episode in train_episode_2:
        fig, ax = plt.subplots(figsize=(10, 6))

        combined_df = df2.loc[(df2["inst_env"] == env_type_plot) & (df2["trained_env"] == trained_env) & (df2["training_episode"] == training_episode)].reset_index(drop=True)
        combined_df.rename(columns={'eval_log_cell_std': 'std'}, inplace=True)
        # combined_df = combined_df.drop_duplicates(subset='source').reset_index(drop=True) ##### temporary due to repeating analysis for sim

        sns.barplot(
            data=combined_df,
            x='inst_combination',
            y='log_diff',
            hue='source',
            errorbar=('ci', None),  # disable automatic CI bars
            err_kws={'linewidth': 1},
            capsize=0.1
        )
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        # for container in ax.containers:
        #     ax.bar_label(container, fmt='%.2f', padding=3)
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        ax.set_title(f'New Agent Performance Against Changes in Cell Parameters\nTrained in {trained_env} for {training_episode} Episodes')
        ax.set_xlabel('')
        ax.set_ylabel(r'$\log(P_{constant})-\log(P_{pulsing})$')
        plt.ylim(-3,4.5)

        ax.legend_.remove()
        ax.set_xticks(combined_df['inst_combination'])
        ax.set_xticklabels(combined_df['env_comb'])
        group_pos = np.linspace(1,len(combined_df['cell_comb'])-2.5,len(combined_df['cell_comb'].drop_duplicates()))
        group_names = combined_df['cell_comb'].drop_duplicates()
        for pos,name in zip(group_pos, group_names):
            ax.text(pos, -5.5, name, ha='center', va='top', rotation=55)
        plt.xticks(rotation=80)
        fig.tight_layout()
        plt.show()

        fig.savefig(BASE_PATH_PLOT / f"figures_pdf/randinit_new_agent_{env_type_plot}_{trained_env}_{training_episode}.pdf")

# %%
