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

BASE_PATH = Path("/Users/Josiah/Documents/OSPool/jun15_gen_eval_GP")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %% ----- ----- ----- ----- sim ----- ----- ----- ----- %% #
param_file = BASE_PATH / f"param_simtest_GP.txt"
sim_folder = BASE_PATH / f"results_generalized_simtest"

n_trials = 50

with open(param_file, "r") as f:
    param_sim = f.readlines()
    # print(param_sim)

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]


df_constant_sim = pd.DataFrame(param_sim)
df_constant_sim.columns = ["half_period", "initialize_app", "antibiotic_value", "ls", "amp", "rep"]
df_constant_sim["inst_combination"] = "ls " + df_constant_sim["ls"] + " amp " + df_constant_sim["amp"]

df_constant_sim["half_period"] = df_constant_sim["half_period"].astype(float)
df_constant_sim["antibiotic_value"] = df_constant_sim["antibiotic_value"].astype(float)

final_cell_list = []
final_cell_std_list = []
freq_list = []
for param in param_sim:
    # if param[5] != 0:
    #     continue

    half_period = int(param[0])
    initialize_app = param[1]
    antibiotic_value = float(param[2])
    folder_name = sim_folder / f"a{param[2]}_GP_ls{float(param[3])}_amp{float(param[4])}_value_check" / f"{param[1]}_{param[0]}/"

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials, False)
    if half_period == 0:
        freq = 0
    else:
        freq = 1 / (half_period * 2 * delta_t)
    freq_list.append(freq)

    final_cell_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean())
    final_cell_std_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std())

df_constant_sim["sim_log_cell"] = final_cell_list
df_constant_sim["sim_log_cell_std"] = final_cell_std_list
df_constant_sim["sim_freq"] = freq_list

df_constant_sim_cst_app = df_constant_sim[df_constant_sim["initialize_app"] == "constant"]


# %% ----- ----- ----- ----- generalized eval ----- ----- ----- ----- %% #
env_type = "generalized"
param_file = BASE_PATH / f"param_agent_eval_GP.txt"
eval_folder = BASE_PATH / f"results_{env_type}_eval"

n_trials_eval = 10

# running_list = ["a3.72_const1_4_T6_24_delay30_episodes600_rep0"]

with open(param_file, "r") as f:
    param_agent = f.readlines()
print(param_agent)
param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

df_generalized_eval = pd.DataFrame(param_agent)
df_generalized_eval.columns = ["antibiotic_value", "trained_env", "delay_embed_len", "rep", "episodes", "training_episode", "ls", "amp"]
df_generalized_eval["inst_combination"] = "ls " + df_generalized_eval["ls"] + " amp " + df_generalized_eval["amp"]

df_generalized_eval["antibiotic_value"] = df_generalized_eval["antibiotic_value"].astype(float)
df_generalized_eval["delay_embed_len"] = df_generalized_eval["delay_embed_len"].astype(int)
df_generalized_eval["rep"] = df_generalized_eval["rep"].astype(int)
df_generalized_eval["episodes"] = df_generalized_eval["episodes"].astype(int)

df_generalized_eval = df_generalized_eval.loc[df_generalized_eval["episodes"] == 500].reset_index(drop=True) ##### temporary

eval_cell_list = []
eval_cell_std_list = []
freq_list = []
for param in param_agent:
    antibiotic_value = float(param[0])
    total_episodes = int(param[4])
    training_episode = param[5]
    folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_episodes{param[4]}_rep{param[3]}_GP_ls{float(param[6])}_amp{float(param[7])}"
    if total_episodes != 500: ##### temporary
        continue
    # if folder_name in running_list:
    #     continue
    folder_name = eval_folder / folder_name / training_episode
    # print(len(os.listdir(folder_name)))

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
    # cell_ave = np.mean(cell_array, axis=0)

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
df_generalized_eval["eval_freq"] = freq_list

# %% ----- ----- ----- ----- gen v.s. sim ----- ----- ----- ----- %% #

sum_df = df_generalized_eval.groupby(["episodes", "trained_env", "rep", "ls", "amp"])["eval_log_cell"].sum().reset_index()
min_rep = sum_df.loc[sum_df.groupby(["episodes", "trained_env", "ls", "amp"])["eval_log_cell"].idxmin()]
if isinstance(min_rep, pd.Series):
    min_rep = min_rep.to_frame().T
df_generalized_eval_app = df_generalized_eval.merge(min_rep[['episodes', "trained_env", 'rep', "ls", "amp"]], on=['episodes', 'rep', "trained_env", "ls", "amp"])
df_generalized_eval_app["inst_combination"] = df_generalized_eval_app["inst_combination"].str.replace(r'^(constenv_\d+)$', r'\1.00', regex=True)

df_sim_cst_app = df_constant_sim_cst_app
df_generalized_eval_app = df_generalized_eval_app.merge(df_sim_cst_app[["inst_combination", "sim_log_cell"]], on="inst_combination")
df_generalized_eval_app["log_diff"] = df_generalized_eval_app["sim_log_cell"] - df_generalized_eval_app["eval_log_cell"]

df_generalized_eval_app.sort_values(by='trained_env')

# %% ----- ----- ----- ----- plot gen v.s. sim (bar) ----- ----- ----- ----- %% #
df1 = df_generalized_eval_app[["inst_combination", "log_diff", "eval_log_cell_std"]].copy()
df1["source"] = "Generalized Agents " + df1["inst_combination"]
# df1["color"] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# COLOR_LIST = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, ax = plt.subplots(figsize=(10, 6))

combined_df = df1
combined_df.rename(columns={'eval_log_cell_std': 'std'}, inplace=True)
combined_df = combined_df.drop_duplicates(subset='source') ##### temporary due to repeating analysis

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
for patch in ax.patches:
    patch.set_edgecolor('black')
    patch.set_linewidth(1)


ax.set_title('Generalized Agent Performance in Different GP Nutrient Environments')
ax.set_xlabel('Evaluation Environment')
ax.set_ylabel(r'$\log(P_{constant})-\log(P_{pulsing})$')
ax.legend_.remove()
plt.xticks(rotation=45)
fig.tight_layout()

Fig_PATH = BASE_PATH / "figures_pdf"
os.makedirs(Fig_PATH, exist_ok=True)
fig.savefig(BASE_PATH / "figures_pdf" / "Generalized_bar_GP.pdf", dpi=600, bbox_inches='tight')


# %% ----- ----- ----- ----- plot individual trajectories ----- ----- ----- ----- %% #

for param in param_agent:
    antibiotic_value = float(param[0])
    total_episodes = int(param[4])
    training_episode = param[5]
    folder_name = f"a{param[0]}_{param[1]}_delay{param[2]}_episodes{param[4]}_rep{param[3]}_GP_ls{float(param[6])}_amp{float(param[7])}"
    if total_episodes != 500: ##### temporary
        continue

    folder_name = eval_folder / folder_name / training_episode

    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)

    freq_param_list = []
    for tcbk in tcbk_list:
        t = tcbk[0,warm_up_embed:]
        c = tcbk[1,warm_up_embed:]
        b = tcbk[2,warm_up_embed:]
        k = tcbk[3,warm_up_embed:]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        axes[2].plot(t, c, color='blue')
        axes[2].set_ylabel("Population Size")
        axes[2].set_yscale('log')

        axes[1].plot(t, k, color='green')
        axes[1].set_ylabel("Nutrient Conc.")

        axes[0].plot(t, b, color='red')
        axes[0].set_ylabel("Antibiotic Conc.")

        plt.subplots_adjust(hspace=0)
        plt.show()



# %%
