# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
# from scipy.optimize import curve_fit
import matplotlib as mpl
from utils import expand_and_fill, estimate_frequency_fft, down_edge_detection, load_logger_data_new, get_best_row_extinct_rate, plot_single_varenv, plot_one_traj
from pathlib import Path


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EPS = 1e-6
# default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# COLOR_LIST = ["#dec60c", "#a7c82f", "#548c6a"]

# BASE_PATH = Path("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/")
BASE_PATH = Path("/home/zihangw/BacteriaAdaptation")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)
n_bootstraps = 1000

color_list = ["#cf7171", "#569122", "#000000"]

# %% ----- ----- ----- ----- control env sim (plot traj) ----- ----- ----- ----- %% #
env_type = "controlenv"
param_file = BASE_PATH / "param_space" / f"param_non_monotonic_pulsing_{env_type}.txt"
sim_folder = BASE_PATH / f"results_sim_{env_type}"
# sim_folder = BASE_PATH / "20250309_constant_app" / sim_folder

n_trials = 100

with open(param_file, "r") as f:
    param_sim = f.readlines()

param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

param_sim = [param_sim[0], param_sim[4]]

df_control_sim = pd.DataFrame(param_sim)
df_control_sim.columns = ["manual_protocol", "antibiotic_value", "nutrient_range"]
df_control_sim["inst_env"] = env_type
df_control_sim["antibiotic_value"] = df_control_sim["antibiotic_value"].astype(float)

final_cell_list = []
final_cell_std_list = []
extinction_frac_list = []
bootstrap_mean_list = []
bootstrap_std_list = []
extinction_rate_list = []
for param in param_sim:
    manual_protocol = param[0]
    antibiotic_value = float(param[1])
    nutrient_range = param[2]

    k_n0_actions = [float(nutr) for nutr in nutrient_range.split('_')]

    folder_name = sim_folder / f"a{antibiotic_value:.2f}_n{nutrient_range}_value_check" / f"{manual_protocol}"
    
    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials)

    out_name = BASE_PATH / "figures_jpg" / "control_nutr" / f"a{antibiotic_value:.2f}_n{nutrient_range}_value_check" / f"{manual_protocol}.jpg"
    # plot_single_varenv(loaded_logger, out_name, color_list, n_trials)
    plot_one_traj(loaded_logger, out_name, color_list, n_trials, warm_up_embed)

    out_name = BASE_PATH / "figures_pdf" / "control_nutr" / f"a{antibiotic_value:.2f}_n{nutrient_range}_value_check" / f"{manual_protocol}.pdf"
    # plot_single_varenv(loaded_logger, out_name, color_list, n_trials)
    plot_one_traj(loaded_logger, out_name, color_list, n_trials, warm_up_embed)

    tcbk_list, _, _, _, cell_array, _ = loaded_logger

    start_1 = 0
    start_1_list = []
    for iii, tcbk in enumerate(tcbk_list):
        if tcbk[3, 0] == 1.0:
            start_1 += 1
            start_1_list.append(iii)
    
    print(start_1)
    print(start_1_list)
            
    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))

    boot_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(extinction, size=len(extinction), replace=True)
        boot_means.append(np.mean(sample))
    
    bootstrap_mean_list.append(np.mean(boot_means))
    bootstrap_std_list.append(np.std(boot_means))

    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))

    final_cell_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean())
    final_cell_std_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std())

df_control_sim["sim_log_cell"] = final_cell_list
df_control_sim["sim_log_cell_std"] = final_cell_std_list
df_control_sim["extinction_frac"] = extinction_frac_list
df_control_sim["extinction_rate"] = extinction_rate_list
df_control_sim["bootstrap_mean"] = bootstrap_mean_list
df_control_sim["bootstrap_std"] = bootstrap_std_list

# %% ----- ----- ----- ----- control env eval ----- ----- ----- ----- %% #
env_type = "controlenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

param_agent = [param_agent[2]]

df_control_eval = pd.DataFrame(param_agent)
df_control_eval.columns = ["antibiotic_value", "nutrient_range", "delay_embed_len", "rep", "b_observation", "k_n0_observation", "training_episode"]

df_control_eval["inst_env"] = env_type
df_control_eval["antibiotic_value"] = df_control_eval["antibiotic_value"].astype(float)
df_control_eval["delay_embed_len"] = df_control_eval["delay_embed_len"].astype(int)
df_control_eval["rep"] = df_control_eval["rep"].astype(int)
df_control_eval["b_observation"] = df_control_eval["b_observation"] == "True"
df_control_eval["k_n0_observation"] = df_control_eval["k_n0_observation"] == "True"

final_cell_list = []
final_cell_std_list = []
extinction_frac_list = []
bootstrap_mean_list = []
bootstrap_std_list = []
extinction_rate_list = []

for param in param_agent:
    antibiotic_value = float(param[0])
    nutrient_range = param[1]
    delay_embed_len = int(param[2])
    rep_run = int(param[3])
    b_observation = param[4] == "True"
    k_n0_observation = param[5] == "True"
    training_episode = param[6]

    trial_name = f"a{antibiotic_value:.2f}_n{nutrient_range}_b{b_observation}_k{k_n0_observation}_delay{delay_embed_len}_rep{rep_run}"
    folder_name = eval_folder / f"{trial_name}" / f"{training_episode}/"
    
    loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval)

    out_name = BASE_PATH / "figures_jpg" / "control_nutr" / trial_name / f"{training_episode}" / "traj.jpg"
    # plot_single_varenv(loaded_logger, out_name, color_list, n_trials_eval)
    plot_one_traj(loaded_logger, out_name, color_list, n_trials_eval, warm_up_embed)

    out_name = BASE_PATH / "figures_pdf" / "control_nutr" / trial_name / f"{training_episode}" / "traj.pdf"
    # plot_single_varenv(loaded_logger, out_name, color_list, n_trials_eval)
    plot_one_traj(loaded_logger, out_name, color_list, n_trials_eval, warm_up_embed)

    tcbk_list, _, _, _, cell_array, _ = loaded_logger
    extinction = [1 if tcbk[1, -1] == 0 else 0 for tcbk in tcbk_list]
    extinction_frac_list.append(np.mean(extinction))
    boot_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(extinction, size=len(extinction), replace=True)
        boot_means.append(np.mean(sample))
    bootstrap_mean_list.append(np.mean(boot_means))
    bootstrap_std_list.append(np.std(boot_means))
    extinction_rate = [i_ext / (tcbk[0, -1] * delta_t) for i_ext, tcbk in zip(extinction, tcbk_list)]
    extinction_rate_list.append(np.mean(extinction_rate))
    final_cell_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean())
    final_cell_std_list.append(np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std())

df_control_eval["eval_log_cell"] = final_cell_list
df_control_eval["eval_log_cell_std"] = final_cell_std_list
df_control_eval["extinction_frac"] = extinction_frac_list
df_control_eval["extinction_rate"] = extinction_rate_list
df_control_eval["bootstrap_mean"] = bootstrap_mean_list
df_control_eval["bootstrap_std"] = bootstrap_std_list

# %%
# df_control_eval_select = df_control_eval[(df_control_eval["extinction_frac"] == 1) & (df_control_eval["nutrient_range"] == "1_3")]
# df_control_eval_select = df_control_eval_select.groupby("nutrient_range", group_keys=False).apply(
#     get_best_row_extinct_rate, include_groups=True
# ).reset_index(drop=True)
# df_control_eval_select = df_control_eval_select.iloc[[0,1]]

# %% ----- ----- ----- ----- control env eval plot traj ----- ----- ----- ----- %% #
# for param in df_control_eval_select.itertuples(index=False):
#     antibiotic_value = param[0]
#     nutrient_range = param[1]
#     delay_embed_len = int(param[2])
#     rep_run = param[3]
#     b_observation = param[4]
#     k_n0_observation = param[5]
#     training_episode = param[6]

#     trial_name = f"a{antibiotic_value:.2f}_n{nutrient_range}_b{b_observation}_k{k_n0_observation}_delay{delay_embed_len}_rep{rep_run}"
#     folder_name = eval_folder / f"{trial_name}" / f"{training_episode}/"
    
#     loaded_logger = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval)

#     out_name = BASE_PATH / "figures_jpg" / "control_nutr" / trial_name / f"{training_episode}" / "traj.jpg"
#     plot_single_varenv(loaded_logger, out_name, color_list, n_trials_eval)
#     plot_one_traj(loaded_logger, out_name, color_list, n_trials_eval, warm_up_embed)

# %%
plt.rcParams.update({"font.size": 14})
fig = plt.figure()

condition_labels = ['Feast', 'Famine', 'Learned Policy']
plt.bar(
    condition_labels[0],
    df_control_sim.loc[df_control_sim["manual_protocol"] == condition_labels[0], "bootstrap_mean"].item(),
    yerr=df_control_sim.loc[df_control_sim["manual_protocol"] == condition_labels[0], "bootstrap_std"].item(),
    capsize=5, color='slateblue', edgecolor='slateblue',
)
plt.bar(
    condition_labels[1],
    df_control_sim.loc[df_control_sim["manual_protocol"] == condition_labels[1], "bootstrap_mean"].item(),
    yerr=df_control_sim.loc[df_control_sim["manual_protocol"] == condition_labels[1], "bootstrap_std"].item(),
    capsize=5, color='purple', edgecolor='purple',
)
plt.bar(
    condition_labels[2],
    df_control_eval["bootstrap_mean"],
    yerr=df_control_eval["bootstrap_std"],
    capsize=5, color='seagreen', edgecolor='seagreen',
)
plt.ylabel('Extinction Fraction')
plt.show()

fig.savefig('/home/zihangw/BacteriaAdaptation/figures_jpg/control_env_comparison.jpg', dpi=300, bbox_inches='tight')
fig.savefig('/home/zihangw/BacteriaAdaptation/figures_pdf/control_env_comparison.pdf', dpi=300, bbox_inches='tight')

# %%
# fff_name = "/home/zihangw/BacteriaAdaptation/plot/run_one_control/results_sim_controlenv/a3.72_n1_3_value_check/Famine/"
# start_1 = 0
# start_1_list = []
# for iii, file in enumerate(os.listdir(fff_name)):
#     with open(os.path.join(fff_name, file), "rb") as f:
#         info = pickle.load(f)
#     tkbc = np.array(info["log"])
#     tcbk = tkbc[:,[0,3,2,1]].T

#     if tcbk[3, 0] == 1.0:
#         start_1 += 1
#         start_1_list.append(file)

# print(start_1)
# print(start_1_list)
# print(iii)

# %%
