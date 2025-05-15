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

from utils import expand_and_fill, estimate_frequency_fft, down_edge_detection, load_logger_data_new, cross_correlation
from pathlib import Path

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EPS = 1e-6

BASE_PATH = Path("/mnt/d/bacterial_adaptation/working_folder")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %%
env_type = "varenv"
T_k_n0 = 12
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval_T{T_k_n0}.txt"
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

ave_max_cross_corr_kn0_list = []
ave_corr_lag_kn0_list = []
extinct_fraction_list = []
ave_ext_time_list = []
ave_reward_list = []

for param in param_agent:
    antibiotic_value = float(param[0])
    training_episode = param[4]
    folder_name = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, False)
    
    max_cross_corr_kn0, lag_kn0 = [], []
    extinct_times, extinct_count = [], 0
    sum_rewards_all = []
    for i_trial in range(n_trials_eval):
        tcbk = tcbk_list[i_trial]
        time, cell_count, drug, nutr = tcbk

        drug = drug[warm_up_embed:]
        nutr = nutr[(warm_up_embed-30):] # embedding len
        # nutr = nutr[(warm_up_embed):] # embedding len
        if np.std(drug) != 0 and np.std(nutr) != 0:
            corr, lag = cross_correlation(drug, nutr)
        
        max_cross_corr_kn0.append(corr)
        lag_kn0.append(lag)

        # check extinction
        # if len(time) != sim_length:
        if cell_count[-1] == 0:
            extinct_times.append(time[-1])
            extinct_count += 1
            cell_count[-1] = 1e-5
        
        sum_rewards_all.append(np.log(cell_count[-1] / cell_count[warm_up_embed-1]))
    
    ave_max_cross_corr_kn0 = sum(max_cross_corr_kn0)/len(max_cross_corr_kn0) if len(max_cross_corr_kn0) > 0 else 0
    ave_corr_lag_kn0 = sum(lag_kn0)/len(lag_kn0) if len(lag_kn0) > 0 else 0
    extinct_fraction = extinct_count / n_trials_eval
    ave_ext_time = sum(extinct_times) / extinct_count if extinct_count > 0 else np.inf
    ave_reward = np.mean(sum_rewards_all)

    ave_max_cross_corr_kn0_list.append(ave_max_cross_corr_kn0)
    ave_corr_lag_kn0_list.append(ave_corr_lag_kn0)
    extinct_fraction_list.append(extinct_fraction)
    ave_ext_time_list.append(ave_ext_time)
    ave_reward_list.append(ave_reward)

df_varenv_eval["ave_max_cross_corr_kn0"] = ave_max_cross_corr_kn0_list
df_varenv_eval["ave_corr_lag_kn0"] = ave_corr_lag_kn0_list
df_varenv_eval["extinct_fraction"] = extinct_fraction_list
df_varenv_eval["ave_ext_time"] = ave_ext_time_list
df_varenv_eval["ave_reward"] = ave_reward_list

# %%
x = np.unique(df_varenv_eval["training_episode_int"].to_numpy())

y1_stack = df_varenv_eval.pivot(index="rep", columns="training_episode_int", values="ave_reward").to_numpy()
y2_stack = df_varenv_eval.pivot(index="rep", columns="training_episode_int", values="ave_max_cross_corr_kn0").to_numpy()
y3_stack = df_varenv_eval.pivot(index="rep", columns="training_episode_int", values="ave_corr_lag_kn0").to_numpy()
y4_stack = df_varenv_eval.pivot(index="rep", columns="training_episode_int", values="extinct_fraction").to_numpy()

y1_mean = np.mean(y1_stack, axis=0)
y2_mean = np.mean(y2_stack, axis=0)
y3_mean = np.mean(y3_stack, axis=0)
y4_mean = np.mean(y4_stack, axis=0)

y1_std = np.std(y1_stack, axis=0)
y2_std = np.std(y2_stack, axis=0)
y3_std = np.std(y3_stack, axis=0)
y4_std = np.std(y4_stack, axis=0)

fig = plt.figure(figsize=(8, 6))
host = host_subplot(111, axes_class=axisartist.Axes, figure=fig)

par1 = host.twinx()
# par2 = host.twinx()

# par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))
par1.axis["right"].toggle(all=True)
# par2.axis["right"].toggle(all=True)

# Plot the lines
p1, = host.plot(x, y1_mean, label="Cost", color='tab:blue')
p2, = par1.plot(x, y2_mean, label="Correlation", color='tab:red')
# p3, = par2.plot(x, y3_mean, label="Lag", color='tab:green')

host.fill_between(x, y1_mean - y1_std, y1_mean + y1_std, color='tab:blue', alpha=0.2)
par1.fill_between(x, y2_mean - y2_std, y2_mean + y2_std, color='tab:red', alpha=0.2)
# par2.fill_between(x, y3_mean - y3_std, y3_mean + y3_std, color='tab:green', alpha=0.2)

# Set labels
host.set(xlabel = "Training Episode (_step)", ylabel = "Average Total Cost")
par1.set(ylabel = "Correlation of nutrient and antibiotic conc.")
# par2.set(ylabel = "Lag of nutrient and antibiotic conc.")

host.legend(loc='upper right')

# Set colors
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
# par2.axis["right"].label.set_color(p3.get_color())

# Title and show
# fig.suptitle("Average Total Reward and Correlation and Lag of Nutrient and Antibiotic conc. vs. Training Episode")
# fig.tight_layout()
fig.savefig(BASE_PATH / "figures_jpg" / f"mean_std_T{T_k_n0}_no_truncate_nutr.jpg", dpi=600, bbox_inches='tight')
fig.savefig(BASE_PATH / "figures_pdf" / f"mean_std_T{T_k_n0}_no_truncate_nutr.pdf", dpi=600, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y3_mean, label="Lag", color='tab:green')
ax.fill_between(x, y3_mean - y3_std, y3_mean + y3_std, color='tab:green', alpha=0.2)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.set(xlabel = "Training Episode (_step)", ylabel = "Lag of nutrient and antibiotic conc.")
ax.legend(loc='upper right')
# ax.set_title("Lag of Nutrient and Antibiotic Concentration vs. Training Episode")
fig.savefig(BASE_PATH / "figures_jpg" / f"mean_std_lag_T{T_k_n0}_no_truncate_nutr.jpg", dpi=600, bbox_inches='tight')
fig.savefig(BASE_PATH / "figures_pdf" / f"mean_std_lag_T{T_k_n0}_no_truncate_nutr.pdf", dpi=600, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y4_mean, label="Extinct Fraction", color='tab:orange')
ax.fill_between(x, y4_mean - y4_std, y4_mean + y4_std, color='tab:orange', alpha=0.2)
ax.set(xlabel = "Training Episode (_step)", ylabel = "Extinct Fraction")
ax.legend(loc='upper right')
# ax.set_title("Extinct Fraction vs. Training Episode")
fig.savefig(BASE_PATH / "figures_jpg" / f"mean_std_extinct_fraction_T{T_k_n0}_no_truncate_nutr.jpg", dpi=600, bbox_inches='tight')
fig.savefig(BASE_PATH / "figures_pdf" / f"mean_std_extinct_fraction_T{T_k_n0}_no_truncate_nutr.pdf", dpi=600, bbox_inches='tight')
plt.close(fig)

# %%
