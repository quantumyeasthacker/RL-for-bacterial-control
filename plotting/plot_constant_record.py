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

from utils import expand_and_fill, estimate_frequency_fft, down_edge_detection, load_logger_data_new
from pathlib import Path


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EPS = 1e-6
default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# BASE_PATH = Path("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/")
BASE_PATH = Path("/home/zihangw/BacteriaAdaptation")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %% ----- ----- ----- ----- constant sim ----- ----- ----- ----- %% #
env_type = "constenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval_0.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]
# param_agent = param_agent[9:] ###### temp

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
freq_std_list = []
for param in param_agent:
    antibiotic_value = float(param[0])
    training_episode = param[4]
    folder_name = f"{eval_folder}/a{param[0]}_n{param[1]}_delay{param[2]}_rep{param[3]}/{training_episode}"
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
        freq_std_list += [0]
    else:
        freq_list += [np.mean(freq_param_list)]
        freq_std_list += [np.std(freq_param_list)]
    
    eval_cell_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).mean()]
    eval_cell_std_list += [np.log10(cell_array[:, warm_up_embed:].mean(axis=1)).std()]

df_constant_eval["eval_log_cell"] = eval_cell_list
df_constant_eval["eval_log_cell_std"] = eval_cell_std_list
# df_constant_eval["eval_log_cell"] = np.log10(df_constant_eval["eval_final_cell"])
df_constant_eval["eval_freq"] = freq_list
df_constant_eval["eval_freq_std"] = freq_std_list

# %%
sum_df = df_constant_eval.groupby(["inst_combination", "rep"])["eval_log_cell"].sum().reset_index()
min_rep = sum_df.loc[sum_df.groupby(["inst_combination"])["eval_log_cell"].idxmin()]
if isinstance(min_rep, pd.Series):
    min_rep = min_rep.to_frame().T

df_constant_eval_app = df_constant_eval.merge(min_rep[['inst_combination', 'rep']], on=['inst_combination', 'rep'])

# %%
nutrient_value_selected = 2.0
df_constant_eval_app_selected = df_constant_eval_app[df_constant_eval_app["nutrient_value"] == 2.0]

# %%
df_plot = df_constant_eval_app_selected[["training_episode_int", "eval_log_cell", "eval_log_cell_std", "eval_freq", "eval_freq_std"]].copy().sort_values("training_episode_int")
x = df_plot["training_episode_int"].values
y1_mean = df_plot["eval_log_cell"].values
y1_std = df_plot["eval_log_cell_std"].values
y2 = df_plot["eval_freq"].values
y2_std = df_plot["eval_freq_std"].values

fig = plt.figure(figsize=(8, 6))

host = host_subplot(111, axes_class=axisartist.Axes, figure=fig)
par1 = host.twinx()

par1.axis["right"].toggle(all=True)

p1, = host.plot(x, y1_mean, label="Average Population Size", color='tab:blue')

p2, = par1.plot(x, y2, label="Pulsing Frequency", color='tab:red')

host.fill_between(x, y1_mean - y1_std, y1_mean + y1_std, color='tab:blue', alpha=0.2)
par1.fill_between(x, y2 - y2_std, y2 + y2_std, color='tab:red', alpha=0.2)

host.set(xlabel = "Training Episode", ylabel = "Average Population Size")
par1.set(ylabel = "Pulsing Frequency")

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

fig.savefig(BASE_PATH / f"figures_pdf" / "pop_size_pulsing_freq_err_area.pdf", dpi=600, bbox_inches='tight')
fig.savefig(BASE_PATH / f"figures_jpg" / "pop_size_pulsing_freq_err_area.jpg", dpi=600, bbox_inches='tight')

# plt.close(fig)

# %%
