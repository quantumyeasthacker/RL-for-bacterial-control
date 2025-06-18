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
# default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# COLOR_LIST = ["#dec60c", "#a7c82f", "#548c6a"]
BASE_PATH = Path("/Users/josiahk/Library/CloudStorage/Box-Box/Zihang_data")

# %%
delta_t = 0.2
init_len = 1
warm_up_embed = 60 + init_len
num_decisions = 300
phiR_max: float = 0.55
phiS_max: float = 0.33

sim_length = num_decisions + warm_up_embed
max_pop: int = int(1e11)

# %%
def detect_drug_switch(signal, low_value=0.0, high_value=3.72):
    
    n = len(signal)
    low_to_high = np.zeros(n)
    high_to_low = np.zeros(n)
    i = 1

    for i in range(n):
        # Detect low_value run
        if signal[i-1] == low_value and signal[i] == high_value:
            low_to_high[i] = 1
        
        if signal[i-1] == high_value and signal[i] == low_value:
            high_to_low[i] = 1

    return low_to_high, high_to_low

# %%
env_type = "varenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / "agents_eval" / f"results_delay_30_record_{env_type}_eval"

n_trials_eval = 100

with open(param_file, "r") as f:
    param_agent = f.readlines()

param_agent = [x.strip() for x in param_agent]
param_agent = [x.split(" ") for x in param_agent]

# %%
# labels = [r"$\phi_R$", r"$\phi_S$", r"$\phi_P$", r"$\phi_Q$"]
labels = [r"$\phi_S$", r"$\phi_R$", r"$\phi_D$", r"$\phi_P$", r"$\phi_Q$"]
# color_list = ["#56e2cf", "#56aee2", "#5668e2", "#8a56e2", "#cf56e2"]
color_list = ["#FE938C", "#E6B89C", "#EAD2AC", "#9CAFB7", "#4281A4"]
# color_list = color_list[1:]

num_bins = 10

# param = param_agent[0]
# parent_folder = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}"
# training_episode_folders = [f.name for f in parent_folder.iterdir() if f.is_dir()]

# %% Probability of switching event occuring, conditional on phi value
# i=0

# training_episode_folders = ['episode_0', 'episode_399']

# for training_episode in training_episode_folders:
#     t = []
#     cell = []
#     drug = []
#     phiR = []
#     phiS = []
#     low_to_high_ind = []
#     high_to_low_ind = []

#     # if i == 1:
#     #     print('stop')
#     #     continue

#     # cycle through each rep
#     for param in param_agent:
#         if param[1] != "12": # for now only considering T=12
#             continue

#         antibiotic_value = float(param[0])

#         folder_name = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
#         tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, True)

#         # cycle through each evaluation
#         for tcbkrs in tcbk_list:
#             b = tcbkrs[2][warm_up_embed:]
#             low_to_high, high_to_low = detect_drug_switch(b)
#             low_to_high_ind.extend(low_to_high.tolist())
#             high_to_low_ind.extend(high_to_low.tolist())

#             t.extend(tcbkrs[0][warm_up_embed:].tolist())
#             cell.extend(tcbkrs[1][warm_up_embed:].tolist())
#             drug.extend(b.tolist())
#             phiR.extend(tcbkrs[4][warm_up_embed:].tolist())
#             phiS.extend(tcbkrs[5][warm_up_embed:].tolist())

#     df = pd.DataFrame({
#         't': t,
#         'pop size': cell,
#         'drug': drug,
#         'phiR': phiR,
#         'phiS': phiS,
#         'low to high':low_to_high_ind,
#         'high to low': high_to_low_ind})


#     df_phiS_sorted = df.sort_values(by='phiS')
#     df_phiS_sorted['bins'] = pd.qcut(df_phiS_sorted['phiS'], num_bins, labels=False)
#     # df_phiS_sorted['bins'] = pd.cut(df_phiS_sorted['phiS'], num_bins, labels=False)
#     binned_h2l_prob = df_phiS_sorted.groupby('bins')['high to low'].mean()
#     binned_l2h_prob = df_phiS_sorted.groupby('bins')['low to high'].mean()
#     binned_phiS = df_phiS_sorted.groupby('bins')['phiS'].mean()

#     plt.plot(binned_phiS,binned_h2l_prob, label='High to low')
#     plt.plot(binned_phiS,binned_l2h_prob, label='Low to high')
#     plt.xlabel('Stress Sector Proteome Fraction, $\phi_S$')
#     plt.ylabel('Probability of Switching')
#     plt.legend()
#     plt.show()
    
#     # i = 1


# %% Frequency of switching, conditional on switching event occuring
# i=0

training_episode_folders = ['episode_0', 'episode_399']

for training_episode in training_episode_folders:
    t = []
    cell = []
    drug = []
    phiR = []
    phiS = []
    low_to_high_ind = []
    high_to_low_ind = []

    # if i == 1:
    #     print('stop')
    #     continue

    # cycle through each rep
    for param in param_agent:
        if param[1] != "12": # for now only considering T=12
            continue

        antibiotic_value = float(param[0])

        folder_name = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
        tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, True)

        # cycle through each evaluation
        for tcbkrs in tcbk_list:
            b = tcbkrs[2][warm_up_embed:]
            low_to_high, high_to_low = detect_drug_switch(b)
            low_to_high_ind.extend(low_to_high.tolist())
            high_to_low_ind.extend(high_to_low.tolist())

            t.extend(tcbkrs[0][warm_up_embed:].tolist())
            cell.extend(tcbkrs[1][warm_up_embed:].tolist())
            drug.extend(b.tolist())
            phiR.extend(tcbkrs[4][warm_up_embed:].tolist())
            phiS.extend(tcbkrs[5][warm_up_embed:].tolist())

    df = pd.DataFrame({
        't': t,
        'pop size': cell,
        'drug': drug,
        'phiR': phiR,
        'phiS': phiS,
        'low to high':low_to_high_ind,
        'high to low': high_to_low_ind})


    df_phiS_sorted = df.sort_values(by='phiS')
    df_phiS_sorted['bins'] = pd.qcut(df_phiS_sorted['phiS'], num_bins, labels=False)
    # df_phiS_sorted['bins'] = pd.cut(df_phiS_sorted['phiS'], num_bins, labels=False)
    binned_h2l_prob = df_phiS_sorted.groupby('bins')['high to low'].sum() / df_phiS_sorted['high to low'].sum()
    binned_l2h_prob = df_phiS_sorted.groupby('bins')['low to high'].sum() / df_phiS_sorted['low to high'].sum()
    binned_phiS = df_phiS_sorted.groupby('bins')['phiS'].mean()

    fig = plt.figure()
    plt.plot(binned_phiS,binned_h2l_prob, label='High to low')
    plt.plot(binned_phiS,binned_l2h_prob, label='Low to high')
    plt.xlabel('Stress Sector Proteome Fraction, $\phi_S$')
    plt.ylabel('Frequency of Switching')
    plt.legend()
    plt.show()

    out_name = BASE_PATH / "figures_pdf" / (training_episode + "_phiS.pdf")
    out_path = os.path.dirname(out_name)
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(
        out_name,
        dpi = 300,
        bbox_inches='tight'
    )


# %% Probability of switching, conditional on switching event occuring, phiR

training_episode_folders = ['episode_0', 'episode_399']

for training_episode in training_episode_folders:
    t = []
    cell = []
    drug = []
    phiR = []
    phiS = []
    low_to_high_ind = []
    high_to_low_ind = []

    # cycle through each rep
    for param in param_agent:
        if param[1] != "12": # for now only considering T=12
            continue

        antibiotic_value = float(param[0])

        folder_name = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
        tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, True)

        # cycle through each evaluation
        for tcbkrs in tcbk_list:
            b = tcbkrs[2][warm_up_embed:]
            low_to_high, high_to_low = detect_drug_switch(b)
            low_to_high_ind.extend(low_to_high.tolist())
            high_to_low_ind.extend(high_to_low.tolist())

            t.extend(tcbkrs[0][warm_up_embed:].tolist())
            cell.extend(tcbkrs[1][warm_up_embed:].tolist())
            drug.extend(b.tolist())
            phiR.extend(tcbkrs[4][warm_up_embed:].tolist())
            phiS.extend(tcbkrs[5][warm_up_embed:].tolist())

    df = pd.DataFrame({
        't': t,
        'pop size': cell,
        'drug': drug,
        'phiR': phiR,
        'phiS': phiS,
        'low to high':low_to_high_ind,
        'high to low': high_to_low_ind})


    df_phi_sorted = df.sort_values(by='phiR')
    df_phi_sorted['bins'] = pd.qcut(df_phi_sorted['phiR'], num_bins, labels=False)
    binned_h2l_prob = df_phi_sorted.groupby('bins')['high to low'].sum() / df_phi_sorted['high to low'].sum()
    binned_l2h_prob = df_phi_sorted.groupby('bins')['low to high'].sum() / df_phi_sorted['low to high'].sum()
    binned_phiS = df_phi_sorted.groupby('bins')['phiR'].mean()

    fig = plt.figure()
    plt.plot(binned_phiS,binned_h2l_prob, label='High to low')
    plt.plot(binned_phiS,binned_l2h_prob, label='Low to high')
    plt.xlabel('Ribosome Sector Proteome Fraction, $\phi_R$')
    plt.ylabel('Frequency of Switching')
    plt.legend()
    plt.show()

    out_name = BASE_PATH / "figures_pdf" / (training_episode + "_phiR.pdf")
    out_path = os.path.dirname(out_name)
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(
        out_name,
        dpi = 300,
        bbox_inches='tight'
    )


# %%