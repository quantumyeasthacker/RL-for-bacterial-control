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
# BASE_PATH = Path("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/")
BASE_PATH = Path("/mnt/d/bacterial_adaptation/working_folder")

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
def detect_low_high_periods(signal, low_value=0.0, high_value=3.72):
    periods = []
    i = 0
    n = len(signal)

    while i < n:
        # Detect low_value run
        if signal[i] == low_value:
            start = i
            while i < n and signal[i] == low_value:
                i += 1
            # Transition to high_value run
            if i < n and signal[i] == high_value:
                transition = i
                while i < n and signal[i] == high_value:
                    i += 1
                end = i - 1
                periods.append((start, transition, end))
            else:
                i += 1
        else:
            i += 1

    return periods

# %%
env_type = "varenv"
param_file = BASE_PATH / "param_space" / f"param_agent_delay_30_{env_type}_eval.txt"
eval_folder = BASE_PATH / f"results_delay_30_record_{env_type}_eval"
# eval_folder = BASE_PATH / "20250325_varenv" / eval_folder

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

num_of_pies = 4
for param in param_agent:
    if param[1] != "12":
        continue

    antibiotic_value = float(param[0])
    training_episode = param[4]
    folder_name = eval_folder / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}" / training_episode
    tcbk_list, _, _, _, cell_array, _ = load_logger_data_new(folder_name, sim_length, max_pop, n_trials_eval, True)

    np.random.seed(1)
    choices = np.random.choice(100, 3)
    fig2, ax2 = plt.subplots()
    ax2_xticks = set()

    for i in choices:
        tcbkrs = tcbk_list[i]
        t = tcbkrs[0]
        cell = tcbkrs[1]
        drug = tcbkrs[2]
        phiR = tcbkrs[4]
        phiS = tcbkrs[5]
        phiP = phiR_max - phiR - phiS
        phiQ = 1 - phiR_max

        phiD = 0.1 * phiR
        phiR -= phiD

        prds = detect_low_high_periods(drug)
        if len(prds) == 0:
            t_choices = np.linspace(warm_up_embed, tcbkrs.shape[1]-2, num_of_pies, dtype=int)
        else:
            prd = prds[-1]
            start, transition, end = prd
            t_choices = np.linspace(start, transition, num_of_pies - 1, dtype=int)
            t_choices = np.append(t_choices, end - 1)
            # t_choices = np.clip(t_choices, warm_up_embed, tcbkrs.shape[1]-2)

        fig, axs = plt.subplots(1, num_of_pies, figsize=(4*num_of_pies, 4))
        for j, ax in enumerate(axs):
            t_point = t_choices[j]
            data_point = np.array([phiS[t_point], phiR[t_point], phiD[t_point], phiP[t_point], phiQ])
            ax.pie(
                data_point,
                labels = labels,
                colors = color_list,
                autopct='%1.1f%%',
                textprops={'fontsize': 20}
            )
            ax.set_title(f"t = {t[t_point]:.1f} h", fontsize=20)

            ax2_xticks.add(t[t_point])
            # ax2.plot([t[t_point], t[t_point]], [0, 1], color="black", linestyle="--", linewidth=1)
            ax2.axvline(x=t[t_point], color="black", linestyle="--", linewidth=1)
        ax2.plot(t, cell)

        fig.tight_layout()

        out_name = BASE_PATH / "figures_pdf" / "var_nutrient" / "eval_pie_2" / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}/{i}.pdf"
        out_path = os.path.dirname(out_name)
        os.makedirs(out_path, exist_ok=True)

        fig.savefig(
            out_name,
            dpi = 300,
            bbox_inches='tight',
        )

        out_name = BASE_PATH / "figures_jpg" / "var_nutrient" / "eval_pie_2" / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}/{i}.jpg"
        out_path = os.path.dirname(out_name)
        os.makedirs(out_path, exist_ok=True)

        fig.savefig(
            out_name,
            dpi = 300,
            bbox_inches='tight',
        )
        plt.close(fig)

    ax2_xticks = np.round(sorted(ax2_xticks), 1)
    
    ax2.set_xticks(ax2_xticks)
    ax2.tick_params(axis='x', labelrotation=45)
    ax2.set_yscale('log')

    fig2.savefig(
        BASE_PATH / "figures_pdf" / "var_nutrient" / "eval_pie_2" / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}/xticks.pdf",
        dpi = 300,
        bbox_inches='tight'
    )
    fig2.savefig(
        BASE_PATH / "figures_jpg" / "var_nutrient" / "eval_pie_2" / f"a{param[0]}_T{param[1]}_delay{param[2]}_rep{param[3]}/xticks.jpg",
        dpi = 300,
        bbox_inches='tight'
    )
    plt.close(fig2)

# %%
