# %%
import wandb
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
from scipy.interpolate import make_interp_spline
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


# BASE_PATH = Path("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/20250407/figures")
BASE_PATH = Path("/home/zihangw/BacteriaAdaptation/figures")

api = wandb.Api()
entity = "zihangwen-carnegie-mellon-university"

# %%

env_type = "varenv"
delay = 30

smoothing_level = 0.2

project = f"antibioticRL-zihang-{env_type}-nutrient-delay-{delay}"
# fig_path = BASE_PATH / f"wandb_{env_type}_delay_{delay}_smooth"
# os.makedirs(fig_path, exist_ok=True)

T_k_n0_list = [6, 12, 18, 24]

# %%
for T_k_n0 in T_k_n0_list:
    x_list = []
    y1_list = []
    y2_list = []
    y3_list = []
    y4_list = []

    for run in api.runs(f"{entity}/{project}"):
        # run = api.runs(f"{entity}/{project}")[0]
        print("Fetching details for run: ", run.id, run.name)
        if run.state != "finished":
            continue
        if run.config["T_k_n0"] != T_k_n0:
            continue
        df = run.history()
        filtered_df = df.dropna(subset=['ave total reward', 'ave_max_cross_corr_kn0'])

        x = filtered_df['_step'].values
        y1 = filtered_df['ave total reward'].values
        y2 = filtered_df['ave_max_cross_corr_kn0'].values
        y3 = filtered_df['ave_corr_lag_kn0'].values
        y4 = filtered_df['extinct_fraction'].values

        x_list.append(x)
        y1_list.append(y1)
        y2_list.append(y2)
        y3_list.append(y3)
        y4_list.append(y4)

    y1_stack = np.stack(y1_list)
    y2_stack = np.stack(y2_list)
    y3_stack = np.stack(y3_list)
    y4_stack = np.stack(y4_list)

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
    p1, = host.plot(x, y1_mean, label="Reward", color='tab:blue')
    p2, = par1.plot(x, y2_mean, label="Correlation", color='tab:red')
    # p3, = par2.plot(x, y3_mean, label="Lag", color='tab:green')

    host.fill_between(x, y1_mean - y1_std, y1_mean + y1_std, color='tab:blue', alpha=0.2)
    par1.fill_between(x, y2_mean - y2_std, y2_mean + y2_std, color='tab:red', alpha=0.2)
    # par2.fill_between(x, y3_mean - y3_std, y3_mean + y3_std, color='tab:green', alpha=0.2)

    # Set labels
    host.set(xlabel = "Training Episode (_step)", ylabel = "Average Total Reward")
    par1.set(ylabel = "Correlation of nutrient and antibiotic conc.")
    # par2.set(ylabel = "Lag of nutrient and antibiotic conc.")

    host.legend(loc='upper right')

    # Set colors
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    # par2.axis["right"].label.set_color(p3.get_color())

    # Title and show
    fig.suptitle("Average Total Reward and Correlation and Lag of Nutrient and Antibiotic conc. vs. Training Episode")
    # fig.tight_layout()
    fig.savefig(BASE_PATH / f"mean_std_T{T_k_n0}.jpg", dpi=600, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y3_mean, label="Lag", color='tab:green')
    ax.fill_between(x, y3_mean - y3_std, y3_mean + y3_std, color='tab:green', alpha=0.2)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set(xlabel = "Training Episode (_step)", ylabel = "Lag of nutrient and antibiotic conc.")
    ax.legend(loc='upper right')
    ax.set_title("Lag of Nutrient and Antibiotic Concentration vs. Training Episode")
    fig.savefig(BASE_PATH / f"mean_std_lag_T{T_k_n0}.jpg", dpi=600, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y4_mean, label="Extinct Fraction", color='tab:orange')
    ax.fill_between(x, y4_mean - y4_std, y4_mean + y4_std, color='tab:orange', alpha=0.2)
    ax.set(xlabel = "Training Episode (_step)", ylabel = "Extinct Fraction")
    ax.legend(loc='upper right')
    ax.set_title("Extinct Fraction vs. Training Episode")
    fig.savefig(BASE_PATH / f"mean_std_extinct_fraction_T{T_k_n0}.jpg", dpi=600, bbox_inches='tight')
    plt.close(fig)

# %%
# for run in api.runs(f"{entity}/{project}"):
#     # run = api.runs(f"{entity}/{project}")[0]
#     print("Fetching details for run: ", run.id, run.name)
#     if run.state != "finished":
#         continue
#     df = run.history()
#     filtered_df = df.dropna(subset=['ave total reward', 'ave_max_cross_corr_kn0'])

#     x = filtered_df['_step'].values
#     y1 = filtered_df['ave total reward'].values
#     y2 = filtered_df['ave_max_cross_corr_kn0'].values
#     y3 = filtered_df['ave_corr_lag_kn0'].values

#     y1_smoothed = lowess(y1, x, frac=smoothing_level)
#     y2_smoothed = lowess(y2, x, frac=smoothing_level)
#     y3_smoothed = lowess(y3, x, frac=smoothing_level)

#     fig = plt.figure(figsize=(10, 6))

#     host = host_subplot(111, axes_class=axisartist.Axes, figure=fig)

#     # Create the parasite axis (second y-axis)
#     par1 = host.twinx()
#     par2 = host.twinx()

#     par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))
#     par1.axis["right"].toggle(all=True)
#     par2.axis["right"].toggle(all=True)

#     host.scatter(x, y1, color='tab:blue', s=10, alpha=0.6)
#     par1.scatter(x, y2, color='tab:red', s=10, alpha=0.6)
#     par2.scatter(x, y3, color='tab:green', s=10, alpha=0.6)

#     # Plot the lines
#     p1, = host.plot(y1_smoothed[:, 0], y1_smoothed[:, 1], label="Reward", color='tab:blue')
#     p2, = par1.plot(y2_smoothed[:, 0], y2_smoothed[:, 1], label="Correlation", color='tab:red')
#     p3, = par2.plot(y3_smoothed[:, 0], y3_smoothed[:, 1], label="Lag", color='tab:green')

#     # Set labels
#     host.set(xlabel = "Training Episode (_step)", ylabel = "Average Total Reward")
#     par1.set(ylabel = "Correlation of nutrient and antibiotic conc.")
#     par2.set(ylabel = "Lag of nutrient and antibiotic conc.")

#     host.legend(loc='upper right')

#     # Set colors
#     host.axis["left"].label.set_color(p1.get_color())
#     par1.axis["right"].label.set_color(p2.get_color())
#     par2.axis["right"].label.set_color(p3.get_color())

#     # Title and show
#     fig.suptitle("Average Total Reward and Correlation and Lag of Nutrient and Antibiotic conc. vs. Training Episode")
#     # fig.tight_layout()
#     fig.savefig(fig_path / f"{run.name}.png", dpi=600, bbox_inches='tight')
#     plt.close(fig)


# %%
# run_data = {
#     "id": run.id,
#     "name": run.name,
#     "url": run.url,
#     "state": run.state,
#     "tags": run.tags,
#     "config": run.config,
#     "created_at": run.created_at,
#     "system_metrics": run.system_metrics,
#     "summary": run.summary,
#     "project": run.project,
#     "entity": run.entity,
#     "user": run.user,
#     "path": run.path,
#     "notes": run.notes,
#     "read_only": run.read_only,
#     "history_keys": run.history_keys,
#     "metadata": run.metadata,
# }
