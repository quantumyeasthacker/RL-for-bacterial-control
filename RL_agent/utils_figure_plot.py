import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from typing import Any, ClassVar, Optional, TypeVar, Union
import wandb


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_trajectory(info_list: Union[dict, list[dict]], episode: int, folder_name) -> None:
    os.makedirs(folder_name, exist_ok=True)
    if isinstance(info_list, dict):
        info_list = [info_list]

    figure, ax = plt.subplots(3,1)
    figure.subplots_adjust(hspace=.0)
    for info in info_list:
        delta_t = info['delta_t']
        warm_up = info['warm_up']
        delay_embed_len = info['delay_embed_len']
        t, k, b, c = list(zip(*info['log']))[:4]

        ax[0].plot(t, b)
        ax[1].plot(t, k)
        ax[2].plot(t, c)

    ax[0].set_ylabel('antibiotic conc.')
    ax[0].axvline(x=delta_t*(warm_up+delay_embed_len), linestyle='--', color='k')
    ax[1].set_ylabel('nutrient conc.')
    ax[1].axvline(x=delta_t*(warm_up+delay_embed_len), linestyle='--', color='k')
    ax[2].set_ylabel('population size')
    ax[2].axvline(x=delta_t*(warm_up+delay_embed_len), linestyle='--', color='k')
    ax[2].set_xlabel('Time (h)')
    ax[2].set_yscale('log')
    image_path = os.path.join(folder_name, f"{episode}.jpg")
    figure.savefig(image_path, dpi=300, bbox_inches='tight')
    wandb.log({"plot_trajectories": wandb.Image(image_path)})
    plt.close()
        


def plot_reward_Q_loss(reward, std, grad_update_num, loss, folder, Q1, Q2, Q1_target, Q2_target):
    # note, since total rewards are not discounted, should be different than Q value
    figure, ax = plt.subplots(3,1)
    figure.subplots_adjust(hspace=.0)
    ax[0].plot(grad_update_num, reward, color='k')
    lb = np.array(reward) - np.array(std)
    ub = np.array(reward) + np.array(std)
    ax[0].fill_between(grad_update_num, lb, ub, color='k', alpha=0.4)
    ax[0].set_ylabel('ave. total eval reward')

    ax[1].plot(grad_update_num, Q1, color='rebeccapurple', label='min Q1')
    ax[1].plot(grad_update_num, Q2, color='#2ca02c', label='min Q2')
    ax[1].plot(grad_update_num, Q1_target, color='#d62728', label='min target Q1')
    ax[1].plot(grad_update_num, Q2_target, color='#1f77b4', label='min target Q2')
    ax[1].set_ylabel('ave. eval Q value')
    ax[1].legend()

    ax[2].plot(range(1,len(loss)+1), loss)
    ax[2].set_ylabel('loss')
    ax[2].legend(['loss 1', 'loss 2'])

    plt.xlabel('num. gradient updates')
    figure.savefig(os.path.join(folder,'reward_Q_loss.jpg'), dpi=300, bbox_inches='tight')
    # wandb.log({"plot_reward_Q_loss": wandb.Image(folder+'/reward_Q_loss.jpg')})
    plt.close()