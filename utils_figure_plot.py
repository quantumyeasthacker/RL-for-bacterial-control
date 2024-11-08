import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


class DynamicUpdate():
    """Plot evaluation trajectories
    """
    def __init__(self, delta_t, embed_len):
        self.folder_name = "./Fluc_nutrient"
        os.system("mkdir " + self.folder_name)
        self.folder_name_test = self.folder_name + "/Eval"
        self.delta_t = delta_t
        self.embed_len = embed_len

    def __call__(self, episode, time, cell_count, kn0_all, b_all):
        figure, ax = plt.subplots(3,1)
        figure.subplots_adjust(hspace=.0)
        for i in range(np.shape(time)[1]):
            if np.all(cell_count[:,i] > 0):
                ind = -1
            else:
                ind = np.where(cell_count == 0)[0][0] + 1
            ax[0].plot(time[:ind,i],b_all[:ind,i])
            ax[1].plot(time[:ind,i],kn0_all[:ind,i])
            ax[2].plot(time[:ind,i], cell_count[:ind,i])
        ax[0].set_ylabel('antibiotic conc.')
        ax[0].axvline(x=self.delta_t*(36+self.embed_len), linestyle='--', color='k')
        ax[1].set_ylabel('nutrient conc.')
        ax[1].axvline(x=self.delta_t*(36+self.embed_len), linestyle='--', color='k')
        ax[2].set_ylabel('population size')
        ax[2].axvline(x=self.delta_t*(36+self.embed_len), linestyle='--', color='k')
        ax[2].set_xlabel('Time (h)')
        ax[2].set_yscale('log')
        figure.savefig(self.folder_name_test+'/'+str(episode)+'.jpg', dpi=300, bbox_inches='tight')
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
    figure.savefig(folder+'/reward_Q_loss.jpg', dpi=300, bbox_inches='tight')
    plt.close()