import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import wandb
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


class DynamicUpdate():
    """Plot evaluation trajectories
    """
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.folder_name_test = self.folder_name + "/Eval"
        os.system("mkdir " + self.folder_name_test)

    def __call__(self, episode, time, cell_count, kn0, b, rand_i):
        figure, ax = plt.subplots(3,1)
        figure.subplots_adjust(hspace=.0)
        start_t = np.min(time[0])

        for i in rand_i:
            ax[0].plot(time[i]-start_t, b[i])
            ax[1].plot(time[i]-start_t, kn0[i])
            ax[2].plot(time[i]-start_t, cell_count[i])
        ax[0].set_ylabel('antibiotic conc.')
        ax[1].set_ylabel('nutrient conc.')
        ax[2].set_ylabel('population size')
        ax[2].set_xlabel('Time (h)')
        ax[2].set_yscale('log')
        figure.savefig(self.folder_name_test+'/'+str(episode)+'.jpg', dpi=300, bbox_inches='tight')
        # wandb.log({"plot_trajectories": wandb.Image(self.folder_name_test+'/'+str(episode)+'.jpg')})
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
    # wandb.log({"plot_reward_Q_loss": wandb.Image(folder+'/reward_Q_loss.jpg')})
    plt.close()