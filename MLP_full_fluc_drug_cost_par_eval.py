import os
import random
import torch
import torch.nn as nn
import numpy as np
from cell_model_fluc_drug_cost_eval import Cell_Population
from replaybuffer import ReplayBuffer
from deepQLnetwork import Model
from joblib import Parallel, delayed
from utils_figure_plot import DynamicUpdate, plot_reward_Q_loss
import copy
from scipy import signal
import matplotlib as mpl
import wandb
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


class CDQL:

    # later add in learning rate and tau (update rate) as arguments to tune over as well
    #  - although learning rate and batch size have similar effects

    def __init__(self, delay_embed_len=10,
            batch_size=512,
            delta_t=0.2,
            omega=0.02,
            gamma=0.99,
            update_freq=2,
            use_gpu=False,
            num_cells_init=60,
            kn0_mean=2.55,
            T=12):
        self.gamma = gamma
        if use_gpu and torch.cuda.is_available(): # and torch.cuda.device_count() > 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        assert (not use_gpu) or (self.device == torch.device('cuda'))
        self.folder_name = "./"

        self.delta_t = delta_t
        self.sim_controller = Cell_Population(num_cells_init, delta_t, omega, kn0_mean, T)
        self.buffer = ReplayBuffer(1e6)
        self.delay_embed_len = delay_embed_len
        self.batch_size = batch_size
        self.b_actions = [0.0, 3.72]
        self.b_num_actions = len(self.b_actions)
        self.num_actions = self.b_num_actions
        self.model = Model(self.device, num_inputs=self.delay_embed_len*3, num_actions=self.num_actions)
        self.b_index = [0, 1]

        self.init = 0.0 # b_init
        self.loss = []
        self.ave_sum_rewards = []
        self.std_sum_rewards = []
        self.ave_Q1 = []
        self.ave_Q2 = []
        self.ave_Q1_target = []
        self.ave_Q2_target = []
        self.grad_updates = []
        self.training_iter = 0
        self.update_freq = update_freq
        self.epsilon = None
        self.episode_num = 0

    def _to_tensor(self, x):
        return torch.tensor(x).float().to(self.device)

    def _save_data(self, folder, sweep_var, episode_num):
        np.save(folder + "/replaybuffer" + str(sweep_var), np.array(self.buffer.buffer, dtype=object))

        self.model.save_networks(folder+'/', sweep_var)
        self.save_episode_num(episode_num)
    
    def save_episode_num(self, episode_num, filename='episode_num.txt'):
        filename = self.folder_name + filename
        with open(filename,"w") as file:
            file.write(str(episode_num))

    def load_data(self, folder_name="./Fluc_nutrient/"):
        self.buffer.load_buffer(folder_name + "replaybuffer.npy")
        self.model.load_networks(folder_name)
        self.episode_num = self.load_episode_num()
    
    def load_episode_num(self, filename='episode_num.txt'):
        filename = self.folder_name + filename
        try:
            with open(filename, "r") as file:
                episode_num = int(file.read().strip()) + 1
            print("Partially trained model found, starting from episode ",episode_num,".")
            return episode_num
        except FileNotFoundError:
            return 0

    def _get_action(self, state, episode=None, eval=False):
        """Gets action given some state using epsilon-greedy algorithm
        Args:
            state: List defining the state
            episode: episode number
        """

        if not eval:
            explore = random.random()
            if (explore < self.epsilon[episode]):
                action = random.choice(list(range(self.num_actions)))
                return action

        self.model.q_1.eval()
        with torch.no_grad():
            curr_state = self._to_tensor(state)
            curr_state = curr_state.unsqueeze(0)
            action = torch.argmin(
                    self.model.q_1(curr_state), dim=1).item()
            if eval:
                store_Q1 = [self.model.q_1(curr_state).squeeze(0)[i].item() for i in range(self.num_actions)]
                store_Q_target1 = [self.model.q_target_1(curr_state).squeeze(0)[i].item() for i in range(self.num_actions)]
                store_Q2 = [self.model.q_2(curr_state).squeeze(0)[i].item() for i in range(self.num_actions)]
                store_Q_target2 = [self.model.q_target_2(curr_state).squeeze(0)[i].item() for i in range(self.num_actions)]
        self.model.q_1.train()

        return (action, store_Q1, store_Q_target1, store_Q2, store_Q_target2) if eval else action

    def _update(self):
        """Updates q1, q2, q1_target and q2_target networks based on clipped Double Q Learning Algorithm
        """
        if (len(self.buffer) < self.batch_size):
            return
        self.training_iter += 1
        # Make sure actor_target and critic_target are in eval mode
        assert not self.model.q_target_1.training
        assert not self.model.q_target_2.training

        assert self.model.q_1.training
        assert self.model.q_2.training
        transitions = self.buffer.sample(self.batch_size)
        batch = self.buffer.transition(*zip(*transitions))
        state_batch = self._to_tensor(batch.state)
        action_batch = self._to_tensor(
            batch.action).unsqueeze(1).to(torch.int64)
        reward_batch = self._to_tensor(batch.reward)
        next_state_batch = self._to_tensor(batch.next_state)

        with torch.no_grad():
            # Add noise to smooth out learning
            Q_next_1 = torch.min(self.model.q_target_1(
                next_state_batch), dim=1)[0].unsqueeze(1)
            Q_next_2 = torch.min(self.model.q_target_2(
                next_state_batch), dim=1)[0].unsqueeze(1)
            # Use max want to avoid underestimation bias #####
            Q_next = torch.max(Q_next_1, Q_next_2)
            Q_expected = reward_batch + self.gamma * Q_next  # No "Terminal State"

        Q_1 = self.model.q_1(state_batch).gather(1, action_batch)
        Q_2 = self.model.q_2(state_batch).gather(1, action_batch)
        L_1 = nn.MSELoss()(Q_1, Q_expected)
        L_2 = nn.MSELoss()(Q_2, Q_expected)

        self.loss.append([L_1.item(), L_2.item()])
        self.model.q_optimizer_1.zero_grad()
        self.model.q_optimizer_2.zero_grad()
        L_1.backward()
        L_2.backward()
        self.model.q_optimizer_1.step()
        self.model.q_optimizer_2.step()

        # Q_1_new = self.model.q_1(state_batch).gather(1, action_batch)
        # print("  %.2f"%Q_expected.min().item())
        # print("    %.2f"%(Q_1_new - Q_1).min().item())
        # self.store_Q.append([Q_1.tolist(), Q_2.tolist(), Q_expected.tolist()])

        if (self.training_iter % self.update_freq) == 0:
            self.model.update_target_nn()
        self.model.grad_update_num +=1
        

    def train(self, sweep_var, num_decisions=250): #200
        """Train q networks
        Args:
            num_decisions: Number of decisions to train algorithm for
        """
        # os.system("mkdir -p " + self.folder_name + "testing!")
        episodes = 350 #400
        e = np.arange(episodes)
        T = 300 # 380, choosing how fast to move from exploration to exploitation
        eps = -np.log10(e/T)
        self.epsilon = np.clip(eps,0.05,1) # clipping to ensure all values are between 0 and 1
        update_plot = DynamicUpdate(self.delta_t,self.delay_embed_len)

        for i in range(self.episode_num,episodes):
            print("Episode: ", i)
            # episode_folder_name = self.folder_name + "Train/" + str(i) + "/"

            # os.system("mkdir -p " + episode_folder_name)
            # filename = episode_folder_name + str(i)

            # warmup
            b = self.init
            state = [0]*self.delay_embed_len*3
            self.sim_controller.initialize(b)
            _, cell_count = self.sim_controller.simulate_population(self.sim_controller.num_cells_init, b)
            for k in range(1,36+self.delay_embed_len):
                _, cell_count = self.sim_controller.simulate_population(cell_count[-1], b)
                if k >= 36:
                    state, _ = self.sim_controller.get_state_reward(state, cell_count, b)

            for j in range(num_decisions):
                action_index = self._get_action(state, i)
                transition_to_add = [copy.deepcopy(state), action_index]

                action_b = self.b_actions[self.b_index[action_index]]

                _, cell_count = self.sim_controller.simulate_population(cell_count[-1], action_b)
                state, reward = self.sim_controller.get_state_reward(state, cell_count, action_b/max(self.b_actions))
                transition_to_add.extend([[reward], copy.deepcopy(state)])
                self.buffer.push(*list(transition_to_add))
                self._update()
                if cell_count[-1] == 0:
                    break

            if (i % 10 == 0) or (i == episodes-1):
                self.eval(i)
                self._save_data(update_plot.folder_name, sweep_var, i)
                if i == episodes-1:
                    plot_reward_Q_loss(self.ave_sum_rewards, self.std_sum_rewards, self.grad_updates, self.loss, update_plot.folder_name_test,
                                   self.ave_Q1, self.ave_Q2, self.ave_Q1_target, self.ave_Q2_target)



    def eval(self, episode, num_decisions=250, num_evals=30): #200, 50
        """Given trained q networks, generate trajectories
        """
        update_plot = DynamicUpdate(self.delta_t,self.delay_embed_len)
        os.system("mkdir " + update_plot.folder_name_test)
        print("Evaluation")

        extinct_times = []
        extinct_count = 0
        max_cross_corr = []
        lag = []

        b_all = np.zeros((num_decisions+self.delay_embed_len,1))
        cell_count_all = np.zeros((num_decisions+self.delay_embed_len,1))
        t_all = np.zeros((num_decisions+self.delay_embed_len,1))
        kn0_all = np.zeros((num_decisions+self.delay_embed_len,1))
        rewards_all = np.zeros((num_decisions+self.delay_embed_len,1))
        Q1_all = np.zeros((num_decisions+self.delay_embed_len,2))
        Q1_target_all = np.zeros((num_decisions+self.delay_embed_len,2))
        Q2_all = np.zeros((num_decisions+self.delay_embed_len,2))
        Q2_target_all = np.zeros((num_decisions+self.delay_embed_len,2))


        results = Parallel(n_jobs=-1)(delayed(self.rollout)(b_all, cell_count_all, t_all, kn0_all, rewards_all, num_decisions,
                                                            Q1_all, Q1_target_all, Q2_all, Q2_target_all)for i in range(num_evals))
        # results = [self.rollout(b_all, cell_count_all, t_all, kn0_all, rewards_all, num_decisions, Q1_all, Q1_target_all, Q2_all, Q2_target_all) for i in range(num_evals)]
        i=0
        for r in results:
            if i == 0:
                b_all, cell_count_all, t_all, kn0_all, rewards_all, Q1, Q1_target, Q2, Q2_target = r
                sum_reward = np.array([rewards_all.sum()])
                min_Q1 = np.array(Q1.min(axis=1))
                min_Q1_target = np.array(Q1_target.min(axis=1))
                min_Q2 = np.array(Q2.min(axis=1))
                min_Q2_target = np.array(Q2_target.min(axis=1))
                if not np.all(cell_count_all > 0):
                    ind = np.where(cell_count_all == 0)[0][0]
                    extinct_times.append(t_all[ind])
                    extinct_count +=1
                    ind +=1
                else:
                    ind = b_all.size
                # compute max cross correlation and lag
                n_points = ind
                cross_corr = signal.correlate(b_all[:ind] - np.mean(b_all[:ind]), kn0_all[:ind] - np.mean(kn0_all[:ind]), mode='full')
                cross_corr /= (np.std(b_all[:ind]) * np.std(kn0_all[:ind]) * n_points)  # Normalize
                max_cross_corr.append(np.max(cross_corr))
                lags = signal.correlation_lags(b_all[:ind].size,kn0_all[:ind].size, mode="full")
                lag.append(lags[np.argmax(cross_corr)])
                i+=1
            else:
                b, cell_count, t, kn0, rewards, Q1, Q1_target, Q2, Q2_target = r
                b_all = np.concatenate((b_all, b), axis=1)
                cell_count_all = np.concatenate((cell_count_all, cell_count), axis=1)
                t_all = np.concatenate((t_all,t), axis=1)
                kn0_all = np.concatenate((kn0_all,kn0), axis=1)
                rewards_all = np.concatenate((rewards_all,rewards), axis=1)
                sum_reward = np.concatenate((sum_reward, np.array([rewards.sum()])))
                min_Q1 = np.vstack((min_Q1, np.array(Q1.min(axis=1))))
                min_Q1_target = np.vstack((min_Q1_target, np.array(Q1_target.min(axis=1))))
                min_Q2 = np.vstack((min_Q2, np.array(Q2.min(axis=1))))
                min_Q2_target = np.vstack((min_Q2_target, np.array(Q2_target.min(axis=1))))
                if not np.all(cell_count > 0):
                    ind = np.where(cell_count == 0)[0][0]
                    extinct_times.append(t[ind])
                    extinct_count +=1
                    ind +=1
                else:
                    ind = b.size
                # compute max cross correlation and lag
                n_points = ind
                cross_corr = signal.correlate(b[:ind] - np.mean(b[:ind]), kn0[:ind] - np.mean(kn0[:ind]), mode='full')
                cross_corr /= (np.std(b[:ind]) * np.std(kn0[:ind]) * n_points)  # Normalize
                max_cross_corr.append(np.max(cross_corr))
                lags = signal.correlation_lags(b[:ind].size,kn0[:ind].size, mode="full")
                lag.append(lags[np.argmax(cross_corr)])

        self.ave_sum_rewards.append(sum_reward.mean())
        self.std_sum_rewards.append(sum_reward.std())
        self.ave_Q1.append(min_Q1.mean())
        self.ave_Q2.append(min_Q2.mean())
        self.ave_Q1_target.append(min_Q1_target.mean())
        self.ave_Q2_target.append(min_Q2_target.mean())
        self.grad_updates.append(self.model.grad_update_num)
        # save extinction quantification results
        if len(extinct_times) > 0:
            ave_ext_time = sum(extinct_times)/len(extinct_times)
        else:
            ave_ext_time = np.Inf

        # log via wandb
        wandb.log({"extinct_fraction": extinct_count/num_evals,
            "ave_ext_rate": 1/ave_ext_time,
            "ave_max_cross_corr": sum(max_cross_corr)/len(max_cross_corr),
            "ave_corr_lag": sum(lag)/len(lag),
            "ave total reward": sum_reward.mean(),
            "ave min Q1": min_Q1.mean()})
        # select five trajectories randomly to plot
        rand_i = random.sample(range(num_evals), 5)
        update_plot(episode, t_all[:,rand_i], cell_count_all[:,rand_i], kn0_all[:,rand_i], b_all[:,rand_i])


    def rollout(self, b_all, cell_count_all, t_all, kn0_all, rewards_all, num_decisions, Q1_all, Q1_target_all, Q2_all, Q2_target_all):
        # warmup
        b = self.init
        state = [0]*self.delay_embed_len*3
        self.sim_controller.initialize(b)
        t, cell_count = self.sim_controller.simulate_population(self.sim_controller.num_cells_init, b)
        for k in range(1,36+self.delay_embed_len):
            t, cell_count = self.sim_controller.simulate_population(cell_count[-1], b)
            if k >= 36:
                _, Q1_all[k-36,:], Q1_target_all[k-36,:], Q2_all[k-36,:], Q2_target_all[k-36,:] = self._get_action(state, eval=True) # just calling this to save Q value for plot
                state, rewards_all[k-36] = self.sim_controller.get_state_reward(state, cell_count, b)

                b_all[k-36] = b
                cell_count_all[k-36] = cell_count[-1]
                t_all[k-36] = t[-1]
                kn0_all[k-36] = self.sim_controller.k_n0

        for j in range(num_decisions):
            action_index, Q1_all[j+self.delay_embed_len,:], Q1_target_all[j+self.delay_embed_len,:], Q2_all[j+self.delay_embed_len,:], Q2_target_all[j+self.delay_embed_len,:] = self._get_action(state, eval=True)
            action_b = self.b_actions[self.b_index[action_index]]

            t, cell_count = self.sim_controller.simulate_population(cell_count[-1], action_b)
            state, rewards_all[j+self.delay_embed_len] = self.sim_controller.get_state_reward(state, cell_count, action_b/max(self.b_actions))

            b_all[j+self.delay_embed_len] = action_b
            cell_count_all[j+self.delay_embed_len] = cell_count[-1]
            t_all[j+self.delay_embed_len] = t[-1]
            kn0_all[j+self.delay_embed_len] = self.sim_controller.k_n0
            if cell_count[-1] == 0:
                break

        return b_all, cell_count_all, t_all, kn0_all, rewards_all, Q1_all, Q1_target_all, Q2_all, Q2_target_all

if __name__ == "__main__":
    pass
    # torch.manual_seed(0)
    # c = CDQL()
    # # c.load_data()
    # c.train()