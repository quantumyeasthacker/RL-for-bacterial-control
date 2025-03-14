import os
import random
import torch
import torch.nn as nn
import numpy as np
from cell_model_full_parallel import Cell_Population
from replaybuffer import ReplayBuffer
from deepQLnetwork import Model
from joblib import Parallel, delayed
from utils_figure_plot import DynamicUpdate, plot_reward_Q_loss
import copy
from scipy import signal
import wandb


class CDQL:
    def __init__(self, batch_size=512,
            delta_t=0.2,
            omega=0.02,
            gamma=0.99,
            update_freq=2,
            use_gpu=False,
            num_cells_init=60,
            kn0_mean=2.55,
            agent_input="no_nutrient",
            training_config={}):

        self.delay_embed_len = training_config["delay_embed_len"]
        self.folder_name = training_config["folder_name"]
        os.makedirs(self.folder_name, exist_ok=True)

        self.gamma = gamma
        if use_gpu and torch.cuda.is_available(): # and torch.cuda.device_count() > 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        assert (not use_gpu) or (self.device == torch.device('cuda'))

        self.delta_t = delta_t
        self.sim_controller = Cell_Population(num_cells_init, delta_t, omega, kn0_mean=kn0_mean)
        self.buffer = ReplayBuffer(1e6)
        self.batch_size = batch_size
        self.b_actions = [0.0, 3.72]
        self.b_num_actions = len(self.b_actions)
        self.num_actions = self.b_num_actions
        self.b_index = [0, 1]

        if agent_input == "full":
            self.get_state_reward = self.sim_controller.get_reward_all
            self.embed_multiplier = 3
        elif agent_input == "no_nutrient":
            self.get_state_reward = self.sim_controller.get_reward_no_nutrient
            self.embed_multiplier = 2
        elif agent_input == "no_act":
            self.get_state_reward = self.sim_controller.get_reward_no_antibiotic
            self.embed_multiplier = 2
        else:
            raise ValueError("Invalid agent_input value.")

        self.model = Model(self.device, num_inputs=self.delay_embed_len*self.embed_multiplier, num_actions=self.num_actions)

        self.init = 0.0 # b_init
        self.max_pop = 1e11 # threshold for terminating episode
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

    def _save_data(self, episode_num):
        np.save(self.folder_name + "/replaybuffer", np.array(self.buffer.buffer, dtype=object))

        self.model.save_networks(self.folder_name+'/')
        self.save_episode_num(episode_num)

    def save_episode_num(self, episode_num, filename='/episode_num.txt'):
        filename = self.folder_name + filename
        with open(filename,"w") as file:
            file.write(str(episode_num))

    def load_data(self, folder_name="./Results"):
        self.folder_name = folder_name
        self.buffer.load_buffer(self.folder_name + "/replaybuffer.npy")
        self.model.load_networks(self.folder_name)
        self.episode_num = self.load_episode_num()

    def load_episode_num(self, filename='/episode_num.txt'):
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
            eval: if False, uses epsilon-greedy action selection, otherwise greedy
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
                store_minQ1 = self.model.q_1(curr_state).min().numpy()
                store_minQtarget1 = self.model.q_target_1(curr_state).min().numpy()
                store_minQ2 = self.model.q_2(curr_state).min().numpy()
                store_minQtarget2 = self.model.q_target_2(curr_state).min().numpy()
        self.model.q_1.train()

        return (action, store_minQ1, store_minQtarget1, store_minQ2, store_minQtarget2) if eval else action

    def _update(self):
        """Updates q1, q2, q1_target and q2_target networks based on clipped Double Q Learning Algorithm
        """
        if (len(self.buffer) < self.batch_size):
            return
        self.training_iter += 1
        # Make sure target networks are in eval mode
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
        terminal = self._to_tensor(batch.terminal)

        with torch.no_grad():
            # Add noise to smooth out learning
            Q_next_1 = torch.min(self.model.q_target_1(
                next_state_batch), dim=1)[0].unsqueeze(1)
            Q_next_2 = torch.min(self.model.q_target_2(
                next_state_batch), dim=1)[0].unsqueeze(1)
            # Use max want to avoid underestimation bias #####
            Q_next = torch.max(Q_next_1, Q_next_2)
            Q_expected = reward_batch + self.gamma * Q_next * (1 - terminal)

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

        if (self.training_iter % self.update_freq) == 0:
            self.model.update_target_nn()
        self.model.grad_update_num +=1


    def train(self, num_decisions=250): ###
        """Train q networks
        Args:
            num_decisions: Number of decisions to train algorithm for
        """
        # self.folder_name = "./Results" + str(sweep_var)
        # os.system("mkdir " + self.folder_name)
        self.update_plot = DynamicUpdate(self.folder_name)

        episodes = 350 ###
        e = np.arange(episodes)
        T = 300 # choosing how fast to move from exploration to exploitation
        eps = -np.log10(e/T)
        self.epsilon = np.clip(eps,0.05,1) # clipping to ensure all values are between 0 and 1

        for i in range(self.episode_num,episodes):
            print("Episode: ", i)

            # warmup
            b = self.init
            state = [0]*self.delay_embed_len*self.embed_multiplier
            self.sim_controller.initialize(b, rand_param=True, rand_per=True)
            _, cell_count = self.sim_controller.simulate_population(self.sim_controller.num_cells_init, b)
            for k in range(1,36+self.delay_embed_len):
                _, cell_count = self.sim_controller.simulate_population(cell_count[-1], b)
                if k >= 36:
                    state, _, _ = self.get_state_reward(state, cell_count, b)

            for j in range(num_decisions):
                action_index = self._get_action(state, i)
                transition_to_add = [copy.deepcopy(state), action_index]

                action_b = self.b_actions[self.b_index[action_index]]

                _, cell_count = self.sim_controller.simulate_population(cell_count[-1], action_b)
                state, reward, terminal = self.get_state_reward(state, cell_count, action_b/max(self.b_actions))
                transition_to_add.extend([[reward], copy.deepcopy(state), [terminal]])
                self.buffer.push(*list(transition_to_add))
                self._update()
                if cell_count[-1] == 0 or cell_count[-1] > self.max_pop:
                    break

            if (i % 10 == 0) or (i == episodes-1):
                self.eval(i)
                self._save_data(i)
                if i == episodes-1:
                    plot_reward_Q_loss(self.ave_sum_rewards, self.std_sum_rewards, self.grad_updates, self.loss, self.update_plot.folder_name_test,
                                   self.ave_Q1, self.ave_Q2, self.ave_Q1_target, self.ave_Q2_target)



    def eval(self, episode, num_decisions=250, num_evals=15): ###
        """Given trained q networks, generate trajectories
        """
        print("Evaluation")

        extinct_times = []
        extinct_count = 0
        max_cross_corr_kn0 = []
        max_cross_corr_U = []
        lag_kn0 = []
        lag_U = []

        results = Parallel(n_jobs=15)(delayed(self.rollout)(num_decisions) for i in range(num_evals))
        # results = [self.rollout(num_decisions) for i in range(num_evals)]

        b, cell_count, t, kn0, sum_rewards, min_Q1, min_Q1_target, min_Q2, min_Q2_target, U = list(zip(*results))
        for drug, nutr, damage, time, count in zip(b,kn0,U,t,cell_count):

            # compute max cross correlation and lag
            if np.std(drug) > 0:
                cross_correlation(drug, nutr, max_cross_corr_kn0, lag_kn0)
                cross_correlation(drug, damage, max_cross_corr_U, lag_U)

            # save extinction times
            if count[-1] == 0:
                    extinct_times.append(time[-1])
                    extinct_count +=1

        self.ave_sum_rewards.append(np.array(sum_rewards).mean())
        self.std_sum_rewards.append(np.array(sum_rewards).std())
        self.ave_Q1.append(np.array(min_Q1).mean())
        self.ave_Q2.append(np.array(min_Q2).mean())
        self.ave_Q1_target.append(np.array(min_Q1_target).mean())
        self.ave_Q2_target.append(np.array(min_Q2_target).mean())
        self.grad_updates.append(self.model.grad_update_num)

        # save results
        ave_ext_time = sum(extinct_times)/len(extinct_times) if len(extinct_times) > 0 else np.Inf
        ave_max_cross_corr_kn0 = sum(max_cross_corr_kn0)/len(max_cross_corr_kn0) if len(max_cross_corr_kn0) > 0 else 0
        ave_corr_lag_kn0 = sum(lag_kn0)/len(lag_kn0) if len(lag_kn0) > 0 else 0
        ave_max_cross_corr_U = sum(max_cross_corr_U)/len(max_cross_corr_U) if len(max_cross_corr_U) > 0 else 0
        ave_corr_lag_U = sum(lag_U)/len(lag_U) if len(lag_U) > 0 else 0
        # log via wandb
        wandb.log({"extinct_fraction": extinct_count/num_evals,
            "ave_ext_rate": 1/ave_ext_time,
            "ave_max_cross_corr_kn0": ave_max_cross_corr_kn0,
            "ave_corr_lag_kn0": ave_corr_lag_kn0,
            "ave_max_cross_corr_U": ave_max_cross_corr_U,
            "ave_corr_lag_U": ave_corr_lag_U,
            "ave total reward": np.array(sum_rewards).mean(),
            "ave min Q1": np.array(min_Q1).mean()})

        # select trajectories randomly to plot
        rand_i = random.sample(range(num_evals), 5)
        self.update_plot(episode, t, cell_count, kn0, b, rand_i)


    def rollout(self, num_decisions):
        b_all = np.zeros((num_decisions,1))
        cell_count_all = np.zeros((num_decisions,1))
        t_all = np.zeros((num_decisions,1))
        kn0_all = np.zeros((num_decisions,1))
        rewards_all = np.zeros((num_decisions,1))
        Q1_all = np.zeros((num_decisions,1))
        Q1_target_all = np.zeros((num_decisions,1))
        Q2_all = np.zeros((num_decisions,1))
        Q2_target_all = np.zeros((num_decisions,1))
        Uave_all = np.zeros((num_decisions,1))

        # warmup
        b = self.init
        state = [0]*self.delay_embed_len*self.embed_multiplier
        self.sim_controller.initialize(b, rand_param=True, rand_per=True)
        _, cell_count = self.sim_controller.simulate_population(self.sim_controller.num_cells_init, b)
        for k in range(1,36+self.delay_embed_len):
            t, cell_count = self.sim_controller.simulate_population(cell_count[-1], b)
            if k >= 36:
                state, _, _ = self.get_state_reward(state, cell_count, b)

        for j in range(num_decisions):
            action_index, Q1_all[j], Q1_target_all[j], Q2_all[j], Q2_target_all[j] = self._get_action(state, eval=True)
            action_b = self.b_actions[self.b_index[action_index]]

            t, cell_count = self.sim_controller.simulate_population(cell_count[-1], action_b)
            state, rewards_all[j], _ = self.get_state_reward(state, cell_count, action_b/max(self.b_actions))

            b_all[j] = action_b
            cell_count_all[j] = cell_count[-1]
            t_all[j] = t[-1]
            kn0_all[j] = self.sim_controller.k_n0
            Uave_all[j] = self.sim_controller.U_ave
            if cell_count[-1] == 0 or cell_count[-1] > self.max_pop:
                # trim arrays to length of episode
                b_all, cell_count_all, t_all, kn0_all, Uave_all = trim([b_all,cell_count_all,t_all,kn0_all,Uave_all], j+1)
                break

        sum_rewards = np.array([rewards_all.sum()])
        return b_all, cell_count_all, t_all, kn0_all, sum_rewards, Q1_all, Q1_target_all, Q2_all, Q2_target_all, Uave_all


def trim(list, trim_ind):
    return [vec[:trim_ind] for vec in list]

def cross_correlation(sig1, sig2, max_cross_corr, lag):
    n_points = sig1.size
    cross_corr = signal.correlate(sig1 - np.mean(sig1), sig2 - np.mean(sig2), mode='full')
    cross_corr /= (np.std(sig1) * np.std(sig2) * n_points)  # Normalize
    max_cross_corr.append(np.max(cross_corr))
    lags = signal.correlation_lags(sig1.size,sig2.size, mode="full")
    lag.append(lags[np.argmax(cross_corr)])


if __name__ == "__main__":
    pass
    # training_config = {"delay_embed_len": 20, "folder_name": "./Results", "T": 12}
    # c = CDQL(training_config=training_config)
    # # c.load_data()
    # c.train()