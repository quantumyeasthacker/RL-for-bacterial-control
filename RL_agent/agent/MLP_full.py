import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joblib import Parallel, delayed
import copy
from scipy import signal
import wandb
# import pickle

from agent.replaybuffer import ReplayBuffer
from agent.deepQLnetwork import Model

from envs.envs import ConstantNutrientEnv, VariableNutrientEnv, ControlNutrientEnv, BaseEnv, EnvConfig
from envs.cell_model import CellConfig

from utils.utils_figure_plot import plot_trajectory, plot_reward_Q_loss


EPS = 1e-10


class CDQL(object):
    def __init__(
        self,
        env: BaseEnv,
        buffer_size: int = 1_000_000,
        batch_size: int = 512,
        gamma: float = 0.99,
        # lambda_smooth: float = 0.1,
        # noise_scale : float = 1.0,
        update_freq: int = 2,
        train_freq: int = 1,
        gradient_steps: int = 1,
        use_gpu: bool = False,
    ) -> None:
        '''
        Args:
            env: environment to train on
            buffer_size: size of replay buffer
            batch_size: batch size for training
            gamma: reward discount factor
            # lambda_smooth: temporal regularization coefficient
            # noise_scale: noise parameters for smooth exploration
            update_freq: frequency of target q updates per q update
            train_freq: frequency of training per step
            gradient_steps: number of gradient steps to take each update
            use_gpu: whether to use gpu
        '''
        if use_gpu and torch.cuda.is_available(): # and torch.cuda.device_count() > 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        assert (not use_gpu) or (self.device == torch.device('cuda'))

        self.env = env
        self.model = Model(self.device,
                           num_inputs = self.env.delay_embed_len*(1 + self.env.k_n0_observation + self.env.b_observation),
                           num_actions = self.env.num_actions)

        # env = env_config.env_name(env_config, cell_config)
        # model = Model(self.device, num_inputs = env_config.delay_embed_len*
        #               (1 + env_config.k_n0_observation + env_config.b_observation), num_actions = 2)

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        # self.lambda_smooth = lambda_smooth
        # self.noise_scale = noise_scale
        self.update_freq = update_freq
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        self.loss = []
        self.ave_sum_rewards = []
        self.std_sum_rewards = []
        self.ave_Q1 = []
        self.ave_Q2 = []
        self.ave_Q1_target = []
        self.ave_Q2_target = []
        self.grad_updates = []
        self.training_iter = 0 # number of updates for q1 and q2
        self.epsilon = None
        # self.episode_num = 0 # episode number for environment

    def _to_tensor(self, x):
        return torch.tensor(x).float().to(self.device)

    def _save_data(self, folder_name, replay_buffer = False):
        os.makedirs(folder_name, exist_ok=True)
        self.model.save_networks(folder_name)
        if replay_buffer:
            np.save(os.path.join(folder_name, "replaybuffer.npy"), np.array(self.buffer.buffer, dtype=object))
        # with open(os.path.join(folder_name, "replaybuffer.pkl"), "wb") as f:
        #     pickle.dump(self.buffer.buffer, f)
        # with open(os.path.join(folder_name, "episode_num.txt"), "w") as f:
        #     f.write(str(self.episode_num))

    def load_data(self, folder_name, replay_buffer = False):
        self.model.load_networks(folder_name)
        if replay_buffer:
            self.buffer.load_buffer(os.path.join(folder_name, "replaybuffer.npy"))
        # with open(os.path.join(folder_name, "replaybuffer.pkl"), "rb") as f:
        #     self.buffer = pickle.load(f)
        # try:
        #     with open(os.path.join(folder_name, "episode_num.txt"), "r") as f:
        #         self.episode_num = int(f.read())
        #     print(f"Partially trained model found, starting from episode {self.episode_num}.")
        # except:
        #     print("No partially trained model found, starting from episode 0.")

    def _update(self) -> None:
        """Updates q1, q2, q1_target and q2_target networks based on clipped Double Q Learning Algorithm
        """
        if (len(self.buffer) < self.batch_size):
            return

        self.model.q_1.train()
        self.model.q_2.train()
        self.model.q_target_1.eval()
        self.model.q_target_2.eval()

        for _ in range(self.gradient_steps):
            transitions = self.buffer.sample(self.batch_size)
            batch = self.buffer.transition(*zip(*transitions))
            state_batch = self._to_tensor(batch.state)
            action_batch = self._to_tensor(batch.action).unsqueeze(-1).to(torch.int64)
            reward_batch = self._to_tensor(batch.reward).unsqueeze(-1)
            next_state_batch = self._to_tensor(batch.next_state)
            terminated_batch = self._to_tensor(batch.terminated).unsqueeze(-1)

            with torch.no_grad():
                # Add noise to smooth out learning
                Q_next_1 = torch.min(self.model.q_target_1(next_state_batch), dim=-1, keepdim=True).values
                Q_next_2 = torch.min(self.model.q_target_2(next_state_batch), dim=-1, keepdim=True).values
                # Use max want to avoid underestimation bias #####
                Q_next = torch.maximum(Q_next_1, Q_next_2)
                Q_expected = reward_batch + self.gamma * Q_next * (1 - terminated_batch)

            Q_1 = self.model.q_1(state_batch).gather(-1, action_batch)
            Q_2 = self.model.q_2(state_batch).gather(-1, action_batch)
            L_1 = nn.MSELoss()(Q_1, Q_expected)
            L_2 = nn.MSELoss()(Q_2, Q_expected)

            # # Action smoothness regularization
            # consecutive_q_1_values = self.model.q_1(state_batch)
            # next_q_1_values = self.model.q_1(next_state_batch)
            # action1_probs = F.softmax(consecutive_q_1_values, dim=-1)
            # next_action1_probs = F.softmax(next_q_1_values, dim=-1)
            # smoothness1_loss = nn.MSELoss()(action1_probs, next_action1_probs)
            # L_1 += self.lambda_smooth * smoothness1_loss

            # consecutive_q_2_values = self.model.q_2(state_batch)
            # next_q_2_values = self.model.q_2(next_state_batch)
            # action2_probs = F.softmax(consecutive_q_2_values, dim=-1)
            # next_action2_probs = F.softmax(next_q_2_values, dim=-1)
            # smoothness2_loss = nn.MSELoss()(action2_probs, next_action2_probs)
            # L_2 += self.lambda_smooth * smoothness2_loss

            self.loss.append([L_1.item(), L_2.item()])
            self.model.q_optimizer_1.zero_grad()
            self.model.q_optimizer_2.zero_grad()
            L_1.backward()
            L_2.backward()
            self.model.q_optimizer_1.step()
            self.model.q_optimizer_2.step()
            self.training_iter += 1
            if (self.training_iter % self.update_freq) == 0:
                self.model.update_target_nn()
        # self.model.grad_update_num +=1

    def train(self, episodes: int, num_decisions: int, num_evals: int = 10, folder_name: str = "./") -> None:
        """Train the model
        Args:
            episodes: number of episodes to train
            gradient_steps: number of gradient steps to take
        """

        T_eps = 300 # 380, choosing how fast to move from exploration to exploitation
        epsilon_list = np.arange(episodes)
        epsilon_list = (-np.log10(epsilon_list/T_eps + EPS)).clip(0.05, 1)

        step_iter = 0
        for episode in range(episodes):
            obs, _ = self.env.reset()
            # while True:
            for _ in range(num_decisions):
                action = self.model.get_action(obs, deterministic = False, epsilon = epsilon_list[episode])
                obs_next, reward, terminated, truncated, _ = self.env.step(action)
                self.buffer.push(obs, action, reward, obs_next, terminated)
                step_iter += 1
                if step_iter % self.train_freq == 0:
                    self._update()
                obs = obs_next
                if terminated or truncated:
                    break
            print(f"Episode {episode} completed.")
            # if episode % 10 == 10 - 1:
            if (episode % 10 == 0) or (episode == episodes - 1):
                # self._save_data(folder_name)
                self._save_data(os.path.join(folder_name, f"episode_{episode}"))
                self.evalulate(episode, num_decisions, num_evals, folder_name)

            if episode == episodes - 1:
                plot_reward_Q_loss(self.ave_sum_rewards, self.std_sum_rewards, self.grad_updates, self.loss, folder_name,
                                   self.ave_Q1, self.ave_Q2, self.ave_Q1_target, self.ave_Q2_target)

    def evalulate(self, episode: int, num_decisions: int, num_evals: int, folder_name: str) -> None:
        """Evaluate the model
        Args:
            num_decisions: number of decisions to make
        """
        self.model.q_1.eval()
        self.model.q_2.eval()
        
        extinct_times = []
        extinct_count = 0
        max_cross_corr_kn0 = []
        max_cross_corr_U = []
        lag_kn0 = []
        lag_U = []


        results = Parallel(n_jobs=10)(delayed(self.eval_step)(num_decisions) for _ in range(num_evals))
        # results = [self.eval_step(num_decisions) for _ in range(num_evals)]
        sum_rewards_all, min_Q_values_all, terminated_all, _, info_all = zip(*results)
        ave_q1, ave_q2, ave_q1_target, ave_q2_target = np.mean(min_Q_values_all, axis=0)

        for terminated, info in zip(terminated_all, info_all):
            time, nutr, drug, _, damage = list(zip(*info['log']))[:5]

            # compute max cross correlation and lag
            drug = drug[(self.env.warm_up+1):]
            nutr = nutr[(self.env.warm_up-self.env.delay_embed_len+1):]
            damage = damage[self.env.warm_up+1:]
            
            # drug = drug[(self.env.warm_up+1):]
            # nutr = nutr[(self.env.warm_up+1):]
            # damage = damage[(self.env.warm_up+1):]

            if np.std(drug) > 0 and np.std(nutr) > 0:
                cross_correlation(drug, nutr, max_cross_corr_kn0, lag_kn0)
            if np.std(drug) > 0 and np.std(damage) > 0:
                cross_correlation(drug, damage, max_cross_corr_U, lag_U)

            # save extinction times            
            if terminated:
                extinct_times.append(time[-1])
                extinct_count += 1

        self.ave_sum_rewards.append(np.mean(sum_rewards_all))
        self.std_sum_rewards.append(np.std(sum_rewards_all))
        self.ave_Q1.append(ave_q1)
        self.ave_Q2.append(ave_q2)
        self.ave_Q1_target.append(ave_q1_target)
        self.ave_Q2_target.append(ave_q2_target)
        self.grad_updates.append(self.training_iter)

        # save results
        ave_ext_time = sum(extinct_times)/len(extinct_times) if len(extinct_times) > 0 else np.Inf
        ave_max_cross_corr_kn0 = sum(max_cross_corr_kn0)/len(max_cross_corr_kn0) if len(max_cross_corr_kn0) > 0 else 0
        ave_corr_lag_kn0 = sum(lag_kn0)/len(lag_kn0) if len(lag_kn0) > 0 else 0
        ave_max_cross_corr_U = sum(max_cross_corr_U)/len(max_cross_corr_U) if len(max_cross_corr_U) > 0 else 0
        ave_corr_lag_U = sum(lag_U)/len(lag_U) if len(lag_U) > 0 else 0
        # log via wandb
        wandb.log({
            "extinct_fraction": extinct_count/num_evals,
            "ave_ext_rate": 1/ave_ext_time,
            "ave_max_cross_corr_kn0": ave_max_cross_corr_kn0,
            "ave_corr_lag_kn0": ave_corr_lag_kn0,
            "ave_max_cross_corr_U": ave_max_cross_corr_U,
            "ave_corr_lag_U": ave_corr_lag_U,
            "ave total reward": np.mean(sum_rewards_all),
            "ave min Q1": ave_q1
        })
        plot_trajectory(random.sample(info_all, 5), episode, os.path.join(folder_name,"Eval"))

    def eval_step(self, num_decisions: int) -> tuple[list, list, bool, bool, dict]:
        obs, _ = self.env.reset()
        rewards = []
        Q_values = []
        for _ in range(num_decisions):
            action = self.model.get_action(obs, deterministic = True)
            obs_tensor = self._to_tensor(obs)
            with torch.no_grad():
                Q_value = [q(obs_tensor) for q in self.model.q_networks]
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            Q_values.append(Q_value)
            if terminated or truncated:
                break
        # return rewards, Q_values, terminated, truncated, info
        return np.sum(rewards), np.array(Q_values).min(-1).mean(0), terminated, truncated, info


def cross_correlation(sig1, sig2, max_cross_corr, lag):
    n_points = len(sig1)
    cross_corr = signal.correlate(sig1 - np.mean(sig1), sig2 - np.mean(sig2), mode='full')
    cross_corr /= (np.std(sig1) * np.std(sig2) * n_points)  # Normalize
    max_cross_corr.append(np.max(cross_corr))
    lags = signal.correlation_lags(len(sig1),len(sig2), mode="full")
    lag.append(lags[np.argmax(cross_corr)])