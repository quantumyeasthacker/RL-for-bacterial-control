import os
import random
import torch
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed
try:
    import wandb
except ImportError:
    # wandb is only used for training-time logging; eval/inference does not need it.
    wandb = None

from .replaybuffer import ReplayBuffer
from .deepQLnetwork import Model

from ..envs.envs import BaseEnv
from ..utils.utils_figure_plot import plot_trajectory, plot_reward_Q_loss
from ..utils.utils_signal import cross_correlation



class CDQL(object):
    def __init__(
        self,
        env: BaseEnv,
        buffer_size: int = 1_000_000,
        batch_size: int = 512,
        gamma: float = 0.99,
        update_freq: int = 2,
        train_freq: int = 1,
        gradient_steps: int = 1,
        use_gpu: bool = False,
        learning_rate: float = 1e-4,
        network_update_rate: float = 0.005
    ) -> None:
        '''
        Args:
            env: environment to train on
            buffer_size: size of replay buffer
            batch_size: batch size for Q network update during training
            gamma: reward discount factor
            update_freq: number of q updates in between each target q update
            train_freq: number of actions in between each online q network update, default 1
            gradient_steps: number of gradient steps to take each update, default 1
            use_gpu: whether to use gpu
            learning_rate: lr for updating Q networks based on Bellman residual
            network_update_rate: Polyak averaging update rate for target network
        '''
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        assert (not use_gpu) or (self.device == torch.device('cuda'))

        self.env = env
        self.model = Model(self.device,
                           num_inputs = self.env.obs_len,
                           num_actions = self.env.num_actions,
                           learning_rate = learning_rate,
                           tau = network_update_rate)

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_freq = update_freq
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        self.loss = []
        self.ave_rewards = []   # per-step mean eval reward (mean over decisions, then over eval rollouts)
        self.std_rewards = []
        self.ave_Q1 = []
        self.ave_Q2 = []
        self.ave_Q1_target = []
        self.ave_Q2_target = []
        self.grad_updates = []
        self.training_iter = 0 # number of gradient updates to q1 and q2, used to make learning curves plotted against gradient updates rather than episodes
        self.epsilon = None

    def _to_tensor(self, x):
        return torch.tensor(x).float().to(self.device)

    def _save_data(self, folder_name, replay_buffer = False):
        os.makedirs(folder_name, exist_ok=True)
        self.model.save_networks(folder_name)
        if replay_buffer:
            np.save(os.path.join(folder_name, "replaybuffer.npy"), np.array(self.buffer.buffer, dtype=object))

    def load_data(self, folder_name, replay_buffer = False):
        self.model.load_networks(folder_name)
        if replay_buffer:
            self.buffer.load_buffer(os.path.join(folder_name, "replaybuffer.npy"))

    def _update(self) -> None:
        """Updates q1, q2, q1_target and q2_target networks based on Clipped Double Q Learning Algorithm (TD3 with modifications for discrete action space with no actor)
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
                Q_next_1 = torch.min(self.model.q_target_1(next_state_batch), dim=-1, keepdim=True).values
                Q_next_2 = torch.min(self.model.q_target_2(next_state_batch), dim=-1, keepdim=True).values
                # max used to avoid underestimation bias
                Q_next = torch.maximum(Q_next_1, Q_next_2)
                Q_expected = reward_batch + self.gamma * Q_next * (1 - terminated_batch)

            Q_1 = self.model.q_1(state_batch).gather(-1, action_batch)
            Q_2 = self.model.q_2(state_batch).gather(-1, action_batch)
            L_1 = nn.MSELoss()(Q_1, Q_expected)
            L_2 = nn.MSELoss()(Q_2, Q_expected)

            self.loss.append([L_1.item(), L_2.item()])
            self.model.q_optimizer_1.zero_grad()
            self.model.q_optimizer_2.zero_grad()
            L_1.backward()
            L_2.backward()
            self.model.q_optimizer_1.step()
            self.model.q_optimizer_2.step()
            self.training_iter += 1
            if (self.training_iter % self.update_freq) == 0:
                self.model.update_target_nn() # perform soft udpate to target

    def train(self, episodes: int, num_decisions: int, num_evals: int = 10, folder_name: str = "./") -> None:
        """Main agent train loop
        Args:
            episodes: number of episodes to train
            num_decisions: number of agent decisions per training and eval episode
            num_evals: number of parallel evals to run to periodically assess agent performance
            folder_name: dir for saving results
        """

        T_eps = 300 # choosing how fast to move from exploration to exploitation
        EPS = 1e-10
        epsilon_list = np.arange(episodes)
        epsilon_list = (-np.log10(epsilon_list/T_eps + EPS)).clip(0.05, 1)

        step_iter = 0
        for episode in range(episodes):
            obs, _ = self.env.reset()

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

            if (episode % 10 == 0) or (episode == episodes - 1):
                # save ckpt
                self._save_data(os.path.join(folder_name, f"episode_{episode}"))
                self.evalulate(episode, num_decisions, num_evals, folder_name)

            if episode == episodes - 1:
                reward_ylabel = ("ave. eval reward per step" if self.env.reward_type == "log10_pop"
                                 else "ave. total eval reward")
                plot_reward_Q_loss(self.ave_rewards, self.std_rewards, self.grad_updates, self.loss, folder_name,
                                   self.ave_Q1, self.ave_Q2, self.ave_Q1_target, self.ave_Q2_target,
                                   reward_ylabel=reward_ylabel)

    def evalulate(self, episode: int, num_decisions: int, num_evals: int, folder_name: str) -> None:
        """Evaluate a given model ckpt
        Args:
            episode: episode number for performance record keeping
            num_decisions: number of decisions to make per eval run
            num_evals: number of evals runs to make (in parallel)
            folder_name: dir to save eval results
        """

        extinct_times = []
        extinct_count = 0
        max_cross_corr_kn0 = []
        max_cross_corr_U = []
        lag_kn0 = []
        lag_U = []


        results = Parallel(n_jobs=10)(delayed(self.eval_step)(num_decisions) for _ in range(num_evals))
        # results = [self.eval_step(num_decisions) for _ in range(num_evals)]
        agg_rewards_all, min_Q_values_all, terminated_all, _, info_all = zip(*results)
        ave_q1, ave_q2, ave_q1_target, ave_q2_target = np.mean(min_Q_values_all, axis=0)

        for terminated, info in zip(terminated_all, info_all):
            time, nutr, drug, _, damage = list(zip(*info['log']))[:5]

            # compute max cross correlation and lag
            drug = drug[(self.env.warm_up+1):]
            nutr = nutr[(self.env.warm_up-self.env.delay_embed_len+1):]
            # nutr = nutr[(self.env.warm_up+1):]
            damage = damage[self.env.warm_up+1:]

            if np.std(drug) > 0 and np.std(nutr) > 0:
                cross_correlation(drug, nutr, max_cross_corr_kn0, lag_kn0)
            if np.std(drug) > 0 and np.std(damage) > 0:
                cross_correlation(drug, damage, max_cross_corr_U, lag_U)

            # save extinction times
            if terminated:
                extinct_times.append(time[-1])
                extinct_count += 1

        self.ave_rewards.append(np.mean(agg_rewards_all))
        self.std_rewards.append(np.std(agg_rewards_all))
        self.ave_Q1.append(ave_q1)
        self.ave_Q2.append(ave_q2)
        self.ave_Q1_target.append(ave_q1_target)
        self.ave_Q2_target.append(ave_q2_target)
        self.grad_updates.append(self.training_iter)

        # save results
        ave_ext_time = sum(extinct_times)/len(extinct_times) if len(extinct_times) > 0 else np.inf
        ave_max_cross_corr_kn0 = sum(max_cross_corr_kn0)/len(max_cross_corr_kn0) if len(max_cross_corr_kn0) > 0 else 0
        ave_corr_lag_kn0 = sum(lag_kn0)/len(lag_kn0) if len(lag_kn0) > 0 else 0
        ave_max_cross_corr_U = sum(max_cross_corr_U)/len(max_cross_corr_U) if len(max_cross_corr_U) > 0 else 0
        ave_corr_lag_U = sum(lag_U)/len(lag_U) if len(lag_U) > 0 else 0
        # log via wandb (skipped if wandb is not installed, e.g. during offline eval)
        # metric name matches the aggregation: per-step mean for log10-pop, episode sum otherwise
        reward_metric_name = "ave reward per step" if self.env.reward_type == "log10_pop" else "ave total reward"
        if wandb is not None:
            wandb.log({
                "extinct_fraction": extinct_count/num_evals,
                "ave_ext_rate": 1/ave_ext_time,
                "ave_max_cross_corr_kn0": ave_max_cross_corr_kn0,
                "ave_corr_lag_kn0": ave_corr_lag_kn0,
                "ave_max_cross_corr_U": ave_max_cross_corr_U,
                "ave_corr_lag_U": ave_corr_lag_U,
                reward_metric_name: np.mean(agg_rewards_all),
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
        # log10-pop reward does not telescope -> report the per-step MEAN (length-normalized, so
        # rollouts of different length are comparable). Growth-rate reward telescopes to endpoints
        # (already length-independent) -> report the episode SUM.
        reward_agg = np.mean if self.env.reward_type == "log10_pop" else np.sum
        return reward_agg(rewards), np.array(Q_values).min(-1).mean(0), terminated, truncated, info
