"""Recurrent (R2D2 / DRQN) clipped-double-Q agent for the rlBacterialControl package.

This is a port of ``RL-for-bacterial-control-RNN/r2d2_RNN_RLmain.py`` adapted to the
package's training/eval ecosystem:
  * dynamics/observations/rewards come from a package env (``VariableNutrientEnv`` etc.),
    which internally uses ``..envs.cell_model.Cell_Population`` -- NOT cell_model_full_parallel;
  * the twin recurrent Q-networks come from ``.deepQLnetwork_RNN`` (stacked RNN + linear
    head) or ``.deepQLnetwork_RNN_encoder_decoder`` (MLP encoder -> 1-layer RNN -> MLP
    decoder), selectable via the ``net_arch`` argument; the sequence replay buffer is
    ``.replaybuffer_RNN`` (a copy of r2d2_RNN_replaybuffer.py);
  * the public API mirrors ``.MLP_full.CDQL`` so it is a drop-in for the package's
    train/eval scripts (e.g. scripts/train/run_w_wandb_single_varenv.py).

Unlike the MLP agent, temporal memory is supplied by the RNN hidden state rather than
delay embedding, so the env is normally configured with ``delay_embed_len = 1`` (the
per-step observation is then ``[growth_rate, (k_n0), (b)]``). Larger ``delay_embed_len``
still works without modification: ``num_inputs`` simply grows to
``delay_embed_len * (1 + k_n0_observation + b_observation)``.

Usage:
    from rlBacterialControl.envs.envs import EnvConfig, VariableNutrientEnv
    from rlBacterialControl.envs.cell_model import CellConfig
    from rlBacterialControl.agent.RNN_full import CDQL

    env = VariableNutrientEnv(EnvConfig(delay_embed_len=1, num_actions=2,
                                        b_actions=[0, 3.72], T_k_n0=..., k_n0_mean=2.55,
                                        sigma_kn0=0.1), CellConfig())
    agent = CDQL(env, rnn_type="LSTM", train_unroll_len=20, net_arch="encoder_decoder", use_gpu=False)
    agent.train(episodes=400, num_decisions=300, num_evals=5, folder_name="./Results")

Output (written under ``folder_name``):
    episode_<n>/q_1, q_2, q_target_1, q_target_2 : saved network checkpoints
    Eval/<episode>.jpg                            : sampled eval trajectories
    reward_Q_loss.jpg                             : training-curve summary (final episode)
    wandb logging of eval metrics (if wandb is installed)
"""

import os
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from joblib import Parallel, delayed
from scipy import signal
try:
    import wandb
except ImportError:
    # wandb is only used for training-time logging; eval/inference does not need it.
    wandb = None

from .replaybuffer_RNN import ReplayBuffer
from .deepQLnetwork_RNN import Model as ModelRNN
from .deepQLnetwork_RNN_encoder_decoder import Model as ModelEncoderDecoder

from ..envs.envs import BaseEnv
from ..utils.utils_figure_plot import plot_trajectory, plot_reward_Q_loss


EPS = 1e-10


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
        network_update_rate: float = 0.005,
        rnn_type: str = "LSTM",
        train_unroll_len: int = 20,
        net_arch: str = "rnn",
    ) -> None:
        '''
        Args:
            env: environment to train on (provides reset()/step()/info['log'])
            buffer_size: number of *sequences* stored in the replay buffer
            batch_size: number of sequences sampled per gradient step
            gamma: reward discount factor
            update_freq: frequency of target-Q soft updates per Q update
            train_freq: frequency of training (gradient updates) per env step
            gradient_steps: number of gradient steps to take per update
            use_gpu: whether to use gpu
            learning_rate: lr for updating Q networks based on Bellman residual
            network_update_rate: soft-update (tau) rate for the target networks
            rnn_type: recurrent cell, "LSTM" (carries h,c) or "GRU" (carries h)
            train_unroll_len: length of the stored training sequences (R2D2 unroll length)
            net_arch: Q-network architecture -- "rnn" (deepQLnetwork_RNN: stacked 3-layer
                RNN + linear head) or "encoder_decoder" (deepQLnetwork_RNN_encoder_decoder:
                MLP encoder -> 1-layer RNN -> MLP decoder)
        '''
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        assert (not use_gpu) or (self.device == torch.device('cuda'))

        self.env = env
        self.rnn_type = rnn_type
        self.train_unroll_len = train_unroll_len
        self.net_arch = net_arch

        num_inputs = self.env.delay_embed_len * (
            1 + self.env.k_n0_observation + self.env.b_observation
        )
        if net_arch == "rnn":
            ModelClass = ModelRNN
        elif net_arch == "encoder_decoder":
            ModelClass = ModelEncoderDecoder
        else:
            raise ValueError(f"Invalid net_arch: {net_arch!r} (expected 'rnn' or 'encoder_decoder')")
        self.model = ModelClass(self.device,
                                num_inputs=num_inputs,
                                num_actions=self.env.num_actions,
                                rnn_type=rnn_type)
        # The copied RNN Model hardcodes lr=1e-4 / tau=0.005; honor the package-style
        # learning_rate / network_update_rate args without modifying the copy.
        self.model.tau = network_update_rate
        self.model.q_optimizer_1 = Adam(self.model.q_1.parameters(), lr=learning_rate)
        self.model.q_optimizer_2 = Adam(self.model.q_2.parameters(), lr=learning_rate)

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
        self.training_iter = 0
        self.epsilon = None

    def _to_tensor(self, x):
        return torch.tensor(np.array(x)).float().to(self.device)

    def _save_data(self, folder_name, replay_buffer=False):
        os.makedirs(folder_name, exist_ok=True)
        # the copied RNN Model.save_networks does ``folder_name + "q_1"`` (string concat),
        # so pass a path with a trailing separator.
        self.model.save_networks(os.path.join(folder_name, ""))
        if replay_buffer:
            np.save(os.path.join(folder_name, "replaybuffer.npy"),
                    np.array(self.buffer.buffer, dtype=object))

    def load_data(self, folder_name, replay_buffer=False):
        self.model.load_networks(os.path.join(folder_name, ""))
        if replay_buffer:
            self.buffer.load_buffer(os.path.join(folder_name, "replaybuffer.npy"))

    def _get_action(self, obs, hidden_state, deterministic: bool = True, epsilon: float = 0.0):
        """Greedy (argmin-cost) action with epsilon exploration, advancing the RNN state.
        Args:
            obs: current single observation vector
            hidden_state: previous RNN hidden state (list; [None] to (re)initialize)
            deterministic: if False, take a random action with probability epsilon
            epsilon: exploration probability
        Returns:
            (action, next_hidden_state)
        """
        self.model.q_1.eval()
        with torch.no_grad():
            curr_state = self._to_tensor(obs).unsqueeze(0)  # (1, num_inputs)
            x = self.model.q_1({'obs': curr_state, 'prev_state': hidden_state}, inference=True)
            action = torch.argmin(x['logit'], dim=1).item()
        next_state = x['next_state']
        if (not deterministic) and (random.random() < epsilon):
            action = random.randrange(self.env.num_actions)
        self.model.q_1.train()
        return action, next_state

    def _update(self) -> None:
        """Clipped double-Q update over sampled sequences (R2D2-style, stored hidden states)."""
        if len(self.buffer) < self.batch_size:
            return

        self.model.q_1.train()
        self.model.q_2.train()
        self.model.q_target_1.eval()
        self.model.q_target_2.eval()

        for _ in range(self.gradient_steps):
            transitions = self.buffer.sample(self.batch_size)
            batch = self.buffer.transition(*zip(*transitions))

            state_batch = self._to_tensor(batch.state)             # (B, T, N)
            next_state_batch = self._to_tensor(batch.next_state)   # (B, T, N)
            # stored inits are length-1 lists from inference forward; unpack to B dicts
            hidden_state = [hidden for hidden, *_ in batch.hidden_state_init]
            next_hidden_state = [hidden for hidden, *_ in batch.next_hidden_state_init]

            action_batch = self._to_tensor(batch.action).transpose(0, 1).unsqueeze(2).to(torch.int64)
            reward_batch = self._to_tensor(batch.reward).transpose(0, 1).unsqueeze(2)
            terminal_batch = self._to_tensor(batch.terminal).transpose(0, 1).unsqueeze(2)

            with torch.no_grad():
                input_dict = {'obs': next_state_batch, 'prev_state': next_hidden_state}
                Q_next_1 = torch.min(self.model.q_target_1(input_dict)['logit'], dim=2)[0].unsqueeze(2)
                Q_next_2 = torch.min(self.model.q_target_2(input_dict)['logit'], dim=2)[0].unsqueeze(2)
                # max over the twin targets to avoid underestimation bias
                Q_next = torch.max(Q_next_1, Q_next_2)
                Q_expected = reward_batch + self.gamma * Q_next * (1 - terminal_batch)

            Q_1 = self.model.q_1({'obs': state_batch, 'prev_state': hidden_state})['logit'].gather(2, action_batch)
            Q_2 = self.model.q_2({'obs': state_batch, 'prev_state': hidden_state})['logit'].gather(2, action_batch)
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
                self.model.update_target_nn()

    def train(self, episodes: int, num_decisions: int, num_evals: int = 10, folder_name: str = "./") -> None:
        """Train the recurrent twin-Q networks.
        Args:
            episodes: number of training episodes
            num_decisions: max decision steps per episode
            num_evals: number of evaluation rollouts at each checkpoint
            folder_name: output directory for checkpoints/plots
        """
        T_eps = 300  # controls how fast exploration decays to exploitation
        epsilon_list = np.arange(episodes)
        epsilon_list = (-np.log10(epsilon_list / T_eps + EPS)).clip(0.05, 1)

        step_iter = 0
        for episode in range(episodes):
            obs, _ = self.env.reset()
            # prime the hidden state on the post-warmup observation so stored
            # hidden_state_init values are always valid (never [None]).
            _, hidden_state = self._get_action(obs, [None], deterministic=True)

            state_seq, action_seq, reward_seq, next_state_seq, terminal_seq = [], [], [], [], []
            it = 1
            hidden_state_init = hidden_state
            next_hidden_state_init = hidden_state

            for _ in range(num_decisions):
                state_seq.append(copy.deepcopy(obs))
                if it == 1:
                    hidden_state_init = hidden_state
                action, hidden_state = self._get_action(
                    obs, hidden_state, deterministic=False, epsilon=epsilon_list[episode])
                if it == 1:
                    next_hidden_state_init = hidden_state
                action_seq.append(action)

                obs_next, reward, terminated, truncated, _ = self.env.step(action)
                reward_seq.append(reward)
                next_state_seq.append(copy.deepcopy(obs_next))
                terminal_seq.append(float(terminated))
                obs = obs_next
                done = terminated or truncated

                if it < self.train_unroll_len:
                    it += 1
                    step_iter += 1
                    if step_iter % self.train_freq == 0:
                        self._update()
                    if done:
                        # drop the trailing partial sequence (batching needs fixed length)
                        break
                else:
                    self.buffer.push(state_seq, action_seq, reward_seq, next_state_seq,
                                     hidden_state_init, next_hidden_state_init, terminal_seq)
                    state_seq, action_seq, reward_seq, next_state_seq, terminal_seq = [], [], [], [], []
                    it = 1
                    step_iter += 1
                    if step_iter % self.train_freq == 0:
                        self._update()
                    if done:
                        break

            print(f"Episode {episode} completed.")
            if (episode % 10 == 0) or (episode == episodes - 1):
                self._save_data(os.path.join(folder_name, f"episode_{episode}"))
                self.evalulate(episode, num_decisions, num_evals, folder_name)

            if episode == episodes - 1:
                reward_ylabel = ("ave. eval reward per step" if self.env.reward_type == "log10_pop"
                                 else "ave. total eval reward")
                plot_reward_Q_loss(self.ave_rewards, self.std_rewards, self.grad_updates,
                                   self.loss, folder_name, self.ave_Q1, self.ave_Q2,
                                   self.ave_Q1_target, self.ave_Q2_target,
                                   reward_ylabel=reward_ylabel)

    def evalulate(self, episode: int, num_decisions: int, num_evals: int, folder_name: str) -> None:
        """Evaluate the model and log/plot metrics (mirrors MLP_full.CDQL.evalulate)."""
        self.model.q_1.eval()
        self.model.q_2.eval()

        extinct_times = []
        extinct_count = 0
        max_cross_corr_kn0 = []
        max_cross_corr_U = []
        lag_kn0 = []
        lag_U = []

        results = Parallel(n_jobs=10)(delayed(self.eval_step)(num_decisions) for _ in range(num_evals))
        agg_rewards_all, min_Q_values_all, terminated_all, _, info_all = zip(*results)
        ave_q1, ave_q2, ave_q1_target, ave_q2_target = np.mean(min_Q_values_all, axis=0)

        for terminated, info in zip(terminated_all, info_all):
            time, nutr, drug, _, damage = list(zip(*info['log']))[:5]

            drug = drug[(self.env.warm_up + 1):]
            nutr = nutr[(self.env.warm_up - self.env.delay_embed_len + 1):]
            damage = damage[self.env.warm_up + 1:]

            if np.std(drug) > 0 and np.std(nutr) > 0:
                cross_correlation(drug, nutr, max_cross_corr_kn0, lag_kn0)
            if np.std(drug) > 0 and np.std(damage) > 0:
                cross_correlation(drug, damage, max_cross_corr_U, lag_U)

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

        ave_ext_time = sum(extinct_times) / len(extinct_times) if len(extinct_times) > 0 else np.inf
        ave_max_cross_corr_kn0 = sum(max_cross_corr_kn0) / len(max_cross_corr_kn0) if len(max_cross_corr_kn0) > 0 else 0
        ave_corr_lag_kn0 = sum(lag_kn0) / len(lag_kn0) if len(lag_kn0) > 0 else 0
        ave_max_cross_corr_U = sum(max_cross_corr_U) / len(max_cross_corr_U) if len(max_cross_corr_U) > 0 else 0
        ave_corr_lag_U = sum(lag_U) / len(lag_U) if len(lag_U) > 0 else 0
        # metric name matches the aggregation: per-step mean for log10-pop, episode sum otherwise
        reward_metric_name = "ave reward per step" if self.env.reward_type == "log10_pop" else "ave total reward"
        if wandb is not None:
            wandb.log({
                "extinct_fraction": extinct_count / num_evals,
                "ave_ext_rate": 1 / ave_ext_time,
                "ave_max_cross_corr_kn0": ave_max_cross_corr_kn0,
                "ave_corr_lag_kn0": ave_corr_lag_kn0,
                "ave_max_cross_corr_U": ave_max_cross_corr_U,
                "ave_corr_lag_U": ave_corr_lag_U,
                reward_metric_name: np.mean(agg_rewards_all),
                "ave min Q1": ave_q1
            })
        plot_trajectory(random.sample(info_all, 5), episode, os.path.join(folder_name, "Eval"))

    def eval_step(self, num_decisions: int) -> tuple:
        """Greedy recurrent rollout; returns (sum_reward, per-net min-Q means, terminated, truncated, info)."""
        obs, _ = self.env.reset()
        _, hidden_state = self._get_action(obs, [None], deterministic=True)
        rewards = []
        Q_values = []
        terminated = truncated = False
        info = {}
        for _ in range(num_decisions):
            obs_t = self._to_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                x1 = self.model.q_1({'obs': obs_t, 'prev_state': hidden_state}, inference=True)
                action = torch.argmin(x1['logit'], dim=1).item()
                # evaluate all four nets at the current obs using the acting net's hidden
                # state (matches r2d2_RNN_RLmain eval bookkeeping).
                q_vals = [
                    x1['logit'].squeeze(0).cpu().numpy(),
                    self.model.q_2({'obs': obs_t, 'prev_state': hidden_state}, inference=True)['logit'].squeeze(0).cpu().numpy(),
                    self.model.q_target_1({'obs': obs_t, 'prev_state': hidden_state}, inference=True)['logit'].squeeze(0).cpu().numpy(),
                    self.model.q_target_2({'obs': obs_t, 'prev_state': hidden_state}, inference=True)['logit'].squeeze(0).cpu().numpy(),
                ]
            hidden_state = x1['next_state']
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            Q_values.append(q_vals)
            if terminated or truncated:
                break
        # log10-pop reward does not telescope -> report the per-step MEAN (length-normalized, so
        # rollouts of different length are comparable). Growth-rate reward telescopes to endpoints
        # (already length-independent) -> report the episode SUM, as originally.
        reward_agg = np.mean if self.env.reward_type == "log10_pop" else np.sum
        return reward_agg(rewards), np.array(Q_values).min(-1).mean(0), terminated, truncated, info


def cross_correlation(sig1, sig2, max_cross_corr, lag):
    n_points = len(sig1)
    cross_corr = signal.correlate(sig1 - np.mean(sig1), sig2 - np.mean(sig2), mode='full')
    cross_corr /= (np.std(sig1) * np.std(sig2) * n_points)  # Normalize
    max_cross_corr.append(np.max(cross_corr))
    lags = signal.correlation_lags(len(sig1), len(sig2), mode="full")
    lag.append(lags[np.argmax(cross_corr)])
