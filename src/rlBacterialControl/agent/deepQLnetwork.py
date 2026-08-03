import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import os
import numpy as np


class Q(nn.Module):
    """Q network parameterization
    """
    def __init__(self, num_inputs: int, num_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, s: torch.Tensor):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        q_a = self.fc4(s)
        return q_a


class Model(object):
    """Defining Q network and optimizer functions
        using Clipped Double Q-learning, so requires two target and two local networks
    """

    def __init__(self, device, num_inputs, num_actions, learning_rate, tau):
        self.device = device
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.q_1 = Q(num_inputs, num_actions).to(device)
        self.q_target_1 = Q(num_inputs, num_actions).to(device)

        self.q_2 = Q(num_inputs, num_actions).to(device)
        self.q_target_2 = Q(num_inputs, num_actions).to(device)
        self.q_networks: list[nn.Module] = [self.q_1, self.q_2, self.q_target_1, self.q_target_2]

        self.q_target_1.eval()
        self.q_target_2.eval()

        self.q_optimizer_1 = Adam(self.q_1.parameters(), lr=learning_rate)
        self.q_optimizer_2 = Adam(self.q_2.parameters(), lr=learning_rate)

        self._update(self.q_target_1, self.q_1)
        self._update(self.q_target_2, self.q_2)
        self.tau = tau

    def get_action(self, obs, deterministic: bool = True, epsilon: float = 0):
        """Returns action based on epsilon-greedy policy
        Args:
            obs: obs of system
            deterministic: if True, takes greedy action
            epsilon: epsilon value for random action selection
        """
        assert len(obs) == self.num_inputs
        self.q_1.eval()

        if not deterministic and np.random.rand() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                curr_obs = torch.tensor(obs).float().to(self.device)
                action = torch.argmin(self.q_1(curr_obs)).item()
        return action

    def _update(self, target, local):
        """
        Sets the parameters of target network to be that of local network, only used at initialization
        """
        target.load_state_dict(local.state_dict())

    def _soft_update(self, target, local):
        """Soft update of parameters in target network via Polyak averaging (EMA) from local
        Args:
            target: target network
            local: local network
        """
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(target_param.data
                                    * (1.0 - self.tau)
                                    + param.data * self.tau)

    def update_target_nn(self):
        """
        Soft update to both target networks
        """
        self._soft_update(self.q_target_1, self.q_1)
        self._soft_update(self.q_target_2, self.q_2)

    def save_networks(self, folder_name="./"):
        """
        Save networks to folder_name
        """

        torch.save({"model_state_dict": self.q_1.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_1.state_dict()
                    }, os.path.join(folder_name, "q_1"))

        torch.save({"model_state_dict": self.q_2.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_2.state_dict()
                    }, os.path.join(folder_name, "q_2"))

        torch.save({"model_state_dict": self.q_target_1.state_dict()},
                   os.path.join(folder_name, "q_target_1"))

        torch.save({"model_state_dict": self.q_target_2.state_dict()},
                   os.path.join(folder_name, "q_target_2"))

    def load_networks(self, folder_name="./"):
        """
        Loads networks and optimizer state from folder_name
        """

        q_checkpoint_1 = torch.load(os.path.join(folder_name, "q_1"),
                                         map_location=self.device)
        self.q_1.load_state_dict(q_checkpoint_1["model_state_dict"])
        self.q_optimizer_1.load_state_dict(q_checkpoint_1[
            "optimizer_state_dict"])

        q_checkpoint_2 = torch.load(os.path.join(folder_name, "q_2"),
                                         map_location=self.device)
        self.q_2.load_state_dict(q_checkpoint_2["model_state_dict"])
        self.q_optimizer_2.load_state_dict(q_checkpoint_2[
            "optimizer_state_dict"])

        q_target_checkpoint_1 = torch.load(os.path.join(folder_name, "q_target_1"),
                                                map_location=self.device)
        self.q_target_1.load_state_dict(
            q_target_checkpoint_1["model_state_dict"])

        q_target_checkpoint_2 = torch.load(os.path.join(folder_name, "q_target_2"),
                                                map_location=self.device)
        self.q_target_2.load_state_dict(
            q_target_checkpoint_2["model_state_dict"])