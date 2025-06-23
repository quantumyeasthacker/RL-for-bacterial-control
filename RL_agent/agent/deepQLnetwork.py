import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import os
import numpy as np


class Q(nn.Module):
    def __init__(self, num_inputs, num_actions, dim_context):
        super().__init__()
        self.hidden_dim = 64
        self.context = nn.Linear(dim_context, self.hidden_dim**2)
        self.fc1 = nn.Linear(num_inputs, self.hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, num_actions)

    def forward(self, s, c):
        s = F.relu(self.fc1(s))
        c = self.context(c)
        c = c.reshape(c.shape[0], self.hidden_dim, self.hidden_dim)
        s = torch.matmul(s.unsqueeze(1), c)
        # s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        q_a = self.fc4(s)
        return q_a.squeeze(1)


class Model(object):
    def __init__(self, device, num_inputs, num_actions, dim_context, learning_rate=1e-4):
        self.device = device
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.dim_context = dim_context
        self.q_1 = Q(num_inputs, num_actions, dim_context).to(device)
        self.q_target_1 = Q(num_inputs, num_actions, dim_context).to(device)

        self.q_2 = Q(num_inputs, num_actions, dim_context).to(device)
        self.q_target_2 = Q(num_inputs, num_actions, dim_context).to(device)
        self.q_networks: list[nn.Module] = [self.q_1, self.q_2, self.q_target_1, self.q_target_2]

        self.q_target_1.eval()
        self.q_target_2.eval()

        self.q_optimizer_1 = Adam(self.q_1.parameters(), lr=learning_rate)
        self.q_optimizer_2 = Adam(self.q_2.parameters(), lr=learning_rate)

        self._update(self.q_target_1, self.q_1)
        self._update(self.q_target_2, self.q_2)
        self.tau = 0.005

    def get_action(self, obs, context, deterministic: bool = True, epsilon: float = 0):
        """Returns action based on epsilon-greedy policy
        Args:
            obs: obs of system
            context: context of system
            deterministic: if True, chooses greedy action
            epsilon: epsilon value
        """
        assert len(obs) == self.num_inputs
        context = [context] if isinstance(context,(float,int)) else context
        assert len([context]) == self.dim_context
        self.q_1.eval()
        self.q_2.eval()

        if not deterministic and np.random.rand() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                curr_obs = torch.tensor(obs).unsqueeze(0).float().to(self.device)
                curr_context = torch.tensor(context).unsqueeze(0).float().to(self.device)
                # action = torch.argmin(self.q_1(curr_obs), dim=-1) # for batch running purposes
                action = torch.argmin(self.q_1(curr_obs, curr_context)).item()
            # self.q_1.train()
        return action


    def _update(self, target, local):
        """Set the parametrs of target network to be that of local network
        Args:
            target: target network
            local: local network
        """
        target.load_state_dict(local.state_dict())

    def _soft_update(self, target, local):
        """Soft update of parameters in target Networks
        """
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(target_param.data
                                    * (1.0 - self.tau)
                                    + param.data * self.tau)

    def update_target_nn(self):
        self._soft_update(self.q_target_1, self.q_1)
        self._soft_update(self.q_target_2, self.q_2)

    def save_networks(self, folder_name="./"):
        """
        Save Networks
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
        """Loads networks and optimizer state
        Args:
            folder_name: folder from which to load networks from
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