import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np


class Q(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        q_a = self.fc4(s)
        return q_a


class Model:
    def __init__(self, device, num_inputs, num_actions):
        self.device = device
        self.q_1 = Q(num_inputs, num_actions).to(device)
        self.q_target_1 = Q(num_inputs, num_actions).to(device)

        self.q_2 = Q(num_inputs, num_actions).to(device)
        self.q_target_2 = Q(num_inputs, num_actions).to(device)

        self.q_target_1.eval()
        self.q_target_2.eval()

        self.q_optimizer_1 = Adam(self.q_1.parameters(), lr=1e-4)
        self.q_optimizer_2 = Adam(self.q_2.parameters(), lr=1e-4)

        self._update(self.q_target_1, self.q_1)
        self._update(self.q_target_2, self.q_2)
        self.tau = 0.005
        self.grad_update_num = 0

    def _smaller_weights_last_layer(self, network, scale):
        """Updates the last layer with smaller weights
        Args:
            network: network to update
            scale: amount to scale down weights of last layer
        """
        last_layers = list(network.state_dict().keys())[-2:]
        for layer in last_layers:
            network.state_dict()[layer] /= scale

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

    def save_networks(self, folder_name):
        """
        Save Networks
        """
        torch.save({"model_state_dict": self.q_1.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_1.state_dict()
                    }, folder_name + "q_1")

        torch.save({"model_state_dict": self.q_2.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_2.state_dict()
                    }, folder_name + "q_2")

        torch.save({"model_state_dict": self.q_target_1.state_dict()},
                   folder_name + "q_target_1")

        torch.save({"model_state_dict": self.q_target_2.state_dict()},
                   folder_name + "q_target_2")

    def load_networks(self, folder_name):
        """Loads networks and optimizer state
        Args:
            folder_name: folder from which to load networks from
        """

        q_checkpoint_1 = torch.load(folder_name + "/q_1",
                                         map_location=self.device)
        self.q_1.load_state_dict(q_checkpoint_1["model_state_dict"])
        self.q_optimizer_1.load_state_dict(q_checkpoint_1[
            "optimizer_state_dict"])

        q_checkpoint_2 = torch.load(folder_name + "/q_2",
                                         map_location=self.device)
        self.q_2.load_state_dict(q_checkpoint_2["model_state_dict"])
        self.q_optimizer_2.load_state_dict(q_checkpoint_2[
            "optimizer_state_dict"])

        q_target_checkpoint_1 = torch.load(folder_name + "/q_target_1",
                                                map_location=self.device)
        self.q_target_1.load_state_dict(
            q_target_checkpoint_1["model_state_dict"])

        q_target_checkpoint_2 = torch.load(folder_name + "/q_target_2",
                                                map_location=self.device)
        self.q_target_2.load_state_dict(
            q_target_checkpoint_2["model_state_dict"])