import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from .twin_q_model import TwinQModel


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


class Model(TwinQModel):
    """Twin MLP Q-networks (clipped double Q-learning); see ``.twin_q_model.TwinQModel``
        for the shared optimizer / soft-update / save-load machinery.
    """

    def __init__(self, device, num_inputs, num_actions, learning_rate, tau):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        super().__init__(device,
                         lambda: Q(num_inputs, num_actions),
                         learning_rate, tau)

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

