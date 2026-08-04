"""Architecture-agnostic twin-Q container shared by the MLP and recurrent agents.

Clipped double Q-learning needs four networks -- two local (``q_1``, ``q_2``) and two
target (``q_target_1``, ``q_target_2``) -- plus one Adam optimizer per local network,
Polyak (soft) target updates, and checkpoint save/load. None of that bookkeeping depends
on how the Q-function is parameterized, so it lives here once and is shared by:
  * ``.deepQLnetwork``                      (MLP ``Q``),
  * ``.deepQLnetwork_RNN``                  (stacked RNN + linear head ``DRQN``),
  * ``.deepQLnetwork_RNN_encoder_decoder``  (MLP encoder -> RNN -> MLP decoder ``DRQN``).

Each of those modules subclasses ``TwinQModel`` and passes a *network factory*: a
zero-argument callable returning one fresh network. The base calls it four times, so the
four networks get independent parameters (targets are then synced to their local twin).

Usage:
    from .twin_q_model import TwinQModel

    class Model(TwinQModel):
        def __init__(self, device, num_inputs, num_actions, learning_rate, tau):
            self.num_inputs = num_inputs
            self.num_actions = num_actions
            super().__init__(device, lambda: Q(num_inputs, num_actions), learning_rate, tau)

Checkpoint layout (written by ``save_networks``, read by ``load_networks``):
    <folder_name>/q_1        : {"model_state_dict", "optimizer_state_dict"}
    <folder_name>/q_2        : {"model_state_dict", "optimizer_state_dict"}
    <folder_name>/q_target_1 : {"model_state_dict"}
    <folder_name>/q_target_2 : {"model_state_dict"}
"""

import os
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Adam


class TwinQModel(object):
    """Twin local + twin target Q-networks with optimizers, soft updates and save/load.

    Subclasses supply the architecture via ``net_factory``; this class never inspects
    the networks it builds, so it works for both feed-forward and recurrent Q-functions.
    """

    def __init__(self, device, net_factory: Callable[[], nn.Module],
                 learning_rate: float, tau: float):
        """
        Args:
            device: torch device the networks are placed on
            net_factory: zero-arg callable returning one fresh (un-synced) Q-network;
                called four times, once per network
            learning_rate: lr for the Adam optimizers on the two local networks
            tau: Polyak coefficient for the target-network soft updates
        """
        self.device = device
        self.q_1 = net_factory().to(device)
        self.q_target_1 = net_factory().to(device)

        self.q_2 = net_factory().to(device)
        self.q_target_2 = net_factory().to(device)
        self.q_networks: list[nn.Module] = [self.q_1, self.q_2, self.q_target_1, self.q_target_2]

        self.q_target_1.eval()
        self.q_target_2.eval()

        self.q_optimizer_1 = Adam(self.q_1.parameters(), lr=learning_rate)
        self.q_optimizer_2 = Adam(self.q_2.parameters(), lr=learning_rate)

        self._update(self.q_target_1, self.q_1)
        self._update(self.q_target_2, self.q_2)
        self.tau = tau

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
