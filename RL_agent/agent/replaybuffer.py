import random
import numpy as np
from collections import namedtuple


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = int(0)
        self.transition = namedtuple("Transition", ("state", "action", \
            "reward", "next_state", "terminated"))

    def push(self, state, action, reward, next_state, terminated):
        """Adds new tuple of (state, action, reward, next_state) sample to replay buffer
        If buffer is full (i.e. more than self. capacity) add new samples to beginning of
        replay buffer
        Args:
            state: state of system
            action: current action
            reward: observed reward
            next_state: next state of system
            terminated: denotes if next state is terminated or not (1=terminated state, 0 otherwise)
        """

        to_add = [state, action, reward, next_state, terminated]
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.transition(*to_add)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        """Samples batch_size samples from replay buffer
        Args:
            batch_size: batch size of sample
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return (len(self.buffer))

    def load_buffer(self, filename):
        self.buffer = np.load(filename, allow_pickle=True).tolist()
        self.position = len(self.buffer)