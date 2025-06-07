# replay_buffer.py
from collections import deque
from typing import Tuple

import torch


class ReplayBuffer:
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        self.buffer.append((state, next_state, action, reward, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        states, next_states, actions, rewards, dones = zip(
            *[self.buffer[i] for i in indices]
        )
        states = torch.cat(states, dim=1)
        next_states = torch.cat(next_states, dim=1)
        actions = torch.cat(actions, dim=1)
        rewards = torch.cat(rewards, dim=1)
        dones = torch.cat(dones, dim=1)
        return states, next_states, actions, rewards, dones

    def get_all(self):
        states, next_states, actions, rewards, dones = zip(*self.buffer)
        states = torch.cat(states, dim=1)
        next_states = torch.cat(next_states, dim=1)
        actions = torch.cat(actions, dim=1)
        rewards = torch.cat(rewards, dim=1)
        dones = torch.cat(dones, dim=1)
        return states, next_states, actions, rewards, dones
