# replay_buffer.py
import torch
from collections import deque
from typing import Tuple


class ReplayBuffer:
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        self.buffer.append((state, action, reward, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        states, actions, rewards, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(dones),
        )

    def get_all(self):
        states, actions, rewards, dones = zip(*self.buffer)
        return {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "rewards": torch.stack(rewards),
            "dones": torch.stack(dones),
        }
