import torch
from typing import NewType

States = NewType("States", torch.Tensor)
Actions = NewType("Actions", torch.Tensor)
Rewards = NewType("Rewards", torch.Tensor)
Dones = NewType("Dones", torch.Tensor)
