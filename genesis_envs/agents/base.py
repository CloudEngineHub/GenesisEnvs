from abc import ABC, abstractmethod

import torch

from ..utils.typing_utils import Actions, Dones, Rewards, States


class AgentInterface(ABC):
    networks: dict[str, torch.nn.Module]
    device: torch.device

    @abstractmethod
    def select_action(self, state: States, deterministic: bool = False) -> Actions:
        pass

    @abstractmethod
    def update_policy(
        self,
        states: States,
        actions: Actions,
        rewards: Rewards,
        dones: Dones,
    ) -> torch.Tensor:
        pass
