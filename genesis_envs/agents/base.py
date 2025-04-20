import torch
from abc import ABC, abstractmethod
from ..utils.typing_utils import States, Actions, Rewards, Dones


class AgentInterface(ABC):
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
