from abc import ABC, abstractmethod
from typing import Tuple

import genesis as gs

from ..utils.typing_utils import Actions, Dones, Rewards, States


class EnvInterface(ABC):
    @abstractmethod
    def build_env(self) -> gs.Scene:
        pass

    @abstractmethod
    def reset(self) -> Tuple[States, Rewards, Dones]:
        pass

    @abstractmethod
    def step(self, actions: Actions) -> States:
        pass
