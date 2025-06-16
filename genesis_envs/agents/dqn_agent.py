import torch
from torch.nn import functional as F

from ..agents.base import AgentInterface
from ..utils.typing_utils import Actions, Dones, Rewards, States


class DQNAgent(AgentInterface):
    def __init__(
        self,
        networks: dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        tau: float = 1.0,
        target_update_interval: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        self.networks = networks
        self.q_network = networks["q_network"]
        self.target_q_network = networks.get("target_q_network", None)
        if self.target_q_network is None:
            import copy

            self.target_q_network = copy.deepcopy(self.q_network)

        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.device = device
        self.update_count = 0

    def select_action(self, state: States, deterministic: bool = False) -> Actions:
        with torch.no_grad():
            q_values = self.q_network(state)

        if deterministic:
            action = torch.argmax(q_values, dim=-1)
        else:
            batch_size = q_values.size(0)
            random_actions = torch.randint(
                0, q_values.size(-1), (batch_size,), device=self.device
            )
            greedy_actions = torch.argmax(q_values, dim=-1)
            mask = torch.rand(batch_size, device=self.device) < self.epsilon
            action = torch.where(mask, random_actions, greedy_actions)
        return action

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_policy(
        self,
        states: States,
        next_states: States,
        actions: Actions,
        rewards: Rewards,
        dones: Dones,
    ) -> dict:
        q_loss = 0.0
        q_value_mean = 0.0
        target_q_value_mean = 0.0
        for i in range(states.size(0)):
            q_values = self.q_network(states[i])
            q_value = q_values.gather(1, actions[i].unsqueeze(dim=-1)).squeeze(dim=-1)

            with torch.no_grad():
                next_q_values = self.target_q_network(next_states[i])
                max_next_q_value = next_q_values.max(dim=-1)[0]
                target_q_value = rewards.squeeze(dim=-1) + (
                    self.discount_factor * max_next_q_value * (~dones[i])
                )

            loss = F.mse_loss(q_value, target_q_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_count += 1
            if self.update_count % self.target_update_interval == 0:
                for target_network_param, q_network_param in zip(
                    self.target_q_network.parameters(), self.q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        self.tau * q_network_param.data
                        + (1.0 - self.tau) * target_network_param.data
                    )

            q_loss += loss.item()
            q_value_mean += q_value.mean().item()
            target_q_value_mean += target_q_value.mean().item()

        self.decay_epsilon()

        q_loss /= states.size(0)
        q_value_mean /= states.size(0)
        target_q_value_mean /= states.size(0)

        logs = {
            "q_loss": q_loss,
            "q_value_mean": q_value_mean,
            "target_q_value_mean": target_q_value_mean,
            "epsilon": self.epsilon,
        }
        return logs
