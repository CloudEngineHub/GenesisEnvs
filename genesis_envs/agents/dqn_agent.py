import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from ..agents.base import AgentInterface
from ..utils.reward_utils import compute_discounted_rewards
from ..utils.typing_utils import Actions, Dones, Rewards, States


class PPOAgent(AgentInterface):
    def __init__(
        self,
        network: dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        epsilon: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.network = network
        self.epsilon = epsilon

        self.device = device

        self.optimizer = optimizer

    def select_action(self, state: States, deterministic: bool = False) -> Actions:
        with torch.no_grad():
            q_values = self.network(state)

        if deterministic:
            action = torch.argmax(q_values, dim=-1)
        else:
            num_envs = q_values.size(0)
            random_action = torch.randint(0, q_values.size(1), (num_envs,)).to(
                self.device
            )
            greedy_action = torch.argmax(q_values, dim=1)
            action = torch.where(
                torch.rand(num_envs < self.epsilon), random_action, greedy_action
            )
        return action

    def update_policy(
        self,
        states: States,
        actions: Actions,
        rewards: Rewards,
        dones: Dones,
    ) -> torch.Tensor:
        logs = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
        }

        discounted_rewards = compute_discounted_rewards(
            rewards,
            dones,
            self.discount_factor,
            normalized=self.normalize_discounted_reward,
            device=self.device,
        )

        with torch.no_grad():
            output = self.network(states)
            logits_old, value_old = output[..., :-1].detach(), output[..., -1].detach()
            dist_old = Categorical(logits=logits_old)
            log_probs_old = dist_old.log_prob(actions)

        advantages = discounted_rewards - value_old
        normalized_advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (
            advantages.std(dim=0, keepdim=True) + 1e-8
        )

        # TODO: support minibatch update
        # TODO: support gradient clipping
        for t in range(self.num_update_steps):
            output = self.network(states)
            logits_new, values = output[..., :-1], output[..., -1]
            dist_new = Categorical(logits=logits_new)

            ratio = torch.exp(dist_new.log_prob(actions) - log_probs_old)

            surrogate_loss1 = ratio * normalized_advantages
            surrogate_loss2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * normalized_advantages
            )
            value_loss = F.mse_loss(values, discounted_rewards)
            entropy = dist_new.entropy().mean()

            policy_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()
            total_loss = (
                policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )

            logs["policy_loss"].append(policy_loss.item())
            logs["value_loss"].append(value_loss.item())
            logs["entropy"].append(entropy.item())
            logs["total_loss"].append(total_loss.item())

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        logs["policy_loss"] = torch.tensor(logs["policy_loss"]).mean().item()
        logs["value_loss"] = torch.tensor(logs["value_loss"]).mean().item()
        logs["total_loss"] = torch.tensor(logs["total_loss"]).mean().item()

        logs["entropy"] = torch.tensor(logs["entropy"]).mean().item()
        logs["discounted_rewards"] = discounted_rewards.detach().mean().item()
        logs["adv_mean"] = advantages.mean().item()
        logs["adv_std"] = advantages.std().item()

        return logs
