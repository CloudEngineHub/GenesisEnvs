import os
import torch
from typing import Tuple

from genesis_envs.envs.base import EnvInterface
from genesis_envs.agents.base import AgentInterface
from genesis_envs.utils.typing_utils import States, Actions, Rewards, Dones


class GenesisEnvTrainer:
    def __init__(
        self,
        env: EnvInterface,
        agent: AgentInterface,
        device: torch.device = torch.device("cpu"),
    ):
        self.env = env
        self.agent = agent
        self.device = device

    def rollout(self, max_step: int = 50) -> Tuple[States, Actions, Rewards, Dones]:
        state = self.env.reset()
        total_reward = torch.zeros(self.env.num_envs, device=self.device)
        done_array = torch.tensor([False] * self.env.num_envs, device=self.device)
        states, actions, rewards, dones = [], [], [], []

        for step in range(max_step):
            action = self.agent.select_action(state)
            next_state, reward, done = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            total_reward += reward.detach()
            done_array = torch.logical_or(done_array, done)
            if done_array.all():
                break

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        return states, actions, rewards, dones

    def train_loop(self, num_epochs: int = 500, max_rollout_step: int = 50):
        for epoch in range(num_epochs):
            states, actions, rewards, dones = self.rollout(max_step=max_rollout_step)
            logs = self.agent.update_policy(states, actions, rewards, dones)
            print(epoch, logs)

            if epoch % 10 == 0:
                print("save checkpoint")
                torch.save(
                    self.agent.actor_critic_model.state_dict(), "actor_critic_model.pth"
                )
