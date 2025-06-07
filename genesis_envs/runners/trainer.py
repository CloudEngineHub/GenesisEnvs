import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from genesis_envs.agents.base import AgentInterface
from genesis_envs.envs.base import EnvInterface
from genesis_envs.utils.typing_utils import Actions, Dones, Rewards, States
from genesis_envs.utils.replay_buffer import ReplayBuffer


def format_log_string(logs: Dict, selected_keys: Optional[List[str]] = None) -> str:
    if selected_keys is None:
        selected_keys = logs.keys()

    return ", ".join([f"{k.replace('_', ' ')}: {v:.4f}" for k, v in logs.items()])


class GenesisEnvTrainer:
    def __init__(
        self,
        env: EnvInterface,
        agent: AgentInterface,
        exp_name: str | os.PathLike,
        horizon: int = 50,
        num_epochs: int = 500,
        replay_buffer_size: int = 1,
        replay_buffer_sample_batch_size: int = 1,
        deterministic_action: bool = False,
        output_dir: str | os.PathLike = "outputs",
        log_every_epochs: int = 1,
        save_checkpoint_every_epochs: int = 10,
        device: torch.device | str = torch.device("cpu"),
    ):
        self.env = env
        self.agent = agent
        self.exp_name = exp_name
        self.horizon = horizon
        self.num_epochs = num_epochs
        self.deterministic_action = deterministic_action
        self.output_dir = os.path.join(output_dir, self.exp_name)
        self.checkpoint_save_path = os.path.join(self.output_dir, "checkpoints")
        self.log_every_epochs = log_every_epochs
        self.save_checkpoint_every_epochs = save_checkpoint_every_epochs
        self.device = device

        self.replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size)
        self.replay_buffer_sample_batch_size = replay_buffer_sample_batch_size

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.checkpoint_save_path):
            os.makedirs(self.checkpoint_save_path, exist_ok=True)

    def rollout(self) -> Tuple[States, Actions, Rewards, Dones]:
        state = self.env.reset()
        total_reward = torch.zeros(self.env.num_envs, device=self.device)
        done_array = torch.tensor([False] * self.env.num_envs, device=self.device)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for step in range(self.horizon):
            action = self.agent.select_action(
                state, deterministic=self.deterministic_action
            )
            next_state, reward, done = self.env.step(action)

            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            total_reward += reward.detach()
            done_array = torch.logical_or(done_array, done)

            if done_array.all():
                break

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        return states, next_states, actions, rewards, dones

    def save_checkpoint(self, checkpoint_name: str):
        check_point_save_path = os.path.join(self.checkpoint_save_path, checkpoint_name)
        logger.info(f"save checkpoint: {check_point_save_path}")

        networks_checkpoint = {}
        for network_name in self.agent.networks:
            networks_checkpoint[network_name] = self.agent.networks[
                network_name
            ].state_dict()
        torch.save(networks_checkpoint, check_point_save_path)

    def train_loop(self):
        start_time = time.time()
        for epoch in range(self.num_epochs):
            states, next_states, actions, rewards, dones = self.rollout()

            self.replay_buffer.add(
                state=states,
                next_state=next_states,
                action=actions,
                reward=rewards,
                done=dones,
            )
            states, next_states, actions, rewards, dones = self.replay_buffer.sample(
                self.replay_buffer_sample_batch_size
            )
            logs = self.agent.update_policy(
                states, next_states, actions, rewards, dones
            )

            elapsed_time = time.time() - start_time
            progress = (epoch + 1) / self.num_epochs
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            percent_complete = progress * 100

            log_string = format_log_string(logs)
            log_string = (
                f"Epoch [{epoch + 1}/{self.num_epochs}] - {log_string} | "
                f"Elapsed: {elapsed_str} Remaining: {remaining_str} ({percent_complete:.2f}%)"
            )
            if (epoch + 1) % self.log_every_epochs == 0:
                logger.info(log_string)

            if (epoch + 1) % self.save_checkpoint_every_epochs == 0:
                self.save_checkpoint(f"checkpoint_{(epoch + 1):04d}.pth")

        self.save_checkpoint("checkpoint_final.pth")
