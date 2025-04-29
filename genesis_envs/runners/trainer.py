import os
import time
import torch
from typing import Tuple, Dict, List, Optional
from loguru import logger

from genesis_envs.envs.base import EnvInterface
from genesis_envs.agents.base import AgentInterface
from genesis_envs.utils.typing_utils import States, Actions, Rewards, Dones


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
        states, actions, rewards, dones = [], [], [], []

        for step in range(self.horizon):
            action = self.agent.select_action(
                state, deterministic=self.deterministic_action
            )
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

    def train_loop(self):
        start_time = time.time()
        for epoch in range(self.num_epochs):
            states, actions, rewards, dones = self.rollout()
            logs = self.agent.update_policy(states, actions, rewards, dones)

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
                check_point_save_path = os.path.join(
                    self.checkpoint_save_path, f"checkpoint_{(epoch + 1):04d}.pth"
                )
                logger.info(f"save checkpoint: {check_point_save_path}")
                torch.save(self.agent.network.state_dict(), check_point_save_path)

        check_point_save_path = os.path.join(
            self.checkpoint_save_path, "checkpoint_final.pth"
        )
        logger.info(f"save checkpoint: {check_point_save_path}")
        torch.save(self.agent.network.state_dict(), check_point_save_path)
