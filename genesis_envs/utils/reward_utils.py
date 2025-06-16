from typing import List, Union

import torch


def compute_discounted_rewards(
    rewards: Union[List[List[float]], torch.Tensor],
    dones: Union[List[List[bool]], torch.Tensor],
    gamma: float,
    normalized: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute discounted rewards for multiple parallel environments.

    Args:
        rewards (Tensor or nested list): [num_steps, num_envs] shaped reward matrix.
        dones (Tensor or nested list): [num_steps, num_envs] done flags.
        gamma (float): Discount factor.
        normalized (bool): If True, normalize discounted rewards for each environment.
        device (torch.device): Target device.

    Returns:
        Tensor: Discounted rewards of shape [num_steps, num_envs]
    """
    # TODO: support GAE
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if not torch.is_tensor(dones):
        dones = torch.tensor(dones, dtype=torch.bool)

    rewards = rewards.to(device)
    dones = dones.to(device)

    num_steps, num_envs = rewards.shape[:2]
    discounted_rewards = torch.zeros_like(rewards)

    R = torch.zeros(num_envs, dtype=torch.float32, device=device)

    for t in reversed(range(num_steps)):
        R = rewards[t] + gamma * R * (~dones[t])
        discounted_rewards[t] = R

    if normalized:
        # Normalize across time for each environment
        mean = discounted_rewards.mean(dim=0, keepdim=True)
        std = discounted_rewards.std(dim=0, keepdim=True) + 1e-8
        discounted_rewards = (discounted_rewards - mean) / std

    return discounted_rewards
