import torch
from genesis_envs.utils.reward_utils import compute_discounted_rewards


def test_reward_utils():
    num_steps, num_envs = 50, 10
    gamma = 0.9

    rewards = torch.rand(num_steps, num_envs)
    dones = torch.randint(low=0, high=2, size=(num_steps, num_envs), dtype=torch.bool)

    discounted_rewards = compute_discounted_rewards(rewards, dones, gamma)
    assert discounted_rewards.shape == torch.Size([num_steps, num_envs])

    discounted_rewards = compute_discounted_rewards(
        rewards, dones, gamma, normalized=True
    )
    assert discounted_rewards.shape == torch.Size([num_steps, num_envs])

    discounted_rewards = compute_discounted_rewards(
        rewards.tolist(), dones.tolist(), gamma, normalized=True
    )
    assert discounted_rewards.shape == torch.Size([num_steps, num_envs])
