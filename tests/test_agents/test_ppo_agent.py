import torch

from genesis_envs.agents import PPOAgent
from genesis_envs.models import MLP


def test_ppo_agent():
    policy_model = {
        "actor_critic_model": MLP(16, 48, 32, 3),
    }
    optimizer = torch.optim.Adam(
        policy_model["actor_critic_model"].parameters(), lr=0.001
    )
    agent = PPOAgent(policy_model, optimizer, discount_factor=0.9, clip_epsilon=0.2)

    action = agent.select_action(torch.rand(4, 16))
    assert action.shape == torch.Size([4])

    loss = agent.update_policy(
        states=torch.rand(50, 10, 16),  # [num_steps, num_envs, num_states]
        next_states=torch.rand(50, 10, 16),
        actions=torch.randint(low=0, high=10, size=(50, 10)),
        rewards=torch.rand(50, 10),
        dones=torch.randint(low=0, high=2, size=(50, 10), dtype=torch.bool),
    )
    print(loss)
