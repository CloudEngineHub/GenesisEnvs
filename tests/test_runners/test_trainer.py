import torch
from genesis_envs.envs import GraspEnv
from genesis_envs.agents import PPOAgent
from genesis_envs.models import MLP
from genesis_envs.runners.trainer import GenesisEnvTrainer


def test_trainer():
    env = GraspEnv(vis=False)
    network = MLP(6, 16, 8, 3)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    agent = PPOAgent(
        network=network,
        optimizer=optimizer,
        discount_factor=0.99,
        clip_epsilon=0.2,
    )
    trainer = GenesisEnvTrainer(
        env=env,
        agent=agent,
        horizon=5,
        num_epochs=3,
        exp_name="pytest_exp",
    )
    states, actions, rewards, dones = trainer.rollout()
    trainer.train_loop()
