from genesis_envs.envs import GraspEnv
from genesis_envs.agents import PPOAgent
from genesis_envs.models import MLP
from genesis_envs.runners.trainer import GenesisEnvTrainer


def test_trainer():
    env = GraspEnv(vis=False)
    agent = PPOAgent(
        actor_critic_model=MLP(6, 16, 8, 3),
        discount_factor=0.99,
        clip_epsilon=0.2,
    )
    trainer = GenesisEnvTrainer(
        env=env,
        agent=agent,
        exp_name="pytest_exp",
    )
    states, actions, rewards, dones = trainer.rollout(horizon=3)
    trainer.train_loop(num_epochs=3, max_rollout_step=5)
