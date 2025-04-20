import os
import torch
import genesis as gs
from genesis_envs.envs import GraspEnv
from genesis_envs.agents import PPOAgent
from genesis_envs.models import MLP
from genesis_envs.tools.trainer import GenesisEnvTrainer


NUM_ENVS = 1
EVAL = NUM_ENVS == 1

gs.init(backend=gs.cpu)
env = GraspEnv(vis=True, num_envs=NUM_ENVS)
actor_critic_model = MLP(6, 64, 9, 3)

if EVAL and os.path.exists("actor_critic_model.pth"):
    print("load_checkpoint")
    actor_critic_model.load_state_dict(torch.load("actor_critic_model.pth"))

agent = PPOAgent(
    actor_critic_model=actor_critic_model,
    discount_factor=0.99,
    clip_epsilon=0.2,
)
trainer = GenesisEnvTrainer(
    env=env,
    agent=agent,
)

if EVAL:
    actor_critic_model.eval()
    while True:
        trainer.rollout(50)
else:
    trainer.train_loop(500, 50)
