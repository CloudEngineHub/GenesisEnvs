import torch
import genesis as gs
from genesis_envs.envs.grasp_env import GraspEnv

def test_grasp_env():
    env = GraspEnv()
    env.reset()

    for i in range(8):
        env.step(torch.tensor([i], dtype=torch.int64))
