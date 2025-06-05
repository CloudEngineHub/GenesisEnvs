import torch

from genesis_envs.models import MLP


def test_mlp():
    mlp_model = MLP(16, 48, 32, 3)
    output = mlp_model(torch.rand(4, 16))
    assert output.shape == torch.Size([4, 32])
