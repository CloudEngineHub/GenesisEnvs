import torch
import pytest
from genesis_envs.utils.replay_buffer import ReplayBuffer

@pytest.fixture
def replay_buffer():
    return ReplayBuffer(buffer_size=5)

def test_add_to_replay_buffer(replay_buffer):
    state = torch.tensor([1.0, 2.0, 3.0])
    next_state = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([0])
    reward = torch.tensor([1.0])
    done = torch.tensor([False])
    
    replay_buffer.add(state, next_state, action, reward, done)
    
    assert len(replay_buffer.buffer) == 1, "Buffer size should be 1"
    
    states, next_states, actions, rewards, dones = replay_buffer.get_all().values()
    assert torch.equal(states[0], state), "State does not match"
    assert torch.equal(next_states[0], next_state), "Next State does not match"
    assert torch.equal(actions[0], action), "Action does not match"
    assert torch.equal(rewards[0], reward), "Reward does not match"
    assert torch.equal(dones[0], done), "Done does not match"

def test_sample_from_replay_buffer(replay_buffer):
    for i in range(5):
        state = torch.tensor([i, i+1, i+2])
        next_state = torch.tensor([i, i+1, i+2])
        action = torch.tensor([i % 2])
        reward = torch.tensor([i * 0.1])
        done = torch.tensor([i % 2 == 0])
        replay_buffer.add(state, next_state, action, reward, done)
    
    batch_size = 3
    states, next_state, actions, rewards, dones = replay_buffer.sample(batch_size)
    
    assert states.shape == (batch_size, 3), f"States shape should be ({batch_size}, 3)"
    assert actions.shape == (batch_size, 1), f"Actions shape should be ({batch_size}, 1)"
    assert rewards.shape == (batch_size, 1), f"Rewards shape should be ({batch_size}, 1)"
    assert dones.shape == (batch_size, 1), f"Dones shape should be ({batch_size}, 1)"

def test_buffer_overflow(replay_buffer):
    for i in range(6):
        state = torch.tensor([i, i+1, i+2])
        next_state = torch.tensor([i, i+1, i+2])
        action = torch.tensor([i % 2])
        reward = torch.tensor([i * 0.1])
        done = torch.tensor([i % 2 == 0])
        replay_buffer.add(state, next_state, action, reward, done)
    
    assert len(replay_buffer.buffer) == 5, "Buffer size should be 5"

def test_get_all_data(replay_buffer):
    for i in range(5):
        state = torch.tensor([i, i+1, i+2])
        next_state = torch.tensor([i, i+1, i+2])
        action = torch.tensor([i % 2])
        reward = torch.tensor([i * 0.1])
        done = torch.tensor([i % 2 == 0])
        replay_buffer.add(state, next_state, action, reward, done)
    
    data = replay_buffer.get_all()
    
    assert data["states"].shape == (5, 3), "States shape should be (5, 3)"
    assert data["next_states"].shape == (5, 3), "States shape should be (5, 3)"
    assert data["actions"].shape == (5, 1), "Actions shape should be (5, 1)"
    assert data["rewards"].shape == (5, 1), "Rewards shape should be (5, 1)"
    assert data["dones"].shape == (5, 1), "Dones shape should be (5, 1)"
