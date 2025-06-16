import pytest
import torch

from genesis_envs.utils.replay_buffer import ReplayBuffer


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(buffer_size=5)


def make_batch(horizon, num_envs, num_actions):
    state = torch.randn(horizon, num_envs, num_actions)
    next_state = torch.randn(horizon, num_envs, num_actions)
    action = torch.randint(0, 2, (horizon, num_envs, num_actions))
    reward = torch.randn(horizon, num_envs, num_actions)
    done = torch.randint(0, 2, (horizon, num_envs, num_actions)).bool()
    return state, next_state, action, reward, done


def test_add_to_replay_buffer(replay_buffer):
    horizon, num_envs, num_actions = 2, 3, 4
    buffer_size = 5
    batch_size = 10
    for i in range(batch_size):
        state, next_state, action, reward, done = make_batch(
            horizon, num_envs, num_actions
        )
        replay_buffer.add(state, next_state, action, reward, done)

    assert len(replay_buffer.buffer) == buffer_size

    data = replay_buffer.get_all()
    states, next_states, actions, rewards, dones = data
    assert states.shape == (horizon, buffer_size * num_envs, num_actions), (
        "State shape mismatch"
    )
    assert next_states.shape == (horizon, buffer_size * num_envs, num_actions), (
        "Next state shape mismatch"
    )
    assert actions.shape == (horizon, buffer_size * num_envs, num_actions), (
        "Action shape mismatch"
    )
    assert rewards.shape == (horizon, buffer_size * num_envs, num_actions), (
        "Reward shape mismatch"
    )
    assert dones.shape == (horizon, buffer_size * num_envs, num_actions), (
        "Done shape mismatch"
    )


def test_sample_from_replay_buffer(replay_buffer):
    horizon, num_envs, num_actions = 5, 2, 3
    for i in range(horizon):
        state, next_state, action, reward, done = make_batch(
            horizon, num_envs, num_actions
        )
        replay_buffer.add(state, next_state, action, reward, done)

    batch_size = 3
    batch = replay_buffer.sample(batch_size)
    states, next_states, actions, rewards, dones = batch

    assert states.shape == (horizon, batch_size * num_envs, num_actions), (
        "States shape mismatch"
    )
    assert next_states.shape == (horizon, batch_size * num_envs, num_actions), (
        "Next states shape mismatch"
    )
    assert actions.shape == (horizon, batch_size * num_envs, num_actions), (
        "Actions shape mismatch"
    )
    assert rewards.shape == (horizon, batch_size * num_envs, num_actions), (
        "Rewards shape mismatch"
    )
    assert dones.shape == (horizon, batch_size * num_envs, num_actions), (
        "Dones shape mismatch"
    )


def test_buffer_overflow(replay_buffer):
    horizon, num_envs, num_actions = 5, 2, 2
    batch_size = 6
    for i in range(batch_size):
        state, next_state, action, reward, done = make_batch(
            horizon, num_envs, num_actions
        )
        replay_buffer.add(state, next_state, action, reward, done)

    assert len(replay_buffer.buffer) == replay_buffer.buffer_size, (
        "Buffer size should be capped"
    )


def test_get_all_data(replay_buffer):
    horizon, num_envs, num_actions = 5, 2, 2
    batch_size = 5
    for i in range(batch_size):
        state, next_state, action, reward, done = make_batch(
            horizon, num_envs, num_actions
        )
        replay_buffer.add(state, next_state, action, reward, done)

    states, next_states, actions, rewards, dones = replay_buffer.get_all()
    assert states.shape == (horizon, batch_size * num_envs, num_actions), (
        "States shape mismatch"
    )
    assert next_states.shape == (horizon, batch_size * num_envs, num_actions), (
        "Next states shape mismatch"
    )
    assert actions.shape == (horizon, batch_size * num_envs, num_actions), (
        "Actions shape mismatch"
    )
    assert rewards.shape == (horizon, batch_size * num_envs, num_actions), (
        "Rewards shape mismatch"
    )
    assert dones.shape == (horizon, batch_size * num_envs, num_actions), (
        "Dones shape mismatch"
    )
