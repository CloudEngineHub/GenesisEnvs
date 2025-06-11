# Genesis Environment
This repository contains **parallelized** RL environment for Genesis general-purpose physics platform.

## Setup
### Prerequisites

- Python >= 3.11
- [uv >= 0.6](https://docs.astral.sh/uv/getting-started/installation/)

### Install
Clone this repository:
``` bash
git clone https://github.com/RochelleNi/GenesisEnvs.git
```

Then install by uv:
```bash
$ uv sync
```

or by PyPI:
```bash
$ pip install -e .
```

## Usage
To train a policy, you can run with:
```bash
uv run run.py --config-path PATH_TO_YOUR_CONFIG --exp-name YOUR_EXP_NAME

# training PPO agent with grasp env:
uv run run.py --config-path configs/GraspFrankaDQN.yaml --exp-name EXP_GraspFrankaDQN

# traininig DQN agent with grasp env:
uv run run.py --config-path configs/GraspFrankaPPO.yaml --exp-name EXP_GraspFrankaPPO
```
