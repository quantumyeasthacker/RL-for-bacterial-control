# RL-for-bacterial-control

Code for [Reinforcement learning for adaptive control of phenotypically heterogeneous bacterial populations](https://www.biorxiv.org/content/10.1101/2025.11.21.689767v1).

## Repository layout

```
rl-for-bacterial-control/
├── pyproject.toml              # package metadata + dependencies
├── src/
│   └── rlBacterialControl/     # the importable package
│       ├── agent/              # CDQL agent, deep Q-network, replay buffer
│       ├── envs/               # cell-population model and RL environments
│       └── utils/              # figure/plot helpers used during training
├── scripts/                    # command-line entry points
│   ├── train/                  # run_w_wandb_single_*.py, RL_test.py
│   ├── eval/                   # eval_trained_agents_*.py, eval_robustness*.py
│   └── simulate/               # simulation_test_*.py
└── plotting/                   # standalone figure-generation scripts
```

The core library lives in `src/rlBacterialControl/`. Everything under `scripts/`
and `plotting/` is a thin command-line driver that imports from the package.

## Installation

Install the package (editable) into your environment:

```bash
pip install -e .
```

To also pull in the extra packages used by the `plotting/` scripts:

```bash
pip install -e ".[plotting]"
```

## Usage

Once installed, the package is importable from anywhere:

```python
from rlBacterialControl.envs.cell_model import CellConfig
from rlBacterialControl.envs.envs import EnvConfig, ConstantNutrientEnv
from rlBacterialControl.agent.MLP_full import CDQL
```

The scripts take positional arguments via `sys.argv` and write outputs to the
`results_dir` you pass (relative to your current working directory). For example,
to train a constant-nutrient agent:

```bash
# args: antibiotic_value nutrient_value delay_embed_len rep results_dir
python scripts/train/run_w_wandb_single_constenv.py 3.72 0.50 10 0 results
```

Plotting scripts are run from within the `plotting/` directory (they import a
local `utils.py`):

```bash
cd plotting
python plot_trajectory.py
```
