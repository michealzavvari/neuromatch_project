> Built as part of the **Neuromatch Academy training program**.  
> This repository implements **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) in PyTorch for **OpenAI Multi-Agent Particle Environments (MPE)**, supporting training and evaluation across cooperative and competitive scenarios, with saved checkpoints/results and optional GIF export (recommended via Linux/Colab).


## MADDPG-MPE-PyTorch

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation for Multi-Agent Particle Environments (MPE). This project implements the MADDPG algorithm for training multiple agents in cooperative and competitive multi-agent scenarios.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Evaluating Models](#evaluating-models)
- [Available Environments](#available-environments)
- [Command Line Arguments](#command-line-arguments)
- [Understanding Outputs](#understanding-outputs)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Platform Compatibility](#platform-compatibility)

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip or conda package manager
- Git (for cloning the multiagent-particle-envs repository)

### Option 1: Using pip (Recommended for macOS/Linux)

1. **Clone or navigate to the project directory:**
   ```bash
   cd neuromatch_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Fix gym version (if needed):**
   ```bash
   pip uninstall gym
   pip install gym==0.10.5
   ```

5. **Install the multiagent package:**
   ```bash
   pip install -e .
   ```

### Option 2: Using conda

1. **Create a conda environment:**
   ```bash
   conda create -n neuromatch python=3.8
   conda activate neuromatch
   ```

2. **Install dependencies:**
   ```bash
   conda install --file conda-requirements.txt
   ```

3. **Install the multiagent package:**
   ```bash
   pip install -e .
   ```

### Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; import numpy; from multiagent import scenarios; print('Installation successful!')"
```

---

## Quick Start

### 1. Train a Simple Model (Quick Test)

Run a short training session to verify everything works:

```bash
python main.py --env simple_tag --episode-num 100 --episode-length 25
```

This will:
- Train agents for 100 episodes
- Save results to `results/simple_tag/{N}/` where `{N}` is the next available folder number (1, 2, 3, etc.)
  - If this is your first run, results will be in `results/simple_tag/1/`
  - If folders 1-7 already exist, results will be in `results/simple_tag/8/`
- Generate training plots and save the model

### 2. Evaluate a Trained Model

After training, evaluate the model. First, check which folder number was created:

```bash
ls results/simple_tag/
```

Then evaluate (replace `{N}` with the actual folder number):

```bash
python evaluate.py --env simple_tag --folder {N} --episode-num 10 --episode-length 25
```

For example, if results are in folder `7`:
```bash
python evaluate.py --env simple_tag --folder 7 --episode-num 10 --episode-length 25
```

This will:
- Load the trained model from `results/simple_tag/{N}/`
- Run 10 evaluation episodes with rendering
- Generate evaluation plots

---

## Training Models

### Basic Training

Train a model with default parameters:

```bash
python main.py --env simple_tag
```

This uses default settings:
- 30,000 episodes
- 25 steps per episode
- Learning starts after 50,000 steps
- Model saves every 100 episodes

### Custom Training Parameters

Train with custom hyperparameters:

```bash
python main.py \
    --env simple_tag \
    --episode-num 10000 \
    --episode-length 50 \
    --actor-lr 0.001 \
    --critic-lr 0.001 \
    --batch-size 512 \
    --gamma 0.99 \
    --tau 0.01
```

### Training Different Environments

Train on different multi-agent scenarios:

```bash
# Cooperative spread task
python main.py --env simple_spread

# Adversarial scenario
python main.py --env simple_adversary

# Communication task
python main.py --env simple_speaker_listener
```

### Training Output

During training, you'll see:
- Episode number and cumulative rewards for each agent
- Total reward sum
- Progress updates

Example output:
```
episode 1: cumulative reward: [-2.5  1.3 -1.2], sum reward: -2.4
episode 2: cumulative reward: [-1.8  2.1 -0.9], sum reward: -0.6
...
```

### Training Results

After training completes, results are saved in:
```
results/{env_name}/{run_number}/
├── model.pt                    # Final trained model
├── model/model_{episode}.pt    # Checkpoints (every save_interval episodes)
├── rewards.npy                 # Reward history (numpy array)
├── maddpg.log                  # Training log file
└── training result of maddpg solve {env_name}.png  # Training plot
```

**Note:** The `{run_number}` is automatically incremented. Each new training run creates a new numbered folder (1, 2, 3, etc.) in the environment's results directory, so your previous runs are preserved.

---

## Evaluating Models

### Basic Evaluation

Evaluate a trained model with visual rendering:

```bash
python evaluate.py --env simple_tag --folder 1
```

Parameters:
- `--env`: Environment name (must match training environment)
- `--folder`: Folder number in results directory (usually `1` for first run)
- `--episode-num`: Number of evaluation episodes (default: 30)
- `--episode-length`: Steps per episode (default: 50)

### Evaluation with Custom Settings

```bash
python evaluate.py \
    --env simple_tag \
    --folder 1 \
    --episode-num 50 \
    --episode-length 100
```

### Evaluation Output

The evaluation script:
- Loads the trained model
- Runs episodes with rendering (visual display)
- Saves evaluation plots to the model directory
- Prints cumulative rewards for each episode

Results saved in:
```
results/{env_name}/{folder}/
└── evaluating result of maddpg solve {env_name}.png
```

### Advanced Evaluation (with GIF generation)
---

#### Exporting GIF Rollouts (Linux / Colab) — macOS note

### TL;DR
If your goal is a **GIF** (not realtime visualization), the most reliable workflow is to export frames on **Linux (Google Colab or a Linux VM)** and generate a GIF there.

### Why?
The renderer in this repo uses legacy OpenGL calls (`glPushMatrix`, `glBegin`, etc.).  
On **macOS**, the OpenGL context is often **Core Profile**, where these calls are invalid, which can cause rendering to crash.  
So for GIF export, prefer Linux.
To prevent rendering in local, the line "env.render()" has been commented in the evaluate.py.
### Guide
See: `docs/GIF_EXPORT_COLAB.md`

---

For generating animated GIFs of agent behavior, use the `z_experiments.py` module:

```python
from z_experiments import evaluate_model, get_args

# Evaluate and create GIF
args = get_args({'env': 'simple_tag', 'episode_num': 5, 'episode_length': 100})
evaluate_model(args, save_video=True, folder_name='simple_tag')
```

This creates:
- Individual frame images in `frames/` directory
- Animated GIF in `animation.gif`
- Copy in `animations/` directory

---

## Available Environments

### Standard Environments

| Environment | Description | Agents | Type |
|------------|-------------|--------|------|
| `simple_tag` | Tag game (chase and evade) | 3 good, 1 adversary | Competitive |
| `simple_adversary` | Adversarial scenario | 2 good, 1 adversary | Competitive |
| `simple_spread` | Cooperative spread task | 3 agents | Cooperative |
| `simple_push` | Cooperative push task | 2 agents | Cooperative |
| `simple_crypto` | Cryptography task | 2 agents | Competitive |
| `simple_reference` | Reference scenario | 2 agents | Mixed |
| `simple_speaker_listener` | Communication task | 2 agents | Cooperative |
| `simple_world_comm` | World communication | 3 agents | Cooperative |

### Custom Environments

| Environment | Description |
|------------|-------------|
| `simple_tag_4_4` | Tag with 4 agents |
| `simple_tag_no_adv_sharing` | Tag without adversary sharing |
| `simple_tag_big_bounds` | Tag with larger boundaries |
| `simple_tag_colab` | Collaborative tag variant |
| `simple_tag_goty_edition` | Configurable tag with many parameters |

### Environment Details

Each environment has different:
- Number of agents
- Observation spaces
- Action spaces
- Reward structures
- Episode lengths

Check `multiagent/scenarios/` for environment-specific details.

---

## Command Line Arguments

### Training (`main.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--env` | str | `simple_tag_no_adv_sharing` | Environment name |
| `--episode-length` | int | 25 | Steps per episode |
| `--episode-num` | int | 30000 | Total training episodes |
| `--gamma` | float | 0.95 | Discount factor (0-1) |
| `--buffer-capacity` | int | 1000000 | Replay buffer size |
| `--batch-size` | int | 1024 | Training batch size |
| `--actor-lr` | float | 0.01 | Actor learning rate |
| `--critic-lr` | float | 0.01 | Critic learning rate |
| `--steps-before-learn` | int | 50000 | Steps before learning starts |
| `--learn-interval` | int | 100 | Learning frequency (every N steps) |
| `--save-interval` | int | 100 | Model save frequency (every N episodes) |
| `--tau` | float | 0.02 | Soft update parameter (0-1) |

### Evaluation (`evaluate.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--env` | str | `simple_tag_big_bounds` | Environment name |
| `--folder` | str | `3` | Results folder number |
| `--episode-length` | int | 50 | Steps per episode |
| `--episode-num` | int | 30 | Number of evaluation episodes |

---

## Understanding Outputs

### Training Plots

The training script generates a plot showing:
- **Individual agent rewards**: Each agent's cumulative reward per episode
- **Running average**: Smoothed reward curve (100-episode window)

File: `results/{env}/1/training result of maddpg solve {env}.png`

### Reward Files

- **`rewards.npy`**: NumPy array of shape `(episodes, num_agents)`
  - Load with: `np.load('rewards.npy')`
  - Contains cumulative rewards for each agent per episode

### Model Files

- **`model.pt`**: Final trained model (PyTorch state dicts)
- **`model/model_{N}.pt`**: Checkpoints saved during training

### Log Files

- **`maddpg.log`**: Training log with detailed information
  - Training progress
  - Loss values (if enabled)
  - Error messages

### Evaluation Plots

Evaluation script generates:
- Reward plots for each agent during evaluation
- File: `results/{env}/{folder}/evaluating result of maddpg solve {env}.png`

---

## Advanced Usage

### Batch Training

Use `z_batch_train.py` for running multiple experiments:

```python
from z_batch_train import run_experiment, get_args

# Run with default args
args = get_args({'env': 'simple_tag'})
run_experiment(args)

# Run with custom world parameters (for goty_edition)
world_args = {
    'num_good_agents': 4,
    'num_adversaries': 2,
    'out_of_bound_punishment': 10,
    'hard_boundary': False
}
args = get_args({'env': 'simple_tag_goty_edition'})
run_experiment(args, world_args=world_args)
```

### Custom Environment Parameters

For `simple_tag_goty_edition`, you can customize:

```python
world_args = {
    'num_good_agents': 4,          # Number of good agents
    'num_adversaries': 2,          # Number of adversaries
    'num_landmarks': 2,            # Number of landmarks
    'out_of_bound_punishment': 10, # Penalty for going out of bounds
    'hard_boundary': False,        # Use soft or hard boundaries
    'good_collaborative': True,    # Good agents share rewards
    'bad_collaborative': False,    # Bad agents share rewards
    'remove_old_adv_sharing': True # Remove old adversary sharing
}
```

### Creating Custom Scenarios

1. Create a new file in `multiagent/scenarios/`
2. Inherit from `BaseScenario`
3. Implement required methods:
   - `make_world()`: Create the world and agents
   - `reset_world()`: Reset world state
   - `reward()`: Calculate rewards
   - `observation()`: Generate observations

See existing scenarios for examples.

### Hyperparameter Tuning

Key hyperparameters to tune:

- **Learning rates** (`--actor-lr`, `--critic-lr`): Start with 0.01, try 0.001 or 0.0001
- **Batch size** (`--batch-size`): Larger = more stable, but slower. Try 256, 512, 1024
- **Gamma** (`--gamma`): Discount factor. Higher = long-term thinking. Try 0.9, 0.95, 0.99
- **Tau** (`--tau`): Soft update rate. Lower = slower target network updates. Try 0.01, 0.02, 0.05
- **Steps before learn** (`--steps-before-learn`): More = better initial data. Try 25k, 50k, 100k

---

## Project Structure

```
neuromatch_project/
├── main.py                 # Main training script
├── evaluate.py             # Model evaluation script
├── MADDPG.py               # MADDPG algorithm implementation
├── Agent.py                # Individual agent (actor-critic)
├── Buffer.py               # Experience replay buffer
├── make_env.py             # Environment creation utility
├── z_batch_train.py        # Batch training utilities
├── z_experiments.py        # Experiment utilities (GIF generation)
├── requirements.txt        # pip dependencies
├── conda-requirements.txt  # conda dependencies
├── setup.py                # Package setup
│
├── multiagent/             # Multi-agent environment package
│   ├── __init__.py
│   ├── core.py            # Core classes (World, Agent, Entity)
│   ├── environment.py     # MultiAgentEnv class
│   ├── scenario.py         # Base scenario class
│   ├── rendering.py       # Visualization
│   └── scenarios/         # Environment scenarios
│       ├── simple_tag.py
│       ├── simple_spread.py
│       └── ...
│
├── results/                # Training results
│   └── {env_name}/
│       └── {run_number}/
│           ├── model.pt
│           ├── rewards.npy
│           └── ...
│
└── animations/            # Generated GIFs
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'multiagent'`

**Solution**:
```bash
pip install -e .
```

#### 2. Gym Version Conflicts

**Problem**: `AttributeError` or version-related errors

**Solution**:
```bash
pip uninstall gym gymnasium
pip install gym==0.10.5
```

#### 3. Pyglet Display Issues (macOS)

**Problem**: Window doesn't open or display errors

**Solution**:
- For headless evaluation, use `save_video=True` in `z_experiments.py`
- Install XQuartz if needed: `brew install --cask xquartz`

#### 4. CUDA/GPU Issues

**Problem**: CUDA errors or GPU not detected

**Solution**:
- The code automatically uses CPU if CUDA is unavailable
- To force CPU: Modify `MADDPG.py` to set `device=torch.device('cpu')`

#### 5. Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce `--batch-size` (try 256 or 512)
- Reduce `--buffer-capacity` (try 100000)
- Use CPU instead of GPU

#### 6. Model Loading Errors

**Problem**: `FileNotFoundError` when evaluating

**Solution**:
- Check that the folder number exists: `ls results/{env}/`
- Ensure `model.pt` exists in the folder
- Verify environment name matches training environment

#### 7. Slow Training

**Problem**: Training is very slow

**Solution**:
- Reduce `--episode-num` for testing
- Reduce `--episode-length`
- Disable rendering during training (already disabled by default)
- Use GPU if available

### Getting Help

1. Check the log file: `results/{env}/{folder}/maddpg.log`
2. Verify all dependencies are installed: `pip list`
3. Test with minimal parameters first
4. Check that environment name is correct

---

## Platform Compatibility

### ✅ macOS
- Fully supported
- All dependencies available
- Rendering works with XQuartz (if needed)

### ✅ Linux
- Fully supported
- All dependencies available
- Rendering works natively

### ⚠️ Windows
- Should work with most features
- `pywin32` removed from requirements (not needed for core functionality)
- Rendering may require additional setup

### Recent Changes

- ✅ Fixed security vulnerabilities (fonttools, idna, requests)
- ✅ Removed platform-specific dependencies (pywin32)
- ✅ Cleaned conda-requirements.txt of conda-specific file paths
- ✅ Refactored main.py for better code structure
- ✅ Added proper error handling
- ✅ Improved cross-platform compatibility

---

## Additional Resources

### Algorithm Details

- **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient
- **Actor-Critic**: Each agent has an actor (policy) and critic (value function)
- **Centralized Training, Decentralized Execution**: Training uses global info, execution is local

### Papers

- MADDPG: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- Multi-Agent Particle Environments: [OpenAI Multi-Agent Particle Environments](https://github.com/openai/multiagent-particle-envs)

### License

See `LICENSE.txt` for details.

---

## Contributing

When contributing:
1. Follow existing code style
2. Test on multiple environments
3. Update documentation
4. Ensure cross-platform compatibility

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Advances in neural information processing systems},
  year={2017}
}
```

---

**Happy Training! 🚀**
