# MA-APG

A PyTorch GRU controller for the nonlinear control theory system implemented in system.py. The controller is trained through differentiable simulated episodes.

[Project homepage](https://paoshou-china.github.io/MA-APG/) *(deployed by the repository Pages workflow)* · [Browse source](https://github.com/PaoShou-China/MA-APG)

![Recurrent controller and nonlinear system rollout](assets/hero.svg)

## Model and objective

The simulator uses state x = [x1, x2]ᵀ and dynamics

```
ẋ1 = x2
ẋ2 = x1 + b u + sin(b)
```

where u is scalar control and b is a sampled system parameter. Each integration step uses fourth-order Runge–Kutta (RK4). The reward penalizes position error, velocity error, and control effort:

```
rₖ = −Δt ((x1,ₖ − x1*)² + 0.1(x2,ₖ − x2*)² + 0.01uₖ²)
```

with target state [0, 0]ᵀ and an additional terminal-state penalty. Training differentiates through the simulated episode to update a two-layer GRU controller. Its default hidden width is 64, recurrent linear maps use spectral normalization, and actions are mapped to [-30, 30] by a sigmoid.

## Repository map

- system.py — nonlinear simulator, RK4 step, reward, sampled initial states, and evaluation helpers.
- train.py — GRU controller, training configuration, training loop, and checkpoint output.
- evaluate.py — fixed-scenario and parameter-sampling evaluation.

## Requirements

Python 3.8 or newer, PyTorch, NumPy, and SciPy. Install the PyTorch build for your CPU/CUDA setup; versions are not currently pinned.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy
python -m pip install torch
```

## Quick start

From the repository root:

```bash
python train.py
python evaluate.py
```

Training writes metrics and full_trained_controller.pth under log/. Evaluation loads log/full_trained_controller.pth and writes CSV results in the configured log directory. If training has not completed, run training first.

## Current defaults and reproduction

TrainingConfig in train.py sets seed 42, time step 0.02, 2,000 parallel simulated trajectories, 500 training episodes, and 15 integration steps per episode. EvaluationConfig in evaluate.py sets its own seed, fixed initial conditions and b values, and random robustness samples. Training and evaluation options are currently edited in Python source; there is no CLI.

Record the source revision, dependency versions, seed, and configuration with reported results. GPU operations may vary across devices and PyTorch versions. The project does not currently include a verified dependency lock or cross-platform version matrix.
