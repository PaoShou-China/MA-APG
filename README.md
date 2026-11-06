# MA-APG

A PyTorch implementation of a GRU-based controller trained on the nonlinear control theory (NCT) system defined in `system.py`.

## Model and objective

The simulator uses the state \(x=[x_1,x_2]^T\), with

\[
\dot{x}_1=x_2,\qquad
\dot{x}_2=x_1+b u+\sin(b),
\]

where \(u\) is the scalar control input and \(b\) is a sampled system parameter. The implementation integrates the dynamics with a fourth-order Runge–Kutta step. The per-step reward penalizes position error, velocity error, and control effort:

\[
r_k=-\Delta t\left((x_{1,k}-x_1^*)^2+
0.1(x_{2,k}-x_2^*)^2+0.01u_k^2\right),
\]

with target state \(x^*=[0,0]^T\), followed by a terminal state penalty. During training, the controller sees batches of sampled initial states and \(b\) values, and updates its parameters by gradient descent through the simulated episode.

The GRU controller in `train.py` uses two recurrent layers of width 64 by default, spectral normalization on the recurrent cell's linear maps, and a sigmoid action mapping to the configured range \([-30,30]\).

## Requirements

Python 3.8 or newer and the following packages are used by the checked-in training/evaluation code:

- PyTorch
- NumPy
- SciPy

Install PyTorch using the command appropriate for your CPU or CUDA setup, then install the other dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy
python -m pip install torch
```

For a CUDA build, follow the official PyTorch selector and install the matching build instead of the generic final command. This repository does not yet lock dependency versions.

## Train

From the repository root:

```bash
python train.py
```

The current defaults are defined in `TrainingConfig` in `train.py`: seed 42, step size 0.02, 2,000 parallel simulated trajectories, 500 training episodes, and 15 integration steps per episode. Training chooses CUDA when available and otherwise uses CPU. It writes training metrics and `full_trained_controller.pth` under `log/`.

## Evaluate

After training:

```bash
python evaluate.py
```

Evaluation loads `log/full_trained_controller.pth`, evaluates the configured fixed scenarios and robustness samples, and writes CSV results under the configured log directory. Check `EvaluationConfig` in `evaluate.py` before changing the evaluation scenarios or horizon.

## Reproducibility

Training and evaluation set random seeds in their entry points, but GPU operations may not be fully deterministic across devices and PyTorch versions. Keep the source revision, package versions, seed, and configuration with any reported result. Training/evaluation parameters are currently edited in Python source rather than through a CLI.
