# MA-APG

This project trains a GRU-based neural controller to manage the Nonlinear Control Theory (NCT) system dynamics.

## 1. Environment Setup

You need Python 3.8+ and PyTorch installed. It is highly recommended to use a virtual environment (e.g., Conda).

**Steps:**

1.  **Activate Environment:** Ensure your environment (assumed name `maapg` based on file context) is active.
    ```bash
    conda activate maapg
    ```
2.  **Install Dependencies:** Install the necessary libraries.
    ```bash
    pip install torch torchvision torchaudio numpy scipy matplotlib
    ```

## 2. How to Train

Run the main training script, `train.py`. This script initializes the simulation environment, the GRU controller, optimizes the controller parameters using the rewards, and saves the resulting model.

**Command:**

```bash
python train.py
