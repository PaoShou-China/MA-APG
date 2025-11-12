# FILE: train.py

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
import csv
from dataclasses import dataclass, field
import os
import datetime


# --- GRU Cell ---
class GRUCell(nn.Module):
    """A single GRU cell."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih_r = spectral_norm(nn.Linear(input_size, hidden_size, bias=False))
        self.weight_hh_r = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
        self.weight_ih_z = spectral_norm(nn.Linear(input_size, hidden_size, bias=False))
        self.weight_hh_z = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
        self.weight_ih_n = spectral_norm(nn.Linear(input_size, hidden_size, bias=False))
        self.weight_hh_n = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))

        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_n = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x, h_prev):
        r_gate = torch.sigmoid(self.weight_ih_r(x) + self.weight_hh_r(h_prev) + self.bias_r)
        z_gate = torch.sigmoid(self.weight_ih_z(x) + self.weight_hh_z(h_prev) + self.bias_z)
        n_gate = torch.tanh(self.weight_ih_n(x) + r_gate * self.weight_hh_n(h_prev) + self.bias_n)
        h_new = (1 - z_gate) * n_gate + z_gate * h_prev
        return h_new


# --- GRU Controller ---
class GRUController(nn.Module):
    """GRU-based controller taking state as input and outputting an action."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, action_low, action_high):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.register_buffer('action_low_tensor', action_low)
        self.register_buffer('action_high_tensor', action_high)

        self.gru_cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.gru_cells.append(GRUCell(layer_input_size, hidden_size))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """Forward pass for the GRU controller."""
        batch_size = x.shape[0]

        if h0 is None:
            h0 = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h0 = [h.clone().to(x.device) for h in h0]

        current_h = h0
        input_to_gru_layer = x

        for layer_idx in range(self.num_layers):
            h_new = self.gru_cells[layer_idx](input_to_gru_layer, current_h[layer_idx])
            current_h[layer_idx] = h_new
            input_to_gru_layer = h_new

        raw_output = self.fc(current_h[-1])
        action = self._map_action(raw_output)
        return action, current_h

    def _map_action(self, raw_output):
        """Maps raw network output to a specified action range."""
        sigmoid_output = torch.sigmoid(raw_output)
        return self.action_low_tensor + sigmoid_output * (self.action_high_tensor - self.action_low_tensor)


# --- Training Loop ---
def _run_training_loop(
        controller, controller_optimizer, simulator, cfg, device, log_base_dir
):
    print("\n--- Starting Training ---")
    metric_log_filepath = os.path.join(log_base_dir, "training_metrics.csv")
    csv_fieldnames = ['Episode', 'Avg_Reward', 'Loss']

    with open(metric_log_filepath, 'w', newline='') as metric_csvfile:
        metric_writer = csv.DictWriter(metric_csvfile, fieldnames=csv_fieldnames)
        metric_writer.writeheader()

        for episode in range(cfg.num_episodes):
            # Sample initial conditions and parameters for this batch
            initial_states_batch = simulator.sample_initial_states(cfg.n_envs, device,
                                                                   cfg.initial_state_sampling_base_scale)
            sampled_b = (torch.rand(cfg.n_envs, 1, device=device) * (
                        cfg.b_max_sampling - cfg.b_min_sampling) + cfg.b_min_sampling)

            # --- Controller step (Minimize negative reward) ---
            total_episode_reward_controller = simulator.run_episode(
                controller=controller,
                b_param_dist=sampled_b,
                num_steps=cfg.initial_num_steps,
                device=device,
                initial_states=initial_states_batch,
            )
            controller_loss = -torch.mean(total_episode_reward_controller)

            controller_optimizer.zero_grad()
            controller_loss.backward()

            # Simple gradient clipping
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)

            controller_optimizer.step()

            # --- Logging ---
            if (episode + 1) % 10 == 0 or episode == 0:
                avg_reward_controller = torch.mean(total_episode_reward_controller).item()
                print(
                    f"E {episode + 1}/{cfg.num_episodes} | Avg Rwd: {avg_reward_controller:.2f} | Loss: {controller_loss.item():.4f}")

                metric_writer.writerow({
                    'Episode': episode + 1,
                    'Avg_Reward': avg_reward_controller,
                    'Loss': controller_loss.item()
                })

    print("--- Training Complete ---")


# --- Import components from system.py ---
from system import NCTSystemSimulator


# ==============================================================================
#
#  CONFIGURATION CLASSES
#
# ==============================================================================

@dataclass
class TrainingConfig:
    """Simple configuration for training."""
    seed: int = 42
    sim_dt: float = 0.02
    n_envs: int = 2000  # Reduced batch size
    num_episodes: int = 500  # Reduced episodes
    initial_num_steps: int = 15

    input_dim: int = 2
    hidden_dim: int = 64
    num_layers: int = 2
    action_dim: int = 1
    controller_action_low: torch.Tensor = field(default_factory=lambda: torch.tensor([-30.0]))
    controller_action_high: torch.Tensor = field(default_factory=lambda: torch.tensor([30.0]))
    controller_lr: float = 0.001

    b_min_sampling: float = 0.1
    b_max_sampling: float = 10.0

    initial_state_sampling_base_scale: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 1.0]))

    log_base_dir: str = "log"


# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def train():
    """Orchestrates the training process."""
    cfg = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg.log_base_dir, exist_ok=True)

    # 1. INITIALIZE MODELS AND SIMULATOR
    controller = GRUController(
        cfg.input_dim, cfg.hidden_dim, cfg.num_layers, cfg.action_dim,
        cfg.controller_action_low, cfg.controller_action_high
    ).to(device)
    controller_optimizer = torch.optim.Adam(
        controller.parameters(), lr=cfg.controller_lr
    )

    simulator = NCTSystemSimulator(dt=cfg.sim_dt, action_dim=cfg.action_dim).to(device)

    # 2. TRAIN
    _run_training_loop(
        controller=controller, controller_optimizer=controller_optimizer,
        simulator=simulator, cfg=cfg, device=device, log_base_dir=cfg.log_base_dir
    )

    torch.save(controller.state_dict(), os.path.join(cfg.log_base_dir, "full_trained_controller.pth"))
    print(f"Trained controller saved to {cfg.log_base_dir}.")


if __name__ == "__main__":
    print("Starting training process...")
    train()
    print("\nTraining complete. Run evaluate.py to test the trained model.")
