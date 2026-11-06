# FILE: train.py

import torch
import random
import numpy as np
import csv
from dataclasses import dataclass, field
import os
import datetime


from controller import GRUController


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
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg.log_base_dir, exist_ok=True)
    cfg.initial_state_sampling_base_scale = cfg.initial_state_sampling_base_scale.to(device)

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
