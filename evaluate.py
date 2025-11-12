# FILE: evaluate.py (Final Fix for simulator.A access)

import torch
import torch.nn as nn
import numpy as np
import csv
from dataclasses import dataclass, field
import os
import random

# Import necessary components
from system import NCTSystemSimulator, _run_evaluation_batch
from train import GRUController as GRUController_Full_SN


# ==============================================================================
#  EVALUATION CONFIGURATION
# ==============================================================================

@dataclass
class EvaluationConfig:
    """Simple configuration for evaluation."""
    seed: int = 42
    sim_dt: float = 0.02
    eval_steps: int = 500

    input_dim: int = 2
    hidden_dim: int = 64
    num_layers: int = 2
    action_dim: int = 1
    controller_action_low: torch.Tensor = field(default_factory=lambda: torch.tensor([-30.0]))
    controller_action_high: torch.Tensor = field(default_factory=lambda: torch.tensor([30.0]))

    # Fixed set of initial states for testing
    eval_initial_states: torch.Tensor = field(default_factory=lambda: torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.5], [0.0, -0.5], [0.8, 0.4], [-0.8, -0.4]], dtype=torch.float32))

    # Fixed system parameters (b) for testing
    eval_system_param_sets: dict = field(default_factory=lambda: {
        "b_low": torch.tensor([0.1]),
        "b_nominal": torch.tensor([1.0]),
        "b_high": torch.tensor([10.0]),
    })

    # Fixed sampling boundaries for robustness check (Carried over from previous fix)
    b_min_random_sampling: float = 0.1
    b_max_random_sampling: float = 10.0

    num_box_plot_samples: int = 50  # Reduced sample count for speed

    log_base_dir: str = "log"


# ==============================================================================
#  HELPER FUNCTION FOR SINGLE TRAJECTORY SIMULATION (GRU only)
# ==============================================================================

def simulate_single_reward_gru(
        simulator, initial_state, b_param, num_steps, controller_obj,
        action_low_limit=None, action_high_limit=None
):
    """Simulates a single trajectory using the GRU controller and returns the total reward."""
    # FIX: Use simulator.DT to find the device, replacing the missing simulator.A
    device = simulator.DT.device

    current_state = initial_state.unsqueeze(0).to(device)
    b_param_batch = b_param.unsqueeze(0).to(device)
    total_reward = torch.tensor([0.0], device=device)
    current_h = None
    current_time = 0.0

    with torch.no_grad():
        for i in range(num_steps):
            action, current_h = controller_obj(current_state, h0=current_h)

            if action_low_limit is not None and action_high_limit is not None:
                action = torch.clamp(action, action_low_limit, action_high_limit)

            current_time_batch_for_step = torch.full((current_state.shape[0], 1), current_time, device=device)
            next_state, reward_step = simulator._step(current_state, action, b_param_batch, current_time_batch_for_step)

            total_reward += reward_step.squeeze(0)
            current_state = next_state
            current_time += simulator.DT.item()

    return total_reward.cpu()


# ==============================================================================
#  MAIN EVALUATION FUNCTION FOR A SINGLE MODEL
# ==============================================================================

def evaluate_single_model(
        model_friendly_name, controller_class, cfg, device,
        model_filename, controller_type_for_batch_eval, eval_scenarios_for_csv,
        batch_initial_states_eval, batch_b_params_eval,
        simulator, action_low_limit, action_high_limit,
        random_box_plot_scenarios
):
    """Loads and evaluates a single trained controller model."""
    print(f"\n{'=' * 60}\n  Evaluating Model: {model_friendly_name}\n{'=' * 60}\n")
    controller = controller_class(cfg.input_dim, cfg.hidden_dim, cfg.num_layers, cfg.action_dim,
                                  cfg.controller_action_low, cfg.controller_action_high).to(device)
    controller_path = os.path.join(cfg.log_base_dir, model_filename)
    try:
        controller.load_state_dict(torch.load(controller_path, map_location=device))
        controller.eval();
        print(f"Controller '{model_friendly_name}' loaded from {controller_path}.")
    except FileNotFoundError:
        print(f"Error: {controller_path} not found.");
        return None, {}

    # 1. Batch Evaluation (for fixed test scenarios)
    current_model_rewards_batch = _run_evaluation_batch(
        simulator=simulator, controller=controller, initial_states_batch=batch_initial_states_eval,
        b_batch=batch_b_params_eval, eval_steps=cfg.eval_steps, controller_type_str=controller_type_for_batch_eval,
        device=device, action_low_limit=action_low_limit, action_high_limit=action_high_limit
    )

    csv_output_filepath = os.path.join(cfg.log_base_dir, f"{model_filename.replace('.pth', '')}_results.csv")
    with open(csv_output_filepath, 'w', newline='') as csvfile:
        fieldnames = ['Param_Name', 'b_param', 'Initial_x1', 'Initial_x2', f'{model_friendly_name}_Reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, scenario in enumerate(eval_scenarios_for_csv):
            writer.writerow({'Param_Name': scenario['param_name'], 'b_param': scenario['b_param'].item(),
                             'Initial_x1': scenario['initial_state'][0].item(),
                             'Initial_x2': scenario['initial_state'][1].item(),
                             f'{model_friendly_name}_Reward': current_model_rewards_batch[i, 0].item()})
    print(f"Detailed evaluation results saved to {csv_output_filepath}")

    # 2. Robustness Check (Small sample set)
    model_rewards_for_robustness_check = {}

    for scenario in random_box_plot_scenarios:
        scenario_key = f"Single_Run_{scenario['idx']}"
        reward = simulate_single_reward_gru(
            simulator, scenario['istate'], scenario['b_param'], cfg.eval_steps,
            controller_obj=controller, action_low_limit=action_low_limit, action_high_limit=action_high_limit
        )
        model_rewards_for_robustness_check[scenario_key] = [reward.item()]

    return controller, model_rewards_for_robustness_check


# ==============================================================================
#  MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    overall_seed = 42
    torch.manual_seed(overall_seed);
    np.random.seed(overall_seed);
    random.seed(overall_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cfg = EvaluationConfig()
    os.makedirs(cfg.log_base_dir, exist_ok=True)

    nn_models_to_evaluate = [{"name": "GRU Controller", "controller_class": GRUController_Full_SN,
                              "filename": "full_trained_controller.pth", "controller_type_for_batch_eval": "GRU",
                              "plot_label": "GRU-APG"}]
    simulator = NCTSystemSimulator(dt=cfg.sim_dt, action_dim=cfg.action_dim).to(device)

    action_low_on_device = cfg.controller_action_low.to(device)
    action_high_on_device = cfg.controller_action_high.to(device)

    # --- 1. Prepare Data for Fixed Scenario Batch Evaluation ---
    eval_scenarios_for_csv = []
    for param_name, b_tensor in cfg.eval_system_param_sets.items():
        for initial_state_tensor in cfg.eval_initial_states:
            eval_scenarios_for_csv.append(
                {'param_name': param_name, 'b_param': b_tensor, 'initial_state': initial_state_tensor})

    batch_initial_states_eval = torch.stack([s['initial_state'].to(device) for s in eval_scenarios_for_csv])
    batch_b_params_eval = torch.stack([s['b_param'].to(device) for s in eval_scenarios_for_csv])

    # --- 2. Prepare Scenarios for Robustness Check (Small Sample) ---
    random_box_plot_scenarios: list = []
    # Use only a few random b values for a quick robustness check
    random_b_for_check = (torch.rand(5, device=device) *
                          (cfg.b_max_random_sampling - cfg.b_min_random_sampling) + cfg.b_min_random_sampling)

    istate_rand = cfg.eval_initial_states[0]

    for i, b_rand in enumerate(random_b_for_check):
        random_box_plot_scenarios.append({
            'idx': i,
            'b_param': b_rand.unsqueeze(0).cpu(),
            'istate': istate_rand,
        })

    # --- 3. Evaluate NN Models ---
    all_nn_results = {}

    for model_info in nn_models_to_evaluate:
        _, model_robust_results = evaluate_single_model(
            model_info["name"], model_info["controller_class"], cfg, device,
            model_info["filename"], model_info["controller_type_for_batch_eval"],
            eval_scenarios_for_csv, batch_initial_states_eval, batch_b_params_eval,
            simulator, action_low_on_device, action_high_on_device,
            random_box_plot_scenarios
        )
        if model_robust_results:
            all_nn_results[model_info["plot_label"]] = model_robust_results

    # --- 4. Combine and Print Statistics ---

    print("\n\n\n==================================================================")
    print("  EVALUATION SUMMARY (Fixed Scenarios & Robustness Check)")
    print("==================================================================")

    all_gru_rewards = []
    for scenario in random_box_plot_scenarios:
        scenario_key = f"Single_Run_{scenario['idx']}"
        all_gru_rewards.extend(all_nn_results.get("GRU-APG", {}).get(scenario_key, []))


    def print_summary(label, rewards):
        if not rewards:
            print(f"{label}: N/A (No samples collected)")
            return
        reward_array = np.array(rewards)
        mean_val = np.mean(reward_array)
        std_val = np.std(reward_array)
        print(f"{label:<15}: Mean Reward = {mean_val:.4f}, Std Dev = {std_val:.4f} (Total {len(rewards)} samples)")


    print("\n--- Robustness Check Summary ---")
    print_summary("GRU-APG", all_gru_rewards)

    print("\n==================================================================")
    print("\nEvaluation complete.")
