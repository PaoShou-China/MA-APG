# FILE: system.py

import torch
import torch.nn as nn
import numpy as np
import scipy.linalg


class NCTSystemSimulator(nn.Module):
    """
    Simulator for the system: x_ddot = x1 + bu + sin(b)
    State: x = [position, velocity]^T
    """

    def __init__(self, dt, action_dim):
        super().__init__()
        self.DT = torch.tensor(dt, dtype=torch.float32)
        self.action_dim = action_dim
        self.state_dim = 2

        self.REWARD_X1_PENALTY = 1.0
        self.REWARD_X2_PENALTY = 0.1
        self.REWARD_ACTION_PENALTY = 0.01
        self.TARGET_STATE = torch.tensor([0.0, 0.0])

    def to(self, device):
        """Moves internal tensors to the specified device."""
        self.device = device
        self.TARGET_STATE = self.TARGET_STATE.to(device)
        self.DT = self.DT.to(device)
        return self

    def _system_dynamics(self, state, u, b_param, t_current):
        """Calculates the system derivatives (x_dot)."""
        x1, x2 = state[:, 0:1], state[:, 1:2]
        b_val = b_param[:, 0:1]
        x1_dot = x2
        x2_dot = x1 + b_val * u + torch.sin(b_val)
        return torch.cat((x1_dot, x2_dot), dim=1)

    def _step(self, state, action, b_param, current_time):
        """Performs one simulation step using RK4."""
        u_cmd = action[:, 0:1]
        ode = lambda s, t_ode: self._system_dynamics(s, u_cmd, b_param, t_ode)
        k1 = ode(state, current_time)
        k2 = ode(state + 0.5 * self.DT * k1, current_time + 0.5 * self.DT)
        k3 = ode(state + 0.5 * self.DT * k2, current_time + 0.5 * self.DT)
        k4 = ode(state + self.DT * k3, current_time + self.DT)
        next_state = state + (self.DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        reward = -(self.REWARD_X1_PENALTY * (next_state[:, 0:1] - self.TARGET_STATE[0]) ** 2 +
                   self.REWARD_X2_PENALTY * (next_state[:, 1:2] - self.TARGET_STATE[1]) ** 2 +
                   self.REWARD_ACTION_PENALTY * u_cmd ** 2) * self.DT
        return next_state, reward

    def calculate_truncated_reward(self, state):
        """Final state penalty."""
        return -(self.REWARD_X1_PENALTY * (state[:, 0:1] - self.TARGET_STATE[0]) ** 2 +
                 self.REWARD_X2_PENALTY * (state[:, 1:2] - self.TARGET_STATE[1]) ** 2) * 10.0

    def sample_initial_states(self, num_envs, device, current_sampling_scale):
        """Samples initial states for training."""
        # Initial states centered around zero, scaled by current_sampling_scale
        initial_states = (torch.rand(num_envs, 2, device=device) * 2 - 1) * current_sampling_scale.to(device)
        return initial_states

    def run_episode(self, controller, b_param_dist, num_steps, device, initial_states):
        """Runs a single simulation episode for training."""
        n_envs = b_param_dist.shape[0]
        total_episode_reward = torch.zeros((n_envs, 1), device=device, requires_grad=True)
        current_h = None
        current_states = initial_states
        current_time_scalar = 0.0
        for step in range(num_steps):
            h0_for_controller = None
            if hasattr(controller, 'gru_cells') and current_h is not None:
                h0_for_controller = [h_state.clone() for h_state in current_h]

            action, next_h = controller(current_states, h0=h0_for_controller)

            next_states, step_reward = self._step(current_states, action, b_param_dist, current_time_scalar)
            current_states = next_states
            total_episode_reward = total_episode_reward + step_reward
            current_h = next_h
            current_time_scalar += self.DT.item()

        total_episode_reward = total_episode_reward + self.calculate_truncated_reward(current_states)
        return total_episode_reward


def calculate_lqr_gain(b_param_np, simulator):
    """Calculates the LQR gain K for the linearized system."""
    b = b_param_np[0]
    A_aug = simulator.A.cpu().numpy()
    B_aug = np.array([[0.0], [b]])
    Q = np.diag([simulator.REWARD_X1_PENALTY, simulator.REWARD_X2_PENALTY])
    R = np.array([[simulator.REWARD_ACTION_PENALTY]])
    try:
        if abs(b) < 1e-6:
            return np.zeros((1, 2))
        P = scipy.linalg.solve_continuous_are(A_aug, B_aug, Q, R)
        K = np.linalg.inv(R) @ B_aug.T @ P
        return K
    except Exception:
        return np.zeros((1, 2))


def _run_evaluation_batch(
        simulator, controller, initial_states_batch,
        b_batch, eval_steps, controller_type_str, device,
        lqr_gains_batch=None, action_low_limit=None, action_high_limit=None
):
    """Runs a batch evaluation (simplified for GRU only)."""
    batch_size = initial_states_batch.shape[0]
    current_states = initial_states_batch.to(device)
    b_batch = b_batch.to(device)
    h_batch = None

    total_rewards = torch.zeros(batch_size, 1, device=device)
    current_time_batch = torch.zeros(batch_size, 1, device=device)

    clamp_low = action_low_limit
    clamp_high = action_high_limit
    if controller is not None and hasattr(controller, 'action_low_tensor'):
        clamp_low = controller.action_low_tensor.to(device)
        clamp_high = controller.action_high_tensor.to(device)

    with torch.no_grad():
        for _ in range(eval_steps):
            if controller_type_str == 'GRU':
                action, h_batch = controller(current_states, h0=h_batch)
                action = torch.clamp(action, clamp_low, clamp_high)
            else:
                raise ValueError("Only GRU evaluation supported in this simplified batch function.")

            next_states, rewards = simulator._step(current_states, action, b_batch, current_time_batch)
            total_rewards += rewards
            current_states = next_states
            current_time_batch += simulator.DT

    return total_rewards
