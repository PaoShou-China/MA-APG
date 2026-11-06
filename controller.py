"""Recurrent controller used by MA-APG training and evaluation."""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

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
            # The cell updates a new list and never mutates hidden-state tensors in place.
            # to() is a no-op for the normal same-device rollout and remains differentiable if needed.
            h0 = [h.to(x.device) for h in h0]

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
