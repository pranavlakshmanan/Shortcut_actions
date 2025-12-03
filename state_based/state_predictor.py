import torch
import torch.nn as nn

class StatePredictor(nn.Module):
    """Neural network that predicts final state directly s_theta(s, a, t, d)

    Key difference from VelocityFieldNet:
    - Output is final state directly, NOT velocity
    - No multiplication by step_size after network output
    - Same architecture (411K params) for fair comparison
    """

    def __init__(self, state_dim=4, action_dim=2, max_seq_len=10,
                 hidden_dims=[256, 256, 128]):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len

        # Input: state (4) + flattened actions (2*max_seq_len) + time (1) + step_size (1)
        input_dim = state_dim + action_dim * max_seq_len + 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer: predict final state directly
        layers.append(nn.Linear(prev_dim, state_dim))

        self.network = nn.Sequential(*layers)

        # Use PyTorch's default initialization for output layer

    def forward(self, state, action_seq, time, step_size):
        """
        Args:
            state: (batch, state_dim)
            action_seq: (batch, seq_len, action_dim) - can be variable length
            time: (batch, 1)
            step_size: (batch, 1)

        Returns:
            final_state: (batch, state_dim) - predicted state at t+d
        """
        batch_size = state.shape[0]

        # Flatten and pad action sequence
        seq_len = action_seq.shape[1]
        action_flat = action_seq.reshape(batch_size, -1)  # (batch, seq_len * action_dim)

        # Pad if necessary
        if seq_len < self.max_seq_len:
            padding = torch.zeros(batch_size,
                                 (self.max_seq_len - seq_len) * self.action_dim,
                                 device=action_seq.device)
            action_flat = torch.cat([action_flat, padding], dim=1)
        elif seq_len > self.max_seq_len:
            action_flat = action_flat[:, :self.max_seq_len * self.action_dim]

        # Concatenate all inputs
        x = torch.cat([state, action_flat, time, step_size], dim=1)

        # Predict final state directly (KEY DIFFERENCE: no * step_size)
        final_state = self.network(x)

        return final_state
