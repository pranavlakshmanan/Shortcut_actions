import torch
import torch.nn as nn

class ShortcutPredictor(nn.Module):
    """Shortcut-based state predictor using velocity field"""

    def __init__(self, velocity_net):
        super().__init__()
        self.velocity_net = velocity_net

    def predict_one_step(self, state, action_seq, time, step_size):
        """
        Shortcut prediction: s(t+d) ≈ s(t) + v(s(t), a, t) * d

        Args:
            state: (batch, state_dim)
            action_seq: (batch, seq_len, action_dim)
            time: (batch, 1)
            step_size: (batch, 1)

        Returns:
            predicted_state: (batch, state_dim)
        """
        velocity = self.velocity_net(state, action_seq, time, step_size)
        predicted_state = state + velocity * step_size
        return predicted_state

    def predict_multi_step(self, state, action_seq, time, num_steps, step_size):
        """
        Sequential rollout (baseline comparison)

        Args:
            state: (batch, state_dim)
            action_seq: (batch, total_seq_len, action_dim)
            time: (batch, 1)
            num_steps: int
            step_size: (batch, 1)

        Returns:
            predicted_state: (batch, state_dim)
        """
        current_state = state
        current_time = time

        actions_per_step = action_seq.shape[1] // num_steps

        for i in range(num_steps):
            # Get actions for this step
            start_idx = i * actions_per_step
            end_idx = (i + 1) * actions_per_step
            current_actions = action_seq[:, start_idx:end_idx, :]

            velocity = self.velocity_net(current_state, current_actions,
                                        current_time, step_size)
            current_state = current_state + velocity * step_size
            current_time = current_time + step_size

        return current_state