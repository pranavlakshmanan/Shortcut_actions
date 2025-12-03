import torch
import torch.nn.functional as F

def state_matching_loss(pred_state, true_state):
    """
    L_state = ||s_pred - s_true||^2

    Direct supervised learning of final state.
    NO DIVISION BY dt! Clean gradient signal even during collisions.

    Args:
        pred_state: (batch, state_dim) - Predicted final state
        true_state: (batch, state_dim) - True final state from simulation

    Returns:
        loss: scalar
    """
    return F.mse_loss(pred_state, true_state)


def state_self_consistency_loss(model, state, action_seq, time, step_size):
    """
    L_sc = ||s_theta(s, a, t, 2d) - s_theta(s_theta(s, a, t, d), a', t+d, d)||^2

    Enforces: One big 2d jump should equal two sequential d jumps
    Teaches temporal composition at the state level

    Args:
        model: StatePredictor
        state: (batch, state_dim)
        action_seq: (batch, seq_len, action_dim)
        time: (batch, 1)
        step_size: (batch, 1)

    Returns:
        loss: scalar
    """
    batch_size = state.shape[0]
    seq_len = action_seq.shape[1]

    # One big jump of 2d
    s_2d = model(state, action_seq, time, 2 * step_size)

    # Split action sequence for two jumps
    mid_point = seq_len // 2
    action_seq_first = action_seq[:, :mid_point, :]
    action_seq_second = action_seq[:, mid_point:, :]

    # First small jump of d
    s_d1 = model(state, action_seq_first, time, step_size)

    # Second small jump of d from intermediate state
    # Use .detach() to avoid second-order gradients (faster, more stable)
    s_d2 = model(s_d1.detach(), action_seq_second, time + step_size, step_size)

    # Compare final states
    return F.mse_loss(s_2d, s_d2)
