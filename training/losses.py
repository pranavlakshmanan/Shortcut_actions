import torch
import torch.nn.functional as F
import numpy as np

def velocity_matching_loss(velocity_pred, velocity_true):
    """
    L_v = ||s_θ(s_t, a, t, d) - v_avg_true||^2

    Compares model's predicted average velocity over small step d
    with actual average velocity from physics simulation.

    Args:
        velocity_pred: (batch, state_dim) - Model predicted average velocity s_θ(s,a,t,d)
        velocity_true: (batch, state_dim) - True average velocity over step d

    Returns:
        loss: scalar
    """
    return F.mse_loss(velocity_pred, velocity_true)


def compute_true_average_velocity(env, state, actions, step_size):
    """
    Compute actual average velocity over step_size using physics simulation

    This implements: v_avg = (s(t+d) - s(t)) / d
    where s(t+d) comes from actual physics simulation.

    Args:
        env: Physics environment
        state: (batch, state_dim) - Current state
        actions: (batch, seq_len, action_dim) - Action sequence
        step_size: (batch, 1) - Time step size

    Returns:
        velocity_avg: (batch, state_dim) - True average velocity
    """
    batch_size = state.shape[0]
    velocity_avg = torch.zeros_like(state)

    for i in range(batch_size):
        # Extract single sample
        current_state = state[i].detach().cpu().numpy()
        action_seq = actions[i].detach().cpu().numpy()
        dt = step_size[i].item()

        # Store current environment state and simulate
        original_state = getattr(env, 'state', None)

        # Set environment to current state and simulate
        if hasattr(env, 'particles'):
            # For multi-particle environment
            env.clear_particles()
            x, y, vx, vy = current_state
            env.add_particle(x, y, vx, vy, mass=getattr(env, 'mass', 1.0), radius=0.15)
        else:
            # For single particle environment
            env.state = current_state

        # Simulate for the step size duration using action sequence
        num_sub_steps = max(1, int(dt / env.dt))
        for step_idx in range(num_sub_steps):
            # Use actions from sequence, cycling if needed
            action_idx = min(step_idx, len(action_seq) - 1)
            action = action_seq[action_idx] if len(action_seq) > 0 else np.zeros(env.action_dim)
            next_state, _, _ = env.step(action)

        # Compute average velocity
        displacement = next_state - current_state
        avg_velocity = displacement / dt
        velocity_avg[i] = torch.tensor(avg_velocity, dtype=state.dtype, device=state.device)

        # Restore original state
        if original_state is not None:
            env.state = original_state

    return velocity_avg


def self_consistency_loss(model, state, action_seq, time, step_size):
    """
    L_sc = ||s(x_t, t, 2d) - [s(x_t, t, d) + s(x_{t+d}, t+d, d)]/2||^2

    Paper Equation 4: The shortcut velocity for a 2d jump should equal
    the average of shortcut velocities for two d jumps.

    Args:
        model: ShortcutPredictor
        state: (batch, state_dim)
        action_seq: (batch, seq_len, action_dim)
        time: (batch, 1)
        step_size: (batch, 1)

    Returns:
        loss: scalar
    """
    # Get shortcut VELOCITY for one big jump of 2d
    s_2d = model.velocity_net(state, action_seq, time, 2 * step_size)

    # Get shortcut VELOCITY for first small jump of d
    s_d = model.velocity_net(state, action_seq, time, step_size)

    # Compute intermediate state after first jump
    state_intermediate = state + s_d * step_size

    # Split action sequence for second jump (use second half)
    seq_len = action_seq.shape[1]
    mid_point = seq_len // 2
    action_seq_second = action_seq[:, mid_point:, :]

    # Get shortcut VELOCITY for second small jump of d from intermediate state
    s_d_second = model.velocity_net(state_intermediate, action_seq_second,
                                    time + step_size, step_size)

    # Average of the two shortcut velocities (as per paper equation 4)
    s_target = (s_d + s_d_second) / 2.0

    # Compare shortcut velocities, not final states
    return F.mse_loss(s_2d, s_target)


def velocity_magnitude_loss(velocity_pred, velocity_true):
    """
    Option A: Velocity Magnitude Loss

    Forces the model to predict velocities with correct magnitudes, preventing
    the zero-velocity problem identified in diagnostics.

    L_v_mag = ||v_pred|| - ||v_true||²

    Args:
        velocity_pred: (batch, state_dim) - Predicted velocity
        velocity_true: (batch, state_dim) - True velocity

    Returns:
        loss: scalar - Magnitude difference loss
    """
    # Compute velocity magnitudes (L2 norm)
    mag_pred = torch.norm(velocity_pred, dim=1)  # (batch,)
    mag_true = torch.norm(velocity_true, dim=1)   # (batch,)

    # MSE loss on magnitudes
    return F.mse_loss(mag_pred, mag_true)


def cascaded_self_consistency_loss(model, state, action_seq, time, step_size):
    """
    Option B: Cascaded Self-Consistency Loss

    Enforces consistency at MULTIPLE scales, not just 2d = d + d.
    This teaches the model compositional reasoning across different time scales.

    Tests:
    - 4d = d + d + d + d  (four steps)
    - 8d = 2d + 2d + 2d + 2d (four 2d steps)
    - 4d = 2d + 2d (two 2d steps)

    Args:
        model: ShortcutPredictor
        state: (batch, state_dim)
        action_seq: (batch, seq_len, action_dim)
        time: (batch, 1)
        step_size: (batch, 1) - Base step size (d)

    Returns:
        loss: scalar - Combined cascaded consistency loss
    """
    batch_size = state.shape[0]
    device = state.device

    total_loss = torch.tensor(0.0, device=device)
    num_terms = 0

    # Test 1: 4d = d + d + d + d (four small steps)
    s_4d_direct = model.velocity_net(state, action_seq, time, 4 * step_size)

    # Chain four d-steps
    current_state = state
    current_time = time
    seq_len = action_seq.shape[1]
    actions_per_step = seq_len // 4

    for i in range(4):
        start_idx = i * actions_per_step
        end_idx = (i + 1) * actions_per_step
        action_subset = action_seq[:, start_idx:end_idx, :]

        velocity_d = model.velocity_net(current_state, action_subset, current_time, step_size)
        current_state = current_state + velocity_d * step_size
        current_time = current_time + step_size

    s_4d_chained = (current_state - state) / (4 * step_size)  # Average velocity

    total_loss = total_loss + F.mse_loss(s_4d_direct, s_4d_chained)
    num_terms += 1

    # Test 2: 4d = 2d + 2d (two 2d steps)
    s_2d_first = model.velocity_net(state, action_seq[:, :seq_len//2, :], time, 2 * step_size)
    state_mid = state + s_2d_first * 2 * step_size

    s_2d_second = model.velocity_net(state_mid, action_seq[:, seq_len//2:, :],
                                   time + 2 * step_size, 2 * step_size)

    s_4d_from_2d = (s_2d_first + s_2d_second) / 2.0  # Average velocity

    total_loss = total_loss + F.mse_loss(s_4d_direct, s_4d_from_2d)
    num_terms += 1

    # Test 3: 8d = 4d + 4d (if step size allows)
    if torch.all(8 * step_size <= 1.0):  # Only if within reasonable bounds
        s_8d_direct = model.velocity_net(state, action_seq, time, 8 * step_size)

        s_4d_first = model.velocity_net(state, action_seq[:, :seq_len//2, :], time, 4 * step_size)
        state_mid_8 = state + s_4d_first * 4 * step_size

        s_4d_second_8 = model.velocity_net(state_mid_8, action_seq[:, seq_len//2:, :],
                                         time + 4 * step_size, 4 * step_size)

        s_8d_from_4d = (s_4d_first + s_4d_second_8) / 2.0

        total_loss = total_loss + F.mse_loss(s_8d_direct, s_8d_from_4d)
        num_terms += 1

    # Average across all consistency tests
    return total_loss / num_terms if num_terms > 0 else total_loss