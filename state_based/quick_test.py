#!/usr/bin/env python3
"""
Quick collision test: 5-minute sanity check

Tests gradient stability on a single hard collision scenario.
Compares velocity-based vs state-based prediction.

Expected result:
- State-based loss decreases smoothly
- Velocity-based loss oscillates/spikes

If this passes, proceed to full training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from models.velocity_field import VelocityFieldNet
from state_predictor import StatePredictor
from envs import PointMass2D


def simulate_collision_scenario(env, state, actions, dt, device='cpu'):
    """Simulate physics to get ground truth final state"""
    # Set environment to initial state
    env.state = state.cpu().numpy().squeeze()

    # Simulate for dt duration
    num_steps = int(dt / env.dt)
    for step in range(num_steps):
        # Use actions from sequence (cycling if needed)
        action_idx = min(step, actions.shape[1] - 1)
        action = actions[0, action_idx].cpu().numpy()
        next_state, _, _ = env.step(action)

    # Return final state
    final_state = torch.FloatTensor(env.state).unsqueeze(0).to(device)
    return final_state


def main():
    print("="*80)
    print("QUICK COLLISION TEST: Gradient Stability Check")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ===== CREATE HARD COLLISION SCENARIO =====
    # Particle near right wall (x=4.5), moving right (vx=2.0)
    # Will collide with wall at x=5.0 and bounce back
    state = torch.FloatTensor([[4.5, 0.0, 2.0, 0.0]]).to(device)
    actions = torch.zeros(1, 10, 2).to(device)  # No external forces
    time = torch.zeros(1, 1).to(device)
    dt = torch.FloatTensor([[1.0]]).to(device)

    print("Collision Scenario:")
    print(f"  Initial state: x={state[0,0]:.2f}, y={state[0,1]:.2f}, "
          f"vx={state[0,2]:.2f}, vy={state[0,3]:.2f}")
    print(f"  Timestep: dt={dt.item():.2f}s")
    print(f"  Expected: Particle hits wall, bounces, possibly multiple times\n")

    # Get ground truth from physics simulator
    env = PointMass2D(dt=0.01, damping=0.05)
    true_final_state = simulate_collision_scenario(env, state, actions, dt.item(), device)

    print(f"  Ground truth final state: x={true_final_state[0,0]:.2f}, "
          f"y={true_final_state[0,1]:.2f}, "
          f"vx={true_final_state[0,2]:.2f}, vy={true_final_state[0,3]:.2f}\n")

    # ===== CREATE BOTH MODELS =====
    print("Creating models (identical architecture)...")
    vel_model = VelocityFieldNet(
        state_dim=4,
        action_dim=2,
        max_seq_len=10,
        hidden_dims=[64, 64, 64, 64]
    ).to(device)

    state_model = StatePredictor(
        state_dim=4,
        action_dim=2,
        max_seq_len=10,
        hidden_dims=[64, 64, 64, 64]
    ).to(device)

    vel_params = sum(p.numel() for p in vel_model.parameters())
    state_params = sum(p.numel() for p in state_model.parameters())
    print(f"  Velocity model: {vel_params:,} parameters")
    print(f"  State model: {state_params:,} parameters")
    print(f"  Identical: {vel_params == state_params}\n")

    # ===== OPTIMIZERS =====
    vel_optimizer = torch.optim.Adam(vel_model.parameters(), lr=0.001)
    state_optimizer = torch.optim.Adam(state_model.parameters(), lr=0.001)

    # ===== TRAINING LOOP (200 iterations) =====
    print("Training both models on this single collision...")
    print("(200 iterations, should take ~30 seconds)\n")

    num_iterations = 200
    vel_losses = []
    state_losses = []

    for i in range(num_iterations):
        # ===== VELOCITY-BASED MODEL =====
        vel_model.train()
        v_pred = vel_model(state, actions, time, dt)
        s_pred_vel = state + v_pred * dt
        loss_vel = F.mse_loss(s_pred_vel, true_final_state)

        vel_optimizer.zero_grad()
        loss_vel.backward()
        vel_optimizer.step()

        vel_losses.append(loss_vel.item())

        # ===== STATE-BASED MODEL =====
        state_model.train()
        s_pred_state = state_model(state, actions, time, dt)
        loss_state = F.mse_loss(s_pred_state, true_final_state)

        state_optimizer.zero_grad()
        loss_state.backward()
        state_optimizer.step()

        state_losses.append(loss_state.item())

        # Progress
        if (i + 1) % 50 == 0:
            print(f"Iter {i+1:3d}: Velocity loss={loss_vel.item():.6f}, "
                  f"State loss={loss_state.item():.6f}")

    # ===== ANALYSIS =====
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Final losses
    print(f"\nFinal losses:")
    print(f"  Velocity-based: {vel_losses[-1]:.6f}")
    print(f"  State-based:    {state_losses[-1]:.6f}")

    # Loss reduction
    vel_reduction = (vel_losses[0] - vel_losses[-1]) / vel_losses[0] * 100
    state_reduction = (state_losses[0] - state_losses[-1]) / state_losses[0] * 100
    print(f"\nLoss reduction:")
    print(f"  Velocity-based: {vel_reduction:.1f}%")
    print(f"  State-based:    {state_reduction:.1f}%")

    # Stability (std dev of losses in last 50 iterations)
    vel_std = np.std(vel_losses[-50:])
    state_std = np.std(state_losses[-50:])
    print(f"\nLoss stability (std dev, last 50 iters):")
    print(f"  Velocity-based: {vel_std:.6f}")
    print(f"  State-based:    {state_std:.6f}")

    # ===== VISUALIZATION =====
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Full training curves
    axes[0].plot(vel_losses, label='Velocity-based', alpha=0.7, linewidth=2)
    axes[0].plot(state_losses, label='State-based', alpha=0.7, linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Curves: Velocity vs State')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Plot 2: Last 100 iterations (zoom in)
    axes[1].plot(range(100, 200), vel_losses[100:], label='Velocity-based', alpha=0.7, linewidth=2)
    axes[1].plot(range(100, 200), state_losses[100:], label='State-based', alpha=0.7, linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Last 100 Iterations (Detail)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(__file__).parent / 'quick_test_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # ===== DECISION =====
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)

    success_criteria = [
        ("State loss decreases smoothly", state_std < vel_std * 0.5),
        ("State loss converges better", state_losses[-1] < vel_losses[-1]),
        ("State loss reduces more", state_reduction > vel_reduction * 0.8)
    ]

    passed = sum([criterion[1] for criterion in success_criteria])
    total = len(success_criteria)

    print(f"\nSuccess criteria ({passed}/{total} passed):")
    for desc, passed_flag in success_criteria:
        status = "✓" if passed_flag else "✗"
        print(f"  {status} {desc}")

    if passed >= 2:
        print("\n✅ SUCCESS: State-based approach shows better gradient stability!")
        print("   → Proceed to full training with train_state_shortcut.py")
        return 0
    else:
        print("\n❌ FAILURE: State-based approach did not show clear advantage")
        print("   → Debug before scaling to full training")
        return 1


if __name__ == "__main__":
    exit(main())
