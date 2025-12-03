#!/usr/bin/env python3
"""
Evaluation script for state-based shortcut predictor

Compares three models:
1. Sequential baseline (velocity model, autoregressive at dt=0.01)
2. Velocity-based shortcut (existing approach)
3. State-based shortcut (new approach)

Tests at multiple dt values: [0.05, 0.1, 0.2, 0.5, 1.0]
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
import argparse

from state_predictor import StatePredictor
from models.velocity_field import VelocityFieldNet
from models.shortcut_predictor import ShortcutPredictor
from envs import PointMass2D


def simulate_ground_truth(env, state, actions, dt):
    """Simulate physics to get ground truth final state"""
    env.reset()
    env.particles[0]["position"] = state[:2].astype(np.float32)
    env.particles[0]["velocity"] = state[2:].astype(np.float32)
    num_steps = int(dt / env.dt)

    for step in range(num_steps):
        action_idx = min(step, len(actions) - 1)
        action = actions[action_idx]
        next_state, _, _ = env.step(action)

    return next_state.copy()


def evaluate_sequential_model(model, test_data, dt, env, device):
    """Evaluate sequential baseline (autoregressive)"""
    errors = []

    for sample in tqdm(test_data, desc=f"Sequential dt={dt}", leave=False):
        initial_state = sample['scenario']['initial_state']
        force_pattern = sample['scenario']['force_pattern']

        # Autoregressive rollout at dt=0.01
        current_state = torch.FloatTensor(initial_state).unsqueeze(0).to(device)
        num_steps = int(dt / env.dt)

        for step in range(num_steps):
            action_idx = min(step, len(force_pattern) - 1)
            action = torch.FloatTensor(force_pattern[action_idx:action_idx+20]).unsqueeze(0).to(device)
            time = torch.zeros(1, 1).to(device)
            dt_tensor = torch.FloatTensor([[env.dt]]).to(device)

            with torch.no_grad():
                v_pred = model(current_state, action, time, dt_tensor)
                current_state = current_state + v_pred * dt_tensor

        pred_state = current_state.cpu().numpy().squeeze()
        true_state = simulate_ground_truth(env, initial_state, force_pattern, dt)
        error = np.linalg.norm(pred_state - true_state)
        errors.append(error)

    return np.mean(errors), np.std(errors)


def evaluate_velocity_shortcut(model, test_data, dt, env, device):
    """Evaluate velocity-based shortcut model"""
    errors = []

    for sample in tqdm(test_data, desc=f"Velocity dt={dt}", leave=False):
        initial_state = sample['scenario']['initial_state']
        force_pattern = sample['scenario']['force_pattern']

        state = torch.FloatTensor(initial_state).unsqueeze(0).to(device)
        actions = torch.FloatTensor(force_pattern[:20]).unsqueeze(0).to(device)
        time = torch.zeros(1, 1).to(device)
        dt_tensor = torch.FloatTensor([[dt]]).to(device)

        with torch.no_grad():
            v_pred = model.velocity_net(state, actions, time, dt_tensor)
            pred_state = (state + v_pred * dt_tensor).cpu().numpy().squeeze()

        true_state = simulate_ground_truth(env, initial_state, force_pattern, dt)
        error = np.linalg.norm(pred_state - true_state)
        errors.append(error)

    return np.mean(errors), np.std(errors)


def evaluate_state_shortcut(model, test_data, dt, env, device):
    """Evaluate state-based shortcut model"""
    errors = []

    for sample in tqdm(test_data, desc=f"State dt={dt}", leave=False):
        initial_state = sample['scenario']['initial_state']
        force_pattern = sample['scenario']['force_pattern']

        state = torch.FloatTensor(initial_state).unsqueeze(0).to(device)
        actions = torch.FloatTensor(force_pattern[:20]).unsqueeze(0).to(device)
        time = torch.zeros(1, 1).to(device)
        dt_tensor = torch.FloatTensor([[dt]]).to(device)

        with torch.no_grad():
            pred_state = model(state, actions, time, dt_tensor).cpu().numpy().squeeze()

        true_state = simulate_ground_truth(env, initial_state, force_pattern, dt)
        error = np.linalg.norm(pred_state - true_state)
        errors.append(error)

    return np.mean(errors), np.std(errors)


def main():
    parser = argparse.ArgumentParser(description="Evaluate state-based shortcut")
    parser.add_argument("--state_model", type=str, default='experiments/state_based/state_best_model.pt',
                       help="Path to trained state model")
    parser.add_argument("--velocity_model", type=str, default='experiments/shortcut_bootstrap_model.pt',
                       help="Path to trained velocity shortcut model")
    parser.add_argument("--sequential_model", type=str, default='experiments/sequential_baseline_model.pt',
                       help="Path to trained sequential model")
    parser.add_argument("--test_data", type=str, default='data/collision_test.pkl',
                       help="Path to test dataset")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of test samples to evaluate")
    parser.add_argument("--device", type=str, default='auto', help="Device")

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*80)
    print("STATE-BASED SHORTCUT EVALUATION")
    print("="*80)
    print(f"Device: {device}\n")

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, 'rb') as f:
        test_data = pickle.load(f)[:args.num_samples]
    print(f"  Loaded {len(test_data)} samples\n")

    # Create environment
    env = PointMass2D(dt=0.01, damping=0.05)

    # Load models
    print("Loading models...")

    # Sequential baseline
    print("  1. Sequential baseline...")
    sequential_net = VelocityFieldNet(
        state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64, 64, 64, 64]
    ).to(device)
    checkpoint = torch.load(args.sequential_model, map_location=device)
    sequential_net.load_state_dict(checkpoint['model_state_dict'])
    sequential_net.eval()

    # Velocity shortcut
    print("  2. Velocity-based shortcut...")
    velocity_net = VelocityFieldNet(
        state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64, 64, 64, 64]
    )
    velocity_shortcut = ShortcutPredictor(velocity_net).to(device)
    checkpoint = torch.load(args.velocity_model, map_location=device)
    velocity_shortcut.load_state_dict(checkpoint['model_state_dict'])
    velocity_shortcut.eval()

    # State shortcut
    print("  3. State-based shortcut...")
    state_shortcut = StatePredictor(
        state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64, 64, 64, 64]
    ).to(device)
    checkpoint = torch.load(args.state_model, map_location=device)
    state_shortcut.load_state_dict(checkpoint['model_state_dict'])
    state_shortcut.eval()

    print("  All models loaded!\n")

    # Evaluate at multiple dt values
    dt_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = {}

    print("Evaluating models at multiple timesteps...")
    print("="*80)

    for dt in dt_values:
        print(f"\ndt = {dt:.2f}s")
        print("-" * 40)

        # Sequential
        seq_mean, seq_std = evaluate_sequential_model(
            sequential_net, test_data, dt, env, device
        )

        # Velocity shortcut
        vel_mean, vel_std = evaluate_velocity_shortcut(
            velocity_shortcut, test_data, dt, env, device
        )

        # State shortcut
        state_mean, state_std = evaluate_state_shortcut(
            state_shortcut, test_data, dt, env, device
        )

        results[dt] = {
            'sequential': {'mean': seq_mean, 'std': seq_std},
            'velocity': {'mean': vel_mean, 'std': vel_std},
            'state': {'mean': state_mean, 'std': state_std}
        }

        print(f"  Sequential:  {seq_mean:.4f} ± {seq_std:.4f}")
        print(f"  Velocity:    {vel_mean:.4f} ± {vel_std:.4f}")
        print(f"  State:       {state_mean:.4f} ± {state_std:.4f}")

        # Ratios
        vel_ratio = seq_mean / vel_mean if vel_mean > 0 else 0
        state_ratio = seq_mean / state_mean if state_mean > 0 else 0

        print(f"\n  Ratios (Sequential / Shortcut):")
        print(f"    Velocity: {vel_ratio:.2f}x")
        print(f"    State:    {state_ratio:.2f}x")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'dt':>6} | {'Sequential':>12} | {'Velocity':>12} | {'State':>12} | "
          f"{'Vel Ratio':>10} | {'State Ratio':>12}")
    print("-" * 80)

    for dt in dt_values:
        r = results[dt]
        vel_ratio = r['sequential']['mean'] / r['velocity']['mean'] if r['velocity']['mean'] > 0 else 0
        state_ratio = r['sequential']['mean'] / r['state']['mean'] if r['state']['mean'] > 0 else 0

        print(f"{dt:6.2f} | {r['sequential']['mean']:12.4f} | {r['velocity']['mean']:12.4f} | "
              f"{r['state']['mean']:12.4f} | {vel_ratio:10.2f}x | {state_ratio:12.2f}x")

    print("="*80)

    # Success analysis
    print("\nSUCCESS CRITERIA ANALYSIS")
    print("="*80)

    dt_1_state = results[1.0]['state']['mean']
    dt_1_vel = results[1.0]['velocity']['mean']
    dt_1_seq = results[1.0]['sequential']['mean']

    state_ratio_1 = dt_1_seq / dt_1_state if dt_1_state > 0 else 0
    vel_ratio_1 = dt_1_seq / dt_1_vel if dt_1_vel > 0 else 0

    print(f"\nAt dt=1.0:")
    print(f"  State-based error: {dt_1_state:.4f}")
    print(f"  State-based ratio: {state_ratio_1:.2f}x (target: >3.0x)")
    print(f"  Velocity-based ratio: {vel_ratio_1:.2f}x (baseline)")

    if state_ratio_1 > 3.0:
        print("\n✅ SUCCESS: State-based achieves >3x improvement at dt=1.0!")
    elif state_ratio_1 > vel_ratio_1 * 1.5:
        print("\n⚠️  PARTIAL SUCCESS: State-based better than velocity, but <3x")
    else:
        print("\n❌ FAILURE: State-based does not show clear advantage")

    # Flat error profile check
    dt_05_state = results[0.05]['state']['mean']
    error_ratio = dt_1_state / dt_05_state

    print(f"\nError profile (flat = good):")
    print(f"  State at dt=0.05: {dt_05_state:.4f}")
    print(f"  State at dt=1.0:  {dt_1_state:.4f}")
    print(f"  Ratio: {error_ratio:.2f}x (target: <2.0x)")

    if error_ratio < 2.0:
        print("  ✅ Flat error profile achieved!")
    else:
        print("  ❌ Error increases too much at large dt")


if __name__ == "__main__":
    main()
