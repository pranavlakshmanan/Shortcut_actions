#!/usr/bin/env python3
"""
Verification: Are state-based results too good to be true?
Tests on UNSEEN dt values and visualizes predictions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'state_based'))

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from state_based.state_predictor import StatePredictor
from envs import PointMass2D

def simulate_ground_truth(env, state, actions, dt):
    """Get ground truth final state"""
    env.reset()
    env.particles[0]["position"] = state[:2].astype(np.float32)
    env.particles[0]["velocity"] = state[2:].astype(np.float32)
    num_steps = int(dt / env.dt)
    collision_count = 0
    
    for step in range(num_steps):
        old_vx, old_vy = env.state[2], env.state[3]
        action_idx = min(step, len(actions) - 1)
        env.step(actions[action_idx])
        # Detect collision (velocity sign flip)
        if np.sign(env.state[2]) != np.sign(old_vx) and abs(old_vx) > 0.1:
            collision_count += 1
        if np.sign(env.state[3]) != np.sign(old_vy) and abs(old_vy) > 0.1:
            collision_count += 1
    
    return next_state.copy(), collision_count

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("VERIFICATION: Are state-based results real?")
    print("="*70)
    print(f"Device: {device}\n")

    # Load model
    print("Loading state-based model...")
    model = StatePredictor(
        state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64, 64, 64, 64]
    ).to(device)
    checkpoint = torch.load('experiments/state_based/state_best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}\n")

    # Load test data
    print("Loading test data...")
    with open('data/collision_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print(f"  Test samples: {len(test_data)}\n")

    # Check train/test overlap
    print("Checking train/test overlap...")
    with open('data/collision_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    train_states = set()
    for s in train_data[:1000]:
        state = tuple(np.round(s['scenario']['initial_state'], 4))
        train_states.add(state)
    
    overlap = 0
    for s in test_data[:100]:
        state = tuple(np.round(s['scenario']['initial_state'], 4))
        if state in train_states:
            overlap += 1
    print(f"  Overlap in first 100 test samples: {overlap}\n")

    env = PointMass2D(dt=0.01, damping=0.05)

    # TEST 1: Unseen dt values
    print("="*70)
    print("TEST 1: Performance on UNSEEN dt values")
    print("="*70)
    
    seen_dt = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
    unseen_dt = [0.03, 0.12, 0.25, 0.45, 0.75, 0.90]
    
    print(f"Seen during training: {seen_dt}")
    print(f"Testing on UNSEEN: {unseen_dt}\n")

    results = {}
    
    for dt in unseen_dt:
        errors = []
        collision_errors = []
        smooth_errors = []
        
        for sample in test_data[:50]:
            initial_state = sample['scenario']['initial_state']
            force_pattern = sample['scenario']['force_pattern']
            
            # Get ground truth
            true_state, collision_count = simulate_ground_truth(
                env, initial_state, force_pattern, dt
            )
            
            # Get prediction
            state_t = torch.FloatTensor(initial_state).unsqueeze(0).to(device)
            actions_t = torch.FloatTensor(force_pattern[:20]).unsqueeze(0).to(device)
            time_t = torch.zeros(1, 1).to(device)
            dt_t = torch.FloatTensor([[dt]]).to(device)
            
            with torch.no_grad():
                pred_state = model(state_t, actions_t, time_t, dt_t)
            
            pred_np = pred_state.cpu().numpy().squeeze()
            error = np.linalg.norm(pred_np - true_state)
            errors.append(error)
            
            if collision_count > 0:
                collision_errors.append(error)
            else:
                smooth_errors.append(error)
        
        results[dt] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'collision_mean': np.mean(collision_errors) if collision_errors else 0,
            'smooth_mean': np.mean(smooth_errors) if smooth_errors else 0,
            'n_collision': len(collision_errors),
            'n_smooth': len(smooth_errors)
        }
        
        print(f"dt={dt:.2f} (UNSEEN): error={results[dt]['mean']:.4f} ± {results[dt]['std']:.4f}")
        print(f"         Collision scenarios ({results[dt]['n_collision']}): {results[dt]['collision_mean']:.4f}")
        print(f"         Smooth scenarios ({results[dt]['n_smooth']}): {results[dt]['smooth_mean']:.4f}")

    # TEST 2: Compare seen vs unseen
    print("\n" + "="*70)
    print("TEST 2: Seen vs Unseen dt comparison")
    print("="*70)
    
    seen_errors = []
    for dt in [0.04, 0.16, 0.64]:  # Sample of seen dt values
        errors = []
        for sample in test_data[:50]:
            initial_state = sample['scenario']['initial_state']
            force_pattern = sample['scenario']['force_pattern']
            true_state, _ = simulate_ground_truth(env, initial_state, force_pattern, dt)
            
            state_t = torch.FloatTensor(initial_state).unsqueeze(0).to(device)
            actions_t = torch.FloatTensor(force_pattern[:20]).unsqueeze(0).to(device)
            time_t = torch.zeros(1, 1).to(device)
            dt_t = torch.FloatTensor([[dt]]).to(device)
            
            with torch.no_grad():
                pred_state = model(state_t, actions_t, time_t, dt_t)
            
            error = np.linalg.norm(pred_state.cpu().numpy().squeeze() - true_state)
            errors.append(error)
        
        seen_errors.append(np.mean(errors))
        print(f"dt={dt:.2f} (SEEN):   error={np.mean(errors):.4f}")

    unseen_errors = [results[dt]['mean'] for dt in unseen_dt]
    
    print(f"\nAverage error on SEEN dt:   {np.mean(seen_errors):.4f}")
    print(f"Average error on UNSEEN dt: {np.mean(unseen_errors):.4f}")
    print(f"Ratio (unseen/seen): {np.mean(unseen_errors)/np.mean(seen_errors):.2f}x")

    # TEST 3: Visualize predictions
    print("\n" + "="*70)
    print("TEST 3: Visualizing predictions vs ground truth")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Pick samples with collisions
    collision_samples = []
    for i, sample in enumerate(test_data[:100]):
        initial_state = sample['scenario']['initial_state']
        force_pattern = sample['scenario']['force_pattern']
        _, collision_count = simulate_ground_truth(env, initial_state, force_pattern, 1.0)
        if collision_count > 0:
            collision_samples.append((i, sample, collision_count))
        if len(collision_samples) >= 6:
            break
    
    for idx, (sample_idx, sample, n_col) in enumerate(collision_samples):
        initial_state = sample['scenario']['initial_state']
        force_pattern = sample['scenario']['force_pattern']
        
        # Simulate full trajectory
        env.state = initial_state.copy()
        trajectory = [initial_state.copy()]
        for step in range(100):  # 1 second
            action_idx = min(step, len(force_pattern) - 1)
            env.step(force_pattern[action_idx])
            trajectory.append(env.state.copy())
        trajectory = np.array(trajectory)
        
        # Get predictions at various dt
        predictions = {}
        for dt in [0.1, 0.25, 0.5, 0.75, 1.0]:
            state_t = torch.FloatTensor(initial_state).unsqueeze(0).to(device)
            actions_t = torch.FloatTensor(force_pattern[:20]).unsqueeze(0).to(device)
            time_t = torch.zeros(1, 1).to(device)
            dt_t = torch.FloatTensor([[dt]]).to(device)
            
            with torch.no_grad():
                pred = model(state_t, actions_t, time_t, dt_t)
            predictions[dt] = pred.cpu().numpy().squeeze()
        
        # Plot
        ax = axes[idx]
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5, label='True trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='blue', s=100, marker='s', label='True end')
        
        for dt, pred in predictions.items():
            ax.scatter(pred[0], pred[1], s=50, marker='x', label=f'Pred dt={dt}')
        
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=-5, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=-5, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(f'Sample {sample_idx} ({n_col} collisions)')
        ax.legend(fontsize=6)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('verification_trajectories.png', dpi=150)
    print(f"Saved trajectory visualization to verification_trajectories.png")

    # VERDICT
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    ratio = np.mean(unseen_errors) / np.mean(seen_errors)
    
    if ratio < 1.5:
        print("✅ RESULTS APPEAR GENUINE")
        print("   Model generalizes well to unseen dt values")
        print("   Error ratio (unseen/seen) < 1.5x")
    elif ratio < 3.0:
        print("⚠️  PARTIAL GENERALIZATION")
        print("   Model works on unseen dt but with degraded performance")
        print(f"   Error ratio: {ratio:.2f}x")
    else:
        print("❌ POOR GENERALIZATION")
        print("   Model may be memorizing training dt values")
        print(f"   Error ratio: {ratio:.2f}x")
    
    # Check if collision scenarios are harder
    all_collision = [results[dt]['collision_mean'] for dt in unseen_dt if results[dt]['collision_mean'] > 0]
    all_smooth = [results[dt]['smooth_mean'] for dt in unseen_dt if results[dt]['smooth_mean'] > 0]
    
    if all_collision and all_smooth:
        print(f"\nCollision vs Smooth performance:")
        print(f"   Collision scenarios: {np.mean(all_collision):.4f}")
        print(f"   Smooth scenarios: {np.mean(all_smooth):.4f}")

if __name__ == "__main__":
    main()
