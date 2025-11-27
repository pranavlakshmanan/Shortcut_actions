#!/usr/bin/env python3
"""
Evaluation script that properly tests Sequential vs Shortcut differentiation
Focus on collision scenarios where the difference should be most apparent
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import wandb
import argparse

sys.path.append(str(Path(__file__).parent))

from models import VelocityFieldNet, ShortcutPredictor
from envs import PointMass2D

def evaluate_on_collisions():
    """Evaluate models specifically on collision scenarios"""

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Sequential model
    sequential_checkpoint = torch.load('experiments/sequential_baseline_model.pt')
    sequential_model = VelocityFieldNet(
        state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64, 64, 64, 64]
    ).to(device)
    sequential_model.load_state_dict(sequential_checkpoint['model_state_dict'])
    sequential_model.eval()

    # Load Shortcut model
    shortcut_checkpoint = torch.load('experiments/shortcut_bootstrap_model.pt')
    velocity_net = VelocityFieldNet(
        state_dim=4, action_dim=2, max_seq_len=20, hidden_dims=[64, 64, 64, 64]
    )
    shortcut_model = ShortcutPredictor(velocity_net).to(device)
    shortcut_model.load_state_dict(shortcut_checkpoint['model_state_dict'])
    shortcut_model.eval()

    # Load test data
    with open('data/collision_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Separate collision vs smooth samples
    collision_samples = []
    smooth_samples = []

    for sample in test_data:
        trajectory = sample['trajectory']
        has_collision = False

        for i in range(1, len(trajectory)):
            vel_change = np.abs(trajectory[i][2:] - trajectory[i-1][2:])
            if np.any(vel_change > 1.0):
                has_collision = True
                break

        if has_collision:
            collision_samples.append(sample)
        else:
            smooth_samples.append(sample)

    print(f"\nTest Set Distribution:")
    print(f"  Collision samples: {len(collision_samples)}")
    print(f"  Smooth samples: {len(smooth_samples)}")

    # Test at dt=1.0 (100x larger than training for Sequential)
    results = {
        'sequential_collision_error': [],
        'shortcut_collision_error': [],
        'sequential_smooth_error': [],
        'shortcut_smooth_error': []
    }

    print("\n🎯 Evaluating on COLLISION scenarios (dt=1.0)...")
    for sample in tqdm(collision_samples[:50]):  # Test subset
        initial_state = torch.FloatTensor(sample['trajectory'][0]).unsqueeze(0).to(device)
        force_pattern = torch.FloatTensor(sample['scenario']['force_pattern']).unsqueeze(0).to(device)
        true_final = sample['trajectory'][-1]

        dt = torch.FloatTensor([[1.0]]).to(device)
        time = torch.FloatTensor([[0.0]]).to(device)

        # Sequential prediction (out-of-distribution)
        with torch.no_grad():
            seq_velocity = sequential_model(initial_state, force_pattern, time, dt)
            seq_pred = initial_state + seq_velocity * dt
            seq_error = np.linalg.norm(seq_pred.cpu().numpy() - true_final)
            results['sequential_collision_error'].append(seq_error)

        # Shortcut prediction (in-distribution)
        with torch.no_grad():
            short_velocity = shortcut_model.velocity_net(initial_state, force_pattern, time, dt)
            short_pred = initial_state + short_velocity * dt
            short_error = np.linalg.norm(short_pred.cpu().numpy() - true_final)
            results['shortcut_collision_error'].append(short_error)

    print("\n📊 Results on COLLISION scenarios:")
    seq_mean = np.mean(results['sequential_collision_error'])
    short_mean = np.mean(results['shortcut_collision_error'])
    ratio = seq_mean / short_mean

    print(f"  Sequential error: {seq_mean:.4f}")
    print(f"  Shortcut error: {short_mean:.4f}")
    print(f"  Ratio (Sequential/Shortcut): {ratio:.1f}x")

    if ratio > 5.0:
        print("  ✅ SUCCESS: Clear differentiation achieved!")
    else:
        print("  ❌ FAILURE: Models still performing similarly")

    return results

def main():
    """Main evaluation function with wandb integration"""
    parser = argparse.ArgumentParser(description="Evaluate collision physics models")
    parser.add_argument("--wandb", action="store_true", default=True,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="collision-physics-evaluation",
                       help="Weights & Biases project name")

    args = parser.parse_args()

    # Initialize wandb for evaluation tracking
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name="collision-evaluation-fixed-dataset",
            config={
                "dataset_type": "collision_physics_70pct",
                "evaluation_type": "sequential_vs_shortcut",
                "test_dt": 1.0,
                "collision_ratio": 0.7,
            },
            tags=["evaluation", "collision-physics", "data-fix", "dt-1.0"],
            notes="Evaluation of Sequential vs Shortcut models on fixed 70% collision dataset"
        )
        print(f"📊 Weights & Biases initialized: {wandb.run.name}")

    print("="*80)
    print("FIXED EVALUATION - COLLISION PHYSICS")
    print("="*80)

    results = evaluate_on_collisions()

    # Log results to wandb
    if args.wandb and wandb.run is not None:
        seq_mean = np.mean(results['sequential_collision_error'])
        short_mean = np.mean(results['shortcut_collision_error'])
        ratio = seq_mean / short_mean

        wandb.log({
            "evaluation/sequential_collision_error_mean": seq_mean,
            "evaluation/shortcut_collision_error_mean": short_mean,
            "evaluation/error_ratio_sequential_vs_shortcut": ratio,
            "evaluation/differentiation_success": ratio > 5.0,
            "evaluation/test_dt": 1.0,
            "evaluation/collision_scenarios_tested": len(results['sequential_collision_error'])
        })

        # Create summary table
        wandb.log({
            "evaluation_summary": wandb.Table(
                columns=["Model", "Mean Error", "Error Ratio", "Status"],
                data=[
                    ["Sequential", f"{seq_mean:.4f}", f"{ratio:.1f}x", "FAIL" if ratio > 5.0 else "SIMILAR"],
                    ["Shortcut", f"{short_mean:.4f}", "1.0x", "SUCCESS" if ratio > 5.0 else "SIMILAR"]
                ]
            )
        })

        print(f"📊 Results logged to: {wandb.run.url}")
        wandb.finish()

    print("\n" + "="*80)
    print("Expected after fix:")
    print("  - Sequential should FAIL on dt=1.0 (15-20x error)")
    print("  - Shortcut should WORK on dt=1.0 (2-4x error)")
    print("="*80)

if __name__ == "__main__":
    main()