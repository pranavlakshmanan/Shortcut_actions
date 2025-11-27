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

    # Test at MULTIPLE dt values to find where differentiation is strongest
    test_dts = [0.05, 0.1, 0.2, 0.5, 1.0]
    results_by_dt = {}

    for test_dt in test_dts:
        print(f"\n🎯 Evaluating on COLLISION scenarios (dt={test_dt})...")
        
        results = {
            'sequential_collision_error': [],
            'shortcut_collision_error': []
        }
        
        for sample in tqdm(collision_samples[:50], desc=f"dt={test_dt}"):
            trajectory = sample['trajectory']
            
            # For smaller dt, sample from trajectory at appropriate intervals
            # dt=0.05 means 5 steps (since base dt=0.01), dt=0.1 means 10 steps, etc.
            steps = int(test_dt / 0.01)
            
            # Make sure we have enough trajectory length
            if len(trajectory) <= steps:
                continue
            
            # Test from beginning of trajectory
            initial_state = torch.FloatTensor(trajectory[0]).unsqueeze(0).to(device)
            true_final = trajectory[steps]
            
            # Get force pattern for this interval (pad to max_seq_len=20)
            force_pattern_full = sample['scenario']['force_pattern']
            force_window = force_pattern_full[:min(steps, len(force_pattern_full))]
            force_pattern_padded = np.zeros((20, 2))
            # Only copy up to max_seq_len=20
            copy_len = min(len(force_window), 20)
            force_pattern_padded[:copy_len] = force_window[:copy_len]
            force_pattern = torch.FloatTensor(force_pattern_padded).unsqueeze(0).to(device)
            
            dt_tensor = torch.FloatTensor([[test_dt]]).to(device)
            time = torch.FloatTensor([[0.0]]).to(device)
            
            # Sequential prediction (out-of-distribution for dt > 0.01)
            with torch.no_grad():
                seq_velocity = sequential_model(initial_state, force_pattern, time, dt_tensor)
                seq_pred = initial_state + seq_velocity * dt_tensor
                seq_error = np.linalg.norm(seq_pred.cpu().numpy() - true_final)
                results['sequential_collision_error'].append(seq_error)
            
            # Shortcut prediction (in-distribution)
            with torch.no_grad():
                short_velocity = shortcut_model.velocity_net(initial_state, force_pattern, time, dt_tensor)
                short_pred = initial_state + short_velocity * dt_tensor
                short_error = np.linalg.norm(short_pred.cpu().numpy() - true_final)
                results['shortcut_collision_error'].append(short_error)
        
        # Store results for this dt
        if results['sequential_collision_error']:
            seq_mean = np.mean(results['sequential_collision_error'])
            short_mean = np.mean(results['shortcut_collision_error'])
            ratio = seq_mean / short_mean if short_mean > 0 else float('inf')
            
            results_by_dt[test_dt] = {
                'sequential_error': seq_mean,
                'shortcut_error': short_mean,
                'ratio': ratio,
                'num_samples': len(results['sequential_collision_error'])
            }
            
            print(f"  Sequential error: {seq_mean:.4f}")
            print(f"  Shortcut error: {short_mean:.4f}")
            print(f"  Ratio (Sequential/Shortcut): {ratio:.1f}x")
            
            if ratio > 5.0:
                print(f"  ✅ Strong differentiation at dt={test_dt}!")
            elif ratio > 2.0:
                print(f"  ⚠️  Moderate differentiation at dt={test_dt}")
            else:
                print(f"  ❌ Weak differentiation at dt={test_dt}")

    # Print summary across all dt values
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL dt VALUES")
    print("="*80)
    print(f"{'dt':>6} | {'Sequential':>12} | {'Shortcut':>12} | {'Ratio':>8} | {'Samples':>8}")
    print("-" * 80)
    for dt_val in test_dts:
        if dt_val in results_by_dt:
            r = results_by_dt[dt_val]
            print(f"{dt_val:6.2f} | {r['sequential_error']:12.4f} | {r['shortcut_error']:12.4f} | {r['ratio']:8.1f}x | {r['num_samples']:8d}")
    print("="*80)
    
    # Find best differentiation
    best_dt = max(results_by_dt.keys(), key=lambda d: results_by_dt[d]['ratio'])
    best_ratio = results_by_dt[best_dt]['ratio']
    print(f"\n🎯 Best differentiation: dt={best_dt} with {best_ratio:.1f}x ratio")
    
    if best_ratio > 5.0:
        print("✅ SUCCESS: Clear differentiation achieved!")
    else:
        print("❌ FAILURE: Models still performing similarly across all dt values")

    return results_by_dt

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

    # Log results to wandb (use dt=1.0 as primary metric)
    if args.wandb and wandb.run is not None and 1.0 in results:
        dt_1_results = results[1.0]
        seq_mean = dt_1_results['sequential_error']
        short_mean = dt_1_results['shortcut_error']
        ratio = dt_1_results['ratio']

        wandb.log({
            "evaluation/sequential_collision_error_mean": seq_mean,
            "evaluation/shortcut_collision_error_mean": short_mean,
            "evaluation/error_ratio_sequential_vs_shortcut": ratio,
            "evaluation/differentiation_success": ratio > 5.0,
            "evaluation/test_dt": 1.0,
            "evaluation/collision_scenarios_tested": dt_1_results['num_samples']
        })
        
        # Log all dt results
        for dt_val, dt_results in results.items():
            wandb.log({
                f"evaluation/dt_{dt_val}_sequential_error": dt_results['sequential_error'],
                f"evaluation/dt_{dt_val}_shortcut_error": dt_results['shortcut_error'],
                f"evaluation/dt_{dt_val}_ratio": dt_results['ratio']
            })

        # Create summary table for all dt values
        table_data = []
        for dt_val in sorted(results.keys()):
            dt_res = results[dt_val]
            table_data.append([
                f"dt={dt_val}",
                f"{dt_res['sequential_error']:.4f}",
                f"{dt_res['shortcut_error']:.4f}",
                f"{dt_res['ratio']:.2f}x"
            ])
        
        wandb.log({
            "evaluation_summary": wandb.Table(
                columns=["Timestep", "Sequential Error", "Shortcut Error", "Ratio"],
                data=table_data
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