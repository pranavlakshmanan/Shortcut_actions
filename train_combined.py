#!/usr/bin/env python3
"""
Two-Network Training and Comparison Script

Implements the complete two-network evaluation system:
1. Sequential Baseline: Optimized for d=0.01 physics accuracy
2. Shortcut Predictor: Bootstrap trained across multiple time scales

This provides the fair comparison framework requested in the shortcuts paper analysis.
"""

import torch
import numpy as np
import yaml
import wandb
from pathlib import Path
import argparse

# Import modules
from envs import PointMass2D
from models import VelocityFieldNet, ShortcutPredictor
from training.two_network_trainer import TwoNetworkTrainer
from utils import generate_training_data, create_dataloader

def set_seed(seed):
    """Set random seeds for reproducible experiments"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_identical_networks(config, device):
    """Create identical network architectures for fair comparison"""

    # Sequential network (VelocityFieldNet only)
    sequential_net = VelocityFieldNet(
        state_dim=config['model']['state_dim'],
        action_dim=config['model']['action_dim'],
        max_seq_len=config['model']['max_seq_len'],
        hidden_dims=config['model']['hidden_dims']
    ).to(device)

    # Shortcut network (wrapped VelocityFieldNet)
    velocity_net = VelocityFieldNet(
        state_dim=config['model']['state_dim'],
        action_dim=config['model']['action_dim'],
        max_seq_len=config['model']['max_seq_len'],
        hidden_dims=config['model']['hidden_dims']
    )
    shortcut_net = ShortcutPredictor(velocity_net).to(device)

    return sequential_net, shortcut_net

def main():
    parser = argparse.ArgumentParser(description="Two-Network Training and Comparison")
    parser.add_argument("--config", type=str, default="configs/two_network_comparison.yaml",
                       help="Configuration file path")
    parser.add_argument("--multi_collision_data", type=str,
                       default="data/multi_collision_stress_test.pkl",
                       help="Multi-collision test dataset")

    args = parser.parse_args()

    print("🔄 Two-Network Training and Comparison Pipeline")
    print("="*70)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default bootstrap configuration...")
        config_path = Path("configs/shortcut_bootstrap.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    print(f"🎲 Random seed set to: {config['seed']}")

    # Device selection
    device_config = config.get('device', 'cpu')
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config

    print(f"💻 Using device: {device}")

    # Initialize wandb for two-network comparison
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', True):
        wandb.init(
            project=wandb_config.get('project', "shortcut-two-network-comparison"),
            entity=wandb_config.get('entity'),
            name=f"two-network-{config.get('seed', 42)}-{device}",
            config=config,
            tags=wandb_config.get('tags', ["two-network", "sequential-vs-shortcut"]),
            notes=wandb_config.get('notes', "Sequential baseline vs Shortcut predictor comparison")
        )

    # Create physics environment
    print("\n🔬 Creating Physics Environment...")
    env = PointMass2D(
        dt=config['environment']['dt'],
        mass=config['environment']['mass'],
        damping=config['environment']['damping']
    )
    print(f"   ✓ Environment: dt={env.dt}, mass={env.mass}, damping={env.damping}")

    # Generate training data
    print("\n📊 Generating Training Data...")
    train_data = generate_training_data(
        env,
        num_trajectories=config['training']['num_trajectories'],
        traj_length=config['training']['traj_length'],
        action_seq_len=config['training']['action_seq_len']
    )
    print(f"   ✓ Generated {len(train_data)} training samples")

    val_data = generate_training_data(
        env,
        num_trajectories=config['training']['val_trajectories'],
        traj_length=config['training']['traj_length'],
        action_seq_len=config['training']['action_seq_len']
    )
    print(f"   ✓ Generated {len(val_data)} validation samples")

    # Create data loaders
    train_loader = create_dataloader(train_data, batch_size=config['training']['batch_size'])
    val_loader = create_dataloader(val_data, batch_size=config['training']['batch_size'])

    # Create identical network architectures
    print("\n🧠 Creating Identical Network Architectures...")
    sequential_model, shortcut_model = create_identical_networks(config, device)

    sequential_params = sum(p.numel() for p in sequential_model.parameters())
    shortcut_params = sum(p.numel() for p in shortcut_model.parameters())

    print(f"   ✓ Sequential Model: {sequential_params:,} parameters")
    print(f"   ✓ Shortcut Model: {shortcut_params:,} parameters")
    print(f"   ✓ Architecture identical: {sequential_params == shortcut_params}")

    # Create two-network trainer
    print(f"\n🎯 Setting up Two-Network Training System...")
    trainer = TwoNetworkTrainer(
        sequential_model=sequential_model,
        shortcut_model=shortcut_model,
        device=device,
        env=env,
        config=config
    )

    # Training phase
    epochs = config['training']['epochs']
    print(f"\n🚂 Starting Two-Network Training ({epochs} epochs)...")
    print("   📈 Sequential: Pure velocity matching (d=0.01 physics grounding)")
    print("   📈 Shortcut: Bootstrap hierarchy (d=[0.01, 0.02, ..., 1.0] temporal scaling)")

    training_results = trainer.train_both_networks(train_loader, val_loader, epochs)

    print(f"\n✅ Training completed!")
    print(f"   Sequential best val loss: {training_results['best_sequential_loss']:.6f}")
    print(f"   Shortcut best val loss: {training_results['best_shortcut_loss']:.6f}")

    # Evaluation phase
    print(f"\n🧪 Starting Comprehensive Evaluation...")

    # Load multi-collision test data if available
    test_data = val_data  # Default to validation data
    multi_collision_path = Path(args.multi_collision_data)
    if multi_collision_path.exists():
        print(f"   📁 Loading multi-collision stress-test data: {multi_collision_path}")
        try:
            from data_generation.multi_collision_scenarios import load_multi_collision_dataset
            stress_dataset = load_multi_collision_dataset(multi_collision_path.name)
            # Convert to format expected by evaluation
            test_data = []
            for sample in stress_dataset[:500]:  # Use subset for evaluation speed
                test_data.append({
                    'state': sample['scenario']['initial_state'],
                    'actions': sample['scenario']['force_pattern'],
                    'time': 0.0,  # Default time
                    'velocity': np.zeros(2)  # Will be computed by physics
                })
            print(f"   ✓ Using {len(test_data)} multi-collision test scenarios")
        except Exception as e:
            print(f"   ⚠️  Could not load multi-collision data: {e}")
            print("   ✓ Using validation data for evaluation")
    else:
        print(f"   ✓ Using validation data for evaluation ({len(test_data)} samples)")

    # Run comprehensive comparison
    d_levels = config.get('evaluation', {}).get('test_d_levels', [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    evaluation_results = trainer.evaluate_comparison(test_data, d_levels=d_levels)

    # Generate comparison report
    print(f"\n📊 Generating Comparison Report...")
    report = trainer.generate_comparison_report(evaluation_results)

    # Log final results to wandb
    if wandb.run:
        final_metrics = {}

        # Log accuracy results
        for d in evaluation_results['sequential']:
            final_metrics.update({
                f"final/sequential_error_d_{d}": evaluation_results['sequential'][d]['mean_error'],
                f"final/shortcut_error_d_{d}": evaluation_results['shortcut'][d]['mean_error'],
                f"final/speedup_d_{d}": evaluation_results['speedup_analysis'][d]['speedup'],
                f"final/accuracy_ratio_d_{d}": evaluation_results['speedup_analysis'][d]['accuracy_ratio'],
                f"final/efficiency_score_d_{d}": evaluation_results['speedup_analysis'][d]['efficiency_score']
            })

        # Overall summary metrics
        all_speedups = [evaluation_results['speedup_analysis'][d]['speedup'] for d in evaluation_results['speedup_analysis']]
        all_accuracy_ratios = [evaluation_results['speedup_analysis'][d]['accuracy_ratio'] for d in evaluation_results['speedup_analysis']]

        final_metrics.update({
            "final/avg_speedup": np.mean(all_speedups),
            "final/avg_accuracy_ratio": np.mean(all_accuracy_ratios),
            "final/successful_shortcuts": sum(1 for r in all_accuracy_ratios if r < 2.0),
            "final/total_test_cases": len(all_accuracy_ratios),
            "final/sequential_best_loss": training_results['best_sequential_loss'],
            "final/shortcut_best_loss": training_results['best_shortcut_loss']
        })

        wandb.log(final_metrics)

        # Upload models and report
        sequential_artifact = wandb.Artifact("sequential_baseline_model", type="model")
        sequential_artifact.add_file('experiments/sequential_baseline_model.pt')
        wandb.log_artifact(sequential_artifact)

        shortcut_artifact = wandb.Artifact("shortcut_predictor_model", type="model")
        shortcut_artifact.add_file('experiments/shortcut_bootstrap_model.pt')
        wandb.log_artifact(shortcut_artifact)

        report_artifact = wandb.Artifact("comparison_report", type="report")
        report_artifact.add_file('experiments/two_network_comparison.txt')
        wandb.log_artifact(report_artifact)

    # Final summary
    print(f"\n🎉 Two-Network Comparison Pipeline Completed!")
    print("="*50)
    print(f"📁 Results saved to:")
    print(f"   • Sequential model: experiments/sequential_best_model.pt")
    print(f"   • Shortcut model: experiments/shortcut_best_model.pt")
    print(f"   • Comparison report: experiments/two_network_comparison.txt")

    if wandb.run:
        print(f"   • Wandb run: {wandb.run.url}")
        wandb.finish()

    # Success criteria check
    avg_speedup = np.mean(all_speedups) if all_speedups else 0
    avg_accuracy_ratio = np.mean(all_accuracy_ratios) if all_accuracy_ratios else float('inf')

    print(f"\n🎯 Success Criteria Analysis:")
    print(f"   Average speedup: {avg_speedup:.1f}x (target: >10x)")
    print(f"   Average accuracy ratio: {avg_accuracy_ratio:.3f} (target: <1.5)")

    if avg_speedup > 10 and avg_accuracy_ratio < 1.5:
        print(f"   ✅ SUCCESS: Shortcut models achieve significant speedup with acceptable accuracy!")
    elif avg_speedup > 10:
        print(f"   ⚠️  PARTIAL SUCCESS: Good speedup but accuracy needs improvement")
    elif avg_accuracy_ratio < 1.5:
        print(f"   ⚠️  PARTIAL SUCCESS: Good accuracy but speedup needs improvement")
    else:
        print(f"   ❌ NEEDS WORK: Both speedup and accuracy need improvement")

if __name__ == "__main__":
    main()