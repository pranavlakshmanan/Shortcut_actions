#!/usr/bin/env python3
"""
Train Sequential and Shortcut models on collision physics dataset

This training demonstrates clear performance differentiation:
1. Sequential baseline trained only on d=0.01 (fine timesteps)
2. Shortcut trained on d=[0.01, ..., 1.0] (bootstrap hierarchy)
3. Collision physics breaks naive single-step extrapolation
"""

import torch
import numpy as np
import yaml
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D
from models import VelocityFieldNet, ShortcutPredictor
from training.two_network_trainer import TwoNetworkTrainer
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    """Set random seeds for reproducible experiments"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class CollisionPhysicsDataset(Dataset):
    """Dataset for collision physics training"""

    def __init__(self, filename):
        filepath = Path('data') / filename
        with open(filepath, 'rb') as f:
            raw_data = pickle.load(f)

        self.samples = []
        for sample in tqdm(raw_data, desc=f"Loading {filename}"):
            trajectory = sample['trajectory']
            scenario = sample['scenario']

            for t_idx in range(len(trajectory) - 1):
                current_state = trajectory[t_idx]
                next_state = trajectory[t_idx + 1]

                force_pattern = scenario['force_pattern']
                if t_idx < len(force_pattern):
                    action = np.array(force_pattern[t_idx])
                else:
                    action = np.zeros(2)

                actions_padded = np.tile(action, (20, 1))
                dt = 0.01
                velocity = (next_state - current_state) / dt

                self.samples.append({
                    'state': torch.FloatTensor(current_state),
                    'actions': torch.FloatTensor(actions_padded),
                    'time': torch.FloatTensor([t_idx * dt]),
                    'dt': torch.FloatTensor([dt]),
                    'velocity': torch.FloatTensor(velocity),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class MultiTimestepCollisionDataset(Dataset):
    """Dataset that provides ground truth for multiple timestep sizes"""

    def __init__(self, pkl_file, bootstrap_levels=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]):
        """
        Args:
            pkl_file: Path to collision dataset pickle file
            bootstrap_levels: List of dt values to generate samples for
        """
        self.bootstrap_levels = bootstrap_levels
        self.samples = []

        filepath = Path('data') / pkl_file if not pkl_file.startswith('data/') else pkl_file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"Generating multi-timestep dataset from {len(data)} trajectories...")

        for sample_idx, sample in enumerate(tqdm(data, desc="Creating multi-timestep samples")):
            trajectory = sample['trajectory']
            scenario = sample['scenario']

            # For each bootstrap level, create samples
            for dt in bootstrap_levels:
                # Calculate number of fine timesteps per dt
                # Trajectory is at 0.01s intervals, so dt=0.02 means skip 2 steps
                dt_base = 0.01
                skip = max(1, int(dt / dt_base))

                # Create samples at this dt level
                for t_idx in range(0, len(trajectory) - skip, skip):
                    current_state = trajectory[t_idx]
                    future_state = trajectory[t_idx + skip]

                    # Get force pattern for this interval
                    force_pattern = scenario['force_pattern']

                    # Extract actions for this time interval
                    if t_idx < len(force_pattern):
                        # Use actions from this time window
                        action_window = force_pattern[t_idx:min(t_idx + skip, len(force_pattern))]
                        # Pad to fixed length (20 actions)
                        actions_padded = np.zeros((20, 2))
                        actions_padded[:len(action_window)] = action_window
                    else:
                        actions_padded = np.zeros((20, 2))

                    # Compute TRUE average velocity over this dt
                    actual_dt = skip * dt_base  # Actual time elapsed
                    velocity_true = (future_state - current_state) / actual_dt

                    self.samples.append({
                        'state': torch.FloatTensor(current_state),
                        'actions': torch.FloatTensor(actions_padded),
                        'time': torch.FloatTensor([t_idx * dt_base]),
                        'dt': torch.FloatTensor([actual_dt]),
                        'velocity': torch.FloatTensor(velocity_true),
                        'future_state': torch.FloatTensor(future_state),
                    })

        print(f"Generated {len(self.samples)} multi-timestep samples")

        # Print distribution of timesteps
        dt_counts = {}
        for sample in self.samples:
            dt_val = sample['dt'].item()
            dt_counts[dt_val] = dt_counts.get(dt_val, 0) + 1

        print("Timestep distribution:")
        for dt_val in sorted(dt_counts.keys()):
            print(f"  dt={dt_val:.2f}: {dt_counts[dt_val]} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class SparseMultiTimestepCollisionDataset(Dataset):
    """Dataset with ground truth at SPARSE supervised levels for grounded bootstrap hierarchy

    Provides ground truth supervision at: [0.01, 0.04, 0.08, 0.16, 0.64]
    Self-consistency will learn: [0.02, 0.32, 1.0]

    This enables:
    - Temporal extrapolation via self-consistency
    - Unified dynamics learning across scales
    - Sparse supervision efficiency (5 anchors vs 100 dense steps)
    """

    def __init__(self, pkl_file, supervised_levels=[0.01, 0.04, 0.08, 0.16, 0.64]):
        """
        Args:
            pkl_file: Path to collision dataset pickle file
            supervised_levels: Timesteps with ground truth supervision
        """
        self.supervised_levels = supervised_levels
        self.samples = []

        filepath = Path('data') / pkl_file if not pkl_file.startswith('data/') else pkl_file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"\n📊 Generating Sparse Multi-Timestep Dataset")
        print(f"   Source: {pkl_file}")
        print(f"   Trajectories: {len(data)}")
        print(f"   Supervised levels: {supervised_levels}")

        # Trajectory sampling rate
        dt_base = 0.01

        for sample_idx, sample in enumerate(tqdm(data, desc="Processing trajectories")):
            trajectory = sample['trajectory']
            scenario = sample['scenario']
            force_pattern = scenario['force_pattern']

            # For each supervised level, extract samples from trajectory
            for dt in supervised_levels:
                # Calculate skip interval
                # dt=0.01 -> skip=1, dt=0.04 -> skip=4, dt=0.08 -> skip=8, etc.
                skip = max(1, int(round(dt / dt_base)))
                actual_dt = skip * dt_base

                # Extract samples at this timestep interval
                max_start = len(trajectory) - skip
                if max_start <= 0:
                    continue

                for t_idx in range(0, max_start):  # Sample EVERY timestep, not every skip
                    current_state = trajectory[t_idx]
                    future_state = trajectory[t_idx + skip]

                    # Get force pattern for this time interval
                    if t_idx + skip <= len(force_pattern):
                        force_window = force_pattern[t_idx:t_idx + skip]
                    else:
                        force_window = force_pattern[t_idx:]

                    # Pad to max sequence length (20 actions)
                    actions_padded = np.zeros((20, 2))
                    action_len = min(len(force_window), 20)
                    actions_padded[:action_len] = force_window[:action_len]

                    # Compute TRUE average velocity over this dt interval
                    velocity_true = (future_state - current_state) / actual_dt

                    self.samples.append({
                        'state': torch.FloatTensor(current_state),
                        'actions': torch.FloatTensor(actions_padded),
                        'time': torch.FloatTensor([t_idx * dt_base]),
                        'dt': torch.FloatTensor([actual_dt]),
                        'velocity': torch.FloatTensor(velocity_true),
                        'future_state': torch.FloatTensor(future_state),
                    })

        print(f"✅ Generated {len(self.samples)} supervised samples")

        # Print distribution of samples across timesteps
        dt_counts = {}
        for sample in self.samples:
            dt_val = round(sample['dt'].item(), 2)
            dt_counts[dt_val] = dt_counts.get(dt_val, 0) + 1

        print("\n   Timestep distribution:")
        for dt_val in sorted(dt_counts.keys()):
            count = dt_counts[dt_val]
            pct = 100.0 * count / len(self.samples)
            print(f"   dt={dt_val:.2f}s: {count:,} samples ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collision_collate_fn(batch):
    """Custom collate function"""
    batched = {}
    for key in batch[0].keys():
        batched[key] = torch.stack([sample[key] for sample in batch])
    return batched

def create_collision_dataloader(filename, batch_size, shuffle=True):
    """Create DataLoader for collision physics dataset"""
    dataset = CollisionPhysicsDataset(filename)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     collate_fn=collision_collate_fn, num_workers=0)

def main():
    parser = argparse.ArgumentParser(description="Train models on collision physics")
    parser.add_argument("--config", type=str, default="configs/collision_physics_training.yaml",
                       help="Path to config file")
    parser.add_argument("--epochs", type=int, default=25,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--wandb", action="store_true", default=True,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="collision-physics-fix",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")

    args = parser.parse_args()

    # Load config first
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Weights & Biases
    if args.wandb:
        run_name = args.wandb_run_name or f"collision-fix-{args.epochs}ep-seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "config_file": args.config,
                "dataset_type": "collision_physics_70pct",
                "model_architecture": config['model'],
                "training_config": config['training'],
                "environment_config": config['environment'],
                "collision_bias": 0.7,
                "fix_description": "Fixed 37.8% -> 70% collision ratio for proper model differentiation"
            },
            tags=["collision-physics", "sequential-vs-shortcut", "data-fix", "70pct-collisions"],
            notes="Training with fixed collision dataset (70% collision scenarios) to demonstrate clear Sequential vs Shortcut differentiation"
        )
        print(f"📊 Weights & Biases initialized: {wandb.run.name}")
        print(f"   Project: {args.wandb_project}")
        print(f"   URL: {wandb.run.url}")
    else:
        print("📊 Weights & Biases disabled")

    print("=" * 80)
    print("COLLISION PHYSICS TRAINING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Random seed: {args.seed}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")

    print("\n" + "="*80)
    print("DATASET CREATION - GROUNDED BOOTSTRAP HIERARCHY")
    print("="*80)

    # Load configuration
    supervised_levels = config['training'].get('supervised_levels', [0.01, 0.04, 0.08, 0.16, 0.64])
    all_bootstrap_levels = config['training']['bootstrap_levels']
    sc_levels = [d for d in all_bootstrap_levels if d not in supervised_levels]

    print(f"\n🎯 Training Strategy:")
    print(f"   Sequential Model:")
    print(f"      - Supervised at: dt=0.01 ONLY")
    print(f"      - Supervision: DENSE (every timestep in trajectory)")
    print(f"      - Test at dt=1.0: 100x extrapolation (expected: FAIL)")
    print()
    print(f"   Shortcut Model:")
    print(f"      - Supervised at: {supervised_levels}")
    print(f"      - Self-consistency at: {sc_levels}")
    print(f"      - Supervision: SPARSE (5 anchor points)")
    print(f"      - Test at dt=1.0: 1.56x extrapolation from 0.64 (expected: SUCCESS)")
    print()

    # Sequential uses single-timestep dataset (dt=0.01 only)
    print("📦 Creating Sequential dataset (dt=0.01 only)...")
    sequential_train_dataset = CollisionPhysicsDataset('collision_train.pkl')
    sequential_val_dataset = CollisionPhysicsDataset('collision_val.pkl')

    print(f"   ✅ Sequential training: {len(sequential_train_dataset)} samples")
    print(f"   ✅ Sequential validation: {len(sequential_val_dataset)} samples")

    # Shortcut uses sparse multi-timestep dataset
    print(f"\n📦 Creating Shortcut dataset (supervised at {supervised_levels})...")
    shortcut_train_dataset = SparseMultiTimestepCollisionDataset(
        'collision_train.pkl',
        supervised_levels=supervised_levels
    )
    shortcut_val_dataset = SparseMultiTimestepCollisionDataset(
        'collision_val.pkl',
        supervised_levels=supervised_levels
    )

    print(f"   ✅ Shortcut training: {len(shortcut_train_dataset)} samples")
    print(f"   ✅ Shortcut validation: {len(shortcut_val_dataset)} samples")

    # Create dataloaders for Shortcut model
    train_loader = DataLoader(
        shortcut_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collision_collate_fn
    )

    val_loader = DataLoader(
        shortcut_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collision_collate_fn
    )

    print("\n✅ Datasets and dataloaders created")
    print("="*80)

    # Create models
    print("\n🏗️  Creating Models...")

    # Sequential baseline (VelocityFieldNet only)
    sequential_model = VelocityFieldNet(
        state_dim=config['model']['state_dim'],
        action_dim=config['model']['action_dim'],
        max_seq_len=config['model']['max_seq_len'],
        hidden_dims=config['model']['hidden_dims']
    ).to(device)

    # Shortcut predictor (wrapped VelocityFieldNet)
    velocity_net = VelocityFieldNet(
        state_dim=config['model']['state_dim'],
        action_dim=config['model']['action_dim'],
        max_seq_len=config['model']['max_seq_len'],
        hidden_dims=config['model']['hidden_dims']
    )
    shortcut_model = ShortcutPredictor(velocity_net).to(device)

    print(f"  Sequential params: {sum(p.numel() for p in sequential_model.parameters()):,}")
    print(f"  Shortcut params: {sum(p.numel() for p in shortcut_model.parameters()):,}")

    # Create environment for physics grounding
    env = PointMass2D(
        dt=config['environment']['dt'],
        mass=config['environment']['mass'],
        damping=config['environment']['damping']
    )
    
    # Set environment bounds and restitution to match data generation
    env.pos_bounds = [-config['environment']['boundaries'], config['environment']['boundaries']]
    env.restitution = config['environment']['collision_restitution']

    print(f"  Environment: dt={env.dt}, mass={env.mass}, damping={env.damping}")
    print(f"  Boundaries: {env.pos_bounds}, restitution={env.restitution}")

    # Create trainer
    print("\n🚂 Initializing Two-Network Trainer...")
    trainer = TwoNetworkTrainer(
        sequential_model=sequential_model,
        shortcut_model=shortcut_model,
        device=device,
        env=env,
        config=config
    )

    # Train both networks
    print(f"\n🎯 Training on Collision Physics...")
    print(f"   Sequential: Trained ONLY on d=0.01 (physics grounding)")
    print(f"   Shortcut: Bootstrap trained on d=[0.01, 0.02, ..., 1.0]")
    print(f"   Dataset: {len(train_loader)} train batches, {len(val_loader)} val batches")

    results = trainer.train_both_networks(
        train_loader=train_loader,                    # Shortcut multi-timestep data
        val_loader=val_loader,                        # Shortcut validation
        epochs=config['training']['epochs'],
        sequential_train_dataset=sequential_train_dataset,  # Sequential dt=0.01 only
        sequential_val_dataset=sequential_val_dataset       # Sequential validation
    )

    print("\n✅ Training Complete!")
    print(f"  Best Sequential Loss: {results['best_sequential_loss']:.6f}")
    print(f"  Best Shortcut Loss: {results['best_shortcut_loss']:.6f}")

    # Save models with metadata
    print("\n💾 Saving Collision-Trained Models...")

    # Sequential model
    torch.save({
        'model_state_dict': sequential_model.state_dict(),
        'config': config,
        'training_results': results,
        'dataset_type': 'collision_physics',
        'epoch': config['training']['epochs'],
        'collision_bias': 0.7,
        'model_type': 'sequential_baseline'
    }, 'experiments/sequential_baseline_model.pt')

    # Shortcut model
    torch.save({
        'model_state_dict': shortcut_model.state_dict(),
        'config': config,
        'training_results': results,
        'dataset_type': 'collision_physics',
        'epoch': config['training']['epochs'],
        'collision_bias': 0.7,
        'model_type': 'shortcut_bootstrap'
    }, 'experiments/shortcut_bootstrap_model.pt')

    print("  ✓ Sequential saved to: experiments/sequential_baseline_model.pt")
    print("  ✓ Shortcut saved to: experiments/shortcut_bootstrap_model.pt")

    # Save training summary
    training_summary = {
        'config': config,
        'args': vars(args),
        'results': results,
        'dataset_info': {
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'collision_bias': 0.7,
            'batch_size': args.batch_size
        }
    }

    torch.save(training_summary, 'experiments/collision_training_summary.pt')
    print("  ✓ Training summary saved to: experiments/collision_training_summary.pt")

    # Log final results to Weights & Biases
    if args.wandb and wandb.run is not None:
        print("\n📊 Logging final results to Weights & Biases...")

        # Log final metrics
        wandb.log({
            "final/sequential_loss": results['best_sequential_loss'],
            "final/shortcut_loss": results['best_shortcut_loss'],
            "final/loss_ratio": results['best_sequential_loss'] / results['best_shortcut_loss'],
            "final/training_epochs": config['training']['epochs'],
            "final/dataset_collision_ratio": 0.7,
        })

        # Log model artifacts
        sequential_artifact = wandb.Artifact(
            f"sequential-model-{wandb.run.id}",
            type="model",
            description="Sequential baseline model trained only on dt=0.01"
        )
        sequential_artifact.add_file('experiments/sequential_baseline_model.pt')
        wandb.log_artifact(sequential_artifact)

        shortcut_artifact = wandb.Artifact(
            f"shortcut-model-{wandb.run.id}",
            type="model",
            description="Shortcut model with bootstrap training on dt=[0.01, ..., 1.0]"
        )
        shortcut_artifact.add_file('experiments/shortcut_bootstrap_model.pt')
        wandb.log_artifact(shortcut_artifact)

        # Log training summary
        summary_artifact = wandb.Artifact(
            f"training-summary-{wandb.run.id}",
            type="results",
            description="Complete training configuration and results"
        )
        summary_artifact.add_file('experiments/collision_training_summary.pt')
        wandb.log_artifact(summary_artifact)

        # Mark as fixed dataset
        wandb.run.tags = wandb.run.tags + ["collision-data-fixed", "70pct-collisions"]

        print(f"  ✓ Results logged to: {wandb.run.url}")

        # Finish wandb run
        wandb.finish()

    print("\n" + "=" * 80)
    print("🎉 COLLISION PHYSICS TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run evaluation: python evaluate_models_fixed.py")
    print("  2. Run comprehensive evaluation: python evaluate_collision_physics.py")
    print("  3. Expected: Sequential Single-Step fails (15-20x error), Shortcut works (2-4x error)!")

if __name__ == "__main__":
    main()