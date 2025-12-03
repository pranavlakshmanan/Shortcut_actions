#!/usr/bin/env python3
"""
Full training script for state-based shortcut predictor

Uses bootstrap hierarchy with:
- Supervised levels: [0.01, 0.04, 0.08, 0.16, 0.64]
- Self-consistency levels: [0.02, 0.32, 1.0]

Trains for 40 epochs, logs to wandb
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import argparse
import wandb

from state_predictor import StatePredictor
from state_trainer import StateBootstrapTrainer
from envs import PointMass2D


class StateDataset(Dataset):
    """Dataset for state-based shortcut learning

    Generates training samples with final states at various dt values
    """

    def __init__(self, data_path, dt_levels, env):
        """
        Args:
            data_path: Path to collision dataset pkl
            dt_levels: List of timestep sizes to generate samples for
            env: Physics environment for simulation
        """
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.dt_levels = dt_levels
        self.env = env
        self.samples = []

        print(f"Loading dataset from {data_path}...")
        print(f"  Raw samples: {len(self.raw_data)}")
        print(f"  Generating samples for dt levels: {dt_levels}")

        # Pre-generate samples
        self._generate_samples()

        print(f"  Generated samples: {len(self.samples)}")

    def _generate_samples(self):
        """Generate training samples with final states"""
        for raw_sample in self.raw_data:
            scenario = raw_sample['scenario']
            initial_state = scenario['initial_state']
            force_pattern = scenario['force_pattern']

            # For each dt level, create a sample
            for dt in self.dt_levels:
                # Simulate to get final state
                final_state = self._simulate(initial_state, force_pattern, dt)

                self.samples.append({
                    'state': initial_state.astype(np.float32),
                    'actions': force_pattern[:20].astype(np.float32),  # Use first 20 timesteps
                    'time': np.array([0.0], dtype=np.float32),
                    'dt': np.array([dt], dtype=np.float32),
                    'final_state': final_state.astype(np.float32)
                })

    def _simulate(self, initial_state, force_pattern, dt):
        """Simulate physics to get final state"""
        self.env.reset()
        self.env.particles[0]["position"] = initial_state[:2].astype(np.float32)
        self.env.particles[0]["velocity"] = initial_state[2:].astype(np.float32)

        num_steps = int(dt / self.env.dt)
        for step in range(num_steps):
            action_idx = min(step, len(force_pattern) - 1)
            action = force_pattern[action_idx]
            next_state, _, _ = self.env.step(action)

        return next_state.copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'state': torch.FloatTensor(sample['state']),
            'actions': torch.FloatTensor(sample['actions']),
            'time': torch.FloatTensor(sample['time']),
            'dt': torch.FloatTensor(sample['dt']),
            'final_state': torch.FloatTensor(sample['final_state'])
        }


def main():
    parser = argparse.ArgumentParser(description="Train state-based shortcut predictor")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default='auto', help="Device (auto/cuda/cpu)")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable wandb logging")
    parser.add_argument("--save_dir", type=str, default='experiments/state_based',
                       help="Directory to save models")

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*80)
    print("STATE-BASED SHORTCUT TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}\n")

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="shortcut-state-based",
            name=f"state_shortcut_e{args.epochs}_bs{args.batch_size}",
            config={
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'architecture': 'StatePredictor',
                'hidden_dims': [64, 64, 64, 64],
                'supervised_levels': [0.01, 0.04, 0.08, 0.16, 0.64],
                'sc_levels': [0.02, 0.32, 1.0],
                'lambda_state': 0.6,
                'lambda_sc': 0.4
            },
            tags=["state-based", "bootstrap", "collision-physics"]
        )

    # Create physics environment
    env = PointMass2D(dt=0.01, damping=0.05)

    # Create datasets
    dt_levels = [0.01, 0.04, 0.08, 0.16, 0.64]  # Supervised levels only for dataset

    print("Creating training dataset...")
    train_dataset = StateDataset(
        'data/collision_train.pkl',
        dt_levels=dt_levels,
        env=env
    )

    print("Creating validation dataset...")
    val_dataset = StateDataset(
        'data/collision_val.pkl',
        dt_levels=dt_levels,
        env=env
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Single process for now
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    print("\nCreating StatePredictor model...")
    model = StatePredictor(
        state_dim=4,
        action_dim=2,
        max_seq_len=20,
        hidden_dims=[64, 64, 64, 64]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    trainer = StateBootstrapTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        supervised_levels=[0.01, 0.04, 0.08, 0.16, 0.64],
        sc_levels=[0.02, 0.32, 1.0],
        lambda_state=0.6,
        lambda_sc=0.4
    )

    # Train
    print("\nStarting training...")
    save_path = Path(args.save_dir) / 'state_best_model.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = trainer.train(
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=str(save_path)
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
