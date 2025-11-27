#!/usr/bin/env python3
"""
Two-Network Training System

Implements fair comparison between Sequential baseline and Shortcut predictor.
- Sequential Network: Optimized for d=0.01 only (physics grounding)
- Shortcut Network: Bootstrap trained across d=[0.01, 0.02, ..., 1.0]
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import wandb
from pathlib import Path

from .losses import velocity_matching_loss, self_consistency_loss, compute_true_average_velocity
from .bootstrap_trainer import BootstrapTrainer

def collision_collate_fn(batch):
    """Collate function for collision physics dataset

    Combines individual samples into batched tensors for training.
    Handles both single-timestep and multi-timestep collision datasets.

    Args:
        batch: List of dictionaries containing sample data

    Returns:
        Dictionary with batched tensors for state, actions, velocity, time, dt
    """
    return {
        'state': torch.stack([item['state'] for item in batch]),
        'actions': torch.stack([item['actions'] for item in batch]),
        'velocity': torch.stack([item['velocity'] for item in batch]),
        'time': torch.stack([item['time'] for item in batch]),
        'dt': torch.stack([item['dt'] for item in batch])
    }

class SequentialTrainer:
    """Trainer for Sequential baseline network (d=0.01 only)"""

    def __init__(self, model, optimizer, device, env=None):
        """
        Args:
            model: VelocityFieldNet for sequential prediction
            optimizer: PyTorch optimizer
            device: 'cuda' or 'cpu'
            env: Physics environment for ground truth
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.env = env
        self.step_count = 0

    def train_epoch(self, dataloader, epoch=None):
        """Train sequential model on d=0.01 physics grounding only"""

        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Sequential Epoch {epoch+1 if epoch is not None else ""}')
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)

            # Sequential training: only d=0.01 velocity matching
            step_size = torch.full((len(batch['state']), 1), 0.01, device=self.device)

            # Predict velocity at d=0.01
            velocity_pred = self.model(
                batch['state'],
                batch['actions'],
                batch['time'],
                step_size
            )

            # Compute true velocity using physics simulation
            if self.env is not None:
                velocity_true = compute_true_average_velocity(
                    self.env,
                    batch['state'],
                    batch['actions'],
                    step_size
                )
            else:
                velocity_true = batch['velocity']

            # Pure velocity matching loss (no self-consistency)
            loss = velocity_matching_loss(velocity_pred, velocity_true)
            loss = torch.clamp(loss, max=500.0)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.step_count += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log to wandb
            if wandb.run is not None and self.step_count % 10 == 0:
                wandb.log({
                    "sequential/batch_loss": loss.item(),
                    "sequential/step": self.step_count
                })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log epoch summary
        if wandb.run is not None and epoch is not None:
            wandb.log({
                "sequential/epoch_loss": avg_loss,
                "epoch": epoch
            })

        return {'total': avg_loss, 'velocity': avg_loss}

    def validate(self, dataloader, epoch=None):
        """Validate sequential model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)

                step_size = torch.full((len(batch['state']), 1), 0.01, device=self.device)

                velocity_pred = self.model(
                    batch['state'],
                    batch['actions'],
                    batch['time'],
                    step_size
                )

                if self.env is not None:
                    velocity_true = compute_true_average_velocity(
                        self.env,
                        batch['state'],
                        batch['actions'],
                        step_size
                    )
                else:
                    velocity_true = batch['velocity']

                loss = velocity_matching_loss(velocity_pred, velocity_true)
                loss = torch.clamp(loss, max=500.0)  # Add missing loss clipping
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss

class TwoNetworkTrainer:
    """Coordinates training and evaluation of both Sequential and Shortcut networks"""

    def __init__(self, sequential_model, shortcut_model, device, env, config):
        """
        Args:
            sequential_model: VelocityFieldNet for sequential baseline
            shortcut_model: ShortcutPredictor for temporal scaling
            device: 'cuda' or 'cpu'
            env: Physics environment
            config: Training configuration
        """
        self.sequential_model = sequential_model
        self.shortcut_model = shortcut_model
        self.device = device
        self.env = env
        self.config = config

        # Create optimizers
        self.sequential_optimizer = torch.optim.Adam(
            sequential_model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.shortcut_optimizer = torch.optim.Adam(
            shortcut_model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Create trainers
        self.sequential_trainer = SequentialTrainer(
            sequential_model, self.sequential_optimizer, device, env
        )

        self.shortcut_trainer = BootstrapTrainer(
            shortcut_model, self.shortcut_optimizer, device, env,
            supervised_levels=config['training'].get('supervised_levels', [0.01, 0.04, 0.08, 0.16, 0.64]),
            all_bootstrap_levels=config['training']['bootstrap_levels']
        )

        print("🔄 Two-Network Training System Initialized")
        print(f"   Sequential Model: {sum(p.numel() for p in sequential_model.parameters()):,} parameters")
        print(f"   Shortcut Model: {sum(p.numel() for p in shortcut_model.parameters()):,} parameters")

    def validate_multi_horizon(self, val_loader, epoch=None):
        """Validation across multiple horizons to show differentiation during training"""

        # Sample small subset of validation data for multi-horizon testing
        val_samples = []
        sample_count = 0
        max_samples = 50  # Limit for speed during training

        for batch in val_loader:
            for i in range(len(batch['state'])):
                if sample_count >= max_samples:
                    break
                val_samples.append({
                    'state': batch['state'][i].cpu().numpy(),
                    'actions': batch['actions'][i].cpu().numpy(),
                    'time': batch['time'][i].cpu().numpy()
                })
                sample_count += 1
            if sample_count >= max_samples:
                break

        horizons = [0.01, 0.1, 0.5, 1.0]  # Test key horizons including large ones
        results = {}

        self.sequential_model.eval()
        self.shortcut_model.eval()

        with torch.no_grad():
            for horizon in horizons:
                seq_errors = []
                shortcut_errors = []

                for sample in val_samples:
                    # Simulate ground truth
                    if self.env is not None:
                        self.env.clear_particles()
                        x, y, vx, vy = sample['state']
                        self.env.add_particle(x, y, vx, vy, mass=self.env.mass)

                        num_steps = max(1, int(horizon / self.env.dt))
                        current_state = sample['state'].copy()

                        for step_idx in range(num_steps):
                            action_idx = min(step_idx, len(sample['actions']) - 1)
                            action = sample['actions'][action_idx] if len(sample['actions']) > 0 else np.zeros(2)
                            current_state, _, _ = self.env.step(action)

                        gt_state = current_state

                        # Sequential single-step prediction (out-of-distribution for horizon > 0.01)
                        # Create tensors in batch format (batch_size=1)
                        state_tensor = torch.FloatTensor(sample['state']).unsqueeze(0).to(self.device)  # (1, state_dim)

                        # Actions are shape (seq_len, action_dim) from data, need (1, seq_len, action_dim) for batch
                        actions_tensor = torch.FloatTensor(sample['actions']).unsqueeze(0).to(self.device)  # (1, seq_len, action_dim)

                        # Handle time - sample['time'] is a scalar from numpy array
                        time_scalar = float(sample['time']) if hasattr(sample['time'], '__iter__') else float(sample['time'])
                        time_tensor = torch.FloatTensor([[time_scalar]]).to(self.device)  # (1, 1)

                        horizon_tensor = torch.FloatTensor([[horizon]]).to(self.device)  # (1, 1)

                        seq_velocity = self.sequential_model(state_tensor, actions_tensor, time_tensor, horizon_tensor)
                        seq_pred = state_tensor + seq_velocity * horizon_tensor
                        seq_error = torch.norm(seq_pred[0, :2] - torch.FloatTensor(gt_state[:2]).to(self.device)).item()

                        # Shortcut single-step prediction (in-distribution)
                        shortcut_velocity = self.shortcut_model.velocity_net(state_tensor, actions_tensor, time_tensor, horizon_tensor)
                        shortcut_pred = state_tensor + shortcut_velocity * horizon_tensor
                        shortcut_error = torch.norm(shortcut_pred[0, :2] - torch.FloatTensor(gt_state[:2]).to(self.device)).item()

                        seq_errors.append(seq_error)
                        shortcut_errors.append(shortcut_error)

                results[horizon] = {
                    'seq_error': np.mean(seq_errors) if seq_errors else float('inf'),
                    'shortcut_error': np.mean(shortcut_errors) if shortcut_errors else float('inf')
                }

        return results

    def train_both_networks(self, train_loader, val_loader, epochs,
                           sequential_train_dataset=None, sequential_val_dataset=None):
        """Train both networks with separate datasets

        Args:
            train_loader: DataLoader for Shortcut model (multi-timestep data)
            val_loader: Validation loader for Shortcut model
            epochs: Number of training epochs
            sequential_train_dataset: Dataset for Sequential model (dt=0.01 only)
            sequential_val_dataset: Validation dataset for Sequential model
        """

        print(f"\n{'='*80}")
        print("STARTING TWO-NETWORK TRAINING")
        print(f"{'='*80}")
        print(f"\n🎯 TRAINING CONFIGURATION:")
        print(f"   Epochs: {epochs}")
        print()
        print(f"   📊 Sequential Model:")
        print(f"      - Supervised: dt=0.01 ONLY")
        print(f"      - Training samples: {len(sequential_train_dataset) if sequential_train_dataset else 'N/A'}")
        print(f"      - Expected at dt=1.0: CATASTROPHIC FAILURE (100x extrapolation)")
        print()
        print(f"   📊 Shortcut Model:")
        supervised_levels = self.config['training'].get('supervised_levels', [0.01, 0.04, 0.08, 0.16, 0.64])
        all_levels = self.config['training']['bootstrap_levels']
        sc_levels = [d for d in all_levels if d not in supervised_levels]
        print(f"      - Supervised: {supervised_levels}")
        print(f"      - Self-consistency: {sc_levels}")
        print(f"      - Training samples: {len(train_loader.dataset)}")
        print(f"      - Expected at dt=1.0: MAINTAINS ACCURACY (1.56x extrapolation)")
        print(f"{'='*80}\n")

        # Create separate dataloaders for Sequential if provided
        if sequential_train_dataset is not None:
            from torch.utils.data import DataLoader
            sequential_train_loader = DataLoader(
                sequential_train_dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                collate_fn=collision_collate_fn
            )
            sequential_val_loader = DataLoader(
                sequential_val_dataset,
                batch_size=val_loader.batch_size,
                shuffle=False,
                collate_fn=collision_collate_fn
            )
            print("✅ Separate dataloaders created for Sequential model")
        else:
            # Fallback: use same loaders (not recommended for this experiment)
            sequential_train_loader = train_loader
            sequential_val_loader = val_loader
            print("⚠️  WARNING: Using same dataloaders for both models (not ideal for this experiment)")

        sequential_losses = {'train': [], 'val': []}
        shortcut_losses = {'train': [], 'val': []}

        best_sequential_loss = float('inf')
        best_shortcut_loss = float('inf')

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{epochs}")
            print(f"{'='*60}")

            # Train Sequential Network (dt=0.01 only)
            print("\n🔵 Training Sequential Model...")
            seq_train_loss = self.sequential_trainer.train_epoch(sequential_train_loader, epoch=epoch)
            seq_val_loss = self.sequential_trainer.validate(sequential_val_loader, epoch=epoch)

            sequential_losses['train'].append(seq_train_loss['total'])
            sequential_losses['val'].append(seq_val_loss)

            # Train Shortcut Network
            shortcut_train_loss = self.shortcut_trainer.train_epoch(
                train_loader,
                lambda_v=self.config['training']['lambda_v'],
                lambda_sc=self.config['training']['lambda_sc'],
                lambda_v_mag=self.config['training'].get('lambda_v_mag', 0.2),  # Default to 0.2
                lambda_casc=self.config['training'].get('lambda_casc', 0.2),   # Default to 0.2
                epoch=epoch,
                config=self.config
            )
            shortcut_val_loss = self.shortcut_trainer.validate(val_loader, epoch=epoch)

            shortcut_losses['train'].append(shortcut_train_loss['total'])
            shortcut_losses['val'].append(shortcut_val_loss)

            print(f"   Sequential  - Train: {seq_train_loss['total']:.4f}, Val: {seq_val_loss:.4f}")
            print(f"   Shortcut    - Train: {shortcut_train_loss['total']:.4f}, Val: {shortcut_val_loss:.4f}")

            # Multi-horizon validation every epoch to show differentiation
            if True:  # Run every epoch to see differentiation
                print(f"   🔍 Multi-Horizon Validation:")
                horizon_results = self.validate_multi_horizon(val_loader, epoch=epoch)
                print(f"       Testing {len(horizon_results)} different time horizons...")

                for horizon, errors in horizon_results.items():
                    seq_err = errors['seq_error']
                    shortcut_err = errors['shortcut_error']
                    ratio = seq_err / shortcut_err if shortcut_err > 0 else float('inf')

                    print(f"      dt={horizon:.2f}: Sequential={seq_err:.3f}, Shortcut={shortcut_err:.3f} (ratio: {ratio:.1f}x)")

                # Log the key differentiation metric (dt=1.0 ratio)
                if 1.0 in horizon_results:
                    dt1_ratio = horizon_results[1.0]['seq_error'] / horizon_results[1.0]['shortcut_error'] if horizon_results[1.0]['shortcut_error'] > 0 else float('inf')
                    print(f"      🎯 Large timestep performance gap: {dt1_ratio:.1f}x (Sequential/Shortcut at dt=1.0)")

                    if dt1_ratio > 5.0:
                        print(f"      ✅ Clear differentiation achieved! Sequential fails, Shortcut works")
                    else:
                        print(f"      ⚠️  Differentiation still developing...")

            # Save best models
            if seq_val_loss < best_sequential_loss:
                best_sequential_loss = seq_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.sequential_model.state_dict(),
                    'optimizer_state_dict': self.sequential_optimizer.state_dict(),
                    'val_loss': seq_val_loss,
                    'config': self.config,
                    'model_type': 'sequential_baseline'
                }, 'experiments/sequential_baseline_model.pt')
                print(f"     ✓ Saved best Sequential model (val_loss={seq_val_loss:.4f})")

            if shortcut_val_loss < best_shortcut_loss:
                best_shortcut_loss = shortcut_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.shortcut_model.state_dict(),
                    'optimizer_state_dict': self.shortcut_optimizer.state_dict(),
                    'val_loss': shortcut_val_loss,
                    'config': self.config,
                    'model_type': 'shortcut_bootstrap'
                }, 'experiments/shortcut_bootstrap_model.pt')
                print(f"     ✓ Saved best Shortcut model (val_loss={shortcut_val_loss:.4f})")

        return {
            'sequential_losses': sequential_losses,
            'shortcut_losses': shortcut_losses,
            'best_sequential_loss': best_sequential_loss,
            'best_shortcut_loss': best_shortcut_loss
        }

    def evaluate_comparison(self, test_data, d_levels=None):
        """Comprehensive evaluation comparing both networks"""

        if d_levels is None:
            d_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        print(f"\n🧪 Two-Network Comparison Evaluation")
        print(f"   Testing d-levels: {d_levels}")

        results = {
            'sequential': {},
            'shortcut': {},
            'speedup_analysis': {}
        }

        self.sequential_model.eval()
        self.shortcut_model.eval()

        with torch.no_grad():
            for d in d_levels:
                print(f"\n   Evaluating d={d:.2f}s...")

                sequential_errors = []
                shortcut_errors = []
                sequential_times = []
                shortcut_times = []

                for sample in test_data[:100]:  # Test on subset for speed
                    state = torch.FloatTensor(sample['state']).unsqueeze(0).to(self.device)
                    actions = torch.FloatTensor(sample['actions']).unsqueeze(0).to(self.device)
                    time = torch.FloatTensor([sample['time']]).unsqueeze(0).to(self.device)
                    step_size = torch.FloatTensor([[d]]).to(self.device)

                    # Ground truth via physics simulation
                    if self.env is not None:
                        self.env.clear_particles()
                        x, y, vx, vy = sample['state']
                        self.env.add_particle(x, y, vx, vy, mass=getattr(self.env, 'mass', 1.0))

                        num_steps = max(1, int(d / self.env.dt))
                        actions_np = sample['actions']

                        current_state = sample['state'].copy()
                        for step_idx in range(num_steps):
                            action_idx = min(step_idx, len(actions_np) - 1)
                            action = actions_np[action_idx] if len(actions_np) > 0 else np.zeros(2)
                            current_state, _, _ = self.env.step(action)

                        state_true = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)

                        # Sequential prediction (iterative rollout for d > 0.01)
                        import time as timer
                        start_time = timer.time()

                        if d <= 0.01:
                            # Direct prediction for d=0.01
                            velocity_pred = self.sequential_model(state, actions, time, step_size)
                            state_pred_seq = state + velocity_pred * step_size
                        else:
                            # Iterative rollout for d > 0.01
                            current_state_seq = state.clone()
                            num_steps_seq = int(d / 0.01)
                            dt_tensor = torch.FloatTensor([[0.01]]).to(self.device)

                            for _ in range(num_steps_seq):
                                velocity_pred = self.sequential_model(current_state_seq, actions, time, dt_tensor)
                                current_state_seq = current_state_seq + velocity_pred * dt_tensor

                            state_pred_seq = current_state_seq

                        sequential_time = timer.time() - start_time
                        sequential_times.append(sequential_time)

                        # Shortcut prediction (single forward pass)
                        start_time = timer.time()
                        velocity_pred_shortcut = self.shortcut_model.velocity_net(state, actions, time, step_size)
                        state_pred_shortcut = state + velocity_pred_shortcut * step_size
                        shortcut_time = timer.time() - start_time
                        shortcut_times.append(shortcut_time)

                        # Compute errors
                        seq_error = torch.norm(state_pred_seq - state_true).item()
                        shortcut_error = torch.norm(state_pred_shortcut - state_true).item()

                        sequential_errors.append(seq_error)
                        shortcut_errors.append(shortcut_error)

                # Store results
                results['sequential'][d] = {
                    'mean_error': np.mean(sequential_errors) if sequential_errors else float('inf'),
                    'std_error': np.std(sequential_errors) if sequential_errors else 0.0,
                    'mean_time': np.mean(sequential_times) if sequential_times else 0.0,
                    'num_samples': len(sequential_errors)
                }

                results['shortcut'][d] = {
                    'mean_error': np.mean(shortcut_errors) if shortcut_errors else float('inf'),
                    'std_error': np.std(shortcut_errors) if shortcut_errors else 0.0,
                    'mean_time': np.mean(shortcut_times) if shortcut_times else 0.0,
                    'num_samples': len(shortcut_errors)
                }

                # Speedup analysis
                if sequential_times and shortcut_times:
                    speedup = np.mean(sequential_times) / np.mean(shortcut_times)
                    accuracy_ratio = results['shortcut'][d]['mean_error'] / results['sequential'][d]['mean_error']

                    results['speedup_analysis'][d] = {
                        'speedup': speedup,
                        'accuracy_ratio': accuracy_ratio,
                        'efficiency_score': speedup / accuracy_ratio if accuracy_ratio > 0 else 0.0
                    }

                    print(f"     Sequential: Error={results['sequential'][d]['mean_error']:.4f}, Time={results['sequential'][d]['mean_time']:.6f}s")
                    print(f"     Shortcut:   Error={results['shortcut'][d]['mean_error']:.4f}, Time={results['shortcut'][d]['mean_time']:.6f}s")
                    print(f"     Speedup: {speedup:.1f}x, Accuracy Ratio: {accuracy_ratio:.3f}")

        return results

    def generate_comparison_report(self, results, save_path="experiments/two_network_comparison.txt"):
        """Generate detailed comparison report"""

        report_lines = [
            "🔍 TWO-NETWORK COMPARISON REPORT",
            "=" * 50,
            "",
            "METHODOLOGY:",
            "- Sequential: Optimized for d=0.01, iterative rollout for longer horizons",
            "- Shortcut: Bootstrap trained across d=[0.01, 0.02, ..., 1.0]",
            "- Evaluation: Physics simulation ground truth",
            "",
            "RESULTS BY TIME HORIZON:",
        ]

        for d in sorted(results['sequential'].keys()):
            seq_results = results['sequential'][d]
            shortcut_results = results['shortcut'][d]
            speedup_results = results['speedup_analysis'].get(d, {})

            report_lines.extend([
                f"",
                f"d = {d:.2f}s:",
                f"  Sequential  - Error: {seq_results['mean_error']:.4f} ± {seq_results['std_error']:.4f}, Time: {seq_results['mean_time']:.6f}s",
                f"  Shortcut    - Error: {shortcut_results['mean_error']:.4f} ± {shortcut_results['std_error']:.4f}, Time: {shortcut_results['mean_time']:.6f}s",
                f"  Speedup: {speedup_results.get('speedup', 0):.1f}x, Accuracy Ratio: {speedup_results.get('accuracy_ratio', 0):.3f}",
                f"  Efficiency Score: {speedup_results.get('efficiency_score', 0):.2f}"
            ])

        # Summary statistics
        all_speedups = [results['speedup_analysis'][d]['speedup'] for d in results['speedup_analysis']]
        all_accuracy_ratios = [results['speedup_analysis'][d]['accuracy_ratio'] for d in results['speedup_analysis']]

        report_lines.extend([
            "",
            "SUMMARY:",
            f"  Average Speedup: {np.mean(all_speedups):.1f}x",
            f"  Average Accuracy Ratio: {np.mean(all_accuracy_ratios):.3f}",
            f"  Shortcut maintains <2x error degradation: {sum(1 for r in all_accuracy_ratios if r < 2.0)} / {len(all_accuracy_ratios)} cases",
            "",
            "INTERPRETATION:",
            "- Speedup > 10x with accuracy ratio < 1.5 indicates successful shortcut learning",
            "- Accuracy ratio < 1.0 means shortcut outperforms sequential (rare but possible)",
            "- Efficiency score combines speed and accuracy for overall performance"
        ])

        # Save report
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))

        # Print to console
        for line in report_lines:
            print(line)

        return '\n'.join(report_lines)