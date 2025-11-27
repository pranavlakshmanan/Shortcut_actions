import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import wandb
from .losses import (velocity_matching_loss, self_consistency_loss, compute_true_average_velocity,
                     velocity_magnitude_loss, cascaded_self_consistency_loss)

class BootstrapBatchSampler:
    """Sample training batches with grounded bootstrap hierarchy

    Supervised levels: [0.01, 0.04, 0.08, 0.16, 0.64] - ground truth from dataset
    Self-consistency levels: [0.02, 0.32, 1.0] - learned via consistency loss
    """

    def __init__(self, supervised_levels: List[float],
                 all_bootstrap_levels: List[float],
                 velocity_ratio: float = 0.6):
        """
        Args:
            supervised_levels: Levels with ground truth [0.01, 0.04, 0.08, 0.16, 0.64]
            all_bootstrap_levels: All levels [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
            velocity_ratio: Fraction for velocity matching (0.6 = 60%)
        """
        self.supervised_levels = supervised_levels
        self.all_bootstrap_levels = all_bootstrap_levels
        self.velocity_ratio = velocity_ratio

        # Self-consistency levels = all levels NOT in supervised
        self.sc_levels = [d for d in all_bootstrap_levels
                         if d not in supervised_levels]

        print(f"\n🎯 Grounded Bootstrap Hierarchy:")
        print(f"   Supervised (ground truth): {supervised_levels}")
        print(f"   Self-consistency (learned): {self.sc_levels}")
        print(f"   Batch split: {velocity_ratio*100:.0f}% velocity / {(1-velocity_ratio)*100:.0f}% self-consistency")

    def sample_batch(self, batch_data: Dict) -> Tuple[Dict, Dict]:
        """Split batch into velocity matching (ground truth) and self-consistency

        Key Design:
        - 60% of batch: Velocity matching using actual dt from dataset (supervised levels)
        - 40% of batch: Self-consistency at random d from [0.02, 0.32, 1.0]

        This enables the model to:
        1. Learn real physics at anchored timesteps [0.01, 0.04, 0.08, 0.16, 0.64]
        2. Learn temporal scaling via consistency at [0.02, 0.32, 1.0]
        """
        batch_size = len(batch_data['state'])
        velocity_size = int(batch_size * self.velocity_ratio)

        # ================================================================
        # 60% for VELOCITY MATCHING at supervised dt levels
        # Dataset contains only [0.01, 0.04, 0.08, 0.16, 0.64] samples
        # ================================================================
        velocity_batch = {
            'state': batch_data['state'][:velocity_size],
            'actions': batch_data['actions'][:velocity_size],
            'velocity': batch_data['velocity'][:velocity_size],
            'time': batch_data['time'][:velocity_size],
            'step_size': batch_data['dt'][:velocity_size]  # Uses actual dt from dataset
        }

        # ================================================================
        # 40% for SELF-CONSISTENCY at learned levels [0.02, 0.32, 1.0]
        # These are learned WITHOUT ground truth via consistency loss
        # ================================================================
        sc_size = batch_size - velocity_size
        if sc_size > 0 and len(self.sc_levels) > 0:
            # Sample random d levels from self-consistency set
            random_d_levels = np.random.choice(
                self.sc_levels,
                size=sc_size,
                replace=True
            )

            # Use same states but at self-consistency timesteps
            sc_batch = {
                'state': batch_data['state'][velocity_size:],
                'actions': batch_data['actions'][velocity_size:],
                'time': batch_data['time'][velocity_size:],
                'step_size': torch.FloatTensor(random_d_levels).unsqueeze(1).to(batch_data['state'].device)
            }
        else:
            sc_batch = None

        return velocity_batch, sc_batch

class BootstrapTrainer:
    """Training loop with bootstrap hierarchy across multiple time scales"""

    def __init__(self, model, optimizer, device, env=None,
                 supervised_levels=None, all_bootstrap_levels=None):
        """
        Args:
            model: ShortcutPredictor model
            optimizer: PyTorch optimizer
            device: 'cuda' or 'cpu'
            env: Physics environment (kept for compatibility, not used for velocity computation)
            supervised_levels: Levels with ground truth [0.01, 0.04, 0.08, 0.16, 0.64]
            all_bootstrap_levels: All levels [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.env = env
        self.step_count = 0

        # Default supervised levels (grounded anchors)
        if supervised_levels is None:
            self.supervised_levels = [0.01, 0.04, 0.08, 0.16, 0.64]
        else:
            self.supervised_levels = supervised_levels

        # Default all bootstrap levels
        if all_bootstrap_levels is None:
            self.all_bootstrap_levels = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
        else:
            self.all_bootstrap_levels = all_bootstrap_levels

        # Create batch sampler with grounded hierarchy
        self.sampler = BootstrapBatchSampler(
            supervised_levels=self.supervised_levels,
            all_bootstrap_levels=self.all_bootstrap_levels,
            velocity_ratio=0.6
        )

        print(f"✅ Bootstrap Trainer initialized with Grounded Hierarchy")

    def train_epoch(self, dataloader, lambda_v=0.6, lambda_sc=0.4, lambda_v_mag=0.2, lambda_casc=0.2, epoch=None, config=None):
        """Train one epoch with bootstrap hierarchy"""

        self.model.train()
        total_loss = 0
        total_loss_v = 0
        total_loss_sc = 0
        total_loss_v_mag = 0
        total_loss_casc = 0

        # Track losses by d-level for monitoring
        level_losses = {str(d): [] for d in self.all_bootstrap_levels}

        pbar = tqdm(dataloader, desc=f'Bootstrap Epoch {epoch+1 if epoch is not None else ""}')
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)

            # Split batch into velocity matching and self-consistency
            velocity_batch, sc_batch = self.sampler.sample_batch(batch)

            # ================================================================
            # LOSS 1: VELOCITY MATCHING (60% of batch, supervised levels only)
            # Ground truth comes from dataset - no environment simulation needed
            # Supervised at: [0.01, 0.04, 0.08, 0.16, 0.64]
            # ================================================================
            velocity_pred = self.model.velocity_net(
                velocity_batch['state'],
                velocity_batch['actions'],
                velocity_batch['time'],
                velocity_batch['step_size']
            )

            # Ground truth velocity pre-computed from trajectories in dataset
            velocity_true = velocity_batch['velocity']

            loss_v = velocity_matching_loss(velocity_pred, velocity_true)
            loss_v = torch.clamp(loss_v, max=500.0)  # Prevent outliers

            # ================================================================
            # LOSS 1B: VELOCITY MAGNITUDE MATCHING
            # Prevents zero-velocity predictions
            # ================================================================
            loss_v_mag = velocity_magnitude_loss(velocity_pred, velocity_true)
            loss_v_mag = torch.clamp(loss_v_mag, max=500.0)

            # Track losses by dt level for monitoring
            unique_dts = torch.unique(velocity_batch['step_size'])
            for dt_val in unique_dts:
                dt_key = f"{dt_val.item():.2f}"
                if dt_key not in level_losses:
                    level_losses[dt_key] = []

                # Get samples at this dt
                mask = (velocity_batch['step_size'] == dt_val).squeeze()
                if mask.sum() > 0:
                    level_loss = torch.nn.functional.mse_loss(
                        velocity_pred[mask],
                        velocity_true[mask]
                    )
                    level_losses[dt_key].append(level_loss.item())

            # ================================================================
            # LOSS 2: SELF-CONSISTENCY (40% of batch, random d levels)
            # ================================================================
            loss_sc = torch.tensor(0.0, device=self.device)
            loss_casc = torch.tensor(0.0, device=self.device)  # Cascaded consistency

            if sc_batch is not None and len(sc_batch['state']) > 0:
                # Group by step size for efficient computation
                unique_step_sizes = torch.unique(sc_batch['step_size'].flatten())

                for step_size_val in unique_step_sizes:
                    mask = (sc_batch['step_size'].flatten() == step_size_val)
                    if mask.sum() == 0:
                        continue

                    # Get samples for this step size
                    state_subset = sc_batch['state'][mask]
                    actions_subset = sc_batch['actions'][mask]
                    time_subset = sc_batch['time'][mask]
                    step_size_subset = sc_batch['step_size'][mask]

                    # Compute self-consistency loss for this d level
                    sc_loss_subset = self_consistency_loss(
                        self.model,
                        state_subset,
                        actions_subset,
                        time_subset,
                        step_size_subset
                    )
                    
                    sc_loss_subset = torch.clamp(sc_loss_subset, max=500.0)

                    # ================================================================
                    # LOSS 2B: CASCADED SELF-CONSISTENCY (Option B)
                    # ================================================================
                    # Only apply cascaded consistency for larger step sizes (d >= 0.04)
                    if step_size_val >= 0.04:
                        casc_loss_subset = cascaded_self_consistency_loss(
                            self.model,
                            state_subset,
                            actions_subset,
                            time_subset,
                            step_size_subset
                        )
                        casc_loss_subset = torch.clamp(casc_loss_subset, max=500.0)

                        # Accumulate cascaded loss
                        loss_casc = loss_casc + casc_loss_subset * (mask.sum().float() / len(sc_batch['state']))

                    # Accumulate loss (weighted by number of samples)
                    loss_sc = loss_sc + sc_loss_subset * (mask.sum().float() / len(sc_batch['state']))

                    # Track loss by level
                    level_key = f"{step_size_val:.2f}"
                    if level_key not in level_losses:
                        level_losses[level_key] = []
                    level_losses[level_key].append(sc_loss_subset.item())

            # ================================================================
            # COMBINED LOSS: Enhanced with velocity magnitude + cascaded consistency
            # ================================================================
            total_batch_loss = (lambda_v * loss_v +
                              lambda_sc * loss_sc +
                              lambda_v_mag * loss_v_mag +
                              lambda_casc * loss_casc)

            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_loss_v += loss_v.item()
            total_loss_sc += loss_sc.item() if torch.is_tensor(loss_sc) else 0.0
            total_loss_v_mag += loss_v_mag.item() if torch.is_tensor(loss_v_mag) else 0.0
            total_loss_casc += loss_casc.item() if torch.is_tensor(loss_casc) else 0.0

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'loss_v': f'{loss_v.item():.4f}',
                'loss_sc': f'{loss_sc.item() if torch.is_tensor(loss_sc) else 0.0:.4f}',
                'loss_v_mag': f'{loss_v_mag.item() if torch.is_tensor(loss_v_mag) else 0.0:.4f}',
                'loss_casc': f'{loss_casc.item() if torch.is_tensor(loss_casc) else 0.0:.4f}'
            })

            self.step_count += 1

            # Log to wandb every 10 steps
            if wandb.run is not None and self.step_count % 10 == 0:
                log_dict = {
                    "bootstrap/batch_loss": total_batch_loss.item(),
                    "bootstrap/velocity_loss": loss_v.item(),
                    "bootstrap/consistency_loss": loss_sc.item() if torch.is_tensor(loss_sc) else 0.0,
                    "bootstrap/velocity_magnitude_loss": loss_v_mag.item() if torch.is_tensor(loss_v_mag) else 0.0,
                    "bootstrap/cascaded_consistency_loss": loss_casc.item() if torch.is_tensor(loss_casc) else 0.0,
                    "bootstrap/velocity_contribution": lambda_v * loss_v.item(),
                    "bootstrap/consistency_contribution": lambda_sc * (loss_sc.item() if torch.is_tensor(loss_sc) else 0.0),
                    "bootstrap/velocity_magnitude_contribution": lambda_v_mag * (loss_v_mag.item() if torch.is_tensor(loss_v_mag) else 0.0),
                    "bootstrap/cascaded_contribution": lambda_casc * (loss_casc.item() if torch.is_tensor(loss_casc) else 0.0),
                    "bootstrap/step": self.step_count
                }

                # Log per-level losses
                for level, losses in level_losses.items():
                    if losses:
                        log_dict[f"bootstrap/loss_d_{level}"] = np.mean(losses)

                wandb.log(log_dict)

        # Calculate epoch averages
        epoch_metrics = {
            'total': total_loss / len(dataloader),
            'velocity': total_loss_v / len(dataloader),
            'consistency': total_loss_sc / len(dataloader),
            'velocity_magnitude': total_loss_v_mag / len(dataloader),
            'cascaded_consistency': total_loss_casc / len(dataloader),
            'level_losses': {level: np.mean(losses) if losses else 0.0
                           for level, losses in level_losses.items()}
        }

        # Log epoch summary
        if wandb.run is not None and epoch is not None:
            epoch_log = {
                "bootstrap/epoch_loss": epoch_metrics['total'],
                "bootstrap/epoch_velocity_loss": epoch_metrics['velocity'],
                "bootstrap/epoch_consistency_loss": epoch_metrics['consistency'],
                "bootstrap/epoch_velocity_magnitude_loss": epoch_metrics['velocity_magnitude'],
                "bootstrap/epoch_cascaded_consistency_loss": epoch_metrics['cascaded_consistency'],
                "epoch": epoch
            }

            # Log per-level epoch averages
            for level, avg_loss in epoch_metrics['level_losses'].items():
                epoch_log[f"bootstrap/epoch_loss_d_{level}"] = avg_loss

            wandb.log(epoch_log)

        return epoch_metrics

    def validate(self, dataloader, epoch=None):
        """Validate model across all bootstrap levels"""
        from .trainer import ShortcutTrainer  # Import to reuse validation logic

        # Use base trainer validation but with bootstrap sampling
        base_trainer = ShortcutTrainer(self.model, self.optimizer, self.device, self.env)
        return base_trainer.validate(dataloader, epoch)

    def evaluate_by_d_level(self, test_data, horizons=None):
        """Evaluate model performance at each bootstrap level"""

        if horizons is None:
            horizons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        self.model.eval()
        results = {}

        print(f"\\nEvaluating across d-levels: {horizons}")

        with torch.no_grad():
            for d in horizons:
                errors = []

                for sample in test_data[:100]:  # Test on first 100 samples
                    state = torch.FloatTensor(sample['state']).unsqueeze(0).to(self.device)
                    actions = torch.FloatTensor(sample['actions']).unsqueeze(0).to(self.device)
                    time = torch.FloatTensor([sample['time']]).unsqueeze(0).to(self.device)
                    step_size = torch.FloatTensor([[d]]).to(self.device)

                    # Predict using shortcut
                    velocity_pred = self.model.velocity_net(state, actions, time, step_size)
                    state_pred = state + velocity_pred * step_size

                    # Ground truth using physics simulation
                    if self.env is not None:
                        # Run physics simulation for ground truth
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
                        error = torch.norm(state_pred - state_true).item()
                        errors.append(error)

                results[d] = {
                    'mean_error': np.mean(errors) if errors else float('inf'),
                    'std_error': np.std(errors) if errors else 0.0,
                    'num_samples': len(errors)
                }

                print(f"  d={d:4.2f}: Error = {results[d]['mean_error']:.4f} ± {results[d]['std_error']:.4f}")

        return results