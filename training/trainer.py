import torch
from tqdm import tqdm
import wandb
import numpy as np

class ShortcutTrainer:
    """Training loop for shortcut predictor with wandb integration"""

    def __init__(self, model, optimizer, device='cuda', env=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.env = env  # Physics environment for computing true average velocities
        self.step_count = 0


    def train_epoch(self, dataloader, lambda_v=0.6, lambda_sc=0.4, epoch=None,
                    use_improved_velocity_loss=True, config=None):
        """
        Train for one epoch with 60/40 loss weighting (NO curriculum learning)
        """
        from .losses import velocity_matching_loss, self_consistency_loss, compute_true_average_velocity

        self.model.train()
        total_loss = 0
        total_loss_v = 0
        total_loss_sc = 0

        # Fixed max horizon - no curriculum learning
        max_horizon = config['training']['max_horizon'] if config else 1.0

        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch+1 if epoch is not None else ""}')
        for batch_idx, batch in enumerate(pbar):
            state = batch['state'].to(self.device)
            actions = batch['actions'].to(self.device)
            velocity_data = batch['velocity'].to(self.device)
            time = batch['time'].to(self.device)

            # ================================================================
            # LOSS 1: VELOCITY MATCHING (d=0.01) - 60% weight
            # Teaches model to match physics at small time scales
            # ================================================================
            small_step = torch.full((state.shape[0], 1), 0.01, device=self.device)
            velocity_pred = self.model.velocity_net(state, actions, time, small_step)

            if use_improved_velocity_loss and self.env is not None:
                velocity_true = compute_true_average_velocity(self.env, state, actions, small_step)
            else:
                velocity_true = velocity_data

            loss_v = velocity_matching_loss(velocity_pred, velocity_true)

            # Clip extreme losses to prevent outliers from destroying training
            # Max loss of 10.0 is ~10x the typical loss of ~1.0
            loss_v = torch.clamp(loss_v, max=500.0)

            # ================================================================
            # LOSS 2: SELF-CONSISTENCY (d ∈ [0, max_horizon]) - 40% weight
            # Teaches model to be consistent across different time scales
            # ================================================================
            step_sizes = torch.rand(state.shape[0], 1, device=self.device) * max_horizon
            loss_sc = self_consistency_loss(self.model, state, actions, time, step_sizes)

            # Clip extreme losses
            loss_sc = torch.clamp(loss_sc, max=500.0)

            # ================================================================
            # COMBINED LOSS: 60/40 weighted sum
            # ================================================================
            loss = lambda_v * loss_v + lambda_sc * loss_sc

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_sc += loss_sc.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'loss_v': f'{loss_v.item():.4f}',
                'loss_sc': f'{loss_sc.item():.4f}'
            })

            self.step_count += 1

            # Log to wandb every 10 steps
            if wandb.run is not None and self.step_count % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_velocity_loss": loss_v.item(),
                    "train/batch_consistency_loss": loss_sc.item(),
                    "train/velocity_contribution": lambda_v * loss_v.item(),
                    "train/consistency_contribution": lambda_sc * loss_sc.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "train/step": self.step_count
                })

        # Calculate epoch averages
        epoch_metrics = {
            'total': total_loss / len(dataloader),
            'velocity': total_loss_v / len(dataloader),
            'consistency': total_loss_sc / len(dataloader)
        }

        # Log epoch metrics
        if wandb.run is not None and epoch is not None:
            wandb.log({
                "train/epoch_loss": epoch_metrics['total'],
                "train/epoch_velocity_loss": epoch_metrics['velocity'],
                "train/epoch_consistency_loss": epoch_metrics['consistency'],
                "epoch": epoch
            })

        return epoch_metrics

    def validate(self, dataloader, epoch=None, use_improved_velocity_loss=True):
        """Validate model with mathematically correct loss functions"""
        from .losses import velocity_matching_loss, self_consistency_loss, compute_true_average_velocity

        self.model.eval()
        total_loss = 0
        total_loss_v = 0
        total_loss_sc = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch+1 if epoch is not None else ""}')
            for batch in pbar:
                state = batch['state'].to(self.device)
                actions = batch['actions'].to(self.device)
                velocity_data = batch['velocity'].to(self.device)
                time = batch['time'].to(self.device)

                # Velocity matching with improved ground truth
                small_step = torch.full((state.shape[0], 1), 0.01, device=self.device)
                velocity_pred = self.model.velocity_net(state, actions, time, small_step)

                if use_improved_velocity_loss and self.env is not None:
                    velocity_true = compute_true_average_velocity(self.env, state, actions, small_step)
                else:
                    velocity_true = velocity_data

                loss_v = velocity_matching_loss(velocity_pred, velocity_true)

                # Clip extreme losses in validation too
                loss_v = torch.clamp(loss_v, max=500.0)

                step_sizes = torch.rand(state.shape[0], 1, device=self.device) * 1.0
                loss_sc = self_consistency_loss(self.model, state, actions, time, step_sizes)

                # Clip extreme losses
                loss_sc = torch.clamp(loss_sc, max=500.0)

                loss = loss_v + 0.5 * loss_sc
                total_loss += loss.item()
                total_loss_v += loss_v.item()
                total_loss_sc += loss_sc.item()

                pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_loss_v': f'{loss_v.item():.4f}',
                    'val_loss_sc': f'{loss_sc.item():.4f}'
                })

        val_loss = total_loss / len(dataloader)
        val_loss_v = total_loss_v / len(dataloader)
        val_loss_sc = total_loss_sc / len(dataloader)

        # Log to wandb
        if wandb.run is not None and epoch is not None:
            wandb.log({
                "val/epoch_loss": val_loss,
                "val/epoch_velocity_loss": val_loss_v,
                "val/epoch_consistency_loss": val_loss_sc,
                "epoch": epoch
            })

        return val_loss