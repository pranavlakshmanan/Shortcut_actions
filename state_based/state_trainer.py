import torch
import numpy as np
from tqdm import tqdm
import wandb
from state_losses import state_matching_loss, state_self_consistency_loss


class StateBootstrapTrainer:
    """Training loop for state-based shortcut prediction with bootstrap hierarchy

    Key differences from velocity trainer:
    1. Training target is final_state directly (no division by dt)
    2. Uses state_matching_loss instead of velocity_matching_loss
    3. Self-consistency operates on states, not velocities
    """

    def __init__(self, model, optimizer, device,
                 supervised_levels=None, sc_levels=None,
                 lambda_state=0.6, lambda_sc=0.4):
        """
        Args:
            model: StatePredictor instance
            optimizer: PyTorch optimizer
            device: 'cuda' or 'cpu'
            supervised_levels: dt values with ground truth [0.01, 0.04, 0.08, 0.16, 0.64]
            sc_levels: dt values learned via self-consistency [0.02, 0.32, 1.0]
            lambda_state: Weight for state matching loss (default 0.6)
            lambda_sc: Weight for self-consistency loss (default 0.4)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.supervised_levels = supervised_levels or [0.01, 0.04, 0.08, 0.16, 0.64]
        self.sc_levels = sc_levels or [0.02, 0.32, 1.0]
        self.lambda_state = lambda_state
        self.lambda_sc = lambda_sc

        print(f"\n🎯 State-Based Bootstrap Trainer")
        print(f"   Supervised levels (ground truth): {self.supervised_levels}")
        print(f"   Self-consistency levels: {self.sc_levels}")
        print(f"   Loss weights: {lambda_state:.1f} state + {lambda_sc:.1f} self-consistency")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_state_loss = 0
        total_sc_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            state = batch['state'].to(self.device)
            actions = batch['actions'].to(self.device)
            time = batch['time'].to(self.device)
            dt = batch['dt'].to(self.device)

            batch_size = state.shape[0]

            # ===== SUPERVISED LOSS (60%) =====
            # Use samples where dt is in supervised_levels
            supervised_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for d_level in self.supervised_levels:
                supervised_mask |= (torch.abs(dt.squeeze() - d_level) < 1e-6)

            if supervised_mask.sum() > 0:
                # Get ground truth final state from dataset
                # (Assuming dataset has 'final_state' field)
                if 'final_state' in batch:
                    true_final_state = batch['final_state'].to(self.device)

                    # Predict final state
                    pred_state = self.model(
                        state[supervised_mask],
                        actions[supervised_mask],
                        time[supervised_mask],
                        dt[supervised_mask]
                    )

                    loss_state = state_matching_loss(pred_state, true_final_state[supervised_mask])
                else:
                    loss_state = torch.tensor(0.0, device=self.device)
            else:
                loss_state = torch.tensor(0.0, device=self.device)

            # ===== SELF-CONSISTENCY LOSS (40%) =====
            # Sample random dt from sc_levels for self-consistency
            sc_size = max(1, batch_size // 3)  # Use ~1/3 of batch for SC
            sc_dt_values = np.random.choice(self.sc_levels, size=sc_size, replace=True)
            sc_dt = torch.FloatTensor(sc_dt_values).unsqueeze(1).to(self.device)

            # Use first sc_size samples for self-consistency
            loss_sc = state_self_consistency_loss(
                self.model,
                state[:sc_size],
                actions[:sc_size],
                time[:sc_size],
                sc_dt
            )

            # ===== COMBINED LOSS =====
            loss = self.lambda_state * loss_state + self.lambda_sc * loss_sc

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            total_state_loss += loss_state.item()
            total_sc_loss += loss_sc.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'state': f'{loss_state.item():.4f}',
                'sc': f'{loss_sc.item():.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_state_loss = total_state_loss / num_batches
        avg_sc_loss = total_sc_loss / num_batches

        return {
            'train_loss': avg_loss,
            'train_state_loss': avg_state_loss,
            'train_sc_loss': avg_sc_loss
        }

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(self.device)
                actions = batch['actions'].to(self.device)
                time = batch['time'].to(self.device)
                dt = batch['dt'].to(self.device)

                if 'final_state' in batch:
                    true_final_state = batch['final_state'].to(self.device)
                    pred_state = self.model(state, actions, time, dt)
                    loss = state_matching_loss(pred_state, true_final_state)
                    total_loss += loss.item()
                    num_batches += 1

        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_val_loss}

    def train(self, epochs, train_loader, val_loader, save_path='state_best_model.pt'):
        """Full training loop"""
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **val_metrics
                })

            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss={train_metrics['train_loss']:.4f}, "
                  f"Val Loss={val_metrics['val_loss']:.4f}")

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, save_path)
                print(f"   ✓ Saved best model (val_loss={best_val_loss:.4f})")

        return best_val_loss
