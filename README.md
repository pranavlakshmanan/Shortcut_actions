# Shortcut Physics Predictor: Velocity vs State-Based Learning

## Executive Summary

This project implements neural network-based physics prediction for 2D particle dynamics with elastic collisions. The current **velocity-based architecture** shows strong performance at small timesteps (dt=0.05-0.1) but degrades significantly at large timesteps (dt=0.5-1.0), achieving only **1.32x improvement** over the sequential baseline at dt=1.0 instead of the expected **10x speedup**.

We propose a **state-based architecture** that predicts final states directly instead of velocities, eliminating a division-by-dt operation that amplifies collision artifacts during training.

---

## Table of Contents

1. [Current System Architecture](#current-system-architecture)
2. [The Problem: Large-Timestep Performance Degradation](#the-problem-large-timestep-performance-degradation)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Proposed Solution: State-Based Prediction](#proposed-solution-state-based-prediction)
5. [Implementation Guide](#implementation-guide)
6. [Key Code Components](#key-code-components)
7. [Validation Strategy](#validation-strategy)
8. [Success Criteria](#success-criteria)
9. [Questions for Expert Review](#questions-for-expert-review)

---

## Current System Architecture

### Overview

The system learns to predict particle trajectories across multiple timescales using a **velocity field network** combined with a **bootstrap training hierarchy**.

### Model Architecture: Velocity-Based Prediction

**Network: `VelocityFieldNet`**
```python
# Location: models/velocity_field.py

class VelocityFieldNet(nn.Module):
    """
    Predicts average velocity over timestep d

    Input:  [state(4D), actions(40D), time(1D), dt(1D)] = 46D
    Output: velocity(4D) - average velocity over timestep dt

    Architecture: 4-layer MLP with LayerNorm
    - Hidden dims: [256, 256, 128]
    - Parameters: ~411,000
    """

    def forward(self, state, action_seq, time, step_size):
        # state: (batch, 4) - [x, y, vx, vy]
        # action_seq: (batch, seq_len, 2) - forces over time
        # time: (batch, 1) - current time
        # step_size: (batch, 1) - timestep size d

        x = torch.cat([state, action_flat, time, step_size], dim=1)
        velocity = self.network(x)  # Predict v̂
        return velocity
```

**State Update Rule:**
```python
# Velocity is integrated to get final state
ŝ(t+d) = s(t) + v̂_θ(s, a, t, d) × d

# Where:
# - v̂_θ: Predicted average velocity from network
# - d: Timestep size
# - ŝ(t+d): Predicted state at time t+d
```

### Training Strategy: Bootstrap Hierarchy

**Supervised Learning Levels** (with ground truth):
- `[0.01, 0.04, 0.08, 0.16, 0.64]` seconds

**Self-Consistency Levels** (learned without ground truth):
- `[0.02, 0.32, 1.0]` seconds

**Training Target Computation:**
```python
# Location: training/losses.py:22-77

def compute_true_average_velocity(env, state, actions, step_size):
    """
    Ground truth velocity computation

    1. Simulate physics from state for duration step_size
    2. Get final state s(t+d) from simulation
    3. Compute average velocity: v_true = (s(t+d) - s(t)) / d

    THIS DIVISION BY d IS THE PROBLEM!
    """
    # Simulate physics
    for step in range(num_sub_steps):
        next_state = env.step(action)

    # Compute average velocity
    displacement = next_state - current_state
    avg_velocity = displacement / step_size  # ⚠️ Division by dt!

    return avg_velocity
```

### Loss Functions

**1. Velocity Matching Loss (60% weight):**
```python
# Location: training/losses.py:5-19

L_v = ||v̂_θ(s, a, t, d) - v_true||²

# Where:
# - v̂_θ: Model's predicted velocity
# - v_true = Δs / d: Ground truth average velocity
```

**2. Self-Consistency Loss (40% weight):**
```python
# Location: training/losses.py:80-119

L_sc = ||v̂_θ(s, a, t, 2d) - (v̂_d1 + v̂_d2)/2||²

# Where:
# - v̂_θ(s, a, t, 2d): Velocity for one big 2d jump
# - v̂_d1: Velocity for first d jump
# - v̂_d2: Velocity for second d jump from intermediate state
#
# Enforces: Taking one 2d jump should equal average of two d jumps
```

**3. Velocity Magnitude Loss (20% weight):**
```python
# Location: training/losses.py:122-143

L_mag = |||v̂_θ|| - ||v_true|||²

# Prevents zero-velocity predictions
```

**4. Cascaded Consistency Loss (20% weight):**
```python
# Location: training/losses.py:146-225

# Multi-scale consistency checks:
# - 4d = d + d + d + d  (four small steps)
# - 4d = 2d + 2d        (two medium steps)
# - 8d = 4d + 4d        (two large steps)
```

---

## The Problem: Large-Timestep Performance Degradation

### Evaluation Results

Results from `evaluate_multi_dt.py` on collision-heavy test scenarios:

| Timestep (dt) | Sequential Error | Shortcut Error | Ratio (Seq/Short) | Expected |
|---------------|------------------|----------------|-------------------|----------|
| **0.05s**     | 4.0847          | 1.4870         | **2.78x** ✅      | 2-5x     |
| **0.1s**      | 4.1473          | 1.2944         | **3.20x** ✅      | 3-5x     |
| **0.2s**      | 4.2305          | 3.9377         | **1.07x** ⚠️      | 5-10x    |
| **0.5s**      | 10.2861         | 9.4441         | **1.09x** ❌      | 10-20x   |
| **1.0s**      | 9.7136          | 7.3533         | **1.32x** ❌      | 10-30x   |

### Key Observations

1. ✅ **Small timesteps (0.05-0.1s):** Shortcut model excels (2.78x-3.20x better)
2. ⚠️ **Medium timesteps (0.2s):** Advantage collapses to 1.07x
3. ❌ **Large timesteps (0.5-1.0s):** Both models fail badly, minimal differentiation
4. 🎯 **Expected behavior:** Shortcut model should maintain flat error across ALL dt values since it trains on them explicitly
5. 📉 **Actual behavior:** Error increases dramatically with dt, indicating temporal scaling is NOT being learned

### What This Means

The shortcut model is **not learning to scale across time**. Instead, it behaves like a sequential model that's slightly better calibrated at small dt. The bootstrap hierarchy and self-consistency losses are failing to teach true temporal composition.

---

## Root Cause Analysis

### The Division-by-dt Problem

**During collision scenarios:**

1. **Large state change:** Particle hits wall, velocity flips sign suddenly
   - Position change: `Δx ≈ 0.05` (small)
   - Velocity change: `Δvx = -2.0 - (+2.0) = -4.0` (large!)

2. **Computing training target:**
   ```python
   v_true = Δs / dt
   ```
   - For `dt = 0.01`: `v_true = [-4.0 / 0.01, ...] = [-400, ...]` 😱
   - For `dt = 0.1`: `v_true = [-4.0 / 0.1, ...] = [-40, ...]`
   - For `dt = 1.0`: `v_true = [-4.0 / 1.0, ...] = [-4, ...]`

3. **Training gradient corruption:**
   ```
   Loss = ||v̂_θ - v_true||²

   At small dt → v_true is HUGE → Gradient explodes
   At large dt → v_true is reasonable → Gradient is normal
   ```

4. **Result:**
   - Model learns to minimize loss at small dt (where gradients are largest)
   - Model ignores large dt (where gradients are reasonable but overwhelmed)
   - Temporal scaling is never learned properly

### Why Self-Consistency Doesn't Save It

The self-consistency loss operates on velocities too:
```python
L_sc = ||v̂_θ(s, 2d) - (v̂_d1 + v̂_d2)/2||²
```

If `v̂_d1` and `v̂_d2` are already corrupted by bad gradients from velocity matching loss, the self-consistency loss just teaches the model to be *consistently wrong*.

### Visualization of the Problem

```
Training Batch with Collision:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sample 1: dt=0.01, collision at t=0.005
    Δs = [0.01, 0.0, -3.8, 0.2]  (small position change, large velocity change)
    v_true = Δs/0.01 = [1.0, 0.0, -380, 20]  ← HUGE!
    Loss gradient: MASSIVE

Sample 2: dt=0.5, collision at t=0.2
    Δs = [0.45, 0.1, -2.1, 0.3]
    v_true = Δs/0.5 = [0.9, 0.2, -4.2, 0.6]  ← Reasonable
    Loss gradient: normal

Result: Model prioritizes Sample 1, ignores Sample 2
```

---

## Proposed Solution: State-Based Prediction

### Core Idea

**Eliminate the intermediate velocity representation entirely.**

Instead of:
```
Network → velocity → integrate → final state
```

Do:
```
Network → final state directly
```

### New Architecture: `StatePredictor`

```python
# Location: models/state_predictor.py (NEW FILE)

class StatePredictor(nn.Module):
    """
    Predicts final state directly after timestep d

    Input:  [state(4D), actions(40D), time(1D), dt(1D)] = 46D
    Output: final_state(4D) - state at time t+d

    Architecture: IDENTICAL to VelocityFieldNet
    - Hidden dims: [256, 256, 128]
    - Parameters: ~411,000 (fair comparison!)
    """

    def __init__(self, state_dim=4, action_dim=2, max_seq_len=10,
                 hidden_dims=[256, 256, 128]):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len

        # Input: state + actions + time + step_size (SAME AS BEFORE)
        input_dim = state_dim + action_dim * max_seq_len + 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer: predict FINAL STATE (not velocity!)
        layers.append(nn.Linear(prev_dim, state_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state, action_seq, time, step_size):
        """
        Args:
            state: (batch, 4) - Current state [x, y, vx, vy]
            action_seq: (batch, seq_len, 2) - Force sequence
            time: (batch, 1) - Current time
            step_size: (batch, 1) - Timestep size d

        Returns:
            final_state: (batch, 4) - Predicted state at t+d
        """
        batch_size = state.shape[0]

        # Flatten and pad action sequence (same as before)
        seq_len = action_seq.shape[1]
        action_flat = action_seq.reshape(batch_size, -1)

        if seq_len < self.max_seq_len:
            padding = torch.zeros(batch_size,
                                 (self.max_seq_len - seq_len) * self.action_dim,
                                 device=action_seq.device)
            action_flat = torch.cat([action_flat, padding], dim=1)

        # Concatenate inputs
        x = torch.cat([state, action_flat, time, step_size], dim=1)

        # Predict FINAL STATE directly
        final_state = self.network(x)

        return final_state
```

### New Loss Functions

**1. State Matching Loss (60% weight):**
```python
# Location: training/losses.py (MODIFIED)

def state_matching_loss(state_pred, state_true):
    """
    L_state = ||ŝ_θ(s, a, t, d) - s_true(t+d)||²

    Direct supervised learning of final state.
    NO DIVISION BY dt! Clean gradient signal.

    Args:
        state_pred: (batch, 4) - Predicted state at t+d
        state_true: (batch, 4) - True state at t+d from simulation

    Returns:
        loss: scalar
    """
    return F.mse_loss(state_pred, state_true)
```

**Key difference:** The training target `s_true(t+d)` is directly from simulation, no division needed!

**2. Self-Consistency Loss (40% weight):**
```python
def state_self_consistency_loss(model, state, action_seq, time, step_size):
    """
    L_sc = ||ŝ_θ(s, a, t, 2d) - ŝ_θ(ŝ_θ(s, a, t, d), a', t+d, d)||²

    Enforces: One big 2d jump = two sequential d jumps

    Still teaches temporal composition, but at state level!

    Args:
        model: StatePredictor
        state: (batch, 4)
        action_seq: (batch, seq_len, 2)
        time: (batch, 1)
        step_size: (batch, 1)

    Returns:
        loss: scalar
    """
    # One big jump of 2d
    state_2d_direct = model(state, action_seq, time, 2 * step_size)

    # First small jump of d
    state_d1 = model(state, action_seq, time, step_size)

    # Second small jump of d from intermediate state
    seq_len = action_seq.shape[1]
    action_seq_second = action_seq[:, seq_len//2:, :]
    state_d2 = model(state_d1, action_seq_second, time + step_size, step_size)

    # Compare final states
    return F.mse_loss(state_2d_direct, state_d2)
```

**3. Displacement Magnitude Loss (20% weight):**
```python
def displacement_magnitude_loss(state_pred, state_init, state_true, state_true_init):
    """
    L_disp = ||Δŝ|| - ||Δs_true|||²

    Where:
    - Δŝ = ŝ(t+d) - s(t): Predicted displacement
    - Δs_true = s_true(t+d) - s(t): True displacement

    Prevents zero-displacement pathology.
    Replaces velocity magnitude loss.

    Args:
        state_pred: (batch, 4) - Predicted final state
        state_init: (batch, 4) - Initial state
        state_true: (batch, 4) - True final state
        state_true_init: (batch, 4) - True initial state

    Returns:
        loss: scalar
    """
    # Compute displacements
    displacement_pred = state_pred - state_init
    displacement_true = state_true - state_true_init

    # Compute magnitudes
    mag_pred = torch.norm(displacement_pred, dim=1)
    mag_true = torch.norm(displacement_true, dim=1)

    return F.mse_loss(mag_pred, mag_true)
```

**4. Cascaded State Consistency (20% weight):**
```python
def cascaded_state_consistency_loss(model, state, action_seq, time, step_size):
    """
    Multi-scale state-level consistency:
    - 4d = d + d + d + d  (four steps)
    - 4d = 2d + 2d        (two steps)
    - 8d = 4d + 4d        (two large steps)

    Same concept as velocity version, but operates on states.
    """
    # Test 1: 4d via four d steps
    s_4d_direct = model(state, action_seq, time, 4 * step_size)

    current_state = state
    current_time = time
    for i in range(4):
        action_subset = action_seq[:, i*seq_len//4:(i+1)*seq_len//4, :]
        current_state = model(current_state, action_subset,
                             current_time, step_size)
        current_time = current_time + step_size

    s_4d_chained = current_state

    loss = F.mse_loss(s_4d_direct, s_4d_chained)

    # Similar tests for 2d and 8d...
    return loss / num_tests
```

### Training Target Computation (SIMPLIFIED!)

```python
def compute_true_final_state(env, state, actions, step_size):
    """
    Much cleaner than velocity computation!

    1. Simulate physics from state for duration step_size
    2. Return final state s(t+d) directly

    NO DIVISION! Just return the simulated state.
    """
    # Set environment state
    env.set_state(state)

    # Simulate
    num_sub_steps = int(step_size / env.dt)
    for step in range(num_sub_steps):
        action = actions[min(step, len(actions)-1)]
        next_state, _, _ = env.step(action)

    # Return final state directly
    return next_state  # ✅ No division!
```

---

## Implementation Guide

### Step 1: Create State Predictor Model

**File: `models/state_predictor.py`**

See [New Architecture: StatePredictor](#new-architecture-statepredictor) section above for complete code.

**Register in `models/__init__.py`:**
```python
from .state_predictor import StatePredictor

__all__ = ['VelocityFieldNet', 'ShortcutPredictor', 'StatePredictor']
```

### Step 2: Adapt Loss Functions

**File: `training/losses.py`**

Add the four new loss functions:
1. `state_matching_loss` (replaces `velocity_matching_loss`)
2. `state_self_consistency_loss` (replaces `self_consistency_loss`)
3. `displacement_magnitude_loss` (replaces `velocity_magnitude_loss`)
4. `cascaded_state_consistency_loss` (replaces `cascaded_self_consistency_loss`)

See [New Loss Functions](#new-loss-functions) section for complete implementations.

### Step 3: Create State-Based Trainer

**File: `training/state_bootstrap_trainer.py`**

```python
class StateBootstrapTrainer:
    """Bootstrap trainer for state-based prediction"""

    def __init__(self, model, optimizer, device, env=None,
                 supervised_levels=None, all_bootstrap_levels=None):
        self.model = model  # StatePredictor instance
        self.optimizer = optimizer
        self.device = device
        self.env = env

        # Bootstrap sampler (SAME AS BEFORE)
        self.batch_sampler = BootstrapBatchSampler(
            supervised_levels=supervised_levels or [0.01, 0.04, 0.08, 0.16, 0.64],
            all_bootstrap_levels=all_bootstrap_levels or [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0],
            velocity_ratio=0.6  # 60% supervised, 40% self-consistency
        )

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            # Move to device
            state = batch['state'].to(self.device)
            actions = batch['actions'].to(self.device)
            time = batch['time'].to(self.device)
            dt = batch['dt'].to(self.device)
            true_final_state = batch['final_state'].to(self.device)  # NEW!

            # Split into supervised/self-consistency batches
            velocity_batch, sc_batch = self.batch_sampler.sample_batch({
                'state': state,
                'actions': actions,
                'time': time,
                'dt': dt,
                'final_state': true_final_state
            })

            # ===== Supervised Loss (60%) =====
            state_pred = self.model(
                velocity_batch['state'],
                velocity_batch['actions'],
                velocity_batch['time'],
                velocity_batch['step_size']
            )

            loss_state = state_matching_loss(state_pred, velocity_batch['final_state'])
            loss_disp_mag = displacement_magnitude_loss(
                state_pred,
                velocity_batch['state'],
                velocity_batch['final_state'],
                velocity_batch['state']
            )

            supervised_loss = 0.6 * loss_state + 0.2 * loss_disp_mag

            # ===== Self-Consistency Loss (40%) =====
            if sc_batch is not None:
                loss_sc = state_self_consistency_loss(
                    self.model,
                    sc_batch['state'],
                    sc_batch['actions'],
                    sc_batch['time'],
                    sc_batch['step_size']
                )

                loss_cascaded = cascaded_state_consistency_loss(
                    self.model,
                    sc_batch['state'],
                    sc_batch['actions'],
                    sc_batch['time'],
                    sc_batch['step_size']
                )

                sc_loss = 0.4 * loss_sc + 0.2 * loss_cascaded
            else:
                sc_loss = 0

            # Total loss
            loss = supervised_loss + sc_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)
```

### Step 4: Modify Data Generation

**File: `utils/data_generation.py`**

Ensure dataset includes final states:

```python
def generate_training_data(env, num_trajectories, traj_length, action_seq_len):
    """Generate training data with FINAL STATES for state-based learning"""
    dataset = []

    for _ in range(num_trajectories):
        env.reset()

        for _ in range(traj_length):
            current_state = env.get_state()

            # Generate random action sequence
            actions = np.random.uniform(-2, 2, (action_seq_len, 2))

            # Sample random dt from bootstrap levels
            dt = np.random.choice([0.01, 0.04, 0.08, 0.16, 0.64])

            # Simulate to get final state
            final_state = simulate_physics(env, current_state, actions, dt)

            dataset.append({
                'state': current_state,
                'actions': actions,
                'time': 0.0,
                'dt': dt,
                'final_state': final_state  # NEW! No velocity computation needed
            })

    return dataset
```

### Step 5: Three-Way Training Comparison

**File: `train_three_way_comparison.py`**

```python
def main():
    """Train and compare THREE models:
    1. Sequential baseline (velocity-based, dt=0.01 only)
    2. Shortcut velocity-based (current approach, multi-dt)
    3. Shortcut state-based (NEW approach, multi-dt)
    """

    # Create three models with IDENTICAL architecture
    sequential_model = VelocityFieldNet(...).to(device)
    shortcut_velocity_model = VelocityFieldNet(...).to(device)
    shortcut_state_model = StatePredictor(...).to(device)  # NEW!

    # Verify parameter counts match
    assert sum(p.numel() for p in sequential_model.parameters()) == 411_000
    assert sum(p.numel() for p in shortcut_velocity_model.parameters()) == 411_000
    assert sum(p.numel() for p in shortcut_state_model.parameters()) == 411_000

    # Train all three
    train_sequential(sequential_model, train_loader)
    train_shortcut_velocity(shortcut_velocity_model, train_loader)
    train_shortcut_state(shortcut_state_model, train_loader)  # NEW!

    # Evaluate on multiple dt
    results = {}
    for dt in [0.05, 0.1, 0.2, 0.5, 1.0]:
        results[dt] = {
            'sequential': evaluate(sequential_model, test_data, dt),
            'shortcut_velocity': evaluate(shortcut_velocity_model, test_data, dt),
            'shortcut_state': evaluate(shortcut_state_model, test_data, dt)  # NEW!
        }

    # Generate comparison report
    print_three_way_comparison(results)
```

### Step 6: Quick Validation Test

**File: `test_state_predictor_collision.py`**

Before full training, validate the hypothesis:

```python
def quick_collision_test():
    """5-minute sanity check to validate state-based approach"""

    # Generate ONE hard collision scenario
    env = PointMass2D()
    state = np.array([4.5, 0.0, 2.0, 0.0])  # Near wall, moving toward it
    actions = np.zeros((10, 2))  # No forces, just collision

    # Create both models
    velocity_model = VelocityFieldNet(...).to(device)
    state_model = StatePredictor(...).to(device)

    # Train for 100 iterations on THIS SINGLE COLLISION
    for i in range(100):
        # Simulate ground truth
        true_final_state = simulate(env, state, actions, dt=1.0)

        # Velocity-based loss
        velocity_pred = velocity_model(state, actions, 0, 1.0)
        state_pred_vel = state + velocity_pred * 1.0
        loss_vel = F.mse_loss(state_pred_vel, true_final_state)

        # State-based loss
        state_pred_state = state_model(state, actions, 0, 1.0)
        loss_state = F.mse_loss(state_pred_state, true_final_state)

        # Backprop both
        optimizer_vel.zero_grad()
        loss_vel.backward()
        optimizer_vel.step()

        optimizer_state.zero_grad()
        loss_state.backward()
        optimizer_state.step()

        print(f"Iter {i}: Velocity loss={loss_vel:.6f}, State loss={loss_state:.6f}")

    # Expected: State loss should converge smoothly, velocity loss should be erratic
```

---

## Key Code Components

### Physics Environment

**Location:** `envs/realistic_physics_2d.py`

- Euler integration with dt=0.01
- Elastic boundary collisions (restitution=0.8)
- Inter-particle collision detection
- Momentum conservation
- Configurable damping and gravity

**Key methods:**
```python
env.reset()  # Initialize particles
env.step(action)  # Simulate one physics timestep (dt=0.01)
env.get_state()  # Get current state [x, y, vx, vy]
```

### Current Models

**Sequential Baseline:** `models/velocity_field.py:VelocityFieldNet`
- Trained only on dt=0.01 (physics grounding)
- Pure velocity matching loss
- 411K parameters

**Shortcut Velocity:** `models/shortcut_predictor.py:ShortcutPredictor`
- Wraps VelocityFieldNet
- Bootstrap training on [0.01, 0.02, ..., 1.0]
- Velocity + self-consistency losses
- 411K parameters

**Shortcut State (PROPOSED):** `models/state_predictor.py:StatePredictor`
- NEW model for proposed approach
- Same architecture as VelocityFieldNet
- State-based losses (no velocity)
- 411K parameters (fair comparison)

### Training Infrastructure

**Bootstrap Trainer:** `training/bootstrap_trainer.py`
- Manages bootstrap hierarchy
- Splits batches into supervised/self-consistency
- Handles multi-dt training

**Two-Network Trainer:** `training/two_network_trainer.py`
- Trains sequential + shortcut models in parallel
- Fair comparison framework
- Evaluation at multiple dt values

### Evaluation Scripts

**Multi-DT Evaluation:** `evaluate_multi_dt.py`
- Tests models at [0.05, 0.1, 0.2, 0.5, 1.0]
- Computes sequential/shortcut error ratios
- Identifies where differentiation breaks down

**Collision Focus:** `evaluate_collision_physics.py`
- Specifically tests on collision scenarios
- Separates smooth vs collision samples
- Validates dt-specific performance

---

## Validation Strategy

### Phase 1: Quick Sanity Check (30 minutes)

**Script:** `test_state_predictor_collision.py`

1. Create single hard collision scenario
2. Train both velocity-based and state-based models for 100 iterations
3. Plot training curves:
   - **Expected:** State-based has smooth gradient descent
   - **Expected:** Velocity-based has spiky, unstable gradients

**Success Criteria:**
- State-based loss decreases monotonically
- Velocity-based loss oscillates during collision phase
- Visual confirmation that state-based handles collisions better

### Phase 2: Full Training (4-8 hours)

**Script:** `train_three_way_comparison.py`

1. Train three models with identical architecture:
   - Sequential baseline (velocity, dt=0.01)
   - Shortcut velocity (velocity, multi-dt)
   - Shortcut state (state, multi-dt)
2. Bootstrap hierarchy: [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
3. 40 epochs, same data, same hyperparameters

**Tracked Metrics:**
- Training loss stability (variance over epochs)
- Validation loss convergence
- Error at each dt level [0.05, 0.1, 0.2, 0.5, 1.0]

### Phase 3: Comprehensive Evaluation (1 hour)

**Script:** `evaluate_three_way_multi_dt.py`

Test all three models on:
1. **Collision-heavy scenarios** (70% of test set)
2. **Smooth motion scenarios** (30% of test set)
3. **Multiple dt values:** [0.05, 0.1, 0.2, 0.5, 1.0]

**Generate reports:**
- Error tables by dt and model
- Speedup ratios (Sequential/Shortcut)
- Efficiency scores (Speedup × Accuracy)

---

## Success Criteria

### Minimum Success (Hypothesis Validated)

✅ State-based model at **dt=1.0**:
- Error < 3.0 (current velocity-based: 7.35)
- Ratio vs sequential > 3.0x (current: 1.32x)

✅ Flat error profile:
- Error at dt=1.0 similar to dt=0.05 (±50%)
- Current velocity-based: 7.35 vs 1.49 (5x worse!)

✅ Training stability:
- Loss decreases smoothly without spikes
- Validation loss continues improving past epoch 20

### Target Success (Production-Ready)

✅ State-based model across ALL dt:
- Error < 3.0 at every dt level
- Ratio vs sequential > 3.0x at every dt level
- Efficiency score (speedup × accuracy) > 10.0

✅ Collision robustness:
- Error on collision scenarios < 4.0
- Error on smooth scenarios < 2.0

✅ Computational speedup:
- 10x faster inference at dt=1.0 vs sequential
- Same memory footprint (411K params)

---

## Questions for Expert Review

### 1. Root Cause Validation

**Q:** Does eliminating the division by dt actually address the root cause of large-dt degradation?

**Context:**
- Division creates extreme velocity targets during collisions: v_true = Δs/d
- At small dt, collision velocities can be 100-1000x larger than normal velocities
- This dominates the loss landscape, making large-dt samples invisible

**Counterargument to consider:**
- Maybe the issue is data quality (not enough collision examples at large dt)?
- Maybe the issue is architecture capacity (network too small)?

### 2. Physical Interpretability

**Q:** Is losing the velocity intermediate representation a significant drawback?

**Current (Velocity-Based):**
- Output has clear physical meaning: average velocity over timestep
- Can verify if predictions respect physics constraints (e.g., velocity magnitude)
- Easier to debug (can plot velocity fields)

**Proposed (State-Based):**
- Output is just "future state" - black box displacement
- Less interpretable, harder to debug
- But: Does interpretability matter if accuracy improves?

### 3. Temporal Scaling Learning

**Q:** Can the model learn temporal scaling without explicit v × d multiplication?

**Current approach explicitly encodes time:**
```python
ŝ(t+d) = s(t) + v̂_θ(s,a,t,d) × d
```
The multiplication by d is built into the architecture.

**Proposed approach relies on the network:**
```python
ŝ(t+d) = f_θ(s,a,t,d)
```
The network must learn that d=2.0 means "twice as far" as d=1.0.

**Question:** Is self-consistency loss sufficient to teach this scaling? Or do we need the explicit multiplication?

### 4. Alternative Solutions

**Q:** Are there better alternatives that preserve velocity representation?

**Option A: Robust Loss Function**
- Use Huber loss instead of MSE to downweight outlier velocities
- Keep velocity-based architecture but make it robust to collision spikes

**Option B: Displacement Prediction**
- Predict Δs = s(t+d) - s(t) instead of velocity or final state
- Then: ŝ(t+d) = s(t) + Δs_θ(s,a,t,d)
- Still avoids division by dt, but conceptually closer to velocity

**Option C: Collision Detection + Specialized Handling**
- Detect collisions in training data
- Use different loss weights or separate networks for collision vs smooth regions

**Option D: Log-Scale Velocity**
- Predict log(||v||) and direction separately
- Reduces magnitude of extreme velocities in loss

### 5. Failure Modes

**Q:** What are the failure modes of state-based prediction that velocity-based might handle better?

**Potential issues:**
- **Extrapolation:** If dt at test time is outside training range, velocity-based might extrapolate better via v × d scaling
- **Conservation laws:** Velocity-based losses might enforce momentum conservation more naturally
- **Multi-step rollouts:** Compounding errors might accumulate differently

---

## Recommended Next Steps

1. **Quick Test (30 min):** Implement `test_state_predictor_collision.py` to validate gradient stability hypothesis

2. **Expert Review:** Get feedback on this document from physics simulation / ML researchers

3. **Full Implementation (2-4 hours):**
   - Implement `StatePredictor` model
   - Adapt loss functions
   - Create `StateBootstrapTrainer`

4. **Three-Way Training (4-8 hours):** Train all three models with identical setup

5. **Comparative Analysis (1-2 hours):** Generate comprehensive evaluation report

6. **Decision Point:** Based on results, decide whether to:
   - ✅ Adopt state-based approach (if successful)
   - 🔄 Iterate on hybrid approaches (if partially successful)
   - ❌ Investigate alternative solutions (if unsuccessful)

---

## Appendix: Key Equations

### Current System (Velocity-Based)

**Model:**
```
v̂ = f_θ(s, a, t, d)
ŝ(t+d) = s(t) + v̂ · d
```

**Training Target:**
```
v_true = (s(t+d) - s(t)) / d  ← PROBLEMATIC DIVISION
```

**Loss:**
```
L_total = 0.6 · ||v̂ - v_true||²
        + 0.4 · ||v̂_2d - (v̂_d1 + v̂_d2)/2||²
        + 0.2 · |||v̂|| - ||v_true|||²
        + 0.2 · L_cascaded
```

### Proposed System (State-Based)

**Model:**
```
ŝ(t+d) = f_θ(s, a, t, d)  ← Direct prediction
```

**Training Target:**
```
s_true(t+d) = simulate(s, a, d)  ← No division!
```

**Loss:**
```
L_total = 0.6 · ||ŝ - s_true||²
        + 0.4 · ||ŝ_2d - ŝ_θ(ŝ_d1)||²
        + 0.2 · |||ŝ - s|| - ||s_true - s|||²
        + 0.2 · L_cascaded_state
```

---

## Contact & References

**Repository:** `/home/pralak/Shortcut_actions`

**Key Papers:**
- Neural Shortcut Models for Physics Simulation
- Temporal Abstraction in Reinforcement Learning
- Learning to Simulate Complex Physics with Graph Networks

**For Questions:**
- Implementation issues → Check code comments in source files
- Theoretical questions → See "Questions for Expert Review" section above
- Results interpretation → Run evaluation scripts with `--wandb` flag for detailed metrics

---

*Last Updated: 2025-12-03*
*Status: Proposed Architecture - Awaiting Validation*
