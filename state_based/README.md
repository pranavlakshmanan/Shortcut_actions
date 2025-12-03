# State-Based Shortcut Experiment

## Overview

This experiment implements **direct state prediction** as an alternative to velocity field prediction for physics shortcut learning.

**Hypothesis:** Predicting final states directly avoids the problematic division-by-dt operation that creates extreme training targets during collisions, leading to better performance at large timesteps.

---

## Quick Start

### 1. Quick Collision Test (5 minutes)

**Run this first** to validate gradient stability:

```bash
cd ~/Shortcut_actions/experiments/state_based
python3 quick_test.py
```

**Expected result:**
- State-based loss decreases smoothly
- Velocity-based loss oscillates/spikes
- Plot saved to `quick_test_results.png`

**Decision:** If state-based shows better stability → proceed to full training

---

### 2. Full Training (4-8 hours)

```bash
python3 train_state_shortcut.py --epochs 40 --batch_size 64 --wandb
```

**What it does:**
- Trains StatePredictor with bootstrap hierarchy
- Supervised levels: [0.01, 0.04, 0.08, 0.16, 0.64]
- Self-consistency levels: [0.02, 0.32, 1.0]
- Saves best model to `state_best_model.pt`
- Logs to wandb project "shortcut-state-based"

---

### 3. Evaluation

```bash
python3 evaluate_state.py --num_samples 100
```

**Compares three models:**
1. Sequential baseline (velocity, autoregressive)
2. Velocity-based shortcut (existing)
3. State-based shortcut (new)

**Tests at:** dt = [0.05, 0.1, 0.2, 0.5, 1.0]

**Success criteria:**
- State-based ratio > 3.0x at dt=1.0
- Flat error profile (dt=1.0 error ≈ dt=0.05 error)

---

## Files

### Core Implementation

- **`state_predictor.py`** - StatePredictor model (identical arch to VelocityFieldNet)
- **`state_losses.py`** - State matching + self-consistency losses
- **`state_trainer.py`** - Bootstrap trainer for state-based learning

### Scripts

- **`quick_test.py`** - 5-min gradient stability test
- **`train_state_shortcut.py`** - Full training script
- **`evaluate_state.py`** - Three-way model comparison

---

## Key Differences from Velocity-Based

### Model Output

**Velocity-based:**
```python
v = velocity_net(s, a, t, d)
s_next = s + v * d  # Explicit integration
```

**State-based:**
```python
s_next = state_predictor(s, a, t, d)  # Direct prediction
```

### Training Target

**Velocity-based:**
```python
v_true = (s_final - s_init) / dt  # ⚠️ Division by dt!
```

**State-based:**
```python
s_true = simulate(s_init, actions, dt)  # ✅ No division!
```

### Loss Functions

**Velocity-based:**
```python
L = 0.6 * ||v_pred - v_true||² + 0.4 * L_self_consistency(velocities)
```

**State-based:**
```python
L = 0.6 * ||s_pred - s_true||² + 0.4 * L_self_consistency(states)
```

---

## Architecture Details

### StatePredictor

- **Input:** [state(4), actions(40), time(1), dt(1)] = 46 dims
- **Hidden:** [64, 64, 64, 64] with LayerNorm + ReLU
- **Output:** 4 dims (final state directly)
- **Parameters:** ~330K (identical to VelocityFieldNet)

### Bootstrap Hierarchy

**Supervised (with ground truth):**
- dt = 0.01s (physics grounding)
- dt = 0.04s
- dt = 0.08s
- dt = 0.16s
- dt = 0.64s

**Self-consistency (learned):**
- dt = 0.02s
- dt = 0.32s
- dt = 1.0s

### Loss Weights

- 60% state matching (supervised)
- 40% self-consistency (temporal composition)

---

## Expected Results

### Minimum Success

At **dt=1.0**:
- State-based error < 3.0
- Ratio (Sequential/State) > 3.0x
- Better than velocity-based (current: 1.32x)

### Target Success

- Flat error across all dt values
- Ratio > 3.0x at all dt
- Smooth training curves (no spikes)

---

## Troubleshooting

### Quick test fails
- Check if environment dt=0.01 matches expected
- Verify collision actually occurs (x=4.5, vx=2.0 → wall at x=5.0)
- Try different learning rates (0.0001 - 0.01)

### Training loss doesn't decrease
- Check dataset format (needs 'final_state' field)
- Verify dt levels in dataset match supervised levels
- Reduce batch size if memory issues

### Evaluation shows no improvement
- Ensure models loaded correctly (check parameter counts)
- Verify test data is collision-heavy (not smooth motion)
- Try more test samples (--num_samples 500)

---

## References

- Main README: `~/Shortcut_actions/README.md`
- Velocity model: `~/Shortcut_actions/models/velocity_field.py`
- Bootstrap trainer: `~/Shortcut_actions/training/bootstrap_trainer.py`
- Evaluation baseline: `~/Shortcut_actions/evaluate_multi_dt.py`

---

## Next Steps

1. ✅ Run `quick_test.py` → validate hypothesis
2. ✅ Run `train_state_shortcut.py` → train model
3. ✅ Run `evaluate_state.py` → compare results
4. 📊 Analyze wandb logs for insights
5. 📝 Document findings in main README

---

*Experiment created: 2025-12-03*
