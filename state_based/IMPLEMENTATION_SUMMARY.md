# State-Based Shortcut: Implementation Summary

## ✅ All Files Created Successfully

### Location
```
~/Shortcut_actions/experiments/state_based/
```

### Files (7 total)

1. **`state_predictor.py`** (2.5 KB) - StatePredictor model
2. **`state_losses.py`** (1.7 KB) - Loss functions
3. **`state_trainer.py`** (7.2 KB) - Bootstrap trainer
4. **`quick_test.py`** (7.5 KB) - 5-minute validation test
5. **`train_state_shortcut.py`** (7.1 KB) - Full training script
6. **`evaluate_state.py`** (11 KB) - Three-way comparison
7. **`README.md`** (4.7 KB) - Experiment documentation

All scripts are executable (`chmod +x` applied).

---

## 🎯 Execution Order

### Step 1: Quick Test (5 minutes) - **RUN THIS FIRST**

```bash
cd ~/Shortcut_actions/experiments/state_based
python3 quick_test.py
```

**What it does:**
- Tests gradient stability on single hard collision
- Compares velocity vs state-based on identical scenario
- Generates plot: `quick_test_results.png`

**Decision criteria:**
- ✅ Pass: State loss smoother than velocity loss → proceed
- ❌ Fail: No clear advantage → debug before scaling

---

### Step 2: Full Training (4-8 hours)

**Only run if Step 1 passes!**

```bash
python3 train_state_shortcut.py --epochs 40 --batch_size 64 --wandb
```

**What it does:**
- Loads collision dataset (5000 train, 500 val samples)
- Trains StatePredictor with bootstrap hierarchy
- Saves best model to `state_best_model.pt`
- Logs training curves to wandb

**Monitor:**
- Training loss should decrease smoothly (no spikes)
- Validation loss should improve consistently
- State loss + SC loss should both converge

---

### Step 3: Evaluation (10-20 minutes)

```bash
python3 evaluate_state.py --num_samples 100
```

**What it does:**
- Loads three models: sequential, velocity shortcut, state shortcut
- Tests at dt = [0.05, 0.1, 0.2, 0.5, 1.0]
- Computes error ratios (Sequential/Shortcut)
- Generates comparison table

**Success criteria:**
- State-based ratio > 3.0x at dt=1.0
- State-based error < 3.0 at dt=1.0
- Flat error profile across all dt

---

## 🔑 Key Implementation Details

### StatePredictor Architecture

**Identical to VelocityFieldNet except:**
```python
# VelocityFieldNet
def forward(self, state, actions, time, dt):
    x = concat([state, actions, time, dt])
    velocity = self.network(x)
    return velocity  # Output is velocity

# StatePredictor
def forward(self, state, actions, time, dt):
    x = concat([state, actions, time, dt])
    final_state = self.network(x)
    return final_state  # Output is state directly
```

**Critical difference:** No multiplication by dt after network output.

---

### Loss Functions

**1. State Matching Loss (60%)**
```python
def state_matching_loss(pred_state, true_state):
    return F.mse_loss(pred_state, true_state)
```
- Direct supervised learning
- Training target: `s_true = simulate(s, a, dt)`
- **No division by dt!** (This is the key fix)

**2. Self-Consistency Loss (40%)**
```python
def state_self_consistency_loss(model, state, actions, time, dt):
    # One big 2d jump
    s_2d = model(state, actions, time, 2*dt)

    # Two small d jumps
    s_d1 = model(state, actions[:half], time, dt)
    s_d2 = model(s_d1.detach(), actions[half:], time+dt, dt)

    return F.mse_loss(s_2d, s_d2)
```
- Teaches temporal composition
- Operates on states, not velocities
- Uses `.detach()` to avoid second-order gradients

---

### Bootstrap Hierarchy

**Supervised levels** (ground truth from dataset):
- 0.01, 0.04, 0.08, 0.16, 0.64 seconds

**Self-consistency levels** (learned without ground truth):
- 0.02, 0.32, 1.0 seconds

**Batch split:**
- 60% of batch: state matching at supervised dt
- 40% of batch: self-consistency at random dt from SC levels

---

### Data Format

**Dataset structure:**
```python
sample = {
    'state': np.array([x, y, vx, vy]),        # Initial state (4,)
    'actions': np.array([[fx, fy], ...]),     # Forces (20, 2)
    'time': np.array([0.0]),                  # Current time (1,)
    'dt': np.array([0.16]),                   # Timestep (1,)
    'final_state': np.array([x', y', vx', vy'])  # Ground truth (4,)
}
```

**Key addition:** `final_state` field computed via physics simulation.

---

## 📊 Expected Results

### Baseline (from evaluate_multi_dt.py)

| dt   | Sequential Error | Velocity Shortcut | Ratio |
|------|-----------------|-------------------|-------|
| 0.05 | 4.08           | 1.49              | 2.78x |
| 0.1  | 4.15           | 1.29              | 3.20x |
| 0.2  | 4.23           | 3.94              | 1.07x ⚠️ |
| 0.5  | 10.29          | 9.44              | 1.09x ❌ |
| 1.0  | 9.71           | 7.35              | 1.32x ❌ |

**Problem:** Velocity-based degrades at large dt.

---

### Target (State-Based)

| dt   | Sequential Error | State Shortcut | Ratio  | Status |
|------|-----------------|----------------|--------|--------|
| 0.05 | 4.08           | 1.50           | 2.7x   | ✅     |
| 0.1  | 4.15           | 1.30           | 3.2x   | ✅     |
| 0.2  | 4.23           | 1.60           | 2.6x   | ✅     |
| 0.5  | 10.29          | 3.00           | 3.4x   | ✅     |
| 1.0  | 9.71           | 2.80           | 3.5x   | ✅     |

**Goal:** Flat error profile, consistent 3x improvement.

---

## 🐛 Troubleshooting

### Quick test shows no difference
- Check collision scenario (particle should hit wall)
- Verify learning rate (try 0.0001 - 0.01)
- Increase iterations (100 → 500)

### Training loss explodes
- Reduce learning rate (0.001 → 0.0001)
- Add gradient clipping (already in trainer, check max_norm)
- Check data normalization

### Evaluation shows no improvement
- Verify models loaded correctly (check parameter counts)
- Ensure test data has collisions (check collision_count field)
- Try more samples (--num_samples 500)

### State-based worse than velocity-based
- Check if dataset has final_state field
- Verify dt levels match between dataset and trainer
- Inspect training curves in wandb (should be smooth)

---

## 🔍 Verification Checklist

Before running experiments:

- [ ] Dataset exists: `data/collision_train.pkl`, `data/collision_val.pkl`
- [ ] Sequential model exists: `experiments/sequential_baseline_model.pt`
- [ ] Velocity shortcut exists: `experiments/shortcut_bootstrap_model.pt`
- [ ] Python path correct: scripts import from parent directories
- [ ] Environment available: `PointMass2D` from `envs` module
- [ ] Dependencies installed: torch, numpy, matplotlib, tqdm, wandb

---

## 📈 Monitoring During Training

**Healthy training signals:**
- Train loss decreases steadily
- Val loss decreases or plateaus (not increasing)
- State loss and SC loss both converge
- No NaN or Inf values

**Warning signs:**
- Loss spikes during training (gradient instability)
- Val loss increasing while train loss decreases (overfitting)
- State loss converges but SC loss stays high (poor composition learning)

---

## 🎓 Scientific Contribution

### Research Question
Can direct state prediction avoid the division-by-dt problem in velocity-based shortcut learning?

### Hypothesis
Division by dt creates extreme training targets during collisions, causing gradient imbalance that prevents learning temporal scaling at large timesteps.

### Test
Train state-based predictor with identical architecture and evaluate at multiple dt values.

### Success Metric
State-based achieves >3x speedup ratio at dt=1.0 (vs 1.32x for velocity-based).

### Impact
If successful, demonstrates that representational choice (velocity vs state) critically affects learning dynamics in physics prediction.

---

## 📝 Next Steps After Evaluation

### If Successful (ratio > 3.0x at dt=1.0)
1. Document results in main README
2. Run ablation studies:
   - Remove self-consistency loss (does it still work?)
   - Try different lambda weights (0.5/0.5, 0.7/0.3)
   - Test on smooth motion dataset (generalization)
3. Write up findings for publication

### If Partially Successful (2.0x < ratio < 3.0x)
1. Analyze failure modes:
   - Which dt values fail?
   - Collision vs smooth scenarios?
2. Try hybrid approach:
   - Predict both state and velocity
   - Use velocity for physics consistency checks
3. Experiment with loss weights

### If Unsuccessful (ratio < 2.0x)
1. Debug thoroughly:
   - Visualize predictions vs ground truth
   - Check if model is learning anything (train accuracy?)
   - Compare gradient magnitudes with velocity-based
2. Consider alternative hypotheses:
   - Maybe issue is data quality, not architecture
   - Maybe need bigger network for direct state prediction
3. Try alternative approaches from README:
   - Displacement prediction (Δs instead of s)
   - Robust loss functions (Huber)
   - Log-scale velocity

---

*Implementation completed: 2025-12-03*
*Ready for execution: All systems go! 🚀*
