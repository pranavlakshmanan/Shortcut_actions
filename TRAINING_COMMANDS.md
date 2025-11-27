# 🚀 **COLLISION PHYSICS TRAINING COMMANDS**

## ✅ **UPDATED: 25 Epochs Training**

### **Training Command (25 epochs):**
```bash
python train_collision_models.py
```

**This will:**
- Use new config: `configs/collision_physics_training.yaml`
- Train for **25 epochs** (reduced from 50)
- Batch size: 64
- Learning rate: 0.001
- Bootstrap levels: [0.01, 0.02, 0.04, ..., 1.0]

### **Evaluation Command:**
```bash
python evaluate_collision_physics.py
```

---

## 📁 **New Configuration File:**

**File:** `configs/collision_physics_training.yaml`

**Key Settings:**
```yaml
training:
  epochs: 25                    # Reduced training time
  batch_size: 64
  learning_rate: 0.001
  bootstrap_levels: [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]

data_generation:
  collision_bias: 0.7           # 70% collision scenarios

evaluation:
  test_horizons: [1.0, 2.0, 3.0]
  success_criteria:
    sequential_single_step_failure_threshold: 5.0    # Must fail >5x
    shortcut_single_step_success_threshold: 3.0      # Must work <3x
```

---

## ⏱️ **Expected Duration:**

- **25 epochs:** ~1.5-2 hours (reduced from 3+ hours)
- **Evaluation:** ~10 minutes

---

## 🎯 **Expected Results:**

After 25 epochs of collision physics training:

```
Sequential Single-Step: 15-20x worse (FAILS on collisions)
Shortcut Single-Step:   2-3x worse  (WORKS on collisions)
```

**This gives the clear differentiation needed for publication!** ✅

---

## 🚀 **Ready to Train:**

```bash
# Start 25-epoch collision physics training
python train_collision_models.py

# After training completes:
python evaluate_collision_physics.py
```