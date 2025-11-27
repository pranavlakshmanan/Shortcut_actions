#!/usr/bin/env python3
"""
Generate collision-heavy dataset with guaranteed 70% collision scenarios
Fixes the data distribution bug that caused Sequential and Shortcut models
to perform identically.
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def generate_collision_heavy_dataset(
    num_samples=5000,
    collision_ratio=0.7,
    trajectory_length=101,  # 1 second at dt=0.01
    seed=42
):
    """
    Generate dataset with guaranteed collision ratio

    Args:
        num_samples: Total number of trajectories
        collision_ratio: Fraction that MUST have collisions (0.7 = 70%)
        trajectory_length: Number of timesteps per trajectory
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create environment with optimal settings for sustained multi-collision bouncing
    env = PointMass2D(dt=0.01, damping=0.0)  # No damping for energy conservation
    env.restitution = 1.0  # Perfect elasticity for sustained bouncing

    # Make box VERY small for rapid bouncing
    env.pos_bounds = [-1.0, 1.0]  # 2x2 box for maximum collision frequency

    # Calculate exact counts
    num_collision_samples = int(num_samples * collision_ratio)
    num_smooth_samples = num_samples - num_collision_samples

    print(f"Generating {num_samples} samples:")
    print(f"  - {num_collision_samples} WITH collisions ({collision_ratio*100:.0f}%)")
    print(f"  - {num_smooth_samples} WITHOUT collisions ({(1-collision_ratio)*100:.0f}%)")

    dataset = []

    # Generate collision samples FIRST to guarantee ratio
    print("\n📌 Generating collision scenarios...")
    samples_generated = 0
    attempts = 0
    max_attempts = num_collision_samples * 10  # Prevent infinite loop

    with tqdm(total=num_collision_samples) as pbar:
        while samples_generated < num_collision_samples and attempts < max_attempts:
            attempts += 1

            # Position very close to walls with high velocities for rapid bouncing
            wall_side = np.random.choice(['left', 'right', 'top', 'bottom',
                                         'corner_tr', 'corner_tl', 'corner_br', 'corner_bl'])

            if wall_side == 'right':
                x = np.random.uniform(0.8, 0.95)  # Close to right wall at x=1.0
                y = np.random.uniform(-0.8, 0.8)  # Fit within 2x2 box
                vx = np.random.uniform(4.0, 6.0)  # Moderate velocities for sustained collision + trainability
                vy = np.random.uniform(-3.0, 3.0)  # Moderate cross-velocity
            elif wall_side == 'left':
                x = np.random.uniform(-0.95, -0.8)  # Close to left wall at x=-1.0
                y = np.random.uniform(-0.8, 0.8)  # Fit within 2x2 box
                vx = np.random.uniform(-6.0, -4.0)  # Moderate velocities for sustained collision + trainability
                vy = np.random.uniform(-3.0, 3.0)  # Moderate cross-velocity
            elif wall_side == 'top':
                x = np.random.uniform(-0.8, 0.8)  # Fit within 2x2 box
                y = np.random.uniform(0.8, 0.95)  # Close to top wall at y=1.0
                vx = np.random.uniform(-3.0, 3.0)  # Moderate cross-velocity
                vy = np.random.uniform(4.0, 6.0)  # Moderate velocities for sustained collision + trainability
            elif wall_side == 'bottom':
                x = np.random.uniform(-0.8, 0.8)  # Fit within 2x2 box
                y = np.random.uniform(-0.95, -0.8)  # Close to bottom wall at y=-1.0
                vx = np.random.uniform(-3.0, 3.0)  # Moderate cross-velocity
                vy = np.random.uniform(-6.0, -4.0)  # Moderate velocities for sustained collision + trainability
            elif wall_side == 'corner_tr':  # Top-right corner - for maximum bouncing
                x = np.random.uniform(0.7, 0.9)  # Top-right corner of 2x2 box
                y = np.random.uniform(0.7, 0.9)
                vx = np.random.uniform(-6.0, -4.0)  # Moderate velocities for diagonal bouncing
                vy = np.random.uniform(-6.0, -4.0)  # Moderate velocities for diagonal bouncing
            elif wall_side == 'corner_tl':  # Top-left corner
                x = np.random.uniform(-0.9, -0.7)  # Top-left corner of 2x2 box
                y = np.random.uniform(0.7, 0.9)
                vx = np.random.uniform(4.0, 6.0)  # Moderate velocities for diagonal bouncing
                vy = np.random.uniform(-6.0, -4.0)  # Moderate velocities for diagonal bouncing
            elif wall_side == 'corner_br':  # Bottom-right corner
                x = np.random.uniform(0.7, 0.9)  # Bottom-right corner of 2x2 box
                y = np.random.uniform(-0.9, -0.7)
                vx = np.random.uniform(-6.0, -4.0)  # Moderate velocities for diagonal bouncing
                vy = np.random.uniform(4.0, 6.0)  # Moderate velocities for diagonal bouncing
            else:  # corner_bl - Bottom-left corner
                x = np.random.uniform(-0.9, -0.7)  # Bottom-left corner of 2x2 box
                y = np.random.uniform(-0.9, -0.7)
                vx = np.random.uniform(4.0, 6.0)  # Moderate velocities for diagonal bouncing
                vy = np.random.uniform(4.0, 6.0)  # Moderate velocities for diagonal bouncing

            # Generate smaller force pattern to not interfere with bouncing
            force_pattern = np.random.uniform(-0.2, 0.2, (trajectory_length, 2))

            # Simulate trajectory
            env.clear_particles()
            env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

            trajectory = [np.array([x, y, vx, vy])]
            collision_count = 0

            for t in range(trajectory_length - 1):
                force = force_pattern[t]
                state, _, _ = env.step(force)
                trajectory.append(state.copy())

                # Check for velocity reversal (indicates collision)
                if t > 0:
                    vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
                    if np.any(vel_change > 1.0):  # Significant velocity change
                        collision_count += 1

            # Keep trajectories with multiple collisions (3+ for sustainable bouncing)
            # Lower threshold to accommodate reduced velocities while maintaining quality
            if collision_count >= 3:
                dataset.append({
                    'scenario': {
                        'initial_state': np.array([x, y, vx, vy]),
                        'force_pattern': force_pattern,
                        'wall_target': wall_side,
                        'collision_count': collision_count
                    },
                    'trajectory': np.array(trajectory)
                })
                samples_generated += 1
                pbar.update(1)

    actual_collision_samples = len([s for s in dataset if 'wall_target' in s['scenario']])
    if actual_collision_samples < num_collision_samples:
        print(f"⚠️ Warning: Only generated {actual_collision_samples}/{num_collision_samples} collision samples")

    # Generate smooth motion samples
    print("\n📌 Generating smooth motion scenarios...")

    with tqdm(total=num_smooth_samples) as pbar:
        smooth_count = 0
        attempts = 0
        max_attempts = num_smooth_samples * 10

        while smooth_count < num_smooth_samples and attempts < max_attempts:
            attempts += 1

            # Start in center of 2x2 box to avoid walls
            x = np.random.uniform(-0.5, 0.5)  # Center of 2x2 box
            y = np.random.uniform(-0.5, 0.5)
            vx = np.random.uniform(-0.5, 0.5)  # Low velocity
            vy = np.random.uniform(-0.5, 0.5)

            # Gentle forces
            force_pattern = np.random.uniform(-0.3, 0.3, (trajectory_length, 2))

            # Simulate
            env.clear_particles()
            env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

            trajectory = [np.array([x, y, vx, vy])]
            had_collision = False

            for t in range(trajectory_length - 1):
                force = force_pattern[t]
                state, _, _ = env.step(force)
                trajectory.append(state.copy())

                # Check for collisions with 2x2 box walls
                if np.any(np.abs(state[:2]) > 0.95):  # Near wall in 2x2 box
                    had_collision = True
                    break

            # Only keep if NO collision
            if not had_collision:
                dataset.append({
                    'scenario': {
                        'initial_state': np.array([x, y, vx, vy]),
                        'force_pattern': force_pattern,
                        'motion_type': 'smooth'
                    },
                    'trajectory': np.array(trajectory)
                })
                smooth_count += 1
                pbar.update(1)

    # Shuffle dataset
    np.random.shuffle(dataset)

    # Validate distribution and collision density
    validate_collision_distribution(dataset)
    validate_collision_density(dataset)

    return dataset

def validate_collision_density(dataset):
    """Measure collision frequency and density in generated dataset"""

    collision_samples = [s for s in dataset if 'wall_target' in s['scenario']]
    smooth_samples = [s for s in dataset if 'motion_type' in s['scenario']]

    if len(collision_samples) == 0:
        print("\n❌ No collision samples found!")
        return 0.0, 0.0

    # Calculate collision statistics
    total_collisions = 0
    total_timesteps = 0
    collision_counts = []

    for sample in collision_samples:
        trajectory = sample['trajectory']
        sample_collisions = 0

        # Count collisions in trajectory
        for i in range(1, len(trajectory)):
            vel_change = np.abs(trajectory[i][2:] - trajectory[i-1][2:])
            if np.any(vel_change > 1.0):
                sample_collisions += 1

        collision_counts.append(sample_collisions)
        total_collisions += sample_collisions
        total_timesteps += len(trajectory)

    # Calculate metrics
    avg_collisions_per_trajectory = np.mean(collision_counts)
    collision_density = total_collisions / total_timesteps * 100  # Percentage

    print(f"\n🏀 High-Frequency Collision Analysis:")
    print(f"   Collision samples: {len(collision_samples)}")
    print(f"   Smooth samples: {len(smooth_samples)}")
    print(f"   Total collisions detected: {total_collisions}")
    print(f"   Average collisions per trajectory: {avg_collisions_per_trajectory:.1f}")
    print(f"   Collision density: {collision_density:.2f}% of timesteps")
    print(f"   Min collisions: {min(collision_counts) if collision_counts else 0}")
    print(f"   Max collisions: {max(collision_counts) if collision_counts else 0}")

    # Target validation (moderate frequency with reduced velocities: 3-8 collisions, 2-8% density)
    if avg_collisions_per_trajectory < 3:
        print("   ⚠️ WARNING: Average collisions below 3 per trajectory!")
    elif avg_collisions_per_trajectory > 8:
        print("   ⚠️ WARNING: Average collisions above 8 per trajectory!")
    else:
        print("   ✅ Collision frequency within target range (3-8 per trajectory)")

    if collision_density < 2.0:
        print("   ⚠️ WARNING: Collision density below 2%!")
    elif collision_density > 8.0:
        print("   ⚠️ WARNING: Collision density above 8%!")
    else:
        print("   ✅ Collision density within target range (2-8%)")

    return avg_collisions_per_trajectory, collision_density

def validate_collision_distribution(dataset):
    """Verify the collision ratio in generated dataset"""

    collision_count = 0
    total = len(dataset)

    for sample in dataset:
        # Check if sample has collisions (either stored count or detect from trajectory)
        if 'wall_target' in sample['scenario']:
            collision_count += 1
        else:
            # Fallback: check trajectory for collisions
            trajectory = sample['trajectory']
            for i in range(1, len(trajectory)):
                vel_change = np.abs(trajectory[i][2:] - trajectory[i-1][2:])
                if np.any(vel_change > 1.0):
                    collision_count += 1
                    break

    collision_ratio = collision_count / total

    print(f"\n✅ Dataset Distribution Validation:")
    print(f"   Total samples: {total}")
    print(f"   Samples with collisions: {collision_count} ({collision_ratio*100:.1f}%)")
    print(f"   Samples without collisions: {total - collision_count} ({(1-collision_ratio)*100:.1f}%)")

    if collision_ratio < 0.65:
        print("   ⚠️ WARNING: Collision ratio below 65% target!")
    elif collision_ratio > 0.75:
        print("   ⚠️ WARNING: Collision ratio above 75% target!")
    else:
        print("   ✅ Collision ratio within target range (65-75%)")

    return collision_ratio

def main():
    """Generate train, val, and test datasets with correct collision distribution"""

    print("="*80)
    print("COLLISION DATASET GENERATION - FIXED VERSION")
    print("="*80)
    print("Target: 70% collision scenarios for proper model differentiation")
    print()

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate datasets with different seeds for variety
    datasets = {
        'train': {'size': 5000, 'seed': 42},
        'val': {'size': 500, 'seed': 123},
        'test': {'size': 500, 'seed': 456}
    }

    for split_name, config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Generating {split_name} dataset...")
        print(f"{'='*60}")

        dataset = generate_collision_heavy_dataset(
            num_samples=config['size'],
            collision_ratio=0.7,
            seed=config['seed']
        )

        # Save dataset
        filename = f"collision_{split_name}.pkl"
        filepath = data_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"💾 Saved to {filepath}")
        print(f"   Size: {filepath.stat().st_size / (1024**2):.1f} MB")

    print("\n" + "="*80)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run training: python train_collision_models.py")
    print("2. Expect clear differentiation after training:")
    print("   - Sequential Single-Step: 15-20x error (catastrophic failure)")
    print("   - Shortcut Single-Step: 2-4x error (maintains accuracy)")

if __name__ == "__main__":
    main()