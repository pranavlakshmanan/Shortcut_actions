#!/usr/bin/env python3
"""
Test ricochet patterns - position particle to bounce between walls rapidly
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def test_horizontal_ricochet():
    """Test horizontal bouncing between left and right walls"""

    # Create environment with minimal damping
    env = PointMass2D(dt=0.01, mass=1.0, damping=0.005)

    # Position in center with high horizontal velocity
    x, y = 0.0, 0.0  # Center position
    vx, vy = 8.0, 0.1  # Fast horizontal, tiny vertical

    print(f"Horizontal ricochet test:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")

    # Generate minimal force pattern
    trajectory_length = 101
    force_pattern = np.random.uniform(-0.05, 0.05, (trajectory_length, 2))

    # Simulate trajectory
    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0
    collision_times = []

    for t in range(trajectory_length - 1):
        force = force_pattern[t]
        state, _, _ = env.step(force)
        trajectory.append(state.copy())

        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
            position = trajectory[-1][:2]
            velocity = trajectory[-1][2:]

            if np.any(vel_change > 1.0):
                collision_count += 1
                collision_times.append(t+1)
                if collision_count <= 15:
                    print(f"Step {t+1}: COLLISION #{collision_count} - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), "
                          f"vel=({velocity[0]:.3f}, {velocity[1]:.3f})")

            if np.any(np.abs(position) > 5.0):
                print(f"   *** OUT OF BOUNDS! ***")
                break

    collision_density = collision_count / len(trajectory) * 100
    print(f"Horizontal ricochet: {collision_count} collisions ({collision_density:.2f}% density)")
    return collision_count, collision_density

def test_diagonal_ricochet():
    """Test diagonal bouncing for maximum collision potential"""

    # Create environment with minimal damping
    env = PointMass2D(dt=0.01, mass=1.0, damping=0.002)  # Even less damping

    # Position slightly off-center with diagonal velocity
    x, y = 1.0, 1.0  # Slightly off-center for asymmetric bouncing
    vx, vy = 7.0, 6.0  # High diagonal velocity

    print(f"\nDiagonal ricochet test:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")

    # Generate minimal force pattern
    trajectory_length = 101
    force_pattern = np.random.uniform(-0.02, 0.02, (trajectory_length, 2))

    # Simulate trajectory
    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0
    collision_times = []

    for t in range(trajectory_length - 1):
        force = force_pattern[t]
        state, _, _ = env.step(force)
        trajectory.append(state.copy())

        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
            position = trajectory[-1][:2]
            velocity = trajectory[-1][2:]

            if np.any(vel_change > 1.0):
                collision_count += 1
                collision_times.append(t+1)
                if collision_count <= 15:
                    print(f"Step {t+1}: COLLISION #{collision_count} - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), "
                          f"vel=({velocity[0]:.3f}, {velocity[1]:.3f})")

            if np.any(np.abs(position) > 5.0):
                print(f"   *** OUT OF BOUNDS! ***")
                break

    collision_density = collision_count / len(trajectory) * 100
    print(f"Diagonal ricochet: {collision_count} collisions ({collision_density:.2f}% density)")
    return collision_count, collision_density

def test_extreme_parameters():
    """Test with extreme parameters for maximum collisions"""

    # Create environment with almost no damping
    env = PointMass2D(dt=0.01, mass=1.0, damping=0.001)

    # Position near wall with extreme velocity
    x, y = 4.0, 3.0  # Near corner but not too close
    vx, vy = 12.0, 10.0  # Extreme velocity

    print(f"\nExtreme parameters test:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")

    # No external forces at all
    trajectory_length = 101
    force_pattern = np.zeros((trajectory_length, 2))

    # Simulate trajectory
    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0
    collision_times = []

    for t in range(trajectory_length - 1):
        force = force_pattern[t]
        state, _, _ = env.step(force)
        trajectory.append(state.copy())

        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
            position = trajectory[-1][:2]
            velocity = trajectory[-1][2:]

            if np.any(vel_change > 1.0):
                collision_count += 1
                collision_times.append(t+1)
                if collision_count <= 20:
                    wall_hit = "X-wall" if vel_change[0] > vel_change[1] else "Y-wall"
                    print(f"Step {t+1}: COLLISION #{collision_count} ({wall_hit}) - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), "
                          f"vel=({velocity[0]:.3f}, {velocity[1]:.3f})")

            if np.any(np.abs(position) > 5.0):
                print(f"   *** OUT OF BOUNDS! ***")
                break

    collision_density = collision_count / len(trajectory) * 100
    print(f"Extreme parameters: {collision_count} collisions ({collision_density:.2f}% density)")
    return collision_count, collision_density

if __name__ == "__main__":
    print("Testing ricochet patterns for high collision frequency...\n")

    # Test different approaches
    count1, density1 = test_horizontal_ricochet()
    count2, density2 = test_diagonal_ricochet()
    count3, density3 = test_extreme_parameters()

    print(f"\n=== SUMMARY ===")
    print(f"Horizontal ricochet: {count1} collisions ({density1:.2f}% density)")
    print(f"Diagonal ricochet: {count2} collisions ({density2:.2f}% density)")
    print(f"Extreme parameters: {count3} collisions ({density3:.2f}% density)")
    print(f"Target: 5-20 collisions (3-15% density)")

    if count3 >= 5:
        print(f"\n✅ SUCCESS: Extreme parameters achieved target collision frequency!")
    else:
        print(f"\n❌ Need further optimization to reach target collision frequency")