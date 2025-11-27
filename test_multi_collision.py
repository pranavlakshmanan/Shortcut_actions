#!/usr/bin/env python3
"""
Test multi-collision scenarios to understand how to create 5+ collisions per trajectory
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def test_corner_bouncing():
    """Test corner bouncing for maximum collisions"""

    # Create environment with minimal damping
    env = PointMass2D(dt=0.01, mass=1.0, damping=0.01)

    # Position in corner with diagonal velocity for maximum bouncing
    x, y = 4.7, 4.7  # Top-right corner
    vx, vy = -6.0, -6.0  # High diagonal velocity toward center

    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Environment bounds: x ∈ [-5.0, 5.0], y ∈ [-5.0, 5.0]")

    # Generate minimal force pattern
    trajectory_length = 101
    force_pattern = np.zeros((trajectory_length, 2))  # No external forces

    # Simulate trajectory
    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0
    collision_times = []

    print("\nSimulating trajectory...")
    for t in range(trajectory_length - 1):
        force = force_pattern[t]
        state, _, _ = env.step(force)
        trajectory.append(state.copy())

        # Check for velocity reversal (indicates collision)
        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
            position = trajectory[-1][:2]
            velocity = trajectory[-1][2:]

            if np.any(vel_change > 1.0):  # Significant velocity change
                collision_count += 1
                collision_times.append(t+1)
                print(f"Step {t+1}: COLLISION #{collision_count} - "
                      f"pos=({position[0]:.3f}, {position[1]:.3f}), "
                      f"vel=({velocity[0]:.3f}, {velocity[1]:.3f}), "
                      f"vel_change=({vel_change[0]:.3f}, {vel_change[1]:.3f})")

            # Check if particle is out of bounds
            if np.any(np.abs(position) > 5.0):
                print(f"   *** PARTICLE OUT OF BOUNDS! ***")
                break

    print(f"\nTotal collisions: {collision_count}")
    print(f"Collision times: {collision_times}")

    # Calculate collision density
    collision_density = collision_count / len(trajectory) * 100
    print(f"Collision density: {collision_density:.2f}% of timesteps")

    return collision_count, collision_density

def test_optimized_bouncing():
    """Test optimized parameters for high collision frequency"""

    # Create environment with even less damping
    env = PointMass2D(dt=0.01, mass=1.0, damping=0.005)  # Half the previous damping

    # Position very close to corner with extreme velocity
    x, y = 4.85, 4.85  # Very close to top-right corner
    vx, vy = -10.0, -10.0  # Extreme diagonal velocity

    print(f"\nOptimized test:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Reduced damping: 0.005")

    # Generate minimal force pattern
    trajectory_length = 101
    force_pattern = np.random.uniform(-0.1, 0.1, (trajectory_length, 2))  # Tiny random forces

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

        # Check for velocity reversal (indicates collision)
        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
            position = trajectory[-1][:2]
            velocity = trajectory[-1][2:]

            if np.any(vel_change > 1.0):  # Significant velocity change
                collision_count += 1
                collision_times.append(t+1)
                if collision_count <= 10:  # Print first 10 collisions
                    print(f"Step {t+1}: COLLISION #{collision_count} - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), "
                          f"vel=({velocity[0]:.3f}, {velocity[1]:.3f})")

            # Check if particle is out of bounds
            if np.any(np.abs(position) > 5.0):
                print(f"   *** PARTICLE OUT OF BOUNDS! ***")
                break

    print(f"\nOptimized total collisions: {collision_count}")
    collision_density = collision_count / len(trajectory) * 100
    print(f"Optimized collision density: {collision_density:.2f}% of timesteps")

    return collision_count, collision_density

if __name__ == "__main__":
    print("Testing corner bouncing scenarios for high collision frequency...")

    # Test basic corner bouncing
    count1, density1 = test_corner_bouncing()

    # Test optimized parameters
    count2, density2 = test_optimized_bouncing()

    print(f"\n=== SUMMARY ===")
    print(f"Basic corner: {count1} collisions ({density1:.2f}% density)")
    print(f"Optimized: {count2} collisions ({density2:.2f}% density)")
    print(f"Target: 5-20 collisions (3-15% density)")