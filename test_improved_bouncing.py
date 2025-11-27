#!/usr/bin/env python3
"""
Test improved bouncing with high restitution and low damping
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def test_improved_bouncing():
    """Test with optimized physics parameters"""

    # Create environment with no damping and perfect elasticity
    env = PointMass2D(dt=0.01, damping=0.0)
    env.restitution = 1.0  # Perfectly elastic collisions

    # Position closer to walls for rapid bouncing, velocity within limits
    x, y = 4.5, 4.3  # Very close to walls
    vx, vy = -9.5, -9.5  # High velocity within bounds, heading toward center for cross-bouncing

    print(f"Improved bouncing test:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Physics: damping=0.0, restitution=1.0 (perfectly elastic)")

    # No external forces
    trajectory_length = 101
    force_pattern = np.zeros((trajectory_length, 2))

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

        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
            position = trajectory[-1][:2]
            velocity = trajectory[-1][2:]
            speed = np.linalg.norm(velocity)

            if np.any(vel_change > 1.0):
                collision_count += 1
                collision_times.append(t+1)
                if collision_count <= 20:
                    wall_hit = "X-wall" if vel_change[0] > vel_change[1] else "Y-wall"
                    print(f"Step {t+1}: COLLISION #{collision_count} ({wall_hit}) - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), "
                          f"vel=({velocity[0]:.3f}, {velocity[1]:.3f}), speed={speed:.3f}")

            if np.any(np.abs(position) > 5.0):
                print(f"   *** OUT OF BOUNDS! ***")
                break

    collision_density = collision_count / len(trajectory) * 100
    print(f"\nImproved bouncing results:")
    print(f"Total collisions: {collision_count}")
    print(f"Collision density: {collision_density:.2f}% of timesteps")
    print(f"Collision times: {collision_times}")

    # Check if we achieved target
    target_achieved = collision_count >= 5 and collision_density >= 3.0
    print(f"Target achieved (5+ collisions, 3%+ density): {'✅ YES' if target_achieved else '❌ NO'}")

    return collision_count, collision_density

if __name__ == "__main__":
    test_improved_bouncing()