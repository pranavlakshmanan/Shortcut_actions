#!/usr/bin/env python3
"""
Debug collision generation to understand why no collisions are detected
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def test_single_collision():
    """Test a single collision scenario to debug the physics"""

    # Create environment with minimal damping
    env = PointMass2D(dt=0.01, mass=1.0, damping=0.01)

    # Position very close to right wall with high velocity
    x, y = 4.8, 0.0  # Close to right wall at x=5.0
    vx, vy = 4.0, 0.0  # Fast rightward velocity

    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Environment bounds: x ∈ [-5.0, 5.0], y ∈ [-5.0, 5.0]")

    # Generate minimal force pattern
    trajectory_length = 101
    force_pattern = np.random.uniform(-0.1, 0.1, (trajectory_length, 2))

    # Simulate trajectory
    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0

    print("\nSimulating trajectory...")
    for t in range(min(20, trajectory_length - 1)):  # Just first 20 steps for debugging
        force = force_pattern[t]
        state, _, _ = env.step(force)
        trajectory.append(state.copy())

        # Check for velocity reversal (indicates collision)
        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
            position = trajectory[-1][:2]
            velocity = trajectory[-1][2:]

            print(f"Step {t+1}: pos=({position[0]:.3f}, {position[1]:.3f}), "
                  f"vel=({velocity[0]:.3f}, {velocity[1]:.3f}), "
                  f"vel_change=({vel_change[0]:.3f}, {vel_change[1]:.3f})")

            if np.any(vel_change > 1.0):  # Significant velocity change
                collision_count += 1
                print(f"   *** COLLISION DETECTED! Count = {collision_count} ***")

            # Check if particle is out of bounds
            if np.any(np.abs(position) > 5.0):
                print(f"   *** PARTICLE OUT OF BOUNDS! ***")
                break

    print(f"\nTotal collisions detected: {collision_count}")
    return collision_count

if __name__ == "__main__":
    test_single_collision()