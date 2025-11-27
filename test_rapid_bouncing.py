#!/usr/bin/env python3
"""
Test rapid bouncing by creating corridor scenarios and smaller timesteps
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def test_rapid_horizontal_bouncing():
    """Test rapid horizontal bouncing in a corridor-like scenario"""

    # Create environment with perfect elasticity
    env = PointMass2D(dt=0.005, damping=0.0)  # Smaller timestep for more resolution
    env.restitution = 1.0

    # Position for horizontal corridor bouncing
    x, y = 0.0, 0.0  # Center horizontally, center vertically
    vx, vy = 9.8, 0.1  # Almost pure horizontal velocity with tiny vertical

    print(f"Rapid horizontal bouncing test:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Physics: dt=0.005, damping=0.0, restitution=1.0")

    # Longer trajectory with smaller timesteps
    trajectory_length = 201  # 201 * 0.005 = 1.005 seconds
    force_pattern = np.zeros((trajectory_length, 2))

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
                    print(f"Step {t+1}: COLLISION #{collision_count} - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), "
                          f"vel=({velocity[0]:.3f}, {velocity[1]:.3f})")

            if np.any(np.abs(position) > 5.0):
                print(f"   *** OUT OF BOUNDS! ***")
                break

    collision_density = collision_count / len(trajectory) * 100
    print(f"Horizontal bouncing: {collision_count} collisions ({collision_density:.2f}% density)")
    return collision_count, collision_density

def test_confined_box_bouncing():
    """Test bouncing in a smaller effective area"""

    # Create environment with perfect elasticity
    env = PointMass2D(dt=0.008, damping=0.0)  # Different timestep
    env.restitution = 1.0

    # Position in smaller area with diagonal velocity
    x, y = 2.5, 1.5  # Offset from center to create asymmetric bouncing
    vx, vy = -8.0, 7.5  # Strong diagonal velocity

    print(f"\nConfined box bouncing test:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Physics: dt=0.008, damping=0.0, restitution=1.0")

    trajectory_length = 151  # 151 * 0.008 = ~1.2 seconds
    force_pattern = np.zeros((trajectory_length, 2))

    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0
    collision_times = []
    speed_history = []

    for t in range(trajectory_length - 1):
        force = force_pattern[t]
        state, _, _ = env.step(force)
        trajectory.append(state.copy())

        position = trajectory[-1][:2]
        velocity = trajectory[-1][2:]
        speed = np.linalg.norm(velocity)
        speed_history.append(speed)

        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])

            if np.any(vel_change > 1.0):
                collision_count += 1
                collision_times.append(t+1)
                wall_hit = "X-wall" if vel_change[0] > vel_change[1] else "Y-wall"
                if collision_count <= 25:
                    print(f"Step {t+1}: COLLISION #{collision_count} ({wall_hit}) - "
                          f"speed={speed:.3f}")

            if np.any(np.abs(position) > 5.0):
                print(f"   *** OUT OF BOUNDS! ***")
                break

    collision_density = collision_count / len(trajectory) * 100
    avg_speed = np.mean(speed_history)
    print(f"Confined box: {collision_count} collisions ({collision_density:.2f}% density)")
    print(f"Average speed maintained: {avg_speed:.3f}")
    print(f"Collision intervals: {np.diff(collision_times) if len(collision_times) > 1 else 'N/A'}")

    return collision_count, collision_density

if __name__ == "__main__":
    print("Testing rapid bouncing scenarios...\n")

    count1, density1 = test_rapid_horizontal_bouncing()
    count2, density2 = test_confined_box_bouncing()

    print(f"\n=== SUMMARY ===")
    print(f"Horizontal rapid: {count1} collisions ({density1:.2f}% density)")
    print(f"Confined box: {count2} collisions ({density2:.2f}% density)")
    print(f"Target: 5-20 collisions (3-15% density)")

    success = max(count1, count2) >= 5
    print(f"Success achieving 5+ collisions: {'✅ YES' if success else '❌ NO'}")