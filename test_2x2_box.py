#!/usr/bin/env python3
"""
Test ultra-small box (2x2) for maximum collision frequency
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def test_2x2_box():
    """Test collision frequency in 2x2 box"""

    # Create environment with 2x2 box
    env = PointMass2D(dt=0.01, damping=0.0)
    env.restitution = 1.0
    env.pos_bounds = [-1.0, 1.0]  # 2x2 box

    # Test scenario: diagonal bouncing in 2x2 box
    x, y = 0.7, 0.7  # Near top-right corner
    vx, vy = -6.0, -6.0  # High diagonal velocity

    print(f"Testing 2x2 box collision frequency:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Box size: 2x2 units (vs original 10x10)")
    print(f"Expected crossing time: 2 units / 6 velocity ≈ 33 timesteps")

    # Simulate trajectory
    trajectory_length = 101
    force_pattern = np.zeros((trajectory_length, 2))

    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0
    collision_times = []
    position_history = []

    print("\nSimulating trajectory...")
    for t in range(trajectory_length - 1):
        force = force_pattern[t]
        state, _, _ = env.step(force)
        trajectory.append(state.copy())

        position = trajectory[-1][:2]
        velocity = trajectory[-1][2:]
        speed = np.linalg.norm(velocity)
        position_history.append(position.copy())

        if t > 0:
            vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])

            if np.any(vel_change > 1.0):
                collision_count += 1
                collision_times.append(t+1)
                wall_hit = "X-wall" if vel_change[0] > vel_change[1] else "Y-wall"
                if collision_count <= 25:
                    print(f"Step {t+1}: COLLISION #{collision_count} ({wall_hit}) - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), speed={speed:.3f}")

            if np.any(np.abs(position) > 1.05):  # Check bounds for 2x2 box
                print(f"   *** OUT OF BOUNDS! pos=({position[0]:.3f}, {position[1]:.3f}) ***")
                break

    collision_density = collision_count / len(trajectory) * 100

    print(f"\n🎯 2x2 BOX RESULTS:")
    print(f"   Total collisions: {collision_count}")
    print(f"   Collision density: {collision_density:.2f}% of timesteps")
    print(f"   Trajectory length: {len(trajectory)} timesteps")

    # Check target achievement
    target_achieved = collision_count >= 5 and collision_density >= 3.0
    print(f"   Target achieved (5+ collisions, 3%+ density): {'✅ YES' if target_achieved else '❌ NO'}")

    if len(collision_times) > 1:
        intervals = np.diff(collision_times)
        avg_interval = np.mean(intervals)
        print(f"   Collision intervals: {intervals}")
        print(f"   Average time between collisions: {avg_interval:.1f} timesteps")

    # Show trajectory bounds
    positions = np.array(position_history)
    x_range = [np.min(positions[:, 0]), np.max(positions[:, 0])]
    y_range = [np.min(positions[:, 1]), np.max(positions[:, 1])]
    print(f"   Position range: x=[{x_range[0]:.3f}, {x_range[1]:.3f}], y=[{y_range[0]:.3f}, {y_range[1]:.3f}]")

    return collision_count, collision_density

def test_extreme_velocity_2x2():
    """Test with extreme velocity in 2x2 box"""

    env = PointMass2D(dt=0.01, damping=0.0)
    env.restitution = 1.0
    env.pos_bounds = [-1.0, 1.0]

    # Start near wall with extreme velocity
    x, y = 0.9, 0.0  # Right edge
    vx, vy = -9.0, 3.0  # Very high velocity

    print(f"\n--- Extreme Velocity Test (2x2 Box) ---")
    print(f"Initial: x={x}, y={y}, vx={vx}, vy={vy}")

    trajectory_length = 101
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
            if np.any(vel_change > 1.0):
                collision_count += 1
                collision_times.append(t+1)

        if np.any(np.abs(trajectory[-1][:2]) > 1.05):
            break

    collision_density = collision_count / len(trajectory) * 100
    intervals = np.diff(collision_times) if len(collision_times) > 1 else []

    print(f"   Collisions: {collision_count}, Density: {collision_density:.2f}%")
    if len(intervals) > 0:
        print(f"   Intervals: {intervals}")
        print(f"   Average interval: {np.mean(intervals):.1f} timesteps")

    return collision_count, collision_density

if __name__ == "__main__":
    # Test basic 2x2 box
    count1, density1 = test_2x2_box()

    # Test extreme velocity
    count2, density2 = test_extreme_velocity_2x2()

    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Basic 2x2: {count1} collisions ({density1:.2f}%)")
    print(f"   Extreme velocity: {count2} collisions ({density2:.2f}%)")

    max_collisions = max(count1, count2)
    max_density = max(density1, density2)

    if max_collisions >= 5 and max_density >= 3.0:
        print(f"   ✅ SUCCESS: 2x2 box achieves target (5+ collisions, 3%+ density)!")
        print(f"   Ready for high-frequency collision dataset generation.")
    else:
        print(f"   ❌ Still below target. May need even smaller box or longer simulation.")