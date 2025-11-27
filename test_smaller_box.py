#!/usr/bin/env python3
"""
Test smaller box (4x4) for high-frequency collision generation
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs import PointMass2D

def test_smaller_box():
    """Test collision frequency in 4x4 box vs original 10x10 box"""

    # Create environment with smaller box
    env = PointMass2D(dt=0.01, damping=0.0)
    env.restitution = 1.0
    env.pos_bounds = [-2.0, 2.0]  # 4x4 box instead of 10x10

    # Test scenario: diagonal bouncing in smaller box
    x, y = 1.5, 1.5  # Near top-right corner
    vx, vy = -6.0, -6.0  # High diagonal velocity toward center

    print(f"Testing smaller box (4x4) collision frequency:")
    print(f"Initial state: x={x}, y={y}, vx={vx}, vy={vy}")
    print(f"Box size: 4x4 units (vs original 10x10)")
    print(f"Expected crossing time: 4 units / 6 velocity ≈ 67 timesteps")

    # Simulate trajectory
    trajectory_length = 101
    force_pattern = np.zeros((trajectory_length, 2))

    env.clear_particles()
    env.add_particle(x, y, vx, vy, mass=1.0, radius=0.15)

    trajectory = [np.array([x, y, vx, vy])]
    collision_count = 0
    collision_times = []
    speed_history = []

    print("\nSimulating trajectory...")
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
                if collision_count <= 20:
                    print(f"Step {t+1}: COLLISION #{collision_count} ({wall_hit}) - "
                          f"pos=({position[0]:.3f}, {position[1]:.3f}), speed={speed:.3f}")

            if np.any(np.abs(position) > 2.1):  # Check bounds for smaller box
                print(f"   *** OUT OF BOUNDS! pos=({position[0]:.3f}, {position[1]:.3f}) ***")
                break

    collision_density = collision_count / len(trajectory) * 100
    avg_speed = np.mean(speed_history)

    print(f"\n🎯 SMALLER BOX RESULTS:")
    print(f"   Total collisions: {collision_count}")
    print(f"   Collision density: {collision_density:.2f}% of timesteps")
    print(f"   Average speed maintained: {avg_speed:.3f}")
    print(f"   Collision intervals: {np.diff(collision_times) if len(collision_times) > 1 else 'N/A'}")

    # Check target achievement
    target_achieved = collision_count >= 5 and collision_density >= 3.0
    print(f"   Target achieved (5+ collisions, 3%+ density): {'✅ YES' if target_achieved else '❌ NO'}")

    if len(collision_times) > 1:
        avg_interval = np.mean(np.diff(collision_times))
        print(f"   Average time between collisions: {avg_interval:.1f} timesteps")

    return collision_count, collision_density

def test_multiple_scenarios():
    """Test several different starting scenarios in smaller box"""

    scenarios = [
        {"name": "Corner diagonal", "x": 1.7, "y": 1.7, "vx": -7.0, "vy": -7.0},
        {"name": "Wall horizontal", "x": 1.8, "y": 0.0, "vx": -8.0, "vy": 2.0},
        {"name": "Center outward", "x": 0.0, "y": 0.0, "vx": 6.0, "vy": 5.0},
    ]

    print(f"\n" + "="*60)
    print("TESTING MULTIPLE SCENARIOS IN SMALLER BOX")
    print("="*60)

    results = []

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")

        env = PointMass2D(dt=0.01, damping=0.0)
        env.restitution = 1.0
        env.pos_bounds = [-2.0, 2.0]

        trajectory_length = 101
        force_pattern = np.zeros((trajectory_length, 2))

        env.clear_particles()
        env.add_particle(scenario['x'], scenario['y'], scenario['vx'], scenario['vy'], mass=1.0, radius=0.15)

        trajectory = [np.array([scenario['x'], scenario['y'], scenario['vx'], scenario['vy']])]
        collision_count = 0

        for t in range(trajectory_length - 1):
            force = force_pattern[t]
            state, _, _ = env.step(force)
            trajectory.append(state.copy())

            if t > 0:
                vel_change = np.abs(trajectory[-1][2:] - trajectory[-2][2:])
                if np.any(vel_change > 1.0):
                    collision_count += 1

            if np.any(np.abs(trajectory[-1][:2]) > 2.1):
                break

        collision_density = collision_count / len(trajectory) * 100
        results.append({"scenario": scenario['name'], "collisions": collision_count, "density": collision_density})

        print(f"   Collisions: {collision_count}, Density: {collision_density:.2f}%")

    print(f"\n🎯 SUMMARY:")
    for result in results:
        status = "✅" if result['collisions'] >= 5 else "❌"
        print(f"   {status} {result['scenario']}: {result['collisions']} collisions ({result['density']:.2f}%)")

    return results

if __name__ == "__main__":
    # Test single scenario
    collision_count, density = test_smaller_box()

    # Test multiple scenarios
    results = test_multiple_scenarios()

    # Overall assessment
    successful_scenarios = sum(1 for r in results if r['collisions'] >= 5)
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Successful scenarios (5+ collisions): {successful_scenarios}/{len(results)}")

    if successful_scenarios > 0:
        print(f"   ✅ SUCCESS: Smaller box generates high-frequency collisions!")
        print(f"   Ready for dataset generation with 5-15 collision target.")
    else:
        print(f"   ❌ Need further optimization for consistent 5+ collisions.")