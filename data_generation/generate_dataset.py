#!/usr/bin/env python3
"""
Multi-Collision Dataset Generator

Integrates physics-informed scenario generation with training pipeline.
Creates stress-testing datasets with 70% multi-collision scenarios.
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from envs import PointMass2D
from data_generation.multi_collision_scenarios import (
    MultiCollisionScenarioGenerator,
    save_multi_collision_dataset,
    load_multi_collision_dataset
)

def main():
    parser = argparse.ArgumentParser(description="Generate multi-collision stress-test dataset")
    parser.add_argument("--num_trajectories", type=int, default=5000,
                       help="Total number of trajectories to generate")
    parser.add_argument("--collision_bias", type=float, default=0.7,
                       help="Fraction of data with collisions (0.7 = 70%)")
    parser.add_argument("--output_filename", type=str, default="multi_collision_stress_test.pkl",
                       help="Output filename for dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible generation")

    args = parser.parse_args()

    print("🎯 Multi-Collision Dataset Generation")
    print("="*50)
    print(f"Target trajectories: {args.num_trajectories:,}")
    print(f"Collision bias: {args.collision_bias*100:.0f}%")
    print(f"Random seed: {args.seed}")

    # Set random seed for reproducible generation
    np.random.seed(args.seed)

    # Create physics environment
    print("\n🔬 Initializing Physics Environment...")
    env = PointMass2D(
        dt=0.01,      # High resolution for accurate collision detection
        mass=1.0,
        damping=0.1
    )
    print(f"   Environment: dt={env.dt}, boundaries=±5.0, damping={env.damping}")

    # Initialize scenario generator
    print("\n🎲 Creating Multi-Collision Scenario Generator...")
    generator = MultiCollisionScenarioGenerator(env)

    print("   Scenario Distribution:")
    for scenario_type, ratio in generator.collision_distribution.items():
        count = int(args.num_trajectories * ratio * args.collision_bias) if scenario_type != 'smooth_motion' else int(args.num_trajectories * (1 - args.collision_bias))
        print(f"     {scenario_type:20s}: {count:4d} ({ratio*100 if scenario_type != 'smooth_motion' else (1-args.collision_bias)*100:4.1f}%)")

    print("   Energy Regimes: low, medium, high")
    print("   Force Patterns: 6 types (zero, constant, impulse, oscillating, opposing, collision-inducing)")

    # Generate complete dataset
    print(f"\n🚀 Generating {args.num_trajectories:,} Physics Trajectories...")
    dataset = generator.generate_dataset(
        total_trajectories=args.num_trajectories,
        collision_bias=args.collision_bias
    )

    # Analyze dataset quality
    print("\n📊 Dataset Quality Analysis:")
    analyze_dataset_quality(dataset)

    # Save dataset
    print(f"\n💾 Saving Dataset...")
    save_multi_collision_dataset(dataset, args.output_filename)

    # Generate summary statistics
    print(f"\n📈 Generation Summary:")
    print(f"   ✓ Total trajectories: {len(dataset):,}")
    print(f"   ✓ Multi-collision scenarios: {sum(1 for d in dataset if d['actual_collisions'] >= 1):,}")
    print(f"   ✓ Collision rate: {sum(1 for d in dataset if d['actual_collisions'] >= 1)/len(dataset)*100:.1f}%")

    # Validate collision complexity distribution
    collision_stats = {}
    for sample in dataset:
        collisions = sample['actual_collisions']
        collision_key = str(collisions) if collisions < 4 else '4+'
        collision_stats[collision_key] = collision_stats.get(collision_key, 0) + 1

    print("\n   Collision Complexity Breakdown:")
    for collision_count in ['0', '1', '2', '3', '4+']:
        count = collision_stats.get(collision_count, 0)
        print(f"     {collision_count} collisions: {count:4d} ({count/len(dataset)*100:4.1f}%)")

    print(f"\n✅ Multi-collision dataset generation completed!")
    print(f"📁 Dataset saved to: data/{args.output_filename}")

def analyze_dataset_quality(dataset):
    """Analyze quality metrics of generated dataset"""

    # Collision distribution analysis
    collision_counts = [sample['actual_collisions'] for sample in dataset]

    # Energy analysis across scenarios
    scenario_types = [sample['scenario']['scenario_type'] for sample in dataset]
    energy_levels = [sample['scenario']['energy_level'] for sample in dataset]

    # Trajectory length analysis
    traj_lengths = [sample['trajectory_length'] for sample in dataset]

    print(f"   Collision Statistics:")
    print(f"     Mean collisions per trajectory: {np.mean(collision_counts):.2f}")
    print(f"     Max collisions observed: {np.max(collision_counts)}")
    print(f"     Multi-collision rate: {sum(1 for c in collision_counts if c >= 2)/len(collision_counts)*100:.1f}%")

    print(f"   Energy Distribution:")
    energy_dist = {level: energy_levels.count(level) for level in set(energy_levels)}
    for energy, count in energy_dist.items():
        print(f"     {energy:6s} energy: {count:4d} ({count/len(energy_levels)*100:4.1f}%)")

    print(f"   Trajectory Quality:")
    print(f"     Mean length: {np.mean(traj_lengths):.1f} steps")
    print(f"     All trajectories complete: {all(length >= 100 for length in traj_lengths)}")

    # Stress-test coverage analysis
    stress_scenarios = ['triple_collision', 'quad_plus_collision']
    stress_count = sum(1 for s_type in scenario_types if s_type in stress_scenarios)
    print(f"   Stress-Test Coverage:")
    print(f"     High-complexity scenarios: {stress_count} ({stress_count/len(scenario_types)*100:.1f}%)")

    # Physics realism checks
    print(f"   Physics Validation:")

    # Check for reasonable velocity ranges
    all_velocities = []
    for sample in dataset:
        trajectory = sample['trajectory']
        velocities = trajectory[:, 2:4]  # vx, vy components
        all_velocities.extend(np.linalg.norm(velocities, axis=1))

    print(f"     Velocity range: [{np.min(all_velocities):.2f}, {np.max(all_velocities):.2f}]")
    print(f"     Mean speed: {np.mean(all_velocities):.2f}")

    # Check boundary compliance
    boundary_violations = 0
    for sample in dataset:
        trajectory = sample['trajectory']
        positions = trajectory[:, :2]  # x, y components
        if np.any(np.abs(positions) > 5.1):  # Small tolerance for boundary
            boundary_violations += 1

    print(f"     Boundary violations: {boundary_violations} ({boundary_violations/len(dataset)*100:.2f}%)")

if __name__ == "__main__":
    main()