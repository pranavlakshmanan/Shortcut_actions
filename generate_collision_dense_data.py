#!/usr/bin/env python3
"""
Generate collision-DENSE dataset with 1 second trajectories (101 steps).
High density of collision moments throughout.
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))
from envs import PointMass2D

PARTICLE_RADIUS = 0.15
WALL = 1.0
DT = 0.01
TRAJ_LENGTH = 101  # 1 second

def generate_collision_dense_dataset(
    num_samples=10000,
    trajectory_length=TRAJ_LENGTH,
    seed=42
):
    np.random.seed(seed)
    
    env = PointMass2D(dt=DT, damping=0.0)
    env.restitution = 1.0
    env.pos_bounds = [-WALL, WALL]
    
    dataset = []
    
    # For 1-second trajectories, we want MANY collisions per trajectory
    scenarios = {
        'high_speed_bouncer': 0.35,    # Fast, many bounces
        'medium_bouncer': 0.30,        # Medium speed, several bounces
        'corner_chaos': 0.20,          # Corner bouncing (hits both walls)
        'smooth': 0.15,                # No collision (control)
    }
    
    for scenario_type, ratio in scenarios.items():
        n_samples = int(num_samples * ratio)
        print(f"\nGenerating {n_samples} '{scenario_type}' samples...")
        
        generated = 0
        attempts = 0
        max_attempts = n_samples * 20
        
        with tqdm(total=n_samples) as pbar:
            while generated < n_samples and attempts < max_attempts:
                attempts += 1
                
                if scenario_type == 'high_speed_bouncer':
                    # Start near wall with high speed - expect 5-10+ collisions
                    wall = np.random.choice(['left', 'right', 'top', 'bottom'])
                    dist = np.random.uniform(0.2, 0.5)
                    speed = np.random.uniform(6.0, 10.0)
                    min_collisions = 5
                    
                elif scenario_type == 'medium_bouncer':
                    # Medium speed - expect 3-6 collisions
                    wall = np.random.choice(['left', 'right', 'top', 'bottom'])
                    dist = np.random.uniform(0.3, 0.6)
                    speed = np.random.uniform(3.0, 6.0)
                    min_collisions = 3
                    
                elif scenario_type == 'corner_chaos':
                    # Start in corner region, diagonal velocity
                    wall = np.random.choice(['corner_tr', 'corner_tl', 'corner_br', 'corner_bl'])
                    dist = np.random.uniform(0.2, 0.5)
                    speed = np.random.uniform(5.0, 8.0)
                    min_collisions = 4
                    
                else:  # smooth
                    wall = 'center'
                    dist = 0.0
                    speed = np.random.uniform(0.1, 0.3)
                    min_collisions = 0
                
                # Set position and velocity
                if wall == 'right':
                    x = WALL - dist
                    y = np.random.uniform(-0.5, 0.5)
                    vx = speed
                    vy = np.random.uniform(-speed*0.5, speed*0.5)
                elif wall == 'left':
                    x = -WALL + dist
                    y = np.random.uniform(-0.5, 0.5)
                    vx = -speed
                    vy = np.random.uniform(-speed*0.5, speed*0.5)
                elif wall == 'top':
                    x = np.random.uniform(-0.5, 0.5)
                    y = WALL - dist
                    vx = np.random.uniform(-speed*0.5, speed*0.5)
                    vy = speed
                elif wall == 'bottom':
                    x = np.random.uniform(-0.5, 0.5)
                    y = -WALL + dist
                    vx = np.random.uniform(-speed*0.5, speed*0.5)
                    vy = -speed
                elif wall == 'corner_tr':
                    x = WALL - dist
                    y = WALL - dist
                    vx = speed * 0.707
                    vy = speed * 0.707
                elif wall == 'corner_tl':
                    x = -WALL + dist
                    y = WALL - dist
                    vx = -speed * 0.707
                    vy = speed * 0.707
                elif wall == 'corner_br':
                    x = WALL - dist
                    y = -WALL + dist
                    vx = speed * 0.707
                    vy = -speed * 0.707
                elif wall == 'corner_bl':
                    x = -WALL + dist
                    y = -WALL + dist
                    vx = -speed * 0.707
                    vy = -speed * 0.707
                else:  # center (smooth)
                    x = np.random.uniform(-0.3, 0.3)
                    y = np.random.uniform(-0.3, 0.3)
                    vx = speed * np.random.choice([-1, 1])
                    vy = speed * np.random.choice([-1, 1])
                
                # Small forces
                force_pattern = np.random.uniform(-0.1, 0.1, (trajectory_length, 2))
                
                # Simulate
                env.clear_particles()
                env.add_particle(x, y, vx, vy, mass=1.0, radius=PARTICLE_RADIUS)
                
                trajectory = [np.array([x, y, vx, vy])]
                collision_steps = []
                
                for t in range(trajectory_length - 1):
                    force = force_pattern[t]
                    state, _, _ = env.step(force)
                    trajectory.append(state.copy())
                    
                    # Detect collision
                    prev_v = trajectory[-2][2:]
                    curr_v = trajectory[-1][2:]
                    if (prev_v[0] * curr_v[0] < -0.1) or (prev_v[1] * curr_v[1] < -0.1):
                        collision_steps.append(t + 1)
                
                # Validate
                if scenario_type == 'smooth':
                    valid = len(collision_steps) == 0
                else:
                    valid = len(collision_steps) >= min_collisions
                
                if valid:
                    dataset.append({
                        'scenario': {
                            'initial_state': np.array([x, y, vx, vy]),
                            'force_pattern': force_pattern,
                            'scenario_type': scenario_type,
                            'wall': wall,
                            'collision_steps': collision_steps,
                            'collision_count': len(collision_steps)
                        },
                        'trajectory': np.array(trajectory)
                    })
                    generated += 1
                    pbar.update(1)
        
        if generated < n_samples:
            print(f"  Warning: only generated {generated}/{n_samples}")
    
    np.random.shuffle(dataset)
    return dataset


def analyze_dataset(dataset):
    total_steps = 0
    total_collisions = 0
    steps_near_collision = 0
    
    for sample in dataset:
        traj = sample['trajectory']
        coll_steps = sample['scenario'].get('collision_steps', [])
        
        total_steps += len(traj) - 1
        total_collisions += len(coll_steps)
        
        for t in range(len(traj) - 1):
            for cs in coll_steps:
                if abs((t+1) - cs) <= 3:
                    steps_near_collision += 1
                    break
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"Total trajectories: {len(dataset)}")
    print(f"Total timesteps: {total_steps}")
    print(f"Total collision events: {total_collisions}")
    print(f"Avg collisions per trajectory: {total_collisions/len(dataset):.1f}")
    print(f"Collision moments: {total_collisions} ({100*total_collisions/total_steps:.1f}%)")
    print(f"Steps near collision (±3): {steps_near_collision} ({100*steps_near_collision/total_steps:.1f}%)")
    
    scenario_counts = {}
    for sample in dataset:
        st = sample['scenario']['scenario_type']
        scenario_counts[st] = scenario_counts.get(st, 0) + 1
    
    print(f"\nScenario breakdown:")
    for st, count in sorted(scenario_counts.items()):
        print(f"  {st}: {count} ({100*count/len(dataset):.1f}%)")


def main():
    print("="*70)
    print("COLLISION-DENSE DATASET (1 second trajectories)")
    print("="*70)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    datasets = {
        'train': {'size': 10000, 'seed': 42},
        'val': {'size': 1000, 'seed': 123},
        'test': {'size': 1000, 'seed': 456}
    }
    
    for split_name, config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Generating {split_name} dataset...")
        print(f"{'='*60}")
        
        dataset = generate_collision_dense_dataset(
            num_samples=config['size'],
            trajectory_length=TRAJ_LENGTH,
            seed=config['seed']
        )
        
        analyze_dataset(dataset)
        
        filename = f"collision_{split_name}.pkl"
        filepath = data_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n💾 Saved to {filepath}")

    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
