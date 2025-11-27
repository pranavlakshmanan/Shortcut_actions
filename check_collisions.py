#!/usr/bin/env python3
"""
Check if collision training data actually contains collisions
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_collision_data():
    print("="*80)
    print("COLLISION DATASET ANALYSIS")
    print("="*80)
    
    # Check if collision data exists
    data_files = ['collision_train.pkl', 'collision_val.pkl', 'collision_test.pkl']
    
    for filename in data_files:
        filepath = Path('data') / filename
        
        if not filepath.exists():
            print(f"\n❌ {filename} not found!")
            continue
        
        print(f"\n📁 Analyzing {filename}...")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   Total samples: {len(data)}")
        
        # Analyze collision counts
        collision_counts = {}
        wall_violations = 0
        max_positions = []
        
        for sample in data:
            # Get collision count
            collisions = sample.get('actual_collisions', 0)
            collision_counts[collisions] = collision_counts.get(collisions, 0) + 1
            
            # Check trajectory for wall violations
            trajectory = sample['trajectory']
            
            # Check if any position exceeds wall boundaries (±5.0)
            max_x = np.max(np.abs(trajectory[:, 0]))  # max |x|
            max_y = np.max(np.abs(trajectory[:, 1]))  # max |y|
            max_pos = max(max_x, max_y)
            max_positions.append(max_pos)
            
            if max_pos > 5.0:
                wall_violations += 1
        
        # Print collision distribution
        print(f"\n   Collision Distribution:")
        total = len(data)
        for count in sorted(collision_counts.keys()):
            num = collision_counts[count]
            pct = num / total * 100
            print(f"      {count} collisions: {num:4d} ({pct:5.1f}%)")
        
        # Print wall boundary info
        print(f"\n   Wall Boundary Analysis:")
        print(f"      Expected boundary: ±5.0")
        print(f"      Max position reached: {np.max(max_positions):.3f}")
        print(f"      Samples exceeding ±5.0: {wall_violations} ({wall_violations/total*100:.1f}%)")
        
        if wall_violations > 0:
            print(f"      ⚠️  WARNING: Particles going through walls!")
        
        # Check a specific collision scenario
        collision_samples = [s for s in data if s.get('actual_collisions', 0) > 0]
        if collision_samples:
            print(f"\n   Example Collision Scenario:")
            example = collision_samples[0]
            traj = example['trajectory']
            
            print(f"      Scenario type: {example['scenario']['scenario_type']}")
            print(f"      Actual collisions: {example['actual_collisions']}")
            print(f"      Initial state: x={traj[0,0]:.2f}, y={traj[0,1]:.2f}, vx={traj[0,2]:.2f}, vy={traj[0,3]:.2f}")
            print(f"      Final state:   x={traj[-1,0]:.2f}, y={traj[-1,1]:.2f}, vx={traj[-1,2]:.2f}, vy={traj[-1,3]:.2f}")
            
            # Check if velocity actually reversed (sign of collision)
            vx_changed_sign = traj[0,2] * traj[-1,2] < 0
            vy_changed_sign = traj[0,3] * traj[-1,3] < 0
            
            if vx_changed_sign or vy_changed_sign:
                print(f"      ✓ Velocity reversed (collision detected)")
            else:
                print(f"      ✗ No velocity reversal (collision NOT visible)")
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")
    
    # Check training data specifically
    train_path = Path('data/collision_train.pkl')
    if train_path.exists():
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        
        collision_count = sum(1 for s in train_data if s.get('actual_collisions', 0) > 0)
        collision_pct = collision_count / len(train_data) * 100
        
        print(f"\nTraining Data Quality:")
        print(f"  Total samples: {len(train_data)}")
        print(f"  Samples with collisions: {collision_count} ({collision_pct:.1f}%)")
        print(f"  Target: 70% collision scenarios")
        
        if collision_pct < 60:
            print(f"\n  ❌ INSUFFICIENT COLLISION SCENARIOS")
            print(f"     Only {collision_pct:.1f}% have collisions, need 70%")
            print(f"     Models trained on mostly smooth physics!")
        elif collision_pct > 75:
            print(f"\n  ⚠️  TOO MANY COLLISION SCENARIOS")
            print(f"     {collision_pct:.1f}% have collisions, target was 70%")
        else:
            print(f"\n  ✓ Collision ratio looks good ({collision_pct:.1f}%)")
        
        # Check if collisions are actually visible in trajectories
        visible_collisions = 0
        for sample in train_data[:100]:  # Check first 100
            if sample.get('actual_collisions', 0) > 0:
                traj = sample['trajectory']
                # Check for velocity sign changes
                vx_changes = np.sum(np.diff(np.sign(traj[:, 2])) != 0)
                vy_changes = np.sum(np.diff(np.sign(traj[:, 3])) != 0)
                if vx_changes > 0 or vy_changes > 0:
                    visible_collisions += 1
        
        visible_pct = visible_collisions / 100 * 100
        print(f"\n  Collision Visibility (first 100 samples):")
        print(f"     Samples with velocity reversals: {visible_collisions}/100")
        
        if visible_collisions < 30:
            print(f"     ❌ COLLISIONS NOT VISIBLE IN DATA")
            print(f"        Trajectories don't show velocity reversals")
            print(f"        Wall collisions might not be properly simulated")
        else:
            print(f"     ✓ Collisions appear visible in trajectories")

if __name__ == "__main__":
    analyze_collision_data()