#!/usr/bin/env python3
"""
Collision Physics Evaluation Script

Evaluates Sequential vs Shortcut models on collision physics dataset.
Demonstrates clear performance differentiation where:
- Sequential Single-Step FAILS on collision scenarios
- Shortcut Single-Step WORKS on collision scenarios
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

from models.velocity_field import VelocityFieldNet
from models.shortcut_predictor import ShortcutPredictor
from envs.realistic_physics_2d import PointMass2D

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

class CollisionPhysicsEvaluator:
    """Evaluator for collision physics demonstrating shortcut value proposition"""

    def __init__(self, config_path="configs/collision_physics_training.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = PointMass2D(
            dt=self.config['environment']['dt'],
            mass=self.config['environment']['mass'],
            damping=self.config['environment']['damping']
        )

        self.output_dir = Path("experiments/collision_evaluation")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_models(self):
        """Load trained collision physics models"""
        models = {}

        # Sequential baseline
        seq_path = "experiments/sequential_baseline_model.pt"
        if Path(seq_path).exists():
            checkpoint = torch.load(seq_path, map_location=self.device)
            model = VelocityFieldNet(**self.config['model'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device).eval()
            models['sequential'] = model
            print(f"✓ Sequential loaded from {seq_path}")

        # Shortcut model
        shortcut_path = "experiments/shortcut_bootstrap_model.pt"
        if Path(shortcut_path).exists():
            checkpoint = torch.load(shortcut_path, map_location=self.device)
            velocity_net = VelocityFieldNet(**self.config['model'])
            shortcut_model = ShortcutPredictor(velocity_net)
            shortcut_model.load_state_dict(checkpoint['model_state_dict'])
            shortcut_model.to(self.device).eval()
            models['shortcut'] = shortcut_model
            print(f"✓ Shortcut loaded from {shortcut_path}")

        return models

    def load_collision_test_data(self):
        """Load collision physics test dataset"""

        test_path = Path('data/collision_test.pkl')
        with open(test_path, 'rb') as f:
            raw_data = pickle.load(f)

        print(f"✓ Loaded {len(raw_data)} collision test scenarios")

        # Convert to evaluation format
        test_scenarios = []
        for sample in raw_data:
            test_scenarios.append({
                'initial_state': sample['scenario']['initial_state'],
                'actions': sample['scenario']['force_pattern'],
                'trajectory': sample['trajectory'],
                'scenario_type': sample['scenario']['scenario_type'],
                'collision_count': sample['actual_collisions']
            })

        return test_scenarios

    def simulate_ground_truth(self, initial_state, actions, horizon):
        """Simulate ground truth physics"""
        self.env.clear_particles()
        x, y, vx, vy = initial_state
        self.env.add_particle(x, y, vx, vy, mass=self.env.mass)

        trajectory = [initial_state.copy()]
        num_steps = int(horizon / self.env.dt)

        for step in range(num_steps):
            action_idx = min(step, len(actions) - 1)
            action = actions[action_idx] if len(actions) > 0 else np.zeros(2)
            state, _, _ = self.env.step(action)
            trajectory.append(state.copy())

        return np.array(trajectory)

    def predict_sequential_multi_step(self, model, initial_state, actions, horizon):
        """Sequential multi-step (gold standard baseline)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor([initial_state]).to(self.device)
            actions_tensor = torch.FloatTensor([actions]).to(self.device)
            time_tensor = torch.zeros(1, 1).to(self.device)

            current_state = state_tensor.clone()
            current_time = time_tensor.clone()
            num_steps = int(horizon / 0.01)
            dt_tensor = torch.FloatTensor([[0.01]]).to(self.device)

            for _ in range(num_steps):
                velocity = model(current_state, actions_tensor, current_time, dt_tensor)
                current_state = current_state + velocity * dt_tensor
                current_time = current_time + dt_tensor

            return current_state.detach().cpu().numpy()[0]

    def predict_sequential_single_step(self, model, initial_state, actions, horizon):
        """Sequential single-step (FORCED out-of-distribution, should FAIL on collisions)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor([initial_state]).to(self.device)
            actions_tensor = torch.FloatTensor([actions]).to(self.device)
            time_tensor = torch.zeros(1, 1).to(self.device)
            step_tensor = torch.FloatTensor([[horizon]]).to(self.device)

            velocity = model(state_tensor, actions_tensor, time_tensor, step_tensor)
            final_state = state_tensor + velocity * step_tensor
            return final_state.detach().cpu().numpy()[0]

    def predict_shortcut_single_step(self, model, initial_state, actions, horizon):
        """Shortcut single-step (DESIGNED for this, should WORK on collisions)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor([initial_state]).to(self.device)
            actions_tensor = torch.FloatTensor([actions]).to(self.device)
            time_tensor = torch.zeros(1, 1).to(self.device)
            step_tensor = torch.FloatTensor([[horizon]]).to(self.device)

            model_output = model.velocity_net(state_tensor, actions_tensor, time_tensor, step_tensor)

            current_velocity = state_tensor[:, 2:4]
            position_displacement = current_velocity * step_tensor
            velocity_change = model_output[:, 2:4] * step_tensor

            new_position = state_tensor[:, :2] + position_displacement
            new_velocity = state_tensor[:, 2:4] + velocity_change

            final_state = torch.cat([new_position, new_velocity], dim=1)
            return final_state.detach().cpu().numpy()[0]

    def evaluate_all_scenarios(self, models, test_scenarios, horizons=[1.0, 2.0, 3.0]):
        """Evaluate all methods on collision test scenarios"""

        results = {
            'horizons': horizons,
            'by_collision_count': {0: [], 1: [], 2: []},
            'overall': []
        }

        print(f"\n🧪 Evaluating {len(test_scenarios)} scenarios at horizons {horizons}")

        for scenario_idx, scenario in enumerate(tqdm(test_scenarios, desc="Evaluating scenarios")):
            initial_state = scenario['initial_state']
            actions = scenario['actions']
            collision_count = scenario['collision_count']

            for horizon in horizons:
                # Ground truth
                gt_traj = self.simulate_ground_truth(initial_state, actions, horizon)
                gt_final = gt_traj[-1]

                # All three methods
                seq_multi = self.predict_sequential_multi_step(models['sequential'], initial_state, actions, horizon)
                seq_single = self.predict_sequential_single_step(models['sequential'], initial_state, actions, horizon)
                sc_single = self.predict_shortcut_single_step(models['shortcut'], initial_state, actions, horizon)

                # Calculate errors (position only)
                seq_multi_error = np.linalg.norm(seq_multi[:2] - gt_final[:2])
                seq_single_error = np.linalg.norm(seq_single[:2] - gt_final[:2])
                sc_single_error = np.linalg.norm(sc_single[:2] - gt_final[:2])

                result = {
                    'scenario_idx': scenario_idx,
                    'collision_count': collision_count,
                    'horizon': horizon,
                    'gt_final': gt_final,
                    'seq_multi_error': seq_multi_error,
                    'seq_single_error': seq_single_error,
                    'sc_single_error': sc_single_error,
                    'seq_single_ratio': seq_single_error / seq_multi_error if seq_multi_error > 0 else float('inf'),
                    'sc_single_ratio': sc_single_error / seq_multi_error if seq_multi_error > 0 else float('inf')
                }

                results['overall'].append(result)

                # Bucket by collision count
                bucket = min(collision_count, 2)  # 0, 1, 2+
                results['by_collision_count'][bucket].append(result)

        return results

    def analyze_results(self, results):
        """Analyze and print collision physics results"""

        print(f"\n{'='*80}")
        print("COLLISION PHYSICS EVALUATION RESULTS")
        print(f"{'='*80}")

        # Overall analysis
        overall = results['overall']
        seq_multi_errors = [r['seq_multi_error'] for r in overall]
        seq_single_errors = [r['seq_single_error'] for r in overall]
        sc_single_errors = [r['sc_single_error'] for r in overall]

        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Sequential Multi-Step:  {np.mean(seq_multi_errors):.6f} ± {np.std(seq_multi_errors):.6f}")
        print(f"  Sequential Single-Step: {np.mean(seq_single_errors):.6f} ± {np.std(seq_single_errors):.6f}")
        print(f"  Shortcut Single-Step:   {np.mean(sc_single_errors):.6f} ± {np.std(sc_single_errors):.6f}")
        print(f"\n  Sequential Single-Step Ratio: {np.mean(seq_single_errors)/np.mean(seq_multi_errors):.2f}x")
        print(f"  Shortcut Single-Step Ratio:   {np.mean(sc_single_errors)/np.mean(seq_multi_errors):.2f}x")

        # Analysis by collision complexity
        print(f"\nBY COLLISION COMPLEXITY:")
        for collision_count, bucket_results in results['by_collision_count'].items():
            if not bucket_results:
                continue

            bucket_seq_multi = [r['seq_multi_error'] for r in bucket_results]
            bucket_seq_single = [r['seq_single_error'] for r in bucket_results]
            bucket_sc_single = [r['sc_single_error'] for r in bucket_results]

            collision_label = f"{collision_count} collision{'s' if collision_count != 1 else ''}"
            if collision_count == 2:
                collision_label = "2+ collisions"

            print(f"\n  {collision_label.upper()} ({len(bucket_results)} samples):")
            print(f"    Sequential Multi-Step:  {np.mean(bucket_seq_multi):.6f}")
            print(f"    Sequential Single-Step: {np.mean(bucket_seq_single):.6f} ({np.mean(bucket_seq_single)/np.mean(bucket_seq_multi):.1f}x)")
            print(f"    Shortcut Single-Step:   {np.mean(bucket_sc_single):.6f} ({np.mean(bucket_sc_single)/np.mean(bucket_seq_multi):.1f}x)")

        # Success criteria check
        print(f"\n{'='*80}")
        print("SUCCESS CRITERIA CHECK:")
        print(f"{'='*80}")

        overall_seq_single_ratio = np.mean(seq_single_errors) / np.mean(seq_multi_errors)
        overall_sc_single_ratio = np.mean(sc_single_errors) / np.mean(seq_multi_errors)

        criteria = [
            ("Sequential Single-Step fails (>5x worse)", overall_seq_single_ratio > 5.0, f"{overall_seq_single_ratio:.1f}x"),
            ("Shortcut Single-Step works (<3x worse)", overall_sc_single_ratio < 3.0, f"{overall_sc_single_ratio:.1f}x"),
            ("Clear differentiation achieved", overall_seq_single_ratio > 2 * overall_sc_single_ratio, "Check"),
        ]

        for criterion, passed, value in criteria:
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}: {value}")

        return results

    def create_visualizations(self, results):
        """Create collision physics evaluation plots"""

        print(f"\n📊 Creating visualizations...")

        # Main comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Collision Physics Evaluation Results', fontsize=16, fontweight='bold')

        # Plot 1: Error by collision count
        ax = axes[0, 0]
        collision_counts = [0, 1, 2]
        collision_labels = ['0 collisions', '1 collision', '2+ collisions']

        seq_multi_means = []
        seq_single_means = []
        sc_single_means = []

        for count in collision_counts:
            bucket = results['by_collision_count'][count]
            if bucket:
                seq_multi_means.append(np.mean([r['seq_multi_error'] for r in bucket]))
                seq_single_means.append(np.mean([r['seq_single_error'] for r in bucket]))
                sc_single_means.append(np.mean([r['sc_single_error'] for r in bucket]))
            else:
                seq_multi_means.append(0)
                seq_single_means.append(0)
                sc_single_means.append(0)

        x = np.arange(len(collision_labels))
        width = 0.25

        ax.bar(x - width, seq_multi_means, width, label='Sequential Multi-Step', alpha=0.8)
        ax.bar(x, seq_single_means, width, label='Sequential Single-Step', alpha=0.8)
        ax.bar(x + width, sc_single_means, width, label='Shortcut Single-Step', alpha=0.8)

        ax.set_xlabel('Collision Complexity')
        ax.set_ylabel('Mean Position Error')
        ax.set_title('Performance by Collision Count')
        ax.set_xticks(x)
        ax.set_xticklabels(collision_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error ratios
        ax = axes[0, 1]
        seq_ratios = [seq_single_means[i]/seq_multi_means[i] if seq_multi_means[i] > 0 else 0 for i in range(len(collision_counts))]
        sc_ratios = [sc_single_means[i]/seq_multi_means[i] if seq_multi_means[i] > 0 else 0 for i in range(len(collision_counts))]

        ax.bar(x - width/2, seq_ratios, width, label='Sequential Single-Step', alpha=0.8)
        ax.bar(x + width/2, sc_ratios, width, label='Shortcut Single-Step', alpha=0.8)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
        ax.set_xlabel('Collision Complexity')
        ax.set_ylabel('Error Ratio vs Multi-Step')
        ax.set_title('Relative Performance (Lower = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(collision_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Error distribution
        ax = axes[1, 0]
        all_seq_single = [r['seq_single_error'] for r in results['overall']]
        all_sc_single = [r['sc_single_error'] for r in results['overall']]

        ax.hist(all_seq_single, bins=30, alpha=0.7, label='Sequential Single-Step', density=True)
        ax.hist(all_sc_single, bins=30, alpha=0.7, label='Shortcut Single-Step', density=True)
        ax.set_xlabel('Position Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution (All Scenarios)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Success rate by horizon
        ax = axes[1, 1]
        horizons = results['horizons']

        seq_success_rates = []
        sc_success_rates = []

        for h in horizons:
            horizon_results = [r for r in results['overall'] if r['horizon'] == h]
            seq_successes = sum(1 for r in horizon_results if r['seq_single_error'] < 2 * r['seq_multi_error'])
            sc_successes = sum(1 for r in horizon_results if r['sc_single_error'] < 2 * r['seq_multi_error'])

            seq_success_rates.append(seq_successes / len(horizon_results) * 100 if horizon_results else 0)
            sc_success_rates.append(sc_successes / len(horizon_results) * 100 if horizon_results else 0)

        x = np.arange(len(horizons))
        ax.bar(x - width/2, seq_success_rates, width, label='Sequential Single-Step', alpha=0.8)
        ax.bar(x + width/2, sc_success_rates, width, label='Shortcut Single-Step', alpha=0.8)
        ax.set_xlabel('Prediction Horizon (s)')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Horizon')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h:.1f}s' for h in horizons])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'collision_physics_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir}/collision_physics_evaluation.png")

    def run_full_evaluation(self):
        """Run complete collision physics evaluation"""

        print("=" * 80)
        print("COLLISION PHYSICS EVALUATION")
        print("=" * 80)

        # Load models
        models = self.load_models()
        if len(models) < 2:
            print("❌ Need both Sequential and Shortcut models trained!")
            print("   Run: python train_collision_models.py")
            return False

        # Load collision test data
        test_scenarios = self.load_collision_test_data()

        # Run evaluation
        results = self.evaluate_all_scenarios(models, test_scenarios)

        # Analyze results
        analyzed_results = self.analyze_results(results)

        # Create visualizations
        self.create_visualizations(results)

        # Save results
        import json

        def convert_numpy(obj):
            """Convert numpy objects for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        with open(self.output_dir / 'collision_evaluation_results.json', 'w') as f:
            json_results = convert_numpy(results)
            json.dump(json_results, f, indent=2)

        print(f"\n  ✓ Results saved to: {self.output_dir}/collision_evaluation_results.json")

        print(f"\n{'='*80}")
        print("✅ COLLISION PHYSICS EVALUATION COMPLETE!")
        print(f"{'='*80}")

        return True

def main():
    evaluator = CollisionPhysicsEvaluator()
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()