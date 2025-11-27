#!/usr/bin/env python3
"""
Comprehensive Ground Truth Ball Trajectory Testing
==================================================

Tests the improved shortcut model against actual physics simulations of ball trajectories
at multiple time horizons to validate the effectiveness of our fixes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import time
from typing import Dict, List, Tuple

from models.velocity_field import VelocityFieldNet
from models.shortcut_predictor import ShortcutPredictor
from envs.realistic_physics_2d import PointMass2D


class GroundTruthTrajectoryTester:
    """Comprehensive testing against ground truth physics simulations"""

    def __init__(self, config_path="configs/two_network_comparison.yaml"):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = self.create_environment()

        # Test scenarios
        self.test_horizons = [0.05, 0.1, 0.2, 0.5, 1.0]  # Time horizons to test
        self.num_trajectories = 50  # Number of test trajectories per horizon

        print(f"🎯 Ground Truth Trajectory Tester Initialized")
        print(f"   Device: {self.device}")
        print(f"   Test horizons: {self.test_horizons}")
        print(f"   Trajectories per horizon: {self.num_trajectories}")

    def load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def create_environment(self):
        """Create physics environment for ground truth simulations"""
        return PointMass2D(
            dt=self.config['environment']['dt'],
            mass=self.config['environment']['mass'],
            damping=self.config['environment']['damping']
        )

    def load_trained_models(self):
        """Load the trained sequential and shortcut models"""
        models = {}

        # Load Sequential Model
        seq_path = "experiments/sequential_baseline_model.pt"
        if Path(seq_path).exists():
            checkpoint = torch.load(seq_path, map_location=self.device)
            model = VelocityFieldNet(**self.config['model'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device).eval()
            models['sequential'] = {
                'model': model,
                'val_loss': checkpoint['val_loss'],
                'type': 'sequential_baseline'
            }
            print(f"   ✓ Sequential model loaded (val_loss: {checkpoint['val_loss']:.4f})")

        # Load Shortcut Model
        shortcut_path = "experiments/shortcut_bootstrap_model.pt"
        if Path(shortcut_path).exists():
            checkpoint = torch.load(shortcut_path, map_location=self.device)
            velocity_net = VelocityFieldNet(**self.config['model'])
            model = ShortcutPredictor(velocity_net)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device).eval()
            models['shortcut'] = {
                'model': model,
                'val_loss': checkpoint['val_loss'],
                'type': 'shortcut_improved'
            }
            print(f"   ✓ Shortcut model loaded (val_loss: {checkpoint['val_loss']:.4f})")

        return models

    def generate_test_scenarios(self):
        """Generate diverse test scenarios for comprehensive evaluation"""
        scenarios = []

        for i in range(self.num_trajectories):
            # Diverse initial conditions
            scenarios.append({
                'name': f'trajectory_{i:03d}',
                'initial_state': np.array([
                    np.random.uniform(-2, 2),      # x position
                    np.random.uniform(-2, 2),      # y position
                    np.random.uniform(-1, 1),      # x velocity
                    np.random.uniform(-1, 1)       # y velocity
                ]),
                'actions': np.random.uniform(-0.3, 0.3, (20, 2))  # Control inputs
            })

        return scenarios

    def simulate_ground_truth(self, scenario: Dict, horizon: float) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Simulate ground truth trajectory using physics environment"""
        self.env.clear_particles()

        # Set initial state
        x, y, vx, vy = scenario['initial_state']
        self.env.add_particle(x, y, vx, vy, mass=self.env.mass)

        # Simulate trajectory
        trajectory = [scenario['initial_state'].copy()]
        current_state = scenario['initial_state'].copy()

        # Number of simulation steps
        dt = self.env.dt
        num_steps = int(horizon / dt)

        for step in range(num_steps):
            # Get action for this step
            action_idx = min(step, len(scenario['actions']) - 1)
            action = scenario['actions'][action_idx] if len(scenario['actions']) > 0 else np.zeros(2)

            # Physics step
            current_state, _, _ = self.env.step(action)
            trajectory.append(current_state.copy())

        final_state = trajectory[-1]
        return final_state, trajectory

    def predict_with_model(self, model_info: Dict, scenario: Dict, horizon: float) -> np.ndarray:
        """Predict final state using neural network model"""
        model = model_info['model']
        model_type = model_info['type']

        with torch.no_grad():
            # Prepare inputs
            initial_state = torch.FloatTensor(scenario['initial_state']).unsqueeze(0).to(self.device)
            actions = torch.FloatTensor(scenario['actions']).unsqueeze(0).to(self.device)
            time = torch.zeros(1, 1).to(self.device)
            step_size = torch.FloatTensor([[horizon]]).to(self.device)

            if model_type == 'sequential_baseline':
                # Sequential model: iterative rollout for horizons > 0.01
                if horizon <= 0.01:
                    velocity = model(initial_state, actions, time, step_size)
                    final_state = initial_state + velocity * step_size
                else:
                    # Iterative rollout
                    current_state = initial_state.clone()
                    num_steps = int(horizon / 0.01)
                    dt_tensor = torch.FloatTensor([[0.01]]).to(self.device)

                    for _ in range(num_steps):
                        velocity = model(current_state, actions, time, dt_tensor)
                        current_state = current_state + velocity * dt_tensor

                    final_state = current_state

            elif model_type == 'shortcut_improved':
                # Shortcut model: direct prediction at any horizon
                velocity = model.velocity_net(initial_state, actions, time, step_size)
                final_state = initial_state + velocity * step_size

            return final_state.cpu().numpy()[0]

    def run_trajectory_comparison(self, models: Dict) -> Dict:
        """Run comprehensive trajectory comparison test"""
        print(f"\n🚀 Running Ground Truth Trajectory Comparison")
        print("=" * 70)

        results = {
            'horizons': self.test_horizons,
            'scenarios': self.generate_test_scenarios(),
            'model_results': {},
            'summary_stats': {}
        }

        for model_name, model_info in models.items():
            print(f"\n📊 Testing {model_name.upper()} model...")
            model_results = {
                'errors_by_horizon': {},
                'computation_times': {},
                'trajectories': {}  # Store sample trajectories for visualization
            }

            for horizon in self.test_horizons:
                print(f"   Testing horizon: {horizon:.2f}s")

                errors = []
                times = []
                sample_trajectories = []

                for i, scenario in enumerate(results['scenarios']):
                    # Ground truth simulation
                    gt_final_state, gt_trajectory = self.simulate_ground_truth(scenario, horizon)

                    # Model prediction with timing
                    start_time = time.time()
                    pred_final_state = self.predict_with_model(model_info, scenario, horizon)
                    pred_time = time.time() - start_time

                    # Calculate error
                    error = np.linalg.norm(pred_final_state - gt_final_state)
                    errors.append(error)
                    times.append(pred_time)

                    # Store sample trajectories for visualization (first 5)
                    if i < 5:
                        sample_trajectories.append({
                            'ground_truth': gt_trajectory,
                            'prediction': {
                                'initial': scenario['initial_state'],
                                'final': pred_final_state,
                                'horizon': horizon
                            },
                            'error': error
                        })

                model_results['errors_by_horizon'][horizon] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'min': np.min(errors),
                    'max': np.max(errors),
                    'median': np.median(errors)
                }

                model_results['computation_times'][horizon] = {
                    'mean': np.mean(times),
                    'std': np.std(times)
                }

                model_results['trajectories'][horizon] = sample_trajectories

                print(f"      Error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
                print(f"      Time:  {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")

            results['model_results'][model_name] = model_results

        # Generate summary statistics
        results['summary_stats'] = self.compute_summary_statistics(results)

        return results

    def compute_summary_statistics(self, results: Dict) -> Dict:
        """Compute comprehensive summary statistics"""
        summary = {}

        for model_name, model_results in results['model_results'].items():
            model_summary = {
                'overall_accuracy': {},
                'temporal_scaling': {},
                'computational_efficiency': {}
            }

            # Overall accuracy metrics
            all_errors = []
            all_times = []

            for horizon in self.test_horizons:
                errors = model_results['errors_by_horizon'][horizon]
                times = model_results['computation_times'][horizon]

                all_errors.extend([errors['mean']] * self.num_trajectories)
                all_times.extend([times['mean']] * self.num_trajectories)

            model_summary['overall_accuracy'] = {
                'mean_error': np.mean(all_errors),
                'error_std': np.std(all_errors),
                'accuracy_score': 1.0 / (1.0 + np.mean(all_errors))  # Higher is better
            }

            # Temporal scaling analysis
            horizon_errors = [model_results['errors_by_horizon'][h]['mean'] for h in self.test_horizons]
            model_summary['temporal_scaling'] = {
                'error_growth_rate': np.polyfit(self.test_horizons, horizon_errors, 1)[0],
                'short_horizon_accuracy': np.mean(horizon_errors[:2]),  # 0.05s, 0.1s
                'long_horizon_accuracy': np.mean(horizon_errors[-2:])   # 0.5s, 1.0s
            }

            # Computational efficiency
            avg_time = np.mean(all_times)
            model_summary['computational_efficiency'] = {
                'avg_prediction_time': avg_time,
                'predictions_per_second': 1.0 / avg_time if avg_time > 0 else float('inf'),
                'efficiency_score': model_summary['overall_accuracy']['accuracy_score'] / avg_time if avg_time > 0 else 0
            }

            summary[model_name] = model_summary

        return summary

    def generate_comparison_report(self, results: Dict) -> str:
        """Generate detailed comparison report"""
        lines = [
            "🎯 GROUND TRUTH TRAJECTORY COMPARISON REPORT",
            "=" * 80,
            "",
            f"Test Configuration:",
            f"  • Horizons tested: {self.test_horizons}",
            f"  • Trajectories per horizon: {self.num_trajectories}",
            f"  • Total test cases: {len(self.test_horizons) * self.num_trajectories}",
            f"  • Physics environment: PointMass2D (dt={self.env.dt}, mass={self.env.mass}, damping={self.env.damping})",
            "",
            "📊 DETAILED RESULTS BY MODEL:"
        ]

        for model_name, model_results in results['model_results'].items():
            model_info = results['summary_stats'][model_name]

            lines.extend([
                "",
                f"🤖 {model_name.upper()} MODEL:",
                f"   Overall Accuracy Score: {model_info['overall_accuracy']['accuracy_score']:.4f}",
                f"   Mean Prediction Error: {model_info['overall_accuracy']['mean_error']:.4f} ± {model_info['overall_accuracy']['error_std']:.4f}",
                f"   Avg Prediction Time: {model_info['computational_efficiency']['avg_prediction_time']*1000:.2f} ms",
                f"   Efficiency Score: {model_info['computational_efficiency']['efficiency_score']:.2f}",
                "",
                f"   Error by Time Horizon:"
            ])

            for horizon in self.test_horizons:
                error_stats = model_results['errors_by_horizon'][horizon]
                time_stats = model_results['computation_times'][horizon]

                lines.append(
                    f"     {horizon:4.2f}s: {error_stats['mean']:.4f} ± {error_stats['std']:.4f} "
                    f"(range: {error_stats['min']:.4f} - {error_stats['max']:.4f}) "
                    f"[{time_stats['mean']*1000:.1f}ms]"
                )

        # Comparative analysis
        if len(results['model_results']) > 1:
            lines.extend([
                "",
                "⚖️  COMPARATIVE ANALYSIS:",
                ""
            ])

            model_names = list(results['model_results'].keys())
            if 'sequential' in model_names and 'shortcut' in model_names:
                seq_stats = results['summary_stats']['sequential']
                sc_stats = results['summary_stats']['shortcut']

                accuracy_improvement = sc_stats['overall_accuracy']['accuracy_score'] / seq_stats['overall_accuracy']['accuracy_score']
                speed_improvement = seq_stats['computational_efficiency']['avg_prediction_time'] / sc_stats['computational_efficiency']['avg_prediction_time']

                lines.extend([
                    f"   Shortcut vs Sequential Performance:",
                    f"     Accuracy: {accuracy_improvement:.2f}x {'better' if accuracy_improvement > 1 else 'worse'}",
                    f"     Speed: {speed_improvement:.2f}x {'faster' if speed_improvement > 1 else 'slower'}",
                    f"     Overall Efficiency: {sc_stats['computational_efficiency']['efficiency_score'] / seq_stats['computational_efficiency']['efficiency_score']:.2f}x",
                    "",
                    f"   Temporal Scaling Analysis:",
                    f"     Sequential error growth rate: {seq_stats['temporal_scaling']['error_growth_rate']:.4f}/s",
                    f"     Shortcut error growth rate: {sc_stats['temporal_scaling']['error_growth_rate']:.4f}/s",
                ])

        lines.extend([
            "",
            "🎯 CONCLUSION:",
            ""
        ])

        # Determine best performing model
        best_model = max(results['summary_stats'].keys(),
                        key=lambda k: results['summary_stats'][k]['overall_accuracy']['accuracy_score'])

        lines.extend([
            f"   Best Overall Performance: {best_model.upper()} model",
            f"   Key Findings:",
        ])

        if 'shortcut' in results['model_results']:
            shortcut_stats = results['summary_stats']['shortcut']
            lines.extend([
                f"     • Shortcut model accuracy score: {shortcut_stats['overall_accuracy']['accuracy_score']:.4f}",
                f"     • Avg prediction time: {shortcut_stats['computational_efficiency']['avg_prediction_time']*1000:.2f} ms",
                f"     • Successfully handles multi-scale temporal prediction: {'✅' if shortcut_stats['temporal_scaling']['error_growth_rate'] < 1.0 else '⚠️'}"
            ])

        report_content = "\n".join(lines)

        # Save report
        report_path = Path("experiments/ground_truth_trajectory_report.txt")
        report_path.parent.mkdir(exist_ok=True, parents=True)
        with open(report_path, 'w') as f:
            f.write(report_content)

        return report_content

    def create_trajectory_visualizations(self, results: Dict):
        """Create trajectory visualization plots"""
        print(f"\n📈 Generating trajectory visualizations...")

        # Create plots directory
        plots_dir = Path("experiments/trajectory_plots")
        plots_dir.mkdir(exist_ok=True, parents=True)

        # Plot 1: Error vs Horizon for all models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for model_name, model_results in results['model_results'].items():
            horizons = self.test_horizons
            mean_errors = [model_results['errors_by_horizon'][h]['mean'] for h in horizons]
            error_stds = [model_results['errors_by_horizon'][h]['std'] for h in horizons]

            ax1.errorbar(horizons, mean_errors, yerr=error_stds,
                        label=f'{model_name.title()} Model', marker='o', capsize=5)

        ax1.set_xlabel('Time Horizon (s)')
        ax1.set_ylabel('Prediction Error')
        ax1.set_title('Prediction Error vs Time Horizon')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Computation Time vs Horizon
        for model_name, model_results in results['model_results'].items():
            times = [model_results['computation_times'][h]['mean']*1000 for h in horizons]  # Convert to ms
            time_stds = [model_results['computation_times'][h]['std']*1000 for h in horizons]

            ax2.errorbar(horizons, times, yerr=time_stds,
                        label=f'{model_name.title()} Model', marker='s', capsize=5)

        ax2.set_xlabel('Time Horizon (s)')
        ax2.set_ylabel('Computation Time (ms)')
        ax2.set_title('Computation Time vs Time Horizon')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Sample trajectory comparisons
        for model_name, model_results in results['model_results'].items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{model_name.title()} Model - Sample Trajectory Predictions')

            for i, horizon in enumerate(self.test_horizons):
                if i >= 5:  # Only plot first 5 horizons
                    break

                ax = axes[i//3, i%3]

                # Plot sample trajectory for this horizon
                if horizon in model_results['trajectories'] and model_results['trajectories'][horizon]:
                    sample = model_results['trajectories'][horizon][0]  # First sample

                    # Ground truth trajectory
                    gt_traj = np.array(sample['ground_truth'])
                    ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=2, label='Ground Truth')
                    ax.plot(gt_traj[0, 0], gt_traj[0, 1], 'go', markersize=8, label='Start')
                    ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'gs', markersize=8, label='GT End')

                    # Model prediction
                    pred_final = sample['prediction']['final']
                    ax.plot([gt_traj[0, 0], pred_final[0]], [gt_traj[0, 1], pred_final[1]],
                           'r--', linewidth=2, alpha=0.7, label='Prediction')
                    ax.plot(pred_final[0], pred_final[1], 'rs', markersize=8, label='Pred End')

                    ax.set_title(f'Horizon: {horizon:.2f}s (Error: {sample["error"]:.4f})')
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.axis('equal')

            plt.tight_layout()
            plt.savefig(plots_dir / f'{model_name}_trajectories.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"   ✓ Visualizations saved to {plots_dir}/")


def main():
    """Main testing function"""
    print("🎯 GROUND TRUTH BALL TRAJECTORY TESTING")
    print("=" * 80)
    print("Testing improved shortcut model against real physics simulations...")

    # Initialize tester
    tester = GroundTruthTrajectoryTester()

    # Load trained models
    print(f"\n🔧 Loading trained models...")
    models = tester.load_trained_models()

    if not models:
        print("❌ No trained models found! Please ensure training has completed.")
        return False

    # Run comprehensive trajectory comparison
    results = tester.run_trajectory_comparison(models)

    # Generate detailed report
    print(f"\n📋 Generating comprehensive report...")
    report = tester.generate_comparison_report(results)
    print(report)

    # Create visualizations
    tester.create_trajectory_visualizations(results)

    # Final summary
    print(f"\n✅ Ground truth trajectory testing completed!")
    print(f"   📊 Report saved: experiments/ground_truth_trajectory_report.txt")
    print(f"   📈 Plots saved: experiments/trajectory_plots/")

    # Highlight key findings
    if 'shortcut' in results['summary_stats']:
        shortcut_stats = results['summary_stats']['shortcut']
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   • Shortcut model accuracy: {shortcut_stats['overall_accuracy']['accuracy_score']:.4f}")
        print(f"   • Mean prediction error: {shortcut_stats['overall_accuracy']['mean_error']:.4f}")
        print(f"   • Avg computation time: {shortcut_stats['computational_efficiency']['avg_prediction_time']*1000:.2f} ms")

        if shortcut_stats['overall_accuracy']['accuracy_score'] > 0.8:
            print(f"   ✅ EXCELLENT: Model shows strong performance across all time horizons!")
        elif shortcut_stats['overall_accuracy']['accuracy_score'] > 0.6:
            print(f"   ✅ GOOD: Model demonstrates reliable trajectory prediction capabilities")
        else:
            print(f"   ⚠️  Model shows room for improvement in trajectory prediction accuracy")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n🎉 Ground truth trajectory testing completed successfully!")
        else:
            print(f"\n❌ Testing failed!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()