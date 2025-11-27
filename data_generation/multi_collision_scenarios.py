import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

class MultiCollisionScenarioGenerator:
    """Generate physics-informed collision scenarios to stress-test shortcut models"""

    def __init__(self, env):
        """
        Args:
            env: PointMass2D physics environment
        """
        self.env = env

        # Collision scenario distribution (stress-test focused)
        self.collision_distribution = {
            'smooth_motion': 0.30,        # No collisions - shortcuts should excel
            'single_collision': 0.25,     # One wall bounce
            'double_collision': 0.20,     # Two sequential collisions
            'triple_collision': 0.15,     # Three sequential collisions
            'quad_plus_collision': 0.10   # 4+ collisions - shortcuts will struggle
        }

        # Force pattern types
        self.force_patterns = [
            'zero_force',          # Pure momentum + damping
            'constant_force',      # Steady acceleration
            'impulse_force',       # Brief high-magnitude forces
            'oscillating_force',   # Sine/cosine patterns
            'opposing_force',      # Force against velocity direction
            'collision_inducing'   # Forces designed to create collisions
        ]

        # Energy regimes for diverse dynamics
        self.energy_regimes = ['low', 'medium', 'high']

    def generate_scenario_specific_data(self, scenario_type: str, energy_level: str, count: int) -> List[Dict]:
        """Generate specific collision scenario data"""

        scenarios = []
        for _ in range(count):
            if scenario_type == 'smooth_motion':
                scenario = self._generate_smooth_trajectory(energy_level)
            elif scenario_type == 'single_collision':
                scenario = self._generate_single_collision(energy_level)
            elif scenario_type == 'double_collision':
                scenario = self._generate_double_collision(energy_level)
            elif scenario_type == 'triple_collision':
                scenario = self._generate_triple_collision(energy_level)
            elif scenario_type == 'quad_plus_collision':
                scenario = self._generate_quad_plus_collision(energy_level)
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")

            scenarios.append(scenario)

        return scenarios

    def _generate_smooth_trajectory(self, energy_level: str) -> Dict:
        """Generate smooth motion without collisions"""

        # Start away from walls to avoid collisions
        x = np.random.uniform(-2.0, 2.0)  # Safe zone
        y = np.random.uniform(-2.0, 2.0)

        # Energy-appropriate velocities
        if energy_level == 'low':
            vx, vy = np.random.uniform(-0.5, 0.5, 2)
        elif energy_level == 'medium':
            vx, vy = np.random.uniform(-1.0, 1.0, 2)
        else:  # high
            vx, vy = np.random.uniform(-1.5, 1.5, 2)

        # Forces that maintain smooth motion (no sudden changes)
        force_pattern = self._generate_smooth_forces()

        return {
            'initial_state': np.array([x, y, vx, vy], dtype=np.float32),
            'force_pattern': force_pattern,
            'expected_collisions': 0,
            'scenario_type': 'smooth_motion',
            'energy_level': energy_level
        }

    def _generate_single_collision(self, energy_level: str) -> Dict:
        """Generate trajectory with exactly one wall collision"""

        # Position to guarantee wall hit
        wall_choice = np.random.choice(['left', 'right', 'top', 'bottom'])

        if wall_choice == 'right':
            x = np.random.uniform(3.0, 4.0)      # Near right wall
            y = np.random.uniform(-2.0, 2.0)
            vx = np.random.uniform(0.5, 2.0)     # Moving toward wall
            vy = np.random.uniform(-1.0, 1.0)
        elif wall_choice == 'left':
            x = np.random.uniform(-4.0, -3.0)    # Near left wall
            y = np.random.uniform(-2.0, 2.0)
            vx = np.random.uniform(-2.0, -0.5)   # Moving toward wall
            vy = np.random.uniform(-1.0, 1.0)
        elif wall_choice == 'top':
            x = np.random.uniform(-2.0, 2.0)
            y = np.random.uniform(3.0, 4.0)      # Near top wall
            vx = np.random.uniform(-1.0, 1.0)
            vy = np.random.uniform(0.5, 2.0)     # Moving toward wall
        else:  # bottom
            x = np.random.uniform(-2.0, 2.0)
            y = np.random.uniform(-4.0, -3.0)    # Near bottom wall
            vx = np.random.uniform(-1.0, 1.0)
            vy = np.random.uniform(-2.0, -0.5)   # Moving toward wall

        # Adjust energy level
        if energy_level == 'low':
            vx, vy = 0.5 * vx, 0.5 * vy
        elif energy_level == 'high':
            vx, vy = 1.5 * vx, 1.5 * vy

        force_pattern = self._generate_collision_forces('single')

        return {
            'initial_state': np.array([x, y, vx, vy], dtype=np.float32),
            'force_pattern': force_pattern,
            'expected_collisions': 1,
            'scenario_type': 'single_collision',
            'energy_level': energy_level,
            'target_wall': wall_choice
        }

    def _generate_double_collision(self, energy_level: str) -> Dict:
        """Generate trajectory with two wall collisions"""

        # Start near corner to encourage multiple bounces
        corner_choice = np.random.choice(['top_right', 'top_left', 'bottom_right', 'bottom_left'])

        if corner_choice == 'top_right':
            x = np.random.uniform(3.5, 4.5)
            y = np.random.uniform(3.5, 4.5)
            vx = np.random.uniform(0.5, 1.5)      # Toward corner
            vy = np.random.uniform(0.5, 1.5)
        elif corner_choice == 'top_left':
            x = np.random.uniform(-4.5, -3.5)
            y = np.random.uniform(3.5, 4.5)
            vx = np.random.uniform(-1.5, -0.5)
            vy = np.random.uniform(0.5, 1.5)
        elif corner_choice == 'bottom_right':
            x = np.random.uniform(3.5, 4.5)
            y = np.random.uniform(-4.5, -3.5)
            vx = np.random.uniform(0.5, 1.5)
            vy = np.random.uniform(-1.5, -0.5)
        else:  # bottom_left
            x = np.random.uniform(-4.5, -3.5)
            y = np.random.uniform(-4.5, -3.5)
            vx = np.random.uniform(-1.5, -0.5)
            vy = np.random.uniform(-1.5, -0.5)

        # Energy adjustment
        if energy_level == 'low':
            vx, vy = 0.6 * vx, 0.6 * vy
        elif energy_level == 'high':
            vx, vy = 1.4 * vx, 1.4 * vy

        force_pattern = self._generate_collision_forces('double')

        return {
            'initial_state': np.array([x, y, vx, vy], dtype=np.float32),
            'force_pattern': force_pattern,
            'expected_collisions': 2,
            'scenario_type': 'double_collision',
            'energy_level': energy_level,
            'target_corner': corner_choice
        }

    def _generate_triple_collision(self, energy_level: str) -> Dict:
        """Generate the wall→wall→ground collision sequence (stress test)"""

        # Position for wall A → wall B → wall C sequence
        x = np.random.uniform(3.5, 4.2)      # Near right wall
        y = np.random.uniform(1.0, 3.0)      # Upper area for gravity effect
        vx = np.random.uniform(1.5, 2.5)     # Strong rightward motion
        vy = np.random.uniform(-0.5, 0.5)    # Slight vertical component

        # Energy scaling
        if energy_level == 'low':
            vx, vy = 0.7 * vx, 0.7 * vy
        elif energy_level == 'high':
            vx, vy = 1.3 * vx, 1.3 * vy

        # Force pattern to encourage three collisions
        force_pattern = self._generate_triple_collision_forces()

        return {
            'initial_state': np.array([x, y, vx, vy], dtype=np.float32),
            'force_pattern': force_pattern,
            'expected_collisions': 3,
            'scenario_type': 'triple_collision',
            'energy_level': energy_level,
            'sequence': 'wall_wall_ground'
        }

    def _generate_quad_plus_collision(self, energy_level: str) -> Dict:
        """Generate high-energy scenarios with 4+ collisions"""

        # High-energy central position
        x = np.random.uniform(-1.0, 1.0)      # Central area
        y = np.random.uniform(-1.0, 1.0)

        # High initial velocity for multiple bounces
        vx = np.random.uniform(-2.5, 2.5)
        vy = np.random.uniform(-2.5, 2.5)

        # Energy scaling (even high energy gets some variation)
        if energy_level == 'low':
            vx, vy = 0.8 * vx, 0.8 * vy
        elif energy_level == 'high':
            vx, vy = 1.2 * vx, 1.2 * vy

        # Complex force pattern for chaotic motion
        force_pattern = self._generate_chaotic_forces()

        return {
            'initial_state': np.array([x, y, vx, vy], dtype=np.float32),
            'force_pattern': force_pattern,
            'expected_collisions': 4,  # Minimum expected
            'scenario_type': 'quad_plus_collision',
            'energy_level': energy_level,
            'motion_type': 'chaotic'
        }

    def _generate_smooth_forces(self) -> np.ndarray:
        """Generate forces that maintain smooth motion"""

        pattern_type = np.random.choice(['zero', 'gentle_constant', 'light_oscillating'])

        if pattern_type == 'zero':
            forces = np.zeros((10, 2), dtype=np.float32)
        elif pattern_type == 'gentle_constant':
            fx = np.random.uniform(-0.3, 0.3)
            fy = np.random.uniform(-0.3, 0.3)
            forces = np.full((10, 2), [fx, fy], dtype=np.float32)
        else:  # light_oscillating
            t = np.linspace(0, 2*np.pi, 10)
            fx = 0.2 * np.sin(t)
            fy = 0.2 * np.cos(t)
            forces = np.column_stack([fx, fy]).astype(np.float32)

        return forces

    def _generate_collision_forces(self, collision_type: str) -> np.ndarray:
        """Generate forces appropriate for collision scenarios"""

        if collision_type == 'single':
            # Minimal forces to let collision occur naturally
            forces = np.random.uniform(-0.5, 0.5, (10, 2)).astype(np.float32)
        else:  # double
            # Moderate forces to encourage second collision
            forces = np.random.uniform(-0.8, 0.8, (10, 2)).astype(np.float32)

        return forces

    def _generate_triple_collision_forces(self) -> np.ndarray:
        """Generate forces for wall→wall→ground sequence"""

        forces = np.zeros((10, 2), dtype=np.float32)

        # Early forces: let natural collision occur
        forces[:3] = np.random.uniform(-0.3, 0.3, (3, 2))

        # Middle forces: encourage second collision
        forces[3:6, 0] = np.random.uniform(-1.0, 0.0, 3)  # Push left after right wall
        forces[3:6, 1] = np.random.uniform(-0.5, 0.5, 3)

        # Late forces: encourage ground collision
        forces[6:, 1] = np.random.uniform(-1.5, -0.5, 4)  # Downward force
        forces[6:, 0] = np.random.uniform(-0.5, 0.5, 4)

        return forces

    def _generate_chaotic_forces(self) -> np.ndarray:
        """Generate complex forces for multiple collisions"""

        # Random walk with moderate magnitude
        forces = np.random.uniform(-1.0, 1.0, (10, 2)).astype(np.float32)

        # Add some structure to encourage bouncing
        forces[::2, :] *= 1.5  # Stronger forces every other step

        return forces

    def _simulate_trajectory(self, scenario: Dict) -> Dict:
        """Run physics simulation to generate complete trajectory"""

        # Set up environment
        self.env.clear_particles()
        initial_state = scenario['initial_state']
        x, y, vx, vy = initial_state
        self.env.add_particle(x, y, vx, vy, mass=getattr(self.env, 'mass', 1.0), radius=0.15)

        # Run simulation
        trajectory = [initial_state.copy()]
        forces = scenario['force_pattern']

        for i in range(100):  # 1.0 second at dt=0.01
            force_idx = min(i, len(forces) - 1)
            force = forces[force_idx]

            next_state, _, _ = self.env.step(force)
            trajectory.append(next_state.copy())

        trajectory = np.array(trajectory, dtype=np.float32)

        # Count actual collisions
        actual_collisions = self._count_collisions_in_trajectory(trajectory)

        return {
            'scenario': scenario,
            'trajectory': trajectory,
            'actual_collisions': actual_collisions,
            'trajectory_length': len(trajectory),
        }

    def _count_collisions_in_trajectory(self, trajectory: np.ndarray) -> int:
        """Count wall collisions in trajectory"""

        collisions = 0
        prev_state = trajectory[0]

        for state in trajectory[1:]:
            # Check if position jumped due to collision (boundary enforcement)
            pos_change = np.abs(state[:2] - prev_state[:2])
            vel_change = np.abs(state[2:] - prev_state[2:])

            # Collision detected if large velocity change with small position change
            if np.any(vel_change > 0.5) and np.any(pos_change < 0.1):
                # Check if near boundary
                if (np.abs(state[0]) > 4.5 or np.abs(state[1]) > 4.5):
                    collisions += 1

            prev_state = state

        return collisions

    def generate_dataset(self, total_trajectories: int = 10000,
                        collision_bias: float = 0.7) -> List[Dict]:
        """
        Generate complete multi-collision dataset

        Args:
            total_trajectories: Total number of trajectories
            collision_bias: Fraction of data that should have collisions (0.7 = 70%)

        Returns:
            List of trajectory dictionaries
        """

        print(f"🎯 Generating Multi-Collision Dataset ({total_trajectories} trajectories)")
        print(f"   Collision bias: {collision_bias*100:.0f}% multi-collision scenarios")

        # Adjust distribution based on collision bias
        if collision_bias > 0.5:
            # Emphasize collision scenarios
            adjusted_dist = {
                'smooth_motion': (1 - collision_bias),
                'single_collision': collision_bias * 0.35,
                'double_collision': collision_bias * 0.30,
                'triple_collision': collision_bias * 0.20,
                'quad_plus_collision': collision_bias * 0.15
            }
        else:
            adjusted_dist = self.collision_distribution

        dataset = []

        for scenario_type, ratio in adjusted_dist.items():
            count = int(total_trajectories * ratio)
            print(f"   Generating {count} {scenario_type} scenarios...")

            # Distribute across energy levels
            count_per_energy = count // len(self.energy_regimes)

            for energy_level in self.energy_regimes:
                scenarios = self.generate_scenario_specific_data(
                    scenario_type, energy_level, count_per_energy
                )

                # Simulate each scenario
                for scenario in scenarios:
                    trajectory_data = self._simulate_trajectory(scenario)
                    dataset.append(trajectory_data)

        # Shuffle dataset
        np.random.shuffle(dataset)

        # Validate collision distribution
        self._validate_dataset_distribution(dataset)

        print(f"✅ Generated {len(dataset)} diverse physics trajectories")
        return dataset

    def _validate_dataset_distribution(self, dataset: List[Dict]):
        """Validate that generated dataset matches target distribution"""

        collision_counts = {}
        scenario_counts = {}

        for sample in dataset:
            # Count by actual collisions
            actual = sample['actual_collisions']
            collision_key = str(actual) if actual < 4 else '4+'
            collision_counts[collision_key] = collision_counts.get(collision_key, 0) + 1

            # Count by scenario type
            scenario = sample['scenario']['scenario_type']
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        total = len(dataset)

        print("\\n📊 Dataset Validation:")
        print("   Collision Distribution:")
        for collision_count, count in sorted(collision_counts.items()):
            percentage = count / total * 100
            print(f"     {collision_count} collisions: {count:4d} ({percentage:4.1f}%)")

        print("   Scenario Distribution:")
        for scenario, count in scenario_counts.items():
            percentage = count / total * 100
            print(f"     {scenario}: {count:4d} ({percentage:4.1f}%)")

def save_multi_collision_dataset(dataset: List[Dict], filename: str = "multi_collision_dataset.pkl"):
    """Save generated dataset to file"""

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    filepath = data_dir / filename

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"💾 Saved dataset to {filepath}")
    print(f"   File size: {filepath.stat().st_size / (1024*1024):.1f} MB")

def load_multi_collision_dataset(filename: str = "multi_collision_dataset.pkl") -> List[Dict]:
    """Load dataset from file"""

    filepath = Path("data") / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)

    print(f"📁 Loaded dataset from {filepath}")
    print(f"   {len(dataset)} trajectories loaded")

    return dataset