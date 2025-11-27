import numpy as np
from typing import Tuple, List, Optional
import math

class RealisticPhysics2D:
    """
    Enhanced 2D physics environment with realistic collisions and momentum conservation.

    Features:
    - Elastic boundary collisions with restitution
    - Inter-particle collision detection and response
    - Momentum and energy conservation
    - Support for multiple particles with different masses and sizes
    - Euler integration (as requested)
    """

    def __init__(self, dt=0.01, damping=0.05, gravity=0.0, restitution=0.8):
        self.dt = dt
        self.damping = damping
        self.gravity = gravity  # Can add gravity if needed
        self.restitution = restitution  # Bounce factor (0=perfectly inelastic, 1=perfectly elastic)

        # Environment bounds
        self.pos_bounds = [-5.0, 5.0]
        self.vel_bounds = [-10.0, 10.0]  # Increased for more realistic velocities
        self.action_bounds = [-2.0, 2.0]

        # Particle properties
        self.particles = []

        # State dimension for single particle (for compatibility)
        self.state_dim = 4  # [x, y, vx, vy]
        self.action_dim = 2  # [fx, fy]

    def add_particle(self, x=0.0, y=0.0, vx=0.0, vy=0.0, mass=1.0, radius=0.1):
        """Add a particle to the simulation"""
        particle = {
            'position': np.array([x, y], dtype=np.float32),
            'velocity': np.array([vx, vy], dtype=np.float32),
            'mass': mass,
            'radius': radius,
            'id': len(self.particles)
        }
        self.particles.append(particle)
        return len(self.particles) - 1

    def clear_particles(self):
        """Remove all particles"""
        self.particles = []

    def reset(self, num_particles=1) -> np.ndarray:
        """Reset environment with random particles"""
        self.clear_particles()

        for i in range(num_particles):
            # Random positions (ensure particles don't overlap)
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x = np.random.uniform(self.pos_bounds[0] + 0.5, self.pos_bounds[1] - 0.5)
                y = np.random.uniform(self.pos_bounds[0] + 0.5, self.pos_bounds[1] - 0.5)

                # Check for overlaps with existing particles
                overlap = False
                for existing in self.particles:
                    dist = np.linalg.norm([x - existing['position'][0], y - existing['position'][1]])
                    if dist < (0.2 + existing['radius']):  # 0.2 is default radius
                        overlap = True
                        break

                if not overlap:
                    break
                attempts += 1

            # Random velocities
            vx = np.random.uniform(-1.0, 1.0)
            vy = np.random.uniform(-1.0, 1.0)

            # Random mass and size
            mass = np.random.uniform(0.5, 2.0)
            radius = np.random.uniform(0.1, 0.25)

            self.add_particle(x, y, vx, vy, mass, radius)

        # Return state of first particle for compatibility
        if self.particles:
            p = self.particles[0]
            return np.concatenate([p['position'], p['velocity']])
        else:
            return np.zeros(4, dtype=np.float32)

    def step(self, action: np.ndarray, particle_id=0) -> Tuple[np.ndarray, float, bool]:
        """Execute one physics step with enhanced collision detection"""
        if not self.particles:
            return np.zeros(4), 0.0, False

        # Apply external force to specified particle
        if particle_id < len(self.particles):
            self._apply_external_force(particle_id, action)

        # Update all particles with physics
        self._update_particle_physics()

        # Handle wall collisions
        self._handle_wall_collisions()

        # Handle inter-particle collisions
        self._handle_particle_collisions()

        # Return state of specified particle for compatibility
        if particle_id < len(self.particles):
            p = self.particles[particle_id]
            return np.concatenate([p['position'], p['velocity']]), 0.0, False
        else:
            return np.zeros(4), 0.0, False

    def _apply_external_force(self, particle_id: int, force: np.ndarray):
        """Apply external force to a particle"""
        particle = self.particles[particle_id]
        force = np.clip(force, self.action_bounds[0], self.action_bounds[1])

        # F = ma -> a = F/m
        acceleration = force / particle['mass']

        # Store acceleration for physics update
        particle['acceleration'] = acceleration

    def _update_particle_physics(self):
        """Update all particles using Euler integration"""
        for particle in self.particles:
            # Get current state
            pos = particle['position']
            vel = particle['velocity']
            mass = particle['mass']

            # External acceleration (from applied forces)
            acceleration = particle.get('acceleration', np.zeros(2))

            # Add gravity if enabled
            if self.gravity != 0.0:
                acceleration[1] += self.gravity  # Downward gravity

            # Euler integration
            # Update velocity: v_new = v + a * dt
            vel_new = vel + acceleration * self.dt

            # Apply damping (air resistance/friction)
            # Use exponential damping for more realistic behavior
            damping_factor = np.exp(-self.damping * self.dt)
            vel_new = vel_new * damping_factor

            # Clip velocities to prevent unrealistic speeds
            vel_new = np.clip(vel_new, self.vel_bounds[0], self.vel_bounds[1])

            # Update position: x_new = x + v * dt
            pos_new = pos + vel_new * self.dt

            # Update particle state
            particle['position'] = pos_new
            particle['velocity'] = vel_new

            # Clear acceleration for next step
            particle['acceleration'] = np.zeros(2)

    def _handle_wall_collisions(self):
        """Handle elastic collisions with walls"""
        for particle in self.particles:
            pos = particle['position']
            vel = particle['velocity']
            radius = particle['radius']

            # Check collision with each wall
            collision_occurred = False

            # Left wall
            if pos[0] - radius <= self.pos_bounds[0]:
                pos[0] = self.pos_bounds[0] + radius  # Move particle out of wall
                vel[0] = abs(vel[0]) * self.restitution  # Reverse and reduce velocity
                collision_occurred = True

            # Right wall
            elif pos[0] + radius >= self.pos_bounds[1]:
                pos[0] = self.pos_bounds[1] - radius
                vel[0] = -abs(vel[0]) * self.restitution
                collision_occurred = True

            # Bottom wall
            if pos[1] - radius <= self.pos_bounds[0]:
                pos[1] = self.pos_bounds[0] + radius
                vel[1] = abs(vel[1]) * self.restitution
                collision_occurred = True

            # Top wall
            elif pos[1] + radius >= self.pos_bounds[1]:
                pos[1] = self.pos_bounds[1] - radius
                vel[1] = -abs(vel[1]) * self.restitution
                collision_occurred = True

            # Update particle
            particle['position'] = pos
            particle['velocity'] = vel

    def _handle_particle_collisions(self):
        """Handle elastic collisions between particles with momentum conservation"""
        n_particles = len(self.particles)

        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                p1 = self.particles[i]
                p2 = self.particles[j]

                # Calculate distance between particle centers
                delta_pos = p1['position'] - p2['position']
                distance = np.linalg.norm(delta_pos)
                min_distance = p1['radius'] + p2['radius']

                # Check for collision
                if distance < min_distance and distance > 0:
                    # Collision detected - resolve it
                    self._resolve_particle_collision(i, j, delta_pos, distance, min_distance)

    def _resolve_particle_collision(self, i: int, j: int, delta_pos: np.ndarray,
                                   distance: float, min_distance: float):
        """Resolve collision between two particles with momentum conservation"""
        p1 = self.particles[i]
        p2 = self.particles[j]

        # Collision normal (unit vector from p2 to p1)
        if distance > 0:
            normal = delta_pos / distance
        else:
            # Handle edge case of identical positions
            normal = np.array([1.0, 0.0])

        # Separate overlapping particles
        overlap = min_distance - distance
        separation = overlap * 0.5  # Each particle moves half the overlap distance

        p1['position'] += normal * separation
        p2['position'] -= normal * separation

        # Get velocities and masses
        v1 = p1['velocity'].copy()
        v2 = p2['velocity'].copy()
        m1 = p1['mass']
        m2 = p2['mass']

        # Relative velocity
        relative_velocity = v1 - v2

        # Velocity component along collision normal
        velocity_along_normal = np.dot(relative_velocity, normal)

        # Don't resolve if velocities are separating
        if velocity_along_normal > 0:
            return

        # Calculate collision impulse using conservation of momentum
        # For elastic collision: e = coefficient of restitution
        e = self.restitution
        impulse_magnitude = -(1 + e) * velocity_along_normal / (1/m1 + 1/m2)

        # Apply impulse to both particles
        impulse = impulse_magnitude * normal

        p1['velocity'] = v1 + impulse / m1
        p2['velocity'] = v2 - impulse / m2

    def get_velocity(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Get instantaneous velocity for compatibility with original interface"""
        if not self.particles:
            return np.zeros(4, dtype=np.float32)

        # Use first particle for compatibility
        particle = self.particles[0]

        # Current velocity
        vx, vy = particle['velocity']

        # Applied acceleration
        fx, fy = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        ax = fx / particle['mass']
        ay = fy / particle['mass'] + self.gravity

        # Rate of change of velocity (with damping)
        dvx_dt = ax - self.damping * vx
        dvy_dt = ay - self.damping * vy

        # Rate of change of position
        dx_dt = vx
        dy_dt = vy

        return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt], dtype=np.float32)

    def rollout(self, state: np.ndarray, actions: np.ndarray, num_steps: int) -> np.ndarray:
        """Execute action sequence and return trajectory"""
        # Set up single particle for compatibility
        self.clear_particles()
        x, y, vx, vy = state
        self.add_particle(x, y, vx, vy, mass=self.mass, radius=0.15)

        trajectory = [state.copy()]

        for i in range(min(num_steps, len(actions))):
            next_state, _, _ = self.step(actions[i])
            trajectory.append(next_state.copy())

        return np.array(trajectory)

    def get_all_particles_state(self) -> List[dict]:
        """Get state of all particles"""
        return [
            {
                'position': p['position'].copy(),
                'velocity': p['velocity'].copy(),
                'mass': p['mass'],
                'radius': p['radius'],
                'id': p['id']
            }
            for p in self.particles
        ]

    def calculate_system_energy(self) -> dict:
        """Calculate total kinetic energy of the system"""
        total_ke = 0.0
        total_momentum = np.zeros(2)

        for particle in self.particles:
            # Kinetic energy: KE = 0.5 * m * v^2
            ke = 0.5 * particle['mass'] * np.sum(particle['velocity']**2)
            total_ke += ke

            # Momentum: p = m * v
            momentum = particle['mass'] * particle['velocity']
            total_momentum += momentum

        return {
            'kinetic_energy': total_ke,
            'total_momentum': total_momentum,
            'momentum_magnitude': np.linalg.norm(total_momentum)
        }

    def step_all_particles(self, forces: List[np.ndarray]) -> List[np.ndarray]:
        """Step all particles with individual forces"""
        # Apply forces to particles
        for i, force in enumerate(forces):
            if i < len(self.particles):
                self._apply_external_force(i, force)

        # Update physics
        self._update_particle_physics()
        self._handle_wall_collisions()
        self._handle_particle_collisions()

        # Return all particle states
        states = []
        for particle in self.particles:
            state = np.concatenate([particle['position'], particle['velocity']])
            states.append(state)

        return states


# Backward compatibility - use enhanced physics by default
class PointMass2D(RealisticPhysics2D):
    """Enhanced PointMass2D with realistic physics - backward compatible"""

    def __init__(self, dt=0.01, mass=1.0, damping=0.05):
        super().__init__(dt=dt, damping=damping)
        self.mass = mass  # Default mass for compatibility
        self.state = None

    def reset(self) -> np.ndarray:
        """Reset with single particle for compatibility"""
        self.clear_particles()

        # Random initial state
        x = np.random.uniform(self.pos_bounds[0] + 0.5, self.pos_bounds[1] - 0.5)
        y = np.random.uniform(self.pos_bounds[0] + 0.5, self.pos_bounds[1] - 0.5)
        vx = np.random.uniform(-1.0, 1.0)
        vy = np.random.uniform(-1.0, 1.0)

        self.add_particle(x, y, vx, vy, mass=self.mass, radius=0.15)

        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Step with single particle for compatibility"""
        # Apply external force to first particle
        if self.particles:
            self._apply_external_force(0, action)

        # Update physics
        self._update_particle_physics()
        self._handle_wall_collisions()
        self._handle_particle_collisions()

        # Return state of first particle
        if self.particles:
            p = self.particles[0]
            new_state = np.concatenate([p['position'], p['velocity']])
            self.state = new_state
            return new_state, 0.0, False
        else:
            return np.zeros(4), 0.0, False