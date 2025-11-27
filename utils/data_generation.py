import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    """Dataset of state-action-velocity tuples"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'state': torch.FloatTensor(sample['state']),
            'actions': torch.FloatTensor(sample['actions']),
            'velocity': torch.FloatTensor(sample['velocity']),
            'time': torch.FloatTensor([sample['time']]),
            'next_states': torch.FloatTensor(sample['next_states'])
        }

def generate_training_data(env, num_trajectories=1000, traj_length=20, action_seq_len=5):
    """
    Generate training dataset from environment

    Args:
        env: PointMass2D environment
        num_trajectories: Number of trajectories
        traj_length: Steps per trajectory
        action_seq_len: Length of action sequences

    Returns:
        dataset: List of dicts
    """
    dataset = []

    for _ in range(num_trajectories):
        state = env.reset()

        for t in range(traj_length):
            # Generate random action sequence
            actions = np.random.uniform(
                env.action_bounds[0],
                env.action_bounds[1],
                size=(action_seq_len, env.action_dim)
            ).astype(np.float32)

            # Compute ground truth velocity
            velocity = env.get_velocity(state, actions[0])

            # Rollout to get future states
            rollout_states = env.rollout(state, actions, num_steps=action_seq_len)

            dataset.append({
                'state': state.copy(),
                'actions': actions.copy(),
                'velocity': velocity.copy(),
                'next_states': rollout_states.copy(),
                'time': t * env.dt
            })

            # Advance to next state
            state, _, _ = env.step(actions[0])

    return dataset

def create_dataloader(data, batch_size=128, shuffle=True):
    """Create PyTorch DataLoader"""
    dataset = TrajectoryDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)