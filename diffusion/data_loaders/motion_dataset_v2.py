import os
import torch
from torch.utils.data import Dataset
from utils.mocap_v2 import MocapDM
import numpy as np
from collections import namedtuple


Batch = namedtuple("Batch", "trajectories motion_class")

motion_classes = {
    "humanoid3d_walk": 0,
    "humanoid3d_run": 1,
    "humanoid3d_spinkick": 2,
    "humanoid3d_roll": 3,
    "humanoid3d_dance_a": 4,
    "humanoid3d_dance_b": 5,
    "humanoid3d_jump": 6,
    "humanoid3d_cartwheel": 7,
    "humanoid3d_backflip": 8
}

def get_motion_class(motion_src_path):
    # motion src path will be like data/motions/humanoid3d_walk.txt
    # we want to get the class from the filename
    filename = motion_src_path.split("/")[-1]
    return motion_classes[filename.split(".")[0]]


class MotionDataset(Dataset):
    def __init__(self, motion_src_path, shuffle=False):
        """
        Args:
            motion_src_path: Path to the motion capture file or directory
            shuffle: Whether to shuffle the data
        """
        self.motion_data = []
        self.motion_classes = []
        self.mocap_dm = MocapDM()
        self.max_sequence_length = 0
        
        # Check if the provided path is a directory
        if not motion_src_path.endswith('.txt'):
            # It's a directory, find all txt files
            motion_files = [os.path.join(motion_src_path, f) for f in os.listdir(motion_src_path) 
                          if f.endswith('.txt')]
            # as a test, only keep the first 2 files
            motion_files = motion_files[:2]
        else:
            # It's a single file
            motion_files = [motion_src_path]
        
        for motion_file in motion_files:
            self._load_motion_file(motion_file, shuffle)
        
        # Convert motion classes to tensor
        self.motion_classes = torch.tensor(self.motion_classes)
        
        # Create nested tensor for motion data
        self.motion_data = torch.nested.nested_tensor(self.motion_data, layout=torch.jagged)

    def _load_motion_file(self, motion_file_path, shuffle):
        """Load a single motion file and add its data to the dataset"""
        # Get motion class for this file
        file_motion_class = get_motion_class(motion_file_path)
        
        self.mocap_dm.load_mocap(filepath=motion_file_path)

        data_config = self.mocap_dm.data_config
        data_vel = self.mocap_dm.data_vel

        data_config = np.array(data_config)
        data_vel = np.array(data_vel)

        num_frames = data_config.shape[0]

        if num_frames > self.max_sequence_length:
            self.max_sequence_length = num_frames

        data_config = data_config[:num_frames, :]
        data_vel = data_vel[:num_frames, :]
        
        combined = np.concatenate([data_config, data_vel], axis=1)

        if shuffle:
            diff = combined[-1] - combined[0]
            print(diff[:3])
            print(diff[:2])
            for i in range(len(combined)):
                prefix = combined[i:].copy()
                suffix = combined[:i].copy()
                if i != 0 and i != len(combined) - 1:
                    suffix[:, :3] += diff[:3]
                    first_diff = prefix[0] - combined[0]
                    prefix[:, :3] -= first_diff[:3]
                    suffix[:, :3] -= first_diff[:3]
                prefix = torch.from_numpy(np.array(prefix)).float()
                suffix = torch.from_numpy(np.array(suffix)).float()
                motion = torch.cat([prefix, suffix], dim=0)
                self.motion_data.append(motion)
                self.motion_classes.append(file_motion_class)
        else:
            for _ in range(1000):
                self.motion_data.append(torch.from_numpy(combined).float())
                self.motion_classes.append(file_motion_class)

    def __len__(self):
        return len(self.motion_classes)

    def __getitem__(self, idx):
        motion = self.motion_data[idx]
        motion_class = self.motion_classes[idx]
        return Batch(trajectories=motion, motion_class=motion_class)

    def collate_fn(self, batch):
        """Custom collate function to batch nested tensors
        
        This creates a single nested tensor for the batch instead of padding.
        """
        # Extract trajectories and classes
        trajectories = [item.trajectories for item in batch]
        motion_classes = torch.stack([item.motion_class for item in batch])
        
        # Create nested tensor for trajectories
        nested_trajectories = torch.nested.nested_tensor(trajectories, layout=torch.jagged)
        
        return Batch(
            trajectories=nested_trajectories,
            motion_class=motion_classes
        )

    def get_conditions(self, data_configs):
        """
        condition on current observation for planning
        """
        return {0: data_configs[0]}


if __name__ == "__main__":
    dataset = MotionDataset(
        "/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_walk.txt"
    )
    # print(dataset[0], dataset[0].trajectories.shape)
