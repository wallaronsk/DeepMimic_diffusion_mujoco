import os
import torch
from torch.utils.data import Dataset
from diffusion.utils.mocap_v2 import MocapDM
import numpy as np

class SpinkickFramesDataset(Dataset):
    def __init__(self, motion_src_path):
        self.motion_data = []
        self.mocap_dm = MocapDM()

        self.mocap_dm.load_mocap(filepath=motion_src_path)

        frame_data = self.mocap_dm.frames_raw
        for i in range(len(frame_data)):
            prefix = frame_data[i:]
            suffix = frame_data[:i]
            motion = np.concatenate((prefix, suffix), axis=0)
            motion = torch.from_numpy(motion).float()
            self.motion_data.append(motion)

        self.motion_data = torch.stack(self.motion_data)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        return self.motion_data[idx]

if __name__ == '__main__':
    dataset = SpinkickFramesDataset("/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_spinkick.txt")
    print(dataset.motion_data.shape)
    print(dataset.motion_data[0][1] == dataset.motion_data[1][0])
    print(dataset.motion_data)