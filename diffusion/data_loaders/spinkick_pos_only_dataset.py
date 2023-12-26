import os
import torch
from torch.utils.data import Dataset
from diffusion.utils.mocap_v2 import MocapDM
import numpy as np

class SpinkickPosOnlyDataset(Dataset):
    def __init__(self, motion_src_path):
        self.motion_data = []
        self.mocap_dm = MocapDM()

        self.mocap_dm.load_mocap(filepath=motion_src_path)

        data_config = self.mocap_dm.data_config
        # for i in range(len(data_config)):
        #     prefix = data_config[i:]
        #     suffix = data_config[:i]
        #     motion = prefix + suffix
        #     motion = torch.from_numpy(np.array(motion)).float()
        #     self.motion_data.append(motion)

        for _ in range(1000):
            self.motion_data.append(torch.from_numpy(np.array(data_config)).float())

        self.motion_data = torch.stack(self.motion_data)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        return self.motion_data[idx]

if __name__ == '__main__':
    dataset = SpinkickPosOnlyDataset("/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_spinkick.txt")
    print(dataset.motion_data.shape)