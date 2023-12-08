import os
import torch
from torch.utils.data import Dataset
from diffusion.utils.mocap_v2 import MocapDM
import numpy as np

class BackflipMotionDataset(Dataset):
    def __init__(self, motion_src_path):
        self.motion_data = []
        self.mocap_dm = MocapDM()

        filename = "humanoid3d_backflip.txt"
        self.mocap_dm.load_mocap(filepath=motion_src_path)

        num_frames = len(self.mocap_dm.data_config)
        for i in range(num_frames):
            data_config = self.mocap_dm.data_config[i:num_frames] + self.mocap_dm.data_config[0:i]
            data_config = torch.tensor(np.array(data_config))

            data_vel = self.mocap_dm.data_vel[i:num_frames] + self.mocap_dm.data_vel[0:i]
            data_vel = torch.tensor(np.array(data_vel))

            motion = torch.cat((data_config, data_vel), dim=1)
            self.motion_data.append(motion)

        # Repeat motion_data 10 times
        self.motion_data *= 10

        # Replace motion data with just the first data but multiply that by 200 times
        # self.motion_data = [self.motion_data[0]] * 1000

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        return self.motion_data[idx]

if __name__ == '__main__':
    dataset = BackflipMotionDataset("/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_backflip.txt")
    print(dataset[0].shape)
    print(dataset[0][1] == dataset[0][1])
    # print(dataset[0][1] == dataset[1][0])
    # print(dataset[0][1] == dataset[2][-1])

    # for data in dataset:
    #     print(data.shape)
