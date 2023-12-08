import os
import torch
from torch.utils.data import Dataset
from diffusion.utils.mocap_v2 import MocapDM
import numpy as np

class SpinkickMotionDataset(Dataset):
    def __init__(self, motion_src_path):
        self.motion_data = []
        self.mocap_dm = MocapDM()

        self.mocap_dm.load_mocap(filepath=motion_src_path)

        num_frames = len(self.mocap_dm.data_config)

        for i in range(num_frames):
            first_pos = self.mocap_dm.data_config[i]

            if i == num_frames: # ith pose will get the 0 to n-1 velocity
                rest_vel = self.mocap_dm.data_vel[:num_frames-1] 
            else: # rest will get i+1th to ith velocity
                rest_vel = self.mocap_dm.data_vel[i+1:num_frames] + self.mocap_dm.data_vel[0:i] 

            motion = [first_pos] + rest_vel
            motion = np.array(motion)
            motion = torch.from_numpy(motion).float()
            self.motion_data.append(motion)

        # Repeat motion_data 20 times
        self.motion_data *= 20

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        return self.motion_data[idx]

if __name__ == '__main__':
    dataset = SpinkickMotionDataset("/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_spinkick.txt")
    print(dataset[0].shape)
    # print(dataset[0][1] == dataset[0][1])
    # print(dataset[0][1] == dataset[1][0])
    # print(dataset[0][1] == dataset[2][-1])

    # for data in dataset:
    #     print(data.shape)
