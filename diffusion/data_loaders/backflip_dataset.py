import os
import torch
from torch.utils.data import Dataset
from utils.mocap_v2 import MocapDM
import numpy as np

class BackflipMotionDataset(Dataset):
    def __init__(self, folder_path='src/data/motions'):
        self.folder_path = folder_path
        self.motion_data = []
        self.mocap_dm = MocapDM()

        filename = "humanoid3d_backflip.txt"
        file_path = os.path.join(folder_path, filename)
        self.mocap_dm.load_mocap(filepath=file_path)

        num_frames = len(self.mocap_dm.data_config)
        for i in range(num_frames):
            data_config = self.mocap_dm.data_config[i:num_frames] + self.mocap_dm.data_config[0:i]
            data_config = torch.tensor(np.array(data_config))

            data_vel = self.mocap_dm.data_vel[i:num_frames] + self.mocap_dm.data_vel[0:i]
            data_vel = torch.tensor(np.array(data_vel))

            motion = torch.cat((data_config, data_vel), dim=0)
            print(motion.shape)
            self.motion_data.append(motion)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        return self.motion_data[idx]

if __name__ == '__main__':
    dataset = BackflipMotionDataset()
    first_data_config, first_data_vel = dataset[0]
    last_data_config, last_data_vel = dataset[-1]

    print(first_data_config[0] == last_data_config[1])
