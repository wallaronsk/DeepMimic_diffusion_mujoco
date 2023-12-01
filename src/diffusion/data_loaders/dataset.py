import os
import json
import torch
from torch.utils.data import Dataset
from diffusion.utils.mocap_v2 import MocapDM
import numpy as np

class MujocoMotionDataset(Dataset):
    def __init__(self, folder_path='src/data/motions'):
        self.folder_path = folder_path
        self.pos = []
        self.vel = []
        self.labels = []

        self.mocap_dm = MocapDM()

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                self.mocap_dm.load_mocap(filepath=file_path)
                self.pos.append(self.mocap_dm.data_config)
                self.vel.append(self.mocap_dm.data_vel)
                label = filename.replace('humanoid3d_', '').replace('.txt', '')
                self.labels.append(label)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        pos = torch.tensor(np.array(self.pos[idx]))
        vel = torch.tensor(np.array(self.vel[idx]))
        label = self.labels[idx]
        return pos, vel, label

if __name__ == '__main__':
    folder_path = 'src/data/motions'
    dataset = MujocoMotionDataset(folder_path)

    for i in range(len(dataset)):
        pos, vel, label = dataset[i]
        print(label, len(pos), len(vel))

        if label == "run":
            print("Pos", pos[0])
            print("Vel", vel[0])
    print(dataset.mocap_dm.all_states[0])