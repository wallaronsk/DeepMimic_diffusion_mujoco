import os
import json
from torch.utils.data import Dataset

class MujocoMotionDataset(Dataset):
    def __init__(self, folder_path='src/data/motions'):
        self.folder_path = folder_path
        self.motion_data = []
        self.labels = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    self.motion_data.append(data['Frames'])

                label = filename.replace('humanoid3d_', '').replace('.txt', '')
                self.labels.append(label)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        return self.motion_data[idx], self.labels[idx]

if __name__ == '__main__':
    folder_path = 'src/data/motions'
    dataset = MujocoMotionDataset(folder_path)

    for i in range(len(dataset)):
        motion_data, label = dataset[i]
        print(label, len(motion_data))