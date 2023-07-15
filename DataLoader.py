import os
from os.path import isfile
import torch
from torch.utils.data import Dataset
import numpy as np


class FastDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                          isfile(os.path.join(folder_path, f))]
        labels = []
        for f in self.file_list:
            label = os.path.basename(f)
            label = label[:label.find("_")]
            if label not in labels: labels.append(label)
        self.dict_emotions = {label: idx for idx, label in enumerate(labels)}
        # self.dict_emotions = {'Cheeky': 0, 'Fear': 1, 'Disgust': 2, 'Excitement': 3, 'Bored': 4, 'Afraid': 5,
        # 'Displeased': 6, 'Concentrate': 7, 'Happy': 8, 'Flirting': 9}
        self.num_classes = len(labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        animation = np.load(self.file_list[idx], allow_pickle=True)
        label = os.path.basename(self.file_list[idx])
        label = label[:label.find("_")]
        path = self.file_list[idx]
        return torch.Tensor(animation), self.dict_emotions[label], path
