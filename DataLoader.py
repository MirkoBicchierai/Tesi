import os
from os.path import isfile
import torch
from torch.utils.data import Dataset
import numpy as np


class FastDataset(Dataset):
    def __init__(self, folder_path, actors, name_actors):
        self.actors = actors
        self.name_actors = name_actors
        self.folder_path = folder_path
        self.file_list = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                          isfile(os.path.join(folder_path, f))]
        labels = []
        for f in self.file_list:
            label = os.path.basename(f)
            if "FaceTalk" in label:
                label = label[:label.find("_FaceTalk")]
            else:
                label = label[:label.find("_")]
            if label not in labels: labels.append(label)
        self.dict_emotions = {label: idx for idx, label in enumerate(labels)}
        self.num_classes = len(labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        animation = np.load(self.file_list[idx], allow_pickle=True)
        tmp = os.path.basename(self.file_list[idx])
        label = os.path.basename(self.file_list[idx])

        if "FaceTalk" in label:
            face = label[label.find("_") + 1:label.find(".")]
            label = label[:label.find("_FaceTalk")]
            face = face[face.index("FaceTalk"):]
        else:
            face = label[label.find("_") + 1:label.find(".")]
            label = label[:label.find("_")]


        path = self.file_list[idx]
        id_template = self.name_actors.index(face)
        template = self.actors[id_template]
        for i in range(animation.shape[0]):
            animation[i] = animation[i] - template
        return torch.Tensor(animation), self.dict_emotions[label], path
