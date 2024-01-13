import os
import re
from os.path import isfile
import torch
from torch.utils.data import Dataset
import numpy as np

def split_line(line):
    # Define the regular expression pattern
    pattern = r'(\w+)_\d+_\d+_(FaceTalk_\d+_\d+_\w+)\.npy'

    # Use re.match to find the groups in the line
    match = re.match(pattern, line)

    # Check if the line matches the pattern
    if match:
        # Extract the groups
        groups = match.groups()
        return list(groups)
    else:
        # Return None if the line does not match the pattern
        return None

class FastDatasetNaima(Dataset):
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
                label = split_line(label)[0]
            else:
                label = label[:label.find("_")]
            if label not in labels: labels.append(label)
        self.dict_emotions = {label: idx for idx, label in enumerate(labels)}
        self.num_classes = len(labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        animation = np.load(self.file_list[idx], allow_pickle=True)
        label = os.path.basename(self.file_list[idx])

        if "FaceTalk" in label:
            face = split_line(label)[1]
            label = split_line(label)[0]
        else:
            face = label[label.find("_") + 1:label.find(".")]
            label = label[:label.find("_")]

        path = self.file_list[idx]
        id_template = self.name_actors.index(face)
        template = self.actors[id_template]
        for i in range(animation.shape[0]):
            animation[i] = animation[i] - template

        animation = torch.Tensor(animation)
        length = animation.shape[0]
        return animation, self.dict_emotions[label], path, length
