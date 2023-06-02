import os
import numpy as np
import torch
from get_landmarks import get_landmarks
import pytorch3d.io
from torch.utils.data import Dataset

train_path = "dataset/SingleExpression/COMA/Partial"
test_path = "dataset/SingleExpression/COMA/Testing"


class SingleEmotionDataset(Dataset):
    def __init__(self, folder_path):
        self.folders = []
        for f in os.listdir(folder_path):
            real_f = os.path.join(folder_path, f)
            self.folders = np.append(self.folders, [os.path.join(real_f, sf) for sf in os.listdir(real_f)])
        self.labels = []
        self.typeface = sum(1 for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item)))
        for f in os.listdir(folder_path):
            real_f = os.path.join(folder_path, f)
            self.labels = np.append(self.labels, [sf for sf in os.listdir(real_f)])
            break
        self.dict_emotions = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        actual_label = self.labels[idx]
        actual_folders = [s for s in self.folders if actual_label in s]
        animation = torch.Tensor([])
        landmark_animation = torch.Tensor([])
        landmark_animations = torch.Tensor([])
        animations = torch.Tensor([])
        for f in actual_folders:
            file_list = sorted([f.decode('utf-8') for f in os.listdir(f)])
            for file_name in file_list:
                if file_name[-3:] == 'obj':
                    frame, _, _ = pytorch3d.io.load_obj(os.path.join(f, file_name), load_textures=False)
                    landmark = torch.tensor(get_landmarks(frame, "template/template/template.obj"))
                    landmark_animation = torch.cat([landmark_animation, landmark.unsqueeze(0)])
                    animation = torch.cat([animation, frame.unsqueeze(0)])

            animations = torch.cat([animations, animation.unsqueeze(0)])
            landmark_animations = torch.cat([landmark_animations, landmark_animation.unsqueeze(0)])
            animation = torch.Tensor([])
            landmark_animation = torch.Tensor([])

        for i, land_anim in enumerate(landmark_animations):
            save = "Landmark_dataset_testing/" + "".join(actual_label) + "_" + str(i + 1) + ".npy"
            np.save(save, land_anim.cpu().numpy())

        return landmark_animations, self.dict_emotions[actual_label], actual_label
