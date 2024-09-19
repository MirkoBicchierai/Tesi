import os
import numpy as np
from Get_landmarks import get_landmarks
import trimesh
from tqdm import tqdm


input_folder = "TrainCOMA/"
output_folder = "datasetCOMA/"

folders = os.listdir(input_folder)

for fold in tqdm(folders):
    labels = os.listdir(input_folder + fold + "/")
    for label in labels:
        tmp_out = output_folder + fold + "/" + label + "/"
        list_dir = sorted(os.listdir(input_folder + fold + "/" + label + "/"))
        landmarks = []
        for i in range(len(list_dir)):
            data_loaded = trimesh.load(os.path.join(input_folder, fold, label, list_dir[i]), process=False)
            landmarks.append(get_landmarks(data_loaded.vertices))

        tmp_path = output_folder + label + "_" + fold + ".npy"
        landmarks = np.array(landmarks)
        np.save(tmp_path, landmarks)
