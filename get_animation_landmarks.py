import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from get_landmarks import get_landmarks
import pytorch3d.io
import rarfile


def unzip_all_files(folder_path, destination):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if file.endswith('.rar'):
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                rar_ref.extractall(destination)


def unzip():
    path_zip = "dataset/SingleExpression/COMA/Complete_Zip"
    path_unzip = "dataset/SingleExpression/COMA/Complete_unZip"
    filelist = [f for f in os.listdir(path_unzip)]
    for f in filelist:
        shutil.rmtree(os.path.join(path_unzip, f), ignore_errors=False, onerror=None)

    for f in tqdm(os.listdir(path_zip)):
        os.mkdir(os.path.join(path_unzip, f))
        unzip_all_files(os.path.join(path_zip, f), os.path.join(path_unzip, f))


def get_folder(path):
    folders = []
    for f in os.listdir(path):
        real_f = os.path.join(path, f)
        folders = np.append(folders, [os.path.join(real_f, sf) for sf in os.listdir(real_f)])
    return folders


def save_array(path, save):
    folders = get_folder(path)
    for f in tqdm(folders):
        actual_label = os.path.basename(os.path.normpath(f))
        actual_face = (f.split(os.sep)[-2]).split("_")[-1]
        landmark_animation = torch.Tensor([])
        landmark_animations = torch.Tensor([])
        file_list = sorted([f.decode('utf-8') for f in os.listdir(f)])
        for file_name in file_list:
            if file_name[-3:] == 'obj':
                frame, _, _ = pytorch3d.io.load_obj(os.path.join(f, file_name), load_textures=False)
                landmark = torch.tensor(get_landmarks(frame, "template/template/template.obj"))
                landmark_animation = torch.cat([landmark_animation, landmark.unsqueeze(0)])
        landmark_animations = torch.cat([landmark_animations, landmark_animation.unsqueeze(0)])

        for j, land_anim in enumerate(landmark_animations):
            file = save + "/" + "".join(actual_label) + "_" + actual_face + ".npy"
            np.save(file, land_anim.cpu().numpy())


def main(z):
    if z:
        unzip()

    filelist = [f for f in os.listdir("Landmark_dataset/dataset_training/Complete/")]
    for f in filelist:
        os.remove(os.path.join("Landmark_dataset/dataset_training/Complete/", f))
    save_array("dataset/SingleExpression/COMA/Complete_Train", "Landmark_dataset/dataset_training/Complete")

    filelist = [f for f in os.listdir("Landmark_dataset/dataset_testing/Complete/")]
    for f in filelist:
        os.remove(os.path.join("Landmark_dataset/dataset_testing/Complete/", f))
    save_array("dataset/SingleExpression/COMA/Complete_Test", "Landmark_dataset/dataset_testing/Complete")


if __name__ == "__main__":
    main(z=False)
