import shutil
import trimesh
from Get_landmarks import get_landmarks
import numpy as np
import os


def main():
    data_path = "Dataset_FLAME_Aligned_COMA/DATASET COMPLETO (copia)"

    shutil.rmtree("Landmark_dataset_flame_aligned_coma/Completo-Mod/", ignore_errors=False, onerror=None)
    os.makedirs("Landmark_dataset_flame_aligned_coma/Completo-Mod/")

    for subdir in os.listdir(data_path):
        os.makedirs("Landmark_dataset_flame_aligned_coma/Completo-Mod/" + subdir + "/")
        for expr_dir in os.listdir(os.path.join(data_path, subdir)):

            list_dir = sorted(os.listdir(os.path.join(data_path, subdir, expr_dir)))
            landmarks = []
            for i in range(len(list_dir)):
                data_loaded = trimesh.load(os.path.join(data_path, subdir, expr_dir, list_dir[i]), process=False)
                landmarks.append(get_landmarks(data_loaded.vertices))

            landmarks = np.array(landmarks)
            np.save(
                'Landmark_dataset_flame_aligned_coma/Completo-Mod/' + subdir + "/" + expr_dir + ".npy",
                landmarks)


if __name__ == '__main__':
    main()
