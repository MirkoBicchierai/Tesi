import shutil
import trimesh
import numpy as np
import os
from Get_landmarks import get_landmarks
from tqdm import tqdm


def main():
    division = {"FaceTalk_170725_00137_TA": "TRAIN",
                "FaceTalk_170728_03272_TA": "TRAIN", "FaceTalk_170731_00024_TA": "TRAIN",
                "FaceTalk_170809_00138_TA": "TRAIN", "FaceTalk_170811_03274_TA": "TRAIN",
                "FaceTalk_170811_03275_TA": "TRAIN", "FaceTalk_170904_00128_TA": "TRAIN",
                "FaceTalk_170904_03276_TA": "TRAIN", "FaceTalk_170908_03277_TA": "TRAIN",
                "FaceTalk_170912_03278_TA": "TEST", "FaceTalk_170913_03279_TA": "TEST",
                "FaceTalk_170915_00223_TA": "TEST"}

    data_path = "sampling/"
    shutil.rmtree("Landmark_dataset_flame_aligned_coma/30frame/dataset_testing/", ignore_errors=False, onerror=None)
    os.makedirs("Landmark_dataset_flame_aligned_coma/30frame/dataset_testing/")

    shutil.rmtree("Landmark_dataset_flame_aligned_coma/30frame/dataset_training/", ignore_errors=False, onerror=None)
    os.makedirs("Landmark_dataset_flame_aligned_coma/30frame/dataset_training/")

    for subdir in tqdm(os.listdir(data_path)):
        for expr_dir in os.listdir(os.path.join(data_path, subdir)):
            expression = expr_dir.replace("_", "-")
            if division[subdir] == "TRAIN":
                tmp_path = 'Landmark_dataset_flame_aligned_coma/30frame/dataset_training/' + expression + "_" + subdir + ".npy"
            else:
                tmp_path = 'Landmark_dataset_flame_aligned_coma/30frame/dataset_testing/' + expression + "_" + subdir + ".npy"

            list_dir = sorted(os.listdir(os.path.join(data_path, subdir, expr_dir)))
            landmarks = []
            for i in range(len(list_dir)):
                data_loaded = trimesh.load(os.path.join(data_path, subdir, expr_dir, list_dir[i]), process=False)
                landmarks.append(get_landmarks(data_loaded.vertices))

            landmarks = np.array(landmarks)
            np.save(tmp_path, landmarks)


if __name__ == '__main__':
    main()
