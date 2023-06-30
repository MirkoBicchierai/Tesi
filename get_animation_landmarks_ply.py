import os
import numpy as np
import trimesh
from tqdm import tqdm

from Get_landmarks import get_landmarks


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

        landmarks = []
        for i in range(61):
            mesh = trimesh.load(os.path.join(f, "".join(actual_label) + '_' + str(i).zfill(2) + '.ply'), process=False)
            landmarks.append(get_landmarks(mesh.vertices))
        landmarks = np.array(landmarks)

        file = save + "/" + "".join(actual_label) + "_" + actual_face + ".npy"
        np.save(file, landmarks)


def main():
    # Complete Partial
    ty = "Complete"
    print("----- START TRAINING DATASET -----")
    filelist = [f for f in os.listdir("Landmark_dataset_flame/dataset_training/"+ty+"/")]
    for f in filelist:
        os.remove(os.path.join("Landmark_dataset_flame/dataset_training/"+ty+"/", f))
    save_array("Dataset_FLAME/dataset_training/"+ty, "Landmark_dataset_flame/dataset_training/"+ty)
    print("----- START TESTING DATASET -----")
    filelist = [f for f in os.listdir("Landmark_dataset_flame/dataset_testing/"+ty+"/")]
    for f in filelist:
        os.remove(os.path.join("Landmark_dataset_flame/dataset_testing/"+ty+"/", f))
    save_array("Dataset_FLAME/dataset_testing/"+ty, "Landmark_dataset_flame/dataset_testing/"+ty)


if __name__ == "__main__":
    main()
