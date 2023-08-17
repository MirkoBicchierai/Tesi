import trimesh
from Get_landmarks import get_landmarks
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def get_folder(path):
    folders = []
    for f in os.listdir(path):
        real_f = os.path.join(path, f)
        folders = np.append(folders, [os.path.join(real_f, sf) for sf in os.listdir(real_f)])
    return folders


def plot_graph(vector, label_ptr, face, main_fold):
    png_files = []
    for j in range(vector.shape[0]):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for point in vector[j]:
            x = np.append(x, point[0])
            y = np.append(y, point[1])
            z = np.append(z, point[2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(x, y, z)
        ax.view_init(90, -90)

        p_save = main_fold + "Animation" + '/frame_' + str(j) + '.png'
        png_files.append(p_save)

        plt.xlim(-0.10, 0.10)
        plt.ylim(-0.115, 0.08)
        step = 0.04
        n_elements = int((0.10 - (-0.10)) / step) + 1
        plt.xticks(np.round(np.linspace(-0.10, 0.10, n_elements), 2))
        step = 0.04
        n_elements = int((0.08 - (-0.115)) / step) + 1
        plt.yticks(np.round(np.linspace(-0.115, 0.08, n_elements), 2))

        plt.savefig(p_save, dpi=300)
        plt.close()

    frames = []
    for png_file in png_files:
        frame = Image.open(png_file)
        frames.append(frame)

    gif_filename = main_fold + 'Animation/' + face + "_" + label_ptr + '.gif'
    frames[0].save(gif_filename, format="GIF", append_images=frames[1:], save_all=True, duration=60, loop=0)
    for file in png_files: os.remove(file)


def save_array(path, save, fold, do_graph):
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
        if do_graph:
            plot_graph(landmarks, "".join(actual_label), actual_face, fold)


def main():
    # Complete Partial
    ty = "Partial2"
    # Landmark_dataset_flame Landmark_dataset_flame_aligned
    folder = "Landmark_dataset_flame_aligned/"
    # Dataset_FLAME Dataset_FLAME_Aligned
    folder_dataset = "Dataset_FLAME_Aligned/"

    plot = False

    print("----- START TRAINING DATASET -----")
    filelist = [f for f in os.listdir(folder + "dataset_training/" + ty + "/")]
    for f in filelist:
        os.remove(os.path.join(folder + "dataset_training/" + ty + "/", f))
    save_array(folder_dataset + "dataset_training/" + ty, folder + "dataset_training/" + ty, folder, plot)
    print("----- START TESTING DATASET -----")
    filelist = [f for f in os.listdir(folder + "/dataset_testing/" + ty + "/")]
    for f in filelist:
        os.remove(os.path.join(folder + "dataset_testing/" + ty + "/", f))
    save_array(folder_dataset + "dataset_testing/" + ty, folder + "dataset_testing/" + ty, folder, plot)


if __name__ == "__main__":
    main()
