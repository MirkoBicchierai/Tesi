import shutil
import trimesh
from Get_landmarks import get_landmarks
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def plot_graph(vector, label_ptr, face, main_fold, aligned):
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

        if aligned:
            plt.xlim(-0.10, 0.10)
            plt.ylim(-0.115, 0.08)
            step = 0.04
            n_elements = int((0.10 - (-0.10)) / step) + 1
            plt.xticks(np.round(np.linspace(-0.10, 0.10, n_elements), 2))
            step = 0.04
            n_elements = int((0.08 - (-0.115)) / step) + 1
            plt.yticks(np.round(np.linspace(-0.115, 0.08, n_elements), 2))
        else:
            plt.xlim(-0.115, 0.115)
            plt.ylim(-0.115, 0.115)

        plt.savefig(p_save, dpi=300)
        plt.close()

    frames = []
    for png_file in png_files:
        frame = Image.open(png_file)
        frames.append(frame)

    gif_filename = main_fold + 'Animation/' + face + "_" + label_ptr + '.gif'
    frames[0].save(gif_filename, format="GIF", append_images=frames[1:], save_all=True, duration=60, loop=0)
    for file in png_files: os.remove(file)


def main():
    aligned = True
    folder_plot = "Landmark_dataset_flame_aligned_coma/"

    data_path = "sampling/"
    shutil.rmtree("Landmark_dataset_flame_aligned_coma/Completo/", ignore_errors=False, onerror=None)
    os.makedirs("Landmark_dataset_flame_aligned_coma/Completo/")

    for subdir in os.listdir(data_path):
        # os.makedirs("Landmark_dataset_flame_aligned_coma/Completo/" + subdir + "/")
        for expr_dir in os.listdir(os.path.join(data_path, subdir)):

            list_dir = sorted(os.listdir(os.path.join(data_path, subdir, expr_dir)))
            landmarks = []
            for i in range(len(list_dir)):
                data_loaded = trimesh.load(os.path.join(data_path, subdir, expr_dir, list_dir[i]), process=False)
                landmarks.append(get_landmarks(data_loaded.vertices))

            landmarks = np.array(landmarks)
            np.save(
                'Landmark_dataset_flame_aligned_coma/Completo/' + expr_dir + "_" + subdir + ".npy",
                landmarks)
            # plot_graph(landmarks, expr_dir, subdir, folder_plot, aligned)


if __name__ == '__main__':
    main()
