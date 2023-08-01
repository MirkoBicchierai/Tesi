import shutil

import pandas as pd
from pyntcloud import PyntCloud
import os
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image
from Get_landmarks import get_landmarks


def copy_file(src_folder, dst_folder, file_name):
    src_path = os.path.join(src_folder, file_name)
    dst_path = os.path.join(dst_folder, file_name)
    shutil.copyfile(src_path, dst_path)


import os
import numpy as np
from pyntcloud import PyntCloud


def interpolate_frames(frame1, frame2, num_interpolated_frames):
    t_values = np.linspace(0, 1, num=num_interpolated_frames + 2)[1:-1]  # Exclude endpoints
    interpolated_frames = [(1 - t) * frame1 + t * frame2 for t in t_values]
    return interpolated_frames


def downsample_animation(input_folder, output_folder, target_frames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    sorted_files = sorted([f for f in files if f.endswith(".ply")])
    num_frames = len(sorted_files)

    if num_frames < 2:
        raise ValueError("The number of original frames must be at least 2 for interpolation.")

    frame_step = num_frames // (target_frames)

    for target_idx in range(target_frames):
        frame1_idx = target_idx * frame_step
        frame2_idx = min((target_idx + 1) * frame_step, num_frames - 1)

        frame1_file = sorted_files[frame1_idx]
        frame2_file = sorted_files[frame2_idx]

        frame1_path = os.path.join(input_folder, frame1_file)
        frame2_path = os.path.join(input_folder, frame2_file)

        frame1_points = np.array(PyntCloud.from_file(frame1_path).points)
        frame2_points = np.array(PyntCloud.from_file(frame2_path).points)

        interpolated_frames = interpolate_frames(frame1_points, frame2_points, num_interpolated_frames=1)

        for i, frame_points in enumerate(interpolated_frames):
            output_path = os.path.join(output_folder, f"interpolated_{target_idx * 6 + i:03d}.ply")
            downsampled_cloud = PyntCloud(pd.DataFrame(frame_points, columns=['x', 'y', 'z']))
            downsampled_cloud.to_file(output_path)


def main():
    input_folder = "Dataset_FLAME_Aligned_COMA/DATASET COMPLETO/FaceTalk_170725_00137_TA/eyebrow"
    output_folder = "test"
    target_frames = 30  # The number of frames in the downsampled animation
    downsample_animation(input_folder, output_folder, target_frames)


def plot_graph(vector):
    png_files = []
    for i in range(vector.shape[0]):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for j in range(vector.shape[1]):
            x = np.append(x, vector[i][j][0])
            y = np.append(y, vector[i][j][1])
            z = np.append(z, vector[i][j][2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(x, y, z)
        ax.view_init(90, -90)

        fol = "test_landmark/"
        if not os.path.exists(fol):
            os.makedirs(fol)
        p_save = fol + '/frame_' + str(i) + '.png'

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

        fol_np = 'test_landmark/np'
        if not os.path.exists(fol_np):
            os.makedirs(fol_np)
        file_np = fol_np + '/generated_' + str(j) + '.npy'
        np.save(file_np, vector[i])

    frames = []
    for png_file in png_files:
        frame = Image.open(png_file)
        frames.append(frame)

    gif_filename = 'test_landmark/gen.gif'
    frames[0].save(gif_filename, format="GIF", append_images=frames[1:], save_all=True, duration=90, loop=0)


def main2():
    main_fold = "test"
    landmarks = []
    for file in sorted(os.listdir(main_fold)):
        data_loaded = trimesh.load(os.path.join(main_fold, file), process=False)
        landmarks.append(get_landmarks(data_loaded.vertices))

    landmarks = np.array(landmarks)
    print(landmarks.shape)
    np.save(
        'test_landmark/test.npy',
        landmarks)
    plot_graph(landmarks)


if __name__ == "__main__":
    main()
    main2()
