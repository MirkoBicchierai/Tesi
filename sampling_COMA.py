import shutil
import os
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd


def copy_file(src_folder, dst_folder, file_name):
    src_path = os.path.join(src_folder, file_name)
    dst_path = os.path.join(dst_folder, file_name)
    shutil.copyfile(src_path, dst_path)


def interpolate_frames(frame1, frame2, num_interpolated_frames):
    t_values = np.linspace(0, 1, num=num_interpolated_frames + 2)[1:-1]  # Exclude endpoints
    interpolated_frames = [(1 - t) * frame1 + t * frame2 for t in t_values]
    return interpolated_frames


def downsample_animation(input_folder, output_folder, target_frames, n_interpolation):
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

        interpolated_frames = interpolate_frames(frame1_points, frame2_points, num_interpolated_frames=n_interpolation)

        for i, frame_points in enumerate(interpolated_frames):
            output_path = os.path.join(output_folder, f"interpolated_{target_idx * 6 + i:03d}.ply")
            downsampled_cloud = PyntCloud(pd.DataFrame(frame_points, columns=['x', 'y', 'z']))
            downsampled_cloud.to_file(output_path)


def main():
    # Dataset_FLAME_Aligned_COMA/DATASET COMPLETO/FaceTalk_170811_03275_TA/mouth_extreme
    input_folder = "Dataset_FLAME_Aligned_COMA/DATASET COMPLETO/"
    output_folder = "sampling/"
    tmp_out_ = "tmp/"
    folders = os.listdir(input_folder)

    shutil.rmtree(output_folder, ignore_errors=False, onerror=None)
    os.makedirs(output_folder)

    shutil.rmtree(tmp_out_, ignore_errors=False, onerror=None)
    os.makedirs(tmp_out_)

    for fold in folders:
        os.makedirs(output_folder + fold + "/")
        labels = os.listdir(input_folder + fold + "/")
        for label in labels:
            shutil.rmtree(tmp_out_, ignore_errors=False, onerror=None)
            os.makedirs(tmp_out_)

            tmp_out = output_folder + fold + "/" + label + "/"
            tmp_in = input_folder + fold + "/" + label + "/"
            os.makedirs(tmp_out)
            tmp_num_frame = len(os.listdir(tmp_in))

            if tmp_num_frame < 41:
                target_frames = tmp_num_frame
                n = 4
                downsample_animation(tmp_in, tmp_out_, target_frames, n_interpolation=n)
                tmp_in = tmp_out_

            target_frames = 41
            n = 1
            downsample_animation(tmp_in, tmp_out, target_frames, n_interpolation=n)


if __name__ == "__main__":
    main()
