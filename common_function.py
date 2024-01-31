import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image
from Get_landmarks import get_landmarks

label_faces_check = []


def plot_gif(npy, path, name):
    p_save_gif = path + "/" + name + ".gif"

    png_files = []
    for frame_index in range(npy.shape[0]):

        x = np.array([])
        y = np.array([])
        z = np.array([])
        for point in npy[frame_index]:
            x = np.append(x, point[0])
            y = np.append(y, point[1])
            z = np.append(z, point[2])

        p_save = path + "/" + "frame_HD_" + str(frame_index) + ".png"
        png_files.append(p_save)
        plt.tight_layout()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        ax.scatter(x, y, z)
        ax.view_init(90, -90)

        ax.set_xlim(-0.10, 0.10)
        ax.set_ylim(-0.115, 0.08)
        step = 0.04
        n_elements = int((0.10 - (-0.10)) / step) + 1
        ax.set_xticks(np.round(np.linspace(-0.10, 0.10, n_elements), 2))
        step = 0.04
        n_elements = int((0.08 - (-0.115)) / step) + 1
        ax.set_yticks(np.round(np.linspace(-0.115, 0.08, n_elements), 2))

        ax.set_zticks([])

        plt.savefig(p_save, dpi=400, pad_inches=0)
        plt.close()

    frames = []
    for png_file in png_files:
        frame = Image.open(png_file)
        width, height = frame.size
        left = (width // 4) + 15
        top = (height // 4) - 45
        right = 3 * width // 4
        bottom = (3 * height // 4) + 125
        cropped_img = frame.crop((left, top, right, bottom))
        cropped_img.save(png_file)
        frames.append(cropped_img)

    frames[0].save(p_save_gif, format="GIF", append_images=frames[1:], save_all=True, duration=60, loop=0)
    # for file in png_files: os.remove(file)


def setup_folder_testing_validation(folder_input, folder_output):
    gen = folder_input + "/GraphTest"
    for folder in sorted(os.listdir(gen)):
        tmp_path = gen + "/" + folder + "/np"
        file = os.listdir(tmp_path)
        f_name = file[0]
        file = tmp_path + "/" + f_name
        out = folder_output + "/" + f_name.replace("generated_", "")
        shutil.copy(file, out)


def get_actor(landmark_animation, path_gen, actors_coma, name_actors):
    landmarks = landmark_animation.cpu().numpy()
    templates = []
    for i in range(landmarks.shape[0]):
        label = os.path.basename(path_gen[i])
        face = label[label.find("_") + 1:label.find(".")]
        # template = actors_coma[int(face[2:]) - 1]
        if "FaceTalk" in face:
            face = face[face.index("FaceTalk"):]
        id_template = name_actors.index(face)
        template = actors_coma[id_template]
        templates.append(template)
    return torch.Tensor(np.array(templates))


def build_face(output, path_gen, actors_coma, name_actors):
    output = output.cpu().numpy()
    for i in range(output.shape[0]):
        label = os.path.basename(path_gen[i])
        face = label[label.find("_") + 1:label.find(".")]
        # template = actors_coma[int(face[2:]) - 1]
        if "FaceTalk" in face:
            face = face[face.index("FaceTalk"):]
        id_template = name_actors.index(face)
        template = actors_coma[id_template]
        for j in range(output.shape[1]):
            output[i][j] = output[i][j] + template
    return output


def import_actor(path):
    file_list = [e for e in sorted(os.listdir(path))]
    actors = []
    actors_name = []
    for file in file_list:
        mesh = trimesh.load(path + file, process=False)
        actors.append(get_landmarks(mesh.vertices))
        actors_name.append(os.path.splitext(os.path.basename(file))[0])
    return np.asarray(actors), actors_name


def plot_frame(animation, path):
    png_files = []
    for j in range(animation.shape[0]):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for point in animation[j]:
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

        p_save = path + "/frame_" + str(j) + ".png"
        png_files.append(p_save)

        plt.xlim(-0.10, 0.10)
        plt.ylim(-0.115, 0.08)
        step = 0.04
        n_elements = int((0.10 - (-0.10)) / step) + 1
        plt.xticks(np.round(np.linspace(-0.10, 0.10, n_elements), 2))
        step = 0.04
        n_elements = int((0.08 - (-0.115)) / step) + 1
        plt.yticks(np.round(np.linspace(-0.115, 0.08, n_elements), 2))

        plt.savefig(p_save, dpi=300)  # dpi = 300
        plt.close()


def plot_frame_pdf(animation, path):
    png_files = []
    for j in range(animation.shape[0]):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for point in animation[j]:
            x = np.append(x, point[0])
            y = np.append(y, point[1])
            z = np.append(z, point[2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('')
        ax.scatter(x, y, z)
        ax.view_init(90, -90)

        p_save = path + "/frame_" + str(j) + ".pdf"
        png_files.append(p_save)

        plt.xlim(-0.10, 0.10)
        plt.ylim(-0.115, 0.08)
        step = 0.04
        n_elements = int((0.10 - (-0.10)) / step) + 1
        plt.xticks(np.round(np.linspace(-0.10, 0.10, n_elements), 2))
        step = 0.04
        n_elements = int((0.08 - (-0.115)) / step) + 1
        plt.yticks(np.round(np.linspace(-0.115, 0.08, n_elements), 2))
        ax.set_zticks([])
        plt.savefig(p_save, bbox_inches='tight', dpi=3000, format="pdf")  # dpi = 300
        plt.close()

    pass


def plot_frames_bis(animation, path, step_c, name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    p_save = path + "/" + name + ".pdf"
    count = 0
    for j in range(animation.shape[0]):
        if j % step_c == 0:

            print("plotting", j)
            x = np.array([])
            y = np.array([])
            for point in animation[j]:
                x = np.append(x, point[0])
                y = np.append(y, point[1])

            ax = axes[count // 3, count % 3]
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            ax.scatter(x, y)
            # ax.view_init(90, -90)

            ax.set_xlim(-0.10, 0.10)
            ax.set_ylim(-0.115, 0.08)
            step = 0.04
            n_elements = int((0.10 - (-0.10)) / step) + 1
            ax.set_xticks(np.round(np.linspace(-0.10, 0.10, n_elements), 2))
            step = 0.04
            n_elements = int((0.08 - (-0.115)) / step) + 1
            if count % 3 == 0:
                ax.set_yticks(np.round(np.linspace(-0.115, 0.08, n_elements), 2))
            else:
                ax.set_yticks([])
            count += 1
    plt.savefig(p_save, dpi=300)  # dpi = 300

    plt.tight_layout()
    plt.close()


def plot_graph(vector, label, epoch):
    for i in range(vector.shape[0]):
        tmp = os.path.basename(label[i])
        if "FaceTalk" in tmp:
            label_ptr = tmp[:tmp.find("_FaceTalk")]
            label_faces_check.append(label_ptr)
            face = (tmp[tmp.find("_") + 1:-4])
            face = face[face.index("FaceTalk"):]
        else:
            label_ptr = tmp[:tmp.find("_")]
            label_faces_check.append(label_ptr)
            face = (tmp[tmp.find("_") + 1:-4])

        png_files = []

        for j in range(vector.shape[1]):
            x = np.array([])
            y = np.array([])
            z = np.array([])
            for point in vector[i][j]:
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
            if epoch == -1:
                fol = "GraphTest/" + face + "_" + label_ptr
                if not os.path.exists(fol):
                    os.makedirs(fol)
                p_save = fol + '/frame_' + str(j) + '.png'
            else:
                p_save = 'GraphTrain/epoch_' + str(epoch + 1) + '_' + label_ptr + '_frame_' + str(
                    j) + '.png'

            png_files.append(p_save)
            plt.xlim(-0.10, 0.10)
            plt.ylim(-0.115, 0.08)
            step = 0.04
            n_elements = int((0.10 - (-0.10)) / step) + 1
            plt.xticks(np.round(np.linspace(-0.10, 0.10, n_elements), 2))
            step = 0.04
            n_elements = int((0.08 - (-0.115)) / step) + 1
            plt.yticks(np.round(np.linspace(-0.115, 0.08, n_elements), 2))

            plt.savefig(p_save, dpi=300, bbox_inches='tight', transparent="True", pad_inches=0)
            plt.close()

        if epoch == -1:
            fol_np = 'GraphTest/' + face + "_" + label_ptr + '/np'
            if not os.path.exists(fol_np):
                os.makedirs(fol_np)
            file_np = fol_np + '/generated_' + tmp[:-4] + '.npy'
            np.save(file_np, vector[i])

        frames = []
        for png_file in png_files:
            frame = Image.open(png_file)
            frames.append(frame)
        if epoch == -1:
            gif_filename = 'GraphTest/' + face + "_" + label_ptr + '/' + face + "_" + label_ptr + '.gif'
        else:
            gif_filename = 'GraphTrain/epoch' + str(epoch + 1) + "_" + label_ptr + "_" + face + '.gif'
        frames[0].save(gif_filename, format="GIF", append_images=frames[1:], save_all=True, duration=60, loop=0)
        for file in png_files: os.remove(file)
