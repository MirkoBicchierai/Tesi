import os
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image
from Get_landmarks import get_landmarks

label_faces_check = []


def build_face(output, path_gen, actors_coma, name_actors):
    output = output.cpu().numpy()
    for i in range(output.shape[0]):
        label = os.path.basename(path_gen[i])
        face = label[label.find("_") + 1:label.find(".")]
        # template = actors_coma[int(face[2:]) - 1]
        id_template = name_actors.index(face)
        template = actors_coma[id_template]
        for j in range(output.shape[1]):
            output[i][j] = output[i][j] + template
    return output


# def import_actor(path):
#     file_list = [path + e for e in sorted(os.listdir(path))]
#     actors = []
#     for file in file_list:
#         mesh = trimesh.load(file, process=False)
#         actors.append(get_landmarks(mesh.vertices))
#     return np.asarray(actors)

def import_actor(path):
    file_list = [e for e in sorted(os.listdir(path))]
    actors = []
    actors_name = []
    for file in file_list:
        mesh = trimesh.load(path + file, process=False)
        actors.append(get_landmarks(mesh.vertices))
        actors_name.append(os.path.splitext(os.path.basename(file))[0])
    return np.asarray(actors), actors_name

def plot_graph(vector, label, epoch, aligned):
    for i in range(vector.shape[0]):
        tmp = os.path.basename(label[i])
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
            gif_filename = 'GraphTrain/epoch' + str(epoch + 1) + "_" + label_ptr + '.gif'
        frames[0].save(gif_filename, format="GIF", append_images=frames[1:], save_all=True, duration=60, loop=0)
        for file in png_files: os.remove(file)
