import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import FastDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

label_faces_check = []


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
                # ax.set_zticks([])
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


def main():
    shutil.rmtree("GraphTest/", ignore_errors=False, onerror=None)
    os.makedirs("GraphTest/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = "Landmark_dataset_flame_aligned/dataset_testing/Partial2"
    save_path = "Models/model_7500_l2_e4.pt"

    aligned = True

    dataset_test = FastDataset(test_path)
    testing_dataloader = DataLoader(dataset_test, batch_size=5, shuffle=False, drop_last=False)

    model = torch.load(save_path)
    model.eval()
    for landmark_animation, label, path_gen in tqdm(testing_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            output = model(landmark_animation[:, 0], label, 60)
            plot_graph(output.cpu().numpy(), path_gen, -1, aligned)

    print(label_faces_check)


if __name__ == "__main__":
    main()
