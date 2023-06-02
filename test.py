import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import FastDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

label_faces_check = {}


def plot_graph(vector, label, epoch):
    for i in range(vector.shape[0]):
        if label[i] not in label_faces_check:
            label_faces_check[label[i]] = 1
        else:
            label_faces_check[label[i]] = label_faces_check[label[i]] + 1

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
                p_save = "Graph/" + label[i] + '_frame_' + str(j) + '.png'
            else:
                p_save = "Graph/" + 'epoch_' + str(epoch + 1) + '_' + label[i] + '_frame_' + str(j) + '.png'

            png_files.append(p_save)
            plt.xlim(-0.2, 0.2)
            plt.ylim(-0.2, 0.2)
            plt.savefig(p_save)
            plt.close()

        frames = []
        for png_file in png_files:
            frame = Image.open(png_file)
            frames.append(frame)
        if epoch == -1:
            gif_filename = 'Graph/' + label[i] + '.gif'
        else:
            gif_filename = 'Graph/epoch' + str(epoch + 1) + "_" + label[i] + '.gif'
        frames[0].save(gif_filename, format="GIF", append_images=frames[1:], save_all=True, duration=120, loop=0)


def main():
    filelist = [f for f in os.listdir("Graph/")]
    for f in filelist:
        os.remove(os.path.join("Graph/", f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = "Landmark_dataset/dataset_testing/Partial"
    save_path = "Models/modelPartial.pt"  # modelComplete.pt

    dataset_test = FastDataset(test_path)
    testing_dataloader = DataLoader(dataset_test, batch_size=10, shuffle=True, drop_last=False)

    model = torch.load(save_path)
    model.eval()
    for landmark_animation, label, str_label in tqdm(testing_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            output = model(landmark_animation[:, 0], label, 60)
            plot_graph(output.cpu().numpy(), str_label, epoch=-1)

    print(label_faces_check)


if __name__ == "__main__":
    main()
