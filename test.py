import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import SingleEmotionDataset
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(vector, label):
    for i in range(vector.shape[0]):
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
            plt.savefig("Grafici/" + 'faccia_' + str(i) + '_' + ''.join(label) + '_frame_' + str(j) + '.png')
            plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = "dataset/SingleExpression/COMA/Testing"
    save_path = "Models/model.pt"

    dataset_test = SingleEmotionDataset(test_path)
    testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True)

    model = torch.load(save_path)
    model.eval()
    for landmark_animation, label, str_label in tqdm(testing_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        landmark_animation = landmark_animation.squeeze(0)
        with torch.no_grad():
            output = model(landmark_animation[:, 0], label)
            output_cpu = output.cpu()
            plot_graph(output_cpu.numpy(), str_label)


if __name__ == "__main__":
    main()
