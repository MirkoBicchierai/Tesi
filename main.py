import os
import shutil
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from DataLoader import FastDataset
from Model import DecoderRNN
from test import plot_graph
from torch.utils.tensorboard import SummaryWriter
import random

seed_value = 27
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = "Landmark_dataset_flame/dataset_training/Partial"
    test_path = "Landmark_dataset_flame/dataset_testing/Partial"
    save_path = "Models/modelPartial5000_partial1.pt"

    hidden_size = 512  # 1024
    num_classes = 10  # number of label (Partial 10 Complete 70)
    output_size = (68 * 3)  # landmark point * coordinates
    frame_generate = 60  # number of frame generate by lstm
    lr = 1e-4  # learning rate

    writer = SummaryWriter("TensorBoard/LABEL:" + str(num_classes) + "_HIDDEN-SIZE:" + str(
        hidden_size) + "_LR:" + str(lr) + "_" + datetime.now().strftime("%m-%d-%Y_%H:%M"))

    shutil.rmtree("GraphTrain/", ignore_errors=False, onerror=None)
    os.makedirs("GraphTrain/")

    dataset_train = FastDataset(train_path)
    training_dataloader = DataLoader(dataset_train, batch_size=25, shuffle=True, drop_last=False, pin_memory=True,
                                     num_workers=5)
    dataset_test = FastDataset(test_path)
    testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)

    model = DecoderRNN(hidden_size, output_size, num_classes, frame_generate, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 5000
    for epoch in tqdm(range(epochs)):
        tot_loss = 0
        for landmark_animation, label, path_gen in training_dataloader:
            for idF in reversed(range(landmark_animation[:, 1:].shape[1])):
                optimizer.zero_grad()
                landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
                output = model(landmark_animation[:, idF], label, 60 - idF)
                loss = F.mse_loss(output, landmark_animation[:, 1 + idF:])
                # loss = F.l1_loss(output, landmark_animation[:, 1 + idF:])
                tot_loss += loss.item()
                loss.backward()
                optimizer.step()

            writer.add_scalar('Loss/train', tot_loss / len(training_dataloader), epoch + 1)

        if not (epoch + 1) % 10:
            print("Epoch: ", epoch + 1, " - Training loss: ", tot_loss / len(training_dataloader))

        if not (epoch + 1) % 50:
            tot_loss_test = 0
            model.eval()
            check = True
            for landmark_animation, label, path_gen in testing_dataloader:
                landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
                with torch.no_grad():
                    output = model(landmark_animation[:, 0], label, 60)
                    if check:
                        plot_graph(output.cpu().numpy(), path_gen, epoch)
                    # check = False
                test_loss = F.mse_loss(output, landmark_animation[:, 1:])
                # test_loss = F.l1_loss(output, landmark_animation[:, 1:])
                tot_loss_test += test_loss.item()

            writer.add_scalar('Loss/validation', tot_loss_test / len(testing_dataloader), epoch + 1)
            print("Epoch: ", epoch + 1, " - Testing loss: ", tot_loss_test / len(testing_dataloader))
            model.train()

    torch.save(model, save_path)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
