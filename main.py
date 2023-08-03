import os
import shutil
import random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import FastDataset
from Model import DecoderRNN
from torch.utils.tensorboard import SummaryWriter
from common_function import plot_graph, import_actor, build_face

seed_value = 27
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligned = True

    coma = True

    if coma:
        train_path = "Landmark_dataset_flame_aligned_coma/dataset_training"
        test_path = "Landmark_dataset_flame_aligned_coma/dataset_testing"
        actors_path = "Actors_Coma/"
        hidden_size = 1024
        num_classes = 12
        output_size = (68 * 3)
        frame_generate = 29
        lr = 1e-3
        epochs = 1200
    else:
        train_path = "Landmark_dataset_flame_aligned/dataset_training/Partial2"
        test_path = "Landmark_dataset_flame_aligned/dataset_testing/Partial2"
        actors_path = "Actors/"
        hidden_size = 1024
        num_classes = 10
        output_size = (68 * 3)
        frame_generate = 60
        lr = 1e-4
        epochs = 1200

    save_path = "Models/model_" + str(epochs) + "_" + str(lr) + "_" + str(hidden_size) + "_COMA.pt"

    actors_coma, name_actors_coma = import_actor(path=actors_path)
    writer = SummaryWriter("TensorBoard/LABEL:" + str(num_classes) + "_HIDDEN-SIZE:" + str(
        hidden_size) + "_LR:" + str(lr) + "_COMA_" + datetime.now().strftime("%m-%d-%Y_%H:%M"))

    shutil.rmtree("GraphTrain/", ignore_errors=False, onerror=None)
    os.makedirs("GraphTrain/")

    dataset_train = FastDataset(train_path, actors_coma, name_actors_coma)
    training_dataloader = DataLoader(dataset_train, batch_size=25, shuffle=True, drop_last=False, pin_memory=True,
                                     num_workers=5)
    dataset_test = FastDataset(test_path, actors_coma, name_actors_coma)
    testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)

    model = DecoderRNN(hidden_size, output_size, num_classes, frame_generate, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        tot_loss = 0
        for landmark_animation, label, path_gen in training_dataloader:
            for idF in reversed(range(landmark_animation[:, 1:].shape[1])):
                optimizer.zero_grad()
                landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
                output = model(landmark_animation[:, idF], label, frame_generate - idF)
                loss = F.mse_loss(output, landmark_animation[:, 1 + idF:])
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
                    output = model(landmark_animation[:, 0], label, frame_generate)
                    if check:
                        plot_graph(build_face(output, path_gen, actors_coma, name_actors_coma), path_gen, epoch,
                                   aligned)
                    check = False
                test_loss = F.mse_loss(output, landmark_animation[:, 1:])
                tot_loss_test += test_loss.item()

            writer.add_scalar('Loss/validation', tot_loss_test / len(testing_dataloader), epoch + 1)
            print("Epoch: ", epoch + 1, " - Testing loss: ", tot_loss_test / len(testing_dataloader))
            model.train()

    torch.save(model, save_path)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
