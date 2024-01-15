import argparse
import os
import shutil
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

from Classification.inting import get_distr_real, get_distr_acc_generated, compute_fid
from DataLoader import FastDataset
from Demo_Only_Meshes import Get_landmarks
from Demo_Only_Meshes.dest import elaborate_landmarks

from Demo_Only_Meshes.new_demo_only_meshes import generate_meshes_from_landmarks

from Model import DecoderRNN
# from torch.utils.tensorboard import SummaryWriter
from common_function import plot_graph, import_actor, build_face
import warnings

warnings.filterwarnings("ignore")

seed_value = 27
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    coma = True
    loss_l2 = False
    if loss_l2:
        loss_ty = "L2"
    else:
        loss_ty = "L1"
    if coma:
        train_path = "Landmark_dataset_flame_aligned_coma/dataset_training"
        test_path = "Landmark_dataset_flame_aligned_coma/dataset_testing"
        actors_path = "Actors_Coma/"
        type_dataset = "COMA"
        hidden_size = 512
        num_classes = 12
        output_size = (68 * 3)
        frame_generate = 40
        lr = 1e-5
        epochs = 1000
    else:
        train_path = "Landmark_dataset_flame_aligned/dataset_training/Partial2"
        test_path = "Landmark_dataset_flame_aligned/dataset_testing/Partial2"
        actors_path = "Actors/"
        type_dataset = "COMA_Florence"
        hidden_size = 1024
        num_classes = 10
        output_size = (68 * 3)
        frame_generate = 60
        lr = 1e-4
        epochs = 1200

    save_path = "Models/model_" + loss_ty + "_" + str(epochs) + "_" + str(lr) + "_" + str(
        hidden_size) + "_" + type_dataset + "_DiffSplit.pt"

    actors_coma, name_actors_coma = import_actor(path=actors_path)
    # writer = SummaryWriter("TensorBoard/" + loss_ty + "_" + str(epochs) + "_" + str(lr) + "_" + str(
    #     hidden_size) + "_" + type_dataset + "_DiffSplit_" + datetime.now().strftime("%m-%d-%Y_%H:%M"))

    shutil.rmtree("GraphTrain/", ignore_errors=False, onerror=None)
    os.makedirs("GraphTrain/")

    m_real, cov_real = get_distr_real()

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
                if loss_l2:
                    loss = F.mse_loss(output, landmark_animation[:, 1 + idF:])
                else:
                    loss = F.l1_loss(output, landmark_animation[:, 1 + idF:])
                tot_loss += loss.item()
                loss.backward()
                optimizer.step()

        # writer.add_scalar('Loss/' + type_dataset + '/train', tot_loss / len(training_dataloader), epoch + 1)

        if not (epoch + 1) % 10:
            print("Epoch: ", epoch + 1, " - Training loss: ", tot_loss / len(training_dataloader))

        if not (epoch ) % 50:
            tot_loss_test = 0
            model.eval()
            check = True

            generated = torch.Tensor([]).to(device)
            labels = torch.Tensor([]).to(device)
            print("Starting Testing")
            count = 0
            mean_err = []
            for landmark_animation, label, path_gen in testing_dataloader:
                landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)

                with torch.no_grad():
                    output = model(landmark_animation[:, 0], label, frame_generate)

                    if check:
                        plot_graph(build_face(output, path_gen, actors_coma, name_actors_coma), path_gen, epoch)
                    check = False
                if loss_l2:
                    test_loss = F.mse_loss(output, landmark_animation[:, 1:])
                else:
                    test_loss = F.l1_loss(output, landmark_animation[:, 1:])
                tot_loss_test += test_loss.item()

                label = label.to(device)
                generated = torch.cat([generated, output])
                lamda = 1000
                if count < 5:
                    count += 1
                    mesh_generated = generate_mesh(build_face(output, path_gen, actors_coma, name_actors_coma),
                                                   path_gen[0]) * lamda
                    mesh_real = generate_mesh(
                        build_face(landmark_animation[:, 1:], path_gen, actors_coma, name_actors_coma),
                        path_gen[0]) * lamda

                    mean_err.append(np.mean(np.sqrt(np.sum((mesh_generated - mesh_real) ** 2, axis=2))))

                labels = torch.cat([labels, label])

            m_fake, cov_fake, acc = get_distr_acc_generated(generated, labels)

            mean_err = np.array(mean_err)
            mean_err = mean_err.mean()

            fid = compute_fid(m_fake, cov_fake, m_real, cov_real)

            # writer.add_scalar('Loss/' + type_dataset + '/validation', tot_loss_test / len(testing_dataloader),
            #                   epoch + 1)
            print("Epoch: ", epoch + 1, " - Testing loss: ", tot_loss_test / len(testing_dataloader))
            print("Accuracy=", acc, "FID=", fid, "mm error=", mean_err)
            model.train()

    torch.save(model, save_path)

    # writer.flush()
    # writer.close()


if __name__ == "__main__":
    main()
