import torch
import random
import numpy as np
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Classification.Model import ClassificationRNN
from DataLoader import FastDataset
from common_function import import_actor

seed_value = 27
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coma = True
    sampling_dataset = True

    if coma:
        if sampling_dataset:
            type_dataset = "COMA"
            train_path = "../Landmark_dataset_flame_aligned_coma/dataset_training"
            test_path = "../Landmark_dataset_flame_aligned_coma/dataset_testing"
            batch_train = 25
            batch_test = 20
        else:
            type_dataset = "COMA_FULL_FRAME"
            train_path = "../Landmark_dataset_flame_aligned_coma/FULL_FRAME/dataset_training"
            test_path = "../Landmark_dataset_flame_aligned_coma/FULL_FRAME/dataset_testing"
            batch_train = 1
            batch_test = 1
        actors_path = "../Actors_Coma/"
        lr = 1e-5
        epochs = 2000
        hidden_size = 1024
        num_classes = 12
        input_size = (68 * 3)
        layers = 3
    else:
        train_path = "../Landmark_dataset_flame_aligned/dataset_training/Partial2"
        test_path = "../Landmark_dataset_flame_aligned/dataset_testing/Partial2"
        actors_path = "../Actors/"
        type_dataset = "COMA_Florence"
        batch_train = 25
        batch_test = 20
        lr = 1e-5
        epochs = 2000
        hidden_size = 256
        num_classes = 10
        input_size = (68 * 3)
        layers = 2

    save_path = "../Classification/Models/model_" + str(layers) + "_" + str(lr) + "_" + str(
        hidden_size) + "_" + type_dataset + ".pt"
    writer = SummaryWriter(
        "../TensorBoard/Classification_" + str(layers) + "_" + str(lr) + "_" + str(
            hidden_size) + "_" + type_dataset + "_" + datetime.now().strftime(
            "%m-%d-%Y_%H:%M"))
    actors_coma, name_actors_coma = import_actor(path=actors_path)

    dataset_train = FastDataset(train_path, actors_coma, name_actors_coma)
    training_dataloader = DataLoader(dataset_train, batch_size=batch_train, shuffle=True, drop_last=False, pin_memory=True,
                                     num_workers=5)
    dataset_test = FastDataset(test_path, actors_coma, name_actors_coma)
    testing_dataloader = DataLoader(dataset_test, batch_size=batch_test, shuffle=False, drop_last=False)

    model = ClassificationRNN(hidden_size, input_size, num_classes, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    for epoch in tqdm(range(epochs)):
        tot_loss = 0
        for landmark_animation, label, path_gen, length in training_dataloader:
            optimizer.zero_grad()
            landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
            label = label.to(device)
            logits = model(landmark_animation)
            loss = loss_fn(logits, label)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        writer.add_scalar('Classification/' + type_dataset + '/Validation_loss_train',
                          tot_loss / len(training_dataloader), epoch + 1)

        if not (epoch + 1) % 10:
            print("Epoch: ", epoch + 1, " - Training loss: ", tot_loss / len(training_dataloader))

        if not (epoch + 1) % 50:
            tot_acc_test = 0
            model.eval()
            for landmark_animation, label, path_gen, length in testing_dataloader:
                landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
                label = label.to(device)
                with torch.no_grad():
                    logits = model(landmark_animation)

                tot_acc_test += accuracy(logits, label).item()

            writer.add_scalar('Classification/' + type_dataset + '/Validation_Accuracy',
                              tot_acc_test / len(testing_dataloader), epoch + 1)
            print("Epoch: ", epoch + 1, " - Testing Accuracy: ", tot_acc_test / len(testing_dataloader))
            model.train()

    torch.save(model, save_path)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()