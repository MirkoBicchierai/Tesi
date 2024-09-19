import shutil
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import FastDataset
from common_function import import_actor, setup_folder_testing_validation
from torchmetrics import Accuracy


def get_distr_real():
    actors_path = "Actors_Coma/"
    train_path = "Classification/datasetCOMA"
    actors_coma, name_actors_coma = import_actor(path=actors_path)

    dataset_training = FastDataset(train_path, actors_coma, name_actors_coma)
    traning_dataloader = DataLoader(dataset_training, batch_size=1, shuffle=False, drop_last=False)

    save_path = "Classification/Models/model_2_0.0001_512_COMA_DiffSplit_Simple_spost_naima.pt"
    model = torch.load(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real = []
    for landmark_animation, label, path_gen in tqdm(traning_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)

        real.append(model(landmark_animation).detach().cpu().squeeze(0).numpy())

    real_arr = np.array(real)

    m_b = real_arr.mean()

    cov_b = np.cov(real_arr.transpose())
    return m_b, cov_b


def get_distr_acc_generated(generated, label):
    save_path = "Classification/Models/model_2_0.0001_512_COMA_DiffSplit_Simple_spost.pt"
    model = torch.load(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 12
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        logits = model(generated)

    acc = accuracy(logits, label).item()

    fake = logits.detach().cpu().squeeze(0).numpy()
    fake_arr = np.array(fake)
    m_a = fake_arr.mean()
    cov_a = np.cov(fake_arr.transpose())
    return m_a, cov_a, acc


def compute_fid(m_a, cov_a, m_b, cov_b):
    fid = abs(m_a - m_b) + np.trace(cov_a + cov_b - 2 * np.sqrt(cov_a * cov_b))
    return fid


def main():
    coma = True
    if coma:
        folder_gen = "../Generated_Animation/L2_1200_1e-05_1024_COMA_DiffSplit"
        actors_path = "../Actors_Coma/"
        num_classes = 12
        save_path = "Models/model_2_0.0001_512_COMA_DiffSplit_Simple_spost.pt"
        train_path = "datasetCOMA"

    else:
        folder_gen = "../Generated_Animation/COMA_FLORENCE_1024_E4_1200"
        actors_path = "../Actors/"
        num_classes = 10
        save_path = "Models/model_COMAFlorence_0.0001_256_LAYER1.pt"
        train_path = "../Landmark_dataset_flame_aligned/dataset_training/Partial2"

    gen_path = folder_gen + "/testing"
    shutil.rmtree(gen_path, ignore_errors=True, onerror=None)
    os.makedirs(gen_path)

    setup_folder_testing_validation(folder_gen, gen_path)

    actors_coma, name_actors_coma = import_actor(path=actors_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset_training = FastDataset(train_path, actors_coma, name_actors_coma)
    traning_dataloader = DataLoader(dataset_training, batch_size=1, shuffle=False, drop_last=False)

    dataset_generated = FastDataset(gen_path, actors_coma, name_actors_coma)
    generated_dataloader = DataLoader(dataset_generated, batch_size=1, shuffle=False, drop_last=False)

    model = torch.load(save_path)
    model.eval()
    tot_acc_test = 0

    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    real = []
    for landmark_animation, label, path_gen, length in tqdm(traning_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)

        real.append(model(landmark_animation).detach().cpu().squeeze(0).numpy())

    fake = []
    for landmark_animation, label, path_gen, length in tqdm(generated_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(landmark_animation)

        tot_acc_test += accuracy(logits, label).item()

        fake.append(model(landmark_animation).detach().cpu().squeeze(0).numpy())

    fake_arr = np.array(fake)
    real_arr = np.array(real)

    m_a = fake_arr.mean()
    m_b = real_arr.mean()

    cov_a = np.cov(fake_arr.transpose())
    cov_b = np.cov(real_arr.transpose())

    fid = abs(m_a - m_b) + np.trace(cov_a + cov_b - 2 * np.sqrt(cov_a * cov_b))
    print(fid)
    print("Accuracy: " + str(tot_acc_test / len(generated_dataloader)))


if __name__ == "__main__":
    main()
