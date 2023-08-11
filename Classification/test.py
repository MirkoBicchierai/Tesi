import shutil
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import FastDataset
from common_function import import_actor, setup_folder_testing_validation
from torchmetrics import Accuracy
import torch.nn.functional as F


def main():
    coma = True
    if coma:
        folder_gen = "../Generated_Animation/COMA_40_1024_E4_1200"
        actors_path = "../Actors_Coma/"
        num_classes = 12
        save_path = "Models/model_COMA_0.0001_256_LAYER2.pt"

    else:
        folder_gen = "../Generated_Animation/COMA_FLORENCE_1024_E4_1200"
        actors_path = "../Actors/"
        num_classes = 10
        save_path = "Models/model_COMAFlorence_0.0001_256_LAYER1.pt"

    test_path = folder_gen + "/testing"
    shutil.rmtree(test_path, ignore_errors=True, onerror=None)
    os.makedirs(test_path)

    setup_folder_testing_validation(folder_gen, test_path)

    actors_coma, name_actors_coma = import_actor(path=actors_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_test = FastDataset(test_path, actors_coma, name_actors_coma)
    testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)

    model = torch.load(save_path)
    model.eval()
    tot_acc_test = 0

    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # sus_gen = torch.tensor([]).to(device)
    # sus_real = torch.tensor([]).to(device)
    for landmark_animation, label, path_gen, length in tqdm(testing_dataloader):
        # real_gen = []
        # for i in range(len(path_gen)):
        #     label_fake = os.path.basename(path_gen[i])
        #     face_fake = label_fake[label_fake.find("_") + 1:label_fake.find(".")]
        #     # template = actors_coma[int(face[2:]) - 1]
        #     if "FaceTalk" in face_fake:
        #         face_fake = face_fake[face_fake.index("FaceTalk"):]
        #     label_fake = label_fake[: label_fake.find("_")]
        #     g = np.load("../Landmark_dataset_flame_aligned_coma/Completo/" + label_fake + "_" + face_fake + ".npy",
        #                 allow_pickle=True)
        #     id_template = name_actors_coma.index(face_fake)
        #     template = actors_coma[id_template]
        #     for j in range(g.shape[0]):
        #         g[j] = g[j] - template
        #     real_gen.append(g)
        #
        # real_gen = torch.tensor(real_gen)[:, 1:].type(torch.FloatTensor).to(device)

        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(landmark_animation)
            # sus_gen = torch.cat([sus_gen, aaa])
            #
            # _, aaa = model(real_gen)
            # sus_real = torch.cat([sus_real, aaa])

        tot_acc_test += accuracy(logits, label).item()

    # mu_gen = torch.mean(sus_gen, dim=0)
    # mu_real = torch.mean(sus_real, dim=0)
    # sigma_gen = torch.cov(sus_gen.t())
    # sigma_real = torch.cov(sus_real.t())
    # bb = F.mse_loss(mu_real, mu_gen)
    # c = - 2 * torch.sqrt(torch.mm(sigma_real,sigma_gen))
    # aa = torch.trace(sigma_gen + sigma_real + c)
    # fid = aa + bb
    # print("FID: " +str(fid / len(testing_dataloader)))

    print("Accuracy: " + str(tot_acc_test / len(testing_dataloader)))


if __name__ == "__main__":
    main()
