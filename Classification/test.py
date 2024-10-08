import shutil
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import FastDatasetC
from common_function import import_actor, setup_folder_testing_validation
from torchmetrics import Accuracy


def main():
    coma = True
    if coma:
        folder_gen = "../Generated_Animation/L1_1200_0.0001_128_COMA"
        actors_path = "../Actors_Coma/"
        num_classes = 12
        save_path = "Models/model_1_0.0001_128_COMA.pt"

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

    dataset_test = FastDatasetC(test_path, actors_coma, name_actors_coma)
    testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)

    model = torch.load(save_path)
    model.eval()
    tot_acc_test = 0

    pd =[]

    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    for landmark_animation, label, path_gen, length in tqdm(testing_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(landmark_animation)

        tot_acc_test += accuracy(logits, label).item()

        pd.append(model.get_distr(landmark_animation))

    print("Accuracy: " + str(tot_acc_test / len(testing_dataloader)))


if __name__ == "__main__":
    main()
