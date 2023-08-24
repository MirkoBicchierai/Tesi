import shutil
import torch
import os
from torch.utils.data import DataLoader
from DataLoader import FastDataset
from common_function import import_actor, setup_folder_testing_validation
from torchmetrics import Accuracy


def search(model_list, actors_path, num_classes, device, test_sets_path):
    actors_coma, name_actors_coma = import_actor(path=actors_path)
    accuracy_list = {}
    for model_path in model_list:
        print("Model: " + model_path[len("Models/"):])
        model = torch.load(model_path)
        model.eval()

        dataset_test = FastDataset(test_sets_path, actors_coma, name_actors_coma)
        testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)
        tot_acc_test = 0
        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        for landmark_animation, label, path_gen, length in testing_dataloader:
            landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
            label = label.to(device)
            with torch.no_grad():
                logits = model(landmark_animation)
            tot_acc_test += accuracy(logits, label).item()

        print("Accuracy: " + str(tot_acc_test / len(testing_dataloader)))
        accuracy_list[model_path[len("Models/"):]] = tot_acc_test / len(testing_dataloader)

    best_model = max_keys_value(accuracy_list)
    return best_model


def max_keys_value(list_):
    max_v = None
    keys_max = {}
    for k, v in list_.items():
        if max_v is None:
            max_v = v
        if v >= max_v:
            max_v = v

    for k, v in list_.items():
        if max_v == v:
            keys_max[k] = v

    return keys_max


def calculate_accuracy_best_gen(best_model_list, best_gen, actors_path, num_classes, device):
    test_path = best_gen + "/testing"
    shutil.rmtree(test_path, ignore_errors=True, onerror=None)
    os.makedirs(test_path)
    setup_folder_testing_validation(best_gen, test_path)
    actors_coma, name_actors_coma = import_actor(path=actors_path)
    accuracy_list = {}
    for p, _ in best_model_list.items():

        model_path = "Models/" + p
        model = torch.load(model_path)
        model.eval()
        dataset_test = FastDataset(test_path, actors_coma, name_actors_coma)
        testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)
        tot_acc_test = 0
        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        for landmark_animation, label, path_gen, length in testing_dataloader:
            landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
            label = label.to(device)
            with torch.no_grad():
                logits = model(landmark_animation)
            tot_acc_test += accuracy(logits, label).item()

        accuracy_list[p] = tot_acc_test / len(testing_dataloader)
        print("Model: " + p + " Accuracy Generation: " + str(tot_acc_test / len(testing_dataloader)))

    return max_keys_value(accuracy_list)


def main():
    default_path = "Models"
    full_list = sorted(os.listdir(default_path))
    list_coma_model = [default_path + "/" + s for s in full_list if "COMA_Florence" not in s]
    list_coma_florence_model = [default_path + "/" + s for s in full_list if "COMA_Florence" in s]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_coma_mm = "../Generated_Animation/L2_1200_1e-05_2048_COMA"
    best_coma_florence_mm = "../Generated_Animation/L1_1200_0.0001_512_COMA_Florence"

    print("---------- COMA -----------")
    best_list = search(list_coma_model, "../Actors_Coma/", 12, device,
                       "../Landmark_dataset_flame_aligned_coma/dataset_testing")
    print("BEST: ", best_list)
    best_list_gen = calculate_accuracy_best_gen(best_list, best_coma_mm, "../Actors_Coma/", 12, device)
    print("BEST GEN: ", best_list_gen)
    print("----------- END -----------")

    print("------ COMA Florence ------")
    best_list = search(list_coma_florence_model, "../Actors/", 10, device,
                       "../Landmark_dataset_flame_aligned/dataset_testing/Partial2")
    print("BEST: ", best_list)
    best_list_gen = calculate_accuracy_best_gen(best_list, best_coma_florence_mm, "../Actors/", 10, device)
    print("BEST GEN: ", best_list_gen)
    print("----------- END -----------")


if __name__ == "__main__":
    main()
