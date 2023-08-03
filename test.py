import shutil
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import FastDataset
from common_function import label_faces_check, plot_graph, import_actor, build_face


def main():
    shutil.rmtree("GraphTest/", ignore_errors=False, onerror=None)
    os.makedirs("GraphTest/")

    actors_coma, name_actors_coma = import_actor(path="Actors_Coma/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = "Landmark_dataset_flame_aligned_coma/dataset_testing"
    save_path = "Models/model_1500_l2_e4_512_only-shift_COMA.pt"

    aligned = True
    dataset_test = FastDataset(test_path, actors_coma, name_actors_coma)
    testing_dataloader = DataLoader(dataset_test, batch_size=5, shuffle=False, drop_last=False)

    model = torch.load(save_path)
    model.eval()
    for landmark_animation, label, path_gen in tqdm(testing_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            output = model(landmark_animation[:, 0], label, 29)
            plot_graph(build_face(output, path_gen, actors_coma, name_actors_coma), path_gen, -1, aligned)

    print(label_faces_check)


if __name__ == "__main__":
    main()
