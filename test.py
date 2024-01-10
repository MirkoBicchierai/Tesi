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
    coma = True

    if coma:
        test_path = "Landmark_dataset_flame_aligned_coma/dataset_testing"
        actors_path = "Actors_Coma/"
        save_path = "Models/model_L1_1200_1e-05_1024_COMA_DiffSplit.pt"
        frame_generate = 40
    else:
        test_path = "Landmark_dataset_flame_aligned/dataset_testing/Partial2"
        actors_path = "Actors/"
        save_path = "Models/model_L1_1200_1e-05_1024_COMA_Florence.pt"
        frame_generate = 60

    actors_coma, name_actors_coma = import_actor(path=actors_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_test = FastDataset(test_path, actors_coma, name_actors_coma)
    testing_dataloader = DataLoader(dataset_test, batch_size=5, shuffle=False, drop_last=False)

    model = torch.load(save_path)
    model.eval()
    for landmark_animation, label, path_gen in tqdm(testing_dataloader):
        landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            output = model(landmark_animation[:, 0], label, frame_generate)
            plot_graph(build_face(output, path_gen, actors_coma, name_actors_coma), path_gen, -1)

    print(label_faces_check)


if __name__ == "__main__":
    main()
