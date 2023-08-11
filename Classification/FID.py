import os
import shutil

import torch
import numpy as np
from scipy.linalg import sqrtm

from common_function import import_actor, setup_folder_testing_validation


# Funzione per calcolare le statistiche di attivazione
def calculate_activation_statistics(sequences, model, device, num_features):
    model.eval()
    act_arr = np.zeros((len(sequences),
                        num_features))  # num_features è il numero di caratteristiche dell'output della tua rete neurale

    for i, sequence in enumerate(sequences):
        sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
        with torch.no_grad():
            activations = model(sequence_tensor)
        act_arr[i] = activations.cpu().numpy()

    mu = np.mean(act_arr, axis=0)
    sigma = np.cov(act_arr, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    eps = 1e-6
    diff = mu1 - mu2

    # Regularize the covariance matrices to make them non-singular
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)

    tr_covmean = np.trace(covmean)
    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


# # TODO GIRO Definire la tua rete neurale per estrarre le caratteristiche dalle sequenze di punti
class ModelFID(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ModelFID, self).__init__()
        # Definire la struttura della rete

    def forward(self, x):
        # Implementare il passaggio in avanti della rete
        return output


def sequences(path, actors_coma, name_actors_coma, shift):
    seq = []
    file_list = [path + e for e in sorted(os.listdir(path))]
    for file in file_list:
        g = np.load(file, allow_pickle=True)

        if shift:
            label_fake = os.path.basename(file)
            face_fake = label_fake[label_fake.find("_") + 1:label_fake.find(".")]
            if "FaceTalk" in face_fake:
                face_fake = face_fake[face_fake.index("FaceTalk"):]
            id_template = name_actors_coma.index(face_fake)
            template = actors_coma[id_template]
            for j in range(g.shape[0]):
                g[j] = g[j] - template

        seq.append(g)

    return seq


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coma = True
    shift = True
    num_features = 512  # Il numero di caratteristiche dell'output della rete neurale

    if coma:
        folder_gen = "../Generated_Animation/COMA_40_1024_E4_1200"
        actors_path = "../Actors_Coma/"
        real_test_path = "../Landmark_dataset_flame_aligned_coma/dataset_testing/"  # dataset_training
        num_classes = 12
        save_path = "Models/model_COMA_0.0001_256_LAYER2.pt"
        sequence_length = 40
    else:
        folder_gen = "../Generated_Animation/COMA_FLORENCE_1024_E4_1200"
        actors_path = "../Actors/"
        real_test_path = "../Landmark_dataset_flame_aligned/dataset_testing/Partial2/"  # dataset_training
        num_classes = 10
        save_path = "Models/model_COMAFlorence_0.0001_256_LAYER1.pt"
        sequence_length = 60

    test_path = folder_gen + "/testing"
    shutil.rmtree(test_path, ignore_errors=True, onerror=None)
    os.makedirs(test_path)
    setup_folder_testing_validation(folder_gen, test_path)
    actors_coma, name_actors_coma = import_actor(path=actors_path)

    # TODO GIRO
    model_real = ModelFID(input_size=sequence_length, output_size=num_features)
    model_generated = ModelFID(input_size=sequence_length, output_size=num_features)
    model_real.to(device)
    model_generated.to(device)

    real_sequences = sequences(real_test_path, actors_coma, name_actors_coma, shift)
    real_sequences = torch.tensor(real_sequences)[:, 1:].type(torch.FloatTensor).to(device)

    generated_sequences = sequences(test_path, actors_coma, name_actors_coma, shift)
    generated_sequences = torch.tensor(generated_sequences).type(torch.FloatTensor).to(device)

    real_mu, real_sigma = calculate_activation_statistics(real_sequences, model_real, device, num_features)
    gen_mu, gen_sigma = calculate_activation_statistics(generated_sequences, model_generated, device, num_features)

    fid_score = calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
    print(f"FID score: {fid_score}")


if __name__ == "__main__":
    main()
