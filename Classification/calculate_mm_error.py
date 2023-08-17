import os
import shutil
import numpy as np

from common_function import setup_folder_testing_validation


def main():
    coma = True
    if coma:
        folder_gen = "../Generated_Animation/COMA_40_1024_E4_1200"
        gt_folder = "../Landmark_dataset_flame_aligned_coma/dataset_testing"
    else:
        folder_gen = "../Generated_Animation/L1_1200_0.0001_256_COMA_Florence"
        gt_folder = "../Landmark_dataset_flame_aligned/dataset_testing/Partial2"

    test_path = folder_gen + "/testing"
    shutil.rmtree(test_path, ignore_errors=True, onerror=None)
    os.makedirs(test_path)
    setup_folder_testing_validation(folder_gen, test_path)
    mse = []
    lamda = 10 ** 4
    for file in sorted(os.listdir(test_path)):
        tmp = os.path.basename(file)
        if "FaceTalk" in tmp:
            label = tmp[:tmp.find("_FaceTalk")]
            face = tmp[1 + tmp.find("_FaceTalk"):-4]
        else:
            label = tmp[:tmp.find("_")]
            face = tmp[1 + tmp.find("_"): -4]

        generated = np.load(test_path + "/" + file, allow_pickle=True)
        real = np.load(gt_folder + "/" + label + "_" + face + ".npy", allow_pickle=True)
        real = real[1:, :, :]

        tmp_mse = np.mean(np.square(real - generated))
        mse.append(tmp_mse * lamda)
        print(face, label, tmp_mse * lamda)

    print("--------------------------------------")
    min_mse = np.min(np.array(mse))
    max_mse = np.max(np.array(mse))

    print("MIN: " + str(min_mse), "MAX: " + str(max_mse))
    mean = (max_mse - min_mse) / 2
    plus = max_mse - mean
    print("ERROR: " + str(mean) + "±" + str(plus))


if __name__ == "__main__":
    main()
