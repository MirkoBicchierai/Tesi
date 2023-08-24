import shutil
import numpy as np
from common_function import setup_folder_testing_validation
import os


def get_loss_err(d_list):
    sum_err = 0
    sum_std = 0
    for key in d_list:
        sum_err = sum_err + d_list[key][0]
        sum_std = sum_std + d_list[key][1]
    return sum_err / len(d_list), sum_std / len(d_list)


def print_result(list_coma, default_path, lamda, gt_folder):
    sum_mean_std = {}
    for folder_gen in sorted(list_coma):
        name = folder_gen
        folder_gen = default_path + "/" + folder_gen
        test_path = folder_gen + "/testing"
        shutil.rmtree(test_path, ignore_errors=True, onerror=None)
        os.makedirs(test_path)
        setup_folder_testing_validation(folder_gen, test_path)
        mean_err = []
        std_err = []
        for file in sorted(os.listdir(test_path)):
            tmp = os.path.basename(file)
            if "FaceTalk" in tmp:
                label = tmp[:tmp.find("_FaceTalk")]
                face = tmp[1 + tmp.find("_FaceTalk"):-4]
            else:
                label = tmp[:tmp.find("_")]
                face = tmp[1 + tmp.find("_"): -4]

            generated = (np.load(test_path + "/" + file, allow_pickle=True)) * lamda
            gt = (np.load(gt_folder + "/" + label + "_" + face + ".npy", allow_pickle=True))[1:, :, :] * lamda

            mean_err.append(np.mean(np.sqrt(np.sum((generated - gt) ** 2, axis=2))))
            std_err.append(np.sqrt(np.sum((generated - gt) ** 2, axis=2)))

        print("--------------------------------------------------")
        print("TEST: " + name)
        print('Error mm:', np.mean(mean_err))
        print('Std:', np.std(std_err))
        print("Error:", str(np.mean(mean_err)) + "±" + str(np.std(std_err)))
        sum_mean_std[name] = [np.mean(mean_err), np.std(std_err)]

    name_min = next(iter(sum_mean_std))
    min_vec = sum_mean_std[name_min][0] + sum_mean_std[name_min][1]
    std = sum_mean_std[name_min][1]
    err = sum_mean_std[name_min][0]
    for key in sum_mean_std:
        if min_vec >= sum_mean_std[key][0] + sum_mean_std[key][1]:
            min_vec = sum_mean_std[key][0] + sum_mean_std[key][1]
            err = sum_mean_std[key][0]
            std = sum_mean_std[key][1]
            name_min = key

    l2_err = {k: v for k, v in sum_mean_std.items() if "L1" not in k}
    l1_err = {k: v for k, v in sum_mean_std.items() if "L2" not in k}

    print("-------------- FINAL RESULT ----------------")
    print("BEST: " + name_min + " error: " + str(err) + "±" + str(std))
    err, std = get_loss_err(l1_err)
    print("MEAN L1 error: " + str(err) + "±" + str(std))
    err, std = get_loss_err(l2_err)
    print("MEAN L2 error: " + str(err) + "±" + str(std))


def main():
    default_path = "../Generated_Animation"
    lamda = 1000
    full = sorted(os.listdir(default_path))
    list_coma = [s for s in full if "COMA_Florence" not in s]
    list_coma_florence = [s for s in full if "COMA_Florence" in s]

    print("---------- ERROR mm COMA ----------")
    print_result(list_coma, default_path, lamda, "../Landmark_dataset_flame_aligned_coma/dataset_testing")
    print("-------------------- END -------------------")
    print("\n")
    print("---------- ERROR mm COMA_Florence ----------")
    print_result(list_coma_florence, default_path, lamda, "../Landmark_dataset_flame_aligned/dataset_testing/Partial2")
    print("-------------------- END -------------------")


if __name__ == "__main__":
    main()
