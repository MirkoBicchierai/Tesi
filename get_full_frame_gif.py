import os
import numpy as np
from tqdm import tqdm
from common_function import plot_frame


def main():
    default_path = "Generated_Animation"
    folder_list = sorted(os.listdir(default_path))
    for f in tqdm(folder_list):
        print(f)
        for a in sorted(os.listdir(default_path + "/" + f + "/GraphTest")):
            animation_path = default_path + "/" + f + "/GraphTest/" + a + "/np"
            animation = animation_path + "/" + sorted(os.listdir(animation_path))[0]
            animation_np = np.load(animation, allow_pickle=True)
            plot_frame(animation_np, default_path + "/" + f + "/GraphTest/" + a)


if __name__ == "__main__":
    main()
