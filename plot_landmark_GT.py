import os

import numpy as np

from common_function import plot_frames_bis

gt = True
if gt:
    file = "Landmark_dataset_flame_aligned/dataset_testing/Partial2/Cheeky_CH01.npy"
    name = os.path.basename(file)
    animation = np.load(file, allow_pickle=True)
    plot_frames_bis(animation, "gt_pdf", 12, name[:name.find(".")])
else:
    f = "Generated_Animation/L2_1200_1e-05_2048_COMA/GraphTest/"
    step = 7  # 10
    # L2_1200_1e-05_2048_COMA
    # L1_1200_0.0001_512_COMA_Florence
    for file in sorted(os.listdir(f)):
        p = f + file + "/np/"
        name = os.path.basename(p + os.listdir(p)[0])
        animation = np.load(p + os.listdir(p)[0], allow_pickle=True)
        plot_frames_bis(animation, "gen_pdf", step, name[:name.find(".")])
