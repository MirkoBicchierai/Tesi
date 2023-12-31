import os
import numpy as np
from common_function import plot_gif

gt = True
# path = "Generated_Animation/L1_1200_0.0001_512_COMA_Florence/GraphTest/CH01_Kissy/HD"
# file = "Generated_Animation/L1_1200_0.0001_512_COMA_Florence/GraphTest/CH01_Kissy/np/generated_Kissy_CH01.npy"

path = "Generated_Animation/L2_1200_1e-05_2048_COMA/GraphTest/FaceTalk_170811_03274_TA_high-smile/HD"
file = "Generated_Animation/L2_1200_1e-05_2048_COMA/GraphTest/FaceTalk_170811_03274_TA_high-smile/np/generated_high-smile_FaceTalk_170811_03274_TA.npy"

name = os.path.basename(file)
animation = np.load(file, allow_pickle=True)
plot_gif(animation, path, name[:name.find(".")] + "_HD")
