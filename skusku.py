import os
import numpy as np

folder_path = 'Dataset_FLAME_Aligned_COMA/DATASET COMPLETO/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000002.ply'
data = np.load(folder_path)
print(data.shape, folder_path)