import numpy as np


animation = np.load("GraphTest/CH02_Cheeky/np/generated_Cheeky_CH02.npy", allow_pickle=True)
print(animation.shape)
""" 
animation = np.load("Landmark_dataset_flame/dataset_training/Partial/Bored_CH09.npy", allow_pickle=True)
print(animation.shape)


animation = np.load("Landmark_dataset/dataset_testing/Partial2/Bored_CH01.npy", allow_pickle=True)
test = torch.Tensor(animation)
fol_np = 'GraphTest_GT'
if not os.path.exists(fol_np):
    os.makedirs(fol_np)
file_np = fol_np + '/frames.npy'
np.save(file_np, test.cpu().flatten(1).numpy())

animation = np.load("landmarks_npy/FaceTalk_170731_00024_TA_sentence40.npy", allow_pickle=True)
print(animation, animation.shape)

animation = np.load(file_np, allow_pickle=True)
print(animation, animation.shape)

"""