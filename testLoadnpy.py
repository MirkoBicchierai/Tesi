import torch
from torch.utils.data import DataLoader

from DataLoader import FastDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = "Landmark_dataset_flame/dataset_training/Partial"
test_path = "Landmark_dataset_flame/dataset_testing/Partial"
save_path = "Models/modelPartial50002.pt"


dataset_train = FastDataset(train_path)
training_dataloader = DataLoader(dataset_train, batch_size=25, shuffle=True, drop_last=False, pin_memory=True,
                                 num_workers=5)
for landmark_animation, label, path_gen in training_dataloader:
    break

dataset_test = FastDataset(test_path)
testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)
for landmark_animation, label, path_gen in testing_dataloader:
    break

"""
# animation = np.load("GraphTest/CH02_Cheeky/np/generated_Cheeky_CH02.npy", allow_pickle=True)
# print(animation.shape)

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