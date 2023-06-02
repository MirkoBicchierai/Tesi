import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from DataLoader import FastDataset
from Model import DecoderRNN
from test import plot_graph


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = "Landmark_dataset/dataset_training/Complete"  # Partial
    test_path = "Landmark_dataset/dataset_testing/Complete"  # Partial
    save_path = "Models/"

    filelist = [f for f in os.listdir("Graph/")]
    for f in filelist:
        os.remove(os.path.join("Graph/", f))

    dataset_train = FastDataset(train_path)
    training_dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True, drop_last=False)
    dataset_test = FastDataset(test_path)
    testing_dataloader = DataLoader(dataset_test, batch_size=2, shuffle=False, drop_last=False)

    hidden_size = 256
    num_classes = 70  # number of label
    output_size = (68 * 3)  # landmark point * coordinates
    frame_generate = 60  # number of frame generate by lstm

    model = DecoderRNN(hidden_size, output_size, num_classes, frame_generate, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epochs = 500
    for epoch in tqdm(range(epochs)):
        tot_loss = 0
        for landmark_animation, label, str_label in training_dataloader:
            for idF in range(landmark_animation[:, 1:].shape[1]):
                optimizer.zero_grad()
                landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
                output = model(landmark_animation[:, idF], label, 60 - idF)
                loss = F.mse_loss(output, landmark_animation[:, 1 + idF:])
                tot_loss += loss.item()
                loss.backward()
                optimizer.step()

        if not (epoch + 1) % 10:
            print("Epoch: ", epoch + 1, " - Training loss: ", tot_loss / len(training_dataloader))

        if not (epoch + 1) % 50:
            tot_loss_test = 0
            model.eval()
            check = True
            for landmark_animation, label, str_label in testing_dataloader:
                landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
                with torch.no_grad():
                    output = model(landmark_animation[:, 0], label, 60)
                    if check:
                        plot_graph(output.cpu().numpy(), str_label, epoch)
                    check = False
                test_loss = F.mse_loss(output, landmark_animation[:, 1:])
                tot_loss_test += test_loss.item()

            print("Epoch: ", epoch + 1, " - Testing loss: ", tot_loss_test / len(testing_dataloader))
            model.train()

    torch.save(model, os.path.join(save_path, "modelComplete.pt"))  # modelPartial.pt


if __name__ == "__main__":
    main()
