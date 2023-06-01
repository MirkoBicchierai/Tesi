import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from DataLoader import SingleEmotionDataset
from Model import DecoderRNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = "dataset/SingleExpression/COMA/Partial"
    test_path = "dataset/SingleExpression/COMA/Testing"
    save_path = "Models/"

    dataset_train = SingleEmotionDataset(train_path)
    training_dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataset_test = SingleEmotionDataset(test_path)
    testing_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True)

    hidden_size = 512
    num_classes = 10  # number of faces
    output_size = (68 * 3)  # landmark point * coordinates
    frame_generate = 60  # number of frame generate by lstm

    model = DecoderRNN(hidden_size, output_size, num_classes, frame_generate, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        tot_loss = 0
        for landmark_animation, label, str_label in tqdm(training_dataloader):
            optimizer.zero_grad()
            landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
            landmark_animation = landmark_animation.squeeze(0)
            output = model(landmark_animation[:, 0], label)
            loss = F.mse_loss(output, landmark_animation)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch: ", epoch, " - Training loss: ", tot_loss / len(training_dataloader))

        tot_loss_test = 0
        model.eval()
        for landmark_animation, label, str_label in tqdm(testing_dataloader):
            landmark_animation = landmark_animation.type(torch.FloatTensor).to(device)
            landmark_animation = landmark_animation.squeeze(0)
            with torch.no_grad():
                output = model(landmark_animation[:, 0], label)
            test_loss = F.mse_loss(output, landmark_animation)
            tot_loss_test += test_loss.item()

        print("Epoch: ", epoch, " - Testing loss: ", tot_loss_test / len(testing_dataloader))
        model.train()

    torch.save(model, os.path.join(save_path, "model.pt"))


if __name__ == "__main__":
    main()
