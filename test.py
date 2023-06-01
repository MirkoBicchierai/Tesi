import torch

from Model import DecoderRNN

hidden_size = 512
num_classes = 10
output_size = (68 * 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DecoderRNN(hidden_size, output_size, num_classes, device).to(device)

inuput = torch.randn(8, 61, 68, 3).to(device)
label = torch.LongTensor([1]).to(device)


output = model(inuput, label)
print("a")
