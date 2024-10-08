import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_classes, length, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(output_size * 2, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, output_size)

        self.fc_inh = nn.Linear(output_size, hidden_size)
        self.fc_inc = nn.Linear(output_size, hidden_size)
        self.num_classes = num_classes
        self.device = device
        self.length = length
        self.fc_enc = nn.Linear(self.num_classes, output_size)

    def forward(self, inputs, labels, length):  # ,actors
        encoding = F.one_hot(labels, num_classes=self.num_classes)
        encoding = encoding.type(torch.FloatTensor).to(self.device)
        encoding = self.fc_enc(encoding)

        frame = inputs.flatten(-2)
        h_t = self.fc_inh(frame)
        c_t = self.fc_inc(frame)
        frame = torch.cat([frame, encoding.to(self.device)], dim=-1)

        output = []
        for _ in range(length):
            h_t, c_t = self.lstm(frame, (h_t, c_t))
            output_ = F.relu(self.fc_1(h_t))
            output_ = self.fc_2(output_)
            frame = torch.cat([output_, encoding.to(self.device)], dim=-1)
            output.append(output_)
        output = torch.stack(output, dim=1).reshape(inputs.shape[0], length, inputs.shape[1], inputs.shape[2])

        return output
