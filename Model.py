import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_classes, length, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(output_size + num_classes, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_classes = num_classes
        self.device = device
        self.length = length

    def forward(self, inputs, labels):
        encoding = F.one_hot(labels, num_classes=self.num_classes)
        encoding = encoding.repeat([inputs.shape[0], 1])

        h_t = torch.zeros(inputs.shape[0], self.hidden_size, dtype=torch.float32).to(self.device)
        c_t = torch.zeros(inputs.shape[0], self.hidden_size, dtype=torch.float32).to(self.device)

        frame = inputs.flatten(-2)
        frame = torch.cat([frame, encoding.to(self.device)], dim=-1)

        output = [inputs.flatten(-2)]
        for _ in range(self.length):
            h_t, c_t = self.lstm(frame, (h_t, c_t))
            output_ = self.fc(h_t)
            output.append(output_)
        output = torch.stack(output, dim=1).reshape(inputs.shape[0], self.length+1, inputs.shape[1], inputs.shape[2])

        return output
