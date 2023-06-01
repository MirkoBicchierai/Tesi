import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_classes, length, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # self.lstm = nn.LSTMCell(output_size + num_classes, hidden_size)
        self.lstm = nn.LSTM(output_size + num_classes, hidden_size, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc_inh = nn.Linear(output_size, hidden_size * 3)
        self.fc_inc = nn.Linear(output_size, hidden_size * 3)
        self.num_classes = num_classes
        self.device = device
        self.length = length

    def forward(self, inputs, labels):
        encoding = F.one_hot(labels, num_classes=self.num_classes)
        encoding = encoding.repeat([inputs.shape[0], 1])
        # LSTM CELL
        # h_t = torch.ones(inputs.shape[0], self.hidden_size, dtype=torch.float32).to(self.device)
        # c_t = torch.ones(inputs.shape[0], self.hidden_size, dtype=torch.float32).to(self.device)

        frame = inputs.flatten(-2)
        h_t = self.fc_inh(frame).reshape(inputs.shape[0], 3, self.hidden_size)  # lstm CELL h_t = self.fc_inh(frame)
        c_t = self.fc_inc(frame).reshape(inputs.shape[0], 3, self.hidden_size)  # lstm CELL c_t = self.fc_inc(frame)
        frame = torch.cat([frame, encoding.to(self.device)], dim=-1)
        # frame = frame.unsqueeze(1).repeat([1, 3, 1])  # lstm

        output = []
        for _ in range(self.length):
            h_t, c_t = self.lstm(frame, (h_t, c_t))
            output_ = self.fc(h_t)
            output.append(output_)
        output = torch.stack(output, dim=1).reshape(inputs.shape[0], self.length, inputs.shape[1], inputs.shape[2])

        return output
