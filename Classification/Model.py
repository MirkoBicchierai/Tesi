import torch
import torch.nn as nn


class ClassificationRNN(nn.Module):
    def __init__(self, hidden_size, input_size, num_classes, layers):
        super(ClassificationRNN, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, bidirectional=True, batch_first=True)
        # self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc_inh = nn.Linear(input_size, hidden_size * 2 * self.layers)
        self.fc_inc = nn.Linear(input_size, hidden_size * 2 * self.layers)
        self.fc_classification = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs):
        frames = inputs.flatten(-2)
        h_t = self.fc_inh(frames[:, 0])
        c_t = self.fc_inc(frames[:, 0])
        h_t = torch.reshape(h_t, (2 * self.layers, frames.shape[0], self.hidden_size))
        c_t = torch.reshape(c_t, (2 * self.layers, frames.shape[0], self.hidden_size))
        h_t, c_t = self.lstm(frames, (h_t, c_t))
        h_t = torch.mean(h_t, dim=1)
        return self.fc_classification(h_t)
