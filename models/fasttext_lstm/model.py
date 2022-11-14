from .config import config
import torch
import torch.nn as nn

class FastTextLSTM(nn.Module):
    def __init__(self, vector_size, out_channels):
        super(FastTextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=vector_size, hidden_size=vector_size, bidirectional=True)
        self.dropout = nn.Dropout(config.DROP_OUT)
        self.linear = nn.Linear(vector_size * 2, out_channels)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        linear = self.dropout(self.linear(concat))
        return linear