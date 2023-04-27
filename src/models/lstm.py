import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vector_size:int, out_channels:int, drop_out:float=0.1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=vector_size, hidden_size=vector_size, bidirectional=True)
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(vector_size * 2, out_channels)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        linear = self.dropout(self.linear(concat))
        return linear