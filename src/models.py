import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.hidden_size, 128)     # (input size, output size)
        self.fc2 = nn.Linear(128, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        h_n = h_n.view(-1, self.hidden_size)

        out = self.relu(h_n)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out