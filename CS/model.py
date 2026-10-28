
import torch
import torch.nn as nn
from torch.autograd import Variable

class CSClass(nn.Module):

    def __init__(self):
        super(CSClass, self).__init__()
        
        self.num_classes = 1
        self.num_layers = 1
        self.input_size = 1
        self.hidden_size = 32
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        return out