from torch import nn
import torch
from torch.autograd import Variable
from opacus.layers import DPLSTM

class BatteryDataModel(nn.Module):
    def __init__(self):
        super(BatteryDataModel,self).__init__()
        self.input_size = 3
        self.num_layers = 1
        self.hidden_dim = 32
        self.output_size = 1
 
        self.lstm = DPLSTM(input_size=self.input_size, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, batch_first=True) 
        self.linear_1 =  nn.Linear(self.hidden_dim, self.output_size) 
        self.tanh = nn.Tanh()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))

        out, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        hn = hn.view(-1, self.hidden_dim) 
        out = self.tanh(hn)
        out = self.linear_1(out) 
        out = self.tanh(out)
        return out