import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,n_hidden,seq_len,pred_len,n_features = 1,n_layers=2,dropout=0.5,**kwargs):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device(0)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            **kwargs)
        self.linear1 = nn.Linear(n_hidden, 256)
        self.relu = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(256,pred_len)
    
    def get_hiddenstate(self):
        self.h_0 = torch.zeros(self.n_layers,self.seq_len,self.n_hidden)
        self.c_0 = torch.zeros(self.n_layers,self.seq_len,self.n_hidden)
        if self.is_cuda:
            self.h_0 = self.h_0.to(self.device)
            self.c_0 = self.c_0.to(self.device)
        self.hidden = (self.h_0,self.c_0)
    
    def forward(self,x,args):
        self.get_hiddenstate()
        y,self.hidden = self.lstm(x,self.hidden)
        y = y.view(self.seq_len,len(x),self.n_hidden)[-1]
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = y.view(y.shape[0],1,-1)
        return y