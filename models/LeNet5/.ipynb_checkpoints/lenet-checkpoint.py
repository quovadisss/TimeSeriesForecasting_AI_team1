import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self,seq_len, pred_len):
        super(LeNet5, self).__init__()
        kernel_1 = 3
        kernel_2 = 3

        self.feature_extractor = nn.Sequential(            
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=kernel_1, stride=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_2, stride=1),
            nn.Tanh()
        )
#         expansion = 3
        expansion = int(round((seq_len - kernel_1 + 1) - kernel_2 + 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*expansion, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features = pred_len),
        )

    def forward(self, x):
        import ipdb
        ipdb.set_trace()
        print(x.shape)
        x = x.permute(0, 2, 1)
        print(x.shape)
        x = self.feature_extractor(x)
        print(x.shape)
        x = x.view(x.size(0),-1)
        print(x.shape)
        
        output = self.classifier(x)
        print(output.shape)
        output = output.unsqueeze(2)
        print(output.shape)        

        return output

    
    