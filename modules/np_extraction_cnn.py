import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 

import torchvision.models as models

import math 

class NP_Extraction_CNN(nn.Module):
    # out_features is equal to the number of noun phrase classes
    def __init__(self, out_features=3000): 
        super(NP_Extraction_CNN,self).__init__()
        self.out_features = out_features

        self.resnet_152 = models.resnet152(pretrained=True)
        self.resnet_152.fc.out_features = 4096
        self.fc_1 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc_2 = nn.Linear(in_features=4096, out_features=self.out_features)

    def forward(self, i):
        f = self.resnet_152(i)
        z = self.fc_1(f)
        z = self.fc_2(z)
        a = F.sigmoid(z)
        return a