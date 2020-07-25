import torch 
import torch.nn as nn

from newnet.config import cfg

class TBNet(nn.Module):
    
    pass

class TBHead(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 2)