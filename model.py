import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GLU

class ConvGLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_out=0.2):
        super(ConvGLUBlock, self).__init__()
        self.conv_data = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(p=drop_out)
        self.glu = GLU()

    def forward(self, X):
        return self.dropout(self.glu(self.conv_data(X), self.conv_gate(X)))


class CharWordSeg(nn.Module):
    def __init__(self):
        super(CharWordSeg).__init__()
    
    def forward(self):
        pass