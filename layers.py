import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, X_input, X_gate):
        """ 
        dim(X_input) = dim(X_gate)

        example:
        >>> x_in = torch.tensor([[0.4288, 0.1976],\
                                 [0.3148, 0.5915]])
        >>> x_gate = torch.tensor([[0.7942, 0.0707],\
                                  [0.9348, 0.8788]])
        >>> GLU().forward(x_in, x_gate)
        tensor([[0.2953, 0.1023],
                [0.2260, 0.4179]])
        """
        return X_input * torch.sigmoid(X_gate)
