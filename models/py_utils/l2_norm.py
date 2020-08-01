import torch
import torch.nn as nn
import torch.nn.init as init


# class L2Normalization(nn.Module):
#     def __init__(self, dim, gamma):
#         super(L2Normalization, self).__init__()
#         self.dim = dim
#         self.gamma = gamma
#
#     def forward(self, x):
#         norm = torch.norm(x, self.dim)
#         output = torch.div(x, norm)
#         output = output*self.gamma
#         return output
class L2Normalization(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Normalization, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
