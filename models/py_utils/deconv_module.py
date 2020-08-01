import torch.nn as nn


class DeconvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(DeconvModule, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.deconv.weight)
        nn.init.constant_(self.deconv.bias, 0)

    def forward(self, x):
        x = self.deconv(x)
        return x
