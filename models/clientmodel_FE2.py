from collections import OrderedDict
import torch
import torch.nn as nn


class UNET_FE(nn.Module):
    def __init__(self, in_channels=3, init_features=32):
        super(UNET_FE, self).__init__()
        features = init_features
        self.encoder1 = UNET_FE._block1(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        return enc1, pool1

    @staticmethod
    def _block1(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
