"""
该模型的思路是：只做卷积/残差，不做池化，维持(192,192)的大小不变。
输出为(1,8,192,192)，对dim=1取均值得到最终结果
"""

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=strides)

        use_1x1conv = False if input_channels == num_channels else True

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)

        else:
            self.conv3 = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals):
    blk = []
    for i in range(num_residuals):
        blk.append(Residual(input_channels, num_channels))
    return blk


class MyResNet(nn.Module):
    def __init__(self, use_softmax=False):
        super(MyResNet, self).__init__()
        self.use_softmax = use_softmax

    def forward(self, X):
        b1 = nn.Sequential(nn.Conv2d(41, 64, kernel_size=3, padding=1), nn.ReLU())
        b2 = nn.Sequential(*resnet_block(64, 64, 2))
        b3 = nn.Sequential(*resnet_block(64, 32, 1))
        b4 = nn.Sequential(*resnet_block(32, 32, 2))
        b5 = nn.Sequential(*resnet_block(32, 16, 1))
        b6 = nn.Sequential(*resnet_block(16, 8, 1))

        net = nn.Sequential(b1, b2, b3, b4, b5, b6)

        Y = net(X)

        if self.use_softmax:
            Y = Y.reshape(Y.shape[0], Y.shape[1], -1)
            Y = Y.softmax(Y, dim=2).reshape(Y.shape[0], Y.shape[1], 192, 192)

        return Y.mean(dim=1).squeeze()


if __name__ == '__main__':
    model = MyResNet()
    data = torch.randn((1, 41, 192, 192))

    output = model(data)
    print(output)
