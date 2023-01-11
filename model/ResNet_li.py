"""
该模型的思路是：只做卷积/残差，不做池化，维持(L,L)的大小不变。
输出为(batch_size,class_num,L,L)
"""

import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size = 3, strides=1):
        super(Residual, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.norm1 = nn.InstanceNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size, padding=self.padding, stride=strides)
        self.norm2 = nn.InstanceNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=self.padding, stride=strides)

        use_1x1conv = False if input_channels == num_channels else True

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size, padding=self.padding, stride=strides)

        else:
            self.conv3 = None

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.conv1(self.norm1(X)))
        Y = self.conv2(self.norm2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals):
    blk = []
    for i in range(num_residuals):
        blk.append(Residual(input_channels, num_channels))
    return blk

# class embed_to_2D():
#     def __init__(self, x):
#
#
#     def forward(self):
#

class MyResNet(nn.Module):
    def __init__(self, num_classes = 2, dim_in = 105, dim_out = 36,
                 resnet_dim = 128, num_block = 8, use_softmax=False):

        super(MyResNet, self).__init__()
        self.use_softmax = use_softmax
        self.num_classes = num_classes
        self.dim_out = dim_out
        self.block = num_block

        self.b1 = nn.Sequential(nn.Conv2d(dim_in, resnet_dim, kernel_size=3, padding=1), nn.ReLU())   # dim: din_in->resnet_dim
        self.b2 = nn.Sequential(*resnet_block(resnet_dim, resnet_dim, num_block))  # dim: resnet_dim->resnet_dim
        self.b3 = nn.Sequential(*resnet_block(resnet_dim, dim_out, 1))  # dim: resnet_dim -> dim_out
        self.linear = nn.Linear(dim_out, num_classes)  # 将dim_out降维成num_classes，该维度中的每一个值代表在该bin中的概率

        self.net = nn.Sequential(self.b1, self.b2, self.b3)

        # self.embed_to_2D = embed_to_2D()
        self.dim_red = nn.Linear(5120, 64)

    def forward(self, X_embed, X_atten):
        # embed [batch_size, L,L,5120]
        # atten [batch_size, 41,L,L]

        # ===========================================
        # =========这一块用Self-Attention改写=========
        # ===========================================
        X_embed = self.dim_red(X_embed)   # (batch_size,L,L,64)
        X_embed = torch.permute(X_embed, (0,3,1,2))  # (batch_size,64,L,L)

        Y = self.net(torch.concat((X_embed, X_atten), dim=1))   # Y: (batch, dim_in, len, len) -> (batch, dim_out, len, len)  dim_in: 105; len: 192
        Y = torch.permute(Y, (0,2,3,1))   # Y: (batch, dim_out, len, len) -> (batch, len, len, dim_out)  len: 192
        Y = self.linear(Y)  # Y: (batch, len, len, dim_out) -> (batch, len, len, num_classes)   num_classes: 2

        return Y.reshape(-1, self.num_classes)


if __name__ == '__main__':
    model = MyResNet()
    embed = torch.randn((1, 129, 129, 5120))
    atten = torch.randn((1, 41, 129,129))

    output = model(embed, atten)
    print(output.shape)

    # from utils.vis_model import vis_model
    # vis_model(model, (embed, atten), filename="ResNet_li")

