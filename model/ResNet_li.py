"""
该模型的思路是：只做卷积/残差，不做池化，维持(L,L)的大小不变。
输出为(batch_size,class_num,L,L)
"""

import torch
from torch import nn

from config import device

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

class MyResNet(nn.Module):
    def __init__(self, num_classes = 2, dim_in = 169, dim_out = 36,
                 resnet_dim = 128, num_block = 8, num_1d_blocks = 8, use_softmax=False):

        super(MyResNet, self).__init__()

        self.use_softmax = use_softmax
        self.num_classes = num_classes
        self.dim_out = dim_out
        self.block = num_block
        self.num_1d_blocks = num_1d_blocks

        self.relu = nn.LeakyReLU(inplace=False)
        self.dim_red_1 = nn.Linear(2560, 512)
        self.dim_red_2 = nn.Linear(512, 256)
        self.conv_1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_2d = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)


        self.b1 = nn.Sequential(nn.Conv2d(dim_in, resnet_dim, kernel_size=3, padding=1), nn.ReLU())   # dim: din_in->resnet_dim
        self.b2 = nn.Sequential(*resnet_block(resnet_dim, resnet_dim, num_block))  # dim: resnet_dim->resnet_dim
        self.b3 = nn.Sequential(*resnet_block(resnet_dim, dim_out, 1))  # dim: resnet_dim -> dim_out
        self.linear = nn.Linear(dim_out, num_classes)  # 将dim_out降维成num_classes，该维度中的每一个值代表在该bin中的概率

        self.net = nn.Sequential(self.b1, self.b2, self.b3)

    @staticmethod
    def embed_2_2d(self, embed):
        batch_size = embed.shape[0]
        L = embed.shape[2]
        # 生成模型结构图; onnx不能接受可变参数
        # batch_size =1
        # L = 129
        embed_zero = torch.zeros((batch_size, 256, L, L))

        embed_zero = embed_zero.to(device)

        embed.unsqueeze_(dim=3)  # (batch_size, 256, L) -> (batch_size, 256, L,1)
        embed_1 = embed + embed_zero
        embed_2 = embed_1.transpose(2,3)
        embed = torch.concat((embed_1, embed_2), dim=1)  # (batch_size,512 L,L)

        return embed

    def forward(self, embed, atten):
        # embed [batch_size, L, 2560]
        # atten [batch_size, 41,L,L]

        # 将(batch_size,L,2560)降维
        embed = self.relu(self.dim_red_1(embed))
        embed = self.dim_red_2(embed) # (batch_size,L,256)

        embed = embed.permute(0, 2, 1)  # (batch_size,256,L)
        for i in range(self.num_1d_blocks):
            embed = self.conv_1d(embed)  # (batch_size,256,L)
            embed = self.relu(embed)

        # 2. 将(batch_size, 256, L)转化为(batch_size, 128, L,L)
        embed = self.embed_2_2d(self,embed=embed)   # (batch_size, 256, L) -> (batch_size, 512, L,L)
        embed = self.conv_2d(embed)  # (batch_size, 128, L,L)

        Y = self.net(torch.concat((embed, atten), dim=1))
        # Y: (batch, dim_in, len, len) -> (batch, dim_out, len, len)  dim_in: 169; len: 192

        Y = torch.permute(Y, (0,2,3,1))
        # Y: (batch, dim_out, len, len) -> (batch, len, len, dim_out)  len: 192

        Y = self.linear(Y)
        # Y: (batch, len, len, dim_out) -> (batch, len, len, num_classes)   num_classes: 2

        return Y.reshape(-1, self.num_classes)


if __name__ == '__main__':
    device = torch.device("cuda:0")

    model = MyResNet()

    #

    #
    # model.apply(weight_init)
    #
    # torch.save(model, "random_model_param.pt")

    model.to(device)

    embed = torch.randn((2, 129, 2560))
    atten = torch.randn((2, 41, 129,129))
    embed = embed.to(device)
    atten = atten.to(device)

    output = model(embed, atten)
    print(output.shape)   # [L*L,dim_out], [129*129,2]



    # from utils.vis_model import vis_model
    # vis_model(model, (embed, atten), filename="ResNet_li")

