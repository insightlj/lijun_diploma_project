import torch
import torch.nn as nn

from utils.vis_model import vis_model

class BasicBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(BasicBlock, self).__init__()
        self.padding = (kernel_size - 1)//2      #这里的kernel_size必须是奇数
        self.norm1 = nn.InstanceNorm2d(dim)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.padding, stride=1)
        self.relu1 = nn.LeakyReLU()

        self.norm2 = nn.InstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.padding, stride=1)
        self.relu2 = nn.LeakyReLU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.norm2(x)
        x = self.conv2(x)
        x = self.relu2(x + residual)
        return x

class NET(nn.Module):
    def __init__(self, dim_in, resnet_channels, class_num, block_num, kernel_size=3, drop=0.1):
        super(NET, self).__init__()
        self.class_num = class_num
        self.reshape = nn.Conv2d(in_channels=dim_in, out_channels=resnet_channels, kernel_size=1)
        self.resnet = self.make_layers(resnet_channels, block_num, kernel_size)
        self.out = nn.Sequential(
            nn.Linear(in_features=resnet_channels, out_features=resnet_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=resnet_channels, out_features=class_num)
        )

    def make_layers(self, resnet_channels, block_num, kernel_size):
        layers = list()
        for _ in range(block_num):
            layers.append(BasicBlock(dim=resnet_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):          # x: [batch, dim_in=41, length, length]
        x = self.reshape(x)        # x -> x: [batch, dim_in=41, length, length] -> [batch, resnet_channels, length, length]
        x = self.resnet(x)         # x [batch, resnet_channels, length, length]
        x = x.permute(0, 2, 3, 1)  # x [batch, length, length, resnet_channels]
        x = self.out(x)            # x [batch, length, length, class_num]
        x = x.reshape(-1, self.class_num)
        return x

net = NET(dim_in=41, resnet_channels=128, class_num=36, block_num=8)
net.train()
print(net)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

for _ in range(8):
    # fake data
    batch = 2
    length = 156
    x = torch.rand(batch, 41, length, length)
    y = torch.randint(low=0, high=36, size=(batch, length, length))
    y = y.reshape(-1)
    # fake data end

    if _ == 0:
        vis_model(net, x, filename="ResNet-hu")

    pred = net(x)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()