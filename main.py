import sys
sys.path.append("D:\desktop\lijun_diploma_project")


from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.MyData import MyData
from model.dilated_ResNet import DRN, Bottleneck, BasicBlock
from model.simple_ResNet import MyResNet

from scripts.test import test
from scripts.train import train

# data preparation
data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
train_file = "example/train_list.txt"
test_file = "example/valid_list.txt"

train_dataset = MyData(data_path, xyz_path, filename = train_file)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = MyData(data_path, xyz_path, filename = test_file)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model from drn.py
model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1],
            channels=(64, 64, 64, 128, 256, 512, 512, 512),
            pool_size=24, arch='C')

# model from myself
model = MyResNet()

# MODEL PARAMETERS
learning_rate = 0.01
momentum = 0.01

batch_size = 1
epoch = 10

# optimize
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)

# loss
l =nn.CrossEntropyLoss()

# 训练开始之前添加tensorboard
writer = SummaryWriter("/logs_train")

# train model
train(train_dataloader, model, l, optimizer, writer)

# test model
test(test_dataloader, model, l)

