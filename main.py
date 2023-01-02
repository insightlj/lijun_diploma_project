import sys
sys.path.append("D:\desktop\lijun_diploma_project")


from torch import optim, nn, save
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
train_file = "/home/rotation3/example/train_list.txt"
test_file = "/home/rotation3/example/valid_list.txt"

train_dataset = MyData(data_path, xyz_path, filename = train_file)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = MyData(data_path, xyz_path, filename = test_file)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # model from drn.py
model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1],
            channels=(64, 64, 64, 128, 256, 512, 512, 512),
            pool_size=24, arch='C')

# model from myself
# model = MyResNet()

# loss
l = nn.CrossEntropyLoss(reduction='none')
# l = nn.MSELoss(reduction='none')

# 训练开始之前添加tensorboard
writer = SummaryWriter("/home/rotation3/lijun-diploma/logs_train")

# train model
trained_model = train(train_dataloader, model, l, writer)

# save model parameters
from datetime import datetime as dt

a = dt.now()
filename = "dilated_resnet"
model_filename = filename + a.strftime("%m%d_%H%M%S") + ".pt"
model_filename = "model/checkpoint/" + model_filename

save(trained_model, model_filename)

# test model
# test(test_dataloader, model, l)

