import sys

sys.path.append("D:\desktop\lijun_diploma_project")

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

from utils.utils import save_model_parameters
from data.MyData import MyData
from model.ResNet_li import MyResNet
from scripts.train import train

now = dt.now()

data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
train_file = "/home/rotation3/example/train_list.txt"
test_file = "/home/rotation3/example/valid_list.txt"

train_dataset = MyData(data_path, xyz_path, filename=train_file, is_train_data=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = MyData(data_path, xyz_path, filename=test_file, is_train_data=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = MyResNet()

l = torch.nn.CrossEntropyLoss()

epoch_num = 30

for i in range(epoch_num):

    logs_name = "/home/rotation3/lijun-diploma/" + "logs_train" + now.strftime("%m%d_%H%M%S")
    writer = SummaryWriter(logs_name)

    trained_model = train(train_dataloader, model, l, writer, use_cuda=True)

    model_filename = save_model_parameters(trained_model, now, filename="ResNet-li")
