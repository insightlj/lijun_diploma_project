import torch
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

from data.MyData import MyData
from model.ResNet_li import MyResNet
from scripts.train import train
from scripts.test import test
from utils.init_parameters import weight_init

from config import loss_fn as l

data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
train_file = "/home/rotation3/example/train_list.txt"
test_file = "/home/rotation3/example/valid_list.txt"

train_dataset = MyData(data_path, xyz_path, filename=train_file, train_mode=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = MyData(data_path, xyz_path, filename=test_file, train_mode=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = MyResNet()


def net_train(epoch_ID, expir_name=None, lr=5e-4):

    if expir_name:
        logs_dir_name = expir_name
        mode_dir_name = expir_name
    else:
        logs_dir_name = now.strftime("%m%d%H%M")
        mode_dir_name = str(epoch_num) + "_epoch" + "_" + now.strftime("%m%d%H%M")

    # logs save path
    logs_name = "/home/rotation3/lijun-diploma/logs/train_" + logs_dir_name + "/"+ "epoch" + str(epoch_ID)
    # /home/rotation3/lijun-diploma/logs/train_01171440/0/events.out.tfevents.1673836666.omnisky-GPU-201.36473.0
    writer_train = SummaryWriter(logs_name)

    train(train_dataloader, model, l, writer_train, epoch_ID, learning_rate=lr)

    # model parameters save path
    dir = "/home/rotation3/lijun-diploma/model/checkpoint/" + mode_dir_name + "/"
    filename = "epoch" + str(epoch_ID) + ".pt"
    # /home/rotation3/lijun-diploma/model/checkpoint/20_epoch_01171441/epoch1.pt

    if not os.path.exists(dir):
        os.mkdir(dir)
    torch.save(model, dir + filename)


def net_test(epoch_ID, expir_name=None):

    if expir_name:
        logs_dir_name = expir_name
    else:
        logs_dir_name = now.strftime("%m%d%H%M")

    logs_name = "/home/rotation3/lijun-diploma/logs/test_" + logs_dir_name + "/" + "epoch" + str(epoch_ID)
    writer_test = SummaryWriter(logs_name)

    test(test_dataloader, model, writer=writer_test)

if __name__ == "__main__":
    now = dt.now()
    model.apply(weight_init)
    expir_name = "3_epoch_try_0118"
    epoch_num = 3

    for epoch in range(epoch_num):
        net_train(epoch, expir_name)
        net_test(epoch, expir_name)


# if __name__ == "__main__":
#     learning_rate_1 = 1e-4
#     learning_rate_2 = 1e-5
#     learning_rate_3 = 5e-6
#
#     now = dt.now()  # Record the id of each training session
#     epoch_num = 30
#     model.apply(weight_init)
#
#     for i in range(10):
#         net_train(learning_rate_1)
#         net_test()
#
#     for j in range(10):
#         i = j + 10
#         net_train(learning_rate_2)
#         net_test()
#
#     for k in range(10):
#         i = k + 20
#         net_train(learning_rate_3)
#         net_test()
#
# if __name__ == "__main__":
#     now = dt.now()  # Record the id of each training session
#     epoch_num = 36
#     model.apply(weight_init)
#
#     learning_rate_1 = 5e-4
#     learning_rate_2 = 1e-4
#     learning_rate_3 = 1e-5
#     learning_rate_4 = 1e-6
#
#     for i in range(1):
#         net_train(learning_rate_1)
#         net_test()
#
#     for j in range(5):
#         i = j + 1
#         net_train(learning_rate_2)
#         net_test()
#
#     for k in range(15):
#         i = k + 6
#         net_train(learning_rate_3)
#         net_test()
#
#     for h in range(15):
#         i = h + 21
#         net_train(learning_rate_4)
#         net_test()