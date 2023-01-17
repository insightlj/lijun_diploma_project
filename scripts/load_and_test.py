import torch
from torch.utils.tensorboard import SummaryWriter

from config import device
from main import test_dataloader
from scripts.test import test
from datetime import datetime as dt

def load_and_test(pt_name, ID):
    net = torch.load(pt_name)
    net.to(device)

    logs_name = "/home/rotation3/lijun-diploma/logs/test" + now.strftime("%m%d%H%M") + "/" + str(ID)
    writer_test = SummaryWriter(logs_name)

    test(test_dataloader, net, writer_test)


# if __name__ == "__main__":
#     from model.ResNet_li import Residual, MyResNet
#     net = Residual(100,100)
#     net = MyResNet()
#
#     now = dt.now()
#     load_and_test("model/checkpoint/random_model_param.pt", 0)


if __name__ == "__main__":
    import os

    pt_list = os.listdir("model/checkpoint/49_epoch_train_20230116")
    num_pt = len(pt_list)

    now = dt.now()
    for i in range(num_pt):
        pt_name = pt_list[i]
        ID = pt_name[:-14][10:]
        print("开始验证模型{}……".format(ID))
        load_and_test("model/checkpoint/49_epoch_train_20230116/" + pt_name, ID)
