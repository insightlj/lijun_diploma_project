import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.MyData import MyData
from main import data_path, xyz_path, test_file, now
from scripts.test import test
from config import device

test_dataset = MyData(data_path, xyz_path, filename=test_file, train_mode=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# net = torch.load("model/checkpoint/ResNet-li-epoch00116_022438.pt")
net = torch.load("random_model_param.pt")

net.to(device)

logs_name = "/home/rotation3/lijun-diploma/logs/test" + now.strftime("%m%d_%H%M%S") + "/" + \
            "logs_train_" + now.strftime("%m%d_%H%M%S") + "_" + str(0)
writer_test = SummaryWriter(logs_name)

test(test_dataloader, net, writer_test)

