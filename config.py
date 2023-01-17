# unify device
import torch

epoch_num = 20
loss_fn = torch.nn.CrossEntropyLoss()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
