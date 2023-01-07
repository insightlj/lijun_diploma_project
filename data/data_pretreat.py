import numpy as np
import torch
import random

def data_pre(embed, atten, is_train_data):
    # 将train data截断
    # coor [L, 4, 3]
    # embed [L, 2560]
    # atten [41, L, L]

    L = embed.shape[0]

    # 对train data来说，应该将L>192的数据随机采样
    if is_train_data and L > 192:

        a = random.randint(0, L - 193)
        b = a + 192

        embed = embed[a:b]
        atten = atten[:, a:b, a:b]

        L = 192

    else:
        INF = 99999
        a = INF

    # 2. 将(L,2560)转化为(L,L,5120)
    embed_zero = torch.zeros((L,L,2560))
    embed.unsqueeze_(dim=1)  # (L,2560) -> (L,1,2560)
    embed_1 = embed + embed_zero
    embed_2 = embed_1.transpose(0,1)
    embed = torch.concat((embed_1, embed_2),dim=2)  # (L,L,5120)

    return embed, atten, a, L   # 对于长度大于192的蛋白质，需要data_pre与contact_label_generate中的截断位置对应起来


if __name__ == '__main__':
    # load 10th aa
    embed = np.load("data/dataset/embed.npy")
    atten = np.load("data/dataset/attention.npy")

    embed = torch.from_numpy(embed)
    atten = torch.from_numpy(atten)

    print(embed.shape, atten.shape)

    input = data_pre(embed, atten, is_train_data=True)

    print(input[1].shape)
