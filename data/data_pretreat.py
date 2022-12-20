import numpy as np
import torch
from torch import nn

def data_pre(coor, embed, atten):
    # 输入为原始coor,embed,atten，输出为embed与atten的contact:(41,192,192)

    L = coor.shape[0]  # the length of aa

    # 1. 将L>192的数据随机采样，将L<=192的数据补零
    if L > 192:
        print("随机采样")
        a = range(L - 192)
        b = a + 192

        coor = coor[a:b]
        embed = embed[a:b]
        atten = atten[:, a:b, a:b]

    elif L < 192:
        print("补全至192")
        coor_ = torch.zeros(192 - L, 4, 3)
        coor = torch.concat((coor, coor_))

        embed_ = torch.zeros(192 - L, 2560)
        embed = torch.concat((embed, embed_))


        tmp1 = torch.zeros((41,192-L,L))
        tmp2 = torch.zeros((41,192,192-L))
        tmp = torch.concat((atten, tmp1), dim=1)
        atten = torch.concat((tmp, tmp2), dim=2)
        atten = atten.permute(1,2,0)

    # 2. 将(192,2560)转化为(192,192,5120)
    embed_zero = torch.zeros((192,192,2560))
    embed = embed.reshape(192,1,2560)
    embed_1 = embed + embed_zero
    print(embed.shape)

    embed_2 = embed_1.transpose(0,1)
    embed_comb = torch.concat((embed_1, embed_2),dim=2)
    print(embed_comb.shape)


    # 将5120降维至64
    red_lin = nn.Linear(5120, 64)
    embed_red = red_lin(embed_comb)

    # 将降维之后的embed与atten拼接起来
    input = torch.concat((embed_red, atten), dim=2)

    return input


if __name__ == '__main__':
    # load 10th aa
    coor = np.load("data\\coor.npy")
    embed = np.load("data\\embed.npy")
    atten = np.load("data\\attention.npy")
    coor = torch.from_numpy(coor)
    embed = torch.from_numpy(embed)
    atten = torch.from_numpy(atten)

    print(coor.shape, embed.shape, atten.shape)

    data_pre(coor, embed, atten)
    print(coor.shape, embed.shape, atten.shape)
