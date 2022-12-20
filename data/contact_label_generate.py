import numpy as np
import torch


def dist(K):
    # 输入为coor_CB (L,3)

    T = K.t()

    K_2 = torch.pow(K, 2).sum(dim=1).unsqueeze(1)
    T_2 = K_2.t()

    dist_2 = K_2 + T_2 - 2 * K @ T
    dist_2[dist_2 < 0] = 0
    dist = pow(dist_2, 0.5)
    return dist


def label_from_dist(P):
    # 输入为距离矩阵(L,L), 输出为label(L,L)

    P[P <= 8] = 1
    P[P > 8] = 0
    return P


def caculator_label(coor):
    # 输入为原始坐标数据(L,4,3), 输出为基于CB计算的label(L,L)

    coor_CB = coor[:, 3, :]
    label = label_from_dist(dist(coor_CB))

    # 将(L,L)转化为(192,192)
    L = label.shape[0]
    if L < 192:
        tmp1 = torch.zeros((192 - L, L))
        tmp2 = torch.zeros((192, 192 - L))
        tmp = torch.concat((label, tmp1), dim=0)
        label = torch.concat((tmp, tmp2), dim=1)

    elif L > 192:
        a = range(L - 192)
        b = a + 192
        label = label[a:b, a:b]

    return label


if __name__ == '__main__':
    # load 10th aa
    coor = np.load("data\\coor.npy")
    coor = torch.from_numpy(coor)

    print(coor.shape)

    # torch.set_printoptions(profile="full")
    contact_label = caculator_label(coor)
    print(contact_label.shape)
