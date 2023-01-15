import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.contact_label_generate import caculator_label

class MyData(Dataset):

    def __init__(self, data_path, xyz_path, filename, train_mode):
        valid = [x.strip() for x in os.popen('cat '+ filename)]
        self.index = valid

        self.train_mode = train_mode

        self.coor = h5py.File(xyz_path, "r")
        self.embed_atten = h5py.File(data_path, "r")

    def __getitem__(self, idx):
        pdb_index = self.index[idx]
        gap = self.coor[pdb_index]['gap'][:]
        coor = self.coor[pdb_index]["xyz"][np.where(gap > 0)[0]]  # [L, 4, 3], 其中L是序列长度，4代表四个原子，顺序是CA， C， N和CB
        embed = self.embed_atten[pdb_index]['token_embeds'][0, np.where(gap > 0)[0]]
        atten = self.embed_atten[pdb_index]['feature_2D'][0, :, np.where(gap > 0)[0]][:, :, np.where(gap > 0)[0]]

        coor = torch.from_numpy(coor)
        embed = torch.from_numpy(embed)
        atten = torch.from_numpy(atten)

        L = embed.shape[0]

        # 对train data来说，应该将L>192的数据随机采样
        if self.train_mode and L > 192:

            a = random.randint(0, L - 193)
            b = a + 192

            embed = embed[a:b]
            atten = atten[:, a:b, a:b]

            L = 192

        else:
            INF = 99999
            a = INF

        contact_label = caculator_label(coor, a, self.train_mode)

        return embed, atten, contact_label, L
        # embed:[L,2560], atten:[41,L,L], contact_label:[L,L]

   def __len__(self):
        return len(self.index)

