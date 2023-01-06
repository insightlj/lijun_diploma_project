import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.contact_label_generate import caculator_label
from data.data_pretreat import data_pre


class MyData(Dataset):

    def __init__(self, data_path, xyz_path, filename, is_train_data):
        valid = [x.strip() for x in os.popen('cat '+ filename)]
        self.index = valid

        self.is_train_data = is_train_data

        self.coor = h5py.File(xyz_path, "r")
        self.embed_atten = h5py.File(data_path, "r")

    def __getitem__(self, idx):
        # 返回值: contact_label, 融合之后的input

        pdb_index = self.index[idx]
        gap = self.coor[pdb_index]['gap'][:]
        coor = self.coor[pdb_index]["xyz"][np.where(gap > 0)[0]]  # [L, 4, 3], 其中L是序列长度，4代表四个原子，顺序是CA， C， N和CB
        embed = self.embed_atten[pdb_index]['token_embeds'][0, np.where(gap > 0)[0]]
        atten = self.embed_atten[pdb_index]['feature_2D'][0, :, np.where(gap > 0)[0]][:, :, np.where(gap > 0)[0]]

        coor = torch.from_numpy(coor)
        embed = torch.from_numpy(embed)
        atten = torch.from_numpy(atten)

        embed, atten, a, L = data_pre(embed, atten, self.is_train_data)
        contact_label = caculator_label(coor, a, self.is_train_data)

        return embed, atten, contact_label, a, L

    def __len__(self):
        return len(self.index)

