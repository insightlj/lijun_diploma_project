# 预测结果的最后一个维度的各个值代表该点落在各个bin之间的概率。现在将概率加权求和，得出预测效果的直观表示

import torch


def merge_dim(pred):
    dim_out = pred.shape[-1]

    if dim_out == 2:
        pred_contact = pred[:, :, 0] * 0 + pred[:, :, 1] * 1

    else:
        for i in range(dim_out):
            if i == 0:
                pred_contact = pred[:, :, 0] * 1

            else:
                pred_contact += pred[:, :, i] * (i + 1)

        pred_contact /= dim_out

    return pred_contact


if __name__ == '__main__':
    # contact_label: [5,5]
    contact_label = torch.tensor(([0, 1, 0, 1, 1],
                                  [0, 0, 0, 1, 1],
                                  [0, 0, 0, 1, 0],
                                  [1, 1, 1, 0, 1],
                                  [0, 1, 0, 1, 0]))

    # pred: [5,5,4]
    pred = contact_label.unsqueeze(dim=2)
    pred = torch.concat((pred, pred), dim=2)
    pred = (pred + 0.8) / 2
    pred = torch.concat((pred, pred), dim=2)

    print(pred)
    print(merge_dim(pred))
