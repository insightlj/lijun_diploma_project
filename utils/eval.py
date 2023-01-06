import torch


def eval_acc(pred, contact_label, error_tolerant=3):
    L = contact_label[0]
    dim_out = pred.shape[2]

    pred = torch.argmax(pred, dim=2)

    if dim_out == 2:
        accuracy = abs(pred - contact_label).sum() / (L * L)

    else:
        diff = abs(pred - contact_label)
        diff[diff <= error_tolerant] = 0   # 对于费二分类问题，误差在3个bin之内代表预测正确
        diff[diff > error_tolerant] = 1
        accuracy = diff.sum() / (L * L)

    return accuracy
