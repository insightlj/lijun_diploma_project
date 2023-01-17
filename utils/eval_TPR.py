import torch


def eval_TPR(pred, contact_label):
    pred = torch.argmax(pred, dim=1)
    diff = contact_label - pred

    FN = diff[diff == 1].sum()
    total_P_num = contact_label.sum()

    TPR = 1 - FN / total_P_num

    return TPR

