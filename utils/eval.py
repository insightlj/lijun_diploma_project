import torch

def eval_acc(pred, contact_label, error_tolerant=3):
    # pred: [batch*L*L, dim_out]
    dim_out = pred.shape[1]
    len = pred.shape[0]
    pred = torch.argmax(pred, dim=1)

    if dim_out == 2:
        accuracy = 1 - (abs(pred - contact_label).sum() / len)

    else:
        diff = abs(pred - contact_label)
        diff[diff <= error_tolerant] = 0   # 对于非二分类问题，误差在3个bin之内代表预测正确
        diff[diff > error_tolerant] = 1
        accuracy = 1 - (diff.sum() / len)

    return accuracy


if __name__ == "__main__":
    pred = torch.ones((2*192*192,2))

    contact_label = torch.zeros((2*192*192))
    contact_label[0:2*20*20] = 1  # 预测错误的数量

    accuracy = eval_acc(pred, contact_label, error_tolerant=3)
    print(accuracy)
