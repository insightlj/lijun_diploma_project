import torch
from utils.eval import eval_acc

def test(test_dataloader, model, writer):

    total_test_step = 1
    total_test_loss = 0
    total_test_accuracy = 0

    test_data_size = len(test_dataloader)

    model.eval()
    l = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_dataloader:
            input, contact_label, L = data
            pred = model(input)

            loss = l(pred, contact_label)
            accuracy = eval_acc(pred, contact_label)

            total_test_loss = total_test_loss + loss.item()
            total_accuracy = total_test_accuracy + accuracy

            writer.add_scalar("test_loss", total_test_loss/total_test_step, total_test_step)
            writer.add_scalar("test_accuracy", total_accuracy/total_test_step, total_test_step)
            total_test_step += 1

    print("整体测试集上的Loss: {}".format(total_test_loss/test_data_size))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))

