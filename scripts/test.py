import torch
from utils.eval import eval_acc
from config import device

def test(test_dataloader, model, writer=None):

    total_test_step = 1
    total_test_loss = 0
    total_test_accuracy = 0

    test_data_size = len(test_dataloader)

    model.eval()
    l = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_dataloader:
            embed, atten, contact_label, L = data
            embed = embed.to(device)
            atten = atten.to(device)
            contact_label = contact_label.to(device)
            contact_label = contact_label.reshape(-1)

            pred = model(embed, atten)

            loss = l(pred, contact_label)
            accuracy = eval_acc(pred, contact_label)

            total_test_loss = total_test_loss + loss.item()
            total_accuracy = total_test_accuracy + accuracy

            if writer:
                writer.add_scalar("loss", loss, total_test_step)
                writer.add_scalar("avg_loss", total_test_loss/total_test_step, total_test_step)
                writer.add_scalar("accuracy", accuracy, total_test_step)
                writer.add_scalar("avg_accuracy", total_accuracy/total_test_step, total_test_step)

            total_test_step += 1
            print("No.{}: loss为{}，准确率为{}".format(total_test_step, loss, accuracy))

    print("整体测试集上的Loss: {}".format(total_test_loss/test_data_size))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
