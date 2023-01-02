import torch

def test(test_dataloader, model, writer):
    from main import l    

    total_test_step = 1
    total_test_loss = 0
    total_test_accuracy = 0

    test_data_size = len(test_dataloader)

    model.eval()

    with torch.no_grad():
        for data in test_dataloader:
            input, contact_label, L = data
            output = model(input).reshape(192,192)

            if L < 192:
                loss = l(output[:L,:L], contact_label[:L,:L])
                accuracy = abs(output[:L,:L], contact_label[:L,:L]).sum() / (L*L)
            else:
                loss = l(output, contact_label)
                accuracy = abs(output-contact_label).sum() / (192*192)

            total_test_loss = total_test_loss + loss.item()
            total_accuracy = total_test_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

