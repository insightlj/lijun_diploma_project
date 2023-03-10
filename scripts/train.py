from torch import optim

from config import device
from utils.eval_TPR import eval_TPR
from utils.eval_acc import eval_acc


def train(train_dataloader, model, loss_fn, writer, epoch_ID, learning_rate=5e-4, use_cuda=True):
    total_train_step = 1
    total_loss = 0
    total_accuracy = 0
    total_TPR = 0

    if epoch_ID == 0:
        if total_train_step < 1000:
            learning_rate = 5e-4
        elif 1000 <= total_train_step < 5000:
            learning_rate = 1e-4
        elif total_train_step >= 5000:
            learning_rate = 2e-5

    elif epoch_ID == 1:
        learning_rate = 1e-5

    else:
        learning_rate = 1e-6

    model.to(device)
    model.train()

    for data in train_dataloader:
        embed, atten, contact_label, L = data

        if use_cuda:
            contact_label = contact_label.to(device)
            embed = embed.to(device)
            atten = atten.to(device)

        pred = model(embed, atten)
        contact_label = contact_label.reshape(-1)

        loss = loss_fn(pred, contact_label)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        l = loss.item() if loss.item() < 1 else 1  # 剔除离群值
        total_loss = total_loss + l

        accuracy = eval_acc(pred, contact_label)
        total_accuracy = total_accuracy + accuracy

        TPR = eval_TPR(pred, contact_label)
        total_TPR = total_TPR + TPR

        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("avg_train_loss", total_loss / total_train_step, total_train_step)
            writer.add_scalar("avg_accuracy", total_accuracy / total_train_step, total_train_step)
            writer.add_scalar("avg_TPR", total_TPR/ total_train_step, total_train_step)
