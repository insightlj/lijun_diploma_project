from torch import optim
import torch


def train(train_dataloader, model, loss_fn, writer, use_cuda=True):

    total_train_step = 0
    model.train()

    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    for data in train_dataloader:
        embed, atten, contact_label, L, a = data

        if use_cuda:
            contact_label = contact_label.to(device)
            embed = embed.to(device)
            atten = atten.to(device)

        contact_label = contact_label.reshape(-1)
        pred = model(embed, atten)

        loss = loss_fn(pred, contact_label)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.mean().item(), total_train_step)

    return model