from torch import optim
from torch.utils.tensorboard import SummaryWriter

from config import device

def train(train_dataloader, model, loss_fn, writer, learning_rate=5e-4, use_cuda=True):
    total_train_step = 0

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

        print(pred.shape, contact_label.shape)

        loss = loss_fn(pred, contact_label)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    return model

if __name__ == "__main__":
    writer_train = SummaryWriter("temp")
    train(train_dataloader,model, l, writer_train)