from torch import optim


def train(train_dataloader, model, l, writer, learning_rate = 0.1, momentum = 0.01):

    total_train_step = 0

    model.train()

    for data in train_dataloader:
        input, contact_label, L = data
        # print(input.shape, contact_label.shape)
        contact_label = contact_label.reshape(192,192)
        output = model(input).reshape(192,192)

        # 这里对于L<192的蛋白质，应该只比较左上角(L,L)的部分
        if L < 192:
            loss = l(output[:L,:L], contact_label[:L,:L])
        else:
            loss = l(output, contact_label)

        # optimize
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        lr_scheduler.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.mean().item()))
            writer.add_scalar("train_loss", loss.mean().item(), total_train_step)

    return model
