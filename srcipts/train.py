

def train(train_dataloader, model, l, optimizer, writer):
    total_train_step = 0

    model.train()

    for data in train_dataloader:
        input, contact_label, L = data
        output = model(input).reshape(192,192)

        # 这里对于L<192的蛋白质，应该只比较左上角(L,L)的部分
        if L < 192:
            loss = l(output[:L,:L], contact_label[:L,:L])
        else:
            loss = l(output, contact_label)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
