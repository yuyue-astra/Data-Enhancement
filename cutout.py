import trainer
import pre_treat

def cutout_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):

    cutout_train_loss = []
    cutout_train_acc = []
    cutout_test_loss = []
    cutout_test_top1_acc = []
    cutout_test_top5_acc = []

    for epoch in range(num_epochs):
        # 训练过程
        model.train().to(device)
        cutout_loss = 0.0
        cutout_correct = 0
        cutout_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = pre_treat.cutout(images, length=16)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cutout_loss += loss.item()
            _, predicted = outputs.max(1)
            cutout_total += labels.size(0)
            cutout_correct += predicted.eq(labels).sum().item()

        cutout_train_loss.append(cutout_loss / len(train_loader))
        cutout_train_acc.append(cutout_correct / cutout_total)

        # 测试过程
        cutout_loss, top1_acc, top5_acc = trainer.test(model, test_loader, criterion, device)
        cutout_test_loss.append(cutout_loss)
        cutout_test_top1_acc.append(top1_acc)
        cutout_test_top5_acc.append(top5_acc)

        scheduler.step() # 更新学习率

        print(f"Cutout: Epoch [{epoch + 1}/{num_epochs}],"
              f"\tTrain Acc: {cutout_train_acc[-1]:.4f},"
              f"\tTest Top1 Acc: {top1_acc:.4f},"
              f"\tTest Top5 Acc: {top5_acc:.4f}")

    # 模型保存
    torch.save(model, 'model_cutout.pth')

    # 结果保存
    with open('cutout.txt', 'w') as file:
        file.write("Train Loss:\n")
        for loss in cutout_train_loss:
            file.write(f"{loss}\n")

        file.write("Train Acc:\n")
        for acc in cutout_train_acc:
            file.write(f"{acc}\n")

        file.write("Test Loss:\n")
        for loss in cutout_test_loss:
            file.write(f"{loss}\n")

        file.write("Test Top1 Acc:\n")
        for acc in cutout_test_top1_acc:
            file.write(f"{acc}\n")

        file.write("Test Top5 Acc:\n")
        for acc in cutout_test_top5_acc:
            file.write(f"{acc}\n")