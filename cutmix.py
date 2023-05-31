import trainer
import pre_treat

def cutmix_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):

    cutmix_train_loss = []
    cutmix_train_acc = []
    cutmix_test_loss = []
    cutmix_test_top1_acc = []
    cutmix_test_top5_acc = []

    for epoch in range(num_epochs):
        # 训练过程
        model.train()
        cutmix_loss = 0.0
        cutmix_correct = 0
        cutmix_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images, labels_a, labels_b, lam = pre_treat.cutmix(images, labels, beta=1.0)

            optimizer.zero_grad()
            model.train().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels_a) * lam + criterion(outputs, labels_b) * (1 - lam)
            loss.backward()
            optimizer.step()

            cutmix_loss += loss.item()
            _, predicted = outputs.max(1)
            cutmix_total += labels.size(0)
            cutmix_correct += predicted.eq(labels).sum().item()

        cutmix_train_loss.append(cutmix_loss / len(train_loader))
        cutmix_train_acc.append(cutmix_correct / cutmix_total)

        # 测试过程
        cutmix_loss, top1_acc, top5_acc = trainer.test(model, test_loader, criterion, device)
        cutmix_test_loss.append(cutmix_loss)
        cutmix_test_top1_acc.append(top1_acc)
        cutmix_test_top5_acc.append(top5_acc)

        scheduler.step() # 更新学习率

        print(f"Cutmix: Epoch [{epoch + 1}/{num_epochs}],"
              f"\tTrain Acc: {cutmix_train_acc[-1]:.4f},"
              f"\tTest Top1 Acc: {top1_acc:.4f},"
              f"\tTest Top5 Acc: {top5_acc:.4f}")
        
    # 模型保存
    torch.save(model, 'model_cutmix.pth')

    # 结果保存
    with open('cutmix.txt', 'w') as file:
        file.write("Train Loss:\n")
        for loss in cutmix_train_loss:
            file.write(f"{loss}\n")

        file.write("Train Acc:\n")
        for acc in cutmix_train_acc:
            file.write(f"{acc}\n")

        file.write("Test Loss:\n")
        for loss in cutmix_test_loss:
            file.write(f"{loss}\n")

        file.write("Test Top1 Acc:\n")
        for acc in cutmix_test_top1_acc:
            file.write(f"{acc}\n")

        file.write("Test Top5 Acc:\n")
        for acc in cutmix_test_top5_acc:
            file.write(f"{acc}\n")