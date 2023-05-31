import trainer
import pre_treat

def mixup_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    
    mixup_train_loss = []
    mixup_train_acc = []
    mixup_test_loss = []
    mixup_test_top1_acc = []
    mixup_test_top5_acc = []

    for epoch in range(num_epochs):
        # 训练过程
        model.train().to(device)
        mixup_loss = 0.0
        mixup_correct = 0
        mixup_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images, labels_a, labels_b, lam = pre_treat.mixup(images, labels, alpha=1.0)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_a) * lam + criterion(outputs, labels_b) * (1 - lam)
            loss.backward()
            optimizer.step()

            mixup_loss += loss.item()
            _, predicted = outputs.max(1)
            mixup_total += labels.size(0)
            mixup_correct += predicted.eq(labels).sum().item()

        mixup_train_loss.append(mixup_loss / len(train_loader))
        mixup_train_acc.append(mixup_correct / mixup_total)

        # 测试过程
        mixup_loss, top1_acc, top5_acc = trainer.test(model, test_loader, criterion, device)
        mixup_test_loss.append(mixup_loss)
        mixup_test_top1_acc.append(top1_acc)
        mixup_test_top5_acc.append(top5_acc)

        scheduler.step() # 更新学习率

        print(f"Mixup: Epoch [{epoch + 1}/{num_epochs}],"
              f"\tTrain Acc: {mixup_train_acc[-1]:.4f},"
              f"\tTest Top1 Acc: {top1_acc:.4f}"
              f"\tTest Top5 Acc: {top5_acc:.4f}")

    # 模型保存
    torch.save(model, 'model_mixup.pth')

    # 结果保存
    with open('mixup.txt', 'w') as file:
        file.write("Train Loss:\n")
        for loss in mixup_train_loss:
            file.write(f"{loss}\n")

        file.write("Train Acc:\n")
        for acc in mixup_train_acc:
            file.write(f"{acc}\n")

        file.write("Test Loss:\n")
        for loss in mixup_test_loss:
            file.write(f"{loss}\n")

        file.write("Test Top1 Acc:\n")
        for acc in mixup_test_top1_acc:
            file.write(f"{acc}\n")

        file.write("Test Top5 Acc:\n")
        for acc in mixup_test_top5_acc:
            file.write(f"{acc}\n")