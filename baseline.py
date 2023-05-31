import trainer
import pre_treat

def baseline_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):

    baseline_train_loss = []
    baseline_train_acc = []
    baseline_test_loss = []
    baseline_test_top1_acc = []
    baseline_test_top5_acc = []

    for epoch in range(num_epochs):
        # 训练过程
        train_loss, train_acc = trainer.train(model, train_loader, criterion, optimizer, scheduler, device)
        baseline_train_loss.append(train_loss)
        baseline_train_acc.append(train_acc)
        
        # 测试过程
        test_loss, top1_acc, top5_acc = trainer.test(model, test_loader, criterion, device)
        baseline_test_loss.append(test_loss)
        baseline_test_top1_acc.append(top1_acc)
        baseline_test_top5_acc.append(top5_acc)

        print(f"Baseline: Epoch [{epoch + 1}/{num_epochs}],"
              f"\tTrain Acc: {train_acc:.4f},"
              f"\tTest Top1 Acc: {top1_acc:.4f},"
              f"\tTest Top5 Acc: {top5_acc:.4f}")

    # 模型保存
    torch.save(model, 'model_baseline.pth')

    # 结果保存
    with open('baseline.txt', 'w') as file:
        file.write("Train Loss:\n")
        for loss in baseline_train_loss:
            file.write(f"{loss}\n")

        file.write("Train Acc:\n")
        for acc in baseline_train_acc:
            file.write(f"{acc}\n")

        file.write("Test Loss:\n")
        for loss in baseline_test_loss:
            file.write(f"{loss}\n")

        file.write("Test Top1 Acc:\n")
        for acc in baseline_test_top1_acc:
            file.write(f"{acc}\n")

        file.write("Test Top5 Acc:\n")
        for acc in baseline_test_top5_acc:
            file.write(f"{acc}\n")