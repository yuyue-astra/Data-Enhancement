import torch

# 训练函数
def train(model, data_loader, criterion, optimizer, scheduler, device):
    model.to(device)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(data_loader)
    train_acc = correct / total
    
    scheduler.step()

    return train_loss, train_acc

# 测试函数
def test(model, data_loader, criterion, device):
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_top1 += predicted.eq(labels).sum().item()

            _, predicted_top5 = outputs.topk(5, dim=1)
            correct_top5 += predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5)).sum().item()

    test_loss = running_loss / len(data_loader)
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    return test_loss, top1_acc, top5_acc