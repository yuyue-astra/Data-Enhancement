import matplotlib.pyplot as plt

def plot_metrics(name):
    filename = name + '.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()

    train_loss = []
    train_acc = []
    test_loss = []
    test_top1_acc = []
    test_top5_acc = []

    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("Train Loss:"):
            current_section = train_loss
        elif line.startswith("Train Acc:"):
            current_section = train_acc
        elif line.startswith("Test Loss:"):
            current_section = test_loss
        elif line.startswith("Test Top1 Acc:"):
            current_section = test_top1_acc
        elif line.startswith("Test Top5 Acc:"):
            current_section = test_top5_acc
        else:
            current_section.append(float(line))

    plt.figure(figsize=(8, 4))        
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_top1_acc, label='Test Top1 Acc')
    plt.plot(test_top5_acc, label='Test Top5 Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    
    filename = name + '.png'
    plt.savefig(filename)
    
    plt.tight_layout()
    plt.show()