import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from baseline import baseline_training
from cutmix import cutmix_training
from cutout import cutout_training
from mixup import mixup_training
import draw
import pre_treat

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 超参数设置
num_epochs = 20 # 训练轮次
batch_size = 128 # 批数量
learning_rate = 0.1 # 初始学习率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载数据集
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

test_dataset = datasets.CIFAR100(root='./data', train=False, download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#### baseline ####
model = models.resnet18(pretrained=False, num_classes=100).to(device) # 加载模型
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

criterion = nn.CrossEntropyLoss() # 使用交叉熵作为损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # SGD优化器
scheduler = StepLR(optimizer, step_size=5, gamma=0.1) #学习率下降

baseline_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device)


#### cutmix ####
model = models.resnet18(pretrained=False, num_classes=100).to(device) # 加载模型
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

criterion = nn.CrossEntropyLoss() # 使用交叉熵作为损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # SGD优化器
scheduler = StepLR(optimizer, step_size=5, gamma=0.1) #学习率下降

cutmix_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device)


#### cutout ####
model = models.resnet18(pretrained=False, num_classes=100).to(device) # 加载模型
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

criterion = nn.CrossEntropyLoss() # 使用交叉熵作为损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # SGD优化器
scheduler = StepLR(optimizer, step_size=5, gamma=0.1) #学习率下降

cutout_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device)


#### mixup ####
model = models.resnet18(pretrained=False, num_classes=100).to(device) # 加载模型
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

criterion = nn.CrossEntropyLoss() # 使用交叉熵作为损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # SGD优化器
scheduler = StepLR(optimizer, step_size=5, gamma=0.1) #学习率下降

mixup_training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device)


# 绘制图像
draw.plot_metrics('baseline')
draw.plot_metrics('cutmix')
draw.plot_metrics('cutout')
draw.plot_metrics('mixup')

img, tar = next(iter(train_loader))
img_cutmix, _, _, _ = pre_treat.cutmix(img, tar, beta=1.0)
img_cutout = pre_treat.cutout(img, length=16)
img_mixup, _, _, _ = pre_treat.mixup(img, tar, alpha=1.0)

for i in range(3):
    image = img[i]
    image_cutmix = img_cutmix[i]
    image_cutout = img_cutout[i]
    image_mixup = img_mixup[i]
    
    plt.subplot(3, 4, i*4+1)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    
    plt.subplot(3, 4, i*4+2)
    plt.imshow(image_cutmix.permute(1, 2, 0))
    plt.axis('off')
    
    plt.subplot(3, 4, i*4+3)
    plt.imshow(image_cutout.permute(1, 2, 0))
    plt.axis('off')
    
    plt.subplot(3, 4, i*4+4)
    plt.imshow(image_mixup.permute(1, 2, 0))
    plt.axis('off')
    
plt.savefig('example.png')