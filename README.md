# Data-Enhancement
## 1.主程序
main.py为运行的主程序，包括超参数的设置，数据集与模型的加载，以及数据的保存。
## 2.模型训练与测试
包括baseline.py, cutmix.py, cutout.py, mixup.py，分别进行baseline以及三种数据增强方法的模型训练、测试步骤及模型的保存。
## 3.辅助函数
draw.py：用于读取训练过程中保存的数据并绘制loss及accuracy曲线图像；
pre-treat.py：用于cutmix, cutout, mixup具体操作的实现；
trainer.py：包括训练函数以及测试函数。
