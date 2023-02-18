# 111
import time

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 准备数据集
train_data = datasets.CIFAR10(root="./dataset",
                              train=True,
                              download=True,
                              transform=ToTensor())

test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=True,
                             transform=ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为{}".format(train_data_size))
print("测试数据集长度为{}".format(test_data_size))

# 利用DatasetLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
print(test_data.data[0].shape)
# print(train_data.targets[0]) 训练集第一个类别
plt.imshow(train_data.data[0])
plt.title("{}".format(train_data.targets[0]))
# plt.axis("off")  # 关闭网格线
plt.xticks([])  # 修改x坐标轴
plt.yticks([])  # 修改y坐标轴
plt.show()


# 搭建神经网络
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


net = Conv()
print(net)
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
Learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), lr=Learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 20
start_time = time.time()
for i in range(epoch):
    # print("------第{}轮训练开始------".format(i + 1))

    # 训练步骤开始
    # 将模块设置为训练模式
    net.train()
    train_loss = 0
    for data_train in train_dataloader:
        img_train, target_train = data_train
        output_train = net(img_train)
        loss = loss_fn(output_train, target_train)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        total_train_step += 1

        # if total_train_step % 100 == 0:
        #     end_time = time.time()
        #     print(end_time - start_time)
        #     print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
    train_loss = train_loss / test_data_size
    end_time = time.time()
    # 测试网络开始
    # 将模块设置为评估模式
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data_test in test_dataloader:
            img_test, target_test = data_test
            output = net(img_test)
            loss = loss_fn(output, target_test)
            total_test_loss += loss.item()
            perds = output.argmax(1)
            accuracy = (perds == target_test).sum()
            total_accuracy = total_accuracy + accuracy

    accuracy = total_accuracy / test_data_size
    print("EPOCH:({}:20), train_loss:{:.4f} test_accuracy:{:.4f} time:{:.4f}".format(i + 1, train_loss, accuracy,
                                                                                     end_time - start_time))
    # torch.save(net, 'train_save1_{}.pth'.format(i + 1))
    # torch.save(net.state_dict(), 'train_{}.pth'.format(i + 1))
    # print("第{}轮模型已保存".format(i + 1))
