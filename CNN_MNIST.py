import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

EPOCH = 10
# 准备数据集
train_data = datasets.MNIST(root="./dataset",
                            train=True,
                            download=True,
                            transform=ToTensor())

test_data = datasets.MNIST(root="./dataset",
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

print(test_data.data[0].size())
# print(train_data.targets[0]) 训练集第一个类别
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title("{}".format(train_data.targets[0].item()))
# plt.axis("off")  # 关闭网格线
plt.xticks([])  # 修改x坐标轴
plt.yticks([])  # 修改y坐标轴
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.net(x)


cnn = CNN()
print(cnn)

optimizer = optim.Adam(cnn.parameters(), lr=0.001)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
# GPU加速
if torch.cuda.is_available():
    cnn = cnn.cuda()
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

start_time = time.time()
# training and testing
best_EPOCH = 0
best_accuracy = 0
for epoch in range(EPOCH):
    # print("------第{}轮训练开始------".format(epoch + 1))

    # 训练步骤开始
    # 将模块设置为训练模式
    cnn.train()
    train_loss = 0
    for step, (b_x, b_y) in enumerate(train_dataloader):  # 分配 batch data, normalize x when iterate train_loader
        # GPU加速
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        train_loss += loss.item()
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        # if (step+1) % 100 == 0:
        #     end_time = time.time()
        #     print("耗时{}秒".format(end_time - start_time))
        #     print("第{}轮训练, 训练次数：{}, loss: {}".format(epoch+1, step+1, loss.item()))
    train_loss = train_loss / test_data_size
    end_time = time.time()
    # 测试网络开始
    # 将模块设置为评估模式
    cnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data_test in test_dataloader:
            img_test, target_test = data_test
            # GPU加速
            if torch.cuda.is_available():
                img_test = img_test.cuda()
                target_test = target_test.cuda()
            output = cnn(img_test)
            loss = loss_func(output, target_test)
            total_test_loss += loss.item()
            perds = output.argmax(1)
            accuracy = (perds == target_test).sum()
            total_accuracy = total_accuracy + accuracy

    # print("整体测试集上的Loss:{}".format(total_test_loss))
    accuracy = total_accuracy / test_data_size
    print("EPOCH:({}:10) train_loss:{:.4f} test_accuracy:{:.4f} time:{:.4f}".format(epoch + 1, train_loss, accuracy,
                                                                                    end_time - start_time))

    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_EPOCH = epoch
# print("训练效果最好的轮数为{}, 准确率为{}".format(best_EPOCH, best_accuracy))

plt.figure(figsize=(9, 9))
for i in range(9):
    data = test_data.data[i].reshape(1, 1, 28, 28)
    data = data / 255
    data = data.cuda()
    out = cnn(data)
    pred_y = torch.max(out, 1)[1]
    target = pred_y.cpu().numpy()
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_data.data[i].numpy(), cmap='gray')
    plt.title("target:{}-pred:{}".format(test_data.targets[i].item(), target.item()))
    plt.axis("off")

plt.show()
