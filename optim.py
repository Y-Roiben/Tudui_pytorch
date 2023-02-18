# 111
from torch import optim
from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class Roiben(Module):
    def __init__(self):
        super(Roiben, self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        input = self.model(input)
        return input


test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=True,
                             transform=ToTensor())

test_loader = DataLoader(dataset=test_data,
                         batch_size=64,
                         shuffle=True)

net = Roiben()
net_optimizer = optim.SGD(net.parameters(), lr=0.01)  # SGD梯度下降
loss_cross = CrossEntropyLoss(reduction='mean')  # 交叉熵损失函数
for epoch in range(10):
    running_loss = 0.0
    for data in test_loader:
        img_tensor, target = data
        net_optimizer.zero_grad()  # 清零梯度
        out = net(img_tensor)
        loss = loss_cross(out, target)  # 计算损失
        loss.backward()       # 反向计算
        net_optimizer.step()     # 更新
        running_loss = running_loss + loss
        print('Epoch:', epoch, "|", 'loss:', running_loss.item())
