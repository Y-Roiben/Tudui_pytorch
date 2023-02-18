# 111
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToPILImage, ToTensor

input = torch.tensor([[1, 0.5, -1],
                      [1, -0.5, 2],
                      [1, 3, -1]])


class Relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.SIG = nn.Sigmoid()

    def forward(self, x):
        return self.SIG(x)


R = Relu()
output = R.forward(input)
# 　print(output)

S = Sigmoid()
# print(S.forward(input))

Test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=False,
                             transform=ToTensor())
i = 1
for data in Test_data:
    img_ten, target = data
    output = S.forward(img_ten)
    if i == 1:
        toPIL = ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        pic1 = toPIL(output)
        plt.imshow(pic1)
        plt.axis('off')
        plt.show()
        pic = toPIL(img_ten)
        plt.imshow(pic)
        plt.axis('off')
        plt.show()
        i = 0
