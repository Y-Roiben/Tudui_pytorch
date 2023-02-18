# 111
import torch
from torch import nn
from torch.nn import Module, Linear
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

Test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=True,
                             transform=ToTensor())

datas = DataLoader(Test_data, batch_size=50)


class linear(Module):
    def __init__(self):
        super(linear, self).__init__()
        self.liner = Linear(153600, 100)  # 输入尺寸，输出尺寸

    def forward(self, x):
        return self.liner(x)


roiben = linear()
for data in datas:
    img_tensor, target = data
    img_tensor1 = torch.flatten(img_tensor)
    print(img_tensor1.size())
    out = roiben.forward(img_tensor1)
    print(out.size())

m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
