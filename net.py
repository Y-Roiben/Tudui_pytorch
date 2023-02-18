# 111
import torch
from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils import tensorboard


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


test = Roiben()
print(test)

x = torch.ones((64, 3, 32, 32))
y = test(x)
print(y)
print(y.size())

writer = tensorboard.SummaryWriter("net")
writer.add_graph(test, x)
writer.close()
