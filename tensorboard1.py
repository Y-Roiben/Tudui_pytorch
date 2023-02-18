# 111

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=True,
                             transform=ToTensor())

test_loader = DataLoader(dataset=test_data,
                         batch_size=64)


class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.conv(x)


net = CONV()

writer = SummaryWriter("show")
step = 0
for data in test_loader:
    img, target = data
    out = net(img)
    out = out.reshape(-1, 3, 30, 30)
    # add_images 而不是add_image
    writer.add_images("CIFAR10", img, step)
    writer.add_images("CIFAR10_conv", out, step)
    step += 1

writer.close()
