# 111
import torch
from torch import nn
from torchvision import models

# 打开方式1 --> 保存方式1
model1 = torch.load('vgg16_method1.pth')
# print(model1)


# 打开方式2 --> 保存方式2
model2 = torch.load('model_weights.pth')
# 输出为字典形式
print(model2)

# 打开方式2 --> 保存方式2
# 恢复成网络模型
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
# print(model)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
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


train = torch.load("cnn_train_save1_10.pth")
# print(train)
