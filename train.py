import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "D:\\DEMO\\dataset\\2.jpg"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
image = transform(image)


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


net = torch.load("cnn_train_save1_9.pth", map_location=torch.device('cpu'))  # 模型为GPU 使用cpu需特殊指定
image = torch.reshape(image, (1, 3, 32, 32))
# 使用GPU训练
# image = image.cuda()
net.eval()
with torch.no_grad():
    output = net(image)
print(output)
print(output.argmax(1))
