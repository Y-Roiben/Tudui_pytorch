# 111


import torch
from torchvision import models


vgg16 = models.vgg16()

# 保存方式1:模型结构+参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2：模型参数
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
torch.save(model.state_dict(), 'model_weights.pth')


