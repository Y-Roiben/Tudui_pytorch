# 111
from torch import nn
from torchvision import models

vgg16_model_none = models.vgg16()
vgg16_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

print(vgg16_model_none)

# 增加net网络
'''
vgg16_model_none.add_module("add_linear", nn.Linear(1000, 10, bias=True))
print(vgg16_model_none)
'''
#  增加classifier网络层数
vgg16_model_none.classifier.add_module("7", nn.Linear(1000, 100, bias=True))
print(vgg16_model_none)

print(vgg16_model)

# 修改net某一个网络
vgg16_model.classifier[6] = nn.Linear(4096, 100)
print(vgg16_model)
