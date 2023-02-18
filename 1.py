# 111
import io

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

Relu = nn.ReLU()
Sigmoid = nn.Sigmoid()
input = torch.rand((3, 3))
print(input)
a = torch.tensor([[1, 3, -1],
                  [4, 1, -4],
                  [3, 5, 3]])
print(Relu(a))
print(Sigmoid(Relu(a)))

m = nn.Linear(2, 3, bias=True)
input = torch.randn(3, 3, 2)
print('in:', input)
output = m(input)
print('out:', output)

# dim=0 跨行球softmax
soft = nn.Softmax(dim=0)
input = torch.randn(3, 3)
output = soft(input)
print('----------------------------')
print(input)
print(output)

a = torch.ones(2, 3)
print(a)

a1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

print(a1.size())
# torch.Size([2, 3])
print(a1.shape)
# torch.Size([2, 3])
print(a1.reshape(3, 2))
print(a1.view(-1, 2))
