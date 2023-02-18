# Designed by Roiben
# 开发时间： 2022-09-19 16:58
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=3)

a = torch.randn(3, 3)
print(a)
out = bn(a)
mean = a[0].mean()
print(mean)
var = (a-mean)**2
print(var)
b = (a-mean)/(var**(1/2))
print(b)

print(out)
