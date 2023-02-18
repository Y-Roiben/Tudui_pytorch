import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

rnn = nn.RNN(input_size=1,
             hidden_size=32,
             num_layers=1)
linear = nn.Linear(32, 1)
h_n = None

steps = np.linspace(0, 2 * np.pi, 10, dtype=np.float32,
                    endpoint=False)  # float32 for converting torch FloatTensor
y_np = np.cos(steps)
x_np = np.sin(steps)
x = torch.from_numpy(x_np[:, np.newaxis, np.newaxis])
print(x.size())

r_out, h_n = rnn(x, h_n)
output = linear(r_out)
print(output)
print(output.size())

outs = []
for time_step in range(r_out.size(0)):
    outs.append(linear(r_out[time_step, :, :]))
print(outs)

print('------------------------------------------------------')

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[10, 20, 30], [40, 50, 60]])
c = torch.tensor([[100, 200, 300], [400, 500, 600]])
d = torch.tensor([[1000, 2000, 3000], [4000, 5000, 6000]])
d0 = torch.stack([a, b, c, d], dim=0)
print(d0)
print(d0.size())
d1 = torch.stack([a, b, c, d], dim=1)
print(d1)
print(d1.size())
d2 = torch.stack([a, b, c], dim=2)
print(d2)
print(d2.size())

print('-----------------------------------------------')

step = 100
steps = np.linspace(0, 2 * np.pi, step, endpoint=True, dtype=np.float32)
for i in range(step):
    x_i = steps[:i]
    y = np.cos(x_i)
    x = np.sin(x_i)
    plt.cla()
    plt.plot(x, y)
    plt.pause(0.01)
plt.show()

# x = list(range(1, 21))  # epoch array
# loss = [2 / (i**2) for i in x]  # loss values array
# plt.ion()
# for i in range(1, len(x)):
#     ix = x[:i]
#     iy = loss[:i]
#     plt.cla()
#     plt.title("loss")
#     plt.plot(ix, iy)
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.pause(0.5)
# plt.ioff()
# plt.show()
