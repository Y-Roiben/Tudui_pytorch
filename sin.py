# Designed by Roiben
# 开发时间： 2022-09-17 16:38
# Designed by Roiben
# 开发时间： 2022-08-29 19:15
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

time_step = 10
Step = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
x_np = np.sin(Step)
y_np = np.cos(Step)

"""plt.plot(Step, x_np, "b-", label='input(sin)')
plt.plot(Step, y_np, 'r-.', label="target(cos)")
plt.legend(loc='best')
plt.show()
"""


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=32,
                           num_layers=1)
        self.linear = nn.Linear(32, 1)

    def forward(self, input, state):
        r_out, state = self.rnn(input, state)
        r_out = self.linear(r_out)
        return r_out, state
        # outs = []
        # for time_step in range(r_out.size(0)):
        #     outs.append(self.linear(r_out[time_step, :, :]))
        # return torch.stack(outs, dim=0), h_n


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
loss_func = nn.MSELoss()
TIME_STEP = 10
plt.figure(1, figsize=(15, 5))
plt.ion()
h_n = torch.randn(1, 1, 32)
c_n = torch.randn(1, 1, 32)
state = (h_n, c_n)
# 训练数据包装, 将（100）的np, 转化为torch.Size([1, 100, 1])的tensor
for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, 10, dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[:, np.newaxis, np.newaxis])
    y = torch.from_numpy(y_np[:, np.newaxis, np.newaxis])
    prediction, (h_n, c_n) = rnn(x, (h_n, c_n))  # rnn output

    loss = loss_func(prediction, y)  # calculate loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # plt.plot(steps, y_np.flatten(), 'r-')
    # plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    # plt.draw()
    # plt.pause(0.05)

plt.ioff()
plt.show()
