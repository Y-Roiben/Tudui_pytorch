import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

data_csv = pd.read_csv('data.csv', usecols=[1])

# plt.plot(data_csv)
# plt.show()
# 数据预处理

data_csv = data_csv.dropna()  # 去掉na数据
dataset = data_csv.values  # 字典(Dictionary) values()：返回字典中的所有值
dataset = dataset.astype('float32')  # astype(type):实现变量类型转换
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))  # 将数据标准化到0~1之间


def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


data_X, data_Y = create_dataset(dataset)


class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


net = lstm_reg(2, 4)

net.load_state_dict(torch.load('net_params.pkl'))

data_X = data_X.reshape(-1, 1, 2)  # reshape中，-1使元素变为一行，然后输出为1列，每列2个子元素
data_X = torch.from_numpy(data_X)  # torch.from_numpy(): numpy中的ndarray转化成pytorch中的tensor（张量）
var_data = Variable(data_X)  # 转为Variable（变量）
pred_test = net(var_data)  # 产生预测结果
pred_test = pred_test.view(-1).data.numpy()  # view(-1)输出为一行

plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')  # loc显示图像  'best'表示自适应方式
plt.show()
