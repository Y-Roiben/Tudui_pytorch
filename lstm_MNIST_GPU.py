import time

import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28  # rnn time step / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.001  # learning rate
DOWNLOAD_MNIST = True  # set to True if haven't download the data

# Mnist digital dataset
train_data = MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,  # download it if you don't have it
)

# Data Loader for easy mini-batch return in training
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# plot one example
print(train_data.data.size())  # (60000, 28, 28)
print(train_data.targets.size())  # (60000)

# 打印图片
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0], fontsize=30)
plt.axis("off")
plt.show()

test_data = MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=28,
            hidden_size=64,  # rnn hidden unit
            num_layers=2,  # number of rnn layer
            batch_first=True,
            # input & output will have batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # default = False
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, _) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = LSTM()
if torch.cuda.is_available():
    rnn = rnn.cuda()

print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# GPU加速
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

test_data_size = len(test_data)
# training and testing
start_time = time.time()

for epoch in range(EPOCH):
    # print("------------------第{}轮训练开始-----------------".format(epoch + 1))

    # 训练步骤开始
    # 将模块设置为训练模式
    rnn.train()
    train_loss = 0
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        # GPU加速
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        train_loss += loss.item()
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    train_loss = train_loss / test_data_size
    end_time = time.time()
    rnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    for data_test in test_loader:
        img_test, target_test = data_test
        img_test = img_test.view(-1, 28, 28)
        # GPU加速
        if torch.cuda.is_available():
            img_test = img_test.cuda()
            target_test = target_test.cuda()

        output = rnn(img_test)
        loss = loss_func(output, target_test)
        total_test_loss += loss.item()
        perds = output.argmax(1)
        accuracy = (perds == target_test).sum()
        total_accuracy = total_accuracy + accuracy

    accuracy = total_accuracy / test_data_size
    print("EPOCH:({}:10) train_loss:{:.4f} test_accuracy:{:.4f} time:{:.4f}".format(epoch+1, train_loss, 
                                                                                    accuracy, end_time - start_time))



plt.figure(figsize=(9, 9))
for i in range(9,18,1):
    data = test_data.data[i].view(-1, 28, 28)
    data = data / 255
    data = data.cuda()
    out = rnn(data)
    pred_y = torch.max(out, 1)[1]
    target = pred_y.cpu().numpy()
    plt.subplot(3, 3, i-8)
    plt.imshow(test_data.data[i].numpy(), cmap='gray')
    plt.title("target:{}-pred:{}".format(test_data.targets[i].item(), target.item()))
    plt.axis("off")
plt.show()
