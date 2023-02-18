import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Hyper Parameters
EPOCH = 2  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28  # rnn time step / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.01  # learning rate
DOWNLOAD_MNIST = True

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


# 显示一张训练集的图片
print(train_data.data.size())  # (60000, 28, 28)
print(train_data.targets.size())  # (60000)
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.axis("off")

plt.show()


# convert test data into Variable, pick 2000 samples to speed up testing
test_data = MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=28,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,
            # MNIST 格式为(batch, high, weight)， 不可以修改格式
            # input & output will have batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # default = False
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = LSTM()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
test_data_size = len(test_data)
# training and testing
for epoch in range(EPOCH):
    print("------第{}轮训练开始------".format(epoch+1))
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if (step + 1) % 100 == 0:
            print("第{}步, loss={}".format(step + 1, loss))
    rnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    for data_test in test_loader:
        img_test, target_test = data_test
        img_test = img_test.view(-1, 28, 28)
        output = rnn(img_test)
        loss = loss_func(output, target_test)
        total_test_loss += loss.item()
        perds = output.argmax(1)
        accuracy = (perds == target_test).sum()
        total_accuracy = total_accuracy + accuracy
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))

'''if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
'''
