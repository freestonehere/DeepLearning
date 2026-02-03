from torch import nn
from d2l import torch as d2l
from tools.ch6 import train_ch6
import matplotlib.pyplot as plt # 用于画图
from datetime import datetime

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

starttime = datetime.now() 
print(starttime) # 打印当前时间

train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

endtime = datetime.now()
print(endtime)
print(endtime-starttime)

plt.show()