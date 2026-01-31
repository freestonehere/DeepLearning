import torch
from torch import nn
import matplotlib.pyplot as plt # 用于画图
from tools.ch3 import train_ch3 # 导入训练函数
from tools.ch3 import load_data_fashion_mnist # 导入数据集加载函数
from tools.ch3 import predict_ch3 # 导入验证函数

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# PyTorch 不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

predict_ch3(net, test_iter)

plt.show()