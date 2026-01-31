'''
一、和 softmax 相比, 代码没多大改动
1. 李沐老师说过: MLP 和 softmax 本质上没什么不同 ⇒ 所以在代码实现上需要改动的内容也比较少。
-----------------------
二、还讲了 为什么 MLP 比 SVN 发展更快？
1. 因为如果我采用 MLP, 效果不好
2. 那我的代码简单改一改, 就能成为 卷积神经网络 / RNN / Transformer
3. 而 SVN 改动就比较大, 改动难度也比较大
'''

import torch
from torch import nn
from tools.ch3 import load_data_fashion_mnist # 导入数据集加载函数
from tools.ch3 import train_ch3 # 导入训练函数
from tools.ch3 import predict_ch3 # 导入验证函数
import matplotlib.pyplot as plt # 用于画图

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

predict_ch3(net, test_iter)
plt.show()