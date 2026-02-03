import torch
from torch import nn
from d2l import torch as d2l
from tools.ch6 import train_ch6
import matplotlib.pyplot as plt # 用于画图
from datetime import datetime

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测（推理）模式下，直接使用传入的全局均值和全局方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 假定 X 要么是 2 维 (batch_size, features)；
        # 要么是 4 维 (batch_size, channels, h, w)
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差（弹幕说类似均方滤波？）
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        '''
        终于明白了, 训练模式下不断更新 moving_mean 和 moving_var,
        1. 在多次训练中，使得 moving_mean 逼近 训练数据集 的整体 mean,
                        moving_var 逼近 训练数据集 的整体 var.
        2. 然后在推理中，直接使用 moving_mean 和 moving_var 直接对数据进行 norm
        '''
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        # 明确：这里的 shape 其实是用于初始化均值和方差的 shape
        # 并不是数据张量的 shape
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        
        '''Batch Normalization 这一层网络要维护 4 个参数'''
        # 这里的 gamma 和 beta 参数为什么不采用正态初始化？反而采用常数初始化？
        # 不是说常数初始化不能训练吗？
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        # 如果 gamma 初始化为 0，那么参数直接变成 0，没法训练了
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    '''
    每一层网络的前向计算要传入数据的！
    但是 class 本身初始化的时候，不需要传入数据。
    '''
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        '''注意: 不同平台上的 eps 还不一样！'''
        return Y

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
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