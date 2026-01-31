import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    # 池化层输出特征图的尺寸计算方式和卷积层一样
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    # 遍历行和列
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max': # 注意切片语法
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], 
                  [3.0, 4.0, 5.0], 
                  [6.0, 7.0, 8.0]])
print('二维最大池化层')
print(pool2d(X, (2, 2)), end='\n\n')

print('二维平均池化层')
print(pool2d(X, (2, 2), 'avg'), end='\n\n\n')


X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

# 默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同 
# ⇒ 两个相邻池化层之间不会有重合
pool2d = nn.MaxPool2d(3)
print('二维最大池化层')
print(pool2d(X), end='\n\n')

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))

X = torch.cat((X, X + 1), 1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))