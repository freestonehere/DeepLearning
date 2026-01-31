import torch
from torch import nn
from d2l import torch as d2l

'''图像中的目标边缘检测: 给定输入和输出矩阵, 学习权重'''

'''计算二维互相关运算 correlation'''    
def corr2d(X, K):  #@save
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() 
            # 注意切片的语法；
            # 另外，这里的 * 是点积（真是没想到矩阵还有点积，不过就是模仿向量点积）
    return Y
    # shape 都是先行后列（ 0 轴是从外面开始计算的 ）

# X 是输入矩阵
X = torch.ones((6, 8))
X[:, 2:6] = 0

# K 是人为定义的卷积核
# 其实很好理解，如果全 1 / 全 0，那么最终输出就是 0
# 如果一个 1 一个 0，那么输出要么是 +1 要么是 -1
K = torch.tensor([[1.0, -1.0]])

# Y 是人为构造的输出矩阵
Y = corr2d(X, K)


# 构造一个二维卷积层，它具有 1 个输出通道和形状为（1，2）的卷积核
# 对于通道的解释：如果输入 / 输出通道数为 1 ⇒ 那么输入 / 输出图就是黑白图
# 因为我们已经知道 corr2d 函数中没有实现偏置，所以这里偏置为 False
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

