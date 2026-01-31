import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from tools.ch3 import train_epoch_ch3 # 导入单次迭代训练函数
from tools import Animator
import matplotlib.pyplot as plt # 用于画图


# 逻辑并不难，但语法没见过，挺陌生
'''生成数据集: 遵从下面的表达式'''
# y = 5 + 1.2x - 3.4 \\frac{x^2}{2!} + 5.6 \\frac{x^3}{3!} + 正态噪音 N(0, 0.1^2)
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 明确切片 [0:4] 的含义

# features 维度 (n_train + n_test, 1)
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)

# 对 features 的每一行做广播幂运算，求出每一行的 0 ~ 19 次幂
# ⇒ poly_features 维度 (n_train + n_test, 20)
# ⇒ power 是幂运算
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    # 对 poly_features 的每一列做 gamma 的缩放
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!

# labels 的维度:(n_train+n_test, 1)
# 计算没有噪音的纯净标签
# ⇒ dot 是两个向量做点积
labels = np.dot(poly_features, true_w)
# 添加噪音后的标签
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray 转换为 tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]


'''评估给定数据集上模型的损失'''
def evaluate_loss(net, data_iter, loss):  #@save
    metric = d2l.Accumulator(2)  # 损失的总和, 样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1] # 获取 train_features 的列数（结合调用代码来看！）
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            # 明确 Animator 其实是个 class，这样就好理解了！
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

'''
# 三阶多项式拟合（正常拟合）
# 从多项式特征中选择前 4 个维度, 即 1, x, x^2/2!, x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
'''

'''
# 线性拟合（欠拟合）
# 从多项式特征中选择前 2 个维度, 即 1 和 x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
'''

'''
# 高阶多项式拟合（过拟合）
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
'''

plt.show()