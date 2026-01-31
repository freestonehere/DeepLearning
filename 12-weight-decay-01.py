import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
from tools import Animator


'''生成数据集, 遵从下面的表达式'''
# y = 0.05 + \\sum_{i=1}^{d} 0.01 x_i + 正态噪音 N(0, 0.01^2)
# 训练集大小、测试集大小、输入大小、批次大小
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 张量 train_data 的维度是 (n_train, 1), train_data 其实就是矩阵！
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    # \lambda 在这里不体现, 在 train() 函数那里体现
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    # lambda 是 Python 中的匿名函数（没有正式函数名的函数），
    # 专门用来快速定义逻辑简单、只用一次的小型函数
    # net是 “新建了一个函数” （匿名）。
    # loss是 “给已有函数起了个新名字” （原函数有正式名称，不是匿名）
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        # 遍历一遍训练集, 先训练一遍
        for X, y in train_iter:
            # 增加了 L2 范数惩罚项，
            # 广播机制使 l2_penalty(w) 成为一个长度为 batch_size 的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0: # 每 5 轮训练才绘图一次，让图表更简洁
            '''
            评估的时候不能用 加入惩罚项后的损失函数！
            '''
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w 的 L2 范数是: ', torch.norm(w).item())

'''
train(lambd=0)
'''

train(lambd=3)

plt.show()
