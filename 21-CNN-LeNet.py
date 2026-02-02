import torch
from torch import nn
from d2l import torch as d2l
from tools.ch3 import load_data_fashion_mnist # 加载 FashionMNIST 数据集
from tools import Animator # 用于绘制动画
import matplotlib.pyplot as plt # 用于画图

'''定义 LeNet 卷积模型'''
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

'''
历史上的 LeNet 数据集 32⨉32 是加入 padding 后的, 
这里把 padding 去掉, 所以成了 28⨉28
'''
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
    # 用于分析张量形状


'''在 FashionMNIST 上训练模型'''
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

'''模型推理：使用 GPU 计算模型在数据集上的精度'''
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device: # 如果没有指定 device
            # 那我就看你模型参数在哪个设备上，我就在哪个设备上计算
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    '''用 GPU 训练模型(在第六章定义)'''
    def init_weights(m): # 在函数定义里面再定义函数，这写法第一次见
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights) # 在内存上初始化参数
    print('training on', device) # 防止在 CPU 上训练
    net.to(device) # 将模型参数搬到 GPU 上
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            # 每次只将一个批次的输入搬到 GPU（并不是一次性将所有输入搬到 GPU 上，防止爆 GPU 内存？）
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward() 
            # 这里为什么不用 l.mean().backward() ？
            # 因为 nn.CrossEntropy 已经完成了「批次内样本损失求平均」存疑？
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    

lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

plt.show()