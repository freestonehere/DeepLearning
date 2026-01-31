'''
一. 明明 d2l 包里面有封装好的下列函数, 那为什么我还要自己再搞一个包？
1. 因为 d2l 包里面的代码是基于 jupyter 的 ⇒ 因此画图代码必须自己重写。
2. 另外, d2l 包里加载数据集的时候直接从网上 download, 而我的网络状况不佳,
    因此, 我更倾向于从本地加载。⇒ 所以加载数据集的代码也必须重写。
'''

import torch

# predict_ch3
from d2l import torch as d2l

# 加载训练集和测试集 load_data_fashion_mnist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 同一个 Module 内部是否可以相互调用？ train_ch3
# . 表示当前目录。突然明白 Python import 的使用方式了
from .Animator import Animator
from .Accumulator import Accumulator


'''加载训练集和测试集'''
# 虽然有下面的解释，但其实并不太理解。以后看代码再理解吧！
def load_data_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root="data/FashionMNIST", train=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="data/FashionMNIST", train=False, transform=transform)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter
    # 这里不能 yield，必须 return ！
    # yield 会让函数变成生成器函数，而非普通函数；
    # 调用该函数时，不会直接返回 train_iter 和 test_iter，而是返回一个生成器对象，
    # 需要通过 next() 或遍历才能拿到值
    # 比如 train_iter, test_iter = next(load_data_fashion_mnist(256, './data'))
    # 而且再说了，train_iter test_iter 本身就是生成器了，直接 return 即可


'''计算预测正确的数量'''
def accuracy(y_hat, y):  #@save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


'''计算在指定数据集上模型的精度'''
def evaluate_accuracy(net, data_iter):  #@save
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


'''训练模型一个迭代周期 (定义见第3章)'''
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


'''训练模型 (定义见第3章)'''
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


'''预测标签 (定义见第3章)'''
def predict_ch3(net, test_iter, n=6):  #@save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])