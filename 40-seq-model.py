import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
import tools.plot as plot
import tools.ch13 as ch13

'''明确：这里全部使用马尔科夫假设，没有使用潜变量模型！'''

T = 1000  # 总共产生 1000 个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
plot.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
fig = plt.gcf()
ch13.set_title(fig, '第一步：展示人造数据')


tau = 4
# 这里的二维张量 features 其实是有 T - tau 个【批次】，每个批次有 tau 的【样本】。
# 一个【批次】负责预测一个新的数据。正好符合马尔科夫假设！
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # 明确：x 是一维数据；并且切片包含起点，不包含终点
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前 n_train 个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机（拥有 2 个全连接层的多层感知机）
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss 计算平方误差时不带系数 1/2
# 因为这是回归问题、不是分类问题，所以采用这个损失函数
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)


onestep_preds = net(features)
plt.figure() # 新开一块画布，否则这里会覆盖前面已经画好的内容！
plot.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
fig = plt.gcf()
ch13.set_title(fig, '第二步：展示预测过程（ 1-step preds ）')


multistep_preds = torch.zeros(T)
# 明确：n_train 其实就是【批次数】，因为前面就是用的 features[:n_train]
# 而训练要用到的一维张量样本是 x[:n_train + tau]，看我之前画的图就很好理解
# multistep_preds 是一维张量
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    # multistep_preds 从 n_train + tau 开始，就全部都是自己预测了！
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

plt.figure()
plot.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
fig = plt.gcf()
ch13.set_title(fig, '第三步：展示预测过程（ 1-step preds, multistep preds ）')


max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列 i（i < tau）是来自 x 的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列 i（i >= tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
# 一共有 T-(tau-1)-max_step 个批次；每个批次有 tau+max_step 个样本（tau 个观测值 + max_step 个预测结果）
# 关于维度的讲解可以看我的图片！
# 每个批次前 tau 个样本都是观测数值；从 tau+1 个样本开始，都是预测的数值
'''【总结】：每个批次都是提供 tau 个样本，预测 max_step 个数据！
疑问：这样的话，那是如何做到提供 tau 个样本分别预测 1 个、4 个、16 个、64 个的呢？
【答】：这是通过【展示】来实现的！每个批次只取 1 个预测值，也就是说，
每个批次都是有 max_step 个预测结果，但是我每次只取对应 step 的一个预测结果！''' 

# 提供 tau 个样本，预测 step 个数据！
steps = (1, 4, 16, 64)
plt.figure()
# 这里使用 plot.plot 的时候：features 的索引很好理解，但是 time 的索引是什么意思呢？
# time 的索引其实是非常极致的索引（左右都正好不多不少）！
# 因为 plot.plot 每个批次只取一个预测结果，也就是对应的 step 步的那个结果
# 不会在 x[:tau-1+i] 和 x[T-max_steps+i:] 的范围上【预测】
# 因此，不用画这些范围！
plot.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
fig = plt.gcf()
ch13.set_title(fig, '第四步：展示预测过程（ 1-step preds, 4-step, 16-step and 64-step preds ）')

plt.show()