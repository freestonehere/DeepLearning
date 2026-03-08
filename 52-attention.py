import torch
from torch import nn
import matplotlib.pyplot as plt # 用于画图
import tools.plot as plot
from tools import Animator
import tools.ch13 as ch13

# 生成数据集
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)
'''x_train 是 n_train 个样本的横坐标！'''

def f(x):
    return 2 * torch.sin(x) + x**0.8


y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
'''y_train 是 n_train 个样本的纵坐标！'''

x_test = torch.arange(0, 5, 0.1)  # 测试样本
'''x_test 是 n_test 个测试点的横坐标。
这 n_test 个点的纵坐标就是我们要预测的东西（预测依据: n_train 个训练样本）！
预测出 n_test 个点的纵坐标是 y_hat ，然后通过 y_hat 可视化预测函数。'''

y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数 n_test = 50（一共有 50 个测试样本）
print(f'\nn_test: {n_test}')


def plot_kernel_reg(y_hat):
    '''绘制训练样本、真实函数和预测函数'''
    plot.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    plt.plot(x_train, y_train, 'o', alpha=0.5)


'''平均汇聚'''
# y_train.mean() 是一个标量，在这里复制 n_test 次
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
fig = plt.gcf()
ch13.set_title(fig, '第一步：平均汇聚')


'''非参数注意力汇聚'''

'''
1. 「查询(Query)」= 测试点的横坐标，「键(Key)」= 训练点的横坐标，「值(Value)」= 训练点的标签。
2. 注意力权重由「查询 - 键」的欧式距离决定：距离越近，权重越大（对近邻点赋予更高关注）。
    ① 由于 X_repeat 是 (n_test, n_train)，因此共有 n_test 次查询，得到 n_test 个结果！
    ② 即 (n_test, 1) = (n_test, n_train) @ (n_train, 1) 左乘行变换，右乘列变换
3. 预测值 = 注意力权重 x 训练标签（加权求和）。
'''

# X_repeat 的形状:(n_test, n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
'''
每行都有相同的测试输入，是指如下！
tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])
'''
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train 包含着键。attention_weights 的形状：(n_test, n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
'''softmax 本身不会使张量降维，只是做归一化！
然后，注意力权重与 y_train 相乘（也就是矩阵乘法），得到【拟合输出】'''
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat 的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights, y_train)
'''(n_test, 1) = (n_test, n_train) @ (n_train, 1) 左乘行变换，右乘列变换'''

plt.figure() # 新开一块画布，否则会覆盖掉原来的图
plot_kernel_reg(y_hat)
fig = plt.gcf()
ch13.set_title(fig, '第二步：非参数注意力汇聚')


# 这里就算不新开画布，也不会覆盖原来的图，这是为什么？
plot.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
fig = plt.gcf()
ch13.set_title(fig, '第三步：非参数注意力汇聚，权重矩阵热图')


'''带参数注意力汇聚'''
# 批量矩阵乘法
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(f'\ntorch.bmm(X, Y).shape: {torch.bmm(X, Y).shape}')


weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
print('\ntorch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)):',
      f'{torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))}')


# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        '''self.w 是形状为 (1) 的张量，用于对权重进行统一缩放。
        另外，真奇妙，单参数统一缩放居然就能拟合一个比较复杂的非线性函数！'''

    def forward(self, queries, keys, values):
        '''① 本来，输入的 queries 是一维张量 (n_train) 或 (n_test)，也就是 (查询个数)。
        repeat_interleave() 后， queries 成为二维矩阵张量 (查询个数, 键值对个数)，
        二维 queries 行内都是相同数值，但是行间是不同数值！
        
        ② keys 和 values 本身就是二维张量 (查询个数, 键值对个数)：
        它们行内是不同数值，但是行间对应位置是相同数值！
        
        ③ 返回一维张量 (查询个数)'''
        # queries 和 attention_weights 的形状为 (查询个数, “键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values 的形状为(查询个数，“键－值”对个数)

        '''为什么注意力机制中要有这么多 repeat()?
        就是因为最后要加权求和（通过矩阵乘法实现的），把 repeat 后的二维张量还原成一维张量！
        但是注意： softmax 本身不改变张量维度，只是归一化！'''
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
        # attention_weights 张量形状 (查询个数, “键——值”对个数) ⇒ (查询个数, 1, “键——值”对个数)
        # values 张量形状 (查询个数, “键——值”对个数) ⇒ (查询个数, “键——值”对个数, 1)
        # 这代码写的真牛！


# 训练
# X_tile 的形状:(n_train, n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile 的形状:(n_train, n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
'''利用布尔张量索引达到修改形状的目的。

同时，定义【键矩阵】和【值矩阵】时，采用留一法：
使得每个训练点作为查询时，排除自身（避免过拟合）'''
# keys 的形状:(n_train, n_train-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values 的形状:(n_train, n_train-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))


net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
fig = plt.gcf()
ch13.set_title(fig, '第四步：训练过程图解')


# keys 的形状:(n_test, n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value 的形状:(n_test, n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plt.figure()
plot_kernel_reg(y_hat)
fig = plt.gcf()
ch13.set_title(fig, '第五步：预测函数（带参数注意力拟合）、真实函数、训练样本')


plot.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
fig = plt.gcf()
ch13.set_title(fig, '第六步：带参数注意力汇聚，权重矩阵热图')


plt.show()