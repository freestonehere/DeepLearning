# Dropout （丢弃法）
1. dropout 主要用于全连接层
2. weight decay 用于各种模型均常见

## 一、概念理解
### （一）、动机
- 一个好的模型需要对输入数据的扰动鲁棒（Robust）
  - 使用**有噪音**的数据等价于**Tikhonov 正则**
  - 丢弃法：相当于在**层之间**加入噪音。

### （二）、无偏差的加入噪音

- 对 ${\bf x}$ 加入噪音得到 ${\bf x\prime}$，我们希望不改变原有期望 $\bf E$ 。即： $${\bf E[x\prime]=x}$$

- 丢弃法对每个元素进行如下扰动：

$$
x_i^\prime=\begin{cases}
0 & \sf{with\ pobability}\ \it p\\
{x_i\over1-p} & \sf otherwise\end{cases}
$$

- 相当于有一定概率 $p$ 使一个值变为零，否则使之变大。这种定义下，可保证期望 $E$ 不变。

### （三）、有意思的问题
1. 丢弃法的使用
   1. 只有在 **训练** 中才会用到丢弃法
   2. **预测（推理）** 的时候是不会用到丢弃的
   3. 因为 `dropout` 是更新参数的过程中使用的正则项。
      1. 训练会更新参数，但是推理不会更新参数
2. 本节中使用的 `dropout` 丢弃的实际上是 **隐藏层的输入**。
3. 先 **过拟合**，再 **正则化** 矫正是个可取的方案
   1. 意思就是先让它具有能力
   2. 然后再提高泛化能力

<br><br><br><br>

## 二、从零实现 `dropout layer`
```python
import torch
from torch import nn
from tools.ch3 import load_data_fashion_mnist # 导入数据集加载函数
from tools.ch3 import train_ch3 # 导入训练函数
import matplotlib.pyplot as plt # 用于画图

'''定义隐藏层'''
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

# 定义随机丢弃的概率
dropout1, dropout2 = 0.2, 0.5

'''Net 继承自 nn.Module'''
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        '''这里居然展平成二维了！以后应该不能全都是二维吧'''
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用 dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个 dropout 层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个 dropout 层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

plt.show()
```

### （一）、理解代码
1. **最终的输出层**
   1. 不需要非线性激活函数，
   2. 也不需要 dropout_layer
2. 多层感知机中**各层都是线性层**的时候，**各层的张量**形状
   1. `(num_inputs, num_hiddens1)`
   2. `(num_hiddens1, num_hiddens2)`
   3. `num_hiddens2, num_outputs`
   4. 这其实是矩阵乘法规律下的必然结果

<br>

### （二）、`mask` 的产生
```python
mask = (torch.rand(X.shape) > dropout).float()
return mask * X / (1.0 - dropout)
```

1. 讲解 `torch.rand()`
   1. [官网 doc - torch.rand](https://docs.pytorch.org/docs/stable/generated/torch.rand.html#torch.rand)
   2. Returns a tensor filled with random numbers from a uniform distribution on the interval $[0,1)$
2. 代码执行流程
   1. 按照 `X.shape` 产生 $[0, 1)$ 之间的随机数
   2. 和 `dropout` 比较，产生布尔值
   3. 但是布尔值不能进行 `*` 运算，所以后面加上 `.float()` 方法
3. 原来，**丢弃张量中的数值真的是按数字随机的**，并不是随机丢弃某一行 / 某一列！

<br>

### （三）、`nn.Module`
[官网 doc - torch.nn 首页](https://docs.pytorch.org/docs/stable/nn.html)
[官网 doc - torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

#### 1、总体认识
1. Base class for **all** neural network modules.
2. Your models should also subclass this class.
3. Modules can also contain other Modules, allowing them to be nested in a tree structure. 
   1. You can assign the submodules as regular attributes.

#### 2、`super.__init__()`
1. 这是 Python 中子类**调用父类**构造函数的写法（ Python3 中可简写为 `super().__init__()` ），
2. 必须调用这句话才能完成 `nn.Module` 基类的初始化，否则模型无法正常注册可训练参数、无法使用 `nn.Module` 提供的各类功能
   1. （比如无法用 `model.parameters()` 获取权重参数）。

<br><br><br><br>

## 三、简洁实现 `dropout layer`
```python
import torch
from torch import nn
from tools.ch3 import load_data_fashion_mnist # 导入数据集加载函数
from tools.ch3 import train_ch3 # 导入训练函数
import matplotlib.pyplot as plt # 用于画图

dropout1, dropout2 = 0.2, 0.5
num_inputs, num_outputs = 784, 10

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = load_data_fashion_mnist(batch_size)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

plt.show()
```

