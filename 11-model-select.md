## 一、概念理解
### 1、模型选择：三种数据集
1. 训练数据集（好理解）：训练模型参数
2. 验证数据集（Validation Datasets）：选择模型超参数
   1. 验证集 和 训练集 不可以混用（验证数据集不可以用来调参）
   2. ~~验证数据集可以反复使用~~
3. 测试数据集（真实的数据集。但是在课程中，我们一般用不到测试数据集）
   1. 测试数据集不可以反复使用（只能使用一次，类比：高考）
   2. 而且测试数据集就是真真正正的真实数据集！
4. **注意：** 因为真实的测试集一般不会开源出来
   1. 所以，很多代码都会把 验证集（Validation）写成 test_data
   2. 所以，自己看代码的时候，一定要注意！
   3. 训练误差 and 泛化误差

        |训练误差|泛化误差|
        |:------|:------|
        |`train_dataset` 上的 error（训练集）|`validation_dataset` 上的 error（验证集）|

- 非大数据集采用 K-折交叉验证（ K-cross validation ）
  - 只是用于调参数，并不解决其他问题
  - K 越大，效果越好，但计算成本也会增大

### 2、过拟合和欠拟合
- 模型容量和数据集复杂度要适配！

    |              | 简单数据 | 复杂数据 |
    | :----------- | :------ | :------- |
    | 低模型复杂度 | 正常     | 欠拟合   |
    | 高模型复杂度 | 过拟合   | 正常     |

<br>

- 如何选择模型复杂度？
  - 这个新的数据集和之前哪个数据集很像？
  - 那就选那个数据集使用的模型的复杂度！

<br><br><br><br>

## 二、代码讲解 模型 and 过拟合 / 欠拟合
```python
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
    input_shape = train_features.shape[-1] # 获取 train_features 的列数（结合调用代码来看）
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
```

### 1、如何判定一个模型是 过拟合 / 欠拟合 / 正常拟合
- 首先：明确这里 **特征** 的概念，其实就是 **可训练参数**
1. 我们构造的数据集的情况：
   1. 我们是根据三次多项式构造的数据集 $y = 5 + 1.2x - 3.4 x^2 + 5.6 x^3$
   2. ⇒ 一共有 4 个特征 `[5, 1.2, -3.4, 5.6]` ，分别对应 `[0 次项, 1 次项, 2 次项, 3 次项]`
2. 你用 **4 个特征的模型**去模拟当然正好
3. 用 **2 个特征的模型**去模拟肯定要欠拟合
4. 用 **20 个特征的模型**去模拟当然会过拟合
5. **另外**，就是在模型**参数规模**相同的情况下，**参数值**的变化也会导致 过拟合 / 欠拟合 / 正常拟合（ 这个过程叫做 **调参** ）

### 2、语法
1. 注意 Python 中的切片
    ```python
    true[0:4]
    ```
    1. 对应的 `[起点: 终点: 步长] (不包括终点)`
    2. 具体索引就是 `0, 1, 2, 3`
2. `numpy.power` 对张量求幂次
3. `numpy.dot` 对向量做点积