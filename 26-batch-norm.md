# batch-norm | 批量归一化
## 一、概念理解
### （一）、网络越深产生的问题
- 反向传播，损失的梯度从输出层向后传，靠近输出的层训练较快
  - 梯度往下越传递越小（小数相乘）
  - 由于梯度较小 ⇒ 因此，权重很快就不变化了
- 数据在最底部（ **数据在靠近输入的那一端** ）
  - 靠近数据的底部层训练较慢
  - 底部层一变化，所有都得跟着变，相当于低层特征改变，不断抽象得到的高层特征也会随之改变
  - 顶部的那些层需要重新学习多次
  - 导致收敛变慢
  - **可能**这个问题只会在 **分布式训练** 的过程中遇到？
    - 毕竟单卡训练的时候，`train` 函数其实就是一个 `for loop`（意味着 靠近输出层 一定是在 靠近输入层 的基础上完成的！）
    - 上面的想法其实不对，虽然代码确实是串行的，但是实际**张量计算**却是**并行**的
    - 那么也就是说，即便是单卡训练，只要你的网络层数很深，同样会出现上述问题！
    - 行了，先理解到这里吧。不要钻牛角尖了。

1. 对于典型的多层感知机或卷积神经网络。
   1. 当我们训练时，**中间层中的变量**（例如，多层感知机中的仿射变换输出）可能具有更广的变化范围：
      1. 不论是沿着从输入到输出的层，跨同一层中的单元，
      2. 或是随着时间的推移，模型参数的随着训练更新变幻莫测。 
   2. 批量归一化的发明者非正式地假设，**这些变量分布**中的这种**偏移**可能会阻碍网络的收敛。 
   3. 直观地说，我们可能会猜想，如果一个层的可变值是另一层的 100 倍，这可能需要对学习率进行补偿调整。
2. 同时，更深层的网络很复杂，容易过拟合。这意味着正则化变得更加重要。
### （二）、数学上的批量归一化（ batch normalization ）
1. 批量归一化（ batch normalization ） [[Ioffe & Szegedy, 2015]](https://arxiv.org/abs/1502.03167)，是一种流行且有效的技术，可持续加速深层网络的收敛速度。
   1. 再结合后期介绍的残差块，
   2. 批量归一化使得研究人员能够训练 100 层以上的网络。
2. 批量规范化应用于单个可选层（也可以应用到所有层），其原理如下：在每次训练迭代中，我们首先**规范化输入**，即通过减去其**均值**并除以其**标准差**，其中 均值 和 标准差 均**基于当前小批量处理**。
3. 批量归一化固定每一个小批量（在不同层输出）里面的均值和方差： $$\mu_B={1 \over |B|} \sum_{i \in B} x_i \\ \sigma_B^2={1 \over |B|} \sum_{i \in B} (x_i-\mu_B)^2 + \epsilon$$
   1. 其中 $B$ 指一个批量 Batch，$\epsilon$ 为一个很小的数，防止方差为零，在下文无法进行除零运算
      1. ![s](https://theaisummer.com/static/d42512016d9b99eabb69a61bb295cd50/2e9f9/normalization.png)
   2. 然后再通过下式对每个批量在不同层的输出值数据做额外的调整，应用**比例系数** $\gamma$ 和 **比例偏移** $\beta$，将每层输出值固定为均值为 ${\beta}$ 、方差为 ${\gamma}$ 的分布：$$x_{i+1}=\gamma{x_i-\mu_B \over \sigma_B} + \beta$$
      1. $\mu_B$ 和 $\sigma_B$ 是小批量的随机噪声
      2. $\gamma$ 和 $\beta$ 是学到的比较稳定的 标准差 + 均值
      3. **一种解释**：这就使得数据确实有一定的噪声，但是又不会变化地特别剧烈。

### （三）、代码中的批量归一化：批量归一化层
- 比例系数 ${\gamma}$ 和偏移系数 ${\beta}$ 是学习出来的
- 批量归一化是一个**线性变换**（因为这是一个**一次函数**）
- **作用位置**
  - 全连接层和卷积层输出上，**激活函数之前**
    - 因为一般激活函数（如 relu） 会将数据映射为正数，所以不能再带回正负各异的状态
  - 全连接层和卷积层输入上
- 对于全连接层，作用在**特征维**（独立改变每个特征的分布）
  - **注意**：全连接层的张量一共就 2 个维度 `(batch_size, features)`
  - 关于全连接层中**特征**的理解：[11-model-select.md](11-model-select.md)
- 对于卷积层，作用于**通道维**（即一个滑动窗口里像素的特征）
  - **注意**：卷积层的张量形状一共 4 个维度 `(batch_size, channels, h, w)`
  - 对于一个卷积层来说，你的**样本数量**其实是 `batch_size x h x w`
  - 那么，你的特征数就是 `channels` ！

### （四）、批量归一化的作用
- 可以**加速收敛并让训练更稳定**（因为可以用更大的学习率，而防止学习率过大造成的无法收敛抖动或者靠近输出层梯度爆炸的问题）
- 一般不改变模型的精度
- 只有**批量足够大**和运用在深层网络时，批量归一化效果才能有效且稳定。如果我们尝试使用大小为1的小批量应用批量规范化，将无法学到任何东西。
  - 因为在减去均值之后，每个隐藏单元将为0。 
  - 所以，只有使用足够大的小批量，批量规范化这种方法才是有效且稳定的。 
  - 请注意，在应用批量规范化时，批量大小的选择可能比没有批量规范化时更重要


### （五）、批量归一化作用的原理
- 最初的论文表示可以减少内部协变量转移
- 后续论文指出 batch normalization 相当于在小批量里**增加噪音**$\mu,\sigma$，对数据进行了随机偏移和缩放（目前还没有一个统一的结论）
- **没必要和丢弃法混合使用**（在[番外 04-Kaggle 竞赛实践经验](extra/番外04-Kaggle竞赛实践经验.md)一篇中有相关实践证明）

### （六）、四种归一化
目前常用的有**BatchNorm、LayerNorm、InstanceNorm、GroupNorm**四种归一化方法。

- **BatchNorm**： 是在 batch 上作用于每个 Channal 维的归一化，多用于 CNN，对小 batchsize 效果不好
- **LayerNorm**： 在通道方向上作用于一个 batch 中一个样本（如一张图的所有通道）的归一化，主要对 RNN 作用明显，现多用于 Transformer，
- **InstanceNorm**： 作用于一个样本的一个通道的归一化，多用在风格化迁移
- **GroupNorm**： 将 channel 分组，然后再做归一化, 在 batchsize<16 的时候, 可以使用这种归一化

具体关于四种归一化的综述，可以参考 AISummer 这篇文章 👉[Aisummer-Norm](https://theaisummer.com/normalization/) 和这篇 [知乎专栏](https://zhuanlan.zhihu.com/p/395855181)

###  （七）、有意思的问题
1. 这里的 batch normalization **其实只是简单的**：均值变 0、方差变 1
   1. 并不是数学上的 Normalization
   2. 另外，这里的 batch norm 其实是 **跨样本** 做的 batch norm
2. Batch Normalization 一般用于深层网络，浅层网络不太会用。
   1. 因为这就是针对深层网络的问题而产生的
3. Batch Normalization 确实做的是线性变换，但是却不能用线性层来替代
   1. 因为 Batch Normalization 主要强调的 **数据分布归一化**
   2. 只用一个线性层的话，未必能达到数据分布归一化的效果

<br><br><br><br>

## 二、从零实现 - 在 `21-CNN-LeNet.py` 的基础上加上 Batch Norm
```python
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测（推理）模式下，直接使用传入的全局均值和全局方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 假定 X 要么是 2 维 (batch_size, features)；
        # 要么是 4 维 (batch_size, channels, h, w)
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差（弹幕说类似均方滤波？）
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        '''
        终于明白了, 训练模式下不断更新 moving_mean 和 moving_var,
        1. 在多次训练中，使得 moving_mean 逼近 训练数据集 的整体 mean,
                        moving_var 逼近 训练数据集 的整体 var.
        2. 然后在推理中，直接使用 moving_mean 和 moving_var 直接对数据进行 norm
        '''
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        # 明确：这里的 shape 其实是用于初始化均值和方差的 shape
        # 并不是数据张量的 shape
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        
        '''Batch Normalization 这一层网络要维护 4 个参数'''
        # 这里的 gamma 和 beta 参数为什么不采用正态初始化？反而采用常数初始化？
        # 不是说常数初始化不能训练吗？
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        # 如果 gamma 初始化为 0，那么后续【数据】直接变成 0，没法训练了
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    '''
    每一层网络的前向计算要传入数据的！
    但是 class 本身初始化的时候，不需要传入数据。
    '''
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        '''注意: 不同平台上的 eps 还不一样！而且非常影响结果'''
        return Y
```

### （一）、理解代码
#### 1、什么时候用 `batch_norm` ？
1. 很显然，**训练的时候要用** batch_norm（由于上面的原因）
2. **推理**的时候也要用 batch_norm，因为
   1. 防止推理的时候数据爆掉？（好像没讲？）

#### 2、`net`（ `整个` 的模型 ）和 `每一层` 网络 之间的关系
1. `net` 要接收**每一批次**的**输入数据**（输入张量）
    1. 下面是**单层线性回归**模型的代码。[08-LinearRegression-01.py](08-LinearRegression-01.py)
    ```python
    def linreg(X, w, b):  #@save
        '''线性回归模型'''
        return torch.matmul(X, w) + b

    net = linreg
    ```

#### 3、为什么 class `BatchNorm` 初始化 $\gamma$ 和 $\beta$ 用常数初始化，却还能正常训练？
1. 因为这是针对数据增加噪声，和 普通全连接 / 卷积 不一样
2. 而且，$\gamma$ 采用常数 1 初始化，$\beta$ 采用常数 0 初始化 还是最好的操作
   1. 因为 $\gamma$ 是学到的标准差，$\beta$ 是学到的均值
   2. 这样初始化也有利于**让数据**均值为 0、标准差为 1

#### 4、明确 class `BatchNorm` 的 可训练参数 和 非可训练参数
1. $\gamma$ 和 $\beta$ 是 class `BatchNorm` 的可训练参数
2. `moving_mean` 和 `moving_var` 则是非可训练参数 

### （二）、`torch.Tensor.mean()` 方法
[官网 doc - torch.Tensor.mean() 方法](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.mean.html)
```python
torch.mean(input, dim, keepdim=False, *, dtype=None, out=None)
```
1. Returns the mean value of each row of the input tensor in the given dimension `dim`. If `dim` is a list of dimensions, reduce over all of them.
2. 例子代码
   1. 要求数据类型必须是 `float` 或 `complex`
   2. **`dim`** 参数 (int or tuple of ints, optional) —— 
      1. the dimension or dimensions **to reduce**. 
      2. If `None`, all dimensions are reduced.
   3. 用的话，其实也有个口诀：**想要保留哪个维度，那就不要添加哪个维度！**
    ```python
    >>> a=torch.tensor(range(12), dtype=float)
    >>> a=a.resize(3, 4)
    >>> a
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]], dtype=torch.float64)
    >>> a.mean(dim=0)
    tensor([4., 5., 6., 7.], dtype=torch.float64)
    >>> a.mean(dim=0, keepdim=True)
    tensor([[4., 5., 6., 7.]], dtype=torch.float64)
    >>> a=a.resize(3, 2, 2, 1)
    >>> a
    tensor([[[[ 0.],
            [ 1.]],

            [[ 2.],
            [ 3.]]],


            [[[ 4.],
            [ 5.]],

            [[ 6.],
            [ 7.]]],


            [[[ 8.],
            [ 9.]],

            [[10.],
            [11.]]]], dtype=torch.float64)
    >>> a.mean(dim=(0,2,3))
    tensor([4.5000, 6.5000], dtype=torch.float64)    
    >>> a.mean(dim=(0,2,3), keepdim=True)
    tensor([[[[4.5000]],

            [[6.5000]]]], dtype=torch.float64) 

    '''
    推演 mean(dim(0,2,3)) 的执行过程
    1. 首先 dim = 0 时
       砍掉 0 维的维度，按 0 轴求平均 (3, 2, 2, 1) ⇒ (1, 2, 2, 1)
       tensor([
            [[[4],
            [5]],

            [[6],
            [7]]]
       ]) 
    2. ··· ···
    3. ··· ···
    '''   
    ```

### （三）、训练结果
```powershell
# batch_size = 256
2026-02-03 16:35:47.761908
training on cuda:0
loss 0.286, train acc 0.894, test acc 0.723
23673.3 examples/sec on cuda:0
2026-02-03 16:37:00.805772
0:01:13.043864
```

```powershell
# batch_size = 128
2026-02-03 16:41:07.805365
training on cuda:0
loss 0.247, train acc 0.908, test acc 0.897
17339.1 examples/sec on cuda:0
2026-02-03 16:42:35.229708
0:01:27.424343
```

- 好像在我的**硬件**上，只要把 `batch_size` 调小一些，性能就会好很多，这是为什么？
  - 是因为消费级显卡的原因吗？
  - 感觉应该不是吧。哪里有这么多专业版显卡？

<br><br><br><br>


## 三、简洁实现 - 在 `21-CNN-LeNet.py` 的基础上加上 Batch Norm
```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```
### （一）、理解代码

### （二）、Pytorch API
- `nn.BatchNorm2d` 和 `nn.BatchNorm1d` 
  - 都只需要 `num_features`
  - 不需要 `num_dims`

### （三）、实验结果
```powershell
# batch_size = 256
2026-02-03 16:44:57.502871
training on cuda:0
loss 0.267, train acc 0.902, test acc 0.853
33766.2 examples/sec on cuda:0
2026-02-03 16:46:04.729064
0:01:07.226193
```

```powershell
# batch_size = 128
2026-02-03 16:46:46.619949
training on cuda:0
loss 0.244, train acc 0.910, test acc 0.893
24241.6 examples/sec on cuda:0
2026-02-03 16:47:56.278802
0:01:09.658853
```