# CNN - channels（卷积神经网络 - 多输入 / 输出通道卷积）

## 一、概念理解
1. 相比之前的 3 个超参数（卷积核、填充、步幅）
   1. 实际上 **通道数** 这个超参数往往会仔细调一调！
2. 如果 **某一个卷积层** 输入 / 输出 **通道数为 1**
   1. 那么，该卷积层的 输入 / 输出 图像就是**灰度图像**

<br><br>

### （一）、多输入通道 and 单输出通道 中的卷积运算
1. 当输入包含多个通道时，需要构造一个**与输入数据具有相同输入通道数的卷积核**，以便与输入数据进行互相关运算。
2. 为解决这个问题，我们可以给每个通道一个卷积核，输出是**所有通道卷积结果的和**。下图表示一个两通道输入的例子 👇
   1. 另外：理解深度学习中的 **卷积运算** 其实就是 **加权和**

![conv-multi-in](https://zh.d2l.ai/_images/conv-multi-in.svg)

3. 公式化表示：
   - 输入 $\bf X$：$c_i\times n_h\times n_w$
   - 核 $\bf W$：$c_i\times k_h\times k_w$
   - 输出 $\bf Y$：$m_h\times m_w$
     - 其中，$c_i$ 表示输入的通道维数


$$
{\bf Y}=\sum_{i=0}^{c_i}{\bf X}_{i,~:,~:}\star {\bf W}_{i,~:,~:}
$$

<br><br>

### （二）、多输入通道 and 多输出通道 中的卷积运算
1. 在最流行的神经网络架构中，随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度。
   1. 直观地说，我们可以将**每个 `输出通道` 看作是对 `不同特征` 的响应**。
      1. ⇒ 因此，就能理解，对于**卷积核张量**
      2. **$c_o$ 在 $c_i$ 的外层**
   2. 而现实可能更为复杂一些，因为每个通道不是独立学习的，而是为了共同使用而优化的。
   3. 因此，多输出通道并不仅是学习多个单通道的检测器。
2. 我们可以有多个三维卷积核，每个核生成一个输出通道，输出时便可叠加为多输出的通道；
3. 公式化表示：
   - 输入 $\bf X$ ：$c_i\times n_h\times n_w$
   - 核 $\bf W$ ：$c_o\times c_i\times k_h\times k_w$
   - 输出 $\bf Y$ ：$c_o\times m_h\times m_w$
     - 其中，$c_o$ 表示输出的通道维

$${\bf Y}_{i,~:,~:}={\bf X} \star {\bf W}_{i,~:,~:,~:} \quad \text{for i=1, $\cdots$, }\mathbb{c_o}$$


<br><br>

### （三）、(1 ⨉ 1) 卷积层 —— 实际上可以看作多通道的 `全连接层`
- $1\times 1$ 卷积层，即 $k_h=k_w=1$ 看起来似乎没有多大意义，失去了卷积层的特的在高度和宽度维度上，识别相邻元素间相互作用的能力。
- 但实际是一个受欢迎的选择，它不识别空间模式，只是**融合输入通道的信息**。

![conv-1x1](https://zh.d2l.ai/_images/conv-1x1.svg)

1. 上图展示了使用 1×1 卷积核与 3 个输入通道和 2 个输出通道的互相关计算。 
   1. 这里输入和输出具有相同的高度和宽度，
   2. **输出中的每个元素都是从输入图像中同一位置的元素的 *线性组合（加权和）*** 
   3. **我们可以将 1×1 卷积层看作是在每个像素位置应用的全连接层**，以 $c_i$ 个输入值转换为 $c_o$ 个输出值。 
   4. 因为这仍然是一个卷积层，所以跨像素的权重是一致的。 同时， 1×1 卷积层需要的权重维度为 $c_o×c_i$ ，再额外加上一个偏置。
2. 注意：这是 1⨉1 **卷积层**，不是 1⨉1 **卷积核**
   1. 也就是说，这一层神经网络用到的全部卷积核都是 1⨉1 的！

<br><br>

### （四）、`一层` 卷积层 的 计算复杂度
- 输入 $\bf X$ ：$c_i\times n_h\times n_w$
- 核 $\bf W$ ：$c_o\times c_i\times k_h\times k_w$
- 偏差 $\bf B$ ：$c_o\times c_i$
- 输出 $\bf Y$ ：$c_o\times m_h\times m_w$

$$
\bf Y =X \star W+B
$$

- 计算复杂度（浮点计算数 FLOP）：$O(c_i ~c_o ~k_h ~k_w ~m_h ~m_w)$
  - 对于 **输出图** 上 **每一个像素点** 的 **每一个卷积核**
    - 是 $k_h ~k_w$ 次计算
  - 又有 **$c_i ~c_o$ 个卷积核**
    - 所以 **输出图** 上 **每一个像素点** 需要 $c_o ~c_i ~k_h ~k_w$ 次运算
  - 又由于 **输出图** 一共有 **$m_h ~m_w$ 个像素点**
    - 所以，**整个输出图** 需要 $c_o ~c_i ~k_h ~k_w ~m_h ~m_w$ 次运算

<br><br>

### （五）、有意思的问题
1. **Q：一般卷积的尺寸和输出通道该怎么设计？**
   1. **🙋‍♂️**：一般如果卷积使得原有输入高宽减半，那么需要将通道数增加为原来的 2 倍，
   2. 以防止因压缩过多而丢失重要信息。
   3. 可以近似看作压缩了空间尺度（高、宽），则需要更多的语义尺度（通道数）来表示提取的特征。
2. **Q：卷积层中的 bias 对结果影响大吗？怎样理解 bias 的作用？**
   1. **🙋‍♂️**：bias 的作用相当于对数据的分布做平移，其实在后期随着各种归一化方法的使用（如 BatchNorm 等），
   2. bias 的作用越来越小，因为 bias 等价于输入数据均值的负数，
   3. 虽然不要 bias 也可以，但其实计算成本来说可以忽略不计，加上也无妨。
3. **如果网络很深的话，那 Padding 很多 0 会不会对模型的分辨能力造成干扰？**
   1. 不会的，因为 Padding 很多 0 相当于一个 bias
   2. 而结合上面的问答，可以知道：bias 对模型性能影响不大
4. 同一层不同通道上的卷积核是一样的
   1. 这样写代码的时候，每层卷积层只写一个卷积操作就行，方便
   2. 而且，这样同一层的卷积核一样，有利于计算
5. 明确一点：卷积核的参数就是我们要学的内容
6. 明确：卷积本身对位置十分敏感
   1. 后面会有池化层让其对位置不那么敏感
7. 输入通道是不能动态变化的

<br><br><br><br>

## 二、代码讲解（卷积操作函数总结）
```python
import torch

'''单输入通道 + 单输出通道 卷积操作'''
def corr2d(X, K):  #@save
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() 
            # 注意切片的语法；
            # 另外，这里的 * 是点积（真是没想到矩阵还有点积，不过就是模仿向量点积，加权和罢了）
    return Y
    # shape 都是先行后列（ 0 轴是从外面开始计算的 ）


'''多输入通道 + 单输出通道 卷积操作'''
def corr2d_multi_in(X, K):
    # 先遍历 X 和 K 的第 0 个维度（通道维度），再把它们加在一起
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print('多输入通道 + 单输出通道 卷积操作')
print(corr2d_multi_in(X, K), end='\n\n')



K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

'''多输入通道 + 多输出通道 卷积操作'''
def corr2d_multi_in_out(X, K):
    # 迭代 “K” 的第 0 个维度，每次都对输入 “X” 执行互相关运算。
    # 最后将所有结果都叠加在一起
    # 先进来的在前面
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

'''构造新的卷积核张量, 适配多输出通道'''
K = torch.stack((K, K + 1, K + 2), 0)

print('多输入通道 + 多输出通道 卷积操作')
print(corr2d_multi_in_out(X, K), end='\n\n')



'''多输入通道 + 多输出通道的 1⨉1 卷积操作 (采用全连接层拉成张量的形式)'''
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape # X = (c_i, h, w)
    c_o = K.shape[0]    # K = (c_o, c_i, h, w)
    X = X.reshape((c_i, h * w)) # 将
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    # 对于卷积层，严格来讲，应当是 XK
    # 但是这里的形式本身就不规范
    # 为了矩阵乘法能够正常进行 K=(c_o, c_i) X=(c_i, h*w)
    # 所以这里就写成 KX
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print('多输入通道 + 多输出通道 1 ⨉ 1 卷积操作')
print(Y1)
print(Y2)
```
### （一）、Python 语法
#### 1、Python `zip` 函数 （返回元素对应的元组）
[官网 doc - Python zip() function](https://docs.python.org/3/library/functions.html#zip)
1. 功能
   1. returns an iterator of tuples, 
   2. where the i-th tuple contains the i-th element from each of the argument iterables.
2. 例子代码
    ```python
    >>> for item in zip([1, 2, 3], ['sugar', 'spice', 'everything nice']):
    ···    print(item)
    (1, 'sugar')
    (2, 'spice')
    (3, 'everything nice')


    >>> for item in ([1, 2, 3], ['sugar', 'spice', 'everything nice']):
    ···    print(item) 
    [1, 2, 3]
    ['sugar', 'spice', 'everything nice']
    ```

<br><br>

### （二）、Pytorch 语法
#### 1、`torch.stack` 函数 （沿某一维度拼接一系列张量）
[官网 doc - torch.stack 函数](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack)

```python
torch.stack(tensors, dim=0, *, out=None)
```
1. 输入必须是 **一系列向量**，单个向量会报错
   1. 而且**多个张量**必须在**括号里面**！
2. 既然是**拼接**，那肯定先进来的在前面！（下面例子代码也可以看出来！）
3. 下面是例子代码
    ```python
    >>> x = torch.randn(2, 3)
    >>> y = torch.randn(2, 3)
    >>> x
    tensor([[ 0.2392,  0.3217,  0.1187],
            [-0.5753,  1.8386, -0.5143]])
    >>> y
    tensor([[-0.5890,  0.3164, -0.0377],
            [-0.0389,  0.6702, -1.4492]])
    >>> torch.stack((x, y))
    tensor([[[ 0.2392,  0.3217,  0.1187],
            [-0.5753,  1.8386, -0.5143]],

            [[-0.5890,  0.3164, -0.0377],
            [-0.0389,  0.6702, -1.4492]]])
    ```
