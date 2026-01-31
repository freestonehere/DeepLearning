## 一、`softmax` 数学理解
1. `softmax` 本身其实是个多分类**函数**，输出概率值
2. **softmax 函数的定义**
   1. 要使输出匹配为概率（非负，和为 1），则需对输出 ${\bf{o}}$ 进行 **Softmax数学变换** ： $$\hat{\bf y}={\rm{softmax}}({\bf o})$$
   2. 为保证 **softmax** 操作满足 “非负，和为 1”，对其中每个类别置信度输出 $\hat y_i$ 为： $$\hat y_i={\exp{o_i}\over\sum_k\exp{o_k}} ~ ~ (o_k = XW + b)$$
      1. 由于 softmax 单个输入为行向量（ X 为行向量 ）
      2. ⇒ W 本身应该是个列向量
      3. ⇒ **对于单个数据** $~ o_k = XW + b ~ ~ ~ ~$ ( 常数 = 一行 ⨉ 一列 + 常数 )
   3. 数据集格式（ **对于单个数据** ）
      1. `ToTensor()` 后的原始**图片**：张量形状 $(1, 28, 28)$
      2. `reshape()` 后的**展平图片**：张量形状 $(784)$
      3. **标签**一直都不变：始终都是一个实数
   4. 预测概率 $\hat{\bf{y}}$ 与真实概率 $\bf{y}$ 的差做损失。
3. 交叉熵损失函数 $$l(\bf{y},\hat{\bf{y}})=\it{-\sum_{i}y_i\log\hat{y_i} = -\log\hat{y}_y}$$

<br><br><br><br>

## 二、从零实现 `softmax` 分类模型
```python
import torch
from d2l import torch as d2l
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

'''和主逻辑无关的辅助包'''
import matplotlib.pyplot as plt # 用于画图
from tools import Animator # 在动画中绘制数据
from tools import Accumulator # 在 n 个变量上累加


'''加载数据集，并转化为张量'''
# 因为数据只有几百 MB, 因此直接将数据加载到内存中
transform = transforms.ToTensor()
# 训练集
train_dataset = datasets.FashionMNIST(
    root='data/FashionMNIST', train=True, transform=transform
)
# 测试集
test_dataset = datasets.FashionMNIST(
    root='data/FashionMNIST', train=False, transform=transform
)
# 构建迭代器: torch.utils.data.DataLoader 可以进行自动批处理，用 batch_size 等来设置
train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False)


'''初始化模型参数 (正推 & 反推 结合用起来)'''
num_inputs = 784
num_outputs = 10
# 已知 FashionMNIST 数据集的 (C, H, W) = (1, 28, 28)
# 同时 softmax 要求输入的单个数据必须是向量
# ⇒ 因此，要把 28 ⨉ 28 展成一维的向量。
# 于是 num_inputs = 28 ⨉ 28 = 784
# 这种展平操作会丢失一些信息，所以卷积神经网络中不能做展平操作
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
# 张量形状: W = (784, 10); b = (10)


'''定义前向计算模型，并展平图像数据集'''
# 所以：softmax 的输入是一个二维矩阵，输出也是一个二维矩阵
def softmax(X):
    X_exp = torch.exp(X)
    # 这里对 X_exp 按行求和，得到 partition
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制（不过是除法的广播机制）
    # 乘法的广播机制保留最后 2 个维度，只广播前 N - 2 个维度

def net(X):
    # softmax 最后的输入是 XW + b
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


'''交叉熵损失函数'''
# torch.log() 底数已经确定：e
# 也就是说：torch.log 是自然对数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


'''计算预测正确的数量'''
def accuracy(y_hat, y):  #@save
    # 辅助理解：a.shape = torch.Size([3, 4]) 时
    # len(a.shape) = 2
    # y_hat 张量形状为 (batch_size, num_labels)
    # 所以才会有下面的条件
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        # 找每个 1 轴内的最大值，并返回索引
        # 这里就是找每行的最大值，并返回索引
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


'''计算在指定数据集上模型的精度'''
# 这个函数其实很简单！
def evaluate_accuracy(net, data_iter):  #@save
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


'''训练模型一个迭代周期 (定义见第3章)'''
# 这个函数也比较简单！
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    # 将模型设置为训练模式。
    # 如果采用 torch.nn.Module 训练，那么直接调用封装好的 train 函数
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 参数 W b 更新（梯度下降）前的数据准备
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用自己实现的的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        '''
        注意一个小点，loss 为均方损失，又要反向求导时：
        使用 optim.Optimizer 时，用 l.mean().backward();
        使用自己实现的 SGD 算法时，用 l.sum().backward()
        '''
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


'''训练模型 (定义见第3章)'''
# 主要是一些陌生的语法，逻辑比较简单
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # 训练指标 train_metrics: 一般是 (训练损失, 训练精度) (train_loss, train_acc)
    # assert (断言)：强制校验训练效果，若不满足条件则直接报错，避免后续使用性能不达标的模型
    # assert 条件表达式, 错误提示信息
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


'''定义优化器: 小批量随机梯度下降算法'''
lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


'''预测标签 (定义见第3章)'''
# 主要也是陌生的语法，逻辑并不难。别怵！
def predict_ch3(net, test_iter, n=6):  #@save
    # 遍历测试数据迭代器，但是只取第一批，然后立即停止
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


'''训练模型 (10 轮)'''
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

predict_ch3(net, test_iter)
plt.show()
```

### （一）、理解代码
#### 1、张量形状
1. **softmax 的输入：** 对于单个数据来讲，softmax 的输入必须是 **向量**！
2. 数据集
    ```python
    '''加载数据集'''
    # 因为数据只有几百 MB, 因此直接将数据加载到内存中
    transform = transforms.ToTensor()
    # 训练集
    train_dataset = datasets.FashionMNIST(
        root='data/FashionMNIST', train=True, transform=transform
    )
    # 测试集
    test_dataset = datasets.FashionMNIST(
        root='data/FashionMNIST', train=False, transform=transform
    )
    # 构建迭代器: torch.utils.data.DataLoader 可以进行自动批处理，用 batch_size 等来设置
    train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False)
    ```
    1. 这里 `transforms.ToTensor()` 并不会将 28 ⨉ 28 的图片展平为 784
    2. `transforms.ToTensor()` 只是将 PIL / numpy 格式的数据转化为张量格式、然后进行归一化操作
    3. DataLoader 构造的迭代器每次抽取 batch_size 个样本（ 即 256 个样本 ），因此张量形状为 $(256, 1, 28, 28)$
        ```python
        '''代码验证张量形状'''

        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        # 复用你的代码逻辑
        transform = transforms.ToTensor()
        train_dataset = datasets.FashionMNIST(
            root='data/FashionMNIST', train=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='data/FashionMNIST', train=False, transform=transform
        )
        train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # 1. 查看单个样本的形状
        single_img, single_label = train_dataset[0]
        print("单个样本图像张量形状：", single_img.shape)  
        # 输出: torch.Size([1, 28, 28])
        print("单个样本标签类型/值：", type(single_label), single_label)  
        # 输出: <class 'int'> 9（或其他整数）
        print("标签转为张量后的形状：", torch.tensor(single_label).shape)  
        # 输出: torch.Size([])（0维）
        # 标签转换为张量后，就是单个实数，自然是 0 维

        # 2. 查看批次数据的形状
        for X, y in train_iter:
            print("批次图像张量形状：", X.shape)  
            # 输出: torch.Size([256, 1, 28, 28])
            print("批次标签张量形状：", y.shape)  
            # 输出: torch.Size([256])
            # 由于 W.shape = (784, 10) ⇒ 因此，W.shape[0] = 784
            print("reshape 后的批次图像张量形状：", X.reshape(-1, 784).size())
            break  # 只看第一个批次即可        
        ```
3. 前向计算（原来 **展平图像** 的操作在 **前向计算** 的 `reshape()` 里边）
    ```python
    '''定义前向计算模型'''
    # 所以：softmax 的输入是一个二维矩阵，输出也是一个二维矩阵
    def softmax(X):
        X_exp = torch.exp(X)
        # 这里对 X_exp 按行求和，得到 partition
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition  # 这里应用了广播机制（不过是除法的广播机制）
        # 乘法的广播机制保留最后 2 个维度，只广播前 N - 2 个维度

    def net(X):
        # softmax 最后的输入是 XW + b
        return softmax(
            torch.matmul(X.reshape((-1, W.shape[0])), W) + b  # 就是这里展平的！！！
        )    
    ```
    1. 展平后，批次图像张量形状为 $X = (256, 784)$
    2. 为了用上 softmax 的函数，使得最后产生 $(256, 10)$ 的数据
    3. ⇒ $W$ 张量形状必须是 $W = (784, 10)$ （隐含条件：**矩阵相乘** 是没有 **广播机制** 的）
    - $W$ 张量形状是怎么来的？起码现在，我是按上面理解的（**反推** 得到 $W$ 张量的形状）
    1. 既然 $XW = (256, 10)$ 并且，**矩阵相加** 有 **广播机制**
    2. ⇒ 张量形状： $b = (10)$
    3. ⇒ **最终 `softmax`** 的输出就是 $(256, 10)$ 即 $(\text{batch\underline{\text{ }}size}, ~ \text{nums\underline{\text{ }}out})$
    4. 也就是说：单独看 $W$ 张量的形状没有意义，而且根本不可能看懂。必须和上下文联系起来！
4. 还是要记一下线性变换的表达式 $$Y = XW + b$$
5. 交叉熵损失函数
    ```python
    '''交叉熵损失函数'''
    # torch.log() 底数已经确定：e
    # 也就是说：torch.log 是自然对数
    def cross_entropy(y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])

    cross_entropy(y_hat, y)
    ```
    1. 已知 $\hat{y} ~ (\text{softmax} ~ 最终的输出)$ 张量形状为 $\hat{y} = (256, 10)$
    2. 批量标签的形状为 $y = (256)$ （256 个实数）

    ```python
    '''索引张量的例子'''

    a = torch.tensor(list(range(12)))
    # Python 默认不进行原地操作
    a = a.reshape(2, 6)
    a
    tensor([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11]])

    # 查看张量的 len() 函数
    len(a)
    2
    a[-1]
    tensor([ 6,  7,  8,  9, 10, 11])

    # 张量的普通索引
    a[0, 1]
    tensor(1)
    a[0, [1, 2, 3]]
    tensor([1, 2, 3])

    '''有了最后这一个索引张量的例子，那就好理解上面代码了！'''
    a[[0, 1], [1, 2]]
    tensor([1, 8])    
    ```    

<br>

### （二）、`torchvision.datasets`
- 话说，`torchvision.datasets` 和 `torch.utils.data` 有什么区别和联系？
  - 一个是数据处理 `torch.utils.data`
  - 一个是数据集 `torchvision.datasets`

[官网 doc - torchvision.datasets 首页](https://docs.pytorch.org/vision/stable/datasets.html)
1. Torchvision provides many 
   1. **built-in** datasets in the `torchvision.datasets` module, 
   2. as well as utility classes for **building your own datasets**.
2. **Built-in datasets**
   1. All datasets are subclasses of `torch.utils.data.Dataset` i.e, they have `__getitem__` and `__len__` methods implemented.（ 所有内置数据集都是 `torch.utils.data.Dataset` 的子类 ）
   2. Hence, they can all be passed to a `torch.utils.data.DataLoader` which can load multiple samples in parallel using `torch.multiprocessing` workers. （ 它们可以传到 `torch.utils.data.Dataloader` 里面，后者可以使用 `torch.multiprocessing` 工作进程并行加载多个样本 ） 
   3. For example:
        ```python
        imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                batch_size=4,
                                                shuffle=True,
                                                num_workers=args.nThreads)
        ```
3. **Build your own datasets using base-class**
   1. 一共有 3 个 base-class
   2. 具体见官网 doc
   3. 这里就不抄了

#### 1、`torchvision.datasets.FashionMNIST`


<br>

### （三）、`torchvision.transforms`
[官网 doc - torchvision 首页](https://docs.pytorch.org/vision/stable/index.html)
[官网 doc - torchvision.transforms 首页](https://docs.pytorch.org/vision/stable/transforms.html)

#### 1、`torchvision.transforms.ToTensor()`
[官网 doc - torchvision.transform.v2.ToTensor()，但是已弃用，下一版本中将会删除](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html#torchvision.transforms.v2.ToTensor)

1. Convert a PIL Image or ndarray to tensor and scale the values accordingly. （ 将 PIL 图像或 ndarray 转换为张量，并相应地缩放值。 ）
2. **if** the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8 <br>（ 如果该 PIL 图像属于以下模式之一（L、LA、P、I、F、RGB、YCbCr、RGBA、CMYK、1），或者该 numpy 数组的数据类型为 np.uint8 ）
   1. Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
   2. （ 那么，将 PIL 图像或 numpy 数组（H x W x C）在 [0, 255] 范围内转换为 `torch.FloatTensor`，形状为（C x H x W），范围在 [0.0, 1.0]，）
3. **else** tensors are returned without scaling.
   1. 返回的张量不进行缩放

- **总结：** 使用 `transforms.ToTensor()` 的核心原因可以归纳为 3 点
  - 格式适配：将原始 PIL / numpy 数据转为 PyTorch 模型能处理的 Tensor 类型；
  - 维度规范：调整为 PyTorch 视觉任务标准的 (C, H, W) 维度顺序；
  - 数值优化：归一化像素值到 0.0 ~ 1.0，提升神经网络训练的稳定性和效率。

<br>

### （四）、再次理解：张量按特定轴求和
```python
import torch

'''定义张量 a 并查看 a 的形状'''
a = torch.ones(2, 5, 7)
a.size()
torch.Size([2, 5, 7])
a
tensor([[[1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1.]]])



'''按 0 轴求和，保持维度'''
a0t = a.sum(axis =0, keepdim=True)
a0t.size()
torch.Size([1, 5, 7])
a0t
tensor([[[2., 2., 2., 2., 2., 2., 2.],
         [2., 2., 2., 2., 2., 2., 2.],
         [2., 2., 2., 2., 2., 2., 2.],
         [2., 2., 2., 2., 2., 2., 2.],
         [2., 2., 2., 2., 2., 2., 2.]]])



'''按 0 轴求和，不保持维度'''
a0f = a.sum(axis =0, keepdim=False)
a0f.size()
torch.Size([5, 7])
a0f
tensor([[2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2.]])



'''注意不同点：a0t 比 a0f 多一对中括号'''


'''按 1 轴求和，不保持维度'''
a1f = a.sum(axis=1, keepdim=False)
a1f.size()
torch.Size([2, 7])
a1f
tensor([[5., 5., 5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5., 5., 5.]])



'''按 1 轴求和，保持维度'''
a1t = a.sum(axis=1, keepdim=True)
a1t.size()
torch.Size([2, 1, 7])
a1t
tensor([[[5., 5., 5., 5., 5., 5., 5.]],

        [[5., 5., 5., 5., 5., 5., 5.]]])



'''按 2 轴求和，不保持维度'''
a2f = a.sum(axis=2, keepdim=False)
a2f.size()
torch.Size([2, 5])
a2f
tensor([[7., 7., 7., 7., 7.],
        [7., 7., 7., 7., 7.]])



'''按 2 轴求和，保持维度'''
a2t = a.sum(axis=2, keepdim=True)
a2t.size()
torch.Size([2, 5, 1])
a2t
tensor([[[7.],
         [7.],
         [7.],
         [7.],
         [7.]],

        [[7.],
         [7.],
         [7.],
         [7.],
         [7.]]])
```
1. 张量按轴求和的几个要点
   1. 维数从最外层开始计算 ( 0 轴就是最外层 )
   2. 元素对应相加
   3. 不要越过自己的小环境

<br>

### （五）、说句题外话：关于参数的问题
$$o_k = XW + b$$
1. 本模型的**可训练参数**就是 $W, b$ ， **超参数** 是 `lr` （学习率）
   1. 而且已知 $W = (784, 10)$ , $b = (10)$
   2. 那么可训练参数总数就是 7840 + 10 = 7850，参数（加上超参数以后）总数就是 7850 + 1 = 7851
   3. 那么该模型是不是就是 千级参数 模型？
   - **答**：确实是 千级参数 模型！
2. 有了上面的回答后，还有疑问：softmax 实际上是
   1. 对每个像素做线性拟合
   2. 然后输出概率
   - 肯定有试一试的成分
   3. 我感觉偏置 $b$ 不应该是 $10$ 啊，感觉是 $(\text{batch\underline{ }size}, 10)$ 更符合**直觉**
   4. 但是从 **逻辑** 上来讲，参数不应该和 批量数 有关，所以：**逻辑** 上，$b$ 还得是 $10$ 。但就是无法从 **直觉** 上理解！
3. 参数管规律，批次管效率

#### Ⅰ、参数数量分析与验证（AI 生成，不一定对，但比较震撼人）

你想确认这个用于 FashionMNIST 分类的 softmax 回归模型中，可训练参数总数是否为 7840 + 10 = 7850 个，同时想知道该模型是否属于 “千级参数模型”。

首先，我们来一步步验证参数数量的计算，以及“千级参数模型”的定义：

##### 1. 可训练参数总数的验证
你的计算是**完全正确的**：
- 权重矩阵 `W`：形状为 `(784, 10)`，参数数量 = 784 × 10 = 7840 个
- 偏置向量 `b`：形状为 `(10,)`，参数数量 = 10 个
- 总可训练参数 = 7840 + 10 = 7850 个

##### 2. “千级参数模型”的判定
行业内通常根据参数数量的**量级**来划分模型规模，常见的划分方式如下：
| 量级       | 参数数量范围       | 示例          |
|------------|--------------------|---------------|
| 百级       | < 1000（1k）       | 简单线性回归  |
| 千级       | 1k ~ 10k（1000~10000） | 本softmax模型 |
| 万级       | 10k ~ 100k         | LeNet-5（~60k）|
| 百万级     | 100k ~ 1M          | 简单CNN       |
| 千万/亿级  | > 10M              | Transformer、大型CNN |

7850 个参数处于 `1k ~ 10k` 的区间内，因此这个模型**确实属于千级参数模型**。

##### 3. 补充说明
这个softmax回归模型是非常轻量的入门级模型：
- 它没有隐藏层，本质是“输入层直接映射到输出层”的线性模型（softmax仅做概率归一化，不增加参数）
- 对比深度学习中常见的模型：
  - LeNet-5（简单CNN）：约6万个参数（万级）
  - ResNet-18：约1100万个参数（百万级）
  - BERT-Base：约1.1亿个参数（亿级）
- 也正因为参数极少，这个模型的拟合能力有限，但足够完成FashionMNIST的基础分类任务（你代码中要求测试精度>0.7，实际训练后通常能达到0.8左右）。

##### 总结
1. 该模型的可训练参数总数计算正确，为**7850个**（W:7840个，b:10个）。
2. 7850属于1k~10k区间，因此该模型**是千级参数模型**。
3. 该模型参数规模极小，是入门级的线性分类模型，拟合能力远低于深层神经网络。

<br><br><br><br>

## 三、简洁实现 `softmax` （ 使用 `torch.nn.Module` ）
```python
import torch
from torch import nn
import matplotlib.pyplot as plt # 用于画图
from tools.ch3 import train_ch3 # 导入训练函数
from tools.ch3 import load_data_fashion_mnist # 导入数据集加载函数

'''从数据迭代器加载数据集（训练集 & 测试集）'''
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

'''定义模型'''
# PyTorch 不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# 对于线性层来讲 Y = XW + b

'''初始化模型参数'''
def init_weights(m):
    if type(m) == nn.Linear:
        # 这里的 input tensor 就是 m.weight
        # 你既然通过 m.weight 初始化参数的话，那就不需要指定参数张量的维度了！省心！
        # 虽然不用指定维度，但最好还是知道张量形状（便于自己分析）
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)


# 交叉熵损失函数 和 优化器 应该是一样的
# 就是它们都只是读取模型产生的数据
# 都不关心模型本身，所以不用在函数接口指定模型
'''定义交叉熵损失函数'''
loss = nn.CrossEntropyLoss(reduction='none')

'''定义优化器（用于更新参数）'''
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

'''开始训练'''
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

plt.show()
```
### （一）、理解代码
#### 1、等一下，这不是 softmax 分类模型吗？为什么你的 `net` 用的是 `Linear Layer` ？
1. PyTorch 的 `nn.CrossEntropyLoss` 有一个关键特性 —— 内置了 softmax 操作，并且做了数值稳定性优化（避免直接计算 $e^x$ 导致的溢出问题）。
2. [官网 doc - torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
   1. Pytorch 官网文档里**明确**说了，Pytorch CELoss **特别适合** 多分类问题！

<br>

### （二）、`torch.nn.Flatten()` 展平层
[官网 doc - torch.nn.modules.flatten.Flatten](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.flatten.Flatten.html#flatten)
[官网 doc - torch.flatten (used with Sequential)](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten)
- 所以，**本代码**用的到底是哪个 `flatten()` 啊？有点不太明白！
  - 经过 `go to definition` 选项查看：是 `class torch.nn.modules.flatten.Flatten`
  - 该 class 的用法抄录如下
    ```python
    '''class Flatten 原型'''
    class torch.nn.modules.flatten.Flatten(start_dim=1, end_dim=-1)


    '''用法实例'''
    # 发现用的时候也没那么麻烦，直接 torch.nn.Flatten() 就好！
    input = torch.randn(32, 1, 5, 5)

    # With default parameters
    m = nn.Flatten()
    output = m(input)
    output.size()

    # With non-default parameters
    m = nn.Flatten(0, 2)
    output = m(input)
    output.size()
    ```

<br>

### （三）、`torch.nn.Squential().apply(init_weight)` 初始化参数
[官网 doc - torch.nn.module.apply](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply)

```python
apply(fn)
```
1. （功能概述）Apply `fn` recursively to every submodule ( as returned by `.children()` ) as well as self.
   1. Typical use includes initializing the parameters of a model (see also torch.nn.init).
2. **Parameters** :
   1. `fn (Module -> None)` – function to be applied to each submodule
   2. 也就是说：*参数* 是 *要应用于每一个子模块的函数*
3. **Returns** :
   1. self
4. **Return type** :
   1. Module

```python
'''抄录实例'''

@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) is nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
```

<br>

### （四）、`torch.nn.init` 初始化参数
[官网 doc - torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html#nn-init-doc)
1. **Warning** : 
   1. All the functions in this module are intended to be used to initialize neural network parameters, 
   2. so they all run in `torch.no_grad()` mode and will not be taken into account by autograd.
   3. 由于参数初始化是 “一次性设置初始值” 的操作，无需计算其梯度，因此这些函数默认运行在 `torch.no_grad()` 模式下

```python
torch.nn.init.normal_(tensor, mean=0.0, std=1.0, generator=None)
```
- Fill the **input Tensor** with values drawn from the normal distribution.

$$N(\text{mean},\text{std}^{2})$$

<br>

### （五）、`torch.nn.CrossEntropyLoss(reduction='none')` 多分类问题中的交叉熵损失函数
[官网 doc - torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

```python
'''class CrossEntropyLoss 原型'''
class torch.nn.CrossEntropyLoss(
    weight=None, size_average=None, ignore_index=-100, 
    reduce=None, reduction='mean', 
    label_smoothing=0.0
)
```

#### 1、功能详解
1. This criterion computes the cross entropy loss between input logits and target. （用于计算对数输入和目标之间的交叉熵损失）
2. It is useful when training a classification problem with **C classes**. （**交叉熵损失函数** 在 **C 分类** 问题中很有用）
   1. If provided, the optional argument **`weight`** should be a **1D Tensor** assigning weight to **each of the classes**.
   2. This is particularly useful when you have an unbalanced training set.（什么叫不平衡的训练集，起码现在还没接触到，以后再说）
3. 对于输入的两个要求
   1. 数据要求
      1. The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general). 
   2. 张量形状
      1. **非批量输入（单样本）** : Input has to be a Tensor of $size (C)$ for unbatched input, 
      2. **批量输入（多样本，但样本本身无空间维度）** : $(minibatch, C)$ 
      3. **高维批量输入（样本本身含空间维度）** : or $(minibatch, C, d_1, d_2 , ..., d_K)$ with $K ≥ 1$ for the K-dimensional case. 
      4. The last being useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images
4. 这段话核心是明确 PyTorch 中 `CrossEntropyLoss` 对 **输入（input）** 的两点关键要求：**数据含义** 与 **张量形状**，具体解析如下：
   1. **输入数据含义：未归一化的 logits**  
      - 输入需是每个类别的“logits”（即模型最后一层未经过softmax等归一化操作的原始输出），其无需满足“值为正”或“所有类别之和为1”——因为 `CrossEntropyLoss` 内部会自动对logits做softmax，将其转化为概率分布后再计算损失。**输入张量形状：适配不同场景**  
   2. 输入需符合特定维度要求，具体分三类场景：
      - **非批量输入（单样本）** : 形状为 `(C)`，其中 `C` 是类别数量（如3分类任务，输入为1个长度为3的张量）；
      - **批量输入（多样本，无空间维度）** : 形状为 `(minibatch, C)`，`minibatch` 是批量大小（如一次输入 10 个 3 分类样本，输入形状为 $(10, 3)$ ）；
      - **高维批量输入（含空间维度）** : 形状为 `(minibatch, C, d₁, d₂, ..., d_K)`（ `K≥1`，`d₁, d₂, …, d_K` 等是空间维度尺寸），
        - 典型场景是 **图像逐像素分类** （如输入 10 张 3 通道、224 × 224 的图像做语义分割，输入形状为(10, C, 224, 224)，`C` 是分割类别数，损失会逐像素计算）。
        - 这个例子是 AI 生成的，不一定对。存疑！（因为我感觉不对）

#### 2、参数详解
1. **`reduction`** (`str`, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
   1. 三种参数讲解
      1. `'none'`: no reduction will be applied, 
      2. `'mean'`: the weighted mean of the output is taken, 
      3. `'sum'`: the output will be summed. 
   2. Note: 
      1. `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`.
      2. 指定 `size_average` 和 `reduce` 种任何一个参数，那么都会覆盖 `reduction`
   3. Default: `'mean'`

#### 3、CELoss 就学到这里吧，再高深的东西，以后遇到再学！

