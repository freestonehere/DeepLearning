# FCN ( Fully Connected Convolution Network ) | 全连接卷积神经网络
## 一、概念理解
![FCN 基本思路](https://zh-v2.d2l.ai/_images/fcn.svg)

1. 就是在**基本 `CNN` 网络**的基础上
   1. 把全连接层换成转置卷积层
   2. 卷积神经网络还是缩小尺寸
   3. 转置卷积层也确实是放大尺寸
      1. 但是放大尺寸之后呢？是怎么进行语义分割的呢？
      2. 这里就不明白了！
   4. 我好像有一点明白了
      1. 因为你要逐像素预测，所以要用 **转置卷积** 把图片 **尺寸还原** 到原来的尺寸
      2. 又因为你是 **分类问题**，所以用 **`channel` 维度**存储类别信息。
      3. 从转置卷积层出来以后，张量形状 `(batch_size, num_classes, h, w)`；标签的张量形状是 `(batch_size, 1, h, w)`
         1. 因为 `VOCSegDataset` class 中，`__getitem__` 函数返回 **特征图** 和 **类别索引**
         2. 注意：是 **类别索引**，不是 **类别图片**！
         3. 所以，标签张量是 `(batch_size, 1, h, w)` 而不是 `(batch_size, 3, h, w)`
      4. 从转置卷积层出来的张量和标签做损失函数
2. 大致思路很好理解
   1. 但是具体细节一点都不懂！
3. 使用一个数据集还真是 **三件套**
   1. 读取数据集 `read`
   2. 构造 `Dataset` class
   3. 构造 `train_iter` 和 `test_iter`
   - 但是在构造这三者的过程中，可能会用到很多很多辅助函数！
4. 转置卷积公式 $n^{\prime} = sn + k - p - s$
   1. 如果 $k = 2s, ~ p = s$ ⇒ 那么图片尺寸倍放大 $s$ 倍！
   2. 其实让图片尺寸放大 $s$ 倍的参数还有很多种，这里选择 $k=2s$ 是为了 **让步幅不至于太大，让相邻的转置卷积核还能重叠一部分！**

<br><br><br><br>

## 二、代码实现
```powershell
# 这是代码运行结果

查看预训练模型的最后 3 层（池化+全连接，之前很熟悉的！）

[Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (downsample): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
), 

AdaptiveAvgPool2d(output_size=(1, 1)), 

Linear(in_features=512, out_features=1000, bias=True)

]

net(X).shape: torch.Size([1, 512, 10, 15])
input image shape: torch.Size([561, 728, 3])
output image shape: torch.Size([1122, 1456, 3])
read 1114 examples
read 1078 examples
2026-02-24 09:58:10.686555
loss 0.416, train acc 0.869, test acc 0.850
121.4 examples/sec on [device(type='cuda', index=0)]
2026-02-24 09:59:31.661618
0:01:20.975063
```
### （一）、理解代码

<br><br>

### （二）、`torch` API
#### 1、`torch.nn.functional.cross_entropy` | 交叉熵损失函数
- [官网 doc - torch.nn.functional.cross_entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy)

```python
torch.nn.functional.cross_entropy(
    input, target, weight=None, size_average=None, 
    ignore_index=-100, reduce=None, reduction='mean', 
    label_smoothing=0.0)
```

1. 张量形状
   1. `input`: Shape $(C)$ $(N,C)$ or $(N,C,d_1,d_2,...,d_K)$ with $K≥1$ in the case of K-dimensional loss.
   2. `Target`: If containing class indices, shape $()$, $(N)$ or $(N,d_1,d_2,...,d_K)$ with $K≥1$ in the case of K-dimensional loss where each value should be between $[0,C)$. 
      1. If containing class probabilities, same shape as the input and each value should be between $[0,1]$.
   3. where: 
      1. $C = \text{number of classes}$
      2. $N = \text{batch size}$
2. 例子代码
    ```python
    # Example of target with class indices
    # input (3, 5) 生成 3 个样本，每个样本 5 个类别
    # target (3,) 生成 3 个取值在 [0,4] 之间的整数（对应 5 个类别），形状为 (3,)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    loss = F.cross_entropy(input, target)
    loss.backward()

    # Example of target with class probabilities
    # input (3, 5) 生成 3 个样本，每个样本 5 个类别
    # taget 和 input 张量形状一致，都是 (3, 5)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    loss = F.cross_entropy(input, target)
    loss.backward()
    ```
<br>

#### 2、`argmax(dim)` 函数 | 指定维度那就不展平 
```python
# 预测
def predict(img):
    # X 张量形状 (1, 3, h, w)
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

- 关于 `argmax` 函数的详细讲解见 [33-CV-AnchorBox.md](33-CV-AnchorBox.md)
  - 不指定维度，那就一律展平
    ```python
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a)
    tensor(0)

    >>> b = torch.tensor(range(1, 17)).reshape(4, 4)
    >>> b
    tensor([[ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16]])
    >>> torch.argmax(b)
    tensor(15)

    # ⇒ 也就是说：把所有维度都展品，然后求最大值的索引！
    ```
  - 指定维度，那就不展平
    ```python
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a, dim=1)
    tensor([ 0,  2,  0,  1])
    ```
- 模仿 `(batch_size, num_classes, h, w)` 的简单例子
  - **`argmax`** 指定维度的时候，**要裁掉这个维度**
  - 直观理解这个 `(batch_size, num_classes, h, w).argmax(dim=1)` 函数：
    - **给定** `(h_i, w_i)` 这个平面图上的**坐标**；
    - 查看 **所有通道** 中，该坐标上的数值；
    - **哪个通道** 中该坐标的数值最大，就返回 **这个通道** 的标号
  - 同时，因为 class `VOCSegDataset` 返回的就是 **特征图** + **类别索引（不是类别图片）**，正好 **类别索引** 也是 `[0, num_classes)` 之间，正好可以学习！
    ```python
    # 模仿一个共有 2 类，尺寸为 2x2 的图片
    # X (batch_size, num_classes, h, w)
    >>> X = torch.randn((1, 2, 2, 2))
    >>> X
    tensor([[[[ 0.6207,  0.2985],
            [-1.3978, -0.2794]],

            [[-1.0836,  0.7870],
            [ 0.5040, -0.1215]]]])
    >>> pred = X.argmax(dim=1)
    >>> pred
    tensor([[[0, 1],
            [1, 1]]])
    ```
<br>

##### (1)、会使张量退化的函数
|会使张量退化的函数|
|:------|
|`sum`|
|`mean`|
|`argmax`|

