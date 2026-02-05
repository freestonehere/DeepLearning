# Data Augmentation | 数据增广
## 一、概念理解
- **增广的一个目的就是让训练集长得更像测试集**
  - 所以还是**数据为王**

### （一）、数据增强（Data Augmentation）
1. 大型数据集是成功应用深度神经网络的先决条件。 
   1. 以图像增广为例，在对训练图像进行一系列的随机变化之后，生成相似但不同的训练样本，从而**扩大了训练集的规模**。 
   2. 此外，应用图像增广的原因是，随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的**泛化能力**。
2. 在一个已有数据集，通过数据变换，使得有更多的多样性
   1. 在语言里加入各种不同的背景噪音
   2. 改变图片的颜色、形状、位置、变形等待属性

### （二）、使用增强数据训练
在训练中随机在线生成增强数据，在测试时不进行增强操作。
- 上下、左右翻转(Flip)：`torchvision.transforms.RandomHorizontalFlip()`
- 切割，从图片中切割一块，变成固定形状（由 CNN 特性决定），所以还存在拉伸等(Crop)
  - 随机高宽比（e.g.[3/4,4/3]）
  - 随机大小（e.g.[80%,100%]）
  - 随机位置
- 颜色(Color)
  - 色调
  - 饱和度
  - 明亮度
- 类似于对图片做 PS 的变换

### （三）、总结
- 数据增广通过变形数据来获取多样性从而使模型泛化性能更好
- 常见方法**翻转、切割、变色**等等

### （四）、有意思的问题
1. `num_workers` 实际上是关注的是 `CPU`
   1. 用于数据处理（注意：不是数据计算，张量计算由 GPU 进行）

<br><br><br><br>

## 二、代码 - 图片语法
```python
import torchvision
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
from tools.ch13 import *

d2l.set_figsize()
img = d2l.Image.open('data/img/cat1.jpg')
plt.imshow(img)

# 获取当前图形对象
fig = plt.gcf()
set_title(fig, '原始图片 - 猫咪')

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    # 这里 aug 其实 augmentation 的缩写，而且是 aug() 函数
    # 生成 num_rows * num_cols 张处理后的图片
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 水平翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
fig = plt.gcf()
set_title(fig, '水平翻转')

# 垂直翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
fig = plt.gcf()
set_title(fig, '垂直翻转')

# 裁切
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
fig = plt.gcf()
set_title(fig, '裁切')

apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
fig = plt.gcf()
set_title(fig, '改变颜色')

apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
fig = plt.gcf()
set_title(fig, '改变色调')

color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
fig = plt.gcf()
set_title(fig, '亮度、对比度、饱和度和色调')

augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
fig = plt.gcf()
set_title(fig, '3 种常用的图像增广方法')

# 没有这行代码的话，那就不会跳出 Figure 来！
plt.show()
```

### （一）、`torchvision` 包中的内容
[官网 doc - torchvision 首页](https://docs.pytorch.org/vision/stable/index.html)
[官网 doc - torchvision.transforms 首页](https://docs.pytorch.org/vision/stable/transforms.html)
<br>

#### 1、`torchvision.transforms.ToTensor()`
- 直接抄的 [09-softmax.md](09-softmax.md)

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

#### 2、水平 + 垂直翻转 `torchvision.transforms.RandomHorizontalFlip()` 和 `torchvision.transforms.RandomVerticalFlip()`
```python
class torchvision.transforms.RandomHorizontalFlip(p=0.5)

class torchvision.transforms.RandomVerticalFlip(p=0.5)
```
<br>

#### 3、裁切 `torchvision.transforms.RandomResizedCrop`
```python
class torchvision.transforms.RandomResizedCrop(
    size, 
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333), 
    interpolation=InterpolationMode.BILINEAR, 
    antialias: Optional[bool] = True)
```
1. Crop a random portion of image and resize it to a given size.
2. A crop of the original image is made:
   1. the crop has a random area $(H * W)$ and a random aspect ratio.
   2. This crop is finally resized to the given size.
   3. This is popularly used to train the `Inception` networks
      1. 居然在这里**遇见刚学过的东西**了 `Inception block in GoogLeNet` [25-CNN-GoogLeNet.md](25-CNN-GoogLeNet.md)
3. `ratio` (tuple of python:float) – 
   1. lower and upper bounds for the random aspect ratio of the crop, before resizing.
   2. 在 resize 之前，裁剪区域**宽高比**的上下限
<br>

#### 3、亮度、对比度、饱和度和色调 `torchvision.transforms.ColorJitter`
- ColorJitter 颜色抖动

```python
class torchvision.transforms.ColorJitter(
    brightness: Union[float, tuple[float, float]] = 0, 
    contrast: Union[float, tuple[float, float]] = 0, 
    saturation: Union[float, tuple[float, float]] = 0, 
    hue: Union[float, tuple[float, float]] = 0
)
```
1. **`brightness`** (float or tuple of python:float (min, max)) – How much to jitter brightness. 
   1. brightness_factor is chosen uniformly from `[max(0, 1 - brightness), 1 + brightness]` or the given `[min, max]`. Should be non negative numbers.
   2. 亮度（浮点型 或 python:float 元组（最小值，最大值））—— 用于指定亮度抖动的程度。亮度因子从 `[max(0, 1 - 亮度), 1 + 亮度]` 中均匀选取，或者从给定的 `[最小值, 最大值]` 中选取。该值应为非负数。
2. **`contrast`** (float or tuple of python:float (min, max)) – How much to jitter contrast. 
   1. contrast_factor is chosen uniformly from `[max(0, 1 - contrast), 1 + contrast]` or the given `[min, max]`. Should be non-negative numbers.
   2. 对比度（浮点数 或 python:float 的元组（最小值，最大值））—— 用于指定对比度抖动的程度。对比度因子从 `[max(0, 1 - 对比度), 1 + 对比度]` 中均匀选取，或者从给定的 `[最小值, 最大值]` 中选取。该值应为非负数。
3. **`saturation`** (float or tuple of python:float (min, max)) – How much to jitter saturation. 
   1. saturation_factor is chosen uniformly from `[max(0, 1 - saturation), 1 + saturation]` or the given `[min, max]`. Should be non negative numbers.
   2. 饱和度（浮点型 或 Python 浮点型元组（最小值，最大值））——用于指定饱和度抖动的程度。饱和度因子从 `[max(0, 1 - 饱和度), 1 + 饱和度]` 中均匀选取，或从给定的 `[最小值, 最大值]` 中选取。该值应为非负数。
4. **`hue`** (float or tuple of python:float (min, max)) – How much to jitter hue. 
   1. hue_factor is chosen uniformly from `[-hue, hue]` or the given `[min, max]`. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space; thus it does not work if you normalize your image to an interval with negative values, or use an interpolation that generates negative values before using this function.
   2. 色调（浮点型 或 Python浮点型元组（最小值，最大值））—— 用于指定色调抖动的程度。色调因子是从 `[-色调，色调]` 或给定的 `[最小值，最大值]` 中均匀选取的。其取值应满足 0 ≤ 色调 ≤ 0.5，或 -0.5 ≤ 最小值 ≤ 最大值 ≤ 0.5。为了实现色调抖动，输入图像的像素值必须为非负值，以便转换到HSV色彩空间；因此，如果你将图像归一化到包含负值的区间，或者在使用此函数之前使用了会产生负值的插值方法，该功能将无法正常工作。
<br>

#### 4、`torchvision.transforms.Compose()`
[官网 doc - torchvision.transforms.Compose](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html?highlight=compose#torchvision.transforms.Compose)
```python
class torchvision.transforms.Compose(transforms)
```
1. **参数**
   1. transforms ( `list` of `Transform` objects ) – list of transforms to compose.
2. 例子
    ```python
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.PILToTensor(),
    >>>     transforms.ConvertImageDtype(torch.float),
    >>> ])
    ```
<br><br><br><br>

## 三、代码 - 训练模型
```python
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
from tools import Animator
from tools.ch13 import *
from datetime import datetime

all_images = torchvision.datasets.CIFAR10(train=True, root="data",
                                          download=False)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);

train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="data", train=is_train,
                                           transform=augs, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    # 在每个小批量函数里面把数据送到 GPU
    # 在最终的训练函数中把 net 送到 GPU
    # 虽然后面有梯度下降、参数更新
    # 但这里 net.train() 仍然不能少，因为这是提供训练环境的，并不是具体计算
    net.train()
    trainer.zero_grad()
    pred = net(X) 
    # 没毛病，训练的过程中就是必须搞预测。
    # 因为需要通过损失函数反向求导
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练: 明明只有 1 个 GPU, 看看代码具体是怎么写的！"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    # 明确：len(train_iter) 其实就是有多少批！并不是一批里有多少个样本！
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    # 也对，在最终的训练函数里把 net 送到 GPU
    # 而不是在每个小批量里面把 net 送到 GPU
    # 另外，把 net 送到 GPU 是指把参数送到 GPU 吗？答：不是！
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            # 由于 num_batches 是1 个 epoch 里面有多少批
            # ⇒ 因此，在一轮训练中，每完成 num_bacthes // 5 的任务 或者 最后一批，
            # 就更新一次可视化
            # 而且，从下面的代码可以看出：train_loss 和 train_acc 的更新频率高于 test_acc
            # 因为 test_acc 只能在训练完一个 epoch 之后才能更新
            # animator.add(x, (y1, y2, y3))
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    
batch_size, devices, net = 64, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

starttime = datetime.now() 
print(starttime) # 打印当前时间

train_with_data_aug(train_augs, test_augs, net)

endtime = datetime.now()
print(endtime)
print(endtime-starttime)

plt.show()
```

```powershell
# 实验结果 batch_size = 256
# 发现过拟合有点严重
2026-02-05 09:47:18.693108
loss 0.160, train acc 0.946, test acc 0.756
1824.7 examples/sec on [device(type='cuda', index=0)]
2026-02-05 09:53:07.321212
0:05:48.628104
```

```powershell
# 实验结果 batch_size = 64
# 还可以对训练数据集做数据增强，但是这里没做
# 发现过拟合已经改善很多了！
2026-02-05 09:56:20.379713
loss 0.161, train acc 0.945, test acc 0.867
1279.3 examples/sec on [device(type='cuda', index=0)]
2026-02-05 10:04:37.832930
0:08:17.453217
```

### （一）、`l.sum().backward()` OR `l.mean().backward()` ？
```python
# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 小批量训练代码中的反向计算
l = loss(pred, y)
l.sum().backward()
```

[关于 train_batch_ch13() 函数中 l.sum().backward() OR l.mean().backward() 的代码问题](https://www.bilibili.com/video/BV17y4y1g76q?comment_on=1&comment_root_id=111166280080&share_tag=s_i#reply111166280080)
1. 从后面 `torch.nn.CrossEntropyLoss(reduction)` 的讲解来看
   1. 其实 `none` 有意义
   2. `mean` 有意义
   3. `sum` 还有意义
   4. 只不过是 **开发者到底选择哪一个 `reduction` 方式** 的问题
2. 上面的代码实际上手动实现了 `nn.CrossEntropyLoss(reduction='sum')`

#### 1、为什么不能直接用 `l.backward()` ？
[官网 doc - torch.Tensor.backward()](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward)
```python
Tensor.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)
```
1. Computes the gradient of current tensor wrt graph leaves.
   1. 计算当前张量相对于图叶子节点的梯度。
2. The graph is differentiated using the chain rule. If the tensor is non-scalar (i.e. its data has more than one element) and requires gradient, the function additionally requires specifying a gradient. It should be a tensor of matching type and shape, that represents the gradient of the differentiated function w.r.t. self.
   1. 图使用链式法则进行微分。
   2. 如果该张量是**非标量**（即其数据包含**不止一个元素**）且需要梯度，则该函数还需要**指定一个 `gradient`**。
   3. 该梯度应该是一个类型和形状匹配的张量，表示微分函数相对于 `self` 的梯度。
   4. 也就是说：**`l.backward()` 不加参数就想直接使用，那么 `l` 就必须是标量（单个实数）**
3. This function **accumulates gradients** in the leaves - you might need to zero `.grad` attributes or set them to None before calling it. See Default gradient layouts for details on the memory layout of accumulated gradients.
   1. 此函数会在叶子节点中累积梯度 —— 在调用它之前，你可能需要将 `.grad` 属性清零或将其设置为 `None`。
   2. 有关累积梯度的内存布局的详细信息，请参见默认梯度布局。
```python
'''
这段例子代码又刷新了我对梯度计算的认识！
原来 sum() 也是独立的 x1**2 + x2**2 + x3**2
⇒ 那看来，以后可以广泛地应用 l.sum().backward() 了！
'''
>>> import torch
>>> x=torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> y = x**2
>>> y.sum().backward(retain_graph=True)
# 如果不指定 retain_graph 参数的话，
# 那么调用一次 backward()，计算图就会自动销毁
>>> x.grad
tensor([2., 4., 6.])
>>> y.mean().backward(retain_graph=True)
>>> x.grad
tensor([0.6667, 1.3333, 2.0000])
```

#### 2、`torch.nn.CrossEntropyLoss(reduction='none')` 多分类问题中的交叉熵损失函数
- 直接抄 [09-softmax.md](09-softmax.md) 中的内容
- [官网 doc - torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

```python
'''class CrossEntropyLoss 原型'''
class torch.nn.CrossEntropyLoss(
    weight=None, size_average=None, ignore_index=-100, 
    reduce=None, reduction='mean', 
    label_smoothing=0.0
)
```

##### (1)、功能详解
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

##### (2)、参数详解
1. **`reduction`** (`str`, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
   1. 三种参数讲解
      1. `'none'`: no reduction will be applied, 
      2. `'mean'`: the weighted mean of the output is taken, 
      3. `'sum'`: the output will be summed. 
   2. Note: 
      1. `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`.
      2. 指定 `size_average` 和 `reduce` 种任何一个参数，那么都会覆盖 `reduction`
   3. Default: `'mean'`
2. **`reduction`** 方式一共有 3 种
   1. `none`：就是原始的 交叉熵损失函数（ **不做 reduction** ）
      1. 返回的还是 `n` 维张量
   2. `mean`：直接对损失函数做 **平均**
      1. 返回的是标量（直接压缩为 `标量` ）
   3. `sum`：对损失函数做 **加和**
      1. 返回的是标量（直接压缩为 `标量` ）
3. **`C` 分类问题中的 `CrossEntropyLoss` 交叉熵损失函数** （ **根据 Pytorch 官网来的**！ ）
   - Class indices in the range $[0,C)$ where $C$ is the number of classes; 
   - if `ignore_index` is specified, this loss also accepts this class index (this index may not necessarily be in the class range).
   - **字母讲解**
     - 这里的 `w` 是**权重** （现在还没用到过！）
     - `n` 是这一批里面的第 `n` 个 （ **`n` 行** ）
     - `C` 是一共有 `C` 类 （ **`C` 列** ）
   1. The **unreduced** (i.e. with `reduction` set to **`'none'`**) loss for this case can be described as:
      1. `loss` 仍然是 **`n` 维张量**
      2. $$ℓ(x,y)=L=\{l_1,…,l_N\}^{T}, ~ ~ l_n=−w_{y_n}log {exp(x_{n,y_n}) \over \sum_{c=1}^{C}exp(x_{n,c})​}$$
         1. 分母是**按行求和**
         2. 分子是按 `(行, 列)` 索引**单个元素**
   2. $$ℓ(x,y)= \begin{cases} \sum_{n=1}^{N}{1 \over \sum_{n=1}^{N} w_n } ℓ_n ~ ~ ~ ~ ~ \text{if~ reduction=\sf{'mean'}} \\ \sum_{n=1}^{N} ℓ_n ~ ~ ~ ~ ~ \text{if~ reduction=\sf{'sum'}}\end{cases}$$
      1. `loss` 被压缩为标量（**一个实数**）

##### (3)、CELoss 就学到这里吧，再高深的东西，以后遇到再学！
<br><br>

### （二）、`resnet.train()`
- 提供 **训练模型** 的环境，并不是真正地做计算
<br><br>

### （三）、看看多 GPU 并行训练代码究竟是怎么写的？必要时结合视频课理解！
[官网 doc - class torch.nn.DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel)

```python
class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```

#### 1、多 GPU 并行的核心：`nn.DataParallel`
1.  **`device_ids=devices`**：指定要使用的 GPU 列表（由 `d2l.try_all_gpus()` 获取，返回当前机器上所有可用的 CUDA GPU，比如 `[cuda:0]` 或 `[cuda:0, cuda:1]` ）。
2.  **包装模型**：`nn.DataParallel` 不会修改原模型，而是创建一个"包装器"，把原模型的参数复制到 `device_ids` 中的**每一个GPU**上（每个 GPU 都会有一个完全相同的模型副本）。
3.  **主GPU（`devices[0]`）**：`DataParallel` 会将 `device_ids[0]` 作为**主GPU（master GPU）**，负责：
    - 接收完整的批量数据并拆分给其他 GPU。
    - 汇总其他 GPU 的计算结果（损失、梯度）。
    - 执行参数更新，并将更新后的参数同步回所有从 GPU。
<br>

#### 2、多 GPU 并行训练的完整执行流程（对应代码）
代码中的多GPU训练是一个"自动封装、透明执行"的过程，你无需手动处理数据拆分和梯度汇总，具体流程对应代码如下：

##### (1)、步骤1：获取可用GPU（初始化阶段）
```python
devices = d2l.try_all_gpus()
```
- 功能：检测当前机器的 CUDA 环境，返回可用的 GPU 设备列表（无 GPU 则返回 CPU ）。
- 作用：为后续 `nn.DataParallel` 提供待使用的GPU列表，实现"多GPU兼容、单GPU退化"的健壮性。

##### (2)、步骤2：包装模型（训练前准备）
```python
# 位于train_ch13函数中
net = nn.DataParallel(net, device_ids=devices).to(devices[0])
```
- 这是多 GPU 的"开关"，包装后，模型的前向传播、反向传播都会被 **`DataParallel` 拦截并做并行处理**。
- 注意：代码中先执行 `net.apply(init_weights)` 初始化参数，再进行 `DataParallel` 包装，这是最佳实践（避免包装后初始化导致多GPU参数同步异常）。

##### (3)、步骤3：数据预处理与加载（无感知适配）
```python
def load_cifar10(...)：
    dataloader = torch.utils.data.DataLoader(...)
```
- 数据加载阶段无需针对多 GPU 做特殊修改，`DataParallel` 会在模型前向传播时自动处理数据拆分。
- 代码中仅在 `train_batch_ch13` 中将数据移到 **主 GPU**（`X = X.to(devices[0])`、`y = y.to(devices[0])`），
  - 这是`DataParallel`的要求（主 GPU 负责接收完整数据后再分发）。

##### (4)、步骤4：多GPU并行前向传播（核心自动逻辑）
```python
# 位于train_batch_ch13函数中
pred = net(X)
```
当调用包装后的 `net(X)` 时，`DataParallel` 会自动执行以下操作（对用户透明）：
1.  **拆分数据**：将主 GPU 上的批量数据（比如 `batch_size=64` ）按GPU数量均分，分发到 `device_ids` 的所有GPU上（比如 2 个GPU，每个 GPU 分到 32 个样本）。
2.  **并行前向计算**：每个 GPU 使用自己的模型副本，对分到的局部数据执行前向传播，得到局部预测结果 `pred` 和局部损失 `l`。
3.  **汇总结果**：将所有从GPU的局部结果汇总到主GPU，拼接成完整的 `pred`（与原始批量数据尺寸一致），供后续计算整体损失。

##### (5)、步骤5：损失计算与反向传播（自动梯度汇总）
```python
# 位于train_batch_ch13函数中
l = loss(pred, y)
l.sum().backward()
```
1.  **损失计算**：代码中损失函数使用 `reduction="none"`，会先得到每个样本的独立损失，再通过 `sum()` 汇总（这是多 GPU 训练的推荐方式，避免默认 `mean` 导致损失计算偏差）。
2.  **自动梯度汇总**：`backward()` 执行时，`DataParallel` 会自动收集所有GPU上的局部梯度，汇总到**主 GPU** 的模型参数上，完成梯度求和。

##### (6)、步骤6：参数更新与同步（主GPU主导）
```python
# 位于train_batch_ch13函数中
trainer.step()
```
1.  只有 **主 GPU** 会执行参数更新（使用汇总后的梯度更新模型参数）。
2.  参数更新完成后，`DataParallel` 会自动将 **主 GPU** 上的新参数同步到所有从 GPU 的模型副本中，确保下一轮训练时，所有GPU的模型参数完全一致。
<br>

#### 3、关键细节补充（对应你代码中的注释）
1.  **为什么损失要用 `reduction="none"`**：
    - 多GPU训练时，每个GPU只处理部分数据，若使用默认的 `reduction="mean"`，会先在每个 GPU 上计算局部均值，再汇总到主 GPU 求全局均值，这会导致损失被重复平均（偏差）。
    - 用 `reduction="none"` 先得到每个样本的损失，再在 主 GPU 上 `sum()` 汇总，能保证损失计算的准确性。
2.  **单GPU场景的兼容性**：
    - 即使 `devices` 只有一个 GPU（比如 `[cuda:0]` ），`nn.DataParallel` 也能正常运行，此时不会进行数据拆分和多 GPU 复制，直接退化为单 GPU 训练，无需修改代码。
3.  **数据移至 主 GPU 的必要性**：
    - `DataParallel` 要求输入数据先位于 **主 GPU**（`devices[0]`），才能进行后续的分发操作，代码中 `X.to(devices[0])` 和 `y.to(devices[0])` 是必要步骤，否则会报设备不匹配错误。

