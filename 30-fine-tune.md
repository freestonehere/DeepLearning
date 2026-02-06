# Fine Tune | 微调
## 一、概念理解
- 越往模型上层，和 **`label`** 的关联就越大
- 越往模型底层，和 **`feature` 特征本身** 的关联就越大
- 微调是 transformer learning （迁移学习）中的一种方法
  - 但迁移学习本身是一个很大的算法
- 比较流行的 CV 预训练模型，有 ResNet ……
- 微调对**学习率**不敏感，微调一般不调学习率

### （一）、为什么需要微调/迁移学习(Fine-tuning / Transfer Learning)
1. 在实际工程应用中，需要深度学习模型对特定的任务具有良好的泛化性，
   1. 如果从头训练该模型，则需要大量该任务专用的数据集才能使模型收敛到一个可接受的结果。
   2. 但是这个体量的专用数据集的标注成本也会很高。
2. 另一方面，例如在图像识别领域，已有许多在 ImageNet 等大规模数据集上训练好的模型，
   1. 其**靠近底层的神经元**，经训练已具备基本的**图像特征提取能力**，这有助于识别边缘、纹理、形状和对象组合。
   2. 我们可以在此模型基础上，使用相对少量的专有样本再训练，就可以得到比较好的结果。

### （二）、微调
#### 1、训练过程
![finetune](https://zh.d2l.ai/_images/finetune.svg)

- 在源数据集（例如 ImageNet 数据集）上预训练神经网络模型，即源模型。
- 创建一个新的神经网络模型，即目标模型。**复制源模型上的所有模型设计及其参数（输出层除外）**。
  - 我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。
  - 我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。
  - 我们假定 **`[1, L-1]`** 层都是用于 **提取特征** 的层，并且这些提取特征的层也使用于目标模型。因此可以采用这些层的参数
- 向目标模型添加输出层，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。
- 在目标数据集（如椅子数据集）上训练目标模型。输出层将从头开始进行训练，而所有其他层的参数将根据源模型的参数进行微调。
- 是一个目标数据集上的正常训练任务，但使用**更强的正则化**
  - 使用**更小的学习率**
  - 使用**更少的数据迭代**
  - 使用**更少的 epoch**
- 源数据集远复杂于目标数据（例如相差 10 倍以上），通常微调效果更好

### （三）、常用技术
#### 1、重用分类器权重
- 源数据集可能也有目标数据中的部分标号
- 可以使用预训练好模型分类器中对应标号对应的向量来做初始化

#### 2、固定一些层
- 神经网络通常学习有层次的特征表示
  - 低层次的特征更加通用
  - 高层次的特征则跟数据集相关
- 可以固定底部一些层的参数，不参与更新
  - 更强的正则

### （四）、总结
- 微调通过使用在大数据集上得到的预训练好的模型来初始化模型权重来完成精度提升，相当于使用了**先验知识**，也相当于将模型初始化在一个距离最优解附近的一个“模型初始化”方法
- 预训练模型质量很重要（比如一般选择 ImageNet 上预训练的 ResNet50 等）
- 微调通常速度更快、精度更高

<br><br><br><br>

## 二、代码讲解

```powershell
print my model pretrained_net.fc: Linear(in_features=512, out_features=1000, bias=True)
2026-02-06 11:03:30.062305
loss 0.190, train acc 0.930, test acc 0.931
237.3 examples/sec on [device(type='cuda', index=0)]
2026-02-06 11:05:15.953484
0:01:45.891179
```

### （一）、理解代码
#### 1、训练代码
```python
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```
#### 2、`torch.optim` 中的内容
- 直接抄 [08-LinearRegression.md](08-LinearRegression.md) 中的内容
- [官网 doc - torch.optim 首页](https://docs.pytorch.org/docs/stable/optim.html#module-torch.optim)

```python
# 没有 class prototype
```

##### (1)、如何使用 optimizer
1. To use `torch.optim` you have to construct an **optimizer object** that 
   1. will hold the current state 
   2. and will update the parameters based on the computed gradients.

- ① **construct optimizer object**
1. To construct an Optimizer you have to give it an **iterable（可迭代对象）** containing 
   1. the parameters (all should be Parameter s) 
   2. or named parameters (tuples of (str, Parameter)) to optimize.
   3. 所有参数都应为 `Parameter` 类型 / 命名参数（ `(str, Parameter)` 元组）
        ```python
        # 参数类型
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam([var1, var2], lr=0.0001)

        # 命名参数类型 (str, Parameter) 元组
        optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam([('layer0', var1), ('layer1', var2)], lr=0.0001)
        ```
- ② **Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc**.

##### (2)、Per-parameter options
1. For example, this is very useful when one wants to specify per-layer learning rates:
    ```python
    optim.SGD([
        {'params': model.base.parameters(), 'lr': 1e-2},
        {'params': model.classifier.parameters()}
    ], lr=1e-3, momentum=0.9)

    optim.SGD([
        {'params': model.base.named_parameters(), 'lr': 1e-2},
        {'params': model.classifier.named_parameters()}
    ], lr=1e-3, momentum=0.9)
    ```

##### (3)、take an optimization step
```python
optimizer.zero_grad()
optimizer.step()
```
1. **All optimizers** implement a `step()` method, that updates the parameters. It can be used in two ways
   1. 见官网
   2. 这里我就不抄了

<br>

### （二）、权重衰退 | weight decay
1. **权重衰退** 就是在 **优化器 `optim`** 里面做！
   1. `weight_decay` 越大，对参数的限制就越大
   2. `weight_decay = ∞` 时，参数就只能取 `0` 了！
   3. `weight_decay = 0` 时，对参数就没有限制了！

<br>

### （三）、`torchvision` 包
#### 1、`torchvision.datasets` 包
[官网 doc - torchvision.datasets.ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html?highlight=imagefolder#torchvision.datasets.ImageFolder)

```python
# ImageFolder 其实是创建了一个对象，也就是说 train_imgs 其实是一个对象
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))


hotdogs = [train_imgs[i][0] for i in range(8)]
# 这里 -i - 1 其实是反向索引
# -1 表示最后一个元素
# -1 - i 表示从最后一个元素开始，往前数 i 个
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
```

1. `ImageFolder` 其实是创建了一个对象，也就是说 `train_imgs` 其实是一个对象
2. `torchvision.datasets.ImageFolder` 是 PyTorch 专门用来加载按「文件夹分类」的图片数据集的工具
3. `train_imgs[i]` 取出第 `i` 个样本
   1. 每个样本是 `(pic, label)` 元组
   2. `train_imgs[i][0]` 其实就是取出每个**样本的图片**，而不取**样本的标签**

<br>

#### 2、`torchvision.transforms.Normalize` | 为什么要做 `channel normalization` ？和 `Batch Normalization` 类似吗 ？
[官网 doc - torchvision.transforms.Normalize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html?highlight=normalize#torchvision.transforms.Normalize)

```python
# 没想到 Normalize 居然也是个 class
class torchvision.transforms.Normalize(mean, std, inplace=False)
```

1. Normalize a tensor image with `mean` and `standard deviation`. 
   1. This transform does not support PIL Image. 
   2. Given `mean`: ( `mean[1],...,mean[n]` ) and `std`: ( `std[1],..,std[n]` ) for **`n` channels**, 
   3. this transform will normalize each channel of the **input `torch.*Tensor`** 
      1. i.e., `output[channel] = (input[channel] - mean[channel]) / std[channel]`
2. Note
   1. This transform acts **out of place**, i.e., it does not mutate the input tensor.
- 核心问题：为什么**一定要做 Normalize 标准化**？
  - 这是深度学习计算机视觉的通用必做步骤，尤其是用预训练模型微调（你的场景），不做的话模型训练会慢、收敛差，甚至完全训不出来，核心原因有 3 个，按重要性排序：
  - **原因 1**：匹配预训练模型的训练条件（微调的核心要求）
    - 你的 ResNet18 预训练权重，是在 ImageNet 数据集上用「这组均值 / 标准差（ `0.485/0.456/0.406` 和 `0.229/0.224/0.225` ）」做标准化训练出来的 —— 预训练模型的所有卷积层、全连接层，已经适应了标准化后的特征分布（均值 0、方差 1）。
    - 如果你的热狗数据集不做相同的标准化，**输入图片的特征分布和预训练模型训练时的分布完全不一致**，模型的**预训练权重相当于白学了**，微调的效果会极差，甚至比从零训练还差。
  - **原因 2**：加速模型的梯度下降收敛，让训练更稳定
    - 深度学习模型的训练核心是梯度下降，而梯度的计算对数值范围非常敏感：
    - 原始图片经 `ToTensor` 后是 0-1，但 RGB 三个通道的**像素值分布可能有差异**（比如 R 通道整体偏亮，均值 0.6；G 通道均值 0.4），数值分布的不一致会导致模型各层的梯度大小差异很大；
    - 梯度差异大的后果：优化器更新参数时会「忽快忽慢」，训练过程震荡，收敛速度极慢，甚至无法收敛到最优解；
  - **原因 3**：减少数值偏差，让模型更关注「特征本身」而非「像素值大小」
    - 未标准化的图片，像素值的绝对大小会影响模型的判断：比如两张热狗图片，一张偏亮（像素值整体大），一张偏暗（像素值整体小），模型可能会错误地把「像素值大小」当成核心特征，而忽略了「热狗的形状、纹理」这些真正的特征。

<br>

##### (1)、`torchvision.transforms.ToTensor()` 方法
- 照搬 [09-softmax.md](09-softmax.md) 中的内容
- [官网 doc - torchvision.transform.v2.ToTensor()，但是已弃用，下一版本中将会删除](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html#torchvision.transforms.v2.ToTensor)

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
