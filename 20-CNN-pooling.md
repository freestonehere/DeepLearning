# CNN - pooling （卷积神经网络 - 池化层）
## 一、概念理解
1. 之前实现的卷积对位置太敏感
   1. 比如：一个物体发生**细微的抖动**，就会导致**输出**发生**剧烈的变化**
   2. 这样其实不好，也就是说，我们希望：
      1. 即便物体发生一定程度的抖动，它输出依然是
      2. 该输出 1 还是 1
      3. 该输出 0 还是 0

### （一）、二维最大池化层
- 返回**窗口平均值**
- 和卷积层类似，也有填充和步幅，
  - 对于池化层**输出特征图的尺寸**，其计算方式和卷积层一样！
  - 但**没有可学习的参数**
  - 意思是**池化层没有卷积核**，直接输出最大值即可
- 在每个输入通道应用池化层以获得相应的输出通道
- 输出通道数 = 输入通道数
  - 也就是说：池化层**不会融合通道数**
  - 融合通道交给卷积层来做
  - 这样，就能让池化层变简单！
  - 事实上，**卷积层融合通道**也就是**将不同通道的输出相加**
- 输出每个窗口最强的信号

### （二）、二维平均池化层
- 和最大池化层一样
  - 只不过这里返回**窗口平均值**

### （三）、有意思的问题
1. 池化层放在卷积层的后面
2. 现在来看，**池化层** 已经不太经常用了
   1. 下面是一种理解
      1. 池化层的作用是让图像更稳定
         1. 而现在直接在卷积层加上 stride 即可！
      2. 另外，现在在数据中本身就会加入扰动
         1. 使得卷积本身已经不太容易过拟合到精准的位置
3. 矩阵拼接
   1. 先用 Python 本身 `list.append`
   2. 然后再将 `list` 转化为 `Tensor`

<br><br><br><br>

## 二、代码
```python
import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    # 池化层输出特征图的尺寸计算方式和卷积层一样
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    # 遍历行和列
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max': # 注意切片语法
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], 
                  [3.0, 4.0, 5.0], 
                  [6.0, 7.0, 8.0]])
print('二维最大池化层')
print(pool2d(X, (2, 2)), end='\n\n')

print('二维平均池化层')
print(pool2d(X, (2, 2), 'avg'), end='\n\n\n')


X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

# 默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同 
# ⇒ 两个相邻池化层之间不会有重合
pool2d = nn.MaxPool2d(3)
print('二维最大池化层')
print(pool2d(X), end='\n\n')

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
```

### （一）、`torch.nn.MaXPool2d` 类
[官网 doc - torch.nn.MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)

```python
class torch.nn.MaxPool2d(
    kernel_size, stride=None, padding=0, dilation=1, 
    return_indices=False, ceil_mode=False
)
```
1. 从 class 原型来看，看不出来 **步幅** 和 **池化窗口** 的关系
2. 另外，Python 中 **class name** `MaxPool2d` 会直接调用 `MaxPool2d.forward()` 前向计算函数

