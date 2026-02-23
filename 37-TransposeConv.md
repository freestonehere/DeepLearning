# Transposed Convolution | 转置卷积
## 一、概念讲解
### （一）、转置卷积的概念
- 卷积不会增大输入的高宽，通常要么不变、要么减半
  - 语义分割需要在像素级别评估
- 转置卷积则可以用来增大输入高宽

![转置卷积的概念](https://zh-v2.d2l.ai/_images/trans_conv.svg)

$$Y[i\!:\!i+h, ~  j\!:\!j+w] ~ +\!= ~ X[i, ~ j] \cdot K$$

1. 转置卷积
   1. 不是为了还原某个像素的 RGB 值
   2. 而是为了预测这个像素的标号！

### （二）、为什么称之为 “转置”
- 对于卷积 $Y=X\star W$
  - 可以对 $W$ 构造一个 $V$ ，使得卷积等价于矩阵乘法 $Y^\prime=VX^\prime$
  - 这里 $Y^\prime,\ X^\prime$ 是 $Y,\ X$ 对应的向量版本
  - 矩阵形状 $(m, 1) = (m, n) \times (n, 1)$
  - 早期卷积就是这么做的
- 转置卷积则等价于 $Y^{\prime}=V^{T}X^{\prime}$
- 如果卷积将输入从 $(h, w)$ 变成了 $(h^\prime, w^\prime)$
  - 同样超参数的转置卷积则从 $(h^\prime, w^\prime)$ 变回 $(h, w)$，逆变换关系
  - 矩阵形状 $(n, 1) = (n, m) \times (m, 1)$
  - 交换了维度，原本的卷积维度减小变成了转置维度增加
- `行向量 = K * 列向量`
- 左乘行变换，右乘列变换；例子： $Ax^T=b$

### （三）、复习一下【卷积】的公式（不是【转置卷积】）
- 照抄 [18-CNN-pad-step.md](18-CNN-pad-step.md) 中的内容
- 给定高度 $s_h$ 和宽度 $s_w$ 的步幅，输出形状是： $$\lfloor \frac{n_h-k_h+p_h+s_h}{s_h}\rfloor\times\lfloor\frac{n_w-k_w+p_w+s_w}{s_w}\rfloor$$
  - **为了便于理解**，上面的公式其实是对的 $$\lfloor \frac{n_h-k_h+p_h}{s_h}+1\rfloor\times\lfloor\frac{n_w-k_w+p_w}{s_w}+1\rfloor$$
  - 这里的 $+1$ 其实就是加上第一个卷积窗口
    - 举例理解：对于 **输出图**，**从第一个**像素**到第二个**像素
    - 只跨了 1 步，但是输出图中却有 2 个像素
    - 所以，必须把第一个像素给加回来！
  - 为什么不是这样呢？**首先明确，下面这个公式一定是错误❌的！** $$\lfloor \frac{n_h-(k_h-1)+p_h}{s_h}+1\rfloor\times\lfloor\frac{n_w-(k_w-1)+p_w}{s_w}+1\rfloor$$
    - **明确正确公式的正确性**
    - `n-k` 是**输入**除去第一个卷积窗口外，**剩余可滑动**的输入长度
    - **至于第一个卷积窗口**产生的像素，在最后面**另加**！
    - **理解到这里就行了，不要钻牛角尖！**
- 如果 $p_h=k_h-1$，$p_w=k_w-1$  $$\lfloor\frac{n_h+s_h-1}{s_h}\rfloor\times\lfloor\frac{n_w+s_w-1}{s_w}\rfloor$$
  - 在步幅为 1 的情况下，可以让输出图尺寸和输入图保持一致！
  - 便于你实际计算
- 如果输入高度和宽度可以被步幅整除（步幅与卷积核大小相当）  $$\frac{n_h}{s_h}\times\frac{n_w}{s_w}$$

### （四）、【转置卷积】的公式
- 卷积是加权求和，转置卷积是加权分散
- **转置卷积的公式** $$(s_h n_h + k_h - p_h - s_h) \times (s_w n_w + k_w - p_w - s_w)$$
  - 即 $n^{\prime} = sn + k - p - s$
- **卷积的公式** $$\lfloor \frac{n_h-k_h+p_h+s_h}{s_h}\rfloor\times\lfloor\frac{n_w-k_w+p_w+s_w}{s_w}\rfloor$$
  - 即 $n^{\prime} = \lfloor \frac{n+p-k+s}{s} \rfloor$
- 可以发现：针对 **卷积公式**，如果把 $n^\prime$ 和 $n$ 调换一下位置，那就是 **转置卷积公式** 了！

<br><br><br><br>

## 二、代码实现
```python
手动实现转置卷积
trans_conv(X, K): tensor([[ 0.,  0.,  1.],
        [ 0.,  4.,  6.],
        [ 4., 12.,  9.]])
tconv(X): tensor([[[[ 0.,  0.,  1.],
          [ 0.,  4.,  6.],
          [ 4., 12.,  9.]]]], grad_fn=<ConvolutionBackward0>)

填充、步幅和多通道
tconv(X): tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)
tconv(X): tensor([[[[0., 0., 0., 1.],
          [0., 0., 2., 3.],
          [0., 2., 0., 3.],
          [4., 6., 6., 9.]]]], grad_fn=<ConvolutionBackward0>)
tconv(conv(X)).shape == X.shape: True

与矩阵变换之间的联系
Y: tensor([[27., 37.],
        [57., 67.]])
W: tensor([[1., 2., 0., 3., 4., 0., 0., 0., 0.],
        [0., 1., 2., 0., 3., 4., 0., 0., 0.],
        [0., 0., 0., 1., 2., 0., 3., 4., 0.],
        [0., 0., 0., 0., 1., 2., 0., 3., 4.]])
Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2):
 tensor([[True, True],
        [True, True]])
Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)
 tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
```
### （一）、理解代码
1. **转置卷积** 的高级 API（其实和 **卷积** 的高级 API 差不多）
    ```python
    # 使用高级 API 获得同样的结果
    X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
    tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
    tconv.weight.data = K
    print(f'tconv(X): {tconv(X)}')
    ```
