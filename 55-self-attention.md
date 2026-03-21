# Self Attention | 自注意力机制
## 一、概念讲解
### （一）、自注意力机制
- **明确：注意力机制本身的结构没有改变，只是 query, key, value 的选择比较特殊（即 query, key, value 都是输入本身！）**
- 给定一个词元序列 ${\bf x_1},...,{\bf x_n},\forall {\bf x_i}\in \mathbb R^d$ ，该序列的自注意力输出为一个长度相同的序列 ${\bf y_1},...,{\bf y_n}$ ，其中 $y_i \in \mathbb{R}^n$
- 自注意力池化层将 ${\bf x_i}$ 同时当作 key, value, query 来对序列抽取特征得到 ${\bf y_1},...,{\bf y_n}$，这里
  ${\bf y_i}=f({\bf x_i},({\bf x_1},{\bf x_1}),...,({\bf x_n},{\bf x_n}))\in \mathbb R^n$
  - 不需要额外的 key, value, query
  - 不需要 encode-decode 来处理序列

#### 1、跟 CNN, RNN对比
![self attention 跟 CNN, RNN 对比](https://zh-v2.d2l.ai/_images/cnn-rnn-self-attention.svg)

- CNN 也可以处理序列，把一个序列看作**高为 1 的图片**，1-d 卷积（注意是 `1-d` 卷积，不是 `1x1` 卷积），将每个 step 的特征 (word2vec) 作为 channel

|   | CNN  |  RNN | 自注意力  |
|---|---|---|---|
| 计算复杂度  | $O(knd^2)$  | $O(nd^2)$  | $O(n^2d)$  |
| 并行度  | $O(n)$  | $O(1)$  | $O(n)$  |
| 最长路径  | $O(n/k)$  | $O(k)$  | $O(1)$  |
  
- n: num_steps
- d: emd_sizes
- k: kernel_size
- 最长路径也叫做视野 (field)：相当于从把 step_1 的信息传递 step_n 需要的计算步数
- RNN 有非常强的时序性；而自注意力对序列长度更敏感，并行度非常好
- 自注意力机制适合长文本，因为视野宽，代价是计算复杂度高
- 理解计算复杂度（从 **张量形状 + 核心公式** 的角度去理解）
  - 卷积 `(batch_size, channel, h, w) → (batch_size, channel, h, w)` **且** 卷积操作就是加权求和！（注意：如果用卷积处理序列信息的话，那么**特征图尺寸**和**通道数**都不变，就是**类比 RNN 嘛**。）
    - 卷积的计算量类比 RNN 即可！
  - RNN `(num_steps, batch_size, num_hiddens) → (num_steps, batch_size, num_hiddens)` **且** $\bf H_n = X_n W_{xh} + H_{n-1} W_{hh} + b$
    - 一个时间步上的隐变量是 `(1, num_hiddens)`，变到下一个时间步上的隐变量 `(1, num_hiddens)`。这个过程要和 `(num_hiddens, num_hiddens)` 的矩阵相乘！也就是 $d^2$ 的计算量
    - n 个时间步的计算量之和就是 $nd^2$ ！
  - 自注意力 `(batch_size, num_steps, embed_size) → (batch_size, num_steps, embed_size)` **且** 【输入到输出的过程是线性的加权求和】，但【注意力权重的计算就比较复杂了，因为 **每一个 `query` 都要与所有 `key` 计算相似度！**】
    - 注意力权重的张量形状 `(batch_size, num_queries, num_keys)`
    - 一个时间步对应一个 query
    - 对一个时间步的 value 加权，计算复杂度是 $d$
    - 计算一个时间步的注意力权重，计算复杂度是 $n$
    - 一个时间步的总复杂度是 $nd$
    - 所有时间步的计算复杂度之和就是 $n^2d$

#### 2、计算复杂度的算法：
- CNN: 相当于每个输入通道 $d$ 做 $k \times n$ 的卷积后叠加，做输出通道数 $d$ 组卷积；
- RNN: 每一步的隐层是 $d\times d$ 的矩阵乘法，做 $n$ 步；
- 自注意力：每个 query 和所有 k-v pairs 做长度为 $d$ 的向量点乘

<br><br>

### （二）、位置编码
- 跟 CNN / RNN 不同，自注意力并没有记录位置信息
- 位置编码将位置信息注入到输入里
  - 假设长度为 $n$ 的序列是 ${\bf X}\in\mathbb R^{n \times d}$ ，那么使用位置编码矩阵 ${\bf P}\in \mathbb R^{n \times d}$ 来输出 ${\bf X} + {\bf P}$ 作为自编码输入
- $\bf P$ 的元素如下计算：
  $p_{i,2j}=\sin\left({i\over 10000^{2j/d}}\right)$ ， $p_{i,2j+1}=\cos\left({i\over10000^{2j/d}}\right)$
  - $i$ 表示一行文本里面第 $i$ 个 `token`，`d` 表示一个 `token` 的维数！
  - 奇数列和偶数列分别是周期不同的正弦余弦函数
  - 列越靠后周期越长


#### 1、绝对位置信息
- 计算机使用二进制编码
  - 0-1 的编码不断条约
  - 越靠前的行/列频率变化越快

#### 2、相对位置信息（数学好证明，不过直观上有什么含义呢？以后遇到再说吧！）
- 位置于 $i+\delta$ 处的位置编码可以线性投影位置 $i$ 处的位置编码来表示
- 记 $\omega_j=1/10000^{2j/d}$ ，那么 

$$
\begin{bmatrix}
    \cos(\delta\omega_j)&\sin(\delta\omega_j) \\
    -\sin(\delta\omega_j)&\cos(\delta\omega_j)
\end{bmatrix}
\begin{bmatrix} 
    p_{i,2j} \\ 
    p_{i,2j+1}
\end{bmatrix} = 
\begin{bmatrix} 
p_{i+\delta,2j} \\ 
p_{i+\delta,2j+1}
\end{bmatrix}
$$

- 投影矩阵跟 $i$ 无关
  - 也就是说：这是一个递推式！

<br><br>

### （三）、总结
- 自注意力池化层将 $x_i$ 当做 key, value, query 来对序列抽取特征
- 完全并行、最长序列为 1、但对长序列计算复杂度高
- 位置编码在输入中加入位置信息，使得子注意力能够记忆位置信息

<br><br>

## 二、代码讲解
```powershell
# 这是代码运行结果！

attention(X, X, X, valid_lens).shape 
torch.Size([2, 4, 100])

0的二进制是：000
1的二进制是：001
2的二进制是：010
3的二进制是：011
4的二进制是：100
5的二进制是：101
6的二进制是：110
7的二进制是：111
```

