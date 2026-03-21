# Attention Score | 注意力分数
## 一、概念讲解
把 [52-attention.py](52-attention.py) 中的核心代码拿出来
```python
# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        '''self.w 是形状为 (1) 的张量，用于对权重进行统一缩放。
        另外，真奇妙，单参数统一缩放居然就能拟合一个比较复杂的非线性函数！'''

    def forward(self, queries, keys, values):
        '''① 本来，输入的 queries 是一维张量 (n_train) 或 (n_test)，也就是 (查询个数)。
        repeat_interleave() 后， queries 成为二维矩阵张量 (查询个数, 键值对个数)，
        二维 queries 行内都是相同数值，但是行间是不同数值！
        
        ② keys 和 values 本身就是二维张量 (查询个数, 键值对个数)：
        它们行内是不同数值，但是行间对应位置是相同数值！
        
        ③ 返回一维张量 (查询个数)'''
        # queries 和 attention_weights 的形状为 (查询个数, “键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values 的形状为(查询个数，“键－值”对个数)

        '''为什么注意力机制中要有这么多 repeat()?
        就是因为最后要加权求和（通过矩阵乘法实现的），把 repeat 后的二维张量还原成一维张量！
        但是注意： softmax 本身不改变张量维度，只是归一化！'''
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
        # attention_weights 张量形状 (查询个数, “键——值”对个数) ⇒ (查询个数, 1, “键——值”对个数)
        # values 张量形状 (查询个数, “键——值”对个数) ⇒ (查询个数, “键——值”对个数, 1)
        # 这代码写的真牛！
```

1. [52-attention.py](52-attention.py) 中，所有东西（查询、键、值）都是标量
   1. 现在，要把它们全都拓展到高维向量
   2. 也就是：把 **查询、键、值** 都从标量变成向量！
   3. 它们都变成向量的话，还可以分成 **2 种情况**
2. 一种情况是：**查询** 和 **键** 的向量长度不同，此时采用 **加性注意力** 方法。
3. 另一种情况是：**查询** 和 **键** 的向量长度相同，此时采用 **缩放点积** 方法。
4. 无论是那种情况，**最终最终的输出一定是 `(查询个数, 1)`**
<br>

### （一）、回顾 [52-attention.py](52-attention.py) 中【`query`, `key`, `value` 全都是标量】的情况
$f(x)=\sum_{i=1}^nsoftmax\left(-{1\over2}((x-x_i))^2\right)y_i$

- 其中，被 normalize 之前的 $-{1\over2}((x-x_i))^2$，称之为注意力分数 (attention scoring function)

**拓展到高维度**

- 假设 query ${\bf q}\in\mathbb R^n$， $m$ 对 key-value $(\bf k_1,v_1),...$ ，这里 ${\bf k_i}\in\mathbb R^k$ ， ${\bf v_i}\in\mathbb R^v$
- 注意力池化层：
  - $$f({\bf q,(k_1,v_1),...,(k_m,v_m)})=\sum_{i=1}^m\alpha({\bf q,k_i}){\bf v_i}\in\mathbb R^v$$
  - $$\alpha({\bf q,k_i})=softmax(a({\bf q,k_i}))={\exp(a({\bf q,k_i}))\over\sum_{j=1}^m\exp(a({\bf q,k_j}))}\in\mathbb R$$
<br>

### （二）、Additive Attention（加性注意力）【向量 `query` 和 `key` 长度不同的情况】
- 可学参数： ${\bf W_k}\in\mathbb R^{h\times k},{\bf W_q}\in\mathbb R^{h\times q},{\bf w_v}\in\mathbb R^h$ <br> $a({\bf k,q})={\bf w_v}^Ttanh({\bf W_kk+W_qq })$
    - 这里的 $h$ 代表 `hidden_size`！
    - 等价于将 key 和 value 合并起来后放入到一个隐藏层大小为 h 输出大小为1的单隐藏层MLP
    - 输出一个标量
    - $\bf q,k,v$ 可以是不同长度
- 拓展维度： ${\bf Q}\in\mathbb R^{n\times q},\ {\bf K}\in\mathbb R^{m\times k},\ V^{m\times v}$
  - $a({\bf K,Q})={\bf w_v}^Ttanh({\bf KW_k^T+QW_q^T })$
  - 代码中使用的是 **矩阵 $\bf Q, K$**
  - 而不是 **向量** $\bf q, k$
- **优点**：`query`, `key`, `value` **的长度可以各不相同！** 

<br>

### （三）、Scaled Dot_product Attention（缩放点积注意力）【向量 `query` 和 `key` 长度相同的情况】
- 如果 query 和 key 都是同样的长度， ${\bf q,k_i}\in\mathbb R^d$ ，那么可以：
        $a({\bf q_i,k})=<{\bf q,k_i}>/\sqrt d$
  - 除以根号 d 使对**向量长度**不敏感
- 向量化版本
  - ${\bf Q}\in\mathbb R^{n\times d},\ {\bf K}\in\mathbb R^{m\times d},\ V^{m\times v}$
  - 注意力分数： $a({\bf Q,K})={\bf QK}^T/\sqrt d\in\mathbb R^{n \times m}$
  - 注意力池化： $f=softmax(a({\bf Q,K})){\bf V}\in\mathbb R^{n \times v}$
  - 有 $n$ 个 query， $m$ 个 key，每个 key 有 $v$ 个 value
- 和 **Additive Attention** 相比，**Scaled Dot_product Attention** 实现简单（超参数少）
  - 但坏处就是：因为你的东西比较少，所以学到的东西可能会比较少。

<br>

### （四）、总结

- 注意力分数是 query 和 key 的相似度，注意力权重是分数的softmax结果
- 两种常见的分数计算：
  - 将 query 和 key 合并起来进入一个单输出单隐藏层的MLP
  - 直接将 query 和 key 做内积

<br><br>

## 二、代码讲解
```powershell
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])): 
tensor([[[0.4307, 0.5693, 0.0000, 0.0000],
         [0.3051, 0.6949, 0.0000, 0.0000]],

        [[0.4187, 0.3693, 0.2121, 0.0000],
         [0.3902, 0.2318, 0.3779, 0.0000]]])

masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])):
tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
         [0.3801, 0.2415, 0.3783, 0.0000]],

        [[0.5659, 0.4341, 0.0000, 0.0000],
         [0.2090, 0.3593, 0.2397, 0.1920]]])

attention(queries, keys, values, valid_lens)
tensor([[[ 2.0000,  3.0000,  4.0000,  5.0000]],

        [[10.0000, 11.0000, 12.0000, 13.0000]]], grad_fn=<BmmBackward0>)

attention(queries, keys, values, valid_lens): 
tensor([[[ 2.0000,  3.0000,  4.0000,  5.0000]],

        [[10.0000, 11.0000, 12.0000, 13.0000]]])
```

### （一）、理解代码
#### 1、原来 `dropout` 保证加和为 1 吗？
- 有点忘了。
- 看看之前 `dropout` 的笔记！


<br>

### （二）、`nn.functional.softmax()` 函数
[官网 doc - class torch.nn.Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax)
```python
class torch.nn.Softmax(dim=None)
```
- 理解 `softmax` 中的样本！
1. `softmax` 中的样本好像是一个序列？
2. `shape` 解释
   1. Input：`(*)`
      1. `*` 代表**任意维度、任意形状**的张量都可以输入。
      2. 比如：1 维、2 维、3 维、4 维 …… 都行。
   2. Output：`(*)`
      1. 输出张量的**维度、形状和输入完全一样**，不会改变尺寸。
3. 一句话总结：Softmax 只改变数值，不改变张量形状。

<br>

### （三）、广播机制
```python
>>> a = torch.tensor((1, 2, 3), dtype=torch.float32)
>>> a = a.reshape(3, 1)
>>> b = torch.tensor((4, 5, 6), dtype=torch.float32).reshape(1, -1)

# 张量 a, b 的形状
>>> a.shape
torch.Size([3, 1])
>>> b.shape
torch.Size([1, 3])

# 张量 a, b 的内容
>>> a
tensor([[1.],
        [2.],
        [3.]])
>>> b
tensor([[4., 5., 6.]])

# 张量 (a + b) 的形状和内容
>>> (a + b).shape
torch.Size([3, 3])
>>> a + b
tensor([[5., 6., 7.],
        [6., 7., 8.],
        [7., 8., 9.]])
```