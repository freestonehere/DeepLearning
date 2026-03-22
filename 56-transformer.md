# Transformer | Encoder-Decoder 架构下的 Transformer
## 一、概念讲解
|架构|图解|
|:---|:--|
|Transformer|![transformer 架构](https://zh-v2.d2l.ai/_images/transformer.svg)|
|seq2seq with<br>attention|![seq2seq with attention](https://zh-v2.d2l.ai/_images/seq2seq-attention-details.svg)|

- 基于编码器-解码器架构来处理序列对
- 跟使用注意力的 seq2seq 不同，Transformer 是纯基于注意力

### （一）、多头注意力
![多头注意力](https://zh-v2.d2l.ai/_images/multi-head-attention.svg)
- 对**同一 key, value, query**, 希望**抽取不同的信息**
  - 例如短距离关系和长距离关系
    - 通过全连接层变成长度小的 $d$
    - 分别进入各自的注意力机制头
- 多头注意力使用 $h$ 个独立的注意力池化
  - 合并 (Concat) 各个头 (head) 输出得到最终输出
  - 最后经过一个全连接层获得输出的维度

- `query` ${\bf q}\in\mathbb R^{d_q}$ ，`key` ${\bf k}\in \mathbb R^{d_k}$ ，`value` ${\bf v}\in\mathbb R^{d_v}$
- 头 $i$ 的可学习参数 ${\bf W}_i^{(q)}\in\mathbb R^{p_q\times d_q}$ ， ${\bf W}_i^{(k)}\in\mathbb R^{p_k\times d_k}$ ， ${\bf W}_i^{(v)}\in\mathbb R^{p_v\times d_v}$
- 头 $i$ 的输出 ${\bf h}_i=f({\bf W}_i^{(q)}{\bf q},{\bf W}_i^{(k)}{\bf k},{\bf W}_i^{(v)}{\bf v})$
- 输出的可学习参数 ${\bf W}_o\in\mathbb R^{p_o\times hp_v}$
- 多头注意力的输出
  
$$
{\bf W_o} 
\begin{bmatrix}
{\bf h}_1 \\ 
\vdots \\ 
{\bf h}_h 
\end{bmatrix} 
\in\mathbb R^{p_o}
$$

```python
# 定义 2 个转置函数
#@save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状（把 num_heads 纳入张量形状！）"""
    
    '''输入 X 的形状: (batch_size, 查询或者“键－值”对的个数, num_hiddens)
    输出 X 的形状: 
    (batch_size, 查询或者“键－值”对的个数, num_heads, num_hiddens / num_heads)'''
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出 X 的形状: (batch_size, num_heads, 查询或者“键－值”对的个数, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状: (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

#@save
def transpose_output(X, num_heads):
    """逆转 transpose_qkv 函数的操作
    
    输入 X 的张量形状 (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens / num_heads)"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

#@save
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = attn.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values 的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens 的形状:
        # (batch_size，) 或 (batch_size，查询的个数)
        # 经过变换后，输出的 queries，keys，values　的形状:
        # (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens / num_heads)
        '''因为不想使用 for loop 遍历 num_heads 次循环，所以要在原来张量的基础上把 num_heads 考虑张量！
        （也就是需要进行转置！）
        
        通过转置 ① 隐含地实现了拼接操作 ② 便于 Q K 进行点积注意力运算
        
        务必明确 2 个原则：① 多头注意力中 num_heads 是和 batch_size 平行的级别！
        ② 所有头的 num_queries 是相同的，同时所有头的 query_size 也都是相同的！'''
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴 0，将第一项（标量或者矢量）复制 num_heads 次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output 的形状: (batch_size * num_heads, 查询的个数, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat 的形状: (batch_size, 查询的个数, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

#### 1、`transpose_qkv` 中，不能拆成 `(batch_size * num_heads, num_queries / num_heads, num_hiddens)`，只能拆成 `(batch_size * num_heads, num_queries, num_hiddens / num_heads)`
1. 序列长度无法被头数整除
   1. `num_queries / num_heads` 是序列长度维度拆分，但序列长度（比如 10）是「实际的文本 / 序列长度」，通常是任意整数（比如 7、10、15），几乎不可能被头数（8/16/32）整除
2. 注意力计算的维度完全错位
   1. 注意力的核心是「每个头对**完整的序列**计算注意力」，而不是把序列拆分给不同的头。
   2. 正确逻辑（拆分隐藏层）：
      1. 每个注意力头看到的是完整的序列（10 个查询），但只用「低维的特征（16 维）」计算
   3. 如果（拆分序列长度）：
      1. 每个注意力头只能看到序列的一部分，完全违背多头注意力的设计初衷：
      2. 头 1：只计算 1 个查询 × 2 个键 的注意力（序列被拆分）；
      3. 头 2：只计算 1 个查询 × 2 个键 的注意力；
      4. ...（每个头只能看到序列的碎片，无法捕捉完整的序列依赖）


### （二）、有掩码的多头注意力
- 解码器对序列中一个元素输出时，不应该考虑该元素之后的元素
  - Attention 的视野可以看到全部的元素
  - 编码器可以，解码器不行
- 可以通过掩码来实现
  - 也就是计算 ${\bf x_i}$ 输出时，假装当前序列长度为 ${\bf i}$
    - 在实现上做 softmax 时不给后面的元素权重

### （三）、基于位置的前馈网络
- 将输入形状由 $(b,n,d)$ 变换成 $(bn,d)$
  - $n$ 是序列长度，与模型无关，会发生改变
  - 所以对序列里每个元素做全连接
- 作用两个全连接层
- 输出形状由 $(bn,d)$ 变化回 $(b,n,d)$
- 等价于两层核窗口为 $1$ 的一维卷积层

### （四）、层归一化
- **层归一化本身是十分灵活的，不必钻牛角尖！**
- 批量归一化对每个特征/通道里元素进行归一化
  - 不适合序列长度会变的NLP应用
  - 也就是 $d$ 对 $b \times n$ 做归一化
- 层归一化对每个样本里的元素进行归一化

![层归一化 Layer Norm](./myPic/56-Transformer/01-层归一化.png)

### （五）、ResNet
- 每一模块会用ResNet合并模块输出一同做归一化
- 因此输出的自注意力模块的输出 dim 与输入相同 

### （六）、信息传递
- 编码器中的输出 ${\bf y_1},...,{\bf y_n}$
- 将其作为解码中第 $i$ 个 Transformer 块中多头注意力的 key 和 value
  - 它的 query 来自目标序列
  - 解码器的每一层（除了第一层SelfAttention以外）都加
- 意味着编码器和解码器中块的个数和输出维度都是一样的
  - 简单对称
  - 也就是贯穿始终的 $d$ 不变性了，非常优秀

### （七）、预测
- 预测第 $t+1$ 个输出时
- 解码器中输入前 $t$ 个预测值
  - 在子注意力中，前 $t$ 个预测值作为 key 和 value，第 t 个预测值还作为 query
  - 还是顺序进行的

### （八）、总结
- Transformer是一个纯使用注意力的编码-解码器
- 编码器和解码器都有 $n$ 个transformer块
- 每个块里使用多头（自）注意力，基于位置的前馈网络，和层归一化

### （九）、看看 Tranformer 的上下文是多长？

<br><br>

## 二、代码讲解
```python
# 这是代码运行结果

ffn(torch.ones((2, 3, 4)))[0]: 
tensor([[ 0.1033, -0.1269, -0.1930, -0.3250, -0.4673,  0.0860, -0.3943,  0.2874],
        [ 0.1033, -0.1269, -0.1930, -0.3250, -0.4673,  0.0860, -0.3943,  0.2874],
        [ 0.1033, -0.1269, -0.1930, -0.3250, -0.4673,  0.0860, -0.3943,  0.2874]],
       grad_fn=<SelectBackward0>)

layer norm:
tensor([[-1.0000,  1.0000],
        [-1.0000,  1.0000]], grad_fn=<NativeLayerNormBackward0>)

batch norm:
tensor([[-1.0000, -1.0000],
        [ 1.0000,  1.0000]], grad_fn=<NativeBatchNormBackward0>)

add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape
torch.Size([2, 3, 4])

encoder_blk(X, valid_lens).shape
torch.Size([2, 100, 24])

encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape
torch.Size([2, 100, 24])

加入位置编码信息前的词向量
nn.Embedding(200, 24, torch.ones(2, 100))[0][0]
tensor([-0.1895, -0.2969, -1.3700, -0.8913, -1.2275, -0.8055,  0.7443,  0.2928,
        -1.3917, -0.0078,  0.0601,  0.7912, -1.2219, -0.9332, -0.0526, -0.5193,
         0.6159,  0.3171, -0.7736, -2.0109, -0.6860, -1.3158, -0.4701, -0.5426],
       grad_fn=<SelectBackward0>)

加入位置编码信息前的词向量均值
nn.Embedding(200, 24, torch.ones(2, 100))[0][0].mean()
0.07729250192642212

加入位置编码信息前的词向量 L2 范数
nn.Embedding(200, 24, torch.ones(2, 100))[0][0].norm()
5.622831344604492

纯位置编码信息
attn.PositionalEncoding(24, 0.5).P[0][0]
tensor([0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
        0., 1., 0., 1., 0., 1.])

加入位置编码信息后的词向量
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0]
tensor([-0.4171, -0.9387, -1.2216,  0.5021, -1.8376,  1.3179, -0.7992, -0.9835,
         0.0476,  0.4639,  1.3233, -0.1767,  0.5068, -0.6837,  1.5620,  1.0962,
         0.0346,  0.2692, -0.5030,  1.8597, -1.7511,  0.3326,  0.0144,  0.7433],
       grad_fn=<SelectBackward0>)

加入位置编码信息后的词向量均值
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].mean()
0.03171844407916069

加入位置编码信息后的词向量的 L2 范数
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].norm()
4.8373122215271

decoder_blk(X, state)[0].shape
torch.Size([2, 100, 24])

loss 0.031, 7991.5 tokens/sec on cuda:0
go . => va !,  bleu 1.000
i lost . => j'ai perdu .,  bleu 1.000
he's calm . => il est calme .,  bleu 1.000
i'm home . => je suis chez moi .,  bleu 1.000

enc_attention_weights.shape
torch.Size([2, 4, 10, 10])

dec_self_attention_weights.shape 
torch.Size([2, 4, 6, 10])

dec_inter_attention_weights.shape
torch.Size([2, 4, 6, 10])
```
### （一）、理解代码
#### 1、整体架构
- 整体架构还是和 `RNN` 一样！因为都是继承的 `EncoderDecoder` 接口！
- 下面把 `class EncoderDecoder` 接口的定义抄过来！

```python
net = nlp.EncoderDecoder(encoder, decoder)


'''编码器-解码器接口'''
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # 从编码器的【输入】得到编码器的【输出】
        enc_outputs = self.encoder(enc_X, *args)
        # 从编码器的【输出】得到解码器的【状态】
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 从解码器的【状态】和解码器的【输入】得到解码器的【输出】
        return self.decoder(dec_X, dec_state)


#@save
class TransformerEncoder(nlp.Encoder):
    """堆叠了 num_layers 个 TransformerEncoderBlock!"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = attn.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        '''因为位置编码值在 -1 和 1 之间，因此嵌入值乘以嵌入维度的平方根进行缩放，
        然后再与位置编码相加。
        
        ① 将嵌入向量的尺度放大，使其与位置编码的尺度处于同一数量级'''
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
            # 这里 blk.attention.attention.attention_weights
            # = TransformerEncoderBlock.MultiHeadAttention.DotProductAttention.attention_weights
        return X


class TransformerDecoder(attn.AttentionDecoder):
    '''和 TransformerEncoder 一样，堆叠了 num_layers 个 TransformerDecoderBlock'''
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = attn.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            '''每个 TransformertDecoderBlock 的【① 输出】和
            【② state = [enc_output, enc_valid_lens, dec_output + 当前输入（预测时使用）]】
            都不必保留！'''
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```
<br><br>


#### 2、`TransformerEncoderBlock` 中张量形状变化
1. 首先明确 `TransformerEncoderBlock` 的结构
   1. 一个 **自注意力** 层
   2. 一个 **PositionWiseFFN（全连接层）**
2. 自注意力层（`query = key = value = X`）不改变输入张量的形状，只是对 `X` 的加权求和
3. **只有全连接层改变输入张量的维度，且只改变最后一个维度！**

<br><br>

#### 3、已知 `TransformerDecoder` 有 `num_layers != num_steps` 个 `TransformerDecoderBlock`，那么每个 `TransformerDecoderBlock` 的 `valid_lens` 是如何确定的？
##### (1)、对于 `enc_balid_lens` 其实比较简单
- 对于 `enc_balid_lens` 其实比较简单
- 因为 `enc_valid_lens` 目的是屏蔽 `enc_output` 中的 `<pad>`
- 然后从代码中追踪 `enc_valid_lens` 的更新路径
1. 发现 `class DecoderBlock` 的定义代码里面，`enc_valid_lens` 来自于 `DecoderBlock` 中的 `state[1]`
   1. 但是 `DecoderBlock` 从来没有更新过 `state[1]`
   2. 那就继续往前回溯，看看 `TransformerDecoder` 的第 `1` 个 `Block` 中的 `enc_valid_lens (也就是 state[1])` 是怎么来的
2. 发现 `TransformerDecoder` 的第 `1` 个 `Block` 中的 `enc_valid_lens (也就是 state[1])` 来自于 `TransformerDecoder.init_state()` 函数
   1. 在这个 `init_state` 函数中，`state` 来自于 `enc_output`！
3. **总结来说**
   1. encoder 生成 `enc_output`
   2. decoder 的 `init_state` 函数依据 `enc_output` 生成 `enc_valid_lens`
   3. 然后在整个 `decode` 过程中，`enc_output` 和 `enc_valid_lens` 都**保持不变**！

<br>

##### (2)、对于 `dec_valid_lens` 呢？
- 还是先明确 `dec_valid_lens` 的**目的**：解码的时候不能看到**未来词元**！
- 回溯 `dec_valid_lens` 的产生过程，发现就是 **每个 `DecoderBlock` 生成自己的 `dec_valid_lens`**！
- **还有一个非常重要的点：`dec_valid_lens` 只在 `train` 模式下启用**，用于屏蔽未来词元。
  - 预测模式下根本不需要 `dec_valid_lens`，因为根本就没有未来词元！
- 具体生成过程如下
1. `dec_valid_lens` 依据每行文本的 `num_steps` 生成
2. `num_steps` 又由 `X.shape[1]` 确定（注意：`X` 是上一个 `DecoderBlock` 的输出）
3. 那么不同的 `DecoderBlock` 中，`num_steps` 数值相同吗？
   1. 显然相同！
   2. （辅助理解：`dec_valid_lens` 只在 train 模式下启用！）

<br><br>

#### 4、`RNN` 和 `Transformer` 在并行度上的根本区别
1. 如果采用 `RNN` 结构，那么就算加入了注意力机制，也依然无法并行！
   1. 参考 [54-seq2seq-attention.md](./54-seq2seq-attention.md) 中的问题 `二/（二）/3`
   2. 不过在计算上虽然无法并行，但是预测准确率可能会提升！
2. 而使用纯注意力机制的 `Transformer` 就可以并行！
3. **总结**
   1. RNN：计算第 t 步的 hidden_state 必须先有 t-1 步的 hidden_state（硬依赖，无法并行）；
   2. Transformer：所有时间步的 query/key/value 可以一次性生成，通过掩码过滤掉未来信息后，一次性计算所有位置的注意力（无硬依赖，可并行）。
<br><br>

#### 5、每个 `DecoderBlock` 是怎么和对应的 `EncoderBlock` 实现信息交互的呢？
- 因为 `Encoder-Decoder` 的代码是【先把所有 encoder 执行完】，【然后再执行所有 decoder】
  - 这样看，`DecoderBlock` 和 `EncoderBlock` 好像没有信息交互的机会啊！
- 解答如下：**编解码器的信息交互路径**
1. Transformer 的编解码器交互不是 “块对块” 的一一对应，而是：
2. 编码器所有块执行完毕 → 输出**完整的编码器全局特征** → 作为解码器所有块的「公共输入」 → 解码器每个块都通过「编码器 - 解码器注意力层」与编码器输出交互
- 既然**编解码器的信息交互路径**是这样的，那么「编码器 - 解码器注意力层」的 `Q/K/V` 分别是从哪里来的呢？
    ```python
    '''自注意力，其中 dec_valid_lens 用于屏蔽未来词元。
    【自注意力层的作用】：建模解码器输出序列内部的依赖关系。
    【原理理解】：以当前输入作为 query ，查询当前输入与之前已生成之间的关系！
    然后对之前已生成的东西进行加权求和。'''
    X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
    Y = self.addnorm1(X, X2)

    '''编码器－解码器注意力，其中 enc_valid_lens 用于屏蔽 <pad> 。
    【编码器-解码器自注意力层的作用】：建立解码器输出与编码器输入的关联（即 “对齐”）。
    【原理理解】：以解码器最新生成的东西作为 query ，查询 decoder 的最新生成与 enc_output 之间的关系！
    然后对 enc_output 进行加权求和。'''
    # enc_outputs 的开头: (batch_size, num_steps, num_hiddens)
    Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
    Z = self.addnorm2(Y, Y2)
    ```
    1. 从代码来看，「编码器 - 解码器注意力层」使用「解码器自注意力层」的输出 `Y` 作为 `query`，使用 `enc_output` 作为 `key, value`
    2. 来实现编码器和解码器的对齐！

<br><br>

#### 6、还有一个问题：encoder 是只给 decoder 一个 `enc_output` 还是每个 `EncoderBlock` 都给出一个 `enc_output`？
1. **答**：从问题 `二/（一）/5` 可以看出，**整个 `TransformerEncoder`** 一共给出**一个** `enc_output`！

<br><br>

### （二）、Norm 层
|Norm 方式|图解|
|:-------|:---|
|Layer Norm|![Layer Norm 图解](https://docs.pytorch.org/docs/stable/_images/layer_norm.jpg)|
||Layer Norm 确实是在一个样本内部**对样本做归一化**|
|Batch Norm 1d|BatchNorm 则是 跨样本**对特征做归一化**|

#### 1、Layer Norm
- [官网 doc - nn.LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm)

```python
class torch.nn.LayerNorm(normalized_shape, eps=1e-05, 
    elementwise_affine=True, bias=True, device=None, 
    dtype=None)
```


```python
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module
layer_norm(embedding)

# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)
```

##### (1)、本代码实例
```python
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# 在训练模式下计算 X 的均值和方差
print('\nlayer norm:', f'\n{ln(X)}\n', '\nbatch norm:', f'\n{bn(X)}')

'''
layer norm 和 batch norm 的输入张量：
X = torch.tensor([[1, 2], 
                 [2, 3]], 
                 dtype=torch.float32)
'''

# 下面是输出张量：layer norm 对样本归一化，batch norm 对特征归一化
layer norm: 
tensor([[-1.0000,  1.0000],
        [-1.0000,  1.0000]], grad_fn=<NativeLayerNormBackward0>)
 
batch norm: 
tensor([[-1.0000, -1.0000],
        [ 1.0000,  1.0000]], grad_fn=<NativeBatchNormBackward0>)
```

#### 2、Batch Norm
- [官网 doc - nn.BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d)
```python
# With Learnable Parameters
m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = torch.randn(20, 100)
output = m(input)
```
<br><br>

### （三）、`TransformerEncoder` 中的 Embedding 层为什么要乘以 `嵌入层维数长度的平方根`（即便有这些数据，我也不理解！先放一放吧！）<br>不过有一点可以确定：这个缩放是为了让 `位置编码的尺度` 和 `词向量的尺度一致`！
```python
class TransformerEncoder(nlp.Encoder):
    """Transformer 编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = attn.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        '''因为位置编码值在 -1 和 1 之间，因此嵌入值乘以嵌入维度的平方根进行缩放，
        然后再与位置编码相加。
        
        ① 将嵌入向量的尺度放大，使其与位置编码的尺度处于同一数量级'''
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

#### 1、直接照抄 [50-seq2seq.md](./50-seq2seq.md) 中的内容
- [官网 doc - class torch.nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding)

```python
class torch.nn.Embedding(
    num_embeddings, embedding_dim, padding_idx=None, 
    max_norm=None, norm_type=2.0, scale_grad_by_freq=False, 
    sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
```

##### (1)、Input：`(*)`, IntTensor / LongTensor
- `(*)` 代表**任意形状**，可以是：
  - 一维：`[1,2,3]`
  - 二维：`[[1,2],[3,4]]`
  - 三维、四维……都行
- 里面必须是**整数索引**，类型是 `int64`（LongTensor）。

##### (2)、Output：`(*, H)`，H = embedding_dim
- 输出形状 = **输入形状后面多一维**
- 多出来的那一维长度 = 你设置的 `embedding_dim`

##### (3)、一句话总结
**输入是任意形状的索引，输出就是在每个索引位置，多嵌一个长度为 `embedding_dim` 的向量。**

举个最简例子：
- 输入形状：`(2, 4)`
- embedding_dim = 3
- 输出形状：`(2, 4, 3)`
<br>

#### 2、看看 `nn.Embedding` 是如何初始化的！
- `weight` 是 **Embedding 层里可学习的参数矩阵**
- 形状是 `(词典大小, 嵌入向量维度)` 也就是 `(num_embeddings, embedding_dim)`
- 初始化方式：**标准正态分布 N(0,1)**，也就是均值为0、方差为1的正态分布随机数。

##### (1)、从 `Embedding` 的【权重】出发，推导 `Embedding` 的【词向量】数量级
1. `Embedding` 的权重张量形状为 `(num_embeddings, embed_size) = (vocab_size, embed_size)`
2. 又由于 **权重** 和 **词向量** 之间有如下关系
    |维度 / 属性|Embedding 层权重矩阵|输出的词向量|
    |:---------|:------------------|:----------|
    |本质| 可学习的参数（全局唯一）| 从权重矩阵中 “索引取值” 得到的结果（动态）|
    |形状| `(vocab_size, H)` （ `H = embedding_dim` ）| `(输入形状，H)` |
    |取值逻辑| 初始化是 `N (0,1)`，训练中不断更新|输入是索引 `idx`，就取权重矩阵第 `idx` 行|
    |数值关系|权重矩阵的第 `i` 行 = 索引 `i` 对应的词向量|输出词向量 ≡ 权重矩阵的某一行（无任何变换）|
- **正式推导如下**
1. Embedding层初始化时，权重矩阵的**每个元素**服从 $w \sim \mathcal{N}(0,1)$ （均值0，方差1）；
2. 输入任意索引 `idx` ，输出的词向量 $\boldsymbol{e} = [e_1, e_2, ..., e_H]$ ，其中 $e_i = w_{idx,i}$ （即权重矩阵第 `idx` 行第 `i` 列的元素）；
3. 因此，词向量的每个元素 $e_i \sim \mathcal{N}(0,1)$ ，且所有 $e_i$ 相互独立。
4. 输出词向量的 L2 范数计算（纯输出视角）：
   1. L2 范数的定义是： $$||\boldsymbol{e}||_2 = \sqrt{e_1^2 + e_2^2 + ... + e_H^2}$$
   2. 我们分步计算其数量级：
      1. **单个元素平方的期望**：因为 $e_i \sim \mathcal{N}(0,1)$ ，所以 $e_i^2$ 的期望 $E[e_i^2] = Var(e_i) + (E[e_i])^2 = 1 + 0 = 1$ ；
      2. **范数平方的期望**：范数平方是所有元素平方的和，因此：$E[||\boldsymbol{e}||_2^2] = E[e_1^2 + e_2^2 + ... + e_H^2] = H \times E[e_i^2] = H \times 1 = H$ ；
      3. **范数的期望（实际数量级）**：范数是范数平方的平方根，因此： $E[||\boldsymbol{e}||_2] = E\left[\sqrt{e_1^2 + ... + e_H^2}\right] \approx \sqrt{H}$ （工程上可直接用这个近似值）。
   3. **另一种视角：平方和 ⇒ $\chi^2$ 分布**

##### (2)、实验验证词向量的数量级
```python

加入位置编码信息前的词向量
nn.Embedding(200, 24, torch.ones(2, 100))[0][0]
tensor([-0.1895, -0.2969, -1.3700, -0.8913, -1.2275, -0.8055,  0.7443,  0.2928,
        -1.3917, -0.0078,  0.0601,  0.7912, -1.2219, -0.9332, -0.0526, -0.5193,
         0.6159,  0.3171, -0.7736, -2.0109, -0.6860, -1.3158, -0.4701, -0.5426],
       grad_fn=<SelectBackward0>)

加入位置编码信息前的词向量均值
nn.Embedding(200, 24, torch.ones(2, 100))[0][0].mean()
0.07729250192642212

加入位置编码信息前的词向量 L2 范数
nn.Embedding(200, 24, torch.ones(2, 100))[0][0].norm()
5.622831344604492

纯位置编码信息
attn.PositionalEncoding(24, 0.5).P[0][0]
tensor([0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
        0., 1., 0., 1., 0., 1.])

加入位置编码信息后的词向量
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0]
tensor([-0.4171, -0.9387, -1.2216,  0.5021, -1.8376,  1.3179, -0.7992, -0.9835,
         0.0476,  0.4639,  1.3233, -0.1767,  0.5068, -0.6837,  1.5620,  1.0962,
         0.0346,  0.2692, -0.5030,  1.8597, -1.7511,  0.3326,  0.0144,  0.7433],
       grad_fn=<SelectBackward0>)

加入位置编码信息后的词向量均值
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].mean()
0.03171844407916069

加入位置编码信息后的词向量的 L2 范数
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].norm()
4.8373122215271
```

<br>

#### 3、发现这个问题，很多人都没有给出很好的解释。我自己还是别钻牛角尖了！先把问题放这吧！
我也许并不能很好的回答你的问题,我找寻了很久也没有获得想要的答案,我思考了很久总结了一些观点,希望能和大家一起讨论讨论这个问题,以下是个人的一些看法.

我们知道，embedding 是一种缩放(缩小或放大)语义信息到一定维度空间的手段，在这里是输入一个 vocab_size 的 query,然后获得hidden_size维度的 embedding, embedding的好处是可以通过学习使得语义相近的样本在 embedding后距离更相近.

一种可能的解释是:当num_hiddens 增大时,嵌入向量的维度也会相应增大,这会增加模型学习到的单词或符号的语义信息.然而,(num_hiddens 较大时)此时我们想要加入位置编码,就有种有心杀贼无力回天的感觉,我们无法确保在茫茫数据海中(而且数据本身也是服从(0, 1)的分布),提取出位置信息,我认为如果不扩大位置编码提供的位置信息,将无法很好地区分不同位置处的单词或符号.因此,为了更好地结合语义信息和位置信息,一种可行的做法是增加位置编码在嵌入中的比例,但是我们知道的是模型的参数最好能控制在较小的范围(稳定性和收敛速度),不能无限制的根据 num_hiddens 放大这个占比,所以选用了平方根进行一定的缩放(你也可以试试log 缩放).

也许我的理解是错误的,希望如果有人能有正确的结论,请不吝赐教,我十分感谢.

