'''
import tools.attention as attn

对注意力机制相关内容单开一个文件。
'''

import torch
import torch.nn as nn
import tools.nlp as nlp


# 掩蔽 softmax 操作
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行 softmax 操作。
    
    输入：① X 是【注意力分数】(batch_size, 查询的个数, “键—值”对的个数);
    ② valid_lens (batch_size,) 或 (batch_size, 查询的个数);
    但是这里的 valid_lens 是谁的 valid_lens ？从后面的使用来看，
    这是【单条查询对应的注意力分数】的 valid_lens!
    更具体一点，其实是【单条查询对应的一系列 “键-值” 对的有效长度】
    
    ③ 与之对比，之前在机器翻译中， valid_lens 是【每行文本的有效长度】"""

    # X: 3D 张量 (batch_size, num_steps, embed_size)；
    # valid_lens: 1D (batch_size) 或 2D (batch_size, num_steps) 张量
    # 形状为 (batch_size, num_steps) 的 valid_lens 张量见【56-transformer.py】
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        '''将 valid_lens 统一处理为 1 维张量，
        长度匹配 X 展平前两个维度后的大小 (batch_size * 查询的个数),
        为逐行掩蔽做准备'''
        shape = X.shape
        if valid_lens.dim() == 1:
            # 不指定维度的 repeat_interleave() 那就一律展平
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其 softmax 输出为 0
        '''输入 sequence_mask 的张量是 (batch_size * 查询的个数, “键——值”对的个数)'''
        X = nlp.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# 加性注意力
#@save
class AdditiveAttention(nn.Module):
    """
    加性注意力

    输入张量的形状：
    ① queries (batch_size, 查询的个数, query_size)
    ② keys (batch_size, “键——值”对的个数, key_size)
    ③ values (batch_size, “键——值”对的个数, value_size)

    tanh(K @ W_k + Q @ W_q) 返回的 features 张量
    形状 (batch_size, 查询的个数, “键-值”对的个数, num_hiddens)

    w_v.(features) 返回的【注意力分数张量】attention_score 的形状：
    (batch_size, 查询的个数, “键-值”对的个数, 1)

    【注意力分数张量】归一化成为【权重张量】
    (batch_size, 查询的个数, “键-值”对的个数, 1)

    可能会有个【疑问】：为什么前面 (K @ W_k + Q @ W_q) 要把看似不相关的 K 和 Q 相加在一起？
    【答】就是为了把【查询的个数、“键-值”对的个数】融合在一起，
    进而让最后的 注意力分数 / 注意力权重 形状为 (batch_size, 查询的个数, “键-值”对的个数, 1)

    class AdditiveAttention 的前向函数最终返回 (batch_size，查询的个数，值的维度)
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        '''明确：全连接层 nn.Linear 其实就是矩阵乘法的抽象！'''
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        '''
        输入张量的形状：
        ① queries (batch_size, 查询的个数, query_size)
        ② keys (batch_size, “键——值”对的个数, key_size)
        ③ values (batch_size, “键——值”对的个数, value_size)
        '''
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries 的形状：(batch_size，查询的个数，1，num_hiddens)
        # key 的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        '''使用广播方式进行求和（ query 的【一个值】会和 key 的【所有值】分别相加），
        这样做完美符合设计初衷，这代码太妙了！'''
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v 仅有一个输出，因此从形状中移除最后那个维度。
        # scores 的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        '''注意力分数归一化 ⇒ 变成注意力权重！'''
        # values 的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

import math

# 缩放点积注意力
#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力（【单个 query】和【单个 key 】的向量长度相等，均为 d 的情况！）"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries 的形状：(batch_size，查询的个数，d)
    # keys 的形状：(batch_size，“键－值”对的个数，d)
    # values 的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens 的形状:(batch_size，) 或者 (batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        '''果然缩放点积注意力的实现就是简单： 1 个矩阵乘法 + 后处理就得到了【注意力权重】！'''
        d = queries.shape[-1]
        # 设置 transpose_b = True 为了交换 keys 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 定义注意力解码器
#@save
class AttentionDecoder(nlp.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
    
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs 的形状为 (batch_size，num_steps，num_hiddens).
        # hidden_state 的形状为 (num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs 的形状为 (batch_size, num_steps, num_hiddens).
        # hidden_state 的形状为 (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出 X 的形状为 (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # hidden_state[-1] 张量形状 (batch_size, num_hiddens)
            # query 的形状为 (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context 的形状为 (batch_size, 查询个数, ) = (batch_size, 1, num_hiddens)
            # x 形状为 (batch_size, embed_size)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            '''enc_outputs 既是 keys 又是 values!
            
            在生成 context 张量的时候，同时也计算出了注意力权重，权重张量形状为
            (batch_size, 查询的个数, “键-值”对的数目) = (batch_size, 1, num_steps)!'''
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 经过 torch.cat 后，张量形状变为 (batch_size, 1, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            '''
            张量形状：
            out (1, batch_size, num_hiddens)
            hidden_state (num_layers, batch_size, num_hiddens)

            out 张量形状和 50-seq2seq.py class Seq2SeqDecoder 中的 out 不同，
            因为 50-seq2seq.py 直接把所有时间步的 X 都传进来了！
            但是这里只传进来 1 个时间步的 X !

            不过这里为什么要这样做呢？是因为加了注意力机制的原因吗？
            '''
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs 的形状为
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

# 位置编码
#@save
class PositionalEncoding(nn.Module):
    """位置编码
    
    位置编码既不是对数据做拼接，也不是改变模型结构；
    而是直接做数学相加，这种处理之前比较少见！
    
    为每行文本每个 token 的每个向量维度做位置编码，最终 forward 函数输出
    (batch_size, num_steps 或 max_len, embed_size)"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        '''创建一个足够长的 P (batch_size, num_steps, dimension) = (1, max_len, num_hiddens)
        一行文本有 num_steps (max_len) 个 token ，每个 token 向量的维数是 num_hiddens'''
        self.P = torch.zeros((1, max_len, num_hiddens))
        # .reshape(-1, 1) 用到了广播机制！
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


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
        self.attention = DotProductAttention(dropout)
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

