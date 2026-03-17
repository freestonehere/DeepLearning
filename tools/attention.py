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

    # X: 3D 张量，valid_lens: 1D 或 2D 张量
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