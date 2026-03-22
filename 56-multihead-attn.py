import torch
from torch import nn
import tools.attention as attn

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
    """逆转 transpose_qkv 函数的操作"""
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


num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()


batch_size, num_queries = 2, 4
num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

print('\nattention(X, Y, Y, valid_lens).shape',
      f'\n{attention(X, Y, Y, valid_lens).shape}')