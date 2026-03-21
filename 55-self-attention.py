import torch
from torch import nn
from d2l import torch as d2l
import tools.plot as plot
import matplotlib.pyplot as plt # 用于画图


num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print('\nattention(X, X, X, valid_lens).shape', 
      f'\n{attention(X, X, X, valid_lens).shape}\n')

# 位置编码
#@save
class PositionalEncoding(nn.Module):
    """位置编码
    
    位置编码既不是对数据做拼接，也不是改变模型结构；
    而是直接做数学相加，这种处理之前比较少见！"""
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


encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
plot.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])


# 绝对位置信息
for i in range(8):
    print(f'{i}的二进制是：{i:>03b}')

P = P[0, :, :].unsqueeze(0).unsqueeze(0)
plot.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')

plt.show()