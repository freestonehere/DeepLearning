import math
import torch
from torch import nn
from d2l import torch as d2l
import tools.attention as attn
import tools.nlp as nlp
import tools.plot as plot
import matplotlib.pyplot as plt # 用于画图
import pandas as pd

'''基于位置的前馈网络（李沐老师说就是 MLP 还真是【单隐藏层的 MLP 】）'''
# 既然就是【单隐藏层的 MLP】，那为什么非要起个新名字呢？肯定有它的道理！
#@save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
print('\nffn(torch.ones((2, 3, 4)))[0]:',
      f'\n{ffn(torch.ones((2, 3, 4)))[0]}')


'''残差连接和层归一化'''
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# 在训练模式下计算 X 的均值和方差
print('\nlayer norm:', f'\n{ln(X)}\n', '\nbatch norm:', f'\n{bn(X)}')


# 现在可以使用残差连接和层规范化来实现 AddNorm 类。暂退法也被作为正则化方法使用。
# 残差连接没那么神秘，就是做了一个加法而已，别怵！
#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        # nn.LayerNorm([100, 24]) 是什么意思？

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# 残差连接要求两个输入的形状相同，以便加法操作后输出张量的形状相同
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
print('\nadd_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape',
      f'\n{add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape}')


'''编码器'''
#@save
class EncoderBlock(nn.Module):
    """Transformer 编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = attn.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

'''
张量形状 (batch_size, num_steps, embed_size)
⇒ 由此可以推出：一个样本其实就是【一行文本】，
每行文本有 num_steps 个 token ，每个 token 向量维数是 embed_size
'''

X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
print('\nencoder_blk(X, valid_lens).shape',
      f'\n{encoder_blk(X, valid_lens).shape}')


# 下面实现的 Transformer 编码器的代码中，堆叠了 num_layers 个 EncoderBlock 类的实例。
# 由于这里使用的是值范围在 -1 和 1 之间的固定位置编码，
# 因此通过学习得到的输入的嵌入表示的值需要先乘以嵌入维度的平方根进行重新缩放，然后再与位置编码相加

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


encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
print('\nencoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape', 
      f'\n{encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape}')

print('\n加入位置编码信息前的词向量\nnn.Embedding(200, 24, torch.ones(2, 100))[0][0]',
      f'\n{nn.Embedding(200, 24)(torch.ones(2, 100, dtype=torch.long))[0][0]}')

print('\n加入位置编码信息前的词向量均值\nnn.Embedding(200, 24, torch.ones(2, 100))[0][0].mean()',
      f'\n{nn.Embedding(200, 24)(torch.ones(2, 100, dtype=torch.long))[0][0].mean()}')

print('\n加入位置编码信息前的词向量 L2 范数\nnn.Embedding(200, 24, torch.ones(2, 100))[0][0].norm()',
      f'\n{nn.Embedding(200, 24)(torch.ones(2, 100, dtype=torch.long))[0][0].norm()}')

print('\n纯位置编码信息\nattn.PositionalEncoding(24, 0.5).P[0][0]',
      f'\n{attn.PositionalEncoding(24, 0.5).P[0][0]}')

print('\n加入位置编码信息后的词向量\nencoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0]',
      f'\n{encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0]}')

print('\n加入位置编码信息后的词向量均值\nencoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].mean()',
      f'\n{encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].mean()}')

print('\n加入位置编码信息后的词向量的 L2 范数\nencoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].norm()',
      f'\n{encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)[0][0].norm()}')


'''解码器'''
class DecoderBlock(nn.Module):
    """解码器中第 i 个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = attn.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = attn.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        '''state[2] 是解码器各块的状态缓存（预测阶段用）'''
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此 state[2][self.i] 初始化为 None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此 state[2][self.i] 包含着直到当前时间步第 i 个块解码的输出表示
        if state[2][self.i] is None:
            '''训练阶段 / 预测初始步： key_values = 当前输入 X'''
            key_values = X
        else:
            '''预测阶段后续步：拼接历史 dec_output 和当前输入（缓存上下文）'''
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        
        if self.training:
            '''训练阶段：生成上三角掩码（防止看到未来词元）'''
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens 的开头: (batch_size, num_steps),
            # 其中每一行是 [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
            '''最里面的维度复制 1 次；往外走一个维度，复制 batch_size 次！'''
        else:
            '''预测阶段：逐词解码，无需提前生成掩码（或后续处理）'''
            dec_valid_lens = None

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
        return self.addnorm3(Z, self.ffn(Z)), state

decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print('\ndecoder_blk(X, state)[0].shape',
      f'\n{decoder_blk(X, state)[0].shape}\n')

# 现在我们构建了由 num_layers 个 DecoderBlock 实例组成的完整的 Transformer 解码器。
# 最后，通过一个全连接层计算所有 vocab_size 个可能的输出词元的预测值。
# 解码器的自注意力权重和编码器解码器注意力权重都被存储下来，方便日后可视化的需要。
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

'''训练和预测'''
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = nlp.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = nlp.EncoderDecoder(encoder, decoder)
nlp.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 训练结束后，使用 Transformer 模型将一些英语句子翻译成法语，并且计算它们的 BLEU 分数。
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = nlp.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {nlp.bleu(translation, fra, k=2):.3f}')


'''可视化注意力权重'''
# 当进行最后一个英语到法语的句子翻译工作时，让我们可视化 Transformer 的注意力权重。
# 编码器自注意力权重的形状为 (编码器层数, 注意力头数, num_steps或查询的数目, num_steps或“键－值”对的数目)
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,
    -1, num_steps))
print('\nenc_attention_weights.shape',
      f'\n{enc_attention_weights.shape}')

plot.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))


dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
print('\ndec_self_attention_weights.shape', f'\n{dec_self_attention_weights.shape}\n', 
      '\ndec_inter_attention_weights.shape', f'\n{dec_inter_attention_weights.shape}')

# 由于解码器自注意力的自回归属性，查询不会对当前位置之后的“键－值”对进行注意力计算。
# Plusonetoincludethebeginning-of-sequencetoken
plot.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

plot.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

plt.show()