import torch
from torch import nn
from d2l import torch as d2l
import tools.nlp as nlp
import matplotlib.pyplot as plt # 用于画图
import tools.plot as plot
import tools.attention as attn

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
        self.attention = attn.AdditiveAttention(
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
    

encoder = nlp.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
print('\noutput.shape:', f'\n{output.shape}\n',
      '\nlen(state):', f'\n{len(state)}\n',
      '\nstate[0].shape', f'\n{state[0].shape}\n',
      '\nlen(state[1]):', f'\n{len(state[1])}\n',
      '\nstate[1][0].shape:', f'\n{state[1][0].shape}\n')


# 训练
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = nlp.load_data_nmt(batch_size, num_steps)
encoder = nlp.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = nlp.EncoderDecoder(encoder, decoder)
nlp.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 模型训练后，我们用它将几个英语句子翻译成法语并计算它们的 BLEU 分数。
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = nlp.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
    
attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
    1, 1, -1, num_steps))

# 加上一个包含序列结束词元
plot.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')

plt.show()