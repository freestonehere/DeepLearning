import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
import tools.nlp as nlp
import matplotlib.pyplot as plt # 用于画图
from tools import Animator

'''老师说：代码虽然看着比较长，但是很多东西和 RNN 类似，采用这种迁移的方式看这些代码，
也许就很好理解！'''

# 编码器
#@save
class Seq2SeqEncoder(nlp.Encoder):
    """用于序列到序列学习的循环神经网络编码器。
    （注意： nn.GRU 和 nn.RNN 的输入 / 输出在张量形状上是完全一样的！）
    
    另外，嵌入层 Embedding 是封装在 encoder 和 decoder 里面的！
    所以 encoder 输入 X 是二维 (batch_size, num_steps)。
    
    encoder 输出的 output 和 state 就是 nn.GRU 直接输出的！"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        # droput 只能在训练时使用，在预测时绝对不能使用！

    def forward(self, X, *args):
        # embeddding 层输出 'X' 的形状：(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # nn.GRU 中，如果未提及状态，则默认为 0
        output, state = self.rnn(X)
        '''【明确】： output 和 h_n 都是隐藏状态，只不过它们隐藏状态所在位置不同！
        output 的形状:(num_steps, batch_size, num_hiddens);
        state 的形状:(num_layers, batch_size, num_hiddens).
        encoder 没有输出层 nn.Linear ，每层最后一个时间步的隐状态直接输入到 decoder 中，
        作为 decoder 的初始隐状态！'''

        return output, state
    
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
print(f'\noutput.shape: \n{output.shape}')
print(f'\nstate.shape: \n{state.shape}')
# output.shape: torch.Size([7, 4, 16])
# state.shape: torch.Size([2, 4, 16])


# 解码器
class Seq2SeqDecoder(nlp.Decoder):
    """用于序列到序列学习的循环神经网络解码器
    
    【和 encoder 相同】嵌入层 Embedding 是封装在 encoder 和 decoder 里面的！
    所以 decoder 输入 X 是二维 (batch_size, num_steps)。

    【和 encoder 不同】decoder ① 输出 output 不是 nn.GRU 直接输出的，而是经过了维度重排
    ② 但是输出 state 就是 nn.GRU 直接输出的！"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 【注意】：decoder 的 input_size 和 encoder 的 input_size 不一样！
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        '''decoder 的初始化隐状态是 encoder 输出的隐状态。
        但是由于 encoder 前向计算函数直接给出 output, state 元组，
        所以，这里要取 enc_outputs[1]!'''
        return enc_outputs[1]

    def forward(self, X, state):
        # embedding.permute 层输出 'X' 的形状：(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播 context，使其具有与 X 相同的 num_steps
        '''
        这里就是不一样的地方：普通的 GRU 就是分别接收 X 和 state ，然后就直接输出结果。
        但是这里把一部分 state 和原始 X 拼接起来，成为新的输入；然后用这个新的输入和 state 用于输出结果！

        state 是最后一个时间步上所有隐藏层的隐状态 (num_layers, batch_size, num_hiddens);
        state[-1] 是最后一个时间步上最后一个隐藏层的隐状态 (batch_size, num_hiddens);
        然后重复 num_steps 次，最终 context = (num_steps, batch_size, num_hiddens).
        
        context 和 X 拼接后，成为 (num_steps, batch_size, num_hiddens + embed_size)
        '''


        '''
        上面是 decoder 用于【训练】的场景。
        
        当 decoder 用于【预测（也就是机器翻译）】时，输入 dec_X = (batch_size, 1) = (1, 1)。
        然后，输入 dec_X 经过 Embedding.permute ，变成 dec_X = (1, batch_size, embed_size)。

        state 是最后一个时间步上所有隐藏层的隐状态 (num_layers, 1, num_hiddens);
        state[-1] 是最后一个时间步上最后一个隐藏层的隐状态 (1, num_hiddens);
        然后重复 dec_X.shape[0] = 1 次，最终 context = (1, 1, num_hiddens).
        
        context 和 dec_X 拼接后，成为 (1, 1, num_hiddens + embed_size)
        '''
        context = state[-1].repeat(X.shape[0], 1, 1)
        # cat 的逻辑是：保持其他维度不变，只有指定维度增长！
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        '''这里对 decoder output 的维度重新排序是有原因的，好像和 loss 有关？
        对的，就是和 loss 有关！
        不过这里不重新排序也可以，因为在输入 loss 前，还要对 decoder_output 重新排序！'''
        # output 的形状:(batch_size, num_steps, vocab_size)
        # state 的形状:(num_layers, batch_size, num_hiddens)
        return output, state
    
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
print(f'\noutput.shape, state.shape: \n{output.shape, state.shape}')


# 损失函数
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项。
    输入 X 是二维数值张量 (行数, num_steps)，其中 num_steps 是每行文本的固定长度；
    valid_len 是一维张量 (行数)，表示第 i 行文本的有效长度是 valid_len[i]。"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    '''上面这行代码 ① [None, :] 升维成 (1, maxlen)
    ② [:, None] 升维成 (行数, 1)
    ③ 然后通过广播 mask = (行数, maxlen)，每个元素的值为布尔值！
    若 X 中 i 行 j 列的字符是有效字符，那么 mask[i, j] = True ；否则 maxk[i, j] = False'''
    # 感觉这样的骚操作（一维通过广播成为二维）在【画锚框】的代码里见过（详见 33-CV-AnchorBox.py ）
    '''虽然 mask 中每个元素的值为布尔值，但是 X 中每个元素的值还是【数值】！
    另外，下面这行代码还用到了布尔索引！'''
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('\nsequence_mask(X, torch.tensor([1, 2])):',
      f'\n{sequence_mask(X, torch.tensor([1, 2]))}')

X = torch.ones(2, 3, 4)
print('\nsequence_mask(X, torch.tensor([1, 2]), value=-1):',
      f'\n{sequence_mask(X, torch.tensor([1, 2]), value=-1)}')

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的 softmax 交叉熵损失函数（其实就是加权，把无效字符的权重设为 0 ）"""
    # pred 的形状：(batch_size, num_steps, vocab_size)
    # label 的形状：(batch_size, num_steps)
    # valid_len 的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        '''由于 class Seq2SeqDecoder 提到：输出的 output 是经过维度重排的，
        所以 output 形状 (batch_size, num_steps, vocab_size)。
        由于这里的 pred 就是 decoder 的输出，所以 pred 形状 (batch_size, num_steps, vocab_size)。
        为了符合交叉熵损失函数的输入，所以 pred 重排成 (batch_size, vocab_size, num_steps)
        
        交叉熵损失函数的输入 (batch_size, vocab_size, num_steps); 输出 (batch_size, num_steps)。
        也就是说：交叉熵损失函数会使得维度退化！'''
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 原生 CrossEntropyLoss 要求输入形状为 (batch_size, num_classes, num_steps)（类别维度在第二个位置），
        # 而模型输出 pred 是 (batch_size, num_steps, vocab_size)，因此需要交换后两个维度
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
        # 由于 keepdim=False，因此 weight_loss 形状 (batch_size)
    
loss = MaskedSoftmaxCELoss()
print('\nloss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])):',
      f'\n{loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))}')
print()

# 训练
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                '''m._flat_weights_names: GRU 模块内部存储的所有权重 / 偏置参数的名称列表
                （比如 weight_ih_l0、weight_hh_l0 等，分别对应输入到隐藏层、隐藏到隐藏层的权重）；
                遍历参数名，筛选出名称包含 "weight" 的参数（排除偏置 bias ）'''
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            # data_iter 每次返回【batch_size 行文本】和【对应文本行的有效长度】
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 先把 dec_input 每行末尾阶段，然后在最前面拼接 <bos> ⇒ 得到 Y
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            '''注意：① 上面 enc_X = X; dec_X = dec_input; (enc_X 传入 encoder, dec_X 传入 decoder);
            ② loss 中使用的是 Y_valid_len ，不是 X_valid_len!'''
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行 “反向传播”
            nlp.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
    
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = nlp.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = nlp.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测（输入： 
    ① src_sentence 是一维 token 列表；
    ② src_vocab 和 tgt_vocab 是存储的【token-数值】映射关系）"""
    # 在预测时将 net 设置为评估模式（防止使用 dropout）
    net.eval()
    # 一维 token 列表变成 一维数值列表
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 将一维数值列表砍成定长
    src_tokens = nlp.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    '''
    添加批量轴 (num_steps) -> (1, num_steps);
    而且由于嵌入层 Embedding 封装在 encoder / decoder 里面，
    所以 encoder / decoder 的输入不需要做词嵌入！
    '''
    
    '''
    【疑问】：这里预测的时候为什么不直接用训练好的 net？反而一步一步地调用 encoder, decoder？
    【答】：训练时它接收的是完整的输入序列 + 完整的目标序列（用于 teacher forcing），
    但预测时我们需要逐词生成目标序列（无法提前知道完整的目标序列），
    因此必须拆解调用 encoder 和 decoder，而不能直接用 net 的前向传播！
    '''
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    
    # 添加批量轴
    '''训练时， dec_X 是一个序列 (batch_size, num_steps); 
    但是预测时不一样， dec_X 只是一个 <bos> (batch_size, 1)。

    也正是因为 dec_X 只有一个时间步，所以预测时 decoder 每次只返回一个 token!

    细节：预测时也要有批次维度，因为训练时 net 就要处理批次维度！
    这里 dec_X 张量形状 (1, 1)'''
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        '''由于训练时，每个时间步只生成一个词；所以预测时，每个时间步也只生成一个词！
        也就是：在这 num_steps 次循环中， dec_X 张量形状始终是 (batch_size, 1) = (1, 1)
        
        这里的 dec_X 张量是没有 vocab_size 的，因为 Embedding 是内置在 net.decoder 中的！'''

        Y, dec_state = net.decoder(dec_X, dec_state)
        # Y 张量形状 (batch_size, num_steps, vocab_size) = (1, 1, vocab_size)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        # item() 将张量转化为标量！
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# 对预测序列的评估
def bleu(pred_seq, label_seq, k):  #@save
    """计算 BLEU （输入 pred_seq 和 label_seq 都是一个字符串！）"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 计算 1~k 阶 n-gram 得分
    for n in range(1, k + 1):
        # label_subs 字典的 key 是 str 类型；value 是 int 类型！
        '''下面这种算法思想其实并不难，但是有点绕！很容易懵！'''
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            '''计算 label_tokens 中【所有 n-gram 语法】的出现次数'''
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            '''数 pred_tokens 中出现了多少次【label_tokens 中的 n-gram 语法】！'''
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

plt.show()