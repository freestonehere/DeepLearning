import re
import tools.cache_load_datasets as cache_load
from torch.nn import functional as F

cache_load.DATA_HUB['time_machine'] = (cache_load.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

# 这里没有数据集三件套，只有 read，可能是因为还没有用到 train_iter？
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    # open(..., 'r')：以只读文本模式打开文件。
    # with 语句：保证文件使用后自动关闭。
    with open(cache_load.download('time_machine'), 'r') as f:
        lines = f.readlines()
        # f.readlines() 返回一个列表，每个元素是文件中的一行（包含换行符 '\n'）
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    # [^...] 表示匹配不在该字符类中的任意字符
    # 将所有非字母字符替换为一个空格 → 去除首尾空白 → 转小写


# 词元化
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
        # 返回一个二维列表，每一行内的元素以 word 为单位
    elif token == 'char':
        return [list(line) for line in lines]
        # 返回一个二维列表，每一行内的元素以 char 为单位
    else:
        print('错误：未知词元类型：' + token)
        # 这里的 + 应该是指字符串拼接。Python 语法真的挺灵活的！

import collections

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的 tokens 是 1D 列表 或 2D 列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表；列表元素就是 char / word；这里是 word
        # 这 2 个 for 要从左往右看！
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 词表
class Vocab:  #@save
    """文本词表（将文本转化为数值）。
    注意：输入的 tokens 是分词后的一维列表！"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率从大到小排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元 <unk> 的索引为 0，<unk> 指 unknown
        # idx_to_token 是列表，token_to_idx 是字典
        # 刚开始，把 reserved_tokens 分别添加到 idx_to_token 和 token_to_idx
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs: # _token_freqs 仍然是字典
            if freq < min_freq: # 出现频率低于设定频率时，直接丢掉
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        '''__getitem__ 一般是代码自动自动调用，所以往往需要返回数值，
        所以才从 token_to_idx 中查找，而不是 idx_to_token！
        不知道这样理解对不对？
        
        上面的理解应该不对，因为下面有调用代码 list(vocab.token_to_idx.item())[:10]'''

        '''1. 如果 tokens 不是列表/元组，则视为单个词元，从 token_to_idx 中查找。
        若找不到，返回 self.unk （即 0 ，代表 <unk>）。
        2. 列表输入：递归地对每个元素调用 __getitem__ ，返回索引列表。
        这样设计使得 vocab[token] 和 vocab[token_list] 都能工作'''

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        '''与 __getitem__ 互为逆操作！'''
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property # @property 表示属性！
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

# 整合所有功能
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的【语料库（将读入的文本一一转化为数值，相同文本 / 数值可以重复出现多次）】
    和【词表（文本-数值字典，相同文本 / 数值只出现一次）】"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
    # 返回索引列表和词表对象

import torch
import random

# 随机采样
'''为什么不采用 40-seq-model.py 中的采样方式？那里其实没听懂！'''
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个【小批量子序列】，一共分为 2 步：
    ① 从语料库中，生成【随机的子序列】（但是子序列内部是连续的！），
    用 initial_indices 来存储【随机子序列】的开头索引；
    ② 从【随机的子序列】中，选 batch_size 个子序列，生成 1 个 batch。
    其中每个子序列长度为 num_steps
    【总结】： 随机操作都是针对【子序列】的，不是针对 batch 的！
    【注意】： seq_data_iter_random 和 sequential 的样本都是【子序列】"""
    # 随机初始化语料库开头，随机范围包括 num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去 1，是因为我们需要考虑标签
    # 这个 1 其实是为最后一个子序列留下来的标签。因为 X = [data(j)] and Y = [data(j+1)]
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为 num_steps 的子序列的起始索引
    # num_subseqs * num_steps = 一共有多少子序列 * 一个子序列内的词语点数量
    # initial_indices 得到每个子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从 pos 位置开始的长度为 num_steps 的序列
        return corpus[pos: pos + num_steps]

    # 下面这行代码 ⇒ 一个【样本】是指一个【子序列】
    # 一个 batch 内有 batch_size 个子序列
    num_batches = num_subseqs // batch_size
    '''将所有数据分批输出，每次输出 1 个 batch'''
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices 包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
        # X, Y 张量形状都是 (batch_size, num_steps)


# 顺序分区
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    '''① seq_data_iter_sequential 和 seq_data_iter_random 【一样】，
    都需要随机初始化语料库开头。offset 就是用于随机初始化语料库开头的！
    ② 二者【不同】的地方在于： seq_data_iter_sequential 不需要【自己再生成顺序子序列】，
    直接用切片从 Xs, Ys 中取【顺序子序列】就行，而 seq_data_iter_random 则需要【自己再生成随机子序列】'''
    # 语料库开头选谁不重要 seq_data_iter_random 中是 random.randint(0, num_steps-1)
    # 而 seq_data_iter_sequential 这里是 random.randint(0, num_steps)
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # 注意：Xs 张量形状依然是 (num_tokens)，Ys 张量形状依然是 (num_tokens)
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    '''将所有数据分批输出，每次输出 1 个 batch'''
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


# 现在，我们将上面的两个采样函数 (seq_data_iter_random & seq_data_iter_sequential) 包装到一个类中， 
# 以便稍后可以将其用作数据迭代器。
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# 定义了所有需要的函数之后，接下来我们创建一个类来包装这些函数，并存储从零开始实现的循环神经网络模型的参数。
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    # 可以定义 forward() 函数，也可以定义 __call__() 函数
    def __call__(self, X, state):
        '''独热编码的位置：① 在数据预处理阶段 (train_iter 迭代器中)，是没有进行独热编码的；
        ② 而是在网络中，先对 train_iter 迭代器出来的数据 (batch_size, num_steps)
        做独热编码 (num_steps * batch_size, 词表大小)，
        然后再进行前向计算'''
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# 预测
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在 prefix 后面生成新字符。
    prefix 是一个字符串； outputs 是一个列表，而且列表元素是数值，不是文本。"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    '''实锤了， outputs = [ vocab[prefix[0]] ] 这行代码证实：
    这里的预测是基于 char 的预测，而不是基于 word 的预测！'''
    # get_input() 函数获取最新的预测
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 上面的循环获取遍历 prefix 序列后的隐藏态
    # 下面的循环才正式开始预测！
    for _ in range(num_preds):  # 预测 num_preds 步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    '''这里并没有进行 softmax 操作（ softmax 操作是内置在损失函数中的，详见 train_ch8 中的交叉熵损失函数）。
    下面梳理一下【预测】的逻辑：①【前向计算】中通过矩阵乘法获得输出 (num_steps * batch_size, 词表大小)；
    ② 然后通过 argmax 获得最有可能的 char 的数值索引'''
    return ''.join([vocab.idx_to_token[i] for i in outputs])

from torch import nn as nn

# 梯度剪裁
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    '''for p in params 将所有梯度拉成一个向量；然后计算该向量的范数 norm。
    该范数其实是向量的 L2-范数（欧几里得长度）

    >>> torch.sqrt(sum(   torch.sum((p.grad ** 2)) for p in params  ))
    for p in params 中， p 各自都是张量，因此：
    ① 先对张量平方求和，同时使得张量退化为标量；② 再对标量求和！'''

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


from d2l import torch as d2l
import math

# 训练
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（ 定义见第 8 章 ）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        '''【张量形状】X, Y 都是 (batch_size, num_steps)'''
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化 state（隐藏层状态）
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            '''关于 GRU 和 LSTM 的代码以后再看！
            反正 RNN 本身是不需要 detach_() 的，因为 RNN 一直在更新隐藏层的参数！'''
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state 对于 nn.GRU 和 nn.RNN 是个张量
                state.detach_()
            else:
                # state 对于 nn.LSTM 或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()

        '''独热编码的位置：① 在数据预处理阶段 (train_iter 迭代器中)，是没有进行独热编码的；
        ② 而是在网络中，先对 train_iter 迭代器出来的数据 (batch_size, num_steps)
        做独热编码 (num_steps * batch_size, 词表大小)，
        然后再进行前向计算'''
        y = Y.T.reshape(-1)
        # Y.T 转置是二维张量 (num_steps, batch_size)
        # y 是一维张量，形状 (num_steps * batch_size)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        # 根据 rnn 前向计算函数得知：
        # y_hat (num_steps * batch_size, 词表大小) state (batch_size, num_hiddens)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            # 为什么这里不先把梯度清零呢？
            # 原来 updater(batch_size=1) 在上一个 epoch 结尾已经把梯度清零了！
            l.backward()
            grad_clipping(net, 1)
            '''① 因为已经调用了 mean 函数，所以这里 batch_size = 1 即可；
            否则， batch_size 就要取真正的 batch_size！
            ② 而且，这里就算前面 loss 中的 redeuction='sum'， 
            这里还是取 batch_size = 1 （还是因为 loss 后面的 .mean() 函数！）'''
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

from tools import Animator

#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（ 定义见第 8 章 ）"""
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            # print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
        if (epoch + 1) % 50 == 0:
            print(predict('time traveller'))
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


# 注意，nn.RNN 只包含隐藏的循环层，我们还需要创建一个单独的输出层。
#@save
class RNNModel(nn.Module):
    """循环神经网络模型（
    ① 【从零实现】用上面的 class RNNModelScratch
    ② 【简洁实现】用这个 class RNNModel）"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果 RNN 是双向的（之后将介绍），num_directions 应该是 2，否则应该是 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        '''① inputs 张量形状是 (num_steps, batch_size, 词表长度)；
        ② 最开始的时候，初始化隐藏层状态 state (1, num_steps * batch_size, num_hiddens)；
        而且，【从零实现】和【简洁实现】一样，隐藏层张量形状在整个更新过程中保持不变！'''
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        '''全连接层首先将 Y 的形状改为 (时间步数 * 批量大小, 隐藏单元数)
        它的输出形状是 (时间步数 * 批量大小, 词表大小)。'''
        # RNN 中的输出不需要激活函数！
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU 以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM 以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

import os

cache_load.DATA_HUB['fra-eng'] = (cache_load.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集（返回一个包含换行符的长字符串）"""
    data_dir = cache_load.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()


#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集（依然返回一个长字符串）"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    # 全角空格，半角空格，用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格（便于把标点符号也做成一个 token，用于翻译）
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

# 词元化
#@save
def tokenize_nmt(text, num_examples=None):
    """词元化 “英语－法语” 数据数据集（返回二维 token 列表 (行数, num_steps) ）
    （将一个长字符串切割为文本 token ，分别返回英语列表和法语列表）。"""
    source, target = [], []
    # 原来文本 text 本身是不换行的！
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            # 注意 .split(' ') 方法返回的是一个 list！
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# 加载数据集
#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列（使得每行文本为定长）。
    输入 line 是【一行文本】转化成的【一个数值列表】；
    num_steps 就是【每行文本的固定长度】！"""
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充


#@save
def build_array_nmt(lines, vocab, num_steps):
    """将【二维 token 列表】转换为【二维数字列表】，同时返回每行文本的有效长度。

    输入 lines 是 list of list of token （二维列表）。
    返回的 array 是二维张量 (行数, num_steps)，
    其中 num_steps 是每行文本的固定长度"""

    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]

    '''lines 经过预处理后，成为 list of list of number_index （二维列表），
    并且每行末尾还有一个 <end of sentence>'''
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # for l in lines ⇒ 意味着 l 是【一行文本】转化成的【一个数值列表】
    '''array 是二维张量 (行数, num_steps)。
    【行数】就是读入的原始字符串的行数；
    【num_steps】是指每行文本固定长度为 num_steps.'''
    # 没错，Python 整型也可以和 torch.tensor 进行广播！
    # 按行求和，得到每行文本的 valid_len！
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# 训练模型
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的【迭代器】和【词表】"""
    # 读入，预处理
    text = preprocess_nmt(read_data_nmt())
    # 分词（返回二维列表，列表内容是【文本 token】）
    source, target = tokenize_nmt(text, num_examples)
    # 构建词表
    # 注意 nlp.Vocab 词表中：<unk> 是 0，自然 ⇒ <pad> 就是 1
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 将【二维 token 列表】转换为【二维数字列表】，同时返回每行文本的有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    '''data_iter 每次返回【batch_size 行文本】和【对应文本行的有效长度】'''
    return data_iter, src_vocab, tgt_vocab


'''编码器接口'''
#@save
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


'''解码器接口'''
#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


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


# seq2seq 的编码器
#@save
class Seq2SeqEncoder(Encoder):
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
    

# seq2seq 解码器
class Seq2SeqDecoder(Decoder):
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
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')



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
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
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