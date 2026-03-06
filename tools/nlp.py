import re
import tools.cache_load_datasets as cache_load
from d2l import torch as d2l
import collections
import torch
import random
from torch import nn as nn
from tools import Animator
import math
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
                # state 对于 nn.GRU 是个张量
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
        
