import random
import torch
from d2l import torch as d2l
import tools.plot as plot
import matplotlib.pyplot as plt # 用于画图
import tools.ch13 as ch13
import tools.nlp as nlp


# 自然语言统计
tokens = d2l.tokenize(nlp.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print('一元语法（普通语法）', '\n',
    f'vocab.token_freqs[:10]:\n{vocab.token_freqs[:10]}', '\n')


freqs = [freq for token, freq in vocab.token_freqs]
plot.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
fig = plt.gcf()
ch13.set_title(fig, '第一步： token-frequency 图')


bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print('二元语法', '\n',
    f'bigram_vocab.token_freqs[:10]:\n{bigram_vocab.token_freqs[:10]}', '\n')

trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print('三元语法', '\n',
    f'trigram_vocab.token_freqs[:10]:\n{trigram_vocab.token_freqs[:10]}', '\n')

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

plt.figure()
plot.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
fig = plt.gcf()
ch13.set_title(fig, '第二步： token-frequency 图（一元、二元和三元语法）')


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

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


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


# 现在，我们将上面的两个采样函数包装到一个类中， 以便稍后可以将其用作数据迭代器。
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

plt.show()