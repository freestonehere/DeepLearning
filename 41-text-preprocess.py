import collections
import re
import tools.cache_load_datasets as cache_load

# 读取数据集
#@save
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

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(f'lines[0]: {lines[0]}')
print(f'lines[10]: {lines[10]}')

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

tokens = tokenize(lines)
# 本代码中的 tokens 是按 word 拆分的二维列表
for i in range(11):
    print(f'tokens[{i}]: {tokens[i]}')
print()

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

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的 tokens 是 1D 列表 或 2D 列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表；列表元素就是 char / word；这里是 word
        # 这 2 个 for 要从左往右看！
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 打印前几个高频词元及其索引
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
print()

# 打印前 11 行文本行的内容
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

# 整合所有功能
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
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

corpus, vocab = load_corpus_time_machine()
print(f'{len(corpus), len(vocab)}')
# len(vocab) = 28 
# = 26 个英文字母 + 1 个 ' ' + 1 个 '<unk>'