import os
import torch
from d2l import torch as d2l
from tools import cache_load_datasets as cache_load
import matplotlib.pyplot as plt # 用于画图
import tools.nlp as nlp

# 下载和预处理数据集
#@save
cache_load.DATA_HUB['fra-eng'] = (cache_load.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集（返回一个包含换行符的长字符串）"""
    data_dir = cache_load.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75], end='\n\n')


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

text = preprocess_nmt(raw_text)
print(text[:80], end='\n\n')


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

source, target = tokenize_nmt(text)
print(source[:6], target[:6], end='\n\n')


#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target)

# 词表
src_vocab = nlp.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(f'len(src_vocab): {len(src_vocab)}')


# 加载数据集
#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列（使得每行文本为定长）。
    输入 line 是【一行文本】转化成的【一个数值列表】；
    num_steps 就是【每行文本的固定长度】！"""
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']),
      end='\n\n')


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
    src_vocab = nlp.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = nlp.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 将【二维 token 列表】转换为【二维数字列表】，同时返回每行文本的有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    '''data_iter 每次返回【batch_size 行文本】和【对应文本行的有效长度】'''
    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break

plt.show()