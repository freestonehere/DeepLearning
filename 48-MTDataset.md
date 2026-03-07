# Machine Translation Dataset | 机器翻译数据集
## 一、概念讲解

<br><br>

## 二、代码讲解
```powershell
Go.     Va !
Hi.     Salut !
Run!    Cours !
Run!    Courez !
Who?    Qui ?
Wow!    Ça alors !


go .    va !
hi .    salut !
run !   cours !
run !   courez !
who ?   qui ?
wow !   ça alors !

[['go', '.'], ['hi', '.'], ['run', '!'], 
['run', '!'], ['who', '?'], ['wow', '!']] 
[['va', '!'], ['salut', '!'], ['cours', '!'], 
['courez', '!'], ['qui', '?'], ['ça', 'alors', '!']]

len(src_vocab): 10012
[47, 4, 1, 1, 1, 1, 1, 1, 1, 1]

X: tensor([[26, 32,  5,  3,  1,  1,  1,  1],
        [ 6, 50,  8,  4,  3,  1,  1,  1]], dtype=torch.int32)
X的有效长度: tensor([4, 5])
Y: tensor([[ 19, 111,   0,   5,   3,   1,   1,   1],
        [  6,  84,   0,   4,   3,   1,   1,   1]], dtype=torch.int32)
Y的有效长度: tensor([5, 5])
```
### （一）、理解代码
```python
'''构建机器翻译数据集的执行流程'''
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的【迭代器】和【词表】"""
    # 1. 读入，预处理
    text = preprocess_nmt(read_data_nmt())
    # 2. 分词（返回二维列表，列表内容是【文本 token】）
    source, target = tokenize_nmt(text, num_examples)
    # 3. 构建词表
    # 注意 nlp.Vocab 词表中：<unk> 是 0，自然 ⇒ <pad> 就是 1
    src_vocab = nlp.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = nlp.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 4. 将【二维 token 列表】转换为【二维数字列表】，同时返回每行文本的有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    '''data_iter 每次返回【batch_size 行文本】和【对应文本行的有效长度】'''
    return data_iter, src_vocab, tgt_vocab
```
1. 数据集三件套
   1. 读入原始数据
   2. ~~定义 `class Dataset` 数据集类~~
      1. 这里虽然没有显式定义数据集类 `class Dataset`
      2. 但是将原始数据处理成了【通过索引来直接使用】的数据！
      3. 具体过程就是
         1. 分词（得到【token 列表】）
         2. 构建 **词表**
         3. 通过 **词表** 将【token 列表】转化成【数字索引列表】
   3. 通过 `class DataLoader`，返回一个 `batch` 的内容

<br>

### （二）、Python 内置方法
#### 1、`str.split()` 方法
[官网 doc - str.split() 方法](https://docs.python.org/3/library/stdtypes.html#str.split)

```python
str.split(sep=None, maxsplit=-1)
```
- Return a `list` of the words in the string, using `sep` as the delimiter string

<br>

### （三）、`torch` 相关
#### 1、没错，`Python` 整型是可以和 `torch.tensor` 进行广播的！
```python
>>> a = torch.tensor(range(12), dtype=torch.float32)
>>> a = a.reshape(3, 4)
>>> a == 1
tensor([[False,  True, False, False],
        [False, False, False, False],
        [False, False, False, False]])
```

#### 2、`tensor.sum(dim=x, keepdim)` 和 `tensor.max(dim=x, keepdim)` 其实是相同的逻辑
- 它们都是：**保持其他维度不变，遍历指定维度 `dim = x`**！
- 另外，别忘了有 `keepdim` 参数！

```python
>>> a
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])

>>> a.sum(dim=1, keepdim=True)
tensor([[ 6.],
        [22.],
        [38.]])

>>> a.max(dim=1, keepdim=True)
torch.return_types.max(
values=tensor([[ 3.],
        [ 7.],
        [11.]]),
indices=tensor([[3],
        [3],
        [3]]))
```