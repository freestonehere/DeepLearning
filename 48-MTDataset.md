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