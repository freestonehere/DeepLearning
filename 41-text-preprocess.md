# Text Preprocess | 文本预处理
## 一、概念理解
- 把 **文本** 当作 **时序信息** 来处理

1. **关于【`vocab` 词表】的注意事项**
   1. **训练集** 和 **测试集** 用的 `vocab` 必须是**一致**的！
<br><br><br><br>

## 二、代码讲解
```powershell
# 下面是代码运行结果

# 文本总行数: 3221
lines[0]: the time machine by h g wells
lines[10]: twinkled and his usually pale face was flushed and animated the
tokens[0]: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
tokens[1]: []
tokens[2]: []
tokens[3]: []
tokens[4]: []
tokens[5]: ['i']
tokens[6]: []
tokens[7]: []
tokens[8]: ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']
tokens[9]: ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
tokens[10]: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']

[('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), ('to', 6), ('was', 7), ('in', 8), ('that', 9)]

文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
索引: [1, 19, 50, 40, 2183, 2184, 400]
文本: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
索引: [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]
(170580, 28)
```

### （一）、理解代码
```python
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
```

- 看来 **分词** 也是三件套
  1. 规范化读取
  2. 分词
  3. 构建 **词语（文本）** 和 **索引（数值）** 之间的映射

### （二）、Python API
#### 1、`class collections.Counter`
[官网 doc - class collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter)

```python
class collections.Counter([iterable-or-mapping])
```

1. A `Counter` is a `dict` subclass for counting hashable objects. 
   1. It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values. 
   2. Counts are allowed to be any integer value including zero or negative counts. 
   3. The `Counter` class is similar to bags or multisets in other languages
2. Elements are counted from an iterable or initialized from another mapping (or counter)
    ```python
    >>> c = Counter()                         # a new, empty counter
    >>> c = Counter('gallahad')               # a new counter from an iterable
    >>> c = Counter({'red': 4, 'blue': 2})    # a new counter from a mapping
    >>> c = Counter(cats=4, dogs=8)           # a new counter from keyword args
    ```
3. `Counter` objects have a **dictionary interface** except that they return a zero count for missing items instead of raising a `KeyError`
   1. 虽然 `Counter` 对象采用 **字典接口**
   2. 但是就算没有对应的索引值，`Counter` 对象也不会 raise `KeyError`
    ```python
    >>> c = Counter(['eggs', 'ham'])
    >>> c['bacon']                        # count of a missing element is zero
    0
    ```
    ```python
    >>> import collections
    >>> c = collections.Counter(['apple', 'apple', 'banana'])
    >>> c
    Counter({'apple': 2, 'banana': 1})
    ```

