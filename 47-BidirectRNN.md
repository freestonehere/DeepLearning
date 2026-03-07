# Bidirectional RNN | 双向循环神经网络
## 一、概念讲解
- 之前 [40-seq-model.md](40-seq-model.md) 中提到 **序列模型** 可以正向进行，也可以反向进行！
  - 于是，这里就开始介绍 **双向循环神经网络**

![双向循环神经网络图解](https://zh-v2.d2l.ai/_images/birnn.svg)

1. 把输入反向，放进去
2. 得到输出后，再把输出反向，得到反向隐藏层的 **最终输出**
3. 把 **正向** 和 **反向** 隐藏层的 **最终输出** 拼接起来，得到 **最终输出**

<br>

### （一）、双向循环神经网络
- 一个前向 RNN 隐藏层
- 一个反向 RNN 隐藏层
- 合并两个隐藏状态得到输出
  - 如果隐藏层 size 为 256，合并 (concat) 就是 512

1. $$\bf H^{(f)}_{t} = \phi( X_t W_{xh}^{(f)}+ H_{t-1}^{(f)} W_{hh}^{(f)}+ b_h^{(f)})$$
2. $$\bf H_{t}^{(b)} = \phi( X_t W_{xh}^{(b)}+ \vec H_{t-1} W_{hh}^{(b)}+ b_h^{(b)})$$
3. $$\bf H_t=\left[ H^{(f)}_{t}, H_t^{(b)}\right]$$
4. $$\bf O_t= H_t W_{hq}+ b_q$$

**推理**

- 可以基于句子做推理（在看到整个句子的情况下）

**总结**

- 双向循环神经网络通过反向更新的隐藏层来利用方向时间信息
- 通常用来对序列**抽取特征、填空**，而不是预测未来

<br>

### （二）、预测相关
```python
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
```

1. 双向 `RNN` 不是不适合做预测，是不适合做时间序列上的预测。
2. 在训练的时候，前向负责向后预测，反向负责向前预测，输入都是一整个序列，所以训练时候效果还好。但是就 d2l 的代码实现来看，predict 函数的输入是一个字符一个字符作为输入的，
   1. 用 rnn 预测还好，能保存前向信息。
   2. 但是用 bi-rnn 预测，前向的信息能保存，但是后向的状态信息每次都是 0（最后一个字符的初始状态默认是 0），就导致了网络一半参数的状态输入为 0，所以就能看到网络一直在输出前向的内容，即重复的字母。
3. **用到的时候再来学双向 `RNN` 吧，现在没有代码，分析起来总是模糊不清，现在只是有个概念就好了！**
4. **不过确实能大概明白：双向 `RNN` 不适合做未来的预测！因为做未来预测的时候，它只有 1 个方向的信息！而训练的时候，它兼具 2 个方向的信息。**

<br><br>

## 二、代码讲解
```powershell
time travellereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
time travellerererererererererererererererererererererererererer
困惑度 1.1, 110311.9 词元/秒 cuda:0
time travellerererererererererererererererererererererererererer
travellerererererererererererererererererererererererererer
```