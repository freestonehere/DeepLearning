import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import tools.nlp as nlp
import matplotlib.pyplot as plt # 用于画图
from tools import Animator

batch_size, num_steps = 32, 35
train_iter, vocab = nlp.load_data_time_machine(batch_size, num_steps)

# 独热编码
print('\nF.one_hot(torch.tensor([0, 2]), len(vocab)):', 
      f'\n{F.one_hot(torch.tensor([0, 2]), len(vocab))}')

# 明确：独热编码实际上是对【所有数据】重编码
# 因此，【独热编码】中的维度变化就是【在原来维度的最后面，添加一个 len(独热编码的码长)】
# 维度变化的例子：(2) ⇒ (2, 28); (5, 2) ⇒ (5, 2, 28)

X = torch.arange(10).reshape((2, 5))
print('\nF.one_hot(X.T, 28).shape:', f'\n{F.one_hot(X.T, 28).shape}')


# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    '''初始化各个网络层（隐藏层、输出层）的参数，并将所有层的参数返回！'''
    # 输入通道数 = 输出通道数 = vocab_size = 28
    # 输出通道数是 vocab_size，因为【预测序列】实际上就是【分类问题】（预测目标到底是 vocab 中的哪一类）
    # 输入通道数是 vocab_size，可能是因为独热编码导致的？？不是太明白
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 循环神经网络模型
def init_rnn_state(batch_size, num_hiddens, device):
    '''返回 RNN 初始化隐藏层张量 (batch_size, num_hiddens)'''

    '''在 RNN 获取第一个数据的时候，是【没有前一个隐藏层】的；
    但是 RNN 的输出却需要依赖【前一个隐藏层】和【本层输入】；
    所以，这个函数就是用于解决这个问题（初始化 RNN 的隐藏层）！'''
    # 在后面的章节中我们将会遇到隐状态包含多个变量的情况，而使用元组可以更容易地处理些
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    '''RNN 前向计算函数'''
    # inputs 的形状：(时间步数量，批量大小，词表大小)
    '''还记得 42-language-model.py 中，采样函数 seq_data_iter_random 和 seq_data_iter_sequential 采的
    【样本】是【子序列】，而【子序列】的长度就是【时间步数量】，
    这两个采样函数返回的张量形状都是 (batch_size, num_steps)。
    将这【采样函数】返回的矩阵转置后进行独热编码，就得到 (num_steps, batch_size, 词表大小)!'''

    '''等一下，如果词表大小 = 28 的话，那就意味着这里是基于 char 的分词，
    而不是基于 word 的分词！真奇怪，基于 char 居然也能训练出不错的效果！'''

    '''还有一个【疑问】：【样本子序列】中的 token 不是已经有数值了吗，为什么还要进行独热编码呢？
    【答】：这是为了 predict_ch8 函数中便于通过 argmax 直接获得【最有可能的字符的数值索引】！'''
    # state 是为 RNN 初始化的隐藏层
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X 的形状：(批量大小，词表大小)。也就是说：X 的所有子序列都是在【同一个时间步上】的
    for X in inputs:
        '''更新隐藏层需要用非线性激活函数；获取输出值则不需要非线性激活函数！'''
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
        # Y 的张量形状是 (batch_size, num_outputs) = (batch_size, 词表大小)
        '''在转置以后，整个前向计算过程中， inputs 张量形状始终都是 (num_steps, batch_size, 词表大小)。
        只有在输出前向计算结果的时候，才把所有结果给拼接起来！成为 (num_steps * batch_size, 词表大小)'''
    return torch.cat(outputs, dim=0), (H,)
    # 返回的两个张量是：基于每个【子序列】的每个【token】的预测输出, 最终的隐藏状态
    # 张量形状分别是 (num_steps * batch_size, 词表大小), (batch_size, num_hiddens)

'''关于 num_steps 含义的说明：
seq_data_iter_sequential 会吐出 (batch_size, num_steps) 张量，
每个子序列的长度就是 num_steps!
也就是说： 序列模型中的【1 个样本】就是【1 个子序列】！

另外， num_steps 不可以理解成隐藏层层数（因为你实际上只有一个隐藏层层数，
只不过在 1 个 batch 内，隐藏层更新 num_steps 次）
'''

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


num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print('\nY.shape, len(new_state), new_state[0].shape:',
      f'\n{Y.shape, len(new_state), new_state[0].shape}')

'''【注意】：从一开始，隐藏层就是 (H,)；又因为这里的 new_state 是最终的隐藏层，
所以 len(new_state) = 1'''


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

print("\npredict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()):",
      f"\n{predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())}",
      end='\n\n')


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


num_epochs, lr = 500, 1
# 采用顺序采样方法训练
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
print()

# 采用随机抽样方法训练
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)

plt.show()