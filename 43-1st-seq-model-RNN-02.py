import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import tools.nlp as nlp
import matplotlib.pyplot as plt # 用于画图

batch_size, num_steps = 32, 35
train_iter, vocab = nlp.load_data_time_machine(batch_size, num_steps)

# 定义模型
num_hiddens = 256
# PyTorch 框架中的 nn.RNN 没有输出层！
rnn_layer = nn.RNN(len(vocab), num_hiddens)
# (输入/输出通道数, 隐变量长度)

state = torch.zeros((1, batch_size, num_hiddens))
print('state.shape:',f'\n{state.shape}')
# 输出 torch.Size([1, 32, 256])

X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(f'\nY.shape, state_new.shape:\n{Y.shape, state_new.shape}')
# 输出 (  torch.Size([35, 32, 256]), torch.Size([1, 32, 256])  )

# 注意，nn.RNN 只包含隐藏的循环层，我们还需要创建一个单独的输出层。
#@save
class RNNModel(nn.Module):
    """循环神经网络模型"""
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
        

# 训练与预测
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
nlp.predict_ch8('time traveller', 10, net, vocab, device)

num_epochs, lr = 500, 1
nlp.train_ch8(net, train_iter, vocab, lr, num_epochs, device)

plt.show()