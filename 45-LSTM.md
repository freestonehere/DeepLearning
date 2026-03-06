# LSTM | 长短期记忆网络
## 一、概念理解
- 其实一直没搞明白过 `LSTM` 为什么这样设计，只是一直在用（因为好用！）
- 另外，虽然看上去 `GRU` 比 `LSTM` 简单，但是 `GRU` 是在 `LSTM` 之后提出的！

### （一）、`LSTM` 介绍
- 忘记门 $\bf F_t$ ：将值朝 0 减少
- 输入门 $\bf I_t$ ：决定不是忽略掉输入数据
- 输出门 $\bf O_t$ ：决定是不是使用隐状态

#### 1、门
- 这三个门的计算方式和 `RNN` 中计算隐变量的方式**完全一样**！
    -  $${\bf I_t=\sigma( X_t W_{xi}+ H_{t-1} W_{hi}+ b_i)} ~ ~ \text{input，输入门}$$
    -  $${\bf F}_t=\sigma({\bf X}_t{\bf W}_{xf}+{\bf H}_{t-1}{\bf W}_{hf}+{\bf b}_f) ~ \text{forget，遗忘门}$$
    -  $${\bf O}_t=\sigma({\bf X}_t{\bf W}_{xo}+{\bf H}_{t-1}{\bf W}_{ho}+{\bf b}_o) ~ ~ \text{output，输出门}$$
<br>

#### 2、候选记忆单元
$${\bf \~C}_t=tanh({\bf X}_t{\bf W}_{xc}+{\bf H}_{t-1}{\bf W}_{hc}+{\bf b}_c)$$

- RNN 的输出（和 `RNN` 中计算隐藏层的方式 **完全一样**）
- ${\bf C, \~C}$ 与 $\bf H$ 形状相同
  - 因为 `LSTM` 中的 $\bf C, \~C$ 的计算方式就 是 和 `RNN` 中 $\bf H$ 的计算方式相同！

##### (1)、如何分析 $\bf C_{t-1}$ 和 $\bf C_{t}$ 的张量形状？
1. 从 **候选记忆单元** 这个名字可以看出来， $\bf C_t$ 和 $\bf C_t$ 的形状应该是一样的！
   1. 又因为 `LSTM` 中 $\bf C_t$ 的计算方式和 `RNN` 中 $\bf H_t$ 的计算方式 **完全相同**
   2. ~~所以可以推知 $\bf C_t$ 张量形状：~~
      1. ~~`(num_steps * batch_size, num_hiddens)`（从零实现）~~
      2. ~~**或** `(num_steps, batch_size, num_hiddens)`（简洁实现）~~
<br>

##### (2)、传入实参的时候， $\bf C_{t-1}$ 应该传入什么？
1. 看看从零实现的代码中是如何传参的吧！
   1. 一个猜测：不会是和 `RNN` 从零实现一样，采用 **随机初始化 `记忆单元`** 吧？（因为 `RNN` 从零实现就是 **随机初始化 `隐变量 state`**）
2. 从下面代码分析可知：果然是随机初始化 **`记忆单元`**，而且是和 **`隐藏态`** 一起随机初始化的！
   1. 但是 `记忆单元` 的张量形状和前面分析的不太一样，前面分析的 `记忆单元 = (num_steps * batch_size, num_hiddens)` 是不对的！
   2. **必须明确**：**`记忆单元`** 和 **`隐藏态`** 性质是一样的，因此
      1. 从 **张量形状** 上来讲：`记忆单元 = (batch_size, num_hiddens)`
      2. 从 **张量用法** 上来讲：每一层的神经网络，只保留**最后一个时间步**的 `记忆单元` 和 `隐藏态`；但是，每一层的神经网络，保留**全部时间步**的 `最终输出`！
   3. 别问【为什么 **记忆单元** 和 **隐藏态** 性质是一样的？】
      1. 论文就是这么设计的！没有为什么！
      2. 想问为什么的话，那就去问论文作者！

```python
# 这是【从零实现】的代码片段！
def init_lstm_state(batch_size, num_hiddens, device):
    '''这里返回 state = (H, C) = (隐藏态, 记忆单元)'''
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

<br>

#### 3、记忆单元
$${\bf C}_t={\bf F}_t⊙{\bf C}_{t-1}+{\bf I}_t⊙{\bf\~C}_t$$

- $F=0$，舍去上一个记忆单元
- $I=0$，舍弃当下的记忆单元
- 和 `GRU` 中 **隐状态** 的计算方式不同 $\bf H_t = Z_t ⊙ X_{t-1} + (1-Z_t) \~H_t$
  - `GRU` 中的两个权重 $\bf Z_t$ 和 $\bf 1 - Z_t$ ：如果一个增大，那么另一个权重就必然减小！
  - 但是 `LSTM` 不同，`LSTM` 中的两个权重 $\bf F_t$ 和 $\bf I_t$ 独立，二者可以同时增大，也可以同时减小！
- 两者相加， $\bf C_{t-1}$ 和 $\bf\~C_t$ 都经过非线性变换，在 $0-1$ 之间，所以 $\bf C_{t}$ 在 $0-2$ 之间
  - 仔细想来， $\bf C$ 没有经过非线性，可以叠加出比较大的值 
<br>

#### 4、隐状态
$${\bf H}_t={\bf O}_t⊙tanh({\bf C}_t)$$
- 【续：前面 **记忆单元** 中的论述】所以要再次非线性把 ${\bf C}_t$ 剪切到 `0-1` 区间。
- ${\bf O}_t=0$，把输入和前一刻状态都舍弃，相当于重置 ${\bf H}_t$
- ${\bf O}$ 是对 ${\bf H}$ 的保存与否
- 相当于 ${\bf C}$ 是对 ${\bf H}$ 状态的中间量
<br>

#### 5、输出

$${\bf O}_t={\bf W}_{ho}{\bf H}_t+{\bf b}_o$$

- 最后输出 $\bf O_t$ 时没有用到 非线性激活函数 吧！
  - 详见 [43-1st-seq-model-RNN-01.py](43-1st-seq-model-RNN-01.py) 代码中 `rnn` 前向计算函数（直接写明了从 隐变量 到 输出 就是矩阵乘法，没有非线性激活函数）
  - **或** [43-1st-seq-model-RNN-02.py](43-1st-seq-model-RNN-02.py) 中 `class RNNModel` 中的 `forward` 前向计算函数（在 `nn.RNN` 后面增加了全连接层 `nn.Linear`）
  - **直观理解**：还是和 [43-1st-seq-model-RNN.md](43-1st-seq-model-RNN.md) 一样，因为 $\bf O_t$ 不需要再次进入神经网络，因此不需要 **非线性激活函数**
    - 如果 $\bf O_t$ 需要再次进入神经网络，那么必须添加非线性激活函数，否则就会产生 **网络坍塌**！

<br>

### （二）、`RNN`, `GRU`, `LSTM` 三者区别的汇总
|网络|图解|
|:--|:---|
|`RNN`|![循环神经网络 (RNN) 的概念](https://zh-v2.d2l.ai/_images/rnn.svg)|
|`GRU`|![更新真正的隐变量](https://zh-v2.d2l.ai/_images/gru-3.svg)|
|`LSTM`|![LSTM 图解](https://zh-v2.d2l.ai/_images/lstm-3.svg)|

#### 0、相似点总结
1. `GRU` 和 `LSTM` 都是
   1. 先通过 **直接的矩阵乘法** 产生 **中间变量**
   2. 然后通过 **中间变量的加权求和**，才得到 **可以输出的变量**！ 

<br>

#### 1、`RNN`

<br>

#### 2、`GRU`
1. $\bf H_{t-1}$ 和 $\bf X_t$ 共同作用
   1. `RNN` 得到 $\bf H_t$ （该时间步的隐藏状态）
   2. `GRU` 得到 $\bf R_t$ 和 $\bf Z_t$ （该时间步的重置门和更新门）
   3. 由于 `GRU` 产生 $\bf R_t, Z_t$ 和 `RNN` 产生 $\bf H_t$ 的运算方式 **完全一样**
      1. 因此， $\bf R_t, Z_t$ 张量形状：
         1. `(num_steps * batch_size, num_hiddens)`（从零实现） 
         2. **或** `(num_steps, batch_size, num_hiddens)`（简洁实现）
      2. 另外，从 **候选隐藏状态** 这个名字也可以看出来： $\bf\~H_t$ 和 $\bf H_t$ 形状应该一致！都是  
         1. `(num_steps * batch_size, num_hiddens)`（从零实现） 
         2. **或** `(num_steps, batch_size, num_hiddens)`（简洁实现）
2. `GRU` 新增的内容
   1. $\bf R_t \text{（重置门）} ~ ~ H_{t-1} ~ \text{和} ~ X_t$ 共同作用，得到 $\bf \~H_t$ （候选隐藏状态）
   2. 由 $\bf Z_t \text{（更新门）}$ 选择 $\bf H_{t-1}$ 和 $\bf \~H_t$ ，得到最终的输出 $\bf H_t$
- $\bf Z_t = 1$ 时，直接用前一时刻状态，舍弃当前状态
- $\bf Z_t = 0$ 时，不起作用；如果配合 $\bf R_t = 1$，就是 RNN。
  - 当 **更新门** $\bf Z_t= 0$，**遗忘门** $\bf R_t= 1$ 时，还真就是 `RNN`！ $\bf H_t = \tanh(X_{t-1} W_{xh} + H_{t-1} W_{hh} + b_h)$
- **注意**：
  - 由于 $\bf R_t, Z_t$ 都是从 $\sigma$ 出来的，所以这两个张量内部的数值都是在 `0-1` 区间上！
  - 生成 **候选隐状态** 需要 **非线性激活函数**；但是更新 **真正的隐状态** 的时候不需要 非线性激活函数！
- **直观理解（但不知道对不对！）**
  - `GRU` 中的候选隐状态 $\bf \~H_t$ 是 在 `RNN` 中 隐状态 $\bf H_t$ 的基础上修改的！
    - 如果 **遗忘门** $\bf R_t = 1$ ，那么 `GRU` 中的 $\bf \~H_t$ 就等于 `RNN` 中的 $\bf H_t$
  - `GRU` 中 **更新门** $\bf Z_t$​ 的作用：
    - 决定「保留多少旧的，写入多少新的」
    - 让 `GRU` 的隐藏层更新时，**可以绕过输入** $\bf X_{t-1}$ （通过 $\bf Z_t = 1$ 来实现！）
- 李沐老师的直观理解：
  - `Reset` $\bf R_t$ 是 我要不要**忘掉**之前的隐状态（信息）。
  - `Update` $\bf Z_t$ 是 我要不要根据之前的隐状态（信息）来**更新**下一个隐状态（信息）。

<br>

#### 3、`LSTM`
1. 权重的矩阵形状
   1. `W_xi`, `W_xo`, `W_xf`, `W_xc` 张量形状 `(num_inputs, num_hiddens)`
   2. `W_hi`, `W_ho`, `W_hf`, `W_hc` 张量形状 `(num_hiddens, num_hiddens)`
   3. 突然意识到：这些 **权重矩阵** 其实都是 **直接的矩阵乘法** 中要用到的权重！
      1. **直接的矩阵乘法** 是指【`RNN, GRU, LSTM` 三者相似点总结】中提出的概念
      2. **直接的矩阵乘法** 计算公式 $~\text{mediumVariable} = \bf XW + HW + b$
      3. 有了这个公式，就非常好理解上述 **权重矩阵的形状** 了！
      4. 同时，因为 **直接的矩阵乘法** 这个概念，也非常容易理解 **权重矩阵的数量** 了！
         1. `GRU` 中有 3 次 **直接的矩阵乘法** ⇒ 因此有 3 * 2 = 6 个权重矩阵
         2. `LSTM` 中有 4 次 **直接的矩阵乘法** ⇒ 因此有 4 * 2 = 8 个权重矩阵

<br><br>

## 二、代码讲解 | 从零实现
```powershell
time traveller ate ate ate ate ate ate ate ate ate ate ate ate a
time traveller the the the the the the the the the the the the t
time traveller and the the the the the the the the the the the t
time traveller the thing the thing the thing the thing the thing
time traveller the paid the peat the peat the peat the peat the 
time traveller the trave traveller the time traveller the trave 
time traveller toun a mithee thisghermanething thit the core tim
time traveller frocknsime i so atlerit weychallowhit s it instar
time traveller abour do and we can flabs dofn any ling and whick
time traveller for so it will be convenient to speak of himwas e
困惑度 1.1, 24976.8 词元/秒 cuda:0
time traveller for so it will be convenient to speak of himwas e
travelleryou can show black is white by argument said filby
```

### （一）、理解代码

<br><br>

## 三、代码讲解 | 简洁实现
```powershell
time traveller the the the the the the the the the the the the t
time travellere the are are are are are are are are are are are 
time traveller this the grace the there the there the there the 
time traveller thing is soulled thing is so it and the said the 
time traveller of the rimensions of space have mand of space and
time traveller for so machler the time travellerit s against net
time traveller for so it will be convenient to speak of himwas e
time traveller for so it will be convenient to speak of himwas e
time travelleryou can show black is white by argument said filby
time travelleryou can show black is white by argument said filby
困惑度 1.1, 105916.5 词元/秒 cuda:0
time travelleryou can show black is white by argument said filby
travelleryou can show black is white by argument said filby
```
### （一）、理解代码

<br>

### （二）、既然学完了 `GRU` 和 `LSTM` 那就分析一下关于 `GRU` 和 `LSTM` 的代码（之前没看的代码）！
```python
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
```
#### 1、【从零实现】时，`state` 用 `turple` 来实现；【简洁实现】时，`state` 用 `增加一个维度` 来实现

<br>

#### 2、`GRU` 和 `LSTM` 中 `detach_()` 的问题