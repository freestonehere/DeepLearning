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
    -  $${\bf F_t=\sigma( X_t W_{xf}+ H_{t-1} W_{hf}+ b_f)} ~ \text{forget，遗忘门}$$
    -  $${\bf O_t=\sigma( X_t W_{xo}+ H_{t-1} W_{ho}+ b_o)} ~ ~ \text{output，输出门}$$
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

### （二）、`nn.LSTM` API
[官网 doc - class torch.nn.LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)

#### 1、符号定义
- $N$：batch size
- $L$：sequence length
- $D$：双向为 2，单向为 1
- $H_{in}$：input_size
- $H_{cell}$：hidden_size
- $H_{out}$：proj_size > 0 时为 proj_size，否则为 hidden_size

#### 2、输入 Inputs
输入为：`input, (h_0, c_0)`

##### (1)、input
- 无批次：$(L, H_{in})$
- `batch_first=False`：$(L, N, H_{in})$
- `batch_first=True`：$(N, L, H_{in})$

支持可变长填充序列，可使用 `pack_padded_sequence` 或 `pack_sequence`。

##### (2)、h_0（初始隐藏状态）
- 无批次：$(D \times num\_layers, H_{out})$
- 有批次：$(D \times num\_layers, N, H_{out})$
未提供时默认为全 0。

##### (3)、c_0（初始单元状态）
- 无批次：$(D \times num\_layers, H_{cell})$
- 有批次：$(D \times num\_layers, N, H_{cell})$
未提供时默认为全 0。

#### 3、输出 Outputs
输出为：`output, (h_n, c_n)`

##### (1)、output
- 无批次：$(L, D \times H_{out})$
- `batch_first=False`：$(L, N, D \times H_{out})$
- `batch_first=True`：$(N, L, D \times H_{out})$

为 LSTM 最后一层每个时间步的输出；双向时为前向与反向隐藏状态拼接。

##### (2)、h_n（最终隐藏状态）
- 无批次：$(D \times num\_layers, H_{out})$
- 有批次：$(D \times num\_layers, N, H_{out})$

包含所有层、所有方向最后时刻隐藏状态，双向为前向与反向最终状态拼接。

##### (4)、c_n（最终单元状态）
- 无批次：$(D \times num\_layers, H_{cell})$
- 有批次：$(D \times num\_layers, N, H_{cell})$

包含所有层、所有方向最后时刻单元状态，双向为前向与反向最终状态拼接。

#### 4、总结
1. 和 `nn.RNN` 相比，`nn.LSTM` 输入输出变化不大，只是在原来的基础上，增加了 `c_n` 而已！

<br>

### （三）、`nn.GRU` API
[官网 doc - class torch.nn.GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU)
#### 1、符号说明
- $N$：batch size
- $L$：sequence length
- $D$：bidirectional=True 时为 2，否则为 1
- $H_{in}$：input_size
- $H_{out}$：hidden_size

#### 2、Inputs: input, h_0
##### (1)、input
- 无批次：$(L, H_{in})$
- batch_first=False：$(L, N, H_{in})$
- batch_first=True：$(N, L, H_{in})$

支持可变长序列，可使用 `torch.nn.utils.rnn.pack_padded_sequence` 或 `torch.nn.utils.rnn.pack_sequence`。

##### (2)、h_0
- 无批次：$(D \times num\_layers, H_{out})$
- 有批次：$(D \times num\_layers, N, H_{out})$

未提供时默认为全 0。

#### 3、Outputs: output, h_n
##### (1)、output
- 无批次：$(L, D \times H_{out})$
- batch_first=False：$(L, N, D \times H_{out})$
- batch_first=True：$(N, L, D \times H_{out})$

为 GRU 最后一层每个时间步的输出特征；若输入为 `PackedSequence`，输出也为 `PackedSequence`。

##### (2)、h_n
- 无批次：$(D \times num\_layers, H_{out})$
- 有批次：$(D \times num\_layers, N, H_{out})$

为序列最终的隐藏状态。

<br>

### （四）、为便于比较学习，把 `nn.RNN` API 也列出来
[官网 doc - class torch.nn.RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN)

- **发现：`nn.GRU` 和 `nn.RNN` 在输入和输出的张量形状上是完全一样的！！！**
- 补充关于高级 API 的理解（照搬 [43-1st-seq-model-RNN.md](43-1st-seq-model-RNN.md) 中的内容）
    |       |高级 API 输出的 `output`|高级 API 输出的 `h_n`|
    |:------|:----------------------|:-------------------|
    |张量形状|`(num_steps, batch_size, num_hiddens)`|`(1, batch_size, num_hiddens)`|
    |含义|既然是 `output` 了，<br>自然是 **最后一层神经网络**<br>（的所有时间步）。|既然 `output` 是最后一层神经网络了，<br>那么 `h_n` 自然是 **最后一个时间步**<br>（的所有层神经网络）|

#### 1、符号说明
- $N$：batch size
- $L$：sequence length
- $D$：bidirectional=True 时为 2，否则为 1
- $H_{in}$：input_size
- $H_{out}$：hidden_size

#### 2、Inputs: input, hx
##### (1)、input
- 无批次：$(L, H_{in})$
- batch_first=False：$(L, N, H_{in})$
- batch_first=True：$(N, L, H_{in})$

支持可变长序列，可通过 `torch.nn.utils.rnn.pack_padded_sequence()` 或 `torch.nn.utils.rnn.pack_sequence()` 处理。

##### (2)、hx
- 无批次：$(D \times num\_layers, H_{out})$
- 有批次：$(D \times num\_layers, N, H_{out})$

未提供时默认为全 0。

#### 3、Outputs: output, h_n
##### (1)、output
- 无批次：$(L, D \times H_{out})$
- batch_first=False：$(L, N, D \times H_{out})$
- batch_first=True：$(N, L, D \times H_{out})$

为 RNN 最后一层每个时间步的输出特征；若输入为 `PackedSequence`，输出也为 `PackedSequence`。

##### (2)、h_n
- 无批次：$(D \times num\_layers, H_{out})$
- 有批次：$(D \times num\_layers, N, H_{out})$

表示批次中每个样本的最终隐藏状态。

<br>

### （五）、既然学完了 `GRU` 和 `LSTM` 那就分析一下关于 `GRU` 和 `LSTM` 的代码（之前没看的代码）！

#### 1、`GRU` 和 `LSTM` 中 `detach_()` 的问题
```python
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
```

```python
'''抽出核心代码如下'''
if state is None or use_random_iter:
    # 在第一次迭代或使用随机抽样时初始化 state（隐藏层状态）
    state = net.begin_state(batch_size=X.shape[0], device=device)
else:
    '''整个 else 的大环境是：
    ① 前面已经迭代过，已经产生了 state
    ② 并且，数据采样方式还是【顺序分区】'''
    
    '''关于 GRU 和 LSTM 的代码以后再看！
    反正 RNN 本身是不需要 detach_() 的，因为 RNN 一直在更新隐藏层的参数！'''
    if isinstance(net, nn.Module) and not isinstance(state, tuple):
        '''如果 ① net 是简洁实现 并且 ② state 不是 turple 形式 
        ⇒ 那就把前面产生的 state 分离出计算图'''
        # state 对于 nn.GRU 和 nn.RNN 是个张量
        state.detach_()
    else:
        # state 对于 nn.LSTM 或对于我们从零开始实现的模型是个元组
        for s in state:
            s.detach_()
```
1. 如果采用**简洁实现**
   1. `RNN` 中的 `state` 是 `(num_layers, batch_size, num_hiddens)` **张量**
      1. 与之对比，如果采用**从零实现**，`RNN` 中的 `state` 则是 `((batch_size, num_hiddens), )`
      2. （**外面**是长为 `1` 的 `turple`，**里面**是二维张量 `(batch_size, num_hiddens)`）
      3. 其实严格来说，外面应该是长为 `num_layers` 的**一元组**，但是由于 **这里无论【从零实现】还是【简洁实现】** 都只有 1 层网络，于是就简化了！
   2. `GRU` 中的 `state` 和 `RNN` 简洁实现的输出一样：`(num_layers, num_steps, num_hiddens)` **张量**
      1. 如果采用**从零实现**，`GRU` 和 `RNN` 中的 `state` **应该一致**！
      2. 如果仅从**输入 / 输出的接口来讲**，`RNN` 和 `GRU` 是完全一致的（不管你是 **从零实现** 还是 **简洁实现**！）
   3. `LSTM` 中的 `state` 是 `(h_n, c_n)` **二元组**
      1. 与之对比，如果采用**从零实现**，`LSTM` 中的 `state` 则是则是 `((batch_size, num_hiddens), (batch_size, num_hiddens))`
      2. （**外面**是长为 `2` 的 `turple`（一个代表 `隐状态`，一个代表 `记忆单元`），**里面**是二维张量 `(batch_size, num_hiddens)`）
      3. 其实严格来说，外面还是是长为 `2` 的**二元组**，但是里面应该是【和 `RNN` 从零实现的 `state` 一样】的**一元组**。由于 **这里无论【从零实现】还是【简洁实现】** 都只有 1 层网络，于是就简化了！（也就是 `torch.stack([h_n, c_n])`，其中 `h_n`, `c_n` 都是 `(batch_size, num_hiddens)` 的张量）
2. `detach_()` 这一操作对于使用连续小批量（非随机抽样）训练循环神经网络**至关重要**，它能保证**梯度计算只依赖**于**当前小批量**内的时间步，而不会通过状态追溯至之前的小批量
   1. 这就是在 **反向传播计算** 中，实现 **马尔科夫假设**！即当前输出只与过去一定数量（ $\tau$ ）的输入有关，并不是与过去所有输入都有关！
   2. 之前在 **前向计算** 中，是如何实现 **马尔科夫假设** 的呢？详见 [43-1st-seq-model-RNN.md](43-1st-seq-model-RNN.md)
      1. 代码中隐藏层是一直 **从【子序列】的开头** 传递下来的
      2. 等等，我明白了！代码就是通过 **控制子序列的长度** 来实现 **马尔科夫假设**！
      3. 代码中 **子序列的长度 `num_steps`** 就是 **马尔科夫假设** 中的 $\tau$ ！

<br>

##### (1)、`detach_()` 对反向传播的影响具体是怎么做到的？
回答来自 DeepSeek。问答链接：[https://chat.deepseek.com/a/chat/s/5463378c-f8cb-4b94-af77-9d84c76310cb](https://chat.deepseek.com/a/chat/s/5463378c-f8cb-4b94-af77-9d84c76310cb) 。看完这个回答以后，好像有一点理解了！不理解的原因应该是**没有学习反向传播算法的原理，只是知道有反向传播这个过程**。

你的疑问非常深刻，触及了截断 BPTT（Truncated Backpropagation Through Time）的核心。让我们一步步理清这个问题。

###### ①. 权重共享与梯度累积
首先，你正确地指出：RNN 的权重矩阵（例如从输入到隐藏、隐藏到隐藏的变换）在所有时间步和所有 batch 中是**同一个参数**。在 PyTorch 中，这些参数是同一个 `nn.Parameter` 对象，无论我们前向计算多少次，它们始终共享。

在反向传播时，梯度会沿着计算图从损失函数一路回传，并累加到这些参数上（通过 `.grad` 属性）。最终，优化器使用所有累积的梯度来更新参数。因此，即使我们使用 `detach_()`，参数仍然是共享的，每个 batch 产生的梯度都会贡献到同一个参数的梯度上，然后一次性更新。

###### ②. 为什么还需要 `detach_()`？
虽然参数共享，但**计算图的长度**决定了梯度传播的范围。如果不使用 `detach_()`，状态 `state` 会携带之前 batch 的计算历史。让我们用一个简化例子说明：

假设我们有连续两个 batch，每个 batch 包含一个时间步（`num_steps=1`）。  
- Batch 1：输入 `X1`，初始状态 `s0`（通常为零），计算输出 `y1` 和状态 `s1`。  
- Batch 2：以 `s1` 作为初始状态，输入 `X2`，计算输出 `y2` 和状态 `s2`。

如果不 `detach` `s1`，那么：
- `s1` 是通过 Batch 1 的计算得到的，它依赖于 Batch 1 的输入和参数。
- 在 Batch 2 的前向传播中，`s2` 依赖于 `s1` 和 Batch 2 的输入。
- 因此，Batch 2 的损失 `L2` 的计算图会一直回溯到 Batch 1 的输入和参数，形成一个跨越两个 batch 的长计算图。

当我们对 `L2` 调用 `.backward()` 时，梯度会沿着这个长图传播，既影响 Batch 2 内部的参数贡献，也影响 Batch 1 内部的参数贡献。也就是说，**梯度会从 Batch 2 一直传到 Batch 1 的所有时间步**。这相当于我们把两个 batch 的序列拼接成一个更长的序列（Batch1 + Batch2）进行了完整的 BPTT。

- 这样做的问题：
  - **内存爆炸**：计算图会随着 batch 数量线性增长。PyTorch 必须保留所有中间变量直到反向传播结束。如果连续多个 batch 都不 `detach`，最终内存会耗尽。
  - **训练不稳定**：过长的 BPTT 容易导致梯度消失/爆炸，且训练效率低。
  - **偏离设计**：在训练语言模型等任务时，我们通常希望每个 batch 独立地计算梯度（即截断 BPTT），只让梯度在 batch 内部的时间步上传播。这样既能捕捉短期依赖，又避免过长依赖带来的计算负担。

###### ③. `detach_()` 的作用
`detach_()` 在将 `s1` 传入 Batch 2 之前，将其从计算图中剥离，使得 `s1` 变成一个**常数**（不再需要梯度）。这样：
- Batch 2 的前向计算仍然使用 `s1` 的**数值**，因此模型能够利用之前 batch 的上下文信息。
- 但是，`s1` 的 `requires_grad` 变为 `False`，所以反向传播时梯度不会通过 `s1` 流入 Batch 1 的计算图。
- 因此，`L2` 的梯度只会计算 Batch 2 内部的时间步，参数更新只基于当前 batch 的贡献。

尽管参数是共享的，但每个 batch 的梯度计算是独立的。这些独立的梯度最终累加到参数上，然后优化器一次性更新。这正是截断 BPTT 的标准实现方式。

###### ④. 类比理解
想象你在训练一个非常深的网络，但你不想一次性反向传播整个网络，而是分阶段进行。`detach_()` 就像是告诉 PyTorch：“这个状态的值很重要，我要用它作为下一阶段的输入，但请别把它之前的计算图带过来，我只想在本阶段内反向传播。”

###### ⑤. 总结
你的理解是正确的：权重矩阵确实是跨 batch 共享的，但 `detach_()` 的目的是**切断跨 batch 的梯度流**，限制反向传播的时间范围，而不是切断参数共享。这样既利用了历史信息进行前向计算，又保证了每个 batch 的梯度计算是局部的、可控的。