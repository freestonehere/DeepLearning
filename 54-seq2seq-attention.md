# Seq2seq using Attention | 使用注意力机制的 seq2seq
## 一、概念讲解
### （一）、动机
- 机器翻译中，每个生成的词可能相关于源句子中不同的词
  - 传统的 seq2seq 模型不能对此直接建模
  - 一个编码-解码器串联的 RNN 里，有一句话"Hello world!"，翻译成其他语言，第一个词看的不是"Hello"，而是"!"，就显得不合理。
  - 引入注意力机制就是要把关注点放在源句子里的对应部分

### （二）、加入注意力
![加入注意力机制的 encoder-decoder](https://zh-v2.d2l.ai/_images/seq2seq-attention-details.svg)

- **直观理解：这是一个【注意力池化层】！**
- **只在解码器加入注意力机制，编码器不需要注意力机制**
- 编码器对每次词的输出作为 key 和 value（这里他们是同一个东西），即第 $i$ 个词在 RNN 里的输出。
- 解码器 RNN 对上一个词的输出是 query
  - 因为当前词还没有被预测，所以只能拿与之语义最接近的上一个词
  - 同一个 RNN，同一个语义空间
- 注意力的输出和下一个词的词嵌入合并进入（取代原先按顺序传入的的隐藏层）
  - 就是把所有的词拿出来，做一个相关性的加权，成为下一刻的输入

### （三）、总结
- Seq2seq 中通过隐状态在编码器和解码器中传递信息
- 注意力机制可以根据解码器 RNN 的输出来匹配到合适的编码器 RNN 的输出来更有效地传递信息。

<br><br>

## 二、代码讲解
```powershell
# 这是代码运行结果
output.shape: 
torch.Size([4, 7, 10])
 
len(state): 
3
 
state[0].shape
torch.Size([4, 7, 16])

len(state[1]):
2

state[1][0].shape:
torch.Size([4, 16])

loss 0.020, 7989.7 tokens/sec on cuda:0
go . => va !,  bleu 1.000
i lost . => j'ai perdu .,  bleu 1.000
he's calm . => il est malade .,  bleu 0.658
i'm home . => je suis chez moi .,  bleu 1.000
```
### （一）、理解代码

<br>

### （二）、回答 `class Seq2SeqAttentionDecoder.forward()` 函数注释中的疑问！
```python
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = attn.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs 的形状为 (batch_size，num_steps，num_hiddens).
        # hidden_state 的形状为 (num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs 的形状为 (batch_size, num_steps, num_hiddens).
        # hidden_state 的形状为 (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出 X 的形状为 (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # hidden_state[-1] 张量形状 (batch_size, num_hiddens)
            # query 的形状为 (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context 的形状为 (batch_size, 查询个数, ) = (batch_size, 1, num_hiddens)
            # x 形状为 (batch_size, embed_size)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            '''enc_outputs 既是 keys 又是 values!
            
            在生成 context 张量的时候，同时也计算出了注意力权重，权重张量形状为
            (batch_size, 查询的个数, “键-值”对的数目) = (batch_size, 1, num_steps)!'''
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 经过 torch.cat 后，张量形状变为 (batch_size, 1, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            '''
            张量形状：
            out (1, batch_size, num_hiddens)
            hidden_state (num_layers, batch_size, num_hiddens)

            out 张量形状和 50-seq2seq.py class Seq2SeqDecoder 中的 out 不同，
            因为 50-seq2seq.py 直接把所有时间步的 X 都传进来了！
            但是这里只传进来 1 个时间步的 X !

            不过这里为什么要这样做呢？是因为加了注意力机制的原因吗？

            其实 50-seq2seq.py 也要一个时间步一个时间步地预测，不过高级 API 封装细节罢了！
            '''
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs 的形状为
        # (num_steps, batch_size, vocab_size) → (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights 
```
#### 1、首先理解 decoder with attention 预测【第一个时间步】的时候做了什么
1. `hidden_state (num_layers, batch_size, num_hiddens)` 是最后一个时间步所有层 RNN 的 **隐变量集合**！
   1. 取出 `hidden_state[-1] (batch_size, num_hiddens)`（也就是最后一个时间步最后一层 RNN 的 **隐变量**）
   2. 然后 `query = hidden_state[-1].unsqueeze(dim=1)` 得到 `query (batch_size, num_queries, query_size) = (batch_size, 1, query_size)`
   3. **直观理解**：由于只预测一个时间步，所以 `查询数目 = 1`，当然很正常！
2. 以最后一个时间步最后一层 encoder 的 `hidden_state` 作为 `query`，以最后一层 encoder **所有时间步上的输出** `enc_outputs (batch_size, num_steps, num_hiddens)` 作为 `key (batch_size, num_keys, key_size) = (batch_size, num_steps, num_hiddens)` 和 `value (batch_size, num_keys, value_size) = (batch_size, num_steps, num_hidddens)`。
   1. 然后用 **`query`** 去查询 **`key`**，也就是 `context = self.attention(···)`
   2. 得到**加权后的 `value` 张量**，也就是 `value = context` `(batch_size, num_queries, value_size) = (batch_size, 1, num_hiddens)`
   3. 然后，**加权后的 `value = context` 张量** 和 `dec_input[i] (batch_size, 1, embed_size)` 拼接，得到 `(batch_size, 1, embed_size + num_hiddens)` 该结果再输入 decoder！
   4. **直观理解**：果然就是比 [50-seq2seq.py](50-seq2seq.py) 改进了！
      1. 之前 [50-seq2seq.py](50-seq2seq.py) 只取 **最后一个时间步最后一层 RNN 的隐变量** `(1, batch_size, num_hiddens)` 重复 `num_steps` 次，成为 `(num_steps, batch_size, num_hiddens)` ；然后 **`(num_steps, batch_size, num_hiddens)` 和 `dec_input (num_steps, batch_size, embed_size)` 拼接**，得到 `(num_steps, batch_size, embed_size + num_hiddens)`。
      2. **也就是说**：之前只取【① 最后一个时间步】和【② `X`】拼接，而现在取【① 所有时间步】和【② `x`】拼接，并且**在所有时间步的跨度上**进行**加权**！
      3. 这里【**在所有时间步的跨度上**进行**加权**】的意思是：要预测 `decoder` 第 `i` 个时间步上的 `token`，那就**主要看** `enc_output` **对应时间步**上的 `数值`。
         1. 【问】：为什么要看 `enc_output` 对应时间步上的数值，而不看 `enc_input` **这个原始 token 列表对应时间步上的文本呢**？
         2. 【答】：可以把 `encoder` 和 `decoder` 内部的数值看作在**同一个语义空间内！**
<br>

#### 2、然后，仿照上面的分析思路，理解 decoder with attention 预测【第 `i` 个时间步】的时候做了什么
- 分析【预测第 `i` 个时间步】时，`query`、`key`、`value` 分别是什么。
1. `query` 是 **从** 上一个时间步输出的 `hidden_state (num_layers, batch_size, num_hiddens)` 中 **取出** 最后一层 RNN 输出的隐变量（即代码中的 `hidden_state[-1]`）
2. `key` 和 `value` 不变，都还是 `enc_output` 还是和【预测第一个时间步的时候】一样！
3. **总结**：decoder with attention 在预测多个时间步时
   1. 是一个时间步一个时间步预测的，就算 **训练** 的时候也是一个一个预测的，**不能同时预测多个时间步**！
      1. 与之对比，decoder without attention ~~**训练** 的时候是可以 **同时预测多个时间步** 的~~；但是 **预测** 的时候就 **不能** 同时预测多个时间步了！
   2. 每次预测时的 `query` **各不相同**
   3. 但每次预测时的 `key` 和 `value` **全都一样**！
<br>

#### 3、理解为什么：即便是【训练】的时候，decoder with attention 【也不能同时预测多个时间步】
- 之前普通 RNN 训练的时候可以同时预测多个时间步，是因为（下面分析 [43-1st-seq-model-RNN-01.py](43-1st-seq-model-RNN-01.py) 中的 RNN）
  - [50-seq2seq.py](50-seq2seq.py) 中 `self.rnn = nn.GRU`，由于 `nn.GRU` 内部实现过于复杂。那就使用 [43-1st-seq-model-RNN-01.py](43-1st-seq-model-RNN-01.py) 中手动实现的 `RNN` 来代替（唯一不同的是 `nn.GRU` 没有输出层，而手动实现的 `RNN` 有输出层！）
  - 然后发现：其实普通 `RNN` 手动实现的时候，**训练** 时也是一个时间步一个时间步预测的。只不过 [50-seq2seq.py](50-seq2seq.py) 中的高级 API 把细节给封装起来了！
    ```python
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
    ```
