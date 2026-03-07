# Deep RNN | 深度循环神经网络
## 一、概念理解
- 问：如何获得更多的非线性性？
- 答：多隐藏层

![多隐藏层的 RNN](https://zh-v2.d2l.ai/_images/deep-rnn.svg)

1. $$\bf H_t^1=f_1( H_{t-1}^1,X_t)$$
2. $$\bf H_t^2=f_2( H_{t-1}^2,H_t^1)$$
3. $$...$$
4. $$\bf H_t^j=f_j( H_{t-1}^j,H_t^{j-1})$$
5. $$...$$
6. $$\bf O_t=g( H_t^L)$$

- 澄清一点，每一个 **横行** 才是一个隐藏层；每一个 **横行内部** 是隐变量！
  - 横向宽度是 **时间步** 的数量
  - 纵向深度是 **隐藏层** 的数量
- 每层模型里打包了非线性函数
  - 当然要打包非线性函数，如果不打包非线性函数，模型就没有非线性能力，导致模型层数的坍塌！
  - 这里指的是在**纵向**维度上打包非线性函数！
  - 如何判断是否需要非线性函数？照搬 [43-1st-seq-model-RNN.md](43-1st-seq-model-RNN.md) 中的内容
    - 单隐藏层的 `RNN` 更新 **隐藏状态** 需要用非线性激活函数
    - 但是获取 **输出** 不需要非线性激活函数（详见代码 [43-1st-seq-model-RNN.py](43-1st-seq-model-RNN-01.py)）
    - 一种理解是：你获取的 **输出** 不需要 **再进入神经网络**，所以自然没必要再增添一个非线性激活函数！
- 对于一个**隐变量** $\bf H_{t-1}^{n-1}$
  - 它既是 **本隐藏层下一个时间步** 的 **【隐变量输入】**，用于生成隐变量 $\bf H_{t}^{n-1}$
  - 又是 **下一个隐藏层在该时间步上** 的 **【输入】**，用于生成隐变量 $\bf H_{t-1}^{n}$
  - 注意：**隐变量输入** 和 **输入** 是不一样的！隐变量输入是 $\bf H$ ，输入是 $\bf X$

### （二）、总结
- 深度循环神经网络使用多个隐藏层来获得更多的非线性性

<br><br>

## 二、代码讲解
```powershell
time traveller te a a a a a a a a a a a a a a a a a a a a a a a 
time traveller and the the the the the the the the the the the t
time traveller thing the merical of the time traveller thing the
time traveller what fist any in in wither at we have expessimed 
time traveller whole were abspathe three dimensional solid and s
time travelleryou can show black is white by argument said filby
time traveller with a slight accession ofcheerfulness really thi
time traveller with a slight accession ofcheerfulness really thi
time travelleryou can show black is white by argument said filby
time travelleryou can show black is white by argument said filby
困惑度 1.0, 200074.3 词元/秒 cuda:0
time travelleryou can show black is white by argument said filby
travelleryou can show black is white by argument said filby
```
