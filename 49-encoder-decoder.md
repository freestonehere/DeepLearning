# Encoder - Decoder | 编码器 - 解码器（只讲架构，不讲具体实现）
## 一、概念讲解
### （一）、重新考察 CNN
- 编码器：将输入编程成中间表达形式（特征）
- 解码器：将中间表示解码成输出

### （二）、重新考察 RNN
- 编码器：将文本表示成向量
- 解码器：将向量表示成输出

### （三）、编码器-解码器架构
![编码器-解码器架构](https://zh-v2.d2l.ai/_images/encoder-decoder.svg)

- 一个模型被分为两块
  - 编码器处理输入
  - 解码器生成输出
    - 解码器可以有额外的输入

<br><br>

## 二、代码讲解
```python
from torch import nn

'''编码器接口'''
#@save
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```python
'''解码器接口'''
#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```python
'''编码器-解码器接口'''
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # 从编码器的【输入】得到编码器的【输出】
        enc_outputs = self.encoder(enc_X, *args)
        # 从编码器的【输出】得到解码器的【状态】
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 从解码器的【状态】和解码器的【输入】得到解码器的【输出】
        return self.decoder(dec_X, dec_state)
```
