# LeNet （经典卷积神经网络 - LeNet）
## 一、概念理解
1. 没有 overfitting ⇒ 意味着很可能 underfitting
2. 卷积模型比 MLP 小
   1. 因为卷积是受限制的全连接层
   2. 卷积一般不会 overfitting
3. LeNet 的思路
   1. 先利用**卷积**，把特征从图片提取到通道中
      1. 反映在代码上就是
      2. 图片尺寸逐渐减小
      3. 通道数目逐渐增多
   2. 然后将这些通道的特征 **展平之后全连接**（拉成一个一维向量），然后**输入 MLP**
      1. 全连接层的命名真是**形象**
         1. 就是将**批次维度保留，其他所有维度**拉成一维向量
         2. 这不正就是全连接的含义吗！
   3. 最后由 MLP 逐步压缩一维向量，最终输出分类结果 label
4. **注意：** 每层卷积层之后也是有激活函数的！
   1. 这进一步验证了 **卷积其实也是 MLP 的一种**！
5. 时序和文本是类似的，都是一条线

### （一）、有意思的问题
1. **Q：max pooling 和 average pooling 哪个用的更多？**
   1. **🙋‍♂️**：二者差别不大（可能在具体问题有细微差别），一般来说 max pooling 用的更多，因为 max pooling 得到的数值更大，相对梯度比 average 也个更大，更好训练。
2. [卷积可视化-用于学习卷积神经网络](https://poloclub.github.io/cnn-explainer/)
3. 跑得动的情况下，可以尽量将中间层的输出通道数调大吗？
   1. 当然不能盲目调大
   2. 因为你模型过大的话，又容易 overfitting（见 `11-model-select.md` 那一章）

<br><br><br><br>

## 二、代码

### （一）、理解代码
#### 1、张量形状（观察特征图尺寸的变化）
```python
# 张量形状 (batch_size, channels, h, w)

'''输入张量'''
X = (1, 1, 28, 28)

'''第 1 层卷积层：尺寸减小，通道增多'''
# 正常来讲，第一层卷积层输出后，图片尺寸应该会比输入尺寸减小
# 这里没有减小，是因为卷积的时候，加入了 padding
Conv2d output shape:     torch.Size([1, 6, 28, 28])
Sigmoid output shape:    torch.Size([1, 6, 28, 28])
AvgPool2d output shape:          torch.Size([1, 6, 14, 14])

'''第 2 层卷积层：尺寸进一步减小，通道进一步增多'''
Conv2d output shape:     torch.Size([1, 16, 10, 10])
Sigmoid output shape:    torch.Size([1, 16, 10, 10])
AvgPool2d output shape:          torch.Size([1, 16, 5, 5])

'''展平之后全连接（批次维度保留，其他所有维度拉成一维向量）'''
Flatten output shape:    torch.Size([1, 400])

'''将全连接后的向量输入 MLP：将一维向量逐步压缩，最终压成一个 label'''
Linear output shape:     torch.Size([1, 120])
Sigmoid output shape:    torch.Size([1, 120])
Linear output shape:     torch.Size([1, 84])
Sigmoid output shape:    torch.Size([1, 84])
Linear output shape:     torch.Size([1, 10])
training on cuda:0
loss 0.467, train acc 0.823, test acc 0.771
29925.1 examples/sec on cuda:0
```