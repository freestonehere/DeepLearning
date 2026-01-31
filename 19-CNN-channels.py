import torch

'''单输入通道 + 单输出通道 卷积操作'''
def corr2d(X, K):  #@save
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() 
            # 注意切片的语法；
            # 另外，这里的 * 是点积（真是没想到矩阵还有点积，不过就是模仿向量点积，加权和罢了）
    return Y
    # shape 都是先行后列（ 0 轴是从外面开始计算的 ）

'''多输入通道 + 单输出通道 卷积操作'''
def corr2d_multi_in(X, K):
    # 先遍历 X 和 K 的第 0 个维度（通道维度），再把它们加在一起
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print('多输入通道 + 单输出通道 卷积操作')
print(corr2d_multi_in(X, K), end='\n\n')



K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

'''多输入通道 + 多输出通道 卷积操作'''
def corr2d_multi_in_out(X, K):
    # 迭代 “K” 的第 0 个维度，每次都对输入 “X” 执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

'''构造新的卷积核张量, 适配多输出通道'''
K = torch.stack((K, K + 1, K + 2), 0)

print('多输入通道 + 多输出通道 卷积操作')
print(corr2d_multi_in_out(X, K), end='\n\n')



'''多输入通道 + 多输出通道的 1⨉1 卷积操作 (采用全连接层拉成张量的形式)'''
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape # X = (c_i, h, w)
    c_o = K.shape[0]    # K = (c_o, c_i, h, w)
    X = X.reshape((c_i, h * w)) # 将
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    # 对于卷积层，严格来讲，应当是 XK
    # 但是这里的形式本身就不规范
    # 为了矩阵乘法能够正常进行 K=(c_o, c_i) X=(c_i, h*w)
    # 所以这里就写成 KX
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print('多输入通道 + 多输出通道 1 ⨉ 1 卷积操作')
print(Y1)
print(Y2)