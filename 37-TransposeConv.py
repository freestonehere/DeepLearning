import torch
from torch import nn
from d2l import torch as d2l

print('手动实现转置卷积')
def trans_conv(X, K):
    '''转置卷积函数'''
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    # 遍历输入的每一个元素
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(f'trans_conv(X, K): {trans_conv(X, K)}')

# 使用高级 API 获得同样的结果
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(f'tconv(X): {tconv(X)}')

# 填充、步幅和多通道（多通道其实比较简单，主要是填充和步幅）
print(f'\n填充、步幅和多通道')
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(f'tconv(X): {tconv(X)}')
# 只考虑尺寸的话，输入 2x2，输出 1x1

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(f'tconv(X): {tconv(X)}')
# 只考虑尺寸的话，输入 2x2，输出 4x4
# 不太懂这个尺寸是怎么变化的！

X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(f'tconv(conv(X)).shape == X.shape: {tconv(conv(X)).shape == X.shape}')

# 与矩阵变换之间的联系
print(f'\n与矩阵变换之间的联系')
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(f'Y: {Y}')
# 从尺寸上来看，输入 3x3，输出 2x2

def kernel2matrix(K):
    '''将【核】拉成【矩阵】；
    不过有个疑问：为什么你生成的这个矩阵既可以用于【卷积】，又可以用于【转置卷积】'''
    # K 是 2x2 矩阵。输入 3x3=9，输出 2x2=4
    # 矩阵形状 (4, 1) = (4, 9) * (9, 1)；左乘行变换
    k, W = torch.zeros(5), torch.zeros((4, 9))
    # K 的第一行分配给 k[0], k[1]
    # K 的第二行分配给 k[3], k[4]
    # k[2] = 0 还是没变。因为左乘行变换，且 2x2 核的窗口在 3x3 输入中滑动时，
    # 不会覆盖到第 3 个位置（对应【列向量】的 x3/x6/x9 边缘）
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W
# 我可以理解：这段代码生成的 W 可以用于【卷积】，因为就是按照【卷积核】生成的！
# 但是我不理解：为什么 W.T 可以用于【转置卷积】！
'''算了，不钻牛角尖了。先记住吧。
另外：卷积是加权求和、转置卷积是加权分散！'''
W = kernel2matrix(K)
print(f'W: {W}')

# Y 是卷积后的结果，Z 是转置卷积后的结果
# 矩阵 W (4, 9)
# Y = corr2d(X, K) 
print('Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2):\n',
      Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))

# 矩阵 W.T (9, 4)
Z = trans_conv(Y, K)
print('Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)\n', 
      Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))