import torch
from d2l import torch as d2l

# 有隐状态的循环神经网络
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
print('torch.matmul(X, W_xh) + torch.matmul(H, W_hh):',
      f'\n{torch.matmul(X, W_xh) + torch.matmul(H, W_hh)}')

'''【发现】：如果把隐藏层 W_hh 去掉，那么其实就是【单层感知机】！
即 torch.matmul(X, W_xh)，就是多层感知机！'''

print('\ntorch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)):',
      f'\n{torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))}')

