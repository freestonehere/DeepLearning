import torch
from torch import nn
from d2l import torch as d2l
from tools.ch6 import train_ch6
import matplotlib.pyplot as plt # 用于画图
from datetime import datetime

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

# Python 默认 float64
X = torch.randn(size=(1, 1, 224, 224), dtype=torch.float32)
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

starttime = datetime.now() 
print(starttime) # 打印当前时间

train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

endtime = datetime.now()
print(endtime)
print(endtime-starttime)

plt.show()