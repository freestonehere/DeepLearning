import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import tools.ch13 as ch13
import matplotlib.pyplot as plt # 用于画图
from datetime import datetime
import tools.cache_load_datasets as cache_load

# 构造模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
print('查看预训练模型的最后 3 层（池化+全连接，之前很熟悉的！）\n',
      list(pretrained_net.children())[-3:])

# 丢掉 RedNet 的【池化层+全连接层】，保留前面的内容！
net = nn.Sequential(*list(pretrained_net.children())[:-2])
X = torch.rand(size=(1, 3, 320, 480))
# net 的前向传播将输入的高和宽减小至原来的 1/32 （ResNet 决定的缩放比例）
# 注意：ResNet 在减小尺寸的同时，也在增大通道数
print(f'net(X).shape: {net(X).shape}')
# 输出 net(X).shape: torch.Size([1, 512, 10, 15])

# 教材上讲了参数是怎么计算的！到时候看看
num_classes = 21 # 这是加上 class background 以后，一共有 21 类！
# 1x1 卷积层不改变图片尺寸，但是可以融合通道数（便于减小计算量）
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

# 初始化转置卷积层（用双线性插值来初始化）
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor
            ) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 套用公式得到 ⇒ 该转置卷积层会将图片尺寸扩大 2 倍。但网络中的转置卷积层就不是这样了！
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

print('\n测试小组件（转置卷积层）的效果（增大图片尺寸）', 
      '但是明确：网络中用的是另一个转置卷积层！')
# 这个 ToTensor()() 的写法好奇怪啊！应该没见过？
img = torchvision.transforms.ToTensor()(d2l.Image.open('./data/img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()

d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
fig = plt.gcf()
ch13.set_title(fig, 'input image shape')

print('output image shape:', out_img.shape)
plt.figure()
d2l.plt.imshow(out_img)
fig = plt.gcf()
ch13.set_title(fig, 'out image shape')

# 对于 1x1 卷积层，用 Xavier 初始化；转置卷积层用双线性插值初始化
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = ch13.load_data_voc(batch_size, crop_size)

# 训练
def loss(inputs, targets):
    '''本质还是一个分类问题，所以还是用交叉熵损失函数。
    input (batch_size, num_classes, h, w); target (batch_size, 1, h, w)。
    返回张量形状 (batch_size,)'''
    # 由于 F.cross_entropy(reduction='none') 返回的张量是 (batch_size, h, w)
    # 所以：先用 mean(keepdim=False) 降维成 (batch_size, w)；
    # 然后再用 mean(keepdim=False) 降维成 (batch_size,)
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
# 优化器不关心模型结构，只对【可训练参数】进行优化！
# 但是话说，weight_decay 是怎么实现的？已经完全忘了。

starttime = datetime.now() 
print(starttime) # 打印当前时间

ch13.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

endtime = datetime.now()
print(endtime)
print(endtime-starttime)

# 预测
def predict(img):
    '''输入三维张量 (channel, h, w)；返回二维张量 (h, w)。
    因为 argmax 会使得张量退化！'''
    # X 张量形状 (1, 3, h, w)
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    '''输入的 pred 张量是二维 (h, w)；
    将 [0, num_classes) 范围内的【类别标号】转换为【图片的 RGB 值】'''
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
    # colormap[X] 会先把 X 的每个元素映射到 colormap 对应的行，得到形状 (H, W, 3)；
    # : 是冗余但清晰的写法（等价于 colormap[X]），明确表示取全部 3 个通道；

# 打印几张图片，看看效果
voc_dir = cache_load.download_extract('voc2012', 'VOCdevkit/VOC2012')
# 这是把全部的 test_image, test_labels 都直接读进内存来了！
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    # 这里的 X 是三维 (channel, h, w)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
    # 虽然 PyTorch 标准张量格式是通道优先
    # 但是图像显示中的张量格式是通道最后
    # 所以，这里必须 permute(1, 2, 0)

# plt.figure() 如果这里加上 plt.figure() 那就会多出一个空白窗口
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
fig = plt.gcf()
ch13.set_title(fig, '打印几张图片，看看效果')

plt.show()