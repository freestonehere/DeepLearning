import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from tools.cache_load_datasets import *
import matplotlib.pyplot as plt # 用于画图
from tools.ch13 import * # 导入 train_ch13 函数
from datetime import datetime

#@save
# 上一个用到 DATA_HUB 的代码是 15-combine-01.py
DATA_HUB['hotdog'] = (DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = download_extract('hotdog')

# ImageFolder 其实是创建了一个对象，也就是说 train_imgs 其实是一个对象
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# 使用RGB通道的均值和标准差，以标准化每个通道
# 但是为什么呢？有点陌生。可能是模仿 Batch Norm 的思路？
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# pretrained_net=True 保证预训练模型的参数也被复制过来
pretrained_net = torchvision.models.resnet18(pretrained=True)

print(f'print my model pretrained_net.fc: {pretrained_net.fc}')

# fc 其实就是 fully_connected 全连接层！
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
# 这里只需要初始化全连接层的参数即可！前面提取特征的参数不要初始化，
# 直接用预训练的参数即可！
nn.init.xavier_uniform_(finetune_net.fc.weight)

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)

starttime = datetime.now() 
print(starttime) # 打印当前时间

train_fine_tuning(finetune_net, 5e-5)

endtime = datetime.now()
print(endtime)
print(endtime-starttime)

plt.show()