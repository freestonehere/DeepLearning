import torchvision
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
from tools.ch13 import *

d2l.set_figsize()
img = d2l.Image.open('data/img/cat1.jpg')
plt.imshow(img)

# 获取当前图形对象
fig = plt.gcf()
set_title(fig, '原始图片 - 猫咪')

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 水平翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
fig = plt.gcf()
set_title(fig, '水平翻转')

# 垂直翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
fig = plt.gcf()
set_title(fig, '垂直翻转')

# 裁切
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
fig = plt.gcf()
set_title(fig, '裁切')

apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
fig = plt.gcf()
set_title(fig, '改变颜色')

apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
fig = plt.gcf()
set_title(fig, '改变色调')

color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
fig = plt.gcf()
set_title(fig, '亮度、对比度、饱和度和色调')

augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
fig = plt.gcf()
set_title(fig, '3 种常用的图像增广方法')

# 没有这行代码的话，那就不会跳出 Figure 来！
plt.show()