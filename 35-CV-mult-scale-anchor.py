import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
from tools.ch13 import *

img = d2l.plt.imread('data/img/catdog.jpg')
h, w = img.shape[:2]
print(f'h, w: {h, w}')

def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
    
plt.figure()
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
fig = plt.gcf()
set_title(fig, '特征图 4x4，面积占比 15%')

plt.figure()
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
fig = plt.gcf()
set_title(fig, '特征图 2x2，面积占比 40%')

plt.figure()
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
fig = plt.gcf()
set_title(fig, '特征图 1x1，面积占比 80%')

plt.show()