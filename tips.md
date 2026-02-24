# Tips | 一些小任务
## 一、可以学一学【窗口动画】和【读取图片】要怎么写代码
- 未删减的代码在 [38-FCN.py](38-FCN.py) 中
- 下面这段代码生成的窗口正好够用（正好没有多余的窗口）
```python
from d2l import torch as d2l
import tools.ch13 as ch13
import matplotlib.pyplot as plt # 用于画图

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

# 打印几张图片，看看效果
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]

# plt.figure() 如果这里加上 plt.figure() 那就会多出一个空白窗口
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
fig = plt.gcf()
ch13.set_title(fig, '打印几张图片，看看效果')

plt.show()
```

<br><br>