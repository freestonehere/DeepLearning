# Tips | 一些小任务
## 一、可以学一学【窗口动画】和【读取图片】要怎么写代码
### （一）、从 [38-FCN.py](38-FCN.py) 中学习
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

### （二）、从 [40-seq-model.py](40-seq-model.py) 中学习
- 下面这段代码生成的窗口也是正好够用（正好没有多余的窗口）
```python
import matplotlib.pyplot as plt # 用于画图
import tools.plot as plot


T = 1000  # 总共产生 1000 个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
plot.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
fig = plt.gcf()
ch13.set_title(fig, '第一步：展示人造数据')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
plt.figure() # 新开一块画布，否则这里会覆盖前面已经画好的内容！
plot.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
fig = plt.gcf()
ch13.set_title(fig, '第二步：展示训练过程')

plt.show()
```

<br><br>

### （三）、绘制直方图（详见 [48-MTDataset.py](48-MTDataset.py)）
```python
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

<br><br>

## 二、读入图片的代码 `read`

<br><br>

## 三、读入文本的代码 `read`

<br><br>