'''计算机视觉相关的内容'''
import matplotlib.pyplot as plt
from d2l import torch as d2l
from tools import Animator
from torch import nn
import torch
from tools.cache_load_datasets import *
import pandas as pd
import torchvision

# 设置 matplotlib 交互式后端（解决 PyCharm 静态渲染问题）
plt.switch_backend('TkAgg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    plt.rcParams['figure.figsize'] = figsize

def set_title(fig, title):
    "set title for TkAgg window"
    # 新版本 matplotlib：先获取 Tk 组件，再获取顶层窗口
    fig.canvas.get_tk_widget().winfo_toplevel().wm_title(title)

#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    # 在每个小批量函数里面把数据送到 GPU
    # 在最终的训练函数中把 net 送到 GPU
    net.train()
    trainer.zero_grad()
    pred = net(X) 
    # 没毛病，训练的过程中就是必须搞预测。
    # 因为需要通过损失函数反向求导
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练: 明明只有 1 个 GPU, 看看代码具体是怎么写的！"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    # 明确：len(train_iter) 其实就是有多少批！并不是一批里有多少个样本！
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    # 也对，在最终的训练函数里把 net 送到 GPU
    # 而不是在每个小批量里面把 net 送到 GPU
    # 另外，把 net 送到 GPU 是指把参数送到 GPU 吗？答：不是！
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            # 由于 num_batches 是1 个 epoch 里面有多少批
            # ⇒ 因此，在一轮训练中，每完成 num_bacthes // 5 的任务 或者 最后一批，就更新一次可视化
            # 而且，从下面的代码可以看出：train_loss 和 train_acc 的更新频率高于 test_acc
            # 因为 test_acc 只能在训练完一个 epoch 之后才能更新
            # animator.add(x, (y1, y2, y3))
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

#@save
DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = download_extract('banana-detection')
    # 训练就用 bananas_train，推理就用 bananas_val
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    # 下面这个 for loop 很简洁，熟悉这种写法
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里返回的 images 形状是 (C, H, W); 即 (3, H, W)
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
    # targets 张量形状是 (len(list), 5); targets.unsqueeze(1) 形状为 (len(list), 1, 5)

#@save
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
    
#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

