'''计算机视觉相关的内容'''
import matplotlib.pyplot as plt
from d2l import torch as d2l
from tools import Animator
from torch import nn
import torch
import tools.cache_load_datasets as cache_load
import pandas as pd
import torchvision
import os

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
cache_load.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = cache_load.download_extract('banana-detection')
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


'''
使用一个数据集还真是【三件套】
1. 读取数据集 read
2. 构造 Dataset class
3. 构造 train_iter 和 test_iter【返回训练 / 测试所需的一个 batch】
但是在构造这 3 者的过程中，可能会用到很多很多辅助函数！
'''
cache_load.DATA_HUB['voc2012'] = (cache_load.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

#@save
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    # 语义分割中，样本和标签都是图片！
    # 不过样本是 jpg 图片，而标签是 png 图片
    # （因为 jpg 图片会压缩，而 png 图片不会压缩。如果标签被压缩，那就很难学习了！）
    # 另外，这里还遵循了 VOC 数据集格式（因为这个格式经典而且好用）
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

# 接下来，我们列举RGB颜色值和类名。
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


#@save
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    # 这里其实就是打一个 RGB 表，将 RGB 三位数视为 256 进制的数！便于快速查找！
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


#@save
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    # 利用前面打的 RGB 表来查找 VOC 类别！
    # 一个 colormap 张量就是一张标签图片！
    # 将颜色维度放在最后面
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

# 预处理数据
#@save
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    # 这里不能 resize，必须 crop
    # resize 会改变像素比例，标签边缘可能失真
    # crop 保持原始分辨率，不拉伸变形，保留细节，同时还可以做数据增强
    # 图片中的【拉伸】是通过【插值（插入像素值）】来实现的！
    '''先获取一个剪裁的窗口，然后利用这个窗口对【特征】和【标签】同时剪裁'''
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# 自定义语义分割数据集类
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 这里的 mean 和 std 都是从 ImageNet 相关模型上抄下来的，
        # 由于我们想用基于 ImageNet 预训练的模型，所以必须这样初始化
        # 以发挥预训练模型的威力！
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        # features 需要做归一化；但是 labels 不需要做归一化，也不能做归一化，因为要将 label 看作 256 进制数 
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        '''要求原图的尺寸大于剪裁后留下的尺寸；如果小于，那就直接舍去！'''
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        # 这里 self.crop_size 就是一个元组，不是元组列表！
        # 返回【特征图】和【类别索引】，居然不是返回【类别图片】
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

#@save
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = cache_load.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter