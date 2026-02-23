import os
import torch
import torchvision
from d2l import torch as d2l
import tools.cache_load_datasets as cache_load # 用于下载数据集
import matplotlib.pyplot as plt # 用于画图
import tools.ch13 as ch13 # 用于改变绘图窗口的标题

#@save
cache_load.DATA_HUB['voc2012'] = (
                        cache_load.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = cache_load.download_extract('voc2012', 'VOCdevkit/VOC2012')

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

train_features, train_labels = read_voc_images(voc_dir, True)

# 下面我们绘制前 5 个输入图像及其标签。
# 在标签图像中，白色和黑色分别表示边框和背景，而其他颜色则对应不同的类别。
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n)
fig = plt.gcf()
ch13.set_title(fig, '前 5 个输入图像和标签')


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

# 查看第一张样本图像
y = voc_label_indices(train_labels[0], voc_colormap2label())
print(f'y[105:115, 130:140], VOC_CLASSES[1]: {y[105:115, 130:140], VOC_CLASSES[1]}')


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

imgs = []
for _ in range(n):
    # imgs 列表的基本元素是【二元组】还是【colormap】？
    # voc_rand_crop 函数返回的确实是【二元组】
    # 但是使用 += 会拆分二元组，应使用 append 保留 (特征，标签) 的配对结构
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
# 这里冒号是切片的意思，类比 [start: end: stride]
# 加了逗号不叫【切片】[:, :, 2]；加了逗号叫【多维索引】
# 更广泛地说，【切片】可以叫【一维索引】！
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
fig = plt.gcf()
ch13.set_title(fig, '查看预处理（随机剪裁）后的图像和标签')

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
    
# 读取数据集
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)

batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(f'X.shape: {X.shape}')
    print(f'Y.shape: {Y.shape}')
    break

# 整合所有组件
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

plt.show()