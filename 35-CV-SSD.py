import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
from tools import Animator # 导入动画类
from tools.ch13 import * # 导入 load_data_bananas() 函数及其依赖

# 反正一个是分类问题；一个是回归问题。但具体是哪一个，我并不清楚！
# 到底是在哪一步只需锚框数量，不需具体锚框内容？

# 类别预测层 class_predictor
def cls_predictor(num_inputs, num_anchors, num_classes):
    '''用通道数存储预测结果（这里是【类别】）'''
    # padding=1 & kernel_size=3 保证输入图 & 输出图尺寸完全一致！
    # num_classes + 1 = 物体类别 + 1 个背景类
    # num_anchors 是 每个像素画 num_anchors 个锚框；但为什么非得是一个像素呢？这个不理解！
    # num_anchors * (num_classes + 1) 意味着 对每个锚框，预测它是什么类别
    '''注意：这里【类别预测】不需要具体锚框坐标，只需要知道【锚框数量+类别数量】即可！
    也就是说：【前向计算】的时候不需要知道具体锚框，但是【反向计算】的时候必须知道具体锚框。
    其实【边缘框预测】也是一样的！'''
    '''可以这样理解：前向计算只是为了得到多尺度的特征图+多尺度的锚框，反向计算才是真正的优化！'''
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

# 边界预测层 bbox_predictor
def bbox_predictor(num_inputs, num_anchors):
    '''用通道数存储预测结果（这里是【边界框】）'''
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    return block(x)

# PyTorch 张量形状 (batch_size, channels, h, w)
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(f'Y1.shape, Y2.shape: {Y1.shape, Y2.shape}')
# 输出 Y1.shape, Y2.shape: (torch.Size([2, 55, 20, 20]), torch.Size([2, 33, 10, 10]))

# 方便连接
def flatten_pred(pred):
    # 对维度重新排序，把通道数放在最后。
    # 使得每个像素的所有通道相邻在一起，便于连接。
    # 保留批量维度，其余展平！
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    '''将预测展平后连接，一律返回二维张量'''
    # 在哪个维度上 cat，哪个维度的维数就必须增加！其他维度的维数就不能改变！
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

print(f'concat_preds([Y1, Y2]).shape: {concat_preds([Y1, Y2]).shape}')
# 输出 concat_preds([Y1, Y2]).shape: torch.Size([2, 25300])

# 宽高减半块（同时指定输出通道数）
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        # 只在第一个卷积层改变通道数，
        # down_sample_blk 块中其他所有层（包括 BN 层）都保持通道数不变
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

tmp = forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
print(f'forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape: {tmp}')
# 输出 forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape: torch.Size([2, 10, 10, 10])

# 基本网络块
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

tmp = forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
print(f'forward(torch.zeros((2, 3, 256, 256)), base_net()).shape: {tmp}')
# 输出 forward(torch.zeros((2, 3, 256, 256)), base_net()).shape: torch.Size([2, 64, 32, 32])
# 基础网络块 base_net 就是由宽高减半块 down_sample_blk 叠加而成的！

# 完整的模型（5 个 stage）
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
    # 返回：输出特征图，锚框，类别预测，边缘框预测
    # 而且，返回的锚框就是开始的时候画的锚框（没有经过 NMS 抑制！）
    # anchors 张量形状 (1, num_anchors_per_pixel * num_piixels, 4)
    
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# 定义完整 TinySSD 模型
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句 self.blk_i = get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))
        '''从这段代码可以看出：一个小型 SSD 共有 5 个 blk；
        每个 blk[i] 都有：基础网络 → 宽高减半块 → 最大池化层。
        然后每个 blk[i] 都要进行【类别预测】和【边缘框预测】。'''

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i) 即访问 self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
            '''用输出的特征图覆盖原图；每次的锚框要保留；
            每次的类别预测要保留；每次的边缘框预测要保留'''

        anchors = torch.cat(anchors, dim=1)
        # ⇒ anchors 张量形状 (1, 一个 SSD 模型 5 个 blk 输出的全部锚框数量之和, 4)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        # 因为 concat_preds 函数一律返回二维张量，为了可读性，这里必须 reshape
        # reshape 完以后，cls_preds (batch_size, num_pixels * num_anchors_per_pixel, num_classes+1)
        # 现在，我好像知道，为什么前面用 num_anchors_per_pixel 了！因为 h, w 都隐藏在卷积层输出中了！
        # 也就是对所有锚框，预测这个锚框到底是什么类
        # 注意：这里是多尺度、多个特征图上经过 NMS 抑制后的所有锚框（并不是只有一个特征图经过 NMS 的锚框！）
        bbox_preds = concat_preds(bbox_preds)
        # 这里的 bbox_preds 为什么不 reshape 呢？
        # 直接就是 (batch_size, num_anchors_per_pixel * 4 * num_pixels)
        # 其实可以直接 reshape，但是没必要！
        return anchors, cls_preds, bbox_preds
    
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)

# 训练模型：这里就和之前思路一致了，再复习一下！
batch_size = 32
train_iter, _ = load_data_bananas(batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评价函数
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    '''【类别预测】无需屏蔽负样本；【边缘框预测】必须屏蔽负样本。'''
    # 输入张量形状如下
    # cls_preds (batch_size, num_pixels * num_anchors_per_pixel, num_classes+1)
    # bbox_preds (batch_size, num_anchors_per_pixel * 4 * num_pixels)
    # 在交叉熵损失函数中，X 是二维张量，Y 是一维张量；并且【列数 / 最后一维的维数】一定是【类别数】
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # 目标检测中，大部分锚框是 “负锚框”（没有匹配到真实目标），
    # 这些锚框不需要计算边界框损失（否则会导致损失被负样本主导，模型学不到有效信息）。
    # bbox_masks 和 bbox_preds 张量形状完全一致
    # 注意：损失函数的 mean() 和 这里的 mean() 会使该维度的【张量】退化为【标量】
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
    # 输出类别预测 + 锚框预测的和，张量形状为 (batch_size)，因为之前 keepdim=False

def cls_eval(cls_preds, cls_labels):
    '''明确：评价函数只是辅助作用，帮助查看模型训练效果！'''
    # 由于类别预测结果放在最后一维，argmax 需要指定最后一维。
    # cls_preds 张量形状 (batch_size, num_anchors * num_pixels, num_classes+1)
    # 这里的 sum() 也使得最后一维退化为标量
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    '''这里的评价函数同样也是辅助作用，不过明确：锚框坐标一定是归一化的！
    同时，也正是因为这里 sum()，所以后面才要除以 .numel()'''
    # 和训练时一样，评价锚框预测也不能带负样本！
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 训练模型的 loop
num_epochs, timer = 20, d2l.Timer()
animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad() # 梯度清零是为了防止参数更新时发生错误
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        # （多尺度锚框的本质是多尺度特征值）
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward() # l 张量形状是 (batch_size)
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))

print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

# 预测目标
X = torchvision.io.read_image('data/img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

# 使用下面的 multibox_detection 函数，我们可以根据锚框及其预测偏移量得到预测边界框。
# 然后，通过非极大值抑制来移除相似的预测边界框。
# NMS 抑制只有在预测的时候才会用到；在训练的时候根本不会用到！
def predict(X):
    net.eval()
    # 张量形状如下
    # anchors (batch_size, 一个 SSD 模型 5 个 blk 输出的所有锚框数量之和, 4)
    # cls_preds (batch_size, num_anchors_per_pixel * num_pixels, num_classes+1)
    # bbox_preds (batch_size, num_anchors_per_pixel * 4 * num_pixels)
    # 事实上，由于这里仅仅输入一张图片，所以这里 batch_size = 1！
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # 在 multibox_detection 函数中，cls_probs 张量形状是 (batch_size, num_classes+1, 锚框总数)
    # 只有这个【类别】在前，其他都是【类别】在后！
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    # output 张量形状 (1, 锚框总数, 6)，其中 6 是（类别 ID + 置信度 + 边界框坐标）
    # output[0] 是消去 batch 维度！(锚框总数, 6)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1] # 把圈到背景的锚框全部移出
    return output[0, idx]

output = predict(X)
# output 是 (有效框数, 6)

# 最后，我们筛选所有置信度不低于 0.9 的边界框，做为最终输出。
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

plt.figure()
display(img, output.cpu(), threshold=0.9)

plt.show()