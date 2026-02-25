import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
import tools.ch13 as ch13
from tools import Animator
from datetime import datetime

d2l.set_figsize()
content_img = d2l.Image.open('./data/img/rainier.jpg')
d2l.plt.imshow(content_img)
fig = plt.gcf()
ch13.set_title(fig, '内容图像')

style_img = d2l.Image.open('./data/img/autumn-oak.jpg')
plt.figure()
d2l.plt.imshow(style_img)
fig = plt.gcf()
ch13.set_title(fig, '风格图像')

# 预处理和后处理（rgb_mean、rgb_std 还是 IamgeNet 上预训练模型需要的 mean 和 std）
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    '''预处理：将图片转化为可以训练的张量 (1, channel, h, w)'''
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    '''后处理：将张量转化为可以显示的图片（就是逆着预处理的步骤来的）'''
    # 这里为什么用 img[0] 呢？难道 img 是个列表吗？
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 用 VGG19 抽取图像特征
pretrained_net = torchvision.models.vgg19(pretrained=True)

# 指定【抽取 style 的层】和【抽取 content 的层】
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# 只取标号在 [0, 28+1) 中的网络层
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])

def extract_features(X, content_layers, style_layers):
    '''定义一个从张量抽取特征的函数'''
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        # net[i](X) 也是四维张量 (1, channnel, h, w)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(image_shape, device):
    '''获取【内容特征】（整合转化张量+抽取特征这 2 步）'''
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    '''获取【风格特征】（整合转化张量+抽取特征这 2 步）'''
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

# 定义损失函数（内容损失，风格损失，全变分损失；然后这三者加权求和）
def content_loss(Y_hat, Y):
    '''内容损失（均方误差损失）
    输入是 4 维张量，但是 mean() 直接把结果降为标量！'''
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X):
    '''输入张量就是一张图片的张量 (1, channel, h, w)'''
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    '''标签的 gram 矩阵是事先算好的，因为不会变嘛，算好放在那等着呗！
    输入是 4 维张量，但是 mean() 直接把结果降为标量！'''
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

def tv_loss(Y_hat):
    '''全变分损失函数（用于去除噪点）；
    输入是 4 维张量，但是 mean() 直接把结果降为标量！'''
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    '''内容损失、风格损失、均方误差损失，三者加权求和'''
    # 按传入顺序依次是：预测的图像，预测图像的内容特征列表，预测图像的风格特征列表，
    # 原始图像的内容特征列表，原始图像的格拉姆矩阵列表（不传【原始图像的风格特征列表】是因为计算 loss 不需要它，
    # 【只需要原始图像风格特征列表】对应的【格拉姆矩阵列表】）
    '''分别计算内容损失、风格损失和全变分损失'''
    '''一种思想：如果相对中间过程进行一些拟合的话，那么就要
    1. 对中间过程设置标签 2. 计算损失'''
    # 明确：compute_loss 中【所有 Y_hat】和【对应的 Y】形状都是一样的；并且都是 4 维张量（
    # 因为它们都是同一个网络的同一层出来的！）
    # 这里 content_loss, style_loss, tv_loss 都是【数学上的减法】，所以 Y_hat 和 Y 张量形状一致正好合适！
    # 不像分类问题中的交叉熵损失函数，Y_hat 和 Y 张量形状不一致！
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    # 含义：1. 用 + 拼接列表，形成一个新的列表（+ 不是数值相加！）；
    # 2. 然后 sum() 对这个新列表中所有元素再求和！
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
    # contents_l, styles_l, tv_l 都是列表，列表中的每个元素都是标量数值！

# 初始化合成图像
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
    
def get_inits(X, device, lr, styles_Y):
    '''返回：3 样东西。
    初始化的目标图像（就是照抄原始内容图片中的数据），
    【原始风格图片对应的风格特征列表】对应的【格拉姆矩阵列表】，
    优化器。'''
    # 把训练时要用的数据都搬到 GPU 上
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    # 在这里，优化器知道要对 gen_img.parameters() 进行优化；
    # 但是，损失函数是如何知道要对 gen_img.parameters() 进行求导的呢？
    # 又没有 requires_grad=True
    # 这是因为 gen_img.weight 是 nn.Parameter 类型，
    # 默认 requires_grad=True，无需手动设置就能被 Autograd 追踪！
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

# 训练模型
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    '''X 是一张图片张量（content_X）；
    contents_Y 和 styles_Y 都是一个列表，一个列表元素就是一个张量！
    它们分别表示【原始内容图片的内容特征列表】和【原始风格图片的风格特征列表】'''
    # 这里为什么不需要 style_X 呢？
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    # 检验模型梯度的 requires_grad 属性
    for param in net.parameters():
        print('检验模型梯度的 requires_grad 属性', param.requires_grad)
        break
    print('检验目标图像的参数', X[0, 0, 0, 0])

    for epoch in range(num_epochs):
        trainer.zero_grad()
        # 【内容预测】的张量列表，【风格预测】的张量列表
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0: # 每 10 轮更新一次可视化
            # 右侧显示合成图片
            animator.axes[1].imshow(postprocess(X))
            # 绘制曲线
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X

device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
# content_X 是内容图片，contents_Y 是内容特征列表
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
# 为什么不需要 style_X ？
# 因为 content_X 是用来初始化目标图像的，没有别的作用。
# 而 style_X 在初始化过程中没有用，所以自然不需要 style_X!

starttime = datetime.now() 
print(starttime) # 打印当前时间

# 不训练模型，而是训练图片！
# 也就是不改变模型的参数，不改变图片的参数！
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)

endtime = datetime.now()
print(endtime)
print(endtime-starttime)

plt.show()