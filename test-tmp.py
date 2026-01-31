'''代码验证'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 复用你的代码逻辑
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(
    root='data/FashionMNIST', train=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='data/FashionMNIST', train=False, transform=transform
)
train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 1. 查看单个样本的形状
single_img, single_label = train_dataset[0]
print("单个样本图像张量形状：", single_img.shape)  
# 输出: torch.Size([1, 28, 28])
print("单个样本标签类型/值：", type(single_label), single_label)  
# 输出: <class 'int'> 9（或其他整数）
print("标签转为张量后的形状：", torch.tensor(single_label).shape)  
# 输出: torch.Size([])（0维）
# 标签转换为张量后，就是单个实数，自然是 0 维

# 2. 查看批次数据的形状
for X, y in train_iter:
    print("批次图像张量形状：", X.shape)  
    # 输出: torch.Size([256, 1, 28, 28])
    print("批次标签张量形状：", y.shape)  
    # 输出: torch.Size([256])
    # 由于 W.shape = (784, 10)
    # ⇒ 因此，W.shape[0] = 784
    print("reshape 后的批次图像张量形状：", X.reshape(-1, 784).size())
    break  # 只看第一个批次即可 