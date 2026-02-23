# Semantic Segmentation Dataset | 语义分割数据（不是语义分割算法！）
- 因为直观上感觉
  - 你这样一个像素一个像素来
  - 好像没有什么特征，没有办法训练！
- 所以，这里先不介绍 **语义分割** 算法，而是先介绍 **语义分割** 数据集

## 一、概念理解

<br><br><br><br>

## 二、代码实现
```python
'''这是代码运行结果'''

y[105:115, 130:140], VOC_CLASSES[1]: (tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]), 'aeroplane')
read 1114 examples
read 1078 examples
X.shape: torch.Size([64, 3, 320, 480])
Y.shape: torch.Size([64, 320, 480])
```
### （一）、理解代码
1. 一些细节
   1. 如果想把图片放到一个 `batch` 中，那么 **图片大小必须相同**
   2. `resize` 和 `crop` 的区别
      1. `resize` 会改变像素比例，标签边缘可能失真
      2. `crop` 保持原始分辨率，不拉伸变形，保留细节，同时还可以做数据增强
   3. 图片中的 **拉伸** 是通过 **插值（插入像素值）** 来实现的！

<br>

### （二）、构造数据集
#### 1、首先照抄 [32-CV-ObjectDetect.md](32-CV-ObjectDetect.md) 中的内容
1. `DataLoader` 中的 **第一个参数 `dataset`** 要求的是：**数据集类型的对象（ `class` ）**
   1. 而且，我目前知道**数据集类型的对象**有 2 种
      1. 一种是 `class torch.utils.data.TensorDataset(*Tensor)`
         1. `TensorDataset` 详情见 [08-LinearRegression.md](08-LinearRegression.md)
         2. 将多个张量 `*Tensor` 转化为 **数据集对象**
         3. 不过以后应该比较少见！
      2. 一种就是**继承并重写** `class torch.utils.data.Dataset` **基类**
         1. 也就是本节的内容
         2. 继承 `torch.utils.data.Dataset` 基类的话，**必须重写这 3 个函数**：`__init__`, `__getitem__`, `__len__`
         3. 如果有需要，还可以 **添加其他辅助函数！**
        ```python
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
        ```

```
最开始，定义 read_data_bananas 函数，
读取原始数据集，返回列表（列表包含【所有】样本）
↓
class BananaDataset 利用 read_data_bananas 函数返回的列表进行索引，
构造数据集对象（数据集对象的索引方法每次返回【一个】样本）
↓
DataLoader 利用数据集对象 class BananaDataset，
构造迭代器（迭代器每次返回【一小批】样本）
```


