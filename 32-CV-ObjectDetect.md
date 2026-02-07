# Object Detection | 目标检测
- 之前讲的都是图片分类
  - 下面开始讲目标检测

## 一、概念理解
### （一）、图片分类、目标检测和实例分割
1. 图像分类任务中，我们假设图像中只有一个主要物体对象，我们只关注如何识别其类别。 
   1. 然而，很多时候图像里有多个我们感兴趣的目标，我们不仅想知道它们的类别，还想得到它们在图像中的具体位置。 
   2. 在计算机视觉里，我们将这类任务称为目标检测（object detection）或目标识别（object recognition）。
2. 目标检测在多个领域中被广泛使用。 
   1. 例如，在**无人驾驶**里，我们需要通过识别拍摄到的视频图像里的车辆、行人、道路和障碍物的位置来规划行进线路。 
   2. **机器人**也常通过该任务来检测感兴趣的目标。
   3. **安防领域**则需要检测异常目标，如歹徒或者炸弹。

### （二）、边界框（bounding box）
一个边界框可以用 4 个数字定义
- 左上 x，左上 y，右下 x，右下 y
- 左上 x，左上 y，宽，高

目标识别的数据集通常比图片分类的数据集小很多。

### （三）、目标检测数据集

- 每行表示一个物体，若一张图里有 n 个物体，则重复 n 行
  - 如：`图片文件名,物体类别,边缘框(x1,y1,x2,y2)`
- `COCO` 数据集：80 类别，330K 图片，1.5M 物体

<br><br><br><br>

## 二、代码 —— 边缘框
```python
#@save
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # torch.stack 和 torch.cat 很像，但是具体内容忘了
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes
```

### （一）、理解代码

<br>

### （二）、`torch.stack` VS `torch.cat`
```python
import torch
# 模拟你的场景：N=2
cx = torch.tensor([2,6])  # [2,]
cy = torch.tensor([3,7])  # [2,]
w = torch.tensor([2,2])   # [2,]
h = torch.tensor([2,2])   # [2,]

# 1. 用torch.stack（正确）：新增最后一维（axis=-1），得到[2,4]
stacked = torch.stack((cx, cy, w, h), axis=-1)
print("stack结果：\n", stacked)
print("stack形状：", stacked.shape)  # torch.Size([2, 4])
# 输出：
# stack结果：
#  tensor([[2, 3, 2, 2],
#          [6, 7, 2, 2]])

# 2. 用torch.cat（错误，不符合需求）：无新增维度，只能在已有维度拼接
cat = torch.cat((cx, cy, w, h), axis=-1)  # axis=-1即第0维（1维张量的唯一维度）
print("\ncat结果：", cat)
print("cat形状：", cat.shape)  # torch.Size([8])
# 输出：cat结果： tensor([2, 6, 3, 7, 2, 2, 2, 2]) → 变成1维的8个元素，完全不是你要的[N,4]
```
- 要新增维度就用 `stack`
- 只扩展已有维度就用 `cat`

<br><br><br><br>

## 三、代码 —— 目标检测的数据集

```python
# 运行效果
read 1000 training examples
read 100 validation examples
batch[0].shape: torch.Size([32, 3, 256, 256]), batch[1].shape: torch.Size([32, 1, 5])
```

1. 明确**边缘框**到底是用**像素**表示，还是用**比例**表示

### （一）、理解代码

<br>

### （二）、`pandas` 包
#### 1、`pandas.read_csv` 函数
```python
pandas.read_csv(
    filepath_or_buffer, *, sep=<no_default>, 
    delimiter=None, header='infer', 
    names=<no_default>, index_col=None, usecols=None, 
    dtype=None, engine=None, converters=None, 
    true_values=None, false_values=None, 
    skipinitialspace=False, skiprows=None, 
    skipfooter=0, nrows=None, na_values=None, 
    keep_default_na=True, na_filter=True, 
    skip_blank_lines=True, parse_dates=None, 
    date_format=None, dayfirst=False, 
    cache_dates=True, iterator=False, chunksize=None, 
    compression='infer', thousands=None, decimal='.', 
    lineterminator=None, quotechar='"', quoting=0, 
    doublequote=True, escapechar=None, comment=None, 
    encoding=None, encoding_errors='strict', 
    dialect=None, on_bad_lines='error', 
    low_memory=True, memory_map=False, 
    float_precision=None, storage_options=None, 
    dtype_backend=<no_default>)
```

[官网 doc - pandas.read_csv 函数](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv)

- **Returns**: `DataFrame` or `TextFileReader`
  - A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.

<br>

#### 2、因为 `read_csv` 返回 `DataFrame`，因此学习 `DataFrame.set_index`
```python
DataFrame.set_index(
    keys, *, drop=True, append=False, inplace=False, verify_integrity=<no_default>)
```

[官网 doc - pandas.DataFrame.set_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html#pandas.DataFrame.set_index)

```python
# 定义 df
>>> df = pd.DataFrame(
...     {
...         "month": [1, 4, 7, 10],
...         "year": [2012, 2014, 2013, 2014],
...         "sale": [55, 40, 84, 31],
...     }
... )
>>> df
   month  year  sale
0      1  2012    55
1      4  2014    40
2      7  2013    84
3     10  2014    31


# 单个字符串作 keys
>>> df.set_index("month")
       year  sale
month
1      2012    55
4      2014    40
7      2013    84
10     2014    31


# 字符串列表作 keys
>>> df.set_index(["year", "month"])
            sale
year  month
2012  1     55
2014  4     40
2013  7     84
2014  10    31
```
<br>

#### 3、`DataFrame.iterrows`
[官网 doc - pandas.DataFrame.iterrows](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows)

- `iterrows()` 指按行迭代

<br>

### （三）、`torch` 包
#### 1、`torch.Tensor.unsqueeze()` 解缩函数 | 在 `dim=dim` 处增加一个一维的维度
[官网 doc - torch.Tensor.squeeze](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html#torch.Tensor.unsqueeze)

```python
torch.unsqueeze(input, dim) 
```
- `unsqueeze`：解缩

1. Returns a new tensor with a dimension of size one inserted at the specified position.
2. `torch.unsqueeze` 的核心作用是**升维**：在张量的指定位置插入一个**长度为 1** 的新维度，返回的新张量与原张量共享数据。
   1. **操作本质**：不改变数据本身，只改变张量的**形状（Shape）**。例如，将形状为 `(4,)` 的一维张量，变为 `(1, 4)` 或 `(4, 1)` 的二维张量。
   2. **维度位置 `dim`**：
      1. 正数：从左往右数，在指定索引处插入维度。如 `dim=0` 表示最前面。
      2. 负数：从右往左数，在指定索引处插入维度。如 `dim=-1` 表示最后面。
   3. **数据共享**：返回的张量是原张量的**视图（View）**，而非副本。修改新张量的数据，原张量也会改变，因此操作效率很高。

```python
>>> x = torch.tensor([1, 2, 3, 4])
# 原始张量形状是 (4)
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
# 上面张量形状是 (1, 4)
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
# 上面张量是 (4, 1)
```
<br>

#### 2、复习一下 `DataLoader` （ `torch.utils.data.DataLoader` ）
[官网 doc - torch.utils.data 首页](https://docs.pytorch.org/docs/stable/data.html#module-torch.utils.data)

##### (1)、大纲：`data` 包的核心
- 直接照搬 [08-LinearRegression.md](08-LinearRegression.md) 中的内容
1. **Pytorch `data` 包的核心**是 `torch.utils.data.Dataloader` class 。它表示一个基于数据集的 Python 可迭代对象，支持
   1. 映射式和可迭代式数据集、
   2. 自定义数据加载顺序、
   3. 自动批处理、
   4. 单进程和多进程数据加载、
   5. 自动内存锁定
   6. 这些选项是通过 `DataLoader` 的构造函数参数来配置的
2. class `DataLoader` constructor
    ```python
    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
    ```
    1. argument `dataset`
       1. 该参数决定 dataset types，一共有 2 种 dataset types
          1. map-style datasets
          2. iterable-style datasets
    2. argument `sampler` （取样器）
       1. For **iterable-style datasets**, data loading order is entirely controlled by the user-defined iterable. 
       2. This allows easier implementations of chunk-reading and dynamic batch size (e.g., by yielding a batched sample at each time).
       3. The rest of this section concerns the case with **map-style datasets**
       4. `torch.utils.data.Sampler` classes are used to specify the sequence of indices/keys used in data loading.（用于指定加载数据时索引 / 键的序列）
          1. They represent iterable objects over the indices to datasets. 
          2. E.g., in the common case with stochastic gradient decent (SGD), 
             1. a `Sampler` could randomly permute a list of indices（随机排列索引列表）
             2. and yield each one at a time, （每次生成一个索引）
             3. or yield a small number of them for mini-batch SGD. （或者为小批量 SGD 生成少量索引）
    3. argument `shuffle`
       1. A sequential or shuffled sampler will be automatically constructed based on the `shuffle` argument to a `DataLoader`. 
       2. Alternatively, users may use the `sampler` argument to specify **a custom Sampler object** that at each time yields the next index/key to fetch
    4. argument `batch_sampler` （取样器）
       1. A custom `Sampler` that yields a list of batch indices at a time can be passed as the `batch_sampler` argument.
       2. 可通过 `batch_sampler` 参数传递一个自定义取样器对象 *custom sampler object*
    5. argument `batch_size` 和 `drop_last`
       1. Automatic batching can also be enabled via `batch_size` and `drop_last` arguments. See the next section for more details on this.
       2. 设置自动批处理的参数：`batch_size`, `drop_last`

<br>

##### (2)、本节的注意事项
1. `DataLoader` 中的 **第一个参数 `dataset`** 要求的是：**数据集类型的对象（ `class` ）**
   1. 而且，我目前知道**数据集类型的对象**有 2 种
      1. 一种是 `class torch.utils.data.TensorDataset(*Tensor)`
         1. `TensorDataset` 详情见 [08-LinearRegression.md](08-LinearRegression.md)
         2. 将多个张量 `*Tensor` 转化为 **数据集对象**
         3. 不过以后应该比较少见！
      2. 一种就是**继承并重写** `class torch.utils.data.Dataset` **基类**
         1. 也就是本节的内容
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
<br>

##### (3)、重新理解代码输出结果
```python
# 运行效果
read 1000 training examples
read 100 validation examples
batch[0].shape: torch.Size([32, 3, 256, 256]), batch[1].shape: torch.Size([32, 1, 5])
```

1. `1000 training examples` 和 `100 validation examples` 
   1. 是初始化数据集对象 `class BananasDataset` 的时候调用的
   2. `1000` 和 `100` 的来源：它们都是是原始数据集的大小！
2. 为什么一个小批量 `batch` 还会有 `batch[0]` 和 `batch[1]` ？
   1. 因为最开始 `read_data_bananas()` 函数返回了 **2 个列表**：一个 `images`；一个 `targets`
   2. `images` 的形状是 `(len(list), C, h, w)`；`targets` 的形状是 `(len(list), 1, 5)`
   3. `batch[0]` 就是 `images` 中的**一小批**。`batch[0]` 的形状是 `(batch_size, C, h, w)`
   4. `batch[1]` 就是 `targets` 中的**一小批**。`batch[1]` 的形状是 `(batch_size, 1, 5)`
3. 有一个比较致命的问题是：这里把**所有数据直接读到内存**，
   1. 这是直接读到内存的具体代码
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
   2. 如果数据集过大可能内存会爆掉
<br>

#### 3、为什么要用 `unsqueeze` ？
1. `unsqueeze(1)` 的具体作用
   1. 对形状为 `(样本数, 5)` 的张量执行 `unsqueeze(1)`，会在 **第 1 维** 插入一个维度，把形状变成 `(样本数, 1, 5)`：
      1. 第 0 维：样本数
      2. 第 1 维：目标数（这里固定为 1，因为每张图只有 1 个香蕉）
      3. 第 2 维：每个目标的 5 个标签值
   2. 后来才发现 `unsqueeze` 函数其实是炫技操作
      1. 本来目标数这个维度是不能少的，但是 `read_data_bananas` 最开始的时候没有加这个维度，所以只能在后续再插入一个维度
      2. 单目标时用 `unsqueeze(1)` 手动加 “目标数 = 1” 的维度，
      3. 多目标时直接生成 `(目标数, 5)` 的张量，无需额外维度操作
<br>

#### 4、`torch.Tensor.permute` | 重新排序函数
[官网 doc - torch.Tensor.permute](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.permute.html#torch.Tensor.permute)

```python
torch.permute(input, dims)
```
1. Returns a view of the original tensor input with its dimensions permuted. （对维度重新排序）
2. 例子代码
    ```python
    >>> x = torch.randn(2, 3, 5)
    >>> x.size()
    torch.Size([2, 3, 5])
    >>> torch.permute(x, (2, 0, 1)).size()
    torch.Size([5, 2, 3])
    ```
<br>

### （四）、展示图片
```python
'''展示图片'''
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```
1. `permute` 函数把通道数放在最后面；
2. 同时 `除以 255` 对 ~~高宽~~ 像素值作归一化操作
3. **切片**：包含起点，不包含终点！
4. `d2l.show_bboxes`：d2l 库封装的绘制边界框函数，在指定图片上画检测框：
   1. `ax`：要绘制框的图片坐标轴；
   2. `[label[0][1:5] * edge_size]`：检测框坐标（关键拆解）：
      1. `label[0]`：取出该样本的唯一目标标签（形状从(1,5)变为(5,)）；
      2. `label[0][1:5]`：截取坐标部分（跳过第一个 “类别” 值，取x1, y1, x2, y2）；
      3. `* edge_size`：还原像素坐标 —— 因为标签坐标之前被归一化到 `[0,1]`（除以 256），乘以 edge_size（256）才能匹配图片的像素尺寸；
      4. 外层 `[]`：show_bboxes 要求传入边界框列表（兼容多目标场景）；
   3. `colors=['w']`：设置检测框为白色（ 'w'=white ），确保在图片上清晰可见
