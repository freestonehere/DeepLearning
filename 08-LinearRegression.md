## 一、代码讲解（从零实现，不使用深度学习框架）
```python
import random
import torch

'''人工合成数据集: y = Xw + b + 噪声（线性数据集）'''
def synthetic_data(w, b, num_examples):  # @save
    # 关键是明确数据形状：(第二维, 第一维)
    # 每个样本是长度为 len(w) 的向量，
    # 每个样本是向量，维数是 (len(w))
    # 全部样本 X 是矩阵，维数是 (num_examples, len(w))
    X = torch.normal(0, 1, (num_examples, len(w)))
    # w 是向量，维数是 (len(w))
    y = torch.matmul(X, w) + b
    # 给 y 添加随机扰动
    y += torch.normal(0, 0.01, y.shape)
    # 这里 -1 就是总的行数；1 表示只有 1 列
    return X, y.reshape((-1, 1))

# 代码里的权重 w 包含了常数 b
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 虽然本代码是回归任务
# 但这里借用了分类任务中的说法
# 但也务必明确：features 是 x 轴，labels 是 y 轴
features, labels = synthetic_data(true_w, true_b, 1000)


'''读取训练数据（小批次数据迭代器）'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 以张量的形式读取数据
        # 1. 如果数据没到头，那就规规矩矩读取 batch_size 个
        # 2. 如果数据到头了，那就读到最后一个数据为止
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]) # 数组减法 / 加法
        yield features[batch_indices], labels[batch_indices]
        # 这里的索引是张量，其实没问题
        # 正是因为如此，所以才能批量输出数据
        # 所以，这里输出的数据也是张量类型，
        # X 维数是 (num_examples, len(w)) ⇒ features 维数是 (len(batch_indices), len(w))
        # 同理，labels 维数是 (len(batch_indices), 1)

# 打印每一批次的训练数据张量
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


'''定义并初始化待求参数'''
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


'''定义模型：线性回归模型'''
def linreg(X, w, b):  # @save
    return torch.matmul(X, w) + b


'''定义训练过程中的损失函数：均方损失'''
# 虽然叫均方损失，但是代码里的定义并没有体现平均
# 也就是说：代码里面没有除以 batch_size
# 但是经过数学推导，发现：在梯度下降求偏导那里再除以 batch_size（体现平均），效果也一样！
# 至于数学推导，下面我也写了
def squared_loss(y_hat, y):  # @save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


'''定义梯度下降算法：小批量随机梯度下降'''
'''明确：因为待求参数未知，所以要用梯度下降算法求解待求参数'''
'''梯度：函数增长最快的方向'''
def sgd(params, lr, batch_size):  # @save
    # 参数更新是手动操作，不需要将这一步纳入梯度追踪计算图
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() 
            # 将梯度原地清零，因为 pytorch 梯度默认累加。如果不清零，后面梯度下降要出错！
# 注意：明确 para.grad 只是用于存储梯度，并不是计算梯度
# 只有在调用 loss.backward() 以后，才会计算梯度，然后将梯度存到 w.grad 和 b.grad 中


'''训练过程：先定义超参数，然后开始训练'''
# 正向计算结果，反向传播梯度
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X 和 y 的小批量损失
        # 因为 l 形状是 (batch_size, 1)，而不是一个标量。
        # 因此，将 l 中的所有元素被加到一起，
        # 并以此计算关于 [w,b] 的梯度
        # Pytorch 中，backward 的输入只能是标量。
        # 因此，需要把张量 l 求和，转化为标量
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    # 本轮训练结束后，计算整个训练集上的损失
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```
### （一）、理解代码
#### 1、张量的形状
1. 要理解代码
   1. 比较重要的点是：明确张量的形状

#### 2、理解梯度下降算法
- **背景**：
  - 因为参数未知，所以才需要用梯度下降算法来求解未知参数
1. 梯度是函数**增长**最快的方向
   1. 明确：就是 **增长**
   2. 不是变化！
2. 沿着上一个估计值使得损失函数增长最快的方向的反向变化
   1. 也就是：沿着上一个估计值**使得损失函数减小最快的方向**变化
   2. 具体求偏导的过程，那就要看数学表达式了
3. 另外，不必死磕：
   1. 机器学习、深度学习处理的问题本身就没有数学最优解
   2. 只是数值模拟最优解
   3. 在数学上肯定不会严格最优！
   4. 另外，由于**未知参数本身**其实**不是关于**估计值 $\bf{w_{t-1}}$ **的函数**，所以求导的时候可以看作常数
   5. **注意：** 深度学习代码中不必关注数学（起码不必关注这里的求导），因为 `w.grad` 是 `loss.backward()` 产生的，`w.grad` 本身就是 $\partial \text{loss} \over \partial \bf{w_{t-1}}$
4. 在数学上，对列向量求导变成行向量，这是对的
   1. 但是 Pytorch 中，为了方便，对列向量求导，其**形状**依然和之前**保持一致**！
<br><br>

- **小批次梯度下降的小批次体现在哪里？**
  - 虽然确实遍历了每一个元素
  - 但是并不是计算每一个元素的梯度
  - 而是 `l.sum().backward()`
<br><br>

- **再次理解小批量梯度下降算法**
```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```
1. 下面严格按照训练过程的代码推导
   1. 单个数据的损失是 $$\frac{1}{2} (y - \hat{y})^2$$
   2. 1 个 `l` 是一批，一批有 `batch_size` 个
   3. 因此，`l.sum().backward()` 就是 $$\frac{1}{\text{batch\underline{ }size}} ~ ~ \sum^\text{batch\underline{ }size}_{i = 1} ~ ~ \{ ~ \partial [\frac{1}{2} (y_i - \hat{y})^2] ~/~ \partial w_{t-1} ~ \} = \partial [\frac{1}{2} (y_i - \hat{y})^2] ~/~ \partial w_{t-1}$$
   
   $$\frac{1}{\text{batch\underline{ }size}} ~ ~ \sum^\text{batch\underline{ }size}_{i = 1} ~ ~ \{ ~ \partial [\text{loss\underline{ }i}] ~/~ \partial w_{t-1} ~ \} = \partial [\text{loss\underline{ }i}] ~/~ \partial w_{t-1}$$

   4. 实际上是这一批次梯度之和的平均，仍然和单个训练数据的梯度具有相同的含义（效力 / 数量级）
2. 均方损失函数实际上是：针对 **这一批次** 的损失之和的 **平均**
3. `l.sum().backward()` 中 `sum()` **必不可少的 2 个原因**
   1. `Pytorch` 的语法限制
   2. 梯度

#### 3、为什么有 `with torch.no_grad` ？
1. **梯度追踪计算图**其实不是特别理解
2. 以后结合其他代码再看一看吧！

#### 4、$\bf{w}$ 不是已经加了常数 $\bf{b}$ 吗（因为 w 维数是 `(2, 1)` ）？为什么在回归模型函数里面还要加常数 b ？ 
1. 不对！我理解错了
2. 线性模型是 $$y = x_1 w_1 + x_2 w_2 + b = Xw + b$$
   1. w 维数是 `(2, 1)` 代表 w 向量是 2 个参数
   2. 并没有将常数 b 加入到 w 中
3. 另外，还发现
   1. Pytorch 假设所有向量都是**行向量**，维数 = $(n)$
   2. 如果是**列向量**，那么维数就是 $(n, 1)$

#### 5、机器学习 & 深度学习的大致流程
1. 数据
2. 模型
3. 损失函数
4. 求解参数：梯度下降算法
5. 训练模型

#### 6、为什么机器学习 & 深度学习这么强调矩阵和线性代数？
1. 也许是因为**输入数据**和**处理数据**都是以**批次**和**矩阵**为单位进行的？

### （二）、`torch` 包中的 ~~函数~~ 内容
#### 1、`matmul()` 函数
```python
torch.matmul(input, other, *, out=None) → Tensor
```
[官网 doc - torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html#torch-matmul)

1. scalar：标量（一维的）
   1. 理解标量这个概念：标量（点，一维） → 向量（二维，线） → 矩阵（三维，面） → 张量（多维）
   2. 这里维数和生活直观并是一样的！（**看下面的英文表述**）

    ||Pytorch（数学）|生活直观|
    |:--|:------|:------|
    |一维|向量，`(a, b)`|直线|
    |二维|矩阵，`[(a, b), (c, d)]`|平面|
    |三维及多维|张量，更多维数||

##### (1)、广播机制
1. If **both tensors are 1-dimensional**, the **dot product (scalar，标量)** is returned.
2. If **both arguments are 2-dimensional**, the **matrix-matrix product（矩阵 乘 矩阵）** is returned
3. If the first argument is **1-dimensional** and the second argument is **2-dimensional**, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
4. If the first argument is **2-dimensional** and the second argument is **1-dimensional**, the matrix-vector product is returned
5. If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), 
   1. then a **batched matrix multiply** is returned.（返回批处理矩阵乘法）
   2. If the first argument is 1-dimensional, a 1 is **prepended（在前面加）** to its dimension for the purpose of the batched matrix multiply and removed after. 
   3. If the second argument is 1-dimensional, a 1 is **appended（在后面加）** to its dimension for the purpose of the batched matrix multiply and removed after.
6. The **first N-2** dimensions of each argument, the batch dimensions, are broadcast (and thus must be broadcastable). （前面的 N - 2 个维数，批维数，被广播）
7. The **last 2,** the matrix dimensions, are handled as in the matrix-matrix product.（最后 2 个维数是矩阵维数，按照矩阵乘法来处理）
8. For example, if input is a $( j × 1 × n × m )$ tensor and other is a $( k × m × p )$ tensor, 
   1. the **batch dimensions** are $( j × 1 )$ and $( k )$ , 
   2. and the **matrix dimensions** are $( n × m )$ and $( m × p )$ .
      1. 反应过来矩阵相乘的知识：$(n × m) × (m × p) = (n × p)$
      2. 不要**乱用广播机制** ！
   3. out will be a $( j × k × n × p)$ tensor.

<br><br><br><br><br>

## 二、代码讲解（简洁实现，使用 Pytorch 的 `nn` 模块）
```python
import torch
from torch.utils import data
from d2l import torch as d2l


'''定义并初始化待求参数'''
true_w = torch.tensor([2, -3.4])
true_b = 4.2


'''人工合成数据集'''
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


'''构造一个小批量数据迭代器'''
def load_array(data_arrays, batch_size, is_train=True):  #@save
    dataset = data.TensorDataset(*data_arrays)
    # 【解包操作 * 】：
    # 把 data_arrays 这个可迭代对象（列表 / 元组）中的元素逐个传入 TensorDataset
    # 例如data_arrays = [X,y] 等价于 TensorDataset(X, y)，如果少了*，
    # 会因传入 “列表” 而非 “多个张量” 报错。
    # 【TensorDataset 作用】：
    # 将多个张量按 “样本索引” 绑定成一个数据集。
    # 比如 data_arrays=(X, y) 时，dataset[i] 会返回 (X[i], y[i])，实现特征和标签的一一对应。
    # ⇒ dataset 实际上成为一个 iterable-style dataset
    # batch_size 用于配置自动批处理
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)


'''读取并打印第一个小批量样本'''
next(iter(data_iter))
# 这行代码是 PyTorch / Python 中快速获取数据加载器首个批次数据的经典写法，
# 核心作用是从DataLoader（即 data_iter ）中取出第一个完整的小批量数据，
# 常用于调试阶段验证数据形状、格式是否符合预期
# 【iter】
# Python 内置函数，作用是把一个可迭代对象（Iterable） 转换成迭代器（Iterator）
# 【next】
# Python 内置函数，作用是从迭代器中取出下一个元素
# 因为这里是第一次执行，所以这里是第一个元素


'''定义模型'''
from torch import nn
# Sequential 指多层神经网络从前到后线性排列
# 虽然本代码中，线性神经网络只有一层
# 启发：既然有 sequential，那么一定就有非 sequential 的
# 以后看一看非线性排列的神经网络是什么样的！
# y = Xw + b 注意：对于 Linear Layer 来讲，X 才是输入，w 不是输入（是参数）
# X 维数 (1, 2); y 维数 ()
net = nn.Sequential(nn.Linear(2, 1))


'''初始化模型参数'''
# net[0] 是指第一层神经网络！
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


'''定义损失函数：均方损失函数'''
loss = nn.MSELoss()


'''定义优化算法：小批量随机梯度下降算法'''
# 其实我有一个【问题】：梯度下降算法居然没有传入损失函数
# 那它是对哪个函数求导呢？
# 前面手动实现 SGD 的时候，是直接硬编码求导表达式，
# 所以不传入损失函数自然没问题！
# 【解答】上面的疑问：
# PyTorch 的优化器和损失函数是解耦的，核心依赖 “计算图 + 反向传播”
# 优化器（trainer）的唯一职责：执行 step() 时，
# 读取每个参数的 .grad 属性（由 loss.backward() 计算好的梯度），按 SGD 公式更新参数
# 突然【反应】过来，其实手动实现线性回归的时候，SGD 也没有现场计算梯度
# 也只是访问 .grad 属性
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# .parameter() 包含所有【可训练】参数！


'''训练'''
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        # 本次梯度下降前，必须将之前的梯度清零。否则本次梯度下降要出错
        trainer.zero_grad()
        # 这里不需要 l.sum().backward() 是因为已经完成求和了
        # 但为什么会自动求和？我不知道
        l.backward()
        # 调用优化算法，更新模型参数
        trainer.step()
    # 本轮训练结束后，计算整个训练集的损失
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w 的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b 的估计误差:', true_b - b)
```
### （一）、理解代码
<br>

### （二）、`torch.utils.data` 包中的内容
[官网 doc - torch.utils.data 首页](https://docs.pytorch.org/docs/stable/data.html#module-torch.utils.data)

- **大纲：`data` 包的核心**
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

- **具体内容**
1. class `torch.utils.data.TensorDataset(*tensors)`
    > 参数讲解：`*tensors` (Tensor) – tensors that have the same size of the first dimension.
    1. Dataset wrapping tensors.
    2. Each sample will be retrieved by indexing tensors along the first dimension.
       1. 每个样本都通过张量的第一个维度来获取
    3. **要求：**
       1. 传入多个张量
       2. 并且，各张量的第一个维度大小相等

<br>

### （三）、`torch.nn.Linear` 包中的函数
[官网 doc - class torch.nn 首页](https://docs.pytorch.org/docs/stable/nn.html#module-torch.nn)

1. `torch.nn.Linear`
    - Linear 层专门做线性变换 $y = xA^{T} + b$ ⇒ 本代码中，$y = Xw + b$
    - 稍微记一记上面的线性变换表达式！
    [官网 doc - class torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
    1. `torch.nn.Linear` constructor 函数原型
        ```python
        class torch.nn.Linear(
            in_features, out_features, bias=True, device=None, dtype=None
        )
        ```
    2. 参数说明
       1. `bias` (bool) – 
          1. If set to `False`, the layer will not learn an additive bias. 
          2. **Default**: `True`
    3. 张量形状
       1. Input: $(∗, H_{in})$ , where $∗$ means any number of dimensions including none and $H_{in} = \text{in\underline{\text{ }}features}$
       2. Output: $(∗, H_{out})$ , where all but the last dimension are the same shape as the input and $H_{out} = \text{out\underline{\text{ }}features}$
       3. 张量形状（最后一维替换，其余维度保留）
          1. 输入形状：`(d1, d2, ..., dn, in_features)`
          2. 输出形状：`(d1, d2, ..., dn, out_features)`
       4. Linear 层自动将张量展平为 2 层 `(批次数, 特征数)`
    4. Linear 层的变量
       1. `weight`
       2. `bias`
       ```python
       # 学习这种参数初始化的方式
       # 其中 normal_ 下划线指原地操作
       linear_layer.weight.data.normal_(0, 0.01)
       ```


#### 1、为什么不能用 import torch.nn 代替 from torch import nn ？

| 导入语句 | 本质含义 | 后续调用方式 |
|:-----|:---|:------|
| `import torch.nn` | 导入 `torch` 模块下的 `nn` 子模块，且仅绑定到 `torch.nn` 这个完整命名空间下 | 必须写完整路径：`torch.nn.Linear` |
| `from torch import nn` | 从 `torch` 模块中把 `nn` 子模块 “提取出来”，直接绑定到当前作用域的 `nn` 变量上 | 可直接简写：`nn.Linear` |

<br>

### （四）、`torch.optim` 中的内容
[官网 doc - torch.optim 首页](https://docs.pytorch.org/docs/stable/optim.html#module-torch.optim)

#### 1、如何使用 optimizer
1. To use `torch.optim` you have to construct an **optimizer object** that 
   1. will hold the current state 
   2. and will update the parameters based on the computed gradients.

##### (1)、construct optimizer object
1. To construct an Optimizer you have to give it an **iterable（可迭代对象）** containing 
   1. the parameters (all should be Parameter s) 
   2. or named parameters (tuples of (str, Parameter)) to optimize.
   3. 所有参数都应为 `Parameter` 类型 / 命名参数（ `(str, Parameter)` 元组）
        ```python
        # 参数类型
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam([var1, var2], lr=0.0001)

        # 命名参数类型 (str, Parameter) 元组
        optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam([('layer0', var1), ('layer1', var2)], lr=0.0001)
        ```
2. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.

#### 2、Per-parameter options
1. For example, this is very useful when one wants to specify per-layer learning rates:
    ```python
    optim.SGD([
        {'params': model.base.parameters(), 'lr': 1e-2},
        {'params': model.classifier.parameters()}
    ], lr=1e-3, momentum=0.9)

    optim.SGD([
        {'params': model.base.named_parameters(), 'lr': 1e-2},
        {'params': model.classifier.named_parameters()}
    ], lr=1e-3, momentum=0.9)
    ```

#### 3、take an optimization step
```python
optimizer.zero_grad()
optimizer.step()
```
1. **All optimizers** implement a `step()` method, that updates the parameters. It can be used in two ways
   1. 见官网
   2. 这里我就不抄了
