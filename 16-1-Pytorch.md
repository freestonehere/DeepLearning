# Pytorch 神经网络基础

## 一、概念理解
### （一）、层和块
#### 1、使用 Block 实现块
```python
class MLP(nn.Module):
    # 必须先使用父类的init初始化，接下来可以定义各层
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    # 必须重新定义前馈过程
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
net(X) 
# 虽然没写 net.forward(X) 但事实上 Python 确实会执行 forward(X) 函数
```

#### 2、自定义 Sequential 实现
```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，`module`是`Module`子类的一个实例。我们把它保存在'Module'类的成员
            # 变量`_modules` 中。`module`的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

<br><br><br>

### （二）、参数
#### 1、参数访问
> `state_dict()` 查看字典形式的模型参数数值

```python
# 可以把Sequential看作一个list，可以用索引拿出每一层的参数。得到一个有序字典。
print(net[2].state_dict())

# module.state_dict().keys()=['weight','bias']
```

```python
print(type(net[2].bias))
# Out:<class 'torch.nn.parameter.Parameter'>
print(net[2].bias)
# Out:Parameter containing:
# tensor([-0.3001], requires_grad=True)
print(net[2].bias.data)
# Out: tensor([-0.3001], requires_grad=True)
net[2].weight.grad == None  # .grad访问梯度
# Out: True
```

> 参数是复合的对象，包含值`.data`、梯度`.grad`和额外信息。 这就是我们需要显式参数值的原因。 除了值之外，我们还可以访问每个参数的梯度。


#### 2、一次访问所有元素
当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。 当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂， 因为我们需要递归整个树来提取每个子块的参数。 下面，我们将通过演示来比较访问第一个全连接层的参数和访问所有层。

> `.named_parameters()` 返回 iterator，用于循环，返回(参数名, 参数数值)。

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# *代表把list/tuple里的元素分开，而非整个输出
```

还提供了另一种访问网络参数的方式，通过名称（默认以`层数序号.weight or .bias`），如下所示。

```python
net.state_dict()['2.bias'].data
```

#### 3、参数初始化（e.g. Xavier 初始化）

**Q：如果在模型参数初始化时，对 W 不采用高斯随机分布`torch.randn()`，而是全零`torch.zeros()`或全 1`torch.ones()`分布，会产生什么？**

**🙋‍♂️**：经过实验，对 W、b 采用全零或全 1 初始化，或 W 全零、b 全 1，或 W 全 1、b 全零，模型都无法训练，推测应该是以上四种初始化下，无法计算梯度，则无法进行参数更新。具体原因有待后续讨论。

- 使用Xavier随机初始化：`torch.nn.init.xavier_uniform(tensor, gain=1)`

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        #uniform distribution
```

#### 4、参数绑定（共享权重）
```python
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                   nn.ReLU(), nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] == 100
# 会同时修改两个shared, 相当于同一个实例的赋值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

#### 5、自定义层（带参数的层）
- 自定义层和自定义模型其实一样的

> nn.Parameter(tensor, required_grad=True) #把传入张量当作模块参数，可以对其求导的

```python
# 定义一个线性层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        #理论上torch.randn(units,)与torch.randn(units)没有区别
        #逗号后省略表示维度只有1
        #如果是randn(2, 1)，就是一个二维张量了。

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5, 3)
dense.weight
```

<br><br><br>

### （三）、读写文件
到目前为止，我们讨论了如何处理数据，以及如何构建、训练和测试深度学习模型。然而，有时我们希望保存训练的模型，以备将来在各种环境中使用（比如在部署中进行预测）。此外，当运行一个耗时较长的训练过程时，最佳的做法是定期保存中间结果，以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。因此，现在是时候学习如何加载和存储权重向量和整个模型了。

Pytorch存储本质上使用的是Python实现的 **Pickle序列化（Serialization）** 操作，有关Pickel序列化的内容可以参考👉[这里](https://docs.python.org/zh-cn/3/library/pickle.html)

- 存储、读取矩阵

> torch.save(tensor, 'filename')
>
> torch.load('filename')

```python
#存储一个tensor
X = torch.arange(4)
torch.save(X, 'x-file')

X2 = torch.load('x-file')
X2
```

```python
#存储高维度
y = torch.zeros(4)
torch.save([X, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```python
#存储字典
mydict = {'x': X, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

- 存储模型参数

> torch.save(net.state_dict(),'net.params')
>
> net.load_state_dict(torch.load('net.params'))

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params') #存储的实际是模型参数而非模型本身

clone = MLP()   #先克隆原模型本身
#再载入参数
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()    #eval()设置模型为评估推理模式，参数为不可导
Y_clone = clone(X)
Y_clone == Y   
# Out: True
```

> 如要存储模型结构定义，需要通过**TorchScript**存储


