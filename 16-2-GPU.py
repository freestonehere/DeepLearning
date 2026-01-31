import torch
from torch import nn

# 这里的 'cuda' 其实是 'cuda:0', 以后也可以有 'cuda:1'
torch.device('cpu'), torch.device('cuda')



def try_gpu(i=0):  #@save
    '''如果存在, 则返回gpu(i), 否则返回cpu()'''
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    '''返回所有可用的GPU, 如果没有GPU, 则返回[cpu(),]'''
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(f'try_gpu(): {try_gpu()}')



# 可以使用的 GPU 数量
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')


X = torch.tensor([1, 2, 3], dtype = torch.float32)
print(f'X.device: {X.device}')


# 将张量 X 搬到 GPU 上
Y = X.cuda(0)
print(f'Y.device: {Y.device}')
Y = X.cuda()
print(f'Y.device: {Y.device}')


'''
把网络、权重、输入 copy 到 GPU
GPU 主要用于前向和反向计算 (也就是模型计算)
'''


'''
人们使用GPU来进行机器学习, 因为单个GPU相对运行速度快。 
但是在设备 (CPU、GPU和其他机器) 之间传输数据比计算慢得多。 
这也使得并行化变得更加困难，因为我们必须等待数据被发送（或者接收）， 然后才能继续进行更多的操作。
这就是为什么拷贝操作要格外小心。 
根据经验，多个小操作比一个大操作糟糕得多。 
此外，一次执行几个操作比代码中散布的许多单个操作要好得多。 
如果一个设备必须等待另一个设备才能执行其他操作， 那么这样的操作可能会阻塞。 

最后, 当我们打印张量或将张量转换为NumPy格式时, 
如果数据不在内存中，框架会首先将其复制到内存中， 这会导致额外的传输开销。 
更糟糕的是, 它现在受制于全局解释器锁, 使得一切都得等待Python完成。
'''


net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu()) # 把模型 (参数) 放到 GPU 上
X = X.to(device=try_gpu()) # 把数据放到 GPU 上，计算数据必须在同一平台上！
print(f'net(X): {net(X)}')
print(f'net[0].weight.data.device: {net[0].weight.data.device}')
