from tools.cache_load_datasets import download, download_extract, download_all
from tools.cache_load_datasets import DATA_HUB, DATA_URL
import matplotlib.pyplot as plt # 用于画图
# 设置 matplotlib 交互式后端（解决 PyCharm 静态渲染问题）
plt.switch_backend('TkAgg')

import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


'''加载数据集'''
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 第一个特征是 ID， 这有助于模型识别每个训练样本。
# 虽然这很方便，但它不携带任何用于预测的信息。 
# 因此，在将数据提供给模型之前，我们将其从数据集中删除
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


'''数据预处理: 将所有东西变为数值（便于进行机器学习）'''
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())) # 按列求均值，求标准差
# 在标准化数据之后，所有均值消失（均值 = 0），因此我们可以将缺失值设置为 0
# 但是感觉下面这行代码好像没必要？但是后来一想，有必要！因为真实数据中就是会有各种各样的缺失！
all_features[numeric_features] = all_features[numeric_features].fillna(0)


'''数据预处理: 类别特征-独热编码（不是特别理解）'''
# “Dummy_na=True” 将 “na”（缺失值）视为有效的特征值，并为其创建指示符特征
# 处理后，all_features中不再有字符串列，
# 所有列都是数值型（0/1 或标准化后的连续值），无任何 NaN，
# 满足机器学习模型的输入要求
all_features = pd.get_dummies(all_features, dummy_na=True)
# all_features.shape
# all_features = (n_train + n_test, in_features)
# 经过独热编码以后，in_features 会变大！


'''定义输入机器学习模型的张量'''
# 从 pandas 格式中提取 numpy 格式，并将其转换为张量表示
n_train = train_data.shape[0] # 感觉 pandas 只处理二维数据
# 这里 .values 方法绝对不能少！
train_features = torch.tensor(all_features[:n_train].values.astype(float), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(float), dtype=torch.float32)
# 这里 train_data.SalePrice 其实是 pandas.DataFrame.__getattr__
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 定义输入特征维数 in_features
in_features = train_features.shape[1]


'''定义损失函数'''
loss = nn.MSELoss()


'''定义模型: 居然只有一层线性层？'''
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


'''做回归时, 如果数值较大, 那么就常取对数'''
# 举例理解，两套房子，可能一个卖 1 万，1 个卖 100 万
# 如果你直接把这 2 套房子的误差相加，那要出问题的
# 因此，我们只关心每套房子的相对误差 (y - \hat{y}) / y
# 对房价、股票做回归的话，都经常先将数值取对数，然后丢进损失函数！
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于 1 的值设置为 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    # RMSE = sqrt(MSE) 对均方误差取平方根，得到均方根误差
    # Root Mean Square = sqrt (Mean Square)
    return rmse.item() # item() 返回普通 Python 标量（在 torch 中定义？）
'''实际训练过程中用的是 log_rmse 吗？'''


'''定义训练函数'''
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是 Adam 优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            '''
            实际训练的时候还是用的 loss = MSELoss, log_rmse 只是用于【评估指标】。
            为什么训练的时候不用 log_rmse 呢？
            反正一个很重要的原因是 log_rmse 梯度不好算
            '''
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
        # 每跑一次训练，把 训练误差 & 泛化误差 添加到 train_ls, test_ls
    return train_ls, test_ls

'''返回 训练 和 验证 数据集'''
def get_k_fold_data(k, i, X, y):
    assert k > 1
    # 21 // 5 = 4 ⇒ 相当于直接丢弃掉后面的数据
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # 注意张量形状：
        # X = (n_train + n_test, in_features) 
        # y = (n_train + n_test) ⇒ 这样就容易理解后面的索引了！
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            # j 代表当前是第 j 折; i 代表第 i 折作验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

'''正式进行 k 折交叉验证 ( 需要提供超参数 )'''
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    # 初始化 train_loss_sum, valid_loss_sum
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # *data 前面的是解包操作; 训练 num_epochs 轮
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # 这种做法也没问题：虽然实际上训练了 num_epochs 轮，但我们只取最后一轮的 loss 加起来！
        # 这也对应了：后面平均损失是除以 (k)，而不是 (k * num_epochs)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            # 显然这是画图的！
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        '''不关心训练误差, 只关心泛化误差'''
        print(f'折 {i + 1}, 训练 log rmse {float(train_ls[-1]):f}, '
              f'验证 log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练 log rmse: {float(train_l):f}, '
      f'平均验证 log rmse: {float(valid_l):f}')


'''上面都是自己训练, 下面就是将训练集提交到 Kaggle'''
'''
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练 log rmse: {float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到 Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
'''

plt.show()
