import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    plt.switch_backend('TkAgg')
    # d2l.use_svg_display() 因为 use_svg_display 是 Jupyter 依赖！
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
tmp = torch.eye(10).reshape((1, 1, 10, 10))
attn_weights_new = torch.concat([attention_weights, tmp], dim=2)
# torch.eye(10) 生成 (10, 10) 的矩阵张量，该矩阵张量主对角线元素全为 1，其他元素均为 0！
# 也就是说：torch.eye(10) 生成单位方阵！
print(attn_weights_new.shape)
show_heatmaps(attn_weights_new, xlabel='Keys', ylabel='Queries')
'''attn_weights_new 的张量形状是 (1, 1, 20, 10)'''

plt.show()