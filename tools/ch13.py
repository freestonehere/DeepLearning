'''计算机视觉相关的内容'''
import matplotlib.pyplot as plt

# 设置 matplotlib 交互式后端（解决 PyCharm 静态渲染问题）
plt.switch_backend('TkAgg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    plt.rcParams['figure.figsize'] = figsize

def set_title(fig, title):
    "set title for TkAgg window"
    # 新版本 matplotlib：先获取 Tk 组件，再获取顶层窗口
    fig.canvas.get_tk_widget().winfo_toplevel().wm_title(title)