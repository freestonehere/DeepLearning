import re
import tools.cache_load_datasets as cache_load

cache_load.DATA_HUB['time_machine'] = (cache_load.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

# 这里没有数据集三件套，只有 read，可能是因为还没有用到 train_iter？
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    # open(..., 'r')：以只读文本模式打开文件。
    # with 语句：保证文件使用后自动关闭。
    with open(cache_load.download('time_machine'), 'r') as f:
        lines = f.readlines()
        # f.readlines() 返回一个列表，每个元素是文件中的一行（包含换行符 '\n'）
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    # [^...] 表示匹配不在该字符类中的任意字符
    # 将所有非字母字符替换为一个空格 → 去除首尾空白 → 转小写

