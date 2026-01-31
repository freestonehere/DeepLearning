'''下载和缓存数据集'''
import hashlib
import os
import tarfile
import zipfile
import requests


'''
download函数用来下载数据集, 
将数据集缓存在本地目录 (默认情况下为./data) 中， 
并返回下载文件的名称。 
如果缓存目录中已经存在此数据集文件, 
并且其 sha-1 与存储在 DATA_HUB 中的相匹配, 
我们将使用缓存的文件, 以避免重复的下载
'''
def download(name, cache_dir=os.path.join('.', 'data')):  #@save
    '''下载一个 DATA_HUB 中的文件, 返回本地文件名'''
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从 {url} 下载 {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  #@save
    '''下载并解压zip/tar文件'''
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    '''下载DATA_HUB中的所有文件'''
    for name in DATA_HUB:
        download(name)

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

'''
DATA_HUB 是个字典 {'数据集名称 (字符串形式)': (数据集 URL, 验证数据集完整性的 sha-1 密钥)}
'''

