## 一、README
### （一）、代码
[d2l - 普通 .py 版本的代码（非 Jupyter）](https://github.com/Miraclelucy/dive_into_deep_learning)

<br><br>

### （二）、`limu` 环境
#### 1、VS code 编辑器
1. color theme
   1. 原本是 Dark Modern (Default Dark Modern)
   2. 现在是 Light Modern (Default Light Modern) / Quiet Light

#### 2、PyCharm 编辑器
1. Theme
   1. 原本是 `4 dark`
   2. 现在是 Islands Light
2. Comment color （注释的颜色）
   1. 在 color-language default 中
      1. 6AAB73（绿色）
      2. 取消斜体（Italic）

#### 3、IDEA 编辑器
1. Theme
   1. 原本是 `1 Dark`
   2. 现在是 Light with Light Header

#### 4、系统终端 （ `Win + R` 呼出的终端 ）
1. 配色方案
   1. 原来是 One Half Dark
   2. 现在是 Solarized Light（过度曝光，便于在阳光下使用）
2. 字号
   1. 原来是 12
3. 背景图像不透明度
   1. 原来是 30%
4. 背景不透明度
   1. 原来是 100%


#### 5、Pytorch 包
```powershell
conda create -n limu python=3.10 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ -y

pip3 install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu126

# d2l 不需要镜像源就能直接下载
# 另外，高版本的 d2l 会与 Pytorch 产生冲突，必须安装低版本的 d2l
pip install d2l==0.17.0
```

<br><br>

### （三）、快速上手 Jupyter
[同济子豪兄 - 快速上手 Jupyter Notebook](https://www.bilibili.com/video/BV1Q4411H7fJ/?spm_id_from=333.337.search-card.all.click&vd_source=774b39ad34e11aca38975d06f9a32cdb)

<br><br><br>

## 二、关于 `./data` 文件夹中文件的说明
1. 因为数据集太大，上传不方便，所以就没上传。
2. 而且只要愿意找，总能找到的
   1. 不过方便起见，这里还是贴出下载链接

```
data
├── 15-01-house-prices-advanced-regression-techniques
|   （ 通过 https://pan.baidu.com/s/1Byx4c1PCYKsuIRpYkEFn1g?pwd=x0j9 下载 ）
├── 15-02-california-house-prices
|   （ 通过 https://pan.baidu.com/s/12aRIWIGSCHJd_IJvFRtD9A?pwd=6666 下载 ）
├── FashionMNIST（ 没招了，只能通过官网下载 ）
├── kaggle_house_pred_test.csv （ 通过 15-combine-01.py 代码下载 ）
└── kaggle_house_pred_train.csv （ 通过 15-combine-01.py 代码下载 ）
```