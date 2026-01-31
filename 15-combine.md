## 一、第一个案例
```python

```
[李沐老师的 10 行代码 - 使用 automl 而非调参](https://www.bilibili.com/video/BV1rh411m7Hb?spm_id_from=333.788.videopod.sections&vd_source=774b39ad34e11aca38975d06f9a32cdb)
1. 使用 `automl` 居然不用调参，而且模型性能还很好！
   1. 非常震惊
   2. 还提到说对于很多问题，`automl` 已经实用了！
   3. 我们要怎么办？

### （一）、理解代码

<br><br>

### （二）、`pandas` API - 用于数据预处理，别怵！
- **看了这么多 *pandas API* 的例子，感觉：好像 *pandas* 只处理二维数据？**

[官网 doc - pandas API reference](https://pandas.pydata.org/docs/reference/index.html)
<br>

#### 1、`pandas.concat` 按列拼接数据
[官网 doc - pandas.concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html#pandas.concat)

```python
pandas.concat(
    objs, *, axis=0, join='outer', ignore_index=False, 
    keys=None, levels=None, names=None, 
    verify_integrity=False, sort=<no_default>, 
    copy=<no_default>
)
```

- **功能概述** ： 
  - Concatenate pandas objects along a particular axis. （沿特定轴连接 pandas 对象）
  - return : **object**, type of objs
    - When concatenating all Series along the index (axis=0), a `Series` is returned. 
    - When objs contains at least one DataFrame, a `DataFrame` is returned. 
    - When concatenating along the columns (axis=1), a `DataFrame` is returned.
  - 由于 `axis = 0`，所以，看起来默认是沿 0 轴连接 pandas 对象
    - 结合下面的例子代码 ⇒ **好像默认是将每一列拼接起来**！
  - 例子代码
    ```python
    >>> df1 = pd.DataFrame([["a", 1], ["b", 2]], columns=["letter", "number"])
    >>> df1
        letter  number
    0      a       1
    1      b       2
    >>> df2 = pd.DataFrame([["c", 3], ["d", 4]], columns=["letter", "number"])
    >>> df2
        letter  number
    0      c       3
    1      d       4
    >>> pd.concat([df1, df2])
        letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4    
    ```
<br>

#### 2、`pandas.read_csv` 读取 csv 文件
[官网 doc - pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv)

```python
# 知道有这么多参数就行，没必要全学
# 掌握几个常用的就行！
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
    dtype_backend=<no_default>
)
```

- **功能概述** ：
  - Read a comma-separated values (csv) file into DataFrame. （ 将逗号分隔值读入 DataFrame ）
  - returns : **DataFrame** or TextFileReader 
    - A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes. （ 逗号分隔值（csv）文件会以带有标记轴的二维数据结构形式返回 ）
  - 例子代码
    ```python
    >>> pd.read_csv("data.csv", dtype={{"Value": float}})
        Name  Value
    0   foo    1.0
    1   bar    2.0
    2  #baz    3.0    
    ```
<br>

#### 3、由于 `pandas.read_csv` 返回 `DataFrame`，因此先学习关于 `DataFrame` 的 `iloc`： `pandas.DataFrame.iloc` （整数索引，先行后列）
[官网 doc - pandas.DataFrame.iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc)
- **功能概述**
  - Purely integer-location based indexing for selection by position. （ `iloc` 应该就是指 `int location` ）
  - 注意：**切片**包含起点，但**不包含终点**！
  - 结合例子代码 ⇒ **先按行索引，然后再按列索引**！
  - 例子代码
    ```python
    >>> mydict = [
    ...     {"a": 1, "b": 2, "c": 3, "d": 4},
    ...     {"a": 100, "b": 200, "c": 300, "d": 400},
    ...     {"a": 1000, "b": 2000, "c": 3000, "d": 4000},
    ... ]
    >>> df = pd.DataFrame(mydict)
    >>> df
        a     b     c     d
    0     1     2     3     4
    1   100   200   300   400
    2  1000  2000  3000  4000


    '''Indexing just the rows ( 仅仅按行索引 )'''
    # With a scalar integer.
    >>> type(df.iloc[0])
    <class 'pandas.Series'>
    >>> df.iloc[0]
    a    1
    b    2
    c    3
    d    4
    Name: 0, dtype: int64

    # With a list of integers
    >>> df.iloc[[0, 1]]
        a    b    c    d
    0    1    2    3    4
    1  100  200  300  400

    # with a slice objects （ Python 中的切片 ）
    >>> df.iloc[:3]
        a     b     c     d
    0     1     2     3     4
    1   100   200   300   400
    2  1000  2000  3000  4000


    '''Indexing both axes ( 多个轴都进行索引 )'''
    # With slice objects
    >>> df.iloc[1:3, 0:3]
        a     b     c
    1   100   200   300
    2  1000  2000  3000
    ```
<br>

#### 4、虽然 `pandas.concat` 返回 *pandas object* ，但是只找到 `pandas.DataFrame.dtypes` （按列返回数据类型）
[官网 doc - pandas.DataFrame.dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html#pandas.DataFrame.dtypes)

- **功能概述**
  - Return the dtypes in the DataFrame.
    - This returns a `Series` with the data type of each column. 
    - The result’s index is the original DataFrame’s columns. 
    - Columns with mixed types are stored with the object dtype
  - 上面的中文解释
    - **核心作用**：明确该属性的核心功能是「返回 DataFrame 中的数据类型（dtypes）」，即告诉使用者每个列的数据属于哪种类型（如整数、浮点数、字符串等）。
    - **返回结果形态**：说明结果的格式是一个 `pandas.Series`（ pandas 中的一维数据结构）：
      - 这个 Series 的「值」是对应列的数据类型（如 `int64`、`float64`）；
      - 这个 Series 的「索引（index）」是原 DataFrame 的列名，确保能清晰对应到每一列。
    - **特殊情况规则**：指出当 DataFrame 某一列存在「混合数据类型」（比如一列中既有数字又有字符串）时，该列的数据类型会统一存储为 `object` 类型，避免因类型混乱导致后续操作报错。
  - 结合例子代码，发现上面的中文解释是完全正确的。即，**按列返回每一列的数据类型**。
  - 例子代码
    ```python
    >>> df = pd.DataFrame(
    ...     {
    ...         "float": [1.0],
    ...         "int": [1],
    ...         "datetime": [pd.Timestamp("20180310")],
    ...         "string": ["foo"],
    ...     }
    ... )
    >>> df.dtypes
    float              float64
    int                  int64
    datetime    datetime64[us]
    string              str
    dtype: object
    ```
<br>

#### 5、由于 `pandas.DataFrame.dtypes` 返回 `Series`，因此，学习 `pandas.Series.index` （只是简单地返回索引）
[官网 doc - pandas.Series.index](https://pandas.pydata.org/docs/reference/api/pandas.Series.index.html#pandas.Series.index)

- **功能概述**
  - return : The index labels of the Series.
  - 例子代码
    ```python
    >>> cities = ['Kolkata', 'Chicago', 'Toronto', 'Lisbon']
    >>> populations = [14.85, 2.71, 2.93, 0.51]
    >>> city_series = pd.Series(populations, index=cities)
    >>> city_series.index
    Index(['Kolkata', 'Chicago', 'Toronto', 'Lisbon'], dtype='object')

    >>> type(city_series.index)
    <class 'pandas.core.indexes.base.Index'>

    >>> city_series.dtypes
    dtype('float64')
    >>> city_series.mean()
    np.float64(5.25)
    >>> city_series.std()
    np.float64(6.4926009169412735)

    >>> city_series
    A    14.85
    B     2.71
    C     2.93
    D     0.51
    dtype: float64
    ```
<br>

#### 6、`pandas.Series.fillna` （填充 缺失值 和 NaN）
```python
Series.fillna(value, *, axis=None, inplace=False, limit=None)
```
[官网 doc - pandas.Series.fillna](https://pandas.pydata.org/docs/reference/api/pandas.Series.fillna.html#pandas.Series.fillna)

- **功能概述**
  - Fill NA / NaN values with value.
  - Returns : Series/DataFrame
    - Object with missing values filled.
<br>

#### 7、`pandas.get_dummies` 用于独热编码
```python
pandas.get_dummies(
    data, prefix=None, prefix_sep='_', dummy_na=False, 
    columns=None, sparse=False, drop_first=False, dtype=None
)
```

[官网 doc - pandas.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html#pandas.get_dummies)

- **功能概述**
  - Convert categorical variable into dummy / indicator variables. （将分类变量转换为虚拟变量 / 指示变量）
    - Each variable is converted in as many 0 / 1 variables as there are different values. （每个变量会被转换为与不同值数量相同的 0 / 1 变量）
    - Columns in the output are each named after a value; （ 输出中的列均以对应的值命名； ）
    - if the input is a DataFrame, the name of the original variable is prepended to the value. （如果输入是一个 DataFrame，原始变量的名称会被添加到该值的前面）
  - **注意**：如果标签很多的话，`pandas.get_dummies` 容易爆内存
    - 这时候就需要采用其他方法了！
  - 例子代码
    ```python
    >>> s = pd.Series(list("abca"))
    >>> pd.get_dummies(s)
        a      b      c
    0   True  False  False
    1  False   True  False
    2  False  False   True
    3   True  False  False

    >>> s1 = ["a", "b", np.nan]
    >>> pd.get_dummies(s1)
        a      b
    0   True  False
    1  False   True
    2  False  False
    >>> pd.get_dummies(s1, dummy_na=True)
        a      b    NaN
    0   True  False  False
    1  False   True  False
    2  False  False   True

    >>> pd.get_dummies(pd.Series(list("abc")), dtype=float)
        a    b    c
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    ```
<br><br>

### （三）、`torch` API - 用于模型训练和预测
#### 1、`torch.clamp` （限制数据在 `[min, max]` 范围中）
[官网 doc - torch.clamp](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp)
```python
torch.clamp(input, min=None, max=None, *, out=None)
```

- **功能概述**
  - Clamps all elements in `input` into the range $[ \text{ min, max }]$. （ 将 `input` 中的所有元素限制在 $[\text{ min, max }]$ 的范围内 ）
<br>

#### 2、`torch.cat` （默认在 0 轴上拼接数据）
[官网 doc - torch.cat](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat)

```python
torch.cat(tensors, dim=0, *, out=None)
```
<br>

## 二、案例总结
1. 集成学习在刷榜的时候经常用到
2. 在本案例中，MLP 仔细调的话，也可以达到比较好的精度。以下是一些比较关键的点
   1. 特征预处理（非常关键，但是我不会）
   2. 超参数的调节

### （一）、难点总结
1. 数值较大 ⇒ 容易梯度爆炸
   1. 如果已知数据都是比较大的整数的话，那么可以取对数
2. 文本特征较多
   1. 用 word2vec / transformer 会比较好
3. 训练使用的是历史数据，
   1. 但是由于时序信息等导致新的数据可能和历史数据有很大差异

### （二）、关于调参的经验
1. 已经调出来一个结果后，再试着往其他方向调一调
   1. 如果后面的结果对变化十分敏感的话，说明这个参数本身不是在一个平滑的曲面上
   2. ⇒ 也就是说，你刚才的结果其实不是最优解，只不过恰好在刚才的数据集上比较吻合
   3. 这种模型的泛化能力一般比较差
2. 另外，实际应用中，调参没那么重要，因为数据一直在变化！参数只要调的差不多就行！
3. 而且，在实际中，数据往往才是决定因素，模型本身其实没那么重要！
   1. 看数据中哪些东西有用，哪些东西没用
   2. 然后做数据清洗

