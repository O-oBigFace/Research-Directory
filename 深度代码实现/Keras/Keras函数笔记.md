# #

[TOC]

## 基本语法

### Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

#### 作用

将 Dropout 应用于输入。Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。

## 层

### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

#### 说明

1D 卷积层 (例如时序卷积)。

该层创建了一个卷积核，该卷积核以 单个空间（或时间）维上的层输入进行卷积， 以生成输出张量。 如果 `use_bias` 为 True， 则会创建一个偏置向量并将其添加到输出中。 最后，如果 `activation` 不是 `None`，它也会应用于输出。

当使用该层作为**模型第一层**时，需要提供 `input_shape` 参数（整数元组或 `None`），例如， `(10, 128)` 表示 10 个 128 维的向量组成的向量序列， `(None, 128)` 表示 128 维的向量组成的变长序列。

#### 参数

-   **filters**: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
-   **kernel_size**: 一个整数，或者单个整数表示的元组或列表， 指明 1D 卷积窗口的长度。
-   **strides**: 一个整数，或者单个整数表示的元组或列表， 指明卷积的步长。 指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
-   **padding**: `"valid"`, `"causal"` 或 `"same"` 之一 (大小写敏感) `"valid"` 表示「不填充」。 `"same"`表示填充输入以使输出具有与原始输入相同的长度。 `"causal"` 表示因果（膨胀）卷积， 例如，`output[t]` 不依赖于 `input[t+1:]`， 在模型不应违反时间顺序的时间数据建模时非常有用。 详见 [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499)。
-   **data_format**: 字符串, `"channels_last"` (默认) 或 `"channels_first"` 之一。输入的各个维度顺序。 `"channels_last"` 对应输入尺寸为 `(batch, steps, channels)` (Keras 中时序数据的默认格式) 而 `"channels_first"` 对应输入尺寸为 `(batch, channels, steps)`。
-   **dilation_rate**: 一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。 当前，指定任何 `dilation_rate` 值 != 1 与指定 stride 值 != 1 两者不兼容。
-   **activation**: 要使用的激活函数 (详见 [activations](https://keras.io/zh/activations/))。 如未指定，则不使用激活函数 (即线性激活： `a(x) = x`)。
-   **use_bias**: 布尔值，该层是否使用偏置向量。
-   **kernel_initializer**: `kernel` 权值矩阵的初始化器 (详见 [initializers](https://keras.io/zh/initializers/))。
-   **bias_initializer**: 偏置向量的初始化器 (详见 [initializers](https://keras.io/zh/initializers/))。
-   **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
-   **bias_regularizer**: 运用到偏置向量的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
-   **activity_regularizer**: 运用到层输出（它的激活值）的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
-   **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
-   **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。

#### 输入尺寸

3D 张量 ，尺寸为 `(batch_size, steps, input_dim)`。

#### 输出尺寸

3D 张量，尺寸为 `(batch_size, new_steps, filters)`。 由于填充或窗口按步长滑动，`steps` 值可能已更改。

### CuDNNLSTM

```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

#### 说明

由 [CuDNN](https://developer.nvidia.com/cudnn) 支持的快速 LSTM 实现。

只能以 TensorFlow 后端运行在 GPU 上。

#### **参数**

-   **units**: 正整数，输出空间的维度。
-   **kernel_initializer**: `kernel` 权值矩阵的初始化器， 用于输入的线性转换 (详见 [initializers](https://keras.io/zh/initializers/))。
-   **unit_forget_bias**: 布尔值。 如果为 True，初始化时，将忘记门的偏置加 1。 将其设置为 True 同时还会强制 `bias_initializer="zeros"`。 这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
-   **recurrent_initializer**: `recurrent_kernel` 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 [initializers](https://keras.io/zh/initializers/))。
-   **bias_initializer**:偏置向量的初始化器 (详见[initializers](https://keras.io/zh/initializers/)).
-   **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
-   **recurrent_regularizer**: 运用到 `recurrent_kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
-   **bias_regularizer**: 运用到偏置向量的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
-   **activity_regularizer**: 运用到层输出（它的激活值）的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
-   **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
-   **recurrent_constraint**: 运用到 `recurrent_kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
-   **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
-   **return_sequences**: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
-   **return_state**: 布尔值。除了输出之外是否返回最后一个状态。
-   **stateful**: 布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品的最后状态 将用作下一批次中索引 i 样品的初始状态。

### Bidirectional

```python
keras.layers.Bidirectional(layer, merge_mode='concat', weights=None)
```

#### 说明

RNN 的双向封装器，对序列进行前向和后向计算。

#### **参数**

-   **layer**: `Recurrent` 实例。
-   **merge_mode**: 前向和后向 RNN 的输出的结合模式。 为 {'sum', 'mul', 'concat', 'ave', None} 其中之一。 如果是 None，输出不会被结合，而是作为一个列表被返回。

#### **异常**

-   **ValueError**: 如果参数 `merge_mode` 非法。

**例**

```
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```