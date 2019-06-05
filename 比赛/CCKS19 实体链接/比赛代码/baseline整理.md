# #baseline整理

[TOC]

## 改进点

#### 数据处理

1.  知识库信息的处理还可以再优化一些

#### 实体识别

1.  *词嵌入可以使用其他方法而不是随机初始化yi*



## 数据处理 

#### 预处理

1.  知识库
    *   所有predicate等值属性全部整合进了实体描述信息
    *   id2kb
    *   kb2id，kb中的一个entity可能会对应多个id
    *   所有的英文entity都转为了小写
2.  训练数据

```python
"""
X1: 句子的char list - list - sent_size * batch_size
X2: 描述文本的转化为的字符列表 - list - desc_size * batch_size
S1: 一串和句子长度相同的数字， 标记了句子中所有mention开始的位置 - np array - sent_size * batch_size
S2: 一串和句子长度相同的数字， 标记了句子中所有mention结束位置的后一位 - np array - sent_size * batch_size
Y : 句子的mention表示形式，随机选择的mention的位置上值为1，其他位置上值为0 - np array - sent_size * batch_size
T : 样本的标签，[1]正例,[2]负例 - list - 1 * batch_size
"""
```

## 实体识别

2BiLSTM+1卷积 ==》两个全连接层预测实体的起始和结束位置

s

## 实体链接

