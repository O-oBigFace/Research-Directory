# # Char Embedding

>   献给焦虑的你，take it easy.

[TOC]

改编自[原文](https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10)

##什么是Char Embedding？

![img](https://cdn-images-1.medium.com/max/1600/1*tMTUPcxnRrc452e5IrLKqA.png)

2016年，[Text Understanding from Scratch](https://arxiv.org/pdf/1502.01710v5.pdf)引入了字符级CNN。作者发现字符中包含了改善模型效果的关键信息。在这篇论文中，他们定义了70个characters包括26个英文字母、10个数字和33个特殊字符。

```
# Copy from Char CNN paper
abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}
```

![img](https://cdn-images-1.medium.com/max/1600/1*VfJM5rAt7xLEr5RGv450qw.png)

另一方面，Google Brain团队在[Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf)中，给出了[lm_1b](https://github.com/tensorflow/models/tree/master/research/lm_1b)数据集，其中包含了256个向量(包括52个字母和特殊字符)，维度仅仅只有16维。而隔壁的词向量，向量的维度和数目都十

## 为什么使用Char Embedding？

在英文中，所有的单词都由26个(或者区分大小写时52)英文字母组成。Char Embedding有以下优点

1.  利用Char Embedding可以对out-of-vocabulary的单词进行嵌入。另一方面，词嵌入只能嵌入见过的词汇；

2.  可以适配拼写错误单词、表情符号、新增词汇的嵌入问题（如新增的"boba tea"）；
3.  对低频词能够更好地进行嵌入：目前的词嵌入方法都依赖大量的训练语料；
4.  向量数量少，维度低，大大降低了模型复杂度，提高了效率。

## 何时使用？

1.  [文本分类任务](https://arxiv.org/pdf/1509.01626.pdf)
2.  [语言模型](https://arxiv.org/pdf/1602.02410.pdf)
3.  [命名实体识别](https://www.aclweb.org/anthology/Q16-1026)

## 一些工作

[A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf)

[Learning Character-level Representations for Part-of-Speech Tagging](http://proceedings.mlr.press/v32/santos14.pdf)

[Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)

[Character Level Deep Learning in Sentiment Analysis](https://offbit.github.io/how-to-read/)