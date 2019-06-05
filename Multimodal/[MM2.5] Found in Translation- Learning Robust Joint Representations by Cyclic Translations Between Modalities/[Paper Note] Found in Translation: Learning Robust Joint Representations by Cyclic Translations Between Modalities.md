# 5
**Related Work真的很好**

## 概括
 - 多模态(视频)情感识别任务, 但本篇论文的重点是多模态数据联合表示，对情感识别任务并没有过多的探讨。
 - 主要的关注点在于获得鲁棒的多模态联合表示，即在多模态数据有噪声或者数据缺失时亦能获得有益的联合表示。
 - 循环翻译cyclic translations在多模态任务中有较好的效果。
 - 层次化结构使输入的模态数据多多益善。

## Abstract
 1. The central challenge in multimodal learning involves inferring joint representations that can process and relate information from these modalities.
多模态学习的难点在于学习到多模态联合表示，这些表示能够处理、关联这些模态的信息

 2. However, existing work learns joint representations by requiring all modalities as input and as a result, the learned representations may be sensitive to noisy or missing modalities at test time.
现有的工作在训练多模态表示时需要将**所有的模态数据**作为输入/输出，这样学习到的表示在测试时会对**噪声、模态缺失的情况敏感**。 {思考一下，这篇和我的论文真的相关吗？}

## Introduction
### 1
1. 情感分析是用来辨识一个演讲者的观点的工具。仅基于文本的情感分析不足够推断情感内容，它忽略了太多非言语信息。
2. Multimodal snetiment analysis: 目前的研究转向利用机器学习方法从其他附加信息中，如：视觉、音频等模态中获得joint representation。
3. 社交网络中包含了大量多模态数据。
4. **However**，以往的研究在learn joint representation时，**需要多重模态的数据作为输入**。这些方法因此在测试时对噪声、模态缺失敏感。

### 2
5. **To address this problem**，作者借鉴了Seq2Seq无监督学习的思想，提出**MCTN**(the Multimodal Cyclic Translation Network model)学习鲁棒的联合模态表示。其主要思想是：将一个源模态S翻译为目标模态T时，模型生成的intermediate representation能够捕捉模态S与T的联合信息。
6. MCTN利用**cyclic translation loss**包含*forward translations*(从源模态到目标模态)和*backward translations*（从预测目标模态到源模态）来保证学习到的联合表示能够捕捉到最丰富的模态间信息。
7. MCTN的损失函数包含了：(1)the cyclic translation loss 前向+逆向, (2)a prediction loss (task-specific)
8. MCTN只要在多模态数据上进行训练，在测试时只需要将输入源模态数据就能推导出联合表示和label(预测结果)

## Proposed Approach
### MCTN
 1. 利用**cycle consistency loss**训练模态*翻译*(modality translation)和*逆向翻译*(back-translation). 在多模态环境中使用**逆向翻译**的目的是激励翻译模型在只有源模态数据作为输入下，也能够学习到有益的联合表示。
 2. 使用Seq2Seq模型来训练joint representation，将原始模态的数据编码、解码到目标模态，利用编码器的结果进行情感分析。




## Experimental Setup
### 数据集和输入模态
1. 数据集
(1) the CMU Multimodal Opinion-level Sentiment Intensity dataset (CMU-MOSI)
(2) ICT-MMMO
(3) Youtube

### Evaluation Metrics
Mean Abosulate Error
Pearson's correlation *r*


## Results and Discussion
1. 与现有的工作比较，MCTN在测试时只利用了语言数据，就已经取得了新的state-of-art结果。。。
2. **Adding More Modalities**: 从结果来看，学习到的联合表示能够利用上输入数据中更多的模态信息。在新的模态加入时，其仅有少量的信息损失。

## Ablation Studies
cyclic translations, modality ordering, and hierarchical structure

实验结果论证了：
1. cyclic translation的重要性：
(1) 激励模型学习源模态到目标模态的对称性，thus adding a source of regularization.
(2) 确保学习到的表示能够最大化保留各个模态的信息。
2. 一个共享参数的Seq2Seq模型大于两个S2S模型，因为两个模型参数多容易过拟合
3. language作为source modality的重要性
4. 层次化结构将学习任务分治，让表示学习更简单