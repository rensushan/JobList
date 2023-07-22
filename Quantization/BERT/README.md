# Examples

At the moment, the Hugging Face library seems to be the most widely accepted and powerful pytorch interface for working with BERT. In addition to supporting a variety of different pre-trained transformer models, the library also includes pre-built modifications of these models suited to your specific task. For example, in this tutorial we will use `BertForSequenceClassification`.


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Details](#details)
- [Result](#result)
- [Reference](#reference)

## Background

BERT的动机

- 基于微调的NLP模型
- 预训练的模型抽取了足够多的信息
- 新的任务只需要增加一个简单的输出层

BERT架构

- 只有编码器的Transformer
- 两个版本
   > Base : #blocks = 12, hidden size = 768, #heads = 12, #parameters = 110M  
   > Large : #blocks = 24, hidden size = 1024, #heads = 16, #parameters = 340M
-  在大规模数据上训练>3B词

BERT针对微调设计

基于Transformer的编码器做了如下修改：
- 模型更大，训练数据更多
- 输入句子对，片段嵌入，可学习的位置编码
- 训练时使用两个任务：
      1.带掩码的语言模型
      2.下一个句子预测

## Install

let’s install the [transformers](https://github.com/huggingface/transformers) package from Hugging Face which will give us a pytorch interface for working with BERT. 

```sh
$ pip install transformers
```

## Usage

```sh
$ python fine-tune.py
```

### Details

We’ll use [The Corpus of Linguistic Acceptability (CoLA)](https://nyu-mll.github.io/CoLA/) dataset for single sentence classification. It’s a set of sentences labeled as grammatically correct or incorrect. It was first published in May of 2018, and is one of the tests included in the “GLUE Benchmark” on which models like BERT are competing.

*Loading CoLA Dataset*

We’ll use the `wget` package to download the dataset to the Colab instance’s file system.

```sh
$ pip install wget
```

The dataset is hosted on GitHub in this repo: https://nyu-mll.github.io/CoLA/

```sh
import wget
import os

print('Downloading dataset...')

# The URL for the dataset zip file.
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

# Download the file (if we haven't already)
if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip')
```

Unzip the dataset to the file system. You can browse the file system of the Colab instance in the sidebar on the left.

```sh
# Unzip the dataset (if we haven't already)
if not os.path.exists('./cola_public/'):
    !unzip cola_public_1.1.zip
```

The path to the dataset:`/data/ice/quantization/master/BERT/cola_public`  
Model:`bert-base-uncased`  
Batch_size:`32`  
Eopch:`2`  
The specific parameters are explained in the `fine-tune.py`

## Result

It takes **42s** to train 2 epochs using one A800 GPU.  
The GPU usage is **95% to 96%**.  
Accuracy on the CoLA benchmark is measured using the **Matthews correlation coefficient** (MCC).
- The MCC of [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) is **0.521**.
- The MCC of [huggingface](https://github.com/huggingface/transformers/tree/5bfcd0485ece086ebcbed2d008813037968a9e58/examples) is **0.489**.
- Our MCC is **0.540** and Acc is **0.83**.

## Reference

[BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  
By Chris McCormick and Nick Ryan
