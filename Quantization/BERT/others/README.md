# BertWithPretrained

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

[参考代码](https://github.com/moon-hotel/BertWithPretrained/tree/main)

## 模型详细解析
- 1. 基于BERT预训练模型的英文多选项(SWAG)任务
- 2. 基于BERT预训练模型的英文问答(SQuAD,Adversarial QA)任务

## 目录结构

在`/data/ice/quantization/master/BERT/others`文件夹下

- `bert_base_uncased`目录中是BERT base英文预训练模型以及配置文件

    模型下载地址：https://huggingface.co/bert-base-uncased/tree/main
    
    注意：`config.json`中需要添加`"pooler_type": "first_token_transform"`
                                `"use_torch_multi_head": true`

- `data`目录中是各个下游任务所使用到的数据集
    - `SWAG`是[SWAG问题选择数据集](https://github.com/rowanz/swagaf/tree/master/data)
    - `SQuAD`是[斯坦福大学开源的问答数据集1.1版本](https://rajpurkar.github.io/SQuAD-explorer/)

- `model`目录中是各个模块的实现
    - `BasicBert`中是基础的BERT模型实现模块
        - `MyTransformer.py`是自注意力机制实现部分；
        - `BertEmbedding.py`是Input Embedding实现部分；
        - `BertConfig.py`用于导入开源的`config.json`配置文件；
        - `Bert.py`是BERT模型的实现部分；
    - `DownstreamTasks`目录是下游任务各个模块的实现
        - `BertForSentenceClassification.py`是单标签句子分类的实现部分；
        - `BertForMultipleChoice.py`是问题选择模型的实现部分；
        - `BertForQuestionAnswering.py`是问题回答（text span）模型的实现部分；
        - `BertForNSPAndMLM.py`是BERT模型预训练的两个任务实现部分；
        - `BertForTokenClassification.py`是字符分类（如：命名体识别）模型的实现部分；

- `Task`目录中是各个具体下游任务的训练和推理实现
    - `TaskForMultipleChoice.py`是问答选择任务的训练和推理实现，可用于问答选择任务（例如SWAG数据集）；
    - `TaskForSQuADQuestionAnswering.py`是问题回答任务的训练和推理实现，可用于问题问答任务（例如SQuAD数据集）；

- `utils`是各个工具类的实现
    - `data_helpers.py`是各个下游任务的数据预处理及数据集构建模块；
    - `log_helper.py`是日志打印模块；
    - `creat_pretraining_data.py`是用于构造BERT预训练任务的数据集；

## 环境
Python版本为3.9，其它相关包的版本如下：
```python
torch==2.0.1
transformers==4.30.0
numpy==1.25.0
pandas==2.0.3
scikit-learn==1.3.0
tqdm==4.65.0
```
```sh
$ conda activate master
```

## 使用方式

### Step 1. 下载数据 
下载完成各个数据集以及相应的BERT预训练模型（如果为空），并放入对应的目录中。具体可以查看每个数据（`data`)目录下的`README.md`文件。

### Step 2. 运行模型 
进入`Tasks`目录，运行相关模型.

### 2.1 SWAG多项选择任务
```python
python TaskForMultipleChoice.py
```

### 2.2 SQuAD问题回答任务
```python
python TaskForSQuADQuestionAnswering.py
```

运行结束后，`data/SQuAD`目录中会生成一个名为`best_result.json`的预测文件，此时只需要切换到该目录下，并运行以下代码即可得到在`dev-v1.1.json`的测试结果：

```python
python evaluate-v1.1.py dev-v1.1.json best_result.json
```

## 结果

*bert-base-uncased* 

| Task           | Metric          | Enpoch | Result      | Training time (s) | 
|----------------|-----------------|--------|-------------|-------------------|
| SQuAD          | exact_match/F1  | 2      | 79.64/87.52 |  2852.253         |
| SWAG           | Accuracy        | 2      | 80.10       |  1253.970         |
