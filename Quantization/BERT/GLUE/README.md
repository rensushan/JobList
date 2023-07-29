# 用GLUE数据集在BERT上训练

[GitHub参考代码](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)

`run_glue.py`位置：`/data/ice/quantization/master/BERT/run_glue.py`

## 环境配置

使用python 3.10的话会报错*段错误（吐核）*，Debug之后发现可能是numpy包版本的问题，需要降版本，降版本又需要低版本的python（也可能不是这个问题），试了很多办法最终还是妥协了

使用**python 3.9**，用conda新建一个环境master（可自行取名）

```sh
$ conda create -n master python=3.9
```

在新环境中安装一些依赖

```sh
$ conda activate master
$ conda install -c huggingface transformers  #或者 pip install transformers
$ pip install torch
$ pip install datasets
$ pip install evaluate
```

## Usage

```sh
$ python run_glue.py --model_name_or_path bert-base-uncased --task_name cola --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --output_dir ./res/cola/
```

### Details

运行程序会自动下载数据集，数据集存在`/data/ice/.cache/huggingface/datasets/glue`
 
--model_name_or_path: `bert-base-uncased`, `bert-base-cased`, `bert-large-uncased`, `bert-large-cased`可选，本次实验均为**bert-base-uncased**

--task_name: `cola`, `sst2`, `mrpc`, `stsb`, `qqp`, `mnli`, `qnli`, `rte`, `wnli`

--do_train: 是否训练

--do_eval: 是否测试

--max_seq_length: `128`填充或裁剪的长度，一般根据数据集句子的长短决定的

在微调过程中，[BERT](https://arxiv.org/abs/1810.04805) 的作者建议使用以下超参(from Appendix A.3 of the BERT paper)

--per_device_train_batch_size: `16`, `32`可选，本次实验均为**32**，此设定默认为多卡分布式训练的batch_size

*如想单卡执行训练可在命令行最前面指定，具体命令行代码如下：*

```sh
$ CUDA_VISIBLE_DEVICES=gpu_ids python run_glue.py ······
```

--learning_rate: `5e-5`, `3e-5`, `2e-5`可选，本次实验均为**2e-5**

--num_train_epochs: BERT作者建议`2`, `3`, `4`可选，但是huggingface跑了5个epoch

--output_dir: 存模型训练/测试结果文件的地址`/data/ice/quantization/master/BERT/res`，其中每个模型内的`all_results.json`包含所有评价指标

## Result

*bert-base-uncased*

| Task  | Metric                       | Result      | Training time (s) | epochs |
|-------|------------------------------|-------------|-------------------|--------|
| CoLA  | Matthews corr                | 57.61       |   225.96          |    5   |
| SST-2 | Accuracy                     | 93.58       |   1788.33         |    5   |
| MRPC  | Accuracy/F1                  | 85.78/90.00 |   97.68           |    5   |
| STS-B | Pearson/Spearman corr        | 89.04/88.59 |   157.78          |    5   |
| QQP   | Accuracy/F1                  | 91.21/88.18 |   24907.99        |    5   |
| MNLI  | Matched acc/Mismatched acc   | 84.63/84.54 |   3756.094        |    3   |
| QNLI  | Accuracy                     | 91.25       |   10564.36        |    5   |
| RTE   | Accuracy                     | 64.98       |   66.29           |    5   |
| WNLI  | Accuracy                     | 56.34       |   17.830          |    5   |

*bert-large-uncased*

| Task  | Metric                       | Result      | Training time (s) | epochs |
|-------|------------------------------|-------------|-------------------|--------|
| CoLA  | Matthews corr                | 60.68       |   723.30          |    5   |
| SST-2 | Accuracy                     | 93.46       |   5701.88         |    5   |
| MRPC  | Accuracy/F1                  | 85.29/89.62 |   309.20          |    5   |
| STS-B | Pearson/Spearman corr        | 88.64/88.63 |   390.32          |    3   |
| QQP   | Accuracy/F1                  | 91.63/88.75 |   29064.75        |    3   |
| MNLI  | Matched acc/Mismatched acc   | 86.51/86.44 |   40030.995       |    3   |
| QNLI  | Accuracy                     | 91.96       |   3328.317        |    3   |
| RTE   | Accuracy                     | 71.84       |   209.97          |    5   |
| WNLI  | Accuracy                     | 53.52       |   42.15           |    3   |
