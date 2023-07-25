# 用GLUE数据集在BERT上训练

[GitHub参考代码](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)

`run_glue.py`

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

或者整理了一个`requirement.txt`，直接读取安装即可

```sh
$ pip install -r requirement.txt
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

--per_device_train_batch_size: `16`, `32`可选，本次实验均为**32**

--learning_rate: `5e-5`, `3e-5`, `2e-5`可选，本次实验均为**2e-5**

--num_train_epochs: `2`, `3`, `4`可选，本次实验均为**3**

--output_dir: 存模型训练/测试结果文件的地址`/data/ice/quantization/master/BERT/res`，其中每个模型内的`all_results.json`包含所有评价指标

## Result

| Task  | Metric                       | Result      | Training time (s) | Evaluating time (s) |
|-------|------------------------------|-------------|-------------------|---------------------|
| CoLA  | Matthews corr                | 53.92       |   67.636          |   3.544             |
| SST-2 | Accuracy                     | 92.09       |   676.429         |   3.816             |
| MRPC  | Accuracy/F1                  | 80.15/86.52 |   34.552          |   1.432             |
| STS-B | Pearson/Spearman corr        | 87.06/86.78 |   93.303          |   6.914             |
| QQP   | Accuracy/F1                  | 90.64/87.51 |   2773.041        |   136.169           |
| MNLI  | Matched acc/Mismatched acc   | 83.91/84.10 |   9323.000        |   
| QNLI  | Accuracy                     | 89.66       |   1857.000        |   
| RTE   | Accuracy                     | 57.04       |   45.743          |   1.227             |
| WNLI  | Accuracy                     | 52.11       |   19.382          |   0.334             |
