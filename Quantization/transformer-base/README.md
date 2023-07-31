# Transformer-based + WMT14-EN-DE 

- [git代码库](https://github.com/facebookresearch/fairseq/tree/main)

- [数据集下载及处理参考](https://github.com/facebookresearch/fairseq/tree/main/examples/translation)

- [调参参考](https://github.com/facebookresearch/fairseq/issues/346)


## 使用说明

### 环境配置

有好几个库只能支持python>=3.8 <3.9,所以conda create了一个python==3.8
```
conda activate trans
```

如果自己配提供了`requirement.txt`
```
pip install -r requirement.txt
```

如果运行fairseq项目报错`please bulid Cython components with **pip inatall --editable .**`
```
git clone https://github.com/pytorch/fairseq.git 
cd fairseq && pip install --editable ./
```

### 命令行输入

**预处理**

```
# Download and prepare the data
cd examples/translation/
bash prepare-wmt14en2de.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/wmt14_en_de
fairseq-preprocess --source-lang en --target-lang de --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/wmt14_en_de --workers 100
```
--source-lang ：source language

--target-lang ：target language

--trainpref ：训练文件所在

--validpref ：验证文件所在

--testpref ：测试文件所在

--destdir ：预处理后的文件存放地

--workers ：并行处理数量

**训练**

参数结合了原文——[Attention Is All You Need](https://doi.org/10.48550/arXiv.1706.03762)以及fairseq的贡献者的回答

```
CUDA_VISIBLE_DEVICES=2  fairseq-train data-bin/wmt14_en_de --arch transformer_wmt_en_de --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 --stop-min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 --max-tokens 4096 --save-dir checkpoints/en-de --update-freq 8 --no-progress-bar --log-format json --log-interval 50 --save-interval-updates 1000 --dropout 0.1 --max-epoch 1 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

--arch ：所使用的模型架构

--share-decoder-input-output-embed    --share-all-embeddings

--optimizer ：指定使用的优化器，`Possible choices: adadelta, adafactor, adagrad, adam, adamax, composite, cpu_adam, lamb, nag, sgd`

--adam-betas ：adam的β1和β2的值

--clip-norm ：梯度裁剪

--lr-scheduler ：学习率调度器，指定学习缩减的方式，`Possible choices: cosine, fixed, inverse_sqrt, manual, pass_through, polynomial_decay, reduce_lr_on_plateau, step, tri_stage, triangular`

--warmup-init-lr ：热身阶段的初始学习率，线性提高到lr

--warmup-updates ：热身4000个updates

--lr ：学习率

--stop-min-lr ：到达此lr停止训练

--criterion ：损失函数

--label-smoothing ：标签平滑，将label_smoothed_cross_entropy损失默认为0的label-smoothing值改为0.1

--weight-decay ：权重衰减

--max-tokens ：按照词的数量来分batch，每个batch包含4096个词

--save-dir ：权重保存文件夹，训练过程保存中间模型

--update-freq ：指定参数更新频率：通常每计算一次梯度，就会更新一次参数，但是某些时候希望多次梯度计算后更新参数。这里设置为8是因为原文是在8个GPU上跑的，我们这里使用单卡，所以设置为8基本等价于8个GPU

--no-progress-bar ：逐行打印日志，方便保存

--log-format ：保存log形式

--log-interval ：每训练50次会打印一次

--save-interval-updates ：每1000步保存一次，通过step来保存模型

最终batch size的大小为max-tokens、GPU数量、update-freq的乘积。

**测试**
```
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt14_en_de --arch transformer_wmt_en_de --path checkpoints/en-de/checkpoint_best.pt --batch-size 128 --beam 4 --remove-bpe --lenpen 0.6 --results-path res/en-de
```
--path ：训练后的模型权重

--gen-subset ：默认解码测试部分。指定解码其他部分，例如--gen-subset train 会翻译

--beam ：设置beam search中的beam size

--remove-bpe ：指定对翻译结果后处理

--lenpen ：设置beam search中的长度惩罚

--quiet ：若不想看到翻译结果，只想看到翻译结果的BLEU分值，使用--quiet参数，只显示翻译进度和最后打分

--results-path ：翻译结果的存放位置

## 结果

| epoch | updatas | training time | valid bleu | test bleu |
|-------|---------|---------------|------------|-----------|
|   1   |  3924   |   00:55:02    |   23.21    |           |
