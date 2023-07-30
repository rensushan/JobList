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

### 命令行输入

参数结合了原文——[Attention Is All You Need](https://doi.org/10.48550/arXiv.1706.03762)以及fairseq的贡献者的回答

```
CUDA_VISIBLE_DEVICES=0  python train.py data-bin/wmt14_en_de --arch transformer_wmt_en_de --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 --stop-min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 --max-tokens 4096 --save-dir checkpoints/en-de --update-freq 8 --no-progress-bar --log-format json --log-interval 50 --save-interval-updates 1000 --keep-interval-updates 20
```
