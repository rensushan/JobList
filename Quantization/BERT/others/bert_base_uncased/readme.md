`bert_base_uncased`目录中是BERT base英文预训练模型以及配置文件

    模型下载地址：https://huggingface.co/bert-base-uncased/tree/main

    需要将`config.json`、`、pytorch_model.bin`、`vocab.txt`三个文件下载并放入此文件夹

    注意：`config.json`中需要添加`"pooler_type": "first_token_transform"`
                                `"use_torch_multi_head": true`
