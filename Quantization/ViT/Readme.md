# ViT

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Download the ImageNet dataset from http://www.image-net.org/
  - Then, move and extract the training and validation images to labeled subfolders

```
# use this env
conda activate master
```

## Dataset processing

quote https://zhuanlan.zhihu.com/p/378991398

The validation set label requires additional processing.

Details can be found in the `imagenet_preprocess.md`

## Details

imagenet-1k数据集位置：/vg_data/ice/dataset/imagenet/1k

**代码位置**：/data/ice/quantization/imagenet2/jeons

预训练权重位置 ： /data/ice/quantization/imagenet2/jeons/ViT/checkpoint

微调模型保存位置： /data/ice/quantization/imagenet2/jeons/ViT/output/ViTScheduler/ViTScheduler_checkpoint.bin

## Training

To train a model, run `train.py`

```
# 终端先限定GPU
export CUDA_VISIBLE_DEVICES=1

""" 训练超参设置, 此处超参设置参考ViT论文 
'AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE' B 1.1章节 Table4
图像分辨率     = 384*384
batch_size    = 512
每 500 steps训练后进行验证
总训练steps    =   20000
"""
python train.py --name ViT_ft --img_size 384 --train_batch_size 512 --eval_every 500 --num_steps 20000 --output_dir /data/ice/quantization/imagenet2/jeons/ViT/output/ViT_ft
```



## Usage

```bash
usage: train.py --name str [--epochs N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [-e] [--pretrained_model_path] [--load_model_path] [--pretrained_model_path] [--gpu GPU] [DIR]

 必要参数：
  --name str                     训练命名，log文件也将以此命名 ''
                                 并保存在/data/ice/quantization/imagenet2/jeons/logs
                                 
  --output_dir str               微调模型保存地址         
 
 可选参数：                             
  DIR                            数据集存储地址 (default: '/vg_data/ice/dataset/imagenet/1k')
  --model_type                   选择网络模型,可选项["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"]
                                 (default: ViT-B_32)
                                 
  --pretrained_dir str           预训练权重保存地址 
                   (defalut: /data/ice/quantization/imagenet2/jeons/ViT/checkpoint/imagenet21k_ViT-B_32.npz)
   
  --img_size N                   图像分辨率 (default: 224)
  --train_batch_size N           训练Batch Size (default: 2048)  batch_size 随着img_size调整                   
  --eval_batch_size N            验证Batch Size (defalut: 64)
  --eval_every N                 验证前需要训练的Steps (default: 500)
  
                                 
  --learning_rate float          学习率 (default: 3e-2)
  --weight_decay float           default：0
  --num_steps N                  总训练Steps (default: 20000)
  --decay_type                   学习率调整方法 可选项["cosine", "linear"]，(default: "cosine")
  --warmup_steps N               Warm Up 的 Steps (default: 500)
  --max_grad_norm float          梯度上限， (default: 1)

```

## Result

plz wait 6h。

## Reference

### Code

jeons Version ViT: https://github.com/jeonsworld/ViT-pytorch

### Pre-trained model (Google's Official Checkpoint)

https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_32.npz;tab=live_object?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))
