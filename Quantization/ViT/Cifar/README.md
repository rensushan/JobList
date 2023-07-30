# Vision Transformer

在cifar10和cifar100数据集上训练ViT，参考：https://github.com/jeonsworld/ViT-pytorch


## Usage
### 1. Download Pre-trained model (Google's Official Checkpoint)
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, **ViT-B_32**, ViT-L_16, **ViT-L_32**, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16

本实验下载了`imagenet21k_ViT-B_32.npz`和`imagenet21k_ViT-L_32.npz`预训练权重，存放在`/data/ice/quantization/master/ViT/checkpoint/`

```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz

```

### 2. Train Model
```
python train.py --name cifar10 --dataset cifar10 --model_type ViT-B_32 --pretrained_dir checkpoint/imagenet21k_ViT-B_32.npz #训练cifar10数据集
python train.py --name cifar100 --dataset cifar100 --model_type ViT-B_32 --pretrained_dir checkpoint/imagenet21k_ViT-B_32.npz #训练cifar100数据集
```
--name ：模型训练名字，同时也是logs存放的文件夹名字

--dataset ：选择数据集`cifar10`,`cifar100`

--model_type ：选择训练所用模型`ViT-B_32`,`ViT-L_32`

--pretrained_dir ：预训练权重所在文件夹`checkpoint/imagenet21k_ViT-B_32.npz`,`checkpoint/imagenet21k_ViT-L_32.npz`

--train_batch_size ：The default batch size is 512.  `

CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).

Also can use [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html) to reduce memory usage and train faster
```
python train.py --name cifar10 --dataset cifar10 --model_type ViT-B_32 --pretrained_dir checkpoint/imagenet21k_ViT-B_32.npz --fp16 --fp16_opt_level O2
```
混合精度相关代码已被我注释掉了，如若需要可以将`#`去掉。修改后代码如下。
```
17  from apex import amp
18  from apex.parallel import DistributedDataParallel as DDP
......
163      if args.fp16:
164          model, optimizer = amp.initialize(models=model,
165                                            optimizers=optimizer,
166                                            opt_level=args.fp16_opt_level)
167          amp._amp_state.loss_scalers[0]._loss_scale = 2**20
168 
169      Distributed training
170      if args.local_rank != -1:
171          model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
......
200         # loss.backward()
             if args.fp16:
                 with amp.scale_loss(loss, optimizer) as scaled_loss:
                     scaled_loss.backward()
             else:
                 loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                 if args.fp16:
                     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                 else:
                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
```

Fine Tune后的权重保存路径为`/data/ice/quantization/master/ViT/output`,可以直接调用在验证集上跑Accuracy

其中`/data/ice/quantization/master/ViT/output/cifar100_checkpoint.bin`对应

```
python train.py --name cifar100 --dataset cifar100  -v true --model_type ViT-L_32
```

`/data/ice/quantization/master/ViT/output/cifar10_checkpoint.bin`对应

```
python train.py --name cifar10 --dataset cifar10  -v true --model_type ViT-B_32 --img_size 384 --load_model_dir /data/ice/quantization/master/ViT/output/cifar10_checkpoint.bin
```

## Results

batch_size=512 (部分batch_size=512单卡显存不足，改为256) , learning_rate=0.03 , weight_decay=0

|  upstream   |  model   |  dataset  | resolution |  steps  |   acc   |    time   | batch_size |
|:-----------:|:--------:|:---------:|:----------:|:-------:|:-------:|:---------:|:----------:|
| imagenet21k | ViT-B_32 | CIFAR-10  |  224x224   |  1000   |  98.89  |  2107.74  |     512    |
| imagenet21k | ViT-B_32 | CIFAR-10  |  384x384   |  1000   |  98.65  |  3346.43  |     512    |
| imagenet21k | ViT-B_32 | CIFAR-10  |  384x384   |  10000  |  99.00  |  32593.36 |     512    |
| imagenet21k | ViT-L_32 | CIFAR-10  |  224x224   |  1000   |  98.83  |  4039.10  |     512    |
| imagenet21k | ViT-L_32 | CIFAR-10  |  384x384   |  1000   |  98.72  |  4487.46  |     256    |
| imagenet21k | ViT-B_32 | CIFAR-100 |  224x224   |  1000   |  90.74  |  1270.46  |     512    |
| imagenet21k | ViT-B_32 | CIFAR-100 |  384x384   |  1000   |  90.61  |  3340.64  |     512    |
| imagenet21k | ViT-L_32 | CIFAR-100 |  224x224   |  1000   |  91.95  |  2970.06  |     512    |
| imagenet21k | ViT-L_32 | CIFAR-100 |  384x384   |  10000  |  93.39  |  45135.64 |     256    |


## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models)


## Citations

```bibtex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
