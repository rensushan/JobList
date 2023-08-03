# MobileNetV2 

## 代码存储位置

`/data/ice/quantization/imagenet2/mobilenet`

## 训练脚本

`torchrun`是`torch.distributed.launch`的超集，新版torch进行了替换。

```
torchrun --nproc_per_node=1 /data/ice/quantization/imagenet2/mobilenet/train.py --model mobilenet_v2 --epochs 300 --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98
```

超参的设置遵循[Improve the accuracy of Classification models by using SOTA recipes and primitives · Issue #3995 · pytorch/vision (github.com)](https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning)的建议。

### 参数

```bash
--nproc_per_node     使用gpu的数量
--model          
--epochs         
--lr
--wd, --weight-decay 
--lr-step-size       每X步长减少一次学习率
--lr-gamma           降低学习率的倍数
```

## Reference

代码源自TorchVision References：[vision/references/classification at main · pytorch/vision (github.com)](https://github.com/pytorch/vision/tree/main/references/classification)
