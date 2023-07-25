# ViT

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install timm
- Download the ImageNet dataset from http://www.image-net.org/
  - Then, move and extract the training and validation images to labeled subfolders

```
# use this env
conda activate t2
```

## Dataset processing

quote https://zhuanlan.zhihu.com/p/378991398

The validation set label requires additional processing.

Details can be found in the `imagenet_preprocess.md`

## Training

To train a model, run `main.py` with ViT and the path to the ImageNet dataset:

```bash
python main.py -p /data/ice/quantization/imagenet/pre-train_pth/vit_b_32.pth --save_model_path
```

## Validation

```bash
python main.py -e --load_model_path /data/ice/quantization/imagenet/res/vit_b_32_ft/model_best.pth
```

Since it is not the `state_dict` but the whole model that is saved during training, the validation session loads the whole model directly using `model = torch.load(*pth)`.The `load_state_dict` and `save_checkpoint` methods will be refactored later.

## Usage

```bash
usage: main.py [-j N] [--epochs N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [-e] [--pretrained_model_path] [--load_model_path] [--pretrained_model_path] [--gpu GPU] [DIR]

optional arguments:
 DIR                                     path to dataset (default: '/vg_data/ice/dataset/imagenet/1k')
  -j N, --workers N                      number of data loading workers (default: 4)
  --epochs N                             number of total epochs to run (defalut: 14)
  -b N, --batch-size N                   mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                                         initial learning rate
  --momentum M                           momentum
  --wd W, --weight-decay W
                                         weight decay (default: 1e-4)
  --print-freq N                         print frequency (default: 10)
  -e, --evaluate                         evaluate model on validation set
  -p N, --pretrained_model_path          pre-trained model path, (default: None)
  --lmp, --load_model_path               Path of load model
  --save_model_path                      Path where the model will be saved
  --gpu GPU                              GPU id to use.(defalut: 0)

```

## Result

The final model is stored in `res`folder. 

| model                                    | top1   | top5   | path                                                         |
| ---------------------------------------- | ------ | ------ | ------------------------------------------------------------ |
| torchvision.models.vit_b_32_ft_14epochs  | 75.178 | 92.350 | /data/ice/quantization/imagenet/res/vit_b_32_ft/model_best.pth |
| timm/vit_base_patch32_clip_224.openai_ft | 81.378 | 95.668 | /data/ice/quantization/imagenet/res/vit_b_32_timm_ft/model_best.pth |

The first line is pre-training based on **Pytorch** public weights.

The second line is pre-training based on the **TIMM library** public weights. (Closer to the int4 paper results)

## Reference

### Code

[examples/imagenet at main · pytorch/examples (github.com)](https://github.com/pytorch/examples/tree/main/imagenet)

### Pre-trained model

[timm/vit_base_patch32_clip_224.openai_ft_in1k · Hugging Face](https://huggingface.co/timm/vit_base_patch32_clip_224.openai_ft_in1k)
